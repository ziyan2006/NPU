/* SPDX-License-Identifier: MIT */
#include "stem_frontend.h"
#include "stem_filterbank.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *name;
    size_t blocks;
    size_t pcm_frames;
} vector_case_t;

static const vector_case_t CASES[] = {
    {"impulse", 1u, 4352u},
    {"dc", 1u, 4352u},
    {"nyquist", 1u, 4352u},
    {"noise", 1u, 4352u},
    {"chirp20", 20u, 82176u},
    {"clip", 1u, 4352u},
};

static size_t clipped_values;

static FILE *open_vector(const char *directory, const char *name,
                         const char *suffix)
{
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s.%s", directory, name, suffix);
    if (length <= 0 || (size_t)length >= sizeof(path)) {
        return NULL;
    }
    return fopen(path, "rb");
}

static void require_read(FILE *stream, void *destination, size_t element_bytes,
                         size_t count, const char *description)
{
    if (fread(destination, element_bytes, count, stream) != count) {
        fprintf(stderr, "short read: %s\n", description);
        exit(2);
    }
}

static void compare_block(const vector_case_t *test_case, size_t block_index,
                          const stem_spectrum_block_t *actual,
                          const int16_t actual_packed[STEM_PACKED_VALUES],
                          FILE *spectrum_file, FILE *magnitude_file,
                          FILE *bands_file, FILE *packed_file)
{
    kiss_fft_cpx *expected_spectrum = malloc(sizeof(*expected_spectrum) *
        STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_FFT_BINS);
    float *expected_magnitude = malloc(sizeof(*expected_magnitude) *
        STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_FFT_BINS);
    float *expected_bands = malloc(sizeof(*expected_bands) *
        STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_BAND_COUNT);
    int16_t *expected_packed = malloc(sizeof(*expected_packed) * STEM_PACKED_VALUES);
    if (expected_spectrum == NULL || expected_magnitude == NULL ||
        expected_bands == NULL || expected_packed == NULL) {
        fprintf(stderr, "test allocation failed\n");
        exit(2);
    }
    require_read(spectrum_file, expected_spectrum, sizeof(*expected_spectrum),
                 STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_FFT_BINS,
                 "spectrum");
    require_read(magnitude_file, expected_magnitude, sizeof(*expected_magnitude),
                 STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_FFT_BINS,
                 "magnitude");
    require_read(bands_file, expected_bands, sizeof(*expected_bands),
                 STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_BAND_COUNT,
                 "bands");
    require_read(packed_file, expected_packed, sizeof(*expected_packed),
                 STEM_PACKED_VALUES, "packed tensor");

    if (actual->first_sample != (int64_t)(block_index * STEM_BLOCK_SAMPLES)) {
        fprintf(stderr, "%s block %zu first sample mismatch\n",
                test_case->name, block_index);
        exit(3);
    }
    float maximum_complex_error = 0.0f;
    float maximum_band_error = 0.0f;
    float maximum_band_actual = 0.0f;
    float maximum_band_expected = 0.0f;
    size_t maximum_band_channel = 0u;
    size_t maximum_band_frame = 0u;
    size_t maximum_band_index = 0u;
    size_t spectrum_index = 0u;
    size_t band_index = 0u;
    for (size_t channel = 0; channel < STEM_INPUT_CHANNELS; ++channel) {
        for (size_t frame = 0; frame < STEM_BLOCK_FRAMES; ++frame) {
            float magnitude[STEM_FFT_BINS];
            for (size_t bin = 0; bin < STEM_FFT_BINS; ++bin, ++spectrum_index) {
                kiss_fft_cpx value = actual->bins[channel][frame][bin];
                kiss_fft_cpx expected = expected_spectrum[spectrum_index];
                float real_error = fabsf(value.r - expected.r);
                float imag_error = fabsf(value.i - expected.i);
                if (real_error > maximum_complex_error) maximum_complex_error = real_error;
                if (imag_error > maximum_complex_error) maximum_complex_error = imag_error;
                magnitude[bin] = hypotf(value.r, value.i);
                if (fabsf(magnitude[bin] - expected_magnitude[spectrum_index]) > 2.0e-4f) {
                    fprintf(stderr, "%s block %zu magnitude mismatch c=%zu t=%zu f=%zu\n",
                            test_case->name, block_index, channel, frame, bin);
                    exit(3);
                }
            }
            for (size_t band = 0; band < STEM_BAND_COUNT; ++band, ++band_index) {
                float value = 0.0f;
                for (size_t bin = 0; bin < STEM_FFT_BINS; ++bin) {
                    value += magnitude[bin] * stem_analysis[bin][band];
                }
                float error = fabsf(value - expected_bands[band_index]);
                if (error > maximum_band_error) {
                    maximum_band_error = error;
                    maximum_band_actual = value;
                    maximum_band_expected = expected_bands[band_index];
                    maximum_band_channel = channel;
                    maximum_band_frame = frame;
                    maximum_band_index = band;
                }
            }
        }
    }
    float band_tolerance = strcmp(test_case->name, "clip") == 0
        ? 1.0e-4f : 2.0e-5f;
    if (maximum_complex_error > 2.0e-4f || maximum_band_error > band_tolerance) {
        fprintf(stderr, "%s block %zu numerical error complex=%g band=%g "
                "at c=%zu t=%zu b=%zu actual=%.9g expected=%.9g\n",
                test_case->name, block_index,
                (double)maximum_complex_error, (double)maximum_band_error,
                maximum_band_channel, maximum_band_frame, maximum_band_index,
                (double)maximum_band_actual, (double)maximum_band_expected);
        exit(3);
    }
    if (strcmp(test_case->name, "clip") != 0 &&
        memcmp(actual_packed, expected_packed,
               sizeof(*actual_packed) * STEM_PACKED_VALUES) != 0) {
        fprintf(stderr, "%s block %zu packed tensor mismatch\n",
                test_case->name, block_index);
        exit(3);
    }
    for (size_t band = 0; band < STEM_BAND_COUNT; ++band) {
        for (size_t frame = 0; frame < STEM_BLOCK_FRAMES; ++frame) {
            size_t base = (band * STEM_BLOCK_FRAMES + frame) * STEM_NHWC8_LANES;
            for (size_t lane = STEM_INPUT_CHANNELS; lane < STEM_NHWC8_LANES; ++lane) {
                if (actual_packed[base + lane] != 0) {
                    fprintf(stderr, "%s block %zu nonzero padding lane\n",
                            test_case->name, block_index);
                    exit(3);
                }
            }
            for (size_t lane = 0; lane < STEM_INPUT_CHANNELS; ++lane) {
                int value = actual_packed[base + lane];
                if (value < -2048 || value > 2047) {
                    fprintf(stderr, "%s block %zu INT12 clipping failure\n",
                            test_case->name, block_index);
                    exit(3);
                }
                if (value == -2048 || value == 2047) {
                    ++clipped_values;
                }
            }
        }
    }
    free(expected_spectrum);
    free(expected_magnitude);
    free(expected_bands);
    free(expected_packed);
}

static void run_case(const char *directory, const vector_case_t *test_case)
{
    FILE *pcm_file = open_vector(directory, test_case->name, "pcm.s16le");
    FILE *spectrum_file = open_vector(directory, test_case->name, "spectrum.c64le");
    FILE *magnitude_file = open_vector(directory, test_case->name, "magnitude.f32le");
    FILE *bands_file = open_vector(directory, test_case->name, "bands.f32le");
    FILE *packed_file = open_vector(directory, test_case->name, "packed.s16le");
    if (pcm_file == NULL || spectrum_file == NULL || magnitude_file == NULL ||
        bands_file == NULL || packed_file == NULL) {
        fprintf(stderr, "cannot open vectors for %s\n", test_case->name);
        exit(2);
    }
    int16_t *pcm_data = malloc(test_case->pcm_frames * 2u * sizeof(*pcm_data));
    if (pcm_data == NULL) {
        fprintf(stderr, "PCM allocation failed\n");
        exit(2);
    }
    require_read(pcm_file, pcm_data, sizeof(*pcm_data),
                 test_case->pcm_frames * 2u, "PCM");

    stem_frontend_t frontend;
    if (stem_frontend_init(&frontend) != STEM_FRONTEND_OK) {
        fprintf(stderr, "frontend init failed\n");
        exit(3);
    }
    static const size_t chunks[] = {1u, 37u, 511u, 1024u, 257u, 409u};
    size_t cursor = 0u;
    size_t block = 0u;
    size_t chunk_index = 0u;
    while (cursor < test_case->pcm_frames) {
        size_t requested = chunks[chunk_index++ % ARRAY_COUNT(chunks)];
        if (requested > test_case->pcm_frames - cursor) {
            requested = test_case->pcm_frames - cursor;
        }
        size_t accepted = stem_frontend_push(
            &frontend, pcm_data + cursor * 2u, requested
        );
        if (accepted != requested) {
            fprintf(stderr, "%s rejected input at frame %zu\n", test_case->name, cursor);
            exit(3);
        }
        cursor += accepted;
        while (stem_frontend_block_ready(&frontend)) {
            stem_spectrum_block_t spectrum_block;
            int16_t packed[STEM_PACKED_VALUES];
            if (stem_frontend_pack(&frontend, packed, &spectrum_block)
                != STEM_FRONTEND_OK) {
                fprintf(stderr, "%s pack failed\n", test_case->name);
                exit(3);
            }
            if (block >= test_case->blocks) {
                fprintf(stderr, "%s produced too many blocks\n", test_case->name);
                exit(3);
            }
            compare_block(test_case, block, &spectrum_block, packed, spectrum_file,
                          magnitude_file, bands_file, packed_file);
            ++block;
        }
    }
    if (block != test_case->blocks || stem_frontend_block_ready(&frontend)) {
        fprintf(stderr, "%s produced %zu blocks, expected %zu\n",
                test_case->name, block, test_case->blocks);
        exit(3);
    }
    free(pcm_data);
    fclose(pcm_file);
    fclose(spectrum_file);
    fclose(magnitude_file);
    fclose(bands_file);
    fclose(packed_file);
}

int main(void)
{
    const char *directory = getenv("STEM_VECTOR_DIR");
    if (directory == NULL || directory[0] == '\0') {
        fprintf(stderr, "STEM_VECTOR_DIR is not set\n");
        return 2;
    }
    stem_spectrum_block_t spectrum;
    int16_t packed[STEM_PACKED_VALUES];
    stem_frontend_t empty;
    if (stem_frontend_init(NULL) != STEM_FRONTEND_E_INVALID ||
        stem_frontend_init(&empty) != STEM_FRONTEND_OK ||
        stem_frontend_block_ready(&empty) ||
        stem_frontend_pack(&empty, packed, &spectrum) != STEM_FRONTEND_E_NOT_READY ||
        stem_frontend_push(&empty, NULL, 1u) != 0u) {
        fprintf(stderr, "frontend state/error contract failed\n");
        return 3;
    }
    size_t filter_terms = 0u;
    for (size_t band = 0u; band < STEM_BAND_COUNT; ++band) {
        size_t begin = empty.analysis_begin[band];
        size_t end = empty.analysis_end[band];
        if (begin > end || end > STEM_FFT_BINS)
            return 3;
        filter_terms += end - begin;
        for (size_t bin = 0u; bin < STEM_FFT_BINS; ++bin) {
            if ((bin < begin || bin >= end) && stem_analysis[bin][band] != 0.0f)
                return 3;
        }
    }
    if (filter_terms == 0u || filter_terms > 2u * STEM_FFT_BINS)
        return 3;
    for (size_t index = 0; index < ARRAY_COUNT(CASES); ++index) {
        run_case(directory, &CASES[index]);
    }
    if (clipped_values == 0u) {
        fprintf(stderr, "vectors did not exercise INT12 saturation\n");
        return 3;
    }
    puts("stem frontend: PASS");
    return 0;
}
