/* SPDX-License-Identifier: MIT */
#include "stem_backend.h"
#include "stem_filterbank.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *name;
    size_t blocks;
} backend_case_t;

static const backend_case_t CASES[] = {
    {"chirp20_zero", 20u},
    {"chirp20_full", 20u},
    {"chirp20_random", 20u},
    {"impulse_full", 1u},
};

static FILE *open_vector(const char *directory, const char *name,
                         const char *suffix)
{
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s.%s", directory, name,
                          suffix);
    if (length <= 0 || (size_t)length >= sizeof(path))
        return NULL;
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

static void run_case(const char *directory, const backend_case_t *test_case,
                     size_t *impulse_vocal_peak)
{
    FILE *spectrum_file = open_vector(directory, test_case->name,
                                      "spectrum.c64le");
    FILE *mask_file = open_vector(directory, test_case->name, "mask.s16le");
    FILE *vocal_file = open_vector(directory, test_case->name, "vocal.f32le");
    stem_backend_t backend;
    double squared_error = 0.0;
    size_t compared = 0u;
    float largest_impulse = 0.0f;

    if (spectrum_file == NULL || mask_file == NULL || vocal_file == NULL) {
        fprintf(stderr, "cannot open backend vectors for %s\n", test_case->name);
        exit(2);
    }
    if (stem_backend_init(&backend) != STEM_BACKEND_OK) {
        fprintf(stderr, "backend init failed\n");
        exit(3);
    }
    for (size_t block = 0u; block < test_case->blocks; ++block) {
        stem_spectrum_block_t spectrum;
        int16_t mask[STEM_PACKED_VALUES];
        float actual[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
        float expected[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
        size_t expected_frames = block == 0u
            ? STEM_BACKEND_STARTUP_FRAMES : STEM_BLOCK_SAMPLES;
        require_read(spectrum_file, spectrum.bins, sizeof(kiss_fft_cpx),
                     STEM_INPUT_CHANNELS * STEM_BLOCK_FRAMES * STEM_FFT_BINS,
                     "backend spectrum");
        spectrum.first_sample = (int64_t)(block * STEM_BLOCK_SAMPLES);
        require_read(mask_file, mask, sizeof(*mask), STEM_PACKED_VALUES,
                     "backend mask");
        require_read(vocal_file, expected, sizeof(float),
                     expected_frames * STEM_INPUT_CHANNELS, "backend vocal");
        size_t produced = stem_backend_process(&backend, &spectrum, mask, actual);
        if (produced != expected_frames) {
            fprintf(stderr, "%s block %zu produced %zu, expected %zu\n",
                    test_case->name, block, produced, expected_frames);
            exit(3);
        }
        for (size_t frame = 0u; frame < produced; ++frame) {
            for (size_t channel = 0u; channel < STEM_INPUT_CHANNELS; ++channel) {
                float error = actual[frame][channel] - expected[frame][channel];
                squared_error += (double)error * (double)error;
                ++compared;
                if (block != 0u && frame == 0u && fabsf(error) > 5.0e-4f) {
                    fprintf(stderr, "%s boundary discontinuity at block %zu\n",
                            test_case->name, block);
                    exit(3);
                }
                if (impulse_vocal_peak != NULL
                    && fabsf(actual[frame][channel]) > largest_impulse) {
                    largest_impulse = fabsf(actual[frame][channel]);
                    *impulse_vocal_peak = frame;
                }
            }
        }
    }
    if (compared == 0u || sqrt(squared_error / (double)compared) > 2.0e-4) {
        fprintf(stderr, "%s RMS error exceeds tolerance\n", test_case->name);
        exit(3);
    }
    fclose(spectrum_file);
    fclose(mask_file);
    fclose(vocal_file);
}

static size_t test_delay(const char *directory)
{
    FILE *expected_file = open_vector(directory, "impulse_full", "delay.f32le");
    stem_backend_t backend;
    float input[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS] = {{0.0f}};
    float actual[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    float expected[STEM_BACKEND_STARTUP_FRAMES][STEM_INPUT_CHANNELS];
    static float next_input[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    static float steady[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    size_t peak = 0u;
    float largest = 0.0f;

    if (expected_file == NULL || stem_backend_init(&backend) != STEM_BACKEND_OK)
        exit(2);
    input[0][0] = 32767.0f / 32768.0f;
    input[0][1] = -1.0f;
    input[257][0] = -12000.0f / 32768.0f;
    input[257][1] = 16000.0f / 32768.0f;
    require_read(expected_file, expected, sizeof(float),
                 STEM_BACKEND_STARTUP_FRAMES * STEM_INPUT_CHANNELS,
                 "delayed mix");
    if (stem_delay_mix(&backend, input, actual, STEM_BLOCK_SAMPLES)
        != STEM_BACKEND_STARTUP_FRAMES)
        exit(3);
    for (size_t frame = 0u; frame < STEM_BACKEND_STARTUP_FRAMES; ++frame) {
        for (size_t channel = 0u; channel < STEM_INPUT_CHANNELS; ++channel) {
            if (actual[frame][channel] != expected[frame][channel]) {
                fprintf(stderr, "delayed mix differs at frame %zu\n", frame);
                exit(3);
            }
            if (fabsf(actual[frame][channel]) > largest) {
                largest = fabsf(actual[frame][channel]);
                peak = frame;
            }
        }
    }
    for (size_t frame = 0u; frame < STEM_BLOCK_SAMPLES; ++frame) {
        next_input[frame][0] = 0.25f + (float)frame / 32768.0f;
        next_input[frame][1] = -0.25f - (float)frame / 32768.0f;
    }
    if (stem_delay_mix(&backend, next_input, steady, STEM_BLOCK_SAMPLES)
        != STEM_BLOCK_SAMPLES)
        exit(3);
    for (size_t frame = 0u; frame < STEM_BLOCK_SAMPLES; ++frame) {
        const float *source = frame < STEM_FFT_SIZE / 2u
            ? input[STEM_BACKEND_STARTUP_FRAMES + frame]
            : next_input[frame - STEM_FFT_SIZE / 2u];
        if (steady[frame][0] != source[0] || steady[frame][1] != source[1]) {
            fprintf(stderr, "steady delayed mix differs at frame %zu\n", frame);
            exit(3);
        }
    }
    fclose(expected_file);
    return peak;
}

int main(void)
{
    const char *directory = getenv("STEM_VECTOR_DIR");
    size_t vocal_peak = 0u;
    stem_backend_t sparse;
    size_t filter_terms = 0u;
    if (stem_backend_init(&sparse) != STEM_BACKEND_OK)
        return 3;
    for (size_t bin = 0u; bin < STEM_FFT_BINS; ++bin) {
        size_t begin = sparse.synthesis_begin[bin];
        size_t end = sparse.synthesis_end[bin];
        if (begin < STEM_BACKEND_PROTECTED_BANDS || begin > end
            || end > STEM_BAND_COUNT)
            return 3;
        filter_terms += end - begin;
        for (size_t band = STEM_BACKEND_PROTECTED_BANDS;
             band < STEM_BAND_COUNT; ++band) {
            if ((band < begin || band >= end) && stem_synthesis[band][bin] != 0.0f)
                return 3;
        }
    }
    if (filter_terms == 0u || filter_terms > 2u * STEM_FFT_BINS)
        return 3;
    if (directory == NULL || directory[0] == '\0')
        return 2;
    if (stem_backend_init(NULL) != STEM_BACKEND_E_INVALID
        || stem_float_to_i16(2.0f) != INT16_MAX
        || stem_float_to_i16(-2.0f) != INT16_MIN
        || stem_float_to_i16(0.0f) != 0)
        return 3;
    for (size_t index = 0u; index < ARRAY_COUNT(CASES); ++index) {
        run_case(directory, &CASES[index],
                 index == ARRAY_COUNT(CASES) - 1u ? &vocal_peak : NULL);
    }
    if (test_delay(directory) != vocal_peak) {
        fprintf(stderr, "delayed mix and vocal impulse peaks are misaligned\n");
        return 3;
    }
    puts("stem backend: PASS");
    return 0;
}
