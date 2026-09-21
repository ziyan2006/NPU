/* SPDX-License-Identifier: MIT */
#include "stem_backend.h"

#include "stem_filterbank.h"

#include <math.h>
#include <string.h>

#define STEM_PI_F 3.14159265358979323846f
#define STEM_OLA_EPSILON 1.0e-12f

static size_t ola_slot(uint64_t sample)
{
    return (size_t)(sample % STEM_BACKEND_OLA_CAPACITY);
}

float stem_decode_mask_q11(int16_t quantized)
{
    int32_t signed_tanh = quantized;

    if (signed_tanh < -2047)
        signed_tanh = -2047;
    else if (signed_tanh > 2047)
        signed_tanh = 2047;
    /* The resident task exposes the signed Q1.11 tanh value.  Training uses
     * mask=(tanh+1)/2, so folding the sign with abs() is not equivalent: it
     * maps a 0.5 mask to zero and reverses the lower half of the range. */
    return (float)(signed_tanh + 2047) / 4094.0f;
}

stem_backend_result_t stem_backend_init(stem_backend_t *backend)
{
    size_t config_bytes;

    if (backend == NULL)
        return STEM_BACKEND_E_INVALID;
    memset(backend, 0, sizeof(*backend));
    config_bytes = sizeof(backend->fft_storage.bytes);
    backend->inverse_fft = kiss_fftr_alloc(
        (int)STEM_FFT_SIZE, 1, backend->fft_storage.bytes, &config_bytes);
    if (backend->inverse_fft == NULL
        || config_bytes > sizeof(backend->fft_storage.bytes)) {
        backend->last_result = STEM_BACKEND_E_CONFIG;
        return STEM_BACKEND_E_CONFIG;
    }
    for (size_t sample = 0u; sample < STEM_FFT_SIZE; ++sample) {
        float phase = 2.0f * STEM_PI_F * (float)sample / (float)STEM_FFT_SIZE;
        backend->window[sample] = 0.5f - 0.5f * cosf(phase);
    }
    /* Skip only leading/trailing zero coefficients, including protected
     * bands. This preserves the original summation order exactly. */
    for (size_t bin = 0u; bin < STEM_FFT_BINS; ++bin) {
        size_t begin = STEM_BACKEND_PROTECTED_BANDS, end = STEM_BAND_COUNT;
        while (begin < end && stem_synthesis[begin][bin] == 0.0f)
            ++begin;
        while (end > begin && stem_synthesis[end - 1u][bin] == 0.0f)
            --end;
        backend->synthesis_begin[bin] = (uint16_t)begin;
        backend->synthesis_end[bin] = (uint16_t)end;
    }
    backend->last_result = STEM_BACKEND_OK;
    return STEM_BACKEND_OK;
}

static void synthesize_frame(stem_backend_t *backend,
                             const stem_spectrum_block_t *spectrum,
                             const int16_t output_nhwc8[STEM_PACKED_VALUES],
                             size_t channel, size_t frame,
                             int64_t frame_start)
{
    float band_mask[STEM_BAND_COUNT];

    for (size_t band = 0u; band < STEM_BAND_COUNT; ++band) {
        size_t index = (band * STEM_BLOCK_FRAMES + frame)
            * STEM_NHWC8_LANES + channel;
        band_mask[band] = band < STEM_BACKEND_PROTECTED_BANDS
            ? 0.0f : stem_decode_mask_q11(output_nhwc8[index]);
    }
    for (size_t bin = 0u; bin < STEM_FFT_BINS; ++bin) {
        float mask = 0.0f;
        for (size_t band = backend->synthesis_begin[bin];
             band < backend->synthesis_end[bin]; ++band) {
            mask += band_mask[band] * stem_synthesis[band][bin];
        }
        backend->masked_bins[bin].r =
            spectrum->bins[channel][frame][bin].r * mask;
        backend->masked_bins[bin].i =
            spectrum->bins[channel][frame][bin].i * mask;
    }
    kiss_fftri(backend->inverse_fft, backend->masked_bins,
               backend->inverse_samples);
    for (size_t sample = 0u; sample < STEM_FFT_SIZE; ++sample) {
        int64_t absolute = frame_start + (int64_t)sample;
        if (absolute >= 0) {
            size_t slot = ola_slot((uint64_t)absolute);
            float window = backend->window[sample];
            backend->accumulation[channel][slot] +=
                backend->inverse_samples[sample]
                * (1.0f / (float)STEM_FFT_SIZE) * window;
            if (channel == 0u)
                backend->window_square[slot] += window * window;
        }
    }
}

size_t stem_backend_process(
    stem_backend_t *backend,
    const stem_spectrum_block_t *spectrum,
    const int16_t output_nhwc8[STEM_PACKED_VALUES],
    float vocal_lr[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS])
{
    uint64_t finalized_end;
    size_t produced;

    if (backend == NULL || spectrum == NULL || output_nhwc8 == NULL
        || vocal_lr == NULL || backend->inverse_fft == NULL) {
        if (backend != NULL)
            backend->last_result = STEM_BACKEND_E_INVALID;
        return 0u;
    }
    if (spectrum->first_sample < 0
        || (uint64_t)spectrum->first_sample != backend->expected_first_sample) {
        backend->last_result = STEM_BACKEND_E_SEQUENCE;
        return 0u;
    }
    for (size_t frame = 0u; frame < STEM_BLOCK_FRAMES; ++frame) {
        int64_t center = spectrum->first_sample
            + (int64_t)(frame * STEM_HOP_SIZE);
        int64_t frame_start = center - (int64_t)(STEM_FFT_SIZE / 2u);
        for (size_t channel = 0u; channel < STEM_INPUT_CHANNELS; ++channel) {
            synthesize_frame(backend, spectrum, output_nhwc8, channel, frame,
                             frame_start);
        }
    }
    finalized_end = backend->expected_first_sample
        + STEM_BLOCK_SAMPLES - STEM_FFT_SIZE / 2u;
    produced = (size_t)(finalized_end - backend->output_cursor);
    if (produced > STEM_BLOCK_SAMPLES) {
        backend->last_result = STEM_BACKEND_E_SEQUENCE;
        return 0u;
    }
    for (size_t frame = 0u; frame < produced; ++frame) {
        size_t slot = ola_slot(backend->output_cursor + frame);
        float weight = backend->window_square[slot];
        for (size_t channel = 0u; channel < STEM_INPUT_CHANNELS; ++channel) {
            vocal_lr[frame][channel] = weight > STEM_OLA_EPSILON
                ? backend->accumulation[channel][slot] / weight : 0.0f;
            backend->accumulation[channel][slot] = 0.0f;
        }
        backend->window_square[slot] = 0.0f;
    }
    backend->output_cursor = finalized_end;
    backend->expected_first_sample += STEM_BLOCK_SAMPLES;
    backend->last_result = STEM_BACKEND_OK;
    return produced;
}

size_t stem_delay_mix(
    stem_backend_t *backend,
    const float input_lr[][STEM_INPUT_CHANNELS],
    float delayed_lr[][STEM_INPUT_CHANNELS],
    size_t frames)
{
    uint64_t available_end;
    size_t produced;

    if (backend == NULL || input_lr == NULL || delayed_lr == NULL
        || frames > STEM_BLOCK_SAMPLES)
        return 0u;
    for (size_t frame = 0u; frame < frames; ++frame) {
        size_t slot = ola_slot(backend->delayed_written + frame);
        backend->delayed_pcm[slot][0] = input_lr[frame][0];
        backend->delayed_pcm[slot][1] = input_lr[frame][1];
    }
    backend->delayed_written += frames;
    available_end = backend->delayed_written > STEM_FFT_SIZE / 2u
        ? backend->delayed_written - STEM_FFT_SIZE / 2u : 0u;
    produced = (size_t)(available_end - backend->delayed_output);
    if (produced > STEM_BLOCK_SAMPLES)
        return 0u;
    for (size_t frame = 0u; frame < produced; ++frame) {
        size_t slot = ola_slot(backend->delayed_output + frame);
        delayed_lr[frame][0] = backend->delayed_pcm[slot][0];
        delayed_lr[frame][1] = backend->delayed_pcm[slot][1];
    }
    backend->delayed_output = available_end;
    return produced;
}

int16_t stem_float_to_i16(float sample)
{
    long quantized;

    if (sample >= 1.0f)
        return INT16_MAX;
    if (sample <= -1.0f)
        return INT16_MIN;
    quantized = lrintf(sample * 32768.0f);
    if (quantized > INT16_MAX)
        return INT16_MAX;
    if (quantized < INT16_MIN)
        return INT16_MIN;
    return (int16_t)quantized;
}
