/* SPDX-License-Identifier: MIT */
#include "stem_frontend.h"
#include "stem_filterbank.h"

#include <math.h>
#include <string.h>

#define STEM_PI_F 3.14159265358979323846f

static uint64_t earliest_retained_sample(const stem_frontend_t *frontend)
{
    uint64_t center = frontend->next_frame * STEM_HOP_SIZE;
    return center > STEM_FFT_SIZE / 2u ? center - STEM_FFT_SIZE / 2u : 0u;
}

static int16_t pcm_sample(const stem_frontend_t *frontend, int64_t index,
                          size_t channel)
{
    uint64_t reflected = index < 0 ? (uint64_t)(-index) : (uint64_t)index;
    return frontend->pcm[reflected % STEM_FRONTEND_PCM_CAPACITY][channel];
}

static int16_t quantize_band(float value)
{
    long quantized = lrintf(value / STEM_INPUT_SCALE);
    if (quantized < -2048L) {
        quantized = -2048L;
    } else if (quantized > 2047L) {
        quantized = 2047L;
    }
    return (int16_t)quantized;
}

stem_frontend_result_t stem_frontend_init(stem_frontend_t *frontend)
{
    if (frontend == NULL) {
        return STEM_FRONTEND_E_INVALID;
    }
    memset(frontend, 0, sizeof(*frontend));
    size_t config_bytes = sizeof(frontend->fft_storage.bytes);
    frontend->fft = kiss_fftr_alloc(
        (int)STEM_FFT_SIZE, 0, frontend->fft_storage.bytes, &config_bytes
    );
    if (frontend->fft == NULL || config_bytes > sizeof(frontend->fft_storage.bytes)) {
        return STEM_FRONTEND_E_CONFIG;
    }
    for (size_t sample = 0; sample < STEM_FFT_SIZE; ++sample) {
        float phase = 2.0f * STEM_PI_F * (float)sample / (float)STEM_FFT_SIZE;
        frontend->window[sample] = 0.5f - 0.5f * cosf(phase);
    }
    return STEM_FRONTEND_OK;
}

size_t stem_frontend_push(stem_frontend_t *frontend, const int16_t *lr,
                          size_t frames)
{
    if (frontend == NULL || (lr == NULL && frames != 0u)) {
        return 0u;
    }
    uint64_t retained = earliest_retained_sample(frontend);
    uint64_t occupied = frontend->samples_written - retained;
    size_t space = occupied < STEM_FRONTEND_PCM_CAPACITY
        ? (size_t)(STEM_FRONTEND_PCM_CAPACITY - occupied) : 0u;
    size_t accepted = frames < space ? frames : space;
    for (size_t frame = 0; frame < accepted; ++frame) {
        size_t slot = (size_t)((frontend->samples_written + frame)
                              % STEM_FRONTEND_PCM_CAPACITY);
        frontend->pcm[slot][0] = lr[frame * 2u];
        frontend->pcm[slot][1] = lr[frame * 2u + 1u];
    }
    frontend->samples_written += accepted;
    return accepted;
}

int stem_frontend_block_ready(const stem_frontend_t *frontend)
{
    if (frontend == NULL || frontend->fft == NULL) {
        return 0;
    }
    uint64_t final_frame = frontend->next_frame + STEM_BLOCK_FRAMES - 1u;
    uint64_t required_samples = final_frame * STEM_HOP_SIZE
        + STEM_FFT_SIZE / 2u;
    return frontend->samples_written >= required_samples;
}

stem_frontend_result_t stem_frontend_pack(
    stem_frontend_t *frontend,
    int16_t nhwc8[STEM_PACKED_VALUES],
    stem_spectrum_block_t *spectrum)
{
    if (frontend == NULL || nhwc8 == NULL || spectrum == NULL) {
        return STEM_FRONTEND_E_INVALID;
    }
    if (!stem_frontend_block_ready(frontend)) {
        return STEM_FRONTEND_E_NOT_READY;
    }
    memset(nhwc8, 0, sizeof(*nhwc8) * STEM_PACKED_VALUES);
    spectrum->first_sample = (int64_t)(frontend->next_frame * STEM_HOP_SIZE);
    for (size_t channel = 0; channel < STEM_INPUT_CHANNELS; ++channel) {
        for (size_t frame = 0; frame < STEM_BLOCK_FRAMES; ++frame) {
            int64_t center = (int64_t)(frontend->next_frame + frame)
                * (int64_t)STEM_HOP_SIZE;
            for (size_t sample = 0; sample < STEM_FFT_SIZE; ++sample) {
                int64_t absolute = center + (int64_t)sample
                    - (int64_t)(STEM_FFT_SIZE / 2u);
                frontend->fft_input[sample] =
                    ((float)pcm_sample(frontend, absolute, channel) / 32768.0f)
                    * frontend->window[sample];
            }
            kiss_fftr(frontend->fft, frontend->fft_input,
                      spectrum->bins[channel][frame]);
            float magnitude[STEM_FFT_BINS];
            for (size_t bin = 0; bin < STEM_FFT_BINS; ++bin) {
                float real = spectrum->bins[channel][frame][bin].r;
                float imaginary = spectrum->bins[channel][frame][bin].i;
                magnitude[bin] = hypotf(real, imaginary);
            }
            for (size_t band = 0; band < STEM_BAND_COUNT; ++band) {
                float value = 0.0f;
                for (size_t bin = 0; bin < STEM_FFT_BINS; ++bin) {
                    value += magnitude[bin] * stem_analysis[bin][band];
                }
                size_t index = (band * STEM_BLOCK_FRAMES + frame)
                    * STEM_NHWC8_LANES + channel;
                nhwc8[index] = quantize_band(value);
            }
        }
    }
    frontend->next_frame += STEM_BLOCK_FRAMES;
    return STEM_FRONTEND_OK;
}
