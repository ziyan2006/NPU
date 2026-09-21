/* SPDX-License-Identifier: MIT */
#ifndef STEM_BACKEND_H
#define STEM_BACKEND_H

#include "stem_frontend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STEM_BACKEND_STARTUP_FRAMES \
    (STEM_BLOCK_SAMPLES - STEM_FFT_SIZE / 2u)
#define STEM_BACKEND_OLA_CAPACITY 8192u
#define STEM_BACKEND_FFT_CONFIG_BYTES 16384u
#define STEM_BACKEND_PROTECTED_BANDS 44u

typedef enum {
    STEM_BACKEND_OK = 0,
    STEM_BACKEND_E_INVALID = -1,
    STEM_BACKEND_E_CONFIG = -2,
    STEM_BACKEND_E_SEQUENCE = -3
} stem_backend_result_t;

typedef union {
    uint64_t alignment;
    unsigned char bytes[STEM_BACKEND_FFT_CONFIG_BYTES];
} stem_backend_fft_storage_t;

typedef struct {
    kiss_fftr_cfg inverse_fft;
    stem_backend_fft_storage_t fft_storage;
    kiss_fft_cpx masked_bins[STEM_FFT_BINS];
    float inverse_samples[STEM_FFT_SIZE];
    float window[STEM_FFT_SIZE];
    uint16_t synthesis_begin[STEM_FFT_BINS];
    uint16_t synthesis_end[STEM_FFT_BINS];
    float accumulation[STEM_INPUT_CHANNELS][STEM_BACKEND_OLA_CAPACITY];
    float window_square[STEM_BACKEND_OLA_CAPACITY];
    float delayed_pcm[STEM_BACKEND_OLA_CAPACITY][STEM_INPUT_CHANNELS];
    uint64_t output_cursor;
    uint64_t expected_first_sample;
    uint64_t delayed_written;
    uint64_t delayed_output;
    stem_backend_result_t last_result;
} stem_backend_t;

stem_backend_result_t stem_backend_init(stem_backend_t *backend);
float stem_decode_mask_q11(int16_t quantized);
size_t stem_backend_process(
    stem_backend_t *backend,
    const stem_spectrum_block_t *spectrum,
    const int16_t output_nhwc8[STEM_PACKED_VALUES],
    float vocal_lr[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS]);
size_t stem_delay_mix(
    stem_backend_t *backend,
    const float input_lr[][STEM_INPUT_CHANNELS],
    float delayed_lr[][STEM_INPUT_CHANNELS],
    size_t frames);
int16_t stem_float_to_i16(float sample);

#ifdef __cplusplus
}
#endif

#endif /* STEM_BACKEND_H */
