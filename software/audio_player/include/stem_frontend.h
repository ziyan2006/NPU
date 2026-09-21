/* SPDX-License-Identifier: MIT */
#ifndef STEM_FRONTEND_H
#define STEM_FRONTEND_H

#include "stem_contract.h"
#include "kiss_fftr.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STEM_FRONTEND_PCM_CAPACITY 8192u
#define STEM_FFT_CONFIG_BYTES      16384u
#define STEM_INPUT_SCALE 0.07046897899364925f

typedef enum {
    STEM_FRONTEND_OK = 0,
    STEM_FRONTEND_E_INVALID = -1,
    STEM_FRONTEND_E_CONFIG = -2,
    STEM_FRONTEND_E_NOT_READY = -3
} stem_frontend_result_t;

typedef struct {
    kiss_fft_cpx bins[STEM_INPUT_CHANNELS][STEM_BLOCK_FRAMES][STEM_FFT_BINS];
    int64_t first_sample;
} stem_spectrum_block_t;

typedef union {
    uint64_t alignment;
    unsigned char bytes[STEM_FFT_CONFIG_BYTES];
} stem_fft_config_storage_t;

typedef struct {
    kiss_fftr_cfg fft;
    stem_fft_config_storage_t fft_storage;
    float window[STEM_FFT_SIZE];
    float fft_input[STEM_FFT_SIZE];
    uint16_t analysis_begin[STEM_BAND_COUNT];
    uint16_t analysis_end[STEM_BAND_COUNT];
    int16_t pcm[STEM_FRONTEND_PCM_CAPACITY][STEM_INPUT_CHANNELS];
    uint64_t samples_written;
    uint64_t next_frame;
} stem_frontend_t;

stem_frontend_result_t stem_frontend_init(stem_frontend_t *frontend);
size_t stem_frontend_push(stem_frontend_t *frontend, const int16_t *lr,
                          size_t frames);
int stem_frontend_block_ready(const stem_frontend_t *frontend);
stem_frontend_result_t stem_frontend_pack(
    stem_frontend_t *frontend,
    int16_t nhwc8[STEM_PACKED_VALUES],
    stem_spectrum_block_t *spectrum);

#ifdef __cplusplus
}
#endif

#endif /* STEM_FRONTEND_H */
