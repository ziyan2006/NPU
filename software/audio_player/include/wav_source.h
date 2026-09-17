/* SPDX-License-Identifier: MIT */
#ifndef STEM_WAV_SOURCE_H
#define STEM_WAV_SOURCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t (*wav_read_fn)(void *context, void *destination, size_t bytes);

typedef enum {
    WAV_SOURCE_OK = 0,
    WAV_SOURCE_E_INVALID = -1,
    WAV_SOURCE_E_IO = -2,
    WAV_SOURCE_E_FORMAT = -3,
    WAV_SOURCE_E_UNSUPPORTED = -4
} wav_source_result_t;

typedef struct {
    wav_read_fn read;
    void *context;
    uint32_t data_bytes_remaining;
    uint32_t total_frames;
    uint8_t error;
} wav_source_t;

wav_source_result_t wav_source_open(wav_source_t *source,
                                    wav_read_fn read,
                                    void *context);
size_t wav_source_read(wav_source_t *source, int16_t *interleaved_lr,
                       size_t frame_count);

#ifdef __cplusplus
}
#endif

#endif /* STEM_WAV_SOURCE_H */
