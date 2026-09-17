/* SPDX-License-Identifier: MIT */
#ifndef STEM_SD_MP3_SOURCE_H
#define STEM_SD_MP3_SOURCE_H

#include <stddef.h>
#include <stdint.h>

#include "minimp3.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MP3_SOURCE_BUFFER_BYTES 16384u
#define MP3_SOURCE_MAX_RESYNC_BYTES 65536u
#define MP3_SOURCE_FRAME_SAMPLES 1152u

typedef size_t (*mp3_read_fn)(void *context, void *destination, size_t bytes);

typedef struct {
    mp3_read_fn read;
    void *context;
} mp3_io_t;

typedef enum {
    MP3_OK = 0,
    MP3_EOF = 1,
    MP3_E_INVALID = -1,
    MP3_E_LAYER = -2,
    MP3_E_RATE = -3,
    MP3_E_CHANNELS = -4,
    MP3_E_SYNC = -5,
    MP3_E_IO = -6
} mp3_result_t;

typedef struct {
    mp3_io_t io;
    mp3dec_t decoder;
    uint8_t buffer[MP3_SOURCE_BUFFER_BYTES];
    size_t buffer_begin;
    size_t buffer_end;
    int16_t pending_pcm[MP3_SOURCE_FRAME_SAMPLES * 2u];
    size_t pending_begin;
    size_t pending_frames;
    uint32_t skipped_bytes;
    uint32_t decoded_frames;
    uint8_t end_of_stream;
    uint8_t format_locked;
    uint8_t io_error;
} mp3_source_t;

mp3_result_t mp3_source_open(mp3_source_t *source, const mp3_io_t *io);
mp3_result_t mp3_source_decode(mp3_source_t *source,
                               int16_t *interleaved_lr,
                               size_t frame_capacity,
                               size_t *frames_out);

#ifdef __cplusplus
}
#endif

#endif /* STEM_SD_MP3_SOURCE_H */
