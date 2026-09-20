/* SPDX-License-Identifier: MIT */
#ifndef STEM_PLAYER_H
#define STEM_PLAYER_H

#include "audio_hw.h"
#include "pcm_ring.h"
#include "stem_backend.h"
#include "stem_contract.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PLAYER_PCM_RING_FRAMES 8192u
#define PLAYER_LOOKAHEAD_FRAMES (STEM_FFT_SIZE / 4u)
#define PLAYER_ANALYSIS_WINDOW_FRAMES \
    (STEM_BLOCK_SAMPLES + PLAYER_LOOKAHEAD_FRAMES)
#define PLAYER_PREFILL_FRAMES \
    (STEM_BACKEND_STARTUP_FRAMES + 3u * STEM_BLOCK_SAMPLES)
#define PLAYER_FADE_FRAMES 1323u
#define PLAYER_BLOCK_DEADLINE_US 92880u
#define PLAYER_TELEMETRY_LINE_BYTES 512u

_Static_assert(PLAYER_PREFILL_FRAMES <= AUDIO_HW_FIFO_CAPACITY,
               "FullStem prefill must fit in the hardware FIFO");

typedef enum {
    PLAYER_BOOT = 0,
    PLAYER_SELF_TEST,
    PLAYER_MOUNT,
    PLAYER_OPEN,
    PLAYER_PREFILL,
    PLAYER_PLAY,
    PLAYER_BYPASS,
    PLAYER_FADE,
    PLAYER_FAIL_MUTE,
    PLAYER_DONE
} player_state_t;

typedef enum {
    PLAYER_OK = 0,
    PLAYER_E_INVALID = -1
} player_result_t;

typedef enum {
    PLAYER_DECODE_OK = 0,
    PLAYER_DECODE_EOF = 1,
    PLAYER_DECODE_ERROR = -1
} player_decode_result_t;

typedef enum {
    PLAYER_PROCESS_OK = 0,
    PLAYER_PROCESS_NPU_ERROR = 1,
    PLAYER_PROCESS_ERROR = -1
} player_process_result_t;

typedef struct {
    uint32_t decode_frontend_us;
    uint32_t npu_us;
    uint32_t backend_sink_us;
} player_block_timing_t;

typedef struct {
    uint16_t fifo_level;
    uint32_t played_frames;
    uint8_t stem_target;
    uint8_t stem_ramping;
    uint8_t codec_error;
    uint32_t underflows;
    uint32_t overflows;
} player_audio_status_t;

typedef struct {
    uint32_t seconds;
    uint32_t decoder_errors;
    uint32_t npu_errors;
    uint32_t codec_errors;
    uint32_t deadline_miss;
    uint32_t block_us_average;
    uint32_t block_us_max;
    uint32_t decode_frontend_us_average;
    uint32_t decode_frontend_us_max;
    uint32_t npu_us_average;
    uint32_t npu_us_max;
    uint32_t backend_sink_us_average;
    uint32_t backend_sink_us_max;
    uint16_t fifo_level;
    uint16_t fifo_minimum;
    uint32_t played_frames;
    uint8_t stem_target;
    uint8_t stem_ramping;
    uint32_t underflows;
    uint32_t overflows;
} player_telemetry_t;

typedef struct {
    void *context;
    int (*self_test)(void *context);
    int (*mount)(void *context);
    int (*open)(void *context);
    player_decode_result_t (*decode)(void *context, int16_t *lr,
                                     size_t capacity, size_t *frames);
    player_process_result_t (*process)(
        void *context, const int16_t *lr, size_t frames, int bypass,
        float mix[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
        float vocal[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
        size_t *output_frames, player_block_timing_t *timing);
    size_t (*audio_space)(void *context);
    int (*audio_write)(void *context, const audio_frame_t *frame);
    void (*audio_enable)(void *context, int enabled);
    void (*audio_status)(void *context, player_audio_status_t *status);
    uint64_t (*time_us)(void *context);
    void (*uart_line)(void *context, const char *line);
} player_deps_t;

typedef struct {
    player_state_t state;
    player_deps_t deps;
    pcm_ring_t pcm_ring;
    pcm_stereo_frame_t pcm_storage[PLAYER_PCM_RING_FRAMES];
    pcm_stereo_frame_t decode_buffer[PLAYER_ANALYSIS_WINDOW_FRAMES];
    pcm_stereo_frame_t analysis_window[PLAYER_ANALYSIS_WINDOW_FRAMES];
    float mix[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    float vocal[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    player_telemetry_t telemetry;
    uint64_t next_telemetry_us;
    uint64_t block_us_total;
    uint64_t decode_frontend_us_total;
    uint64_t npu_us_total;
    uint64_t backend_sink_us_total;
    uint32_t blocks_processed;
    uint32_t frames_submitted;
    uint32_t fade_progress;
    audio_frame_t last_frame;
    uint8_t audio_enabled;
    uint8_t bypass_latched;
    uint8_t end_of_stream;
} player_t;

player_result_t player_init(player_t *player, const player_deps_t *deps);
void player_step(player_t *player);
const char *player_state_name(player_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* STEM_PLAYER_H */
