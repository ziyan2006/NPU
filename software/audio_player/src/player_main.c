/* SPDX-License-Identifier: MIT */
#include "audio_hw.h"
#include "player_platform.h"
#if defined(PLAYER_MODE_MP3_BYPASS) || defined(PLAYER_MODE_FULL_STEM)
#include "sd_mp3_source.h"
#endif
#if defined(PLAYER_MODE_FULL_STEM)
#include "npu_driver.h"
#include "player.h"
#include "stem_backend.h"
#include "stem_frontend.h"
#include "stem_npu_session.h"
#include "stem_task_metadata.h"
#include "xil_cache.h"
#include "xil_types.h"
#endif
#if defined(PLAYER_MODE_WAV)
#include "wav_source.h"
#endif

#include <stddef.h>
#include <stdint.h>

#if (defined(PLAYER_MODE_TONE) + defined(PLAYER_MODE_WAV) \
     + defined(PLAYER_MODE_MP3_BYPASS) + defined(PLAYER_MODE_FULL_STEM)) != 1
#error "build must define exactly one player mode"
#endif

#define PLAYER_LEGACY_PREFILL_FRAMES 8192u
#define PLAYER_BATCH_FRAMES 128u
#define PLAYER_FADE_FRAMES 1323u
#define PLAYER_CODEC_TIMEOUT_MS 3000u
#define PLAYER_TONE_CONTROL 0x100005ceu

static int wait_for_codec(void)
{
    uint32_t start = player_platform_milliseconds();

    while (player_platform_milliseconds() - start < PLAYER_CODEC_TIMEOUT_MS) {
        uint32_t status = player_platform_audio_read(AUDIO_HW_REG_STATUS);
        if ((status & AUDIO_HW_STATUS_CODEC_ERROR) != 0u)
            return 0;
        if ((status & AUDIO_HW_STATUS_CODEC_DONE) != 0u)
            return 1;
    }
    return 0;
}

#if !defined(PLAYER_MODE_FULL_STEM)
static void report_status(uint32_t *next_report, uint32_t *minimum_level)
{
    uint32_t now = player_platform_milliseconds();
    uint32_t level = player_platform_audio_read(AUDIO_HW_REG_FIFO_LEVEL);

    if (level < *minimum_level)
        *minimum_level = level;
    if ((int32_t)(now - *next_report) < 0)
        return;
    player_platform_status(
        player_platform_audio_read(AUDIO_HW_REG_PLAYED_FRAMES), level,
        *minimum_level,
        player_platform_audio_read(AUDIO_HW_REG_UNDERFLOW_COUNT),
        player_platform_audio_read(AUDIO_HW_REG_OVERFLOW_COUNT),
        player_platform_audio_read(AUDIO_HW_REG_CODEC_STATUS));
    *next_report = now + 1000u;
}
#endif

#if defined(PLAYER_MODE_TONE)
static int run_player(audio_hw_t *hardware)
{
    uint32_t next_report = player_platform_milliseconds();
    uint32_t minimum_level = AUDIO_HW_FIFO_CAPACITY;

    (void)hardware;
    player_platform_audio_write(AUDIO_HW_REG_TONE_CONTROL,
                                PLAYER_TONE_CONTROL);
    player_platform_audio_write(AUDIO_HW_REG_CONTROL,
                                AUDIO_HW_CONTROL_TONE_ENABLE);
    player_platform_log("PLAY");
    for (;;) {
        report_status(&next_report, &minimum_level);
        player_platform_delay_ms(10u);
    }
    return 1;
}
#elif defined(PLAYER_MODE_WAV)
static uint32_t minimum_u32(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

static int16_t fade_sample(int16_t sample, uint32_t remaining)
{
    int32_t scaled;

    if (remaining >= PLAYER_FADE_FRAMES)
        return sample;
    scaled = (int32_t)sample * (int32_t)remaining;
    return (int16_t)(scaled / (int32_t)PLAYER_FADE_FRAMES);
}

static size_t read_wav_frames(wav_source_t *source, int16_t *samples,
                              size_t frames)
{
    return wav_source_read(source, samples, frames);
}

static int submit_frames(audio_hw_t *hardware, const int16_t *samples,
                         size_t frames, uint32_t first_frame,
                         uint32_t total_frames)
{
    size_t index;

    for (index = 0u; index < frames; ++index) {
        uint32_t absolute = first_frame + (uint32_t)index;
        uint32_t remaining = total_frames - absolute;
        audio_frame_t frame;

        frame.mix_left = fade_sample(samples[index * 2u], remaining);
        frame.mix_right = fade_sample(samples[index * 2u + 1u], remaining);
        frame.vocal_left = 0;
        frame.vocal_right = 0;
        if (audio_hw_write_frame(hardware, &frame) != AUDIO_HW_OK)
            return 0;
    }
    return 1;
}

static int run_player(audio_hw_t *hardware)
{
    int16_t samples[PLAYER_BATCH_FRAMES * 2u];
    wav_source_t source;
    uint32_t submitted = 0u;
    uint32_t next_report;
    uint32_t minimum_level = AUDIO_HW_FIFO_CAPACITY;

    player_platform_log("MOUNT");
    if (!player_platform_mount()) {
        player_platform_log("FAIL MOUNT");
        return 0;
    }
    player_platform_log("OPEN_WAV");
    if (!player_platform_open_wav()
        || wav_source_open(&source, player_platform_wav_read, NULL)
           != WAV_SOURCE_OK) {
        player_platform_log("FAIL OPEN_WAV");
        return 0;
    }
    if (source.total_frames < PLAYER_LEGACY_PREFILL_FRAMES) {
        player_platform_log("FAIL WAV_TOO_SHORT");
        return 0;
    }

    player_platform_log("PREFILL");
    while (submitted < PLAYER_LEGACY_PREFILL_FRAMES) {
        uint32_t wanted = minimum_u32(
            PLAYER_BATCH_FRAMES, PLAYER_LEGACY_PREFILL_FRAMES - submitted);
        size_t received = read_wav_frames(&source, samples, wanted);
        if (received != wanted
            || !submit_frames(hardware, samples, received, submitted,
                              source.total_frames)) {
            player_platform_log("FAIL WAV_TOO_SHORT");
            return 0;
        }
        submitted += (uint32_t)received;
    }

    player_platform_audio_write(AUDIO_HW_REG_CONTROL,
                                AUDIO_HW_CONTROL_ENABLE);
    player_platform_log("PLAY");
    next_report = player_platform_milliseconds();
    while (submitted < source.total_frames) {
        uint32_t space = (uint32_t)audio_hw_space(hardware);
        uint32_t wanted = minimum_u32(PLAYER_BATCH_FRAMES, space);
        size_t received;

        wanted = minimum_u32(wanted, source.total_frames - submitted);
        if (wanted == 0u) {
            report_status(&next_report, &minimum_level);
            continue;
        }
        received = read_wav_frames(&source, samples, wanted);
        if (received == 0u)
            break;
        if (!submit_frames(hardware, samples, received, submitted,
                           source.total_frames)) {
            player_platform_log("FAIL FIFO_WRITE");
            return 0;
        }
        submitted += (uint32_t)received;
        report_status(&next_report, &minimum_level);
    }
    if (submitted != source.total_frames) {
        player_platform_log("FAIL WAV_READ");
        return 0;
    }
    while (player_platform_audio_read(AUDIO_HW_REG_FIFO_LEVEL) != 0u)
        report_status(&next_report, &minimum_level);
    player_platform_audio_write(AUDIO_HW_REG_CONTROL, 0u);
    player_platform_log("EOF");
    return 1;
}
#elif defined(PLAYER_MODE_MP3_BYPASS)
static uint32_t minimum_u32(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

static int submit_mp3_frames(audio_hw_t *hardware, const int16_t *samples,
                             size_t frames)
{
    size_t index;

    for (index = 0u; index < frames; ++index) {
        audio_frame_t frame;

        frame.mix_left = samples[index * 2u];
        frame.mix_right = samples[index * 2u + 1u];
        frame.vocal_left = 0;
        frame.vocal_right = 0;
        if (audio_hw_write_frame(hardware, &frame) != AUDIO_HW_OK)
            return 0;
    }
    return 1;
}

static const char *mp3_error_name(mp3_result_t result)
{
    switch (result) {
    case MP3_E_LAYER: return "FAIL MP3_LAYER";
    case MP3_E_RATE: return "FAIL MP3_RATE";
    case MP3_E_CHANNELS: return "FAIL MP3_CHANNELS";
    case MP3_E_SYNC: return "FAIL MP3_SYNC";
    case MP3_E_IO: return "FAIL MP3_IO";
    default: return "FAIL MP3_DECODE";
    }
}

static int run_player(audio_hw_t *hardware)
{
    static mp3_source_t source;
    int16_t samples[PLAYER_BATCH_FRAMES * 2u];
    const mp3_io_t io = {player_platform_mp3_read, NULL};
    uint32_t submitted = 0u;
    uint32_t next_report;
    uint32_t minimum_level = AUDIO_HW_FIFO_CAPACITY;
    mp3_result_t result;

    player_platform_log("MOUNT");
    if (!player_platform_mount()) {
        player_platform_log("FAIL MOUNT");
        return 0;
    }
    player_platform_log("OPEN_MP3");
    if (!player_platform_open_mp3()) {
        player_platform_log("FAIL OPEN_MP3");
        return 0;
    }
    result = mp3_source_open(&source, &io);
    if (result != MP3_OK) {
        player_platform_log(mp3_error_name(result));
        return 0;
    }

    player_platform_log("PREFILL");
    while (submitted < PLAYER_LEGACY_PREFILL_FRAMES) {
        uint32_t wanted = minimum_u32(
            PLAYER_BATCH_FRAMES, PLAYER_LEGACY_PREFILL_FRAMES - submitted);
        size_t received = 0u;

        result = mp3_source_decode(&source, samples, wanted, &received);
        if (result != MP3_OK || received == 0u) {
            player_platform_log(result == MP3_EOF
                                ? "FAIL MP3_TOO_SHORT"
                                : mp3_error_name(result));
            return 0;
        }
        if (!submit_mp3_frames(hardware, samples, received)) {
            player_platform_log("FAIL FIFO_WRITE");
            return 0;
        }
        submitted += (uint32_t)received;
    }

    player_platform_audio_write(AUDIO_HW_REG_CONTROL,
                                AUDIO_HW_CONTROL_ENABLE);
    player_platform_log("PLAY");
    next_report = player_platform_milliseconds();
    for (;;) {
        uint32_t wanted = minimum_u32(
            PLAYER_BATCH_FRAMES, (uint32_t)audio_hw_space(hardware));
        size_t received = 0u;

        if (wanted == 0u) {
            report_status(&next_report, &minimum_level);
            continue;
        }
        result = mp3_source_decode(&source, samples, wanted, &received);
        if (result == MP3_EOF)
            break;
        if (result != MP3_OK || received == 0u) {
            player_platform_log(mp3_error_name(result));
            return 0;
        }
        if (!submit_mp3_frames(hardware, samples, received)) {
            player_platform_log("FAIL FIFO_WRITE");
            return 0;
        }
        report_status(&next_report, &minimum_level);
    }
    while (player_platform_audio_read(AUDIO_HW_REG_FIFO_LEVEL) != 0u)
        report_status(&next_report, &minimum_level);
    player_platform_audio_write(AUDIO_HW_REG_CONTROL, 0u);
    player_platform_log("EOF");
    return 1;
}
#else
#define PLAYER_NPU_BASE_ADDRESS 0x43c00000u
#define PLAYER_TASK_DDR_ADDRESS 0x01000000u

typedef struct {
    audio_hw_t *hardware;
    mp3_source_t source;
    stem_frontend_t frontend;
    stem_npu_session_t npu_session;
    stem_backend_t backend;
    npu_device_t npu_device;
    stem_spectrum_block_t spectrum;
    int16_t npu_input[STEM_PACKED_VALUES];
    int16_t npu_output[STEM_PACKED_VALUES];
    float normalized[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS];
    uint64_t decode_us;
    uint32_t processed_blocks;
    uint8_t npu_ready;
} full_stem_context_t;

static full_stem_context_t full_context;
static player_t full_player;

static void full_cache_clean(void *address, size_t bytes, void *context)
{
    (void)context;
    while (bytes != 0u) {
        u32 chunk = bytes > UINT32_MAX ? UINT32_MAX : (u32)bytes;
        Xil_DCacheFlushRange((INTPTR)address, chunk);
        address = (uint8_t *)address + chunk;
        bytes -= chunk;
    }
}

static void full_cache_invalidate(void *address, size_t bytes, void *context)
{
    (void)context;
    while (bytes != 0u) {
        u32 chunk = bytes > UINT32_MAX ? UINT32_MAX : (u32)bytes;
        Xil_DCacheInvalidateRange((INTPTR)address, chunk);
        address = (uint8_t *)address + chunk;
        bytes -= chunk;
    }
}

static void full_barrier(void *context)
{
    (void)context;
    __asm__ volatile("dmb sy" ::: "memory");
}

static int full_self_test(void *context)
{
    (void)context;
    return wait_for_codec();
}

static int full_mount(void *context)
{
    (void)context;
    return player_platform_mount();
}

static int full_open(void *context)
{
    full_stem_context_t *full = context;
    const mp3_io_t io = {player_platform_mp3_read, NULL};

    if (!player_platform_open_mp3())
        return 0;
    return mp3_source_open(&full->source, &io) == MP3_OK;
}

static player_decode_result_t full_decode(void *context, int16_t *lr,
                                          size_t capacity, size_t *frames)
{
    full_stem_context_t *full = context;
    uint64_t start = player_platform_microseconds();
    mp3_result_t result = mp3_source_decode(&full->source, lr, capacity, frames);
    full->decode_us += player_platform_microseconds() - start;
    if (result == MP3_OK)
        return PLAYER_DECODE_OK;
    if (result == MP3_EOF)
        return PLAYER_DECODE_EOF;
    return PLAYER_DECODE_ERROR;
}

static player_process_result_t full_process(
    void *context, const int16_t *lr, size_t frames, int bypass,
    float mix[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
    float vocal[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
    size_t *output_frames, player_block_timing_t *timing)
{
    full_stem_context_t *full = context;
    uint64_t start;
    size_t pushed;
    size_t backend_frames;
    size_t delay_frames;
    npu_result_t npu_result = NPU_OK;
    stem_npu_stats_t npu_stats;

    if (frames != PLAYER_ANALYSIS_WINDOW_FRAMES)
        return PLAYER_PROCESS_ERROR;
    start = player_platform_microseconds();
    if (full->processed_blocks == 0u) {
        pushed = stem_frontend_push(&full->frontend, lr, frames);
    } else {
        pushed = stem_frontend_push(
            &full->frontend, lr + PLAYER_LOOKAHEAD_FRAMES * 2u,
            STEM_BLOCK_SAMPLES);
    }
    if (pushed != (full->processed_blocks == 0u
                   ? PLAYER_ANALYSIS_WINDOW_FRAMES : STEM_BLOCK_SAMPLES)
        || stem_frontend_pack(&full->frontend, full->npu_input,
                              &full->spectrum) != STEM_FRONTEND_OK)
        return PLAYER_PROCESS_ERROR;
    timing->decode_frontend_us = (uint32_t)(
        full->decode_us + player_platform_microseconds() - start);
    full->decode_us = 0u;

    start = player_platform_microseconds();
    if (!bypass && full->npu_ready) {
        npu_result = stem_npu_run_block(&full->npu_session, full->npu_input,
                                       full->npu_output, &npu_stats);
    } else {
        npu_result = bypass ? NPU_OK : NPU_E_HARDWARE;
    }
    timing->npu_us = (uint32_t)(player_platform_microseconds() - start);
    if (bypass || npu_result != NPU_OK)
        memset(full->npu_output, 0, sizeof(full->npu_output));

    start = player_platform_microseconds();
    for (size_t frame = 0u; frame < STEM_BLOCK_SAMPLES; ++frame) {
        full->normalized[frame][0] = (float)lr[frame * 2u] / 32768.0f;
        full->normalized[frame][1] = (float)lr[frame * 2u + 1u] / 32768.0f;
    }
    backend_frames = stem_backend_process(&full->backend, &full->spectrum,
                                          full->npu_output, vocal);
    delay_frames = stem_delay_mix(&full->backend, full->normalized, mix,
                                  STEM_BLOCK_SAMPLES);
    timing->backend_sink_us = (uint32_t)(
        player_platform_microseconds() - start);
    if (backend_frames == 0u || backend_frames != delay_frames)
        return PLAYER_PROCESS_ERROR;
    *output_frames = backend_frames;
    ++full->processed_blocks;
    return npu_result == NPU_OK
        ? PLAYER_PROCESS_OK : PLAYER_PROCESS_NPU_ERROR;
}

static size_t full_audio_space(void *context)
{
    return audio_hw_space(((full_stem_context_t *)context)->hardware);
}

static int full_audio_write(void *context, const audio_frame_t *frame)
{
    return audio_hw_write_frame(((full_stem_context_t *)context)->hardware,
                                frame) == AUDIO_HW_OK;
}

static void full_audio_enable(void *context, int enabled)
{
    (void)context;
    player_platform_audio_write(AUDIO_HW_REG_CONTROL,
        enabled ? AUDIO_HW_CONTROL_ENABLE : 0u);
}

static void full_audio_status(void *context, player_audio_status_t *status)
{
    uint32_t stem_state;
    uint32_t hardware_status;
    (void)context;
    memset(status, 0, sizeof(*status));
    stem_state = player_platform_audio_read(AUDIO_HW_REG_STEM_STATE);
    hardware_status = player_platform_audio_read(AUDIO_HW_REG_STATUS);
    status->fifo_level = (uint16_t)player_platform_audio_read(
        AUDIO_HW_REG_FIFO_LEVEL);
    status->underflows = player_platform_audio_read(
        AUDIO_HW_REG_UNDERFLOW_COUNT);
    status->overflows = player_platform_audio_read(
        AUDIO_HW_REG_OVERFLOW_COUNT);
    status->stem_target = (stem_state & AUDIO_HW_STEM_TARGET) != 0u;
    status->stem_ramping = (stem_state & AUDIO_HW_STEM_RAMPING) != 0u;
    status->codec_error = (hardware_status & AUDIO_HW_STATUS_CODEC_ERROR) != 0u;
}

static uint64_t full_time_us(void *context)
{
    (void)context;
    return player_platform_microseconds();
}

static void full_uart_line(void *context, const char *line)
{
    (void)context;
    player_platform_log(line);
}

static int initialize_full_stem(full_stem_context_t *full,
                                audio_hw_t *hardware)
{
    npu_platform_ops_t platform;

    memset(full, 0, sizeof(*full));
    full->hardware = hardware;
    if (stem_frontend_init(&full->frontend) != STEM_FRONTEND_OK
        || stem_backend_init(&full->backend) != STEM_BACKEND_OK)
        return 0;
    platform.clean = full_cache_clean;
    platform.invalidate = full_cache_invalidate;
    platform.barrier = full_barrier;
    platform.context = NULL;
    if (npu_device_init(&full->npu_device,
                        (volatile void *)PLAYER_NPU_BASE_ADDRESS,
                        &platform) == NPU_OK
        && stem_npu_session_init(
            &full->npu_session, &full->npu_device,
            PLAYER_TASK_DDR_ADDRESS, (void *)PLAYER_TASK_DDR_ADDRESS,
            STEM_TASK_IMAGE_BYTES) == NPU_OK) {
        full->npu_ready = 1u;
    }
    return 1;
}

static int run_player(audio_hw_t *hardware)
{
    player_deps_t deps;

    if (!initialize_full_stem(&full_context, hardware))
        return 0;
    memset(&deps, 0, sizeof(deps));
    deps.context = &full_context;
    deps.self_test = full_self_test;
    deps.mount = full_mount;
    deps.open = full_open;
    deps.decode = full_decode;
    deps.process = full_process;
    deps.audio_space = full_audio_space;
    deps.audio_write = full_audio_write;
    deps.audio_enable = full_audio_enable;
    deps.audio_status = full_audio_status;
    deps.time_us = full_time_us;
    deps.uart_line = full_uart_line;
    if (player_init(&full_player, &deps) != PLAYER_OK)
        return 0;
    player_platform_log("FULL_STEM");
    while (full_player.state != PLAYER_DONE
           && full_player.state != PLAYER_FAIL_MUTE)
        player_step(&full_player);
    return full_player.state == PLAYER_DONE;
}
#endif

int player_main(void)
{
    audio_hw_t hardware;
    int result = 0;

    if (!player_platform_init())
        return 1;
    player_platform_log("BOOT");
    if (audio_hw_init(&hardware, player_platform_audio_base()) != AUDIO_HW_OK) {
        player_platform_log("FAIL AUDIO_HW");
        goto done;
    }
    if (!wait_for_codec()) {
        player_platform_log("FAIL CODEC");
        goto done;
    }
    player_platform_log("CODEC_READY");
    result = run_player(&hardware);
done:
    player_platform_audio_write(AUDIO_HW_REG_CONTROL, 0u);
    player_platform_shutdown();
    return result ? 0 : 1;
}

int main(void)
{
    return player_main();
}
