/* SPDX-License-Identifier: MIT */
#include "player.h"

#include <stdio.h>
#include <string.h>

static int dependencies_valid(const player_deps_t *deps)
{
    return deps != NULL && deps->self_test != NULL && deps->mount != NULL
        && deps->open != NULL && deps->decode != NULL && deps->process != NULL
        && deps->audio_space != NULL && deps->audio_write != NULL
        && deps->audio_enable != NULL && deps->audio_status != NULL
        && deps->time_us != NULL && deps->uart_line != NULL;
}

const char *player_state_name(player_state_t state)
{
    switch (state) {
    case PLAYER_BOOT: return "BOOT";
    case PLAYER_SELF_TEST: return "SELF_TEST";
    case PLAYER_MOUNT: return "MOUNT";
    case PLAYER_OPEN: return "OPEN";
    case PLAYER_PREFILL: return "PREFILL";
    case PLAYER_PLAY: return "PLAY";
    case PLAYER_BYPASS: return "BYPASS";
    case PLAYER_FADE: return "FADE";
    case PLAYER_FAIL_MUTE: return "FAIL_MUTE";
    case PLAYER_DONE: return "DONE";
    default: return "UNKNOWN";
    }
}

player_result_t player_init(player_t *player, const player_deps_t *deps)
{
    if (player == NULL || !dependencies_valid(deps))
        return PLAYER_E_INVALID;
    memset(player, 0, sizeof(*player));
    player->deps = *deps;
    if (pcm_ring_init(&player->pcm_ring, player->pcm_storage,
                      PLAYER_PCM_RING_FRAMES) != PCM_RING_OK)
        return PLAYER_E_INVALID;
    player->state = PLAYER_BOOT;
    player->telemetry.fifo_minimum = AUDIO_HW_FIFO_CAPACITY;
    player->next_telemetry_us = deps->time_us(deps->context) + 1000000u;
    deps->audio_enable(deps->context, 0);
    return PLAYER_OK;
}

static void update_audio_status(player_t *player)
{
    player_audio_status_t status;

    memset(&status, 0, sizeof(status));
    player->deps.audio_status(player->deps.context, &status);
    player->telemetry.fifo_level = status.fifo_level;
    if (status.fifo_level < player->telemetry.fifo_minimum)
        player->telemetry.fifo_minimum = status.fifo_level;
    player->telemetry.underflows = status.underflows;
    player->telemetry.overflows = status.overflows;
    player->telemetry.stem_target = status.stem_target;
    player->telemetry.stem_ramping = status.stem_ramping;
    if (status.codec_error != 0u && player->state != PLAYER_FAIL_MUTE
        && player->state != PLAYER_DONE) {
        ++player->telemetry.codec_errors;
        player->state = PLAYER_FAIL_MUTE;
    }
}

static void emit_telemetry(player_t *player)
{
    uint64_t now = player->deps.time_us(player->deps.context);
    char line[PLAYER_TELEMETRY_LINE_BYTES];

    if (now < player->next_telemetry_us)
        return;
    player->telemetry.seconds = (uint32_t)(now / 1000000u);
    (void)snprintf(
        line, sizeof(line),
        "[AUDIO] sec=%lu state=%s stem=%u ramp=%u fifo=%u fifo_min=%u "
        "uf=%lu of=%lu dec_err=%lu npu_err=%lu codec_err=%lu "
        "deadline_miss=%lu blk_us_avg=%lu blk_us_max=%lu "
        "decfe_us_avg=%lu decfe_us_max=%lu npu_us_avg=%lu npu_us_max=%lu "
        "sink_us_avg=%lu sink_us_max=%lu",
        (unsigned long)player->telemetry.seconds,
        player_state_name(player->state),
        (unsigned)player->telemetry.stem_target,
        (unsigned)player->telemetry.stem_ramping,
        (unsigned)player->telemetry.fifo_level,
        (unsigned)player->telemetry.fifo_minimum,
        (unsigned long)player->telemetry.underflows,
        (unsigned long)player->telemetry.overflows,
        (unsigned long)player->telemetry.decoder_errors,
        (unsigned long)player->telemetry.npu_errors,
        (unsigned long)player->telemetry.codec_errors,
        (unsigned long)player->telemetry.deadline_miss,
        (unsigned long)player->telemetry.block_us_average,
        (unsigned long)player->telemetry.block_us_max,
        (unsigned long)player->telemetry.decode_frontend_us_average,
        (unsigned long)player->telemetry.decode_frontend_us_max,
        (unsigned long)player->telemetry.npu_us_average,
        (unsigned long)player->telemetry.npu_us_max,
        (unsigned long)player->telemetry.backend_sink_us_average,
        (unsigned long)player->telemetry.backend_sink_us_max);
    player->deps.uart_line(player->deps.context, line);
    player->next_telemetry_us = now + 1000000u;
}

static void fail_muted(player_t *player)
{
    if (player->state != PLAYER_FAIL_MUTE)
        player->state = PLAYER_FAIL_MUTE;
    if (player->audio_enabled) {
        player->deps.audio_enable(player->deps.context, 0);
        player->audio_enabled = 0u;
    }
}

static int fill_analysis_window(player_t *player)
{
    while (pcm_ring_size(&player->pcm_ring) < PLAYER_ANALYSIS_WINDOW_FRAMES
           && !player->end_of_stream) {
        size_t ring_space = pcm_ring_space(&player->pcm_ring);
        size_t capacity = ring_space < PLAYER_ANALYSIS_WINDOW_FRAMES
            ? ring_space : PLAYER_ANALYSIS_WINDOW_FRAMES;
        size_t frames = 0u;
        player_decode_result_t result = player->deps.decode(
            player->deps.context, (int16_t *)player->decode_buffer,
            capacity, &frames);
        if (result == PLAYER_DECODE_ERROR || frames > capacity) {
            ++player->telemetry.decoder_errors;
            fail_muted(player);
            return 0;
        }
        if (frames != 0u
            && pcm_ring_push(&player->pcm_ring, player->decode_buffer, frames)
               != frames) {
            ++player->telemetry.decoder_errors;
            fail_muted(player);
            return 0;
        }
        if (result == PLAYER_DECODE_EOF) {
            player->end_of_stream = 1u;
            break;
        }
        if (frames == 0u) {
            ++player->telemetry.decoder_errors;
            fail_muted(player);
            return 0;
        }
    }
    return pcm_ring_size(&player->pcm_ring) >= PLAYER_ANALYSIS_WINDOW_FRAMES;
}

static void record_timing(player_t *player,
                          const player_block_timing_t *timing)
{
    uint32_t total = timing->decode_frontend_us + timing->npu_us
        + timing->backend_sink_us;
    player->block_us_total += total;
    player->decode_frontend_us_total += timing->decode_frontend_us;
    player->npu_us_total += timing->npu_us;
    player->backend_sink_us_total += timing->backend_sink_us;
    ++player->blocks_processed;
    player->telemetry.block_us_average = (uint32_t)(
        player->block_us_total / player->blocks_processed);
    if (total > player->telemetry.block_us_max)
        player->telemetry.block_us_max = total;
    player->telemetry.decode_frontend_us_average = (uint32_t)(
        player->decode_frontend_us_total / player->blocks_processed);
    player->telemetry.npu_us_average = (uint32_t)(
        player->npu_us_total / player->blocks_processed);
    player->telemetry.backend_sink_us_average = (uint32_t)(
        player->backend_sink_us_total / player->blocks_processed);
    if (timing->decode_frontend_us
        > player->telemetry.decode_frontend_us_max) {
        player->telemetry.decode_frontend_us_max = timing->decode_frontend_us;
    }
    if (timing->npu_us > player->telemetry.npu_us_max)
        player->telemetry.npu_us_max = timing->npu_us;
    if (timing->backend_sink_us > player->telemetry.backend_sink_us_max)
        player->telemetry.backend_sink_us_max = timing->backend_sink_us;
    if (total > PLAYER_BLOCK_DEADLINE_US)
        ++player->telemetry.deadline_miss;
}

static int submit_output(player_t *player, size_t frames)
{
    for (size_t index = 0u; index < frames; ++index) {
        audio_frame_t frame;
        frame.mix_left = stem_float_to_i16(player->mix[index][0]);
        frame.mix_right = stem_float_to_i16(player->mix[index][1]);
        frame.vocal_left = stem_float_to_i16(player->vocal[index][0]);
        frame.vocal_right = stem_float_to_i16(player->vocal[index][1]);
        if (!player->deps.audio_write(player->deps.context, &frame)) {
            fail_muted(player);
            return 0;
        }
        player->last_frame = frame;
        ++player->frames_submitted;
    }
    return 1;
}

static void process_block(player_t *player)
{
    player_block_timing_t timing;
    player_process_result_t result;
    size_t output_frames = 0u;
    size_t expected_output = player->blocks_processed == 0u
        ? STEM_BACKEND_STARTUP_FRAMES : STEM_BLOCK_SAMPLES;

    if (player->deps.audio_space(player->deps.context) < STEM_BLOCK_SAMPLES)
        return;
    if (!fill_analysis_window(player)) {
        if (player->state != PLAYER_FAIL_MUTE && player->end_of_stream)
            player->state = PLAYER_FADE;
        return;
    }
    if (pcm_ring_peek(&player->pcm_ring, player->analysis_window,
                      PLAYER_ANALYSIS_WINDOW_FRAMES)
        != PLAYER_ANALYSIS_WINDOW_FRAMES) {
        fail_muted(player);
        return;
    }
    memset(&timing, 0, sizeof(timing));
    result = player->deps.process(
        player->deps.context, (const int16_t *)player->analysis_window,
        PLAYER_ANALYSIS_WINDOW_FRAMES, player->bypass_latched,
        player->mix, player->vocal, &output_frames, &timing);
    if (result == PLAYER_PROCESS_ERROR || output_frames != expected_output) {
        fail_muted(player);
        return;
    }
    if (result == PLAYER_PROCESS_NPU_ERROR) {
        memset(player->vocal, 0,
               output_frames * STEM_INPUT_CHANNELS * sizeof(float));
        ++player->telemetry.npu_errors;
        player->bypass_latched = 1u;
    }
    uint64_t sink_start = player->deps.time_us(player->deps.context);
    if (!submit_output(player, output_frames))
        return;
    uint64_t sink_elapsed = player->deps.time_us(player->deps.context)
        - sink_start;
    if (sink_elapsed > UINT32_MAX - timing.backend_sink_us)
        timing.backend_sink_us = UINT32_MAX;
    else
        timing.backend_sink_us += (uint32_t)sink_elapsed;
    (void)pcm_ring_drop(&player->pcm_ring, STEM_BLOCK_SAMPLES);
    record_timing(player, &timing);
    if (!player->audio_enabled
        && player->frames_submitted >= PLAYER_PREFILL_FRAMES) {
        player->deps.audio_enable(player->deps.context, 1);
        player->audio_enabled = 1u;
    }
    if (player->bypass_latched)
        player->state = PLAYER_BYPASS;
    else if (player->audio_enabled)
        player->state = PLAYER_PLAY;
    else
        player->state = PLAYER_PREFILL;
}

static void fade_and_finish(player_t *player)
{
    size_t space;

    if (!player->audio_enabled && player->frames_submitted != 0u) {
        player->deps.audio_enable(player->deps.context, 1);
        player->audio_enabled = 1u;
    }
    space = player->deps.audio_space(player->deps.context);
    while (space != 0u && player->fade_progress < PLAYER_FADE_FRAMES) {
        uint32_t remaining = PLAYER_FADE_FRAMES - player->fade_progress;
        audio_frame_t frame;
        frame.mix_left = (int16_t)((int32_t)player->last_frame.mix_left
                                   * (int32_t)remaining
                                   / (int32_t)PLAYER_FADE_FRAMES);
        frame.mix_right = (int16_t)((int32_t)player->last_frame.mix_right
                                    * (int32_t)remaining
                                    / (int32_t)PLAYER_FADE_FRAMES);
        frame.vocal_left = (int16_t)((int32_t)player->last_frame.vocal_left
                                     * (int32_t)remaining
                                     / (int32_t)PLAYER_FADE_FRAMES);
        frame.vocal_right = (int16_t)((int32_t)player->last_frame.vocal_right
                                      * (int32_t)remaining
                                      / (int32_t)PLAYER_FADE_FRAMES);
        if (!player->deps.audio_write(player->deps.context, &frame)) {
            fail_muted(player);
            return;
        }
        ++player->fade_progress;
        --space;
    }
    if (player->fade_progress == PLAYER_FADE_FRAMES
        && player->telemetry.fifo_level == 0u) {
        if (player->audio_enabled) {
            player->deps.audio_enable(player->deps.context, 0);
            player->audio_enabled = 0u;
        }
        player->state = PLAYER_DONE;
    }
}

void player_step(player_t *player)
{
    if (player == NULL || !dependencies_valid(&player->deps))
        return;
    update_audio_status(player);
    if (player->state == PLAYER_FAIL_MUTE) {
        fail_muted(player);
        emit_telemetry(player);
        return;
    }
    switch (player->state) {
    case PLAYER_BOOT:
        player->state = PLAYER_SELF_TEST;
        break;
    case PLAYER_SELF_TEST:
        if (!player->deps.self_test(player->deps.context)) {
            ++player->telemetry.codec_errors;
            fail_muted(player);
        } else {
            player->state = PLAYER_MOUNT;
        }
        break;
    case PLAYER_MOUNT:
        if (!player->deps.mount(player->deps.context)) {
            ++player->telemetry.decoder_errors;
            fail_muted(player);
        } else {
            player->state = PLAYER_OPEN;
        }
        break;
    case PLAYER_OPEN:
        if (!player->deps.open(player->deps.context)) {
            ++player->telemetry.decoder_errors;
            fail_muted(player);
        } else {
            player->state = PLAYER_PREFILL;
        }
        break;
    case PLAYER_PREFILL:
    case PLAYER_PLAY:
    case PLAYER_BYPASS:
        process_block(player);
        break;
    case PLAYER_FADE:
        fade_and_finish(player);
        break;
    case PLAYER_FAIL_MUTE:
    case PLAYER_DONE:
        break;
    default:
        fail_muted(player);
        break;
    }
    update_audio_status(player);
    emit_telemetry(player);
}
