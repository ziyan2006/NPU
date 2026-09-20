/* SPDX-License-Identifier: MIT */
#include "player.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#define TEST_AUDIO_FRAMES (60u * STEM_SAMPLE_RATE_HZ + PLAYER_LOOKAHEAD_FRAMES)
#define TEST_MAX_LINES 80u

typedef struct {
    uint64_t now_us;
    uint64_t last_drain_us;
    uint64_t available_frames;
    uint32_t fifo_level;
    uint32_t fifo_min;
    uint32_t played_frames;
    uint32_t underflows;
    uint32_t overflows;
    uint32_t writes;
    uint32_t writes_after_npu_error;
    uint32_t blocks;
    uint32_t npu_error_block;
    uint32_t enabled_at_writes;
    uint32_t enable_count;
    uint32_t disable_count;
    uint32_t mount_count;
    uint32_t open_count;
    uint32_t line_count;
    uint64_t line_time[TEST_MAX_LINES];
    char last_line[PLAYER_TELEMETRY_LINE_BYTES];
    uint32_t stage_us[3];
    int codec_ok;
    int mount_ok;
    int open_ok;
    int decode_error;
    int enabled;
    int saw_zero_vocal_on_failure;
} fake_player_t;

static void drain(fake_player_t *fake)
{
    uint64_t elapsed = fake->now_us - fake->last_drain_us;
    uint64_t frames = fake->enabled
        ? elapsed * STEM_SAMPLE_RATE_HZ / 1000000u : 0u;
    fake->played_frames += (uint32_t)frames;
    if (frames >= fake->fifo_level) {
        fake->underflows += (uint32_t)(frames - fake->fifo_level);
        fake->fifo_level = 0u;
    } else {
        fake->fifo_level -= (uint32_t)frames;
    }
    fake->last_drain_us = fake->now_us;
    if (fake->fifo_level < fake->fifo_min)
        fake->fifo_min = fake->fifo_level;
}

static void advance(fake_player_t *fake, uint32_t microseconds)
{
    drain(fake);
    fake->now_us += microseconds;
    drain(fake);
}

static int fake_self_test(void *context)
{
    return ((fake_player_t *)context)->codec_ok;
}

static int fake_mount(void *context)
{
    fake_player_t *fake = context;
    ++fake->mount_count;
    return fake->mount_ok;
}

static int fake_open(void *context)
{
    fake_player_t *fake = context;
    ++fake->open_count;
    return fake->open_ok;
}

static player_decode_result_t fake_decode(void *context, int16_t *lr,
                                          size_t capacity, size_t *frames)
{
    fake_player_t *fake = context;
    size_t produced;

    advance(fake, fake->stage_us[0]);
    if (fake->decode_error) {
        *frames = 0u;
        return PLAYER_DECODE_ERROR;
    }
    if (fake->available_frames == 0u) {
        *frames = 0u;
        return PLAYER_DECODE_EOF;
    }
    produced = capacity < fake->available_frames
        ? capacity : (size_t)fake->available_frames;
    for (size_t frame = 0u; frame < produced; ++frame) {
        lr[frame * 2u] = (int16_t)(fake->available_frames & 0x7fffu);
        lr[frame * 2u + 1u] = (int16_t)-(int32_t)lr[frame * 2u];
        --fake->available_frames;
    }
    *frames = produced;
    return PLAYER_DECODE_OK;
}

static player_process_result_t fake_process(
    void *context, const int16_t *lr, size_t frames, int bypass,
    float mix[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
    float vocal[STEM_BLOCK_SAMPLES][STEM_INPUT_CHANNELS],
    size_t *output_frames, player_block_timing_t *timing)
{
    fake_player_t *fake = context;
    size_t produced = fake->blocks == 0u
        ? STEM_BACKEND_STARTUP_FRAMES : STEM_BLOCK_SAMPLES;
    int fail = fake->npu_error_block != 0u
        && fake->blocks + 1u == fake->npu_error_block;

    assert(frames == PLAYER_ANALYSIS_WINDOW_FRAMES);
    assert(lr != NULL);
    timing->decode_frontend_us = fake->stage_us[0];
    timing->npu_us = bypass ? 0u : fake->stage_us[1];
    timing->backend_sink_us = fake->stage_us[2];
    advance(fake, fake->stage_us[1] + fake->stage_us[2]);
    for (size_t frame = 0u; frame < produced; ++frame) {
        mix[frame][0] = 0.25f;
        mix[frame][1] = -0.25f;
        vocal[frame][0] = bypass ? 0.0f : 0.125f;
        vocal[frame][1] = bypass ? 0.0f : -0.125f;
    }
    ++fake->blocks;
    *output_frames = produced;
    return fail ? PLAYER_PROCESS_NPU_ERROR : PLAYER_PROCESS_OK;
}

static size_t fake_space(void *context)
{
    fake_player_t *fake = context;
    drain(fake);
    return AUDIO_HW_FIFO_CAPACITY - fake->fifo_level;
}

static int fake_write(void *context, const audio_frame_t *frame)
{
    fake_player_t *fake = context;
    if (fake->fifo_level >= AUDIO_HW_FIFO_CAPACITY) {
        ++fake->overflows;
        return 0;
    }
    if (fake->npu_error_block != 0u
        && fake->blocks == fake->npu_error_block) {
        if (frame->vocal_left == 0 && frame->vocal_right == 0)
            fake->saw_zero_vocal_on_failure = 1;
    }
    if (fake->blocks > fake->npu_error_block && fake->npu_error_block != 0u)
        ++fake->writes_after_npu_error;
    ++fake->fifo_level;
    ++fake->writes;
    return 1;
}

static void fake_enable(void *context, int enabled)
{
    fake_player_t *fake = context;
    if (enabled) {
        fake->enabled = 1;
        fake->enabled_at_writes = fake->writes;
        ++fake->enable_count;
    } else {
        fake->enabled = 0;
        ++fake->disable_count;
    }
}

static void fake_status(void *context, player_audio_status_t *status)
{
    fake_player_t *fake = context;
    drain(fake);
    status->fifo_level = (uint16_t)fake->fifo_level;
    status->played_frames = fake->played_frames;
    status->underflows = fake->underflows;
    status->overflows = fake->overflows;
    status->stem_target = 1u;
    status->stem_ramping = 0u;
    status->codec_error = fake->codec_ok ? 0u : 1u;
}

static uint64_t fake_time(void *context)
{
    return ((fake_player_t *)context)->now_us;
}

static void fake_uart(void *context, const char *line)
{
    fake_player_t *fake = context;
    assert(fake->line_count < TEST_MAX_LINES);
    if (fake->line_count != 0u)
        assert(fake->now_us - fake->line_time[fake->line_count - 1u]
               >= 1000000u);
    fake->line_time[fake->line_count++] = fake->now_us;
    snprintf(fake->last_line, sizeof(fake->last_line), "%s", line);
}

static player_deps_t make_deps(fake_player_t *fake)
{
    player_deps_t deps;
    memset(&deps, 0, sizeof(deps));
    deps.context = fake;
    deps.self_test = fake_self_test;
    deps.mount = fake_mount;
    deps.open = fake_open;
    deps.decode = fake_decode;
    deps.process = fake_process;
    deps.audio_space = fake_space;
    deps.audio_write = fake_write;
    deps.audio_enable = fake_enable;
    deps.audio_status = fake_status;
    deps.time_us = fake_time;
    deps.uart_line = fake_uart;
    return deps;
}

static void initialize_fake(fake_player_t *fake)
{
    memset(fake, 0, sizeof(*fake));
    fake->available_frames = TEST_AUDIO_FRAMES;
    fake->fifo_min = AUDIO_HW_FIFO_CAPACITY;
    fake->codec_ok = 1;
    fake->mount_ok = 1;
    fake->open_ok = 1;
    fake->stage_us[0] = 25000u;
    fake->stage_us[1] = 40000u;
    fake->stage_us[2] = 20000u;
}

static void run_until_terminal(player_t *player, fake_player_t *fake)
{
    for (size_t step = 0u; step < 400000u; ++step) {
        player_step(player);
        if (player->state == PLAYER_DONE || player->state == PLAYER_FAIL_MUTE)
            return;
        advance(fake, 1000u);
    }
    assert(!"player did not reach a terminal state");
}

static void test_sixty_second_playback(void)
{
    fake_player_t fake;
    player_t player;
    initialize_fake(&fake);
    player_deps_t deps = make_deps(&fake);
    assert(player_init(&player, &deps) == PLAYER_OK);
    run_until_terminal(&player, &fake);
    assert(player.state == PLAYER_DONE);
    assert(fake.enable_count == 1u && fake.disable_count >= 1u);
    assert(fake.enabled_at_writes == PLAYER_PREFILL_FRAMES);
    assert(fake.mount_count == 1u && fake.open_count == 1u);
    assert(player.telemetry.deadline_miss == 0u);
    assert(player.telemetry.block_us_max == 85000u);
    assert(player.telemetry.decode_frontend_us_average == 25000u);
    assert(player.telemetry.decode_frontend_us_max == 25000u);
    assert(player.telemetry.npu_us_average == 40000u);
    assert(player.telemetry.npu_us_max == 40000u);
    assert(player.telemetry.backend_sink_us_average == 20000u);
    assert(player.telemetry.backend_sink_us_max == 20000u);
    assert(player.fade_progress == PLAYER_FADE_FRAMES);
    assert(fake.line_count <= 61u);
    assert(strstr(fake.last_line, "[AUDIO] sec=") != NULL);
    assert(strstr(fake.last_line, "deadline_miss=0") != NULL);
    assert(strstr(fake.last_line, "played=") != NULL);
    assert(strstr(fake.last_line, "decfe_us_avg=25000") != NULL);
    assert(strstr(fake.last_line, "decfe_us_max=25000") != NULL);
    assert(strstr(fake.last_line, "npu_us_avg=40000") != NULL);
    assert(strstr(fake.last_line, "npu_us_max=40000") != NULL);
    assert(strstr(fake.last_line, "sink_us_avg=20000") != NULL);
    assert(strstr(fake.last_line, "sink_us_max=20000") != NULL);
}

static void test_npu_failure_latches_bypass(void)
{
    fake_player_t fake;
    player_t player;
    initialize_fake(&fake);
    fake.available_frames = 8u * STEM_BLOCK_SAMPLES + PLAYER_LOOKAHEAD_FRAMES;
    fake.npu_error_block = 3u;
    player_deps_t deps = make_deps(&fake);
    assert(player_init(&player, &deps) == PLAYER_OK);
    run_until_terminal(&player, &fake);
    assert(player.state == PLAYER_DONE);
    assert(player.telemetry.npu_errors == 1u);
    assert(player.bypass_latched == 1u);
    assert(fake.saw_zero_vocal_on_failure);
    assert(fake.writes_after_npu_error > 0u);
}

static void test_deadline_boundary(void)
{
    fake_player_t fake;
    player_t player;
    initialize_fake(&fake);
    fake.available_frames = 3u * STEM_BLOCK_SAMPLES + PLAYER_LOOKAHEAD_FRAMES;
    fake.stage_us[0] = 30000u;
    fake.stage_us[1] = 40000u;
    fake.stage_us[2] = 22880u;
    player_deps_t deps = make_deps(&fake);
    assert(player_init(&player, &deps) == PLAYER_OK);
    run_until_terminal(&player, &fake);
    assert(player.telemetry.deadline_miss == 0u);

    initialize_fake(&fake);
    fake.available_frames = 3u * STEM_BLOCK_SAMPLES + PLAYER_LOOKAHEAD_FRAMES;
    fake.stage_us[0] = 30000u;
    fake.stage_us[1] = 40000u;
    fake.stage_us[2] = 23000u;
    deps = make_deps(&fake);
    assert(player_init(&player, &deps) == PLAYER_OK);
    run_until_terminal(&player, &fake);
    assert(player.telemetry.deadline_miss > 0u);
}

static void test_fatal_failures_mute(void)
{
    for (unsigned failure = 0u; failure < 4u; ++failure) {
        fake_player_t fake;
        player_t player;
        initialize_fake(&fake);
        if (failure == 0u) fake.codec_ok = 0;
        if (failure == 1u) fake.mount_ok = 0;
        if (failure == 2u) fake.open_ok = 0;
        if (failure == 3u) fake.decode_error = 1;
        player_deps_t deps = make_deps(&fake);
        assert(player_init(&player, &deps) == PLAYER_OK);
        run_until_terminal(&player, &fake);
        assert(player.state == PLAYER_FAIL_MUTE);
        assert(fake.enabled == 0);
    }
}

int main(void)
{
    assert(player_init(NULL, NULL) == PLAYER_E_INVALID);
    test_sixty_second_playback();
    test_npu_failure_latches_bypass();
    test_deadline_boundary();
    test_fatal_failures_mute();
    puts("player state machine: PASS");
    return 0;
}
