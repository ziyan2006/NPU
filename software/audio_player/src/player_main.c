/* SPDX-License-Identifier: MIT */
#include "audio_hw.h"
#include "player_platform.h"
#if defined(PLAYER_MODE_MP3_BYPASS)
#include "sd_mp3_source.h"
#endif
#if defined(PLAYER_MODE_WAV)
#include "wav_source.h"
#endif

#include <stddef.h>
#include <stdint.h>

#if (defined(PLAYER_MODE_TONE) + defined(PLAYER_MODE_WAV) \
     + defined(PLAYER_MODE_MP3_BYPASS)) != 1
#error "build must define exactly one player mode"
#endif

#define PLAYER_PREFILL_FRAMES 8192u
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
    if (source.total_frames < PLAYER_PREFILL_FRAMES) {
        player_platform_log("FAIL WAV_TOO_SHORT");
        return 0;
    }

    player_platform_log("PREFILL");
    while (submitted < PLAYER_PREFILL_FRAMES) {
        uint32_t wanted = minimum_u32(
            PLAYER_BATCH_FRAMES, PLAYER_PREFILL_FRAMES - submitted);
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
#else
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
    while (submitted < PLAYER_PREFILL_FRAMES) {
        uint32_t wanted = minimum_u32(
            PLAYER_BATCH_FRAMES, PLAYER_PREFILL_FRAMES - submitted);
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
