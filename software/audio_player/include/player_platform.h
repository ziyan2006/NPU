/* SPDX-License-Identifier: MIT */
#ifndef STEM_PLAYER_PLATFORM_H
#define STEM_PLAYER_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int player_platform_init(void);
void player_platform_shutdown(void);
uintptr_t player_platform_audio_base(void);
uint32_t player_platform_audio_read(uint32_t offset);
void player_platform_audio_write(uint32_t offset, uint32_t value);
uint32_t player_platform_milliseconds(void);
uint64_t player_platform_microseconds(void);
void player_platform_delay_ms(uint32_t milliseconds);
void player_platform_log(const char *message);
void player_platform_status(uint32_t played, uint32_t level,
                            uint32_t minimum_level, uint32_t underflows,
                            uint32_t overflows, uint32_t codec_status);
int player_platform_mount(void);
int player_platform_open_wav(void);
size_t player_platform_wav_read(void *context, void *destination, size_t bytes);
void player_platform_close_wav(void);
int player_platform_open_mp3(void);
size_t player_platform_mp3_read(void *context, void *destination, size_t bytes);
void player_platform_close_mp3(void);

#ifdef __cplusplus
}
#endif

#endif /* STEM_PLAYER_PLATFORM_H */
