/* SPDX-License-Identifier: MIT */
#include "player_platform.h"

#include <limits.h>

#include "ff.h"
#include "sleep.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xiltimer.h"

#if defined(XPAR_AUDIO_OUT_0_S_AXI_AUDIO_BASEADDR)
#define PLAYER_AUDIO_BASE XPAR_AUDIO_OUT_0_S_AXI_AUDIO_BASEADDR
#elif defined(XPAR_AUDIO_OUT_0_BASEADDR)
#define PLAYER_AUDIO_BASE XPAR_AUDIO_OUT_0_BASEADDR
#elif defined(XPAR_AUDIO_OUT_AXI_0_BASEADDR)
#define PLAYER_AUDIO_BASE XPAR_AUDIO_OUT_AXI_0_BASEADDR
#else
#error "XSA xparameters.h has no audio output CSR base macro"
#endif

_Static_assert(PLAYER_AUDIO_BASE == 0x43c10000u,
               "audio CSR base differs from the signed-off mapping");

static FATFS filesystem;
static FIL wav_file;
static int file_open;
static FIL mp3_file;
static int mp3_file_open;

int player_platform_init(void)
{
    Xil_ICacheEnable();
    Xil_DCacheEnable();
    file_open = 0;
    mp3_file_open = 0;
    /* Read-only evidence of the FSBL's actual clock setup, independent of
     * the frequencies generated from the application XSA. */
    xil_printf("[CLOCK] bsp_cpu_hz=%u timer_hz=%u arm_pll=%08x "
               "arm_clk=%08x io_pll=%08x fclk0=%08x gt_ctrl=%08x\r\n",
               (unsigned)XPAR_CPU_CORE_CLOCK_FREQ_HZ,
               (unsigned)(COUNTS_PER_SECOND),
               (unsigned)Xil_In32(0xf8000100u),
               (unsigned)Xil_In32(0xf8000120u),
               (unsigned)Xil_In32(0xf8000108u),
               (unsigned)Xil_In32(0xf8000170u),
               (unsigned)Xil_In32(0xf8f00208u));
    return 1;
}

void player_platform_shutdown(void)
{
    player_platform_close_wav();
    player_platform_close_mp3();
    Xil_DCacheDisable();
    Xil_ICacheDisable();
}

uintptr_t player_platform_audio_base(void)
{
    return (uintptr_t)PLAYER_AUDIO_BASE;
}

uint32_t player_platform_audio_read(uint32_t offset)
{
    return *(volatile const uint32_t *)(PLAYER_AUDIO_BASE + offset);
}

void player_platform_audio_write(uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)(PLAYER_AUDIO_BASE + offset) = value;
}

uint32_t player_platform_milliseconds(void)
{
    XTime now;
    XTime_GetTime(&now);
    return (uint32_t)player_ticks_to_units(now, COUNTS_PER_SECOND, 1000u);
}

uint64_t player_platform_microseconds(void)
{
    XTime now;
    XTime_GetTime(&now);
    return player_ticks_to_units(now, COUNTS_PER_SECOND, 1000000u);
}

void player_platform_delay_ms(uint32_t milliseconds)
{
    while (milliseconds != 0u) {
        uint32_t chunk = milliseconds > 1000u ? 1000u : milliseconds;
        usleep(chunk * 1000u);
        milliseconds -= chunk;
    }
}

void player_platform_log(const char *message)
{
    xil_printf("%s\r\n", message);
}

void player_platform_status(uint32_t played, uint32_t level,
                            uint32_t minimum_level, uint32_t underflows,
                            uint32_t overflows, uint32_t codec_status)
{
    /*
     * Vitis 2026.1 xil_printf interprets every %l conversion as a 64-bit
     * argument on Cortex-A9, although unsigned long is 32-bit for AAPCS32.
     * Keep these CSR/counter arguments and conversions explicitly 32-bit.
     */
    xil_printf("[AUDIO] played=%u fifo=%u fifo_min=%u uf=%u of=%u "
               "codec=%08x\r\n",
               (unsigned int)played, (unsigned int)level,
               (unsigned int)minimum_level, (unsigned int)underflows,
               (unsigned int)overflows, (unsigned int)codec_status);
}

int player_platform_mount(void)
{
    return f_mount(&filesystem, "0:/", 1u) == FR_OK;
}

int player_platform_open_wav(void)
{
    if (f_open(&wav_file, "0:/test.wav", FA_READ) != FR_OK)
        return 0;
    file_open = 1;
    return 1;
}

size_t player_platform_wav_read(void *context, void *destination, size_t bytes)
{
    size_t total = 0u;

    (void)context;
    while (bytes != 0u) {
        UINT received = 0u;
        UINT chunk = bytes > UINT_MAX ? UINT_MAX : (UINT)bytes;
        if (!file_open || f_read(&wav_file, (uint8_t *)destination + total,
                                 chunk, &received) != FR_OK)
            break;
        total += received;
        bytes -= received;
        if (received != chunk)
            break;
    }
    return total;
}

void player_platform_close_wav(void)
{
    if (file_open) {
        (void)f_close(&wav_file);
        file_open = 0;
    }
}

int player_platform_open_mp3(void)
{
    if (f_open(&mp3_file, "0:/music.mp3", FA_READ) != FR_OK)
        return 0;
    mp3_file_open = 1;
    return 1;
}

size_t player_platform_mp3_read(void *context, void *destination, size_t bytes)
{
    size_t total = 0u;

    (void)context;
    while (bytes != 0u) {
        UINT received = 0u;
        UINT chunk = bytes > UINT_MAX ? UINT_MAX : (UINT)bytes;
        if (!mp3_file_open
            || f_read(&mp3_file, (uint8_t *)destination + total,
                      chunk, &received) != FR_OK)
            return SIZE_MAX;
        total += received;
        bytes -= received;
        if (received != chunk)
            break;
    }
    return total;
}

void player_platform_close_mp3(void)
{
    if (mp3_file_open) {
        (void)f_close(&mp3_file);
        mp3_file_open = 0;
    }
}
