/*
 * Self-contained SD cold-boot runner.
 *
 * A legacy FSBL used on the board loads the PL image but omits the candidate
 * XSA's ps7_post_config() sequence.  Calling that sequence here is safe: it
 * releases PL resets and level shifters without reinitializing DDR, unlike a
 * full ps7_init() call.  This file is linked with that generated ps7_init.c.
 */
#include <stddef.h>
#include <stdint.h>

#include "npu_driver.h"
#include "navigator_npu_task_payload.h"

#define NPU_BASE 0x43c00000u
#define UART0_BASE 0xe0000000u
#define UART_SR_OFFSET 0x2cu
#define UART_FIFO_OFFSET 0x30u
#define UART_SR_TXFULL (1u << 4)
#define TASK_TAG 0x53444254u /* "SDBT" */
#define WATCHDOG_CYCLES 10000000u
#define MAXIMUM_POLLS 20000000u

/* Generated from the signed-off candidate XSA; do not call ps7_init() here. */
extern int ps7_post_config(void);

static uint8_t task_buffer[1851712] __attribute__((aligned(64)));

static inline void dsb(void)
{
    __asm__ volatile("dsb sy" ::: "memory");
}

static void disable_caches(void)
{
    uint32_t sctlr;
    __asm__ volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(sctlr));
    sctlr &= ~((1u << 2) | (1u << 12));
    __asm__ volatile("mcr p15, 0, %0, c1, c0, 0\nisb" :: "r"(sctlr)
                     : "memory");
}

static void barrier(void *context)
{
    (void)context;
    dsb();
}

static void uart_putc(char character)
{
    volatile uint32_t *const uart = (volatile uint32_t *)UART0_BASE;
    while ((uart[UART_SR_OFFSET >> 2] & UART_SR_TXFULL) != 0u) { }
    uart[UART_FIFO_OFFSET >> 2] = (uint32_t)(uint8_t)character;
}

static void uart_puts(const char *text)
{
    while (*text != '\0') {
        if (*text == '\n')
            uart_putc('\r');
        uart_putc(*text++);
    }
}

static void copy_bytes(uint8_t *destination, const uint8_t *source, size_t bytes)
{
    while (bytes-- != 0u)
        *destination++ = *source++;
}

static uint32_t fnv1a32(const volatile uint8_t *bytes, size_t count)
{
    uint32_t hash = 2166136261u;
    while (count-- != 0u) {
        hash ^= *bytes++;
        hash *= 16777619u;
    }
    return hash;
}

static int bytes_equal(const volatile uint8_t *left, const uint8_t *right,
                       size_t count)
{
    while (count-- != 0u) {
        if (*left++ != *right++)
            return 0;
    }
    return 1;
}

static void heartbeat_forever(void)
{
    for (;;) {
        uart_puts("[HEARTBEAT] SD coldboot NPU runner healthy\n");
        for (volatile uint32_t delay = 0u; delay != 80000000u; ++delay)
            __asm__ volatile("nop");
    }
}

int main(void)
{
    npu_device_t device;
    npu_completion_t completion;
    npu_platform_ops_t platform = { 0 };
    npu_result_t result;
    uint32_t hash;

    disable_caches();
    uart_puts("STEM NPU SD coldboot runner: applying candidate PL post-config\n");
    if (ps7_post_config() != 0) {
        uart_puts("[FAIL] ps7_post_config\n");
        return 1;
    }
    uart_puts("[INFO] PL post-config complete; probing NPU\n");
    platform.barrier = barrier;
    result = npu_device_init(&device, (volatile void *)NPU_BASE, &platform);
    if (result != NPU_OK) {
        uart_puts("[FAIL] NPU CSR probe\n");
        return 2;
    }
    result = npu_soft_reset(&device);
    if (result != NPU_OK) {
        uart_puts("[FAIL] NPU reset\n");
        return 3;
    }
    for (volatile uint32_t settle = 0u; settle != 1024u; ++settle)
        __asm__ volatile("nop");
    uart_puts("[INFO] staging embedded task\n");
    copy_bytes(task_buffer, navigator_npu_task_image,
               navigator_npu_task_image_bytes);
    dsb();
    result = npu_submit(&device, (uintptr_t)task_buffer, task_buffer,
                        navigator_npu_task_image_bytes, TASK_TAG,
                        WATCHDOG_CYCLES);
    if (result != NPU_OK) {
        uart_puts("[FAIL] NPU submit\n");
        return 4;
    }
    result = npu_wait(&device, MAXIMUM_POLLS, &completion);
    if (result != NPU_OK || completion.completed_tag != TASK_TAG
        || completion.error_code != 0u) {
        uart_puts("[FAIL] NPU completion\n");
        return 5;
    }
    hash = fnv1a32(task_buffer + navigator_npu_output_offset,
                   navigator_npu_expected_output_bytes);
    if (hash != 0x4db54515u || !bytes_equal(
            task_buffer + navigator_npu_output_offset,
            navigator_npu_expected_output,
            navigator_npu_expected_output_bytes)) {
        uart_puts("[FAIL] output verification\n");
        return 6;
    }
    uart_puts("[PASS] SD coldboot NPU inference verified\n");
    heartbeat_forever();
}
