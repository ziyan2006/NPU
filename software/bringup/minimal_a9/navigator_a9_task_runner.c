/* Freestanding A9 task runner for volatile JTAG/DDR verification.
 *
 * The host writes the configuration words at RESULT_BASE + 0x40, loads this
 * ELF at 0x12200000 and starts A9 core 0.  The task image remains in DDR and
 * is submitted through the same portable driver used by the Vitis app.
 */
#include <stdint.h>

#include "npu_driver.h"

#define RESULT_BASE             0x12000000u
#define NPU_BASE                0x43c00000u
#define CONFIG_INDEX            16u
#define CONFIG_MAGIC            0x4e505443u /* "NPTC" */
#define RUNNING                 0x4e505401u
#define PASS                    0x4e5054a5u
#define BAD_CONFIG              0x4e5054c1u
#define BAD_INIT                0x4e5054c2u
#define BAD_SUBMIT              0x4e5054c3u
#define BAD_WAIT                0x4e5054c4u
#define BAD_OUTPUT              0x4e5054c5u
#define BAD_RESET               0x4e5054c6u

enum {
    CFG_MAGIC = CONFIG_INDEX,
    CFG_TASK_BASE,
    CFG_TASK_BYTES,
    CFG_TAG,
    CFG_WATCHDOG,
    CFG_OUTPUT_ADDRESS,
    CFG_OUTPUT_BYTES,
    CFG_EXPECTED_FNV1A,
    CFG_MAX_POLLS
};

static inline void dsb(void)
{
    __asm__ volatile("dsb sy" ::: "memory");
}

static void platform_barrier(void *context)
{
    (void)context;
    dsb();
}

static void disable_caches(void)
{
    uint32_t sctlr;

    __asm__ volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(sctlr));
    sctlr &= ~((1u << 2) | (1u << 12));
    __asm__ volatile("mcr p15, 0, %0, c1, c0, 0\nisb" :: "r"(sctlr)
                     : "memory");
}

static uint32_t fnv1a32(const volatile uint8_t *bytes, uint32_t count)
{
    uint32_t hash = 2166136261u;

    while (count-- != 0u) {
        hash ^= *bytes++;
        hash *= 16777619u;
    }
    return hash;
}

/* A volatile loop avoids a hidden libc memset call in the -nostdlib image. */
static void zero_bytes(volatile uint8_t *bytes, uint32_t count)
{
    while (count-- != 0u)
        *bytes++ = 0u;
}

int main(void)
{
    volatile uint32_t *const result = (volatile uint32_t *)RESULT_BASE;
    npu_platform_ops_t platform = { 0 };
    npu_device_t device;
    npu_completion_t completion;
    uint32_t task_base;
    uint32_t task_bytes;
    uint32_t tag;
    uint32_t watchdog;
    uint32_t output_address;
    uint32_t output_bytes;
    uint32_t expected_hash;
    uint32_t maximum_polls;
    npu_result_t rc;

    disable_caches();
    zero_bytes((volatile uint8_t *)&completion, sizeof(completion));
    result[0] = RUNNING;
    dsb();
    if (result[CFG_MAGIC] != CONFIG_MAGIC) {
        result[0] = BAD_CONFIG;
        return 0;
    }
    task_base = result[CFG_TASK_BASE];
    task_bytes = result[CFG_TASK_BYTES];
    tag = result[CFG_TAG];
    watchdog = result[CFG_WATCHDOG];
    output_address = result[CFG_OUTPUT_ADDRESS];
    output_bytes = result[CFG_OUTPUT_BYTES];
    expected_hash = result[CFG_EXPECTED_FNV1A];
    maximum_polls = result[CFG_MAX_POLLS];
    if ((task_base & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes < NPU_TASK_HEADER_BYTES
        || (task_bytes & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_base > 0x3fffffffu - task_bytes
        || output_bytes == 0u || output_bytes > (1u << 20)
        || output_address > 0x3fffffffu - output_bytes
        || maximum_polls == 0u) {
        result[0] = BAD_CONFIG;
        return 0;
    }

    platform.barrier = platform_barrier;
    rc = npu_device_init(&device, (volatile void *)NPU_BASE, &platform);
    result[1] = (uint32_t)rc;
    if (rc != NPU_OK) {
        result[0] = BAD_INIT;
        return 0;
    }
    /* A JTAG reconfiguration leaves the CSR block idle, but a software task
     * must also clear the core's loader/event state before its first
     * doorbell.  This mirrors the known-good JTAG submission sequence. */
    rc = npu_soft_reset(&device);
    result[11] = (uint32_t)rc;
    if (rc != NPU_OK) {
        result[0] = BAD_RESET;
        return 0;
    }
    /* GP0 writes are ordered, but let the one-cycle PL reset pulse retire
     * before publishing a new doorbell on a freshly configured fabric. */
    for (volatile uint32_t settle = 0u; settle != 1024u; ++settle)
        __asm__ volatile("nop");
    rc = npu_submit(&device, task_base, (void *)task_base, task_bytes, tag,
                    watchdog);
    result[2] = (uint32_t)rc;
    result[12] = npu_read_register(&device, NPU_REG_TASK_BASE_LO);
    result[13] = npu_read_register(&device, NPU_REG_TASK_BASE_HI);
    result[14] = npu_read_register(&device, NPU_REG_TASK_BYTES);
    result[15] = npu_read_register(&device, NPU_REG_STATUS);
    if (rc != NPU_OK) {
        result[0] = BAD_SUBMIT;
        return 0;
    }
    rc = npu_wait(&device, maximum_polls, &completion);
    result[3] = (uint32_t)rc;
    result[4] = completion.completed_tag;
    result[5] = (uint32_t)completion.error_code;
    result[6] = completion.error_pc;
    result[7] = completion.statistics.commands_retired;
    result[8] = (uint32_t)completion.statistics.cycles_total;
    result[9] = (uint32_t)(completion.statistics.cycles_total >> 32);
    dsb();
    if (rc != NPU_OK || completion.completed_tag != tag
        || completion.error_code != 0u) {
        result[0] = BAD_WAIT;
        return 0;
    }
    result[10] = fnv1a32((const volatile uint8_t *)output_address,
                         output_bytes);
    if (result[10] != expected_hash) {
        result[0] = BAD_OUTPUT;
        return 0;
    }
    result[0] = PASS;
    dsb();
    return 0;
}
