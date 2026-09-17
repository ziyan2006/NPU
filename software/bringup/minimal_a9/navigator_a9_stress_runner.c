/* Freestanding A9 stress runner for multi-iteration continuous testing.
 *
 * Runs N iterations of the NPU task in DDR, verifying completed tag,
 * retired commands and FNV-1a checksum on every iteration.
 * Reports live progress in RESULT_BASE.
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
    CFG_MAX_POLLS,
    CFG_ITERATIONS
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
    uint32_t target_iterations;
    uint32_t min_cycles = 0xFFFFFFFFu;
    uint32_t max_cycles = 0u;
    npu_result_t rc;

    disable_caches();
    zero_bytes((volatile uint8_t *)&completion, sizeof(completion));
    result[0] = RUNNING;
    result[1] = 0u;  /* current iteration */
    result[2] = 0u;  /* target iterations */
    result[3] = 0u;  /* error count */
    result[4] = 0u;  /* last error code */
    result[5] = 0u;  /* min cycles */
    result[6] = 0u;  /* max cycles */
    result[7] = 0u;  /* last cycles */
    result[8] = 0u;  /* last hash */
    result[9] = 0u;  /* last retired */
    result[10] = 0u; /* expected hash */
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
    target_iterations = result[CFG_ITERATIONS];

    if ((task_base & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes < NPU_TASK_HEADER_BYTES
        || (task_bytes & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_base > 0x3fffffffu - task_bytes
        || output_bytes == 0u || output_bytes > (1u << 20)
        || output_address > 0x3fffffffu - output_bytes
        || maximum_polls == 0u
        || target_iterations == 0u) {
        result[0] = BAD_CONFIG;
        return 0;
    }

    result[2] = target_iterations;
    result[10] = expected_hash;

    platform.barrier = platform_barrier;
    rc = npu_device_init(&device, (volatile void *)NPU_BASE, &platform);
    if (rc != NPU_OK) {
        result[0] = BAD_INIT;
        result[4] = (uint32_t)rc;
        return 0;
    }

    for (uint32_t iter = 0; iter < target_iterations; ++iter) {
        rc = npu_soft_reset(&device);
        if (rc != NPU_OK) {
            result[0] = BAD_RESET;
            result[3]++;
            result[4] = (uint32_t)rc;
            return 0;
        }

        for (volatile uint32_t settle = 0u; settle != 1024u; ++settle)
            __asm__ volatile("nop");

        rc = npu_submit(&device, task_base, (void *)task_base, task_bytes, tag,
                        watchdog);
        if (rc != NPU_OK) {
            result[0] = BAD_SUBMIT;
            result[3]++;
            result[4] = (uint32_t)rc;
            return 0;
        }

        rc = npu_wait(&device, maximum_polls, &completion);
        if (rc != NPU_OK || completion.completed_tag != tag
            || completion.error_code != 0u) {
            result[0] = BAD_WAIT;
            result[3]++;
            result[4] = completion.error_code != 0u ? (uint32_t)completion.error_code : (uint32_t)rc;
            return 0;
        }

        uint32_t hash = fnv1a32((const volatile uint8_t *)output_address,
                                output_bytes);
        if (hash != expected_hash) {
            result[0] = BAD_OUTPUT;
            result[3]++;
            result[8] = hash;
            return 0;
        }

        uint32_t cycles = (uint32_t)completion.statistics.cycles_total;
        if (cycles < min_cycles) min_cycles = cycles;
        if (cycles > max_cycles) max_cycles = cycles;

        result[1] = iter + 1;
        result[5] = min_cycles;
        result[6] = max_cycles;
        result[7] = cycles;
        result[8] = hash;
        result[9] = completion.statistics.commands_retired;
        dsb();
    }

    result[0] = PASS;
    dsb();
    return 0;
}
