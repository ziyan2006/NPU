/* SPDX-License-Identifier: MIT */
#include "npu_vitis_platform.h"

#include <limits.h>
#include <string.h>

#include "xil_cache.h"
#include "xil_types.h"

static void npu_vitis_clean(void *address, size_t bytes, void *context)
{
    uint8_t *cursor = (uint8_t *)address;

    (void)context;
    while (bytes != 0u) {
        u32 chunk = bytes > UINT_MAX ? UINT_MAX : (u32)bytes;
        Xil_DCacheFlushRange((INTPTR)cursor, chunk);
        cursor += chunk;
        bytes -= chunk;
    }
}

static void npu_vitis_invalidate(void *address, size_t bytes, void *context)
{
    uint8_t *cursor = (uint8_t *)address;

    (void)context;
    while (bytes != 0u) {
        u32 chunk = bytes > UINT_MAX ? UINT_MAX : (u32)bytes;
        Xil_DCacheInvalidateRange((INTPTR)cursor, chunk);
        cursor += chunk;
        bytes -= chunk;
    }
}

static void npu_vitis_barrier(void *context)
{
    (void)context;
#if defined(__arm__) || defined(__aarch64__)
    __asm__ volatile("dmb sy" ::: "memory");
#elif defined(__GNUC__) || defined(__clang__)
    __sync_synchronize();
#endif
}

npu_platform_ops_t npu_vitis_platform_ops(void)
{
    npu_platform_ops_t ops;

    ops.clean = npu_vitis_clean;
    ops.invalidate = npu_vitis_invalidate;
    ops.barrier = npu_vitis_barrier;
    ops.context = NULL;
    return ops;
}

npu_result_t npu_vitis_device_init_at(npu_device_t *device,
                                      uintptr_t csr_base)
{
    npu_platform_ops_t ops = npu_vitis_platform_ops();

    if (csr_base == 0u)
        return NPU_E_INVALID;
    return npu_device_init(device, (volatile void *)csr_base, &ops);
}

npu_result_t npu_vitis_task_buffer_init(npu_vitis_task_buffer_t *buffer,
                                        void *cpu_address,
                                        size_t capacity_bytes)
{
    if (buffer == NULL || cpu_address == NULL || capacity_bytes == 0u
        || ((uintptr_t)cpu_address & (NPU_TASK_ALIGNMENT - 1u)) != 0u)
        return NPU_E_INVALID;
    buffer->cpu_address = cpu_address;
    buffer->dma_address = (uint64_t)(uintptr_t)cpu_address;
    buffer->capacity_bytes = capacity_bytes;
    buffer->staged_bytes = 0u;
    return NPU_OK;
}

npu_result_t npu_vitis_stage_task(npu_vitis_task_buffer_t *buffer,
                                  const void *task_image,
                                  size_t task_bytes)
{
    if (buffer == NULL || buffer->cpu_address == NULL || task_image == NULL
        || task_bytes > buffer->capacity_bytes)
        return NPU_E_INVALID;
    memcpy(buffer->cpu_address, task_image, task_bytes);
    buffer->staged_bytes = task_bytes;
    return NPU_OK;
}

npu_result_t npu_vitis_submit_staged(npu_device_t *device,
                                     npu_vitis_task_buffer_t *buffer,
                                     uint32_t task_tag,
                                     uint32_t watchdog_cycles)
{
    if (buffer == NULL || buffer->staged_bytes == 0u)
        return NPU_E_INVALID;
    return npu_submit(device, buffer->dma_address, buffer->cpu_address,
                      buffer->staged_bytes, task_tag, watchdog_cycles);
}

int npu_vitis_verify_output(const npu_vitis_task_buffer_t *buffer,
                            size_t output_offset,
                            const void *expected_output,
                            size_t output_bytes)
{
    const uint8_t *actual;

    if (buffer == NULL || buffer->cpu_address == NULL
        || expected_output == NULL || output_bytes == 0u
        || output_offset > buffer->staged_bytes
        || output_bytes > buffer->staged_bytes - output_offset)
        return 0;
    actual = (const uint8_t *)buffer->cpu_address + output_offset;
    return memcmp(actual, expected_output, output_bytes) == 0;
}
