/* SPDX-License-Identifier: MIT */
#include "npu_driver.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    unsigned clean_count;
    unsigned invalidate_count;
    unsigned barrier_count;
    void *last_address;
    size_t last_bytes;
} platform_log_t;

static void record_clean(void *address, size_t bytes, void *context)
{
    platform_log_t *log = (platform_log_t *)context;
    ++log->clean_count;
    log->last_address = address;
    log->last_bytes = bytes;
}

static void record_invalidate(void *address, size_t bytes, void *context)
{
    platform_log_t *log = (platform_log_t *)context;
    ++log->invalidate_count;
    log->last_address = address;
    log->last_bytes = bytes;
}

static void record_barrier(void *context)
{
    platform_log_t *log = (platform_log_t *)context;
    ++log->barrier_count;
}

int main(void)
{
    uint32_t registers[0x100 / sizeof(uint32_t)];
    unsigned char task[512];
    platform_log_t log;
    npu_platform_ops_t platform;
    npu_device_t device;
    npu_completion_t completion;

    memset(registers, 0, sizeof(registers));
    memset(task, 0xa5, sizeof(task));
    memset(&log, 0, sizeof(log));
    platform.clean = record_clean;
    platform.invalidate = record_invalidate;
    platform.barrier = record_barrier;
    platform.context = &log;

    registers[NPU_REG_IP_ID >> 2] = NPU_IP_ID_VALUE;
    registers[NPU_REG_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_ISA_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    assert(npu_device_init(&device, registers, &platform) == NPU_OK);
    assert(npu_submit(&device, 0x10000000u, task, sizeof(task),
                      0x12345678u, 4000000u) == NPU_OK);
    assert(log.clean_count == 1u && log.last_address == task
           && log.last_bytes == sizeof(task));
    assert(registers[NPU_REG_TASK_BASE_LO >> 2] == 0x10000000u);
    assert(registers[NPU_REG_TASK_BASE_HI >> 2] == 0u);
    assert(registers[NPU_REG_TASK_BYTES >> 2] == sizeof(task));
    assert(registers[NPU_REG_TASK_TAG >> 2] == 0x12345678u);
    assert(registers[NPU_REG_DOORBELL >> 2] == 1u);

    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_BUSY;
    assert(npu_poll(&device) == NPU_STATE_BUSY);
    assert(npu_soft_reset(&device) == NPU_E_BUSY);
    registers[NPU_REG_COMPLETED_TAG >> 2] = 0x12345678u;
    registers[NPU_REG_COMMANDS_RETIRED >> 2] = 1869u;
    registers[NPU_REG_CYCLES_TOTAL_LO >> 2] = 3254220u;
    registers[NPU_REG_CYCLES_TOTAL_HI >> 2] = 0u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_DONE | NPU_STATUS_IDLE;
    assert(npu_wait(&device, 1u, &completion) == NPU_OK);
    assert(log.invalidate_count == 1u);
    assert(completion.completed_tag == 0x12345678u);
    assert(completion.statistics.commands_retired == 1869u);
    assert(completion.statistics.cycles_total == 3254220u);
    assert(npu_poll(&device) == NPU_STATE_DONE);
    assert(log.invalidate_count == 1u);

    assert(npu_submit(&device, 0x10000001u, task, sizeof(task), 1u, 1u)
           == NPU_E_INVALID);
    assert(npu_submit(&device, 0x10000000u, task, 257u, 1u, 1u)
           == NPU_E_INVALID);
    assert(npu_configure_interrupts(&device,
                                    NPU_IRQ_DONE | NPU_IRQ_ERROR, 1)
           == NPU_OK);
    assert(registers[NPU_REG_IRQ_ENABLE >> 2]
           == (NPU_IRQ_DONE | NPU_IRQ_ERROR));
    assert(registers[NPU_REG_CONTROL >> 2]
           == NPU_CONTROL_IRQ_GLOBAL_ENABLE);

    puts("npu_driver: PASS");
    return 0;
}
