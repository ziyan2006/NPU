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
    volatile uint32_t *registers;
    uint32_t completion_tag;
    uint32_t completion_commands;
    uint16_t completion_error;
    int complete_on_doorbell;
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
    if (log->complete_on_doorbell && log->registers != NULL
        && log->registers[NPU_REG_DOORBELL >> 2] == 1u) {
        log->registers[NPU_REG_DOORBELL >> 2] = 0u;
        log->registers[NPU_REG_COMPLETED_TAG >> 2] = log->completion_tag;
        log->registers[NPU_REG_COMMANDS_RETIRED >> 2] = log->completion_commands;
        log->registers[NPU_REG_ERROR_CODE >> 2] = log->completion_error;
        log->registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE
            | (log->completion_error != 0u ? NPU_STATUS_ERROR : NPU_STATUS_DONE);
        log->complete_on_doorbell = 0;
    }
}

static void test_resident_api(void)
{
    uint32_t registers[0x100 / sizeof(uint32_t)];
    unsigned char task[512];
    platform_log_t log;
    npu_platform_ops_t platform;
    npu_device_t device;
    npu_resident_t resident;
    npu_range_t input;
    npu_range_t output;

    memset(registers, 0, sizeof(registers));
    memset(task, 0, sizeof(task));
    memset(&log, 0, sizeof(log));
    task[0x1c] = 7u;
    platform.clean = record_clean;
    platform.invalidate = record_invalidate;
    platform.barrier = record_barrier;
    platform.context = &log;
    log.registers = registers;
    registers[NPU_REG_IP_ID >> 2] = NPU_IP_ID_VALUE;
    registers[NPU_REG_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_ISA_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    assert(npu_device_init(&device, registers, &platform) == NPU_OK);
    assert(npu_resident_prepare(&resident, &device, 0x20000000u,
                                task, sizeof(task)) == NPU_OK);
    assert(log.clean_count == 1u && log.last_address == task
           && log.last_bytes == sizeof(task));

    input.cpu_address = task + 64u;
    input.physical_address = 0x20000040u;
    input.bytes = 64u;
    output.cpu_address = task + 128u;
    output.physical_address = 0x20000080u;
    output.bytes = 64u;
    log.completion_tag = 11u;
    log.completion_commands = 7u;
    log.complete_on_doorbell = 1;
    assert(npu_resident_submit(&resident, &input, &output, 11u, 100u)
           == NPU_OK);
    assert(log.clean_count == 2u && log.last_address == task + 64u
           && log.last_bytes == 64u);
    assert(log.invalidate_count == 1u && log.last_address == task + 128u
           && log.last_bytes == 64u);
    assert(resident.completion.completed_tag == 11u);
    assert(resident.completion.statistics.commands_retired == 7u);

    input.cpu_address = task + 480u;
    input.physical_address = 0x200001e0u;
    input.bytes = 64u;
    assert(npu_resident_submit(&resident, &input, &output, 12u, 100u)
           == NPU_E_INVALID);
    input.cpu_address = task + 64u;
    input.physical_address = UINT64_MAX - 31u;
    input.bytes = 64u;
    assert(npu_resident_submit(&resident, &input, &output, 12u, 100u)
           == NPU_E_INVALID);
    input.physical_address = 0x20000080u;
    assert(npu_resident_submit(&resident, &input, &output, 12u, 100u)
           == NPU_E_INVALID);
    input.physical_address = 0x20000040u;

    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    log.completion_tag = 999u;
    log.completion_commands = 7u;
    log.completion_error = 0u;
    log.complete_on_doorbell = 1;
    assert(npu_resident_submit(&resident, &input, &output, 12u, 100u)
           == NPU_E_HARDWARE);
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    log.completion_tag = 13u;
    log.completion_commands = 7u;
    log.completion_error = 4u;
    log.complete_on_doorbell = 1;
    assert(npu_resident_submit(&resident, &input, &output, 13u, 100u)
           == NPU_E_HARDWARE);
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    log.completion_tag = 14u;
    log.completion_commands = 6u;
    log.completion_error = 0u;
    log.complete_on_doorbell = 1;
    assert(npu_resident_submit(&resident, &input, &output, 14u, 100u)
           == NPU_E_HARDWARE);
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    log.complete_on_doorbell = 0;
    assert(npu_resident_submit(&resident, &input, &output, 15u, 2u)
           == NPU_E_TIMEOUT);

    assert(npu_resident_prepare(&resident, &device, UINT64_MAX - 255u,
                                task, sizeof(task)) == NPU_E_INVALID);
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
    log.registers = registers;

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

    test_resident_api();

    puts("npu_driver: PASS (legacy + resident)");
    return 0;
}
