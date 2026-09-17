/* SPDX-License-Identifier: MIT */
#include "stem_npu_session.h"
#include "stem_task_metadata.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    volatile uint32_t *registers;
    void *clean_address[4];
    size_t clean_bytes[4];
    void *invalidate_address[4];
    size_t invalidate_bytes[4];
    unsigned clean_count;
    unsigned invalidate_count;
    unsigned barrier_count;
} session_log_t;

static void record_clean(void *address, size_t bytes, void *context)
{
    session_log_t *log = context;
    assert(log->clean_count < 4u);
    log->clean_address[log->clean_count] = address;
    log->clean_bytes[log->clean_count++] = bytes;
}

static void record_invalidate(void *address, size_t bytes, void *context)
{
    session_log_t *log = context;
    assert(log->invalidate_count < 4u);
    log->invalidate_address[log->invalidate_count] = address;
    log->invalidate_bytes[log->invalidate_count++] = bytes;
}

static void complete_on_doorbell(void *context)
{
    session_log_t *log = context;
    ++log->barrier_count;
    if (log->registers[NPU_REG_DOORBELL >> 2] == 1u) {
        log->registers[NPU_REG_DOORBELL >> 2] = 0u;
        log->registers[NPU_REG_COMPLETED_TAG >> 2] =
            log->registers[NPU_REG_TASK_TAG >> 2];
        log->registers[NPU_REG_COMMANDS_RETIRED >> 2] = STEM_TASK_COMMAND_COUNT;
        log->registers[NPU_REG_CYCLES_TOTAL_LO >> 2] = 123456u;
        log->registers[NPU_REG_ERROR_CODE >> 2] = 0u;
        log->registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE | NPU_STATUS_DONE;
    }
}

int main(void)
{
    uint32_t registers[0x100 / sizeof(uint32_t)] = {0};
    session_log_t log = {0};
    npu_platform_ops_t platform;
    npu_device_t device;
    stem_npu_session_t session;
    stem_npu_stats_t stats;
    int16_t input[STEM_PACKED_VALUES];
    int16_t output[STEM_PACKED_VALUES];
    uint8_t *task = malloc(STEM_TASK_IMAGE_BYTES);
    uint8_t *before = malloc(STEM_TASK_IMAGE_BYTES);
    assert(task != NULL && before != NULL);

    log.registers = registers;
    platform.clean = record_clean;
    platform.invalidate = record_invalidate;
    platform.barrier = complete_on_doorbell;
    platform.context = &log;
    registers[NPU_REG_IP_ID >> 2] = NPU_IP_ID_VALUE;
    registers[NPU_REG_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_ISA_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    assert(npu_device_init(&device, registers, &platform) == NPU_OK);
    assert(stem_npu_session_init(&session, &device, 0x30000000u,
                                 task, STEM_TASK_IMAGE_BYTES) == NPU_OK);
    assert(log.clean_count == 1u);
    assert(log.clean_address[0] == task);
    assert(log.clean_bytes[0] == STEM_TASK_IMAGE_BYTES);

    for (size_t index = 0; index < STEM_PACKED_VALUES; ++index) {
        input[index] = (int16_t)(index * 17u + 3u);
        ((int16_t *)(task + STEM_TASK_OUTPUT_OFFSET))[index] =
            (int16_t)(index * 13u - 7u);
    }
    memcpy(before, task, STEM_TASK_IMAGE_BYTES);
    memset(output, 0, sizeof(output));
    assert(stem_npu_run_block(&session, input, output, &stats) == NPU_OK);
    assert(stats.task_tag == 1u);
    assert(stats.commands_retired == STEM_TASK_COMMAND_COUNT);
    assert(stats.cycles_total == 123456u);
    assert(log.clean_count == 2u);
    assert(log.clean_address[1] == task + STEM_TASK_INPUT_OFFSET);
    assert(log.clean_bytes[1] == STEM_TASK_INPUT_BYTES);
    assert(log.invalidate_count == 1u);
    assert(log.invalidate_address[0] == task + STEM_TASK_OUTPUT_OFFSET);
    assert(log.invalidate_bytes[0] == STEM_TASK_OUTPUT_BYTES);
    assert(memcmp(task + STEM_TASK_INPUT_OFFSET, input, sizeof(input)) == 0);
    assert(memcmp(output, task + STEM_TASK_OUTPUT_OFFSET, sizeof(output)) == 0);
    for (size_t index = 0; index < STEM_TASK_IMAGE_BYTES; ++index) {
        int inside_input = index >= STEM_TASK_INPUT_OFFSET
            && index < STEM_TASK_INPUT_OFFSET + STEM_TASK_INPUT_BYTES;
        if (!inside_input) {
            assert(task[index] == before[index]);
        }
    }

    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    assert(stem_npu_run_block(&session, input, output, &stats) == NPU_OK);
    assert(stats.task_tag == 2u);
    assert(log.clean_count == 3u && log.invalidate_count == 2u);

    free(before);
    free(task);
    puts("stem npu session: PASS");
    return 0;
}
