/* SPDX-License-Identifier: MIT */
#include "stem_npu_session.h"

#include "stem_task_metadata.h"
#include "stem_task_payload.h"

#include <string.h>

npu_result_t stem_npu_session_init(stem_npu_session_t *session,
                                   npu_device_t *device,
                                   uint64_t task_physical_address,
                                   void *task_cpu_address,
                                   size_t task_bytes)
{
    npu_result_t result;

    if (session == NULL || task_cpu_address == NULL
        || task_bytes != STEM_TASK_IMAGE_BYTES)
        return NPU_E_INVALID;
    memcpy(task_cpu_address, stem_task_payload, STEM_TASK_IMAGE_BYTES);
    result = npu_resident_prepare(&session->resident, device,
                                  task_physical_address, task_cpu_address,
                                  task_bytes);
    if (result != NPU_OK)
        return result;
    session->next_task_tag = 1u;
    return NPU_OK;
}

npu_result_t stem_npu_run_block(stem_npu_session_t *session,
                                const int16_t input[STEM_PACKED_VALUES],
                                int16_t output[STEM_PACKED_VALUES],
                                stem_npu_stats_t *statistics)
{
    uint8_t *task;
    uint64_t task_physical_address;
    uint32_t task_tag;
    npu_range_t clean_range;
    npu_range_t invalidate_range;
    npu_result_t result;

    if (session == NULL || input == NULL || output == NULL
        || statistics == NULL || session->resident.prepared == 0u)
        return NPU_E_INVALID;
    task = (uint8_t *)session->resident.task_cpu_address;
    task_physical_address = session->resident.task_physical_address;
    task_tag = session->next_task_tag;
    if (task_tag == 0u)
        task_tag = 1u;
    memcpy(task + STEM_TASK_INPUT_OFFSET, input, STEM_TASK_INPUT_BYTES);
    clean_range.cpu_address = task + STEM_TASK_INPUT_OFFSET;
    clean_range.physical_address = task_physical_address + STEM_TASK_INPUT_OFFSET;
    clean_range.bytes = STEM_TASK_INPUT_BYTES;
    invalidate_range.cpu_address = task + STEM_TASK_OUTPUT_OFFSET;
    invalidate_range.physical_address = task_physical_address
        + STEM_TASK_OUTPUT_OFFSET;
    invalidate_range.bytes = STEM_TASK_OUTPUT_BYTES;
    result = npu_resident_submit(&session->resident, &clean_range,
                                 &invalidate_range, task_tag,
                                 STEM_NPU_WATCHDOG_POLLS);
    if (result != NPU_OK)
        return result;
    memcpy(output, task + STEM_TASK_OUTPUT_OFFSET, STEM_TASK_OUTPUT_BYTES);
    statistics->task_tag = task_tag;
    statistics->commands_retired =
        session->resident.completion.statistics.commands_retired;
    statistics->cycles_total = session->resident.completion.statistics.cycles_total;
    session->next_task_tag = task_tag + 1u;
    if (session->next_task_tag == 0u)
        session->next_task_tag = 1u;
    return NPU_OK;
}
