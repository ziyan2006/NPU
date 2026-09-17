/* SPDX-License-Identifier: MIT */
#ifndef STEM_NPU_SESSION_H
#define STEM_NPU_SESSION_H

#include "npu_driver.h"
#include "stem_contract.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STEM_NPU_WATCHDOG_POLLS 8000000u

typedef struct {
    uint32_t task_tag;
    uint32_t commands_retired;
    uint64_t cycles_total;
} stem_npu_stats_t;

typedef struct {
    npu_resident_t resident;
    uint32_t next_task_tag;
} stem_npu_session_t;

npu_result_t stem_npu_session_init(stem_npu_session_t *session,
                                   npu_device_t *device,
                                   uint64_t task_physical_address,
                                   void *task_cpu_address,
                                   size_t task_bytes);
npu_result_t stem_npu_run_block(stem_npu_session_t *session,
                                const int16_t input[STEM_PACKED_VALUES],
                                int16_t output[STEM_PACKED_VALUES],
                                stem_npu_stats_t *statistics);

#ifdef __cplusplus
}
#endif

#endif /* STEM_NPU_SESSION_H */
