/* SPDX-License-Identifier: MIT */
/*
 * Vitis standalone full-task runner for the Navigator Z7020 candidate.
 *
 * Generate generated/navigator_npu_task_payload.[ch] first with
 * scripts/63_generate_vitis_task_payload.py.  The generated binary payload
 * stays local: model weights and test tensors are not committed to this repo.
 */
#include <stddef.h>
#include <stdint.h>

#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"

#include "npu_vitis_platform.h"
#ifndef NPU_VITIS_PAYLOAD_HEADER
#define NPU_VITIS_PAYLOAD_HEADER "generated/navigator_npu_task_payload.h"
#endif
#include NPU_VITIS_PAYLOAD_HEADER

#if defined(XPAR_STEM_NPU_0_S_AXI_CTRL_BASEADDR)
#define NPU_VITIS_CSR_BASE XPAR_STEM_NPU_0_S_AXI_CTRL_BASEADDR
#elif defined(XPAR_STEM_NPU_0_S00_AXI_BASEADDR)
#define NPU_VITIS_CSR_BASE XPAR_STEM_NPU_0_S00_AXI_BASEADDR
#elif defined(XPAR_STEM_NPU_0_BASEADDR)
#define NPU_VITIS_CSR_BASE XPAR_STEM_NPU_0_BASEADDR
#else
#error "XSA xparameters.h has no STEM_NPU CSR base macro"
#endif

#if NPU_VITIS_CSR_BASE != 0x43C00000u
#error "NPU CSR base differs from the signed-off 0x43C00000 mapping"
#endif

#define NPU_TASK_TAG       0x56544953u /* 'VTIS' */
#define NPU_WATCHDOG_CYCLES 10000000u
#define NPU_MAXIMUM_POLLS  20000000u

static uint8_t npu_task_ddr[NPU_VITIS_DEFAULT_TASK_BUFFER_BYTES]
    __attribute__((aligned(NPU_TASK_ALIGNMENT)));

static void print_completion(const npu_completion_t *completion)
{
    xil_printf("NPU completion tag=%08lx commands=%lu cycles=%lu\r\n",
               (unsigned long)completion->completed_tag,
               (unsigned long)completion->statistics.commands_retired,
               (unsigned long)completion->statistics.cycles_total);
    xil_printf("NPU bytes read=%lu written=%lu error=%u pc=%08lx inst=%u\r\n",
               (unsigned long)completion->statistics.bytes_read,
               (unsigned long)completion->statistics.bytes_written,
               (unsigned)completion->error_code,
               (unsigned long)completion->error_pc,
               (unsigned)completion->error_instruction_tag);
}

int main(void)
{
    npu_device_t npu;
    npu_vitis_task_buffer_t task_buffer;
    npu_completion_t completion;
    npu_result_t result;
    int exit_code = 1;

    init_platform();
    xil_printf("Navigator Z7020 NPU Vitis task runner\r\n");
    xil_printf("CSR=%08lx task=%08lx bytes=%lu output+%lu/%lu\r\n",
               (unsigned long)NPU_VITIS_CSR_BASE,
               (unsigned long)(uintptr_t)npu_task_ddr,
               (unsigned long)navigator_npu_task_image_bytes,
               (unsigned long)navigator_npu_output_offset,
               (unsigned long)navigator_npu_expected_output_bytes);

    result = npu_vitis_device_init_at(&npu, NPU_VITIS_CSR_BASE);
    if (result != NPU_OK) {
        xil_printf("NPU DRIVER INIT FAIL %d\r\n", (int)result);
        goto done;
    }
    result = npu_vitis_task_buffer_init(&task_buffer, npu_task_ddr,
                                        sizeof(npu_task_ddr));
    if (result == NPU_OK)
        result = npu_vitis_stage_task(&task_buffer,
                                      navigator_npu_task_image,
                                      navigator_npu_task_image_bytes);
    if (result == NPU_OK)
        result = npu_vitis_submit_staged(&npu, &task_buffer, NPU_TASK_TAG,
                                         NPU_WATCHDOG_CYCLES);
    if (result == NPU_OK)
        result = npu_wait(&npu, NPU_MAXIMUM_POLLS, &completion);

    if (result != NPU_OK) {
        npu_read_completion(&npu, &completion);
        print_completion(&completion);
        xil_printf("NPU TASK FAIL %d\r\n", (int)result);
        goto done;
    }
    print_completion(&completion);
    if (completion.completed_tag != NPU_TASK_TAG
        || completion.statistics.commands_retired
               != navigator_npu_expected_command_count
        || !npu_vitis_verify_output(&task_buffer,
                                    navigator_npu_output_offset,
                                    navigator_npu_expected_output,
                                    navigator_npu_expected_output_bytes)) {
        xil_printf("NPU OUTPUT MISMATCH\r\n");
        goto done;
    }
    xil_printf("NPU TASK PASS\r\n");
    exit_code = 0;
done:
    cleanup_platform();
    return exit_code;
}
