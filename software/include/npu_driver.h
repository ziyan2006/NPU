/* SPDX-License-Identifier: MIT */
#ifndef STEM_NPU_DRIVER_H
#define STEM_NPU_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_REG_IP_ID                 0x000u
#define NPU_REG_VERSION               0x004u
#define NPU_REG_ISA_VERSION           0x008u
#define NPU_REG_CAPABILITY0           0x00cu
#define NPU_REG_CAPABILITY1           0x010u
#define NPU_REG_CONTROL               0x014u
#define NPU_REG_STATUS                0x018u
#define NPU_REG_IRQ_STATUS            0x01cu
#define NPU_REG_IRQ_ENABLE            0x020u
#define NPU_REG_TASK_BASE_LO          0x024u
#define NPU_REG_TASK_BASE_HI          0x028u
#define NPU_REG_TASK_BYTES            0x02cu
#define NPU_REG_TASK_TAG              0x030u
#define NPU_REG_DOORBELL              0x034u
#define NPU_REG_COMPLETED_TAG         0x038u
#define NPU_REG_ERROR_CODE            0x03cu
#define NPU_REG_ERROR_PC              0x040u
#define NPU_REG_ERROR_INST_TAG        0x044u
#define NPU_REG_WATCHDOG_LIMIT        0x048u
#define NPU_REG_COMMANDS_RETIRED      0x04cu
#define NPU_REG_CYCLES_TOTAL_LO       0x050u
#define NPU_REG_CYCLES_TOTAL_HI       0x054u
#define NPU_REG_CYCLES_COMPUTE_LO     0x058u
#define NPU_REG_CYCLES_COMPUTE_HI     0x05cu
#define NPU_REG_CYCLES_RD_WAIT_LO     0x060u
#define NPU_REG_CYCLES_RD_WAIT_HI     0x064u
#define NPU_REG_CYCLES_WR_WAIT_LO     0x068u
#define NPU_REG_CYCLES_WR_WAIT_HI     0x06cu
#define NPU_REG_CYCLES_BANK_STALL_LO  0x070u
#define NPU_REG_CYCLES_BANK_STALL_HI  0x074u
#define NPU_REG_BYTES_READ_LO         0x078u
#define NPU_REG_BYTES_READ_HI         0x07cu
#define NPU_REG_BYTES_WRITTEN_LO      0x080u
#define NPU_REG_BYTES_WRITTEN_HI      0x084u
#define NPU_REG_READ_HIGH_WATER_LO    0x088u
#define NPU_REG_READ_HIGH_WATER_HI    0x08cu
#define NPU_REG_ERROR_COUNT           0x090u
#define NPU_REG_WRITE_HIGH_WATER_LO   0x094u
#define NPU_REG_WRITE_HIGH_WATER_HI   0x098u

#define NPU_IP_ID_VALUE               0x3155504eu
#define NPU_RTL_VERSION_MAJOR         1u
#define NPU_RTL_VERSION_MINOR         0u
#define NPU_DRIVER_ISA_MAJOR          1u
#define NPU_DRIVER_ISA_MINOR          0u

#define NPU_CONTROL_SOFT_RESET        (1u << 0)
#define NPU_CONTROL_IRQ_GLOBAL_ENABLE (1u << 1)

#define NPU_STATUS_IDLE               (1u << 0)
#define NPU_STATUS_BUSY               (1u << 1)
#define NPU_STATUS_RESETTING          (1u << 2)
#define NPU_STATUS_DONE               (1u << 3)
#define NPU_STATUS_ERROR              (1u << 4)

#define NPU_IRQ_DONE                  (1u << 0)
#define NPU_IRQ_ERROR                 (1u << 1)
#define NPU_IRQ_WATCHDOG              (1u << 2)
#define NPU_IRQ_ALL                   (NPU_IRQ_DONE | NPU_IRQ_ERROR \
                                      | NPU_IRQ_WATCHDOG)

#define NPU_TASK_ALIGNMENT            64u
#define NPU_TASK_HEADER_BYTES         256u

typedef enum {
    NPU_OK = 0,
    NPU_E_INVALID = -1,
    NPU_E_INCOMPATIBLE = -2,
    NPU_E_BUSY = -3,
    NPU_E_TIMEOUT = -4,
    NPU_E_HARDWARE = -5
} npu_result_t;

typedef enum {
    NPU_STATE_IDLE = 0,
    NPU_STATE_BUSY,
    NPU_STATE_DONE,
    NPU_STATE_ERROR
} npu_state_t;

typedef void (*npu_cache_fn)(void *address, size_t bytes, void *context);
typedef void (*npu_barrier_fn)(void *context);

typedef struct {
    npu_cache_fn clean;
    npu_cache_fn invalidate;
    npu_barrier_fn barrier;
    void *context;
} npu_platform_ops_t;

typedef struct {
    uint64_t cycles_total;
    uint64_t cycles_compute;
    uint64_t cycles_read_wait;
    uint64_t cycles_write_wait;
    uint64_t cycles_bank_stall;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t read_high_water;
    uint64_t write_high_water;
    uint32_t commands_retired;
    uint32_t error_count;
} npu_statistics_t;

typedef struct {
    uint32_t completed_tag;
    uint16_t error_code;
    uint16_t error_instruction_tag;
    uint32_t error_pc;
    npu_statistics_t statistics;
} npu_completion_t;

typedef struct {
    volatile uint32_t *registers;
    npu_platform_ops_t platform;
    void *submitted_cpu_address;
    size_t submitted_bytes;
    uint32_t submitted_tag;
    uint8_t cache_pending;
} npu_device_t;

npu_result_t npu_device_init(npu_device_t *device,
                             volatile void *register_base,
                             const npu_platform_ops_t *platform);
npu_result_t npu_soft_reset(npu_device_t *device);
npu_result_t npu_configure_interrupts(npu_device_t *device,
                                      uint32_t interrupt_mask,
                                      int global_enable);
void npu_acknowledge_interrupts(npu_device_t *device, uint32_t interrupt_mask);
npu_result_t npu_submit(npu_device_t *device,
                        uint64_t task_physical_address,
                        void *task_cpu_address,
                        size_t task_bytes,
                        uint32_t task_tag,
                        uint32_t watchdog_cycles);
npu_state_t npu_poll(npu_device_t *device);
npu_result_t npu_wait(npu_device_t *device,
                      uint32_t maximum_polls,
                      npu_completion_t *completion);
void npu_read_completion(const npu_device_t *device,
                         npu_completion_t *completion);
uint32_t npu_read_register(const npu_device_t *device, uint32_t offset);
void npu_write_register(npu_device_t *device, uint32_t offset, uint32_t value);

#ifdef __cplusplus
}
#endif

#endif /* STEM_NPU_DRIVER_H */
