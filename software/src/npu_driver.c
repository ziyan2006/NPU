/* SPDX-License-Identifier: MIT */
#include "npu_driver.h"

#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/* Keep the portable driver usable in a freestanding A9 bring-up image.  A
 * normal hosted build is free to provide libc, but requiring memset here made
 * a tiny -nostdlib JTAG test pull an otherwise unnecessary C runtime. */
static void npu_zero(void *address, size_t bytes)
{
    uint8_t *cursor = (uint8_t *)address;

    while (bytes-- != 0u)
        *cursor++ = 0u;
}

static void npu_barrier(npu_device_t *device)
{
    if (device->platform.barrier != NULL) {
        device->platform.barrier(device->platform.context);
        return;
    }
#if defined(__GNUC__) || defined(__clang__)
    __sync_synchronize();
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

static uint32_t npu_read_le32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}

static int npu_add_overflows_u64(uint64_t start, uint64_t bytes)
{
    return bytes > UINT64_MAX - start;
}

static int npu_add_overflows_uintptr(uintptr_t start, size_t bytes)
{
    return bytes > UINTPTR_MAX - start;
}

static int npu_resident_range_valid(const npu_resident_t *resident,
                                    const npu_range_t *range)
{
    uintptr_t task_cpu;
    uintptr_t range_cpu;
    uintptr_t task_cpu_end;
    uint64_t task_dma_end;
    uint64_t range_dma_end;

    if (resident == NULL || range == NULL || range->cpu_address == NULL
        || range->bytes == 0u)
        return 0;
    task_cpu = (uintptr_t)resident->task_cpu_address;
    range_cpu = (uintptr_t)range->cpu_address;
    if (npu_add_overflows_uintptr(task_cpu, resident->task_bytes)
        || npu_add_overflows_uintptr(range_cpu, range->bytes)
        || npu_add_overflows_u64(resident->task_physical_address,
                                 (uint64_t)resident->task_bytes)
        || npu_add_overflows_u64(range->physical_address,
                                 (uint64_t)range->bytes))
        return 0;
    task_cpu_end = task_cpu + resident->task_bytes;
    task_dma_end = resident->task_physical_address + resident->task_bytes;
    range_dma_end = range->physical_address + range->bytes;
    if (range_cpu < task_cpu || range_cpu + range->bytes > task_cpu_end
        || range->physical_address < resident->task_physical_address
        || range_dma_end > task_dma_end)
        return 0;
    return (uint64_t)(range_cpu - task_cpu)
        == range->physical_address - resident->task_physical_address;
}

uint32_t npu_read_register(const npu_device_t *device, uint32_t offset)
{
    return device->registers[offset >> 2];
}

void npu_write_register(npu_device_t *device, uint32_t offset, uint32_t value)
{
    device->registers[offset >> 2] = value;
}

static uint64_t npu_read_counter64(const npu_device_t *device,
                                   uint32_t low_offset,
                                   uint32_t high_offset)
{
    uint32_t high_before;
    uint32_t low;
    uint32_t high_after;

    do {
        high_before = npu_read_register(device, high_offset);
        low = npu_read_register(device, low_offset);
        high_after = npu_read_register(device, high_offset);
    } while (high_before != high_after);
    return ((uint64_t)high_after << 32) | low;
}

static void npu_finish_cache(npu_device_t *device)
{
    if (!device->cache_pending)
        return;
    npu_barrier(device);
    if (device->submitted_cpu_address != NULL
        && device->platform.invalidate != NULL) {
        device->platform.invalidate(device->submitted_cpu_address,
                                    device->submitted_bytes,
                                    device->platform.context);
    }
    npu_barrier(device);
    device->cache_pending = 0u;
}

npu_result_t npu_device_init(npu_device_t *device,
                             volatile void *register_base,
                             const npu_platform_ops_t *platform)
{
    uint32_t rtl_version;
    uint32_t isa_version;

    if (device == NULL || register_base == NULL)
        return NPU_E_INVALID;
    npu_zero(device, sizeof(*device));
    device->registers = (volatile uint32_t *)register_base;
    if (platform != NULL)
        device->platform = *platform;
    if (npu_read_register(device, NPU_REG_IP_ID) != NPU_IP_ID_VALUE)
        return NPU_E_INCOMPATIBLE;
    rtl_version = npu_read_register(device, NPU_REG_VERSION);
    isa_version = npu_read_register(device, NPU_REG_ISA_VERSION);
    if ((rtl_version >> 16) != NPU_RTL_VERSION_MAJOR
        || (isa_version >> 16) != NPU_DRIVER_ISA_MAJOR)
        return NPU_E_INCOMPATIBLE;
#if NPU_DRIVER_ISA_MINOR > 0
    if ((isa_version & 0xffffu) < NPU_DRIVER_ISA_MINOR)
        return NPU_E_INCOMPATIBLE;
#endif
    return NPU_OK;
}

npu_result_t npu_soft_reset(npu_device_t *device)
{
    uint32_t control;

    if (device == NULL || device->registers == NULL)
        return NPU_E_INVALID;
    if (npu_read_register(device, NPU_REG_STATUS) & NPU_STATUS_BUSY)
        return NPU_E_BUSY;
    control = npu_read_register(device, NPU_REG_CONTROL);
    npu_write_register(device, NPU_REG_CONTROL,
                       control | NPU_CONTROL_SOFT_RESET);
    npu_barrier(device);
    device->cache_pending = 0u;
    return NPU_OK;
}

npu_result_t npu_configure_interrupts(npu_device_t *device,
                                      uint32_t interrupt_mask,
                                      int global_enable)
{
    if (device == NULL || device->registers == NULL
        || (interrupt_mask & ~NPU_IRQ_ALL) != 0u)
        return NPU_E_INVALID;
    npu_write_register(device, NPU_REG_IRQ_ENABLE,
                       interrupt_mask & NPU_IRQ_ALL);
    npu_write_register(device, NPU_REG_CONTROL,
                       global_enable ? NPU_CONTROL_IRQ_GLOBAL_ENABLE : 0u);
    npu_barrier(device);
    return NPU_OK;
}

void npu_acknowledge_interrupts(npu_device_t *device, uint32_t interrupt_mask)
{
    if (device == NULL || device->registers == NULL)
        return;
    npu_write_register(device, NPU_REG_IRQ_STATUS,
                       interrupt_mask & NPU_IRQ_ALL);
    npu_barrier(device);
}

npu_result_t npu_submit(npu_device_t *device,
                        uint64_t task_physical_address,
                        void *task_cpu_address,
                        size_t task_bytes,
                        uint32_t task_tag,
                        uint32_t watchdog_cycles)
{
    uint32_t status;

    if (device == NULL || device->registers == NULL
        || (task_physical_address & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes < NPU_TASK_HEADER_BYTES
        || (task_bytes & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes > UINT32_MAX)
        return NPU_E_INVALID;
    status = npu_read_register(device, NPU_REG_STATUS);
    if ((status & NPU_STATUS_IDLE) == 0u)
        return NPU_E_BUSY;

    if (task_cpu_address != NULL && device->platform.clean != NULL) {
        device->platform.clean(task_cpu_address, task_bytes,
                               device->platform.context);
    }
    npu_barrier(device);
    npu_write_register(device, NPU_REG_IRQ_STATUS, NPU_IRQ_ALL);
    npu_write_register(device, NPU_REG_TASK_BASE_HI,
                       (uint32_t)(task_physical_address >> 32));
    npu_write_register(device, NPU_REG_TASK_BASE_LO,
                       (uint32_t)task_physical_address);
    npu_write_register(device, NPU_REG_TASK_BYTES, (uint32_t)task_bytes);
    npu_write_register(device, NPU_REG_TASK_TAG, task_tag);
    npu_write_register(device, NPU_REG_WATCHDOG_LIMIT, watchdog_cycles);

    /* Publish completion/cache bookkeeping before the doorbell.  An enabled
     * IRQ may preempt this function as soon as hardware accepts the task. */
    device->submitted_cpu_address = task_cpu_address;
    device->submitted_bytes = task_bytes;
    device->submitted_tag = task_tag;
    device->cache_pending = 1u;
    npu_barrier(device);
    npu_write_register(device, NPU_REG_DOORBELL, 1u);
    npu_barrier(device);
    return NPU_OK;
}

npu_result_t npu_resident_prepare(npu_resident_t *resident,
                                  npu_device_t *device,
                                  uint64_t task_physical_address,
                                  void *task_cpu_address,
                                  size_t task_bytes)
{
    uintptr_t task_cpu;

    if (resident == NULL || device == NULL || device->registers == NULL
        || task_cpu_address == NULL
        || (task_physical_address & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes < NPU_TASK_HEADER_BYTES
        || (task_bytes & (NPU_TASK_ALIGNMENT - 1u)) != 0u
        || task_bytes > UINT32_MAX
        || npu_add_overflows_u64(task_physical_address, (uint64_t)task_bytes))
        return NPU_E_INVALID;
    task_cpu = (uintptr_t)task_cpu_address;
    if (npu_add_overflows_uintptr(task_cpu, task_bytes))
        return NPU_E_INVALID;

    npu_zero(resident, sizeof(*resident));
    resident->device = device;
    resident->task_cpu_address = task_cpu_address;
    resident->task_physical_address = task_physical_address;
    resident->task_bytes = task_bytes;
    resident->expected_commands = npu_read_le32(
        (const uint8_t *)task_cpu_address + 0x1cu);
    if (resident->expected_commands == 0u)
        return NPU_E_INVALID;
    if (device->platform.clean != NULL)
        device->platform.clean(task_cpu_address, task_bytes,
                               device->platform.context);
    npu_barrier(device);
    resident->prepared = 1u;
    return NPU_OK;
}

npu_result_t npu_resident_submit(npu_resident_t *resident,
                                 const npu_range_t *clean_range,
                                 const npu_range_t *invalidate_range,
                                 uint32_t task_tag,
                                 uint32_t watchdog_cycles)
{
    npu_device_t *device;
    npu_result_t result;

    if (resident == NULL || resident->prepared == 0u
        || resident->device == NULL || task_tag == 0u
        || !npu_resident_range_valid(resident, clean_range)
        || !npu_resident_range_valid(resident, invalidate_range))
        return NPU_E_INVALID;
    device = resident->device;
    if ((npu_read_register(device, NPU_REG_STATUS) & NPU_STATUS_IDLE) == 0u)
        return NPU_E_BUSY;

    if (device->platform.clean != NULL) {
        device->platform.clean(clean_range->cpu_address, clean_range->bytes,
                               device->platform.context);
    }
    npu_barrier(device);
    npu_write_register(device, NPU_REG_IRQ_STATUS, NPU_IRQ_ALL);
    npu_write_register(device, NPU_REG_TASK_BASE_HI,
                       (uint32_t)(resident->task_physical_address >> 32));
    npu_write_register(device, NPU_REG_TASK_BASE_LO,
                       (uint32_t)resident->task_physical_address);
    npu_write_register(device, NPU_REG_TASK_BYTES, (uint32_t)resident->task_bytes);
    npu_write_register(device, NPU_REG_TASK_TAG, task_tag);
    npu_write_register(device, NPU_REG_WATCHDOG_LIMIT, watchdog_cycles);
    npu_barrier(device);
    npu_write_register(device, NPU_REG_DOORBELL, 1u);
    npu_barrier(device);

    result = npu_wait(device, watchdog_cycles, &resident->completion);
    if (result != NPU_E_TIMEOUT && device->platform.invalidate != NULL) {
        device->platform.invalidate(invalidate_range->cpu_address,
                                    invalidate_range->bytes,
                                    device->platform.context);
        npu_barrier(device);
    }
    if (result != NPU_OK)
        return result;
    if (resident->completion.completed_tag != task_tag
        || resident->completion.error_code != 0u
        || resident->completion.statistics.commands_retired
               != resident->expected_commands)
        return NPU_E_HARDWARE;
    return NPU_OK;
}

npu_state_t npu_poll(npu_device_t *device)
{
    uint32_t status;

    if (device == NULL || device->registers == NULL)
        return NPU_STATE_ERROR;
    status = npu_read_register(device, NPU_REG_STATUS);
    if (status & NPU_STATUS_ERROR) {
        npu_finish_cache(device);
        return NPU_STATE_ERROR;
    }
    if (status & NPU_STATUS_DONE) {
        npu_finish_cache(device);
        return NPU_STATE_DONE;
    }
    if (status & NPU_STATUS_BUSY)
        return NPU_STATE_BUSY;
    return NPU_STATE_IDLE;
}

void npu_read_completion(const npu_device_t *device,
                         npu_completion_t *completion)
{
    if (device == NULL || device->registers == NULL || completion == NULL)
        return;
    npu_zero(completion, sizeof(*completion));
    completion->completed_tag = npu_read_register(
        device, NPU_REG_COMPLETED_TAG);
    completion->error_code = (uint16_t)npu_read_register(
        device, NPU_REG_ERROR_CODE);
    completion->error_pc = npu_read_register(device, NPU_REG_ERROR_PC);
    completion->error_instruction_tag = (uint16_t)npu_read_register(
        device, NPU_REG_ERROR_INST_TAG);
    completion->statistics.commands_retired = npu_read_register(
        device, NPU_REG_COMMANDS_RETIRED);
    completion->statistics.cycles_total = npu_read_counter64(
        device, NPU_REG_CYCLES_TOTAL_LO, NPU_REG_CYCLES_TOTAL_HI);
    completion->statistics.cycles_compute = npu_read_counter64(
        device, NPU_REG_CYCLES_COMPUTE_LO, NPU_REG_CYCLES_COMPUTE_HI);
    completion->statistics.cycles_read_wait = npu_read_counter64(
        device, NPU_REG_CYCLES_RD_WAIT_LO, NPU_REG_CYCLES_RD_WAIT_HI);
    completion->statistics.cycles_write_wait = npu_read_counter64(
        device, NPU_REG_CYCLES_WR_WAIT_LO, NPU_REG_CYCLES_WR_WAIT_HI);
    completion->statistics.cycles_bank_stall = npu_read_counter64(
        device, NPU_REG_CYCLES_BANK_STALL_LO,
        NPU_REG_CYCLES_BANK_STALL_HI);
    completion->statistics.bytes_read = npu_read_counter64(
        device, NPU_REG_BYTES_READ_LO, NPU_REG_BYTES_READ_HI);
    completion->statistics.bytes_written = npu_read_counter64(
        device, NPU_REG_BYTES_WRITTEN_LO, NPU_REG_BYTES_WRITTEN_HI);
    completion->statistics.read_high_water = npu_read_counter64(
        device, NPU_REG_READ_HIGH_WATER_LO, NPU_REG_READ_HIGH_WATER_HI);
    completion->statistics.write_high_water = npu_read_counter64(
        device, NPU_REG_WRITE_HIGH_WATER_LO,
        NPU_REG_WRITE_HIGH_WATER_HI);
    completion->statistics.error_count = npu_read_register(
        device, NPU_REG_ERROR_COUNT);
}

npu_result_t npu_wait(npu_device_t *device,
                      uint32_t maximum_polls,
                      npu_completion_t *completion)
{
    uint32_t poll;

    if (device == NULL || device->registers == NULL)
        return NPU_E_INVALID;
    for (poll = 0u; poll < maximum_polls; ++poll) {
        npu_state_t state = npu_poll(device);
        if (state == NPU_STATE_DONE) {
            npu_read_completion(device, completion);
            return NPU_OK;
        }
        if (state == NPU_STATE_ERROR) {
            npu_read_completion(device, completion);
            return NPU_E_HARDWARE;
        }
    }
    return NPU_E_TIMEOUT;
}
