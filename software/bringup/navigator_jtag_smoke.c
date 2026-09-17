/* SPDX-License-Identifier: MIT */
/* First JTAG test for a Z7020 Navigator board after loading the NPU bitstream. */

#include <stddef.h>
#include <stdint.h>

#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "npu_driver.h"

#ifndef NPU_DDR_TEST_MIB
#define NPU_DDR_TEST_MIB 4u
#endif

#if NPU_DDR_TEST_MIB < 1 || NPU_DDR_TEST_MIB > 64
#error "NPU_DDR_TEST_MIB must be in 1..64"
#endif

#define NPU_CSR_BASE       0x43c00000u
#define DDR_TEST_WORDS     (NPU_DDR_TEST_MIB * 1024u * 1024u / 4u)

/* The linker reserves this buffer, so the test cannot trample code or stack. */
static volatile uint32_t ddr_test[DDR_TEST_WORDS] __attribute__((aligned(64)));

static uint32_t pattern(size_t index, unsigned phase)
{
    uint32_t value = (uint32_t)(index + 1u) * 0x9e3779b9u;
    value ^= 0xa5a55a5au;
    return phase ? ~value : value;
}

static int check_ddr(void)
{
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (size_t i = 0; i < DDR_TEST_WORDS; ++i) {
            ddr_test[i] = pattern(i, phase);
        }
        Xil_DCacheFlushRange((INTPTR)(uintptr_t)ddr_test, sizeof(ddr_test));
        Xil_DCacheInvalidateRange((INTPTR)(uintptr_t)ddr_test, sizeof(ddr_test));
        for (size_t i = 0; i < DDR_TEST_WORDS; ++i) {
            uint32_t expected = pattern(i, phase);
            uint32_t actual = ddr_test[i];
            if (actual != expected) {
                xil_printf("DDR FAIL phase=%u word=%u got=%08x expected=%08x\r\n",
                           phase, (unsigned)i, (unsigned)actual, (unsigned)expected);
                return -1;
            }
        }
    }
    xil_printf("DDR PASS %u MiB, two patterns\r\n", (unsigned)NPU_DDR_TEST_MIB);
    return 0;
}

int main(void)
{
    xil_printf("Navigator Z7020 NPU JTAG smoke\r\n");
    xil_printf("DDR buffer=%08x bytes=%u\r\n",
               (unsigned)(uintptr_t)ddr_test, (unsigned)sizeof(ddr_test));
    if (check_ddr() != 0) {
        return 1;
    }

    /* Read CSR only after the NPU bitstream has been programmed via JTAG. */
    uint32_t id = Xil_In32(NPU_CSR_BASE + NPU_REG_IP_ID);
    uint32_t version = Xil_In32(NPU_CSR_BASE + NPU_REG_VERSION);
    uint32_t isa = Xil_In32(NPU_CSR_BASE + NPU_REG_ISA_VERSION);
    uint32_t status = Xil_In32(NPU_CSR_BASE + NPU_REG_STATUS);
    xil_printf("NPU id=%08x rtl=%08x isa=%08x status=%08x\r\n",
               (unsigned)id, (unsigned)version, (unsigned)isa, (unsigned)status);
    if (id != NPU_IP_ID_VALUE || (status & NPU_STATUS_IDLE) == 0u) {
        xil_printf("NPU CSR FAIL\r\n");
        return 2;
    }
    xil_printf("NPU CSR PASS\r\n");
    return 0;
}
