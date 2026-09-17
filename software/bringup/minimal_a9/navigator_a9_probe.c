/* Minimal JTAG-only A9 execution probe. No Vitis runtime or Flash access. */
#include <stdint.h>

#define RESULT_BASE 0x12000000u
#define TEST_BASE   0x12010000u
#define NPU_BASE    0x43c00000u
#define NPU_ID      0x3155504eu
#define RUNNING     0x4e505201u
#define PASS        0x4e5052a5u
#define BAD_CSR     0x4e5052c1u
#define BAD_DDR     0x4e5052d1u

static inline void dsb(void) { __asm__ volatile("dsb sy" ::: "memory"); }
static inline void disable_caches(void) {
    uint32_t sctlr;
    __asm__ volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(sctlr));
    sctlr &= ~((1u << 2) | (1u << 12));
    __asm__ volatile("mcr p15, 0, %0, c1, c0, 0\nisb" :: "r"(sctlr) : "memory");
}

int main(void) {
    volatile uint32_t *const result = (volatile uint32_t *)RESULT_BASE;
    volatile uint32_t *const mem = (volatile uint32_t *)TEST_BASE;
    volatile uint32_t *const csr = (volatile uint32_t *)NPU_BASE;
    const uint32_t words = 4096u; /* 16 KiB, outside probe code/result. */

    disable_caches();
    result[0] = RUNNING;
    result[1] = csr[0];       /* IP_ID */
    result[2] = csr[1];       /* RTL version */
    result[3] = csr[2];       /* ISA version */
    result[4] = csr[3];       /* status */
    dsb();
    if (result[1] != NPU_ID || (result[4] & 1u) == 0u) {
        result[0] = BAD_CSR;
        dsb();
        return 0;
    }
    for (uint32_t phase = 0; phase != 2u; ++phase) {
        for (uint32_t i = 0; i != words; ++i)
            mem[i] = (i * 0x9e3779b9u) ^ (phase ? 0x5a5aa5a5u : 0xa5a55a5au);
        dsb();
        for (uint32_t i = 0; i != words; ++i) {
            uint32_t expected = (i * 0x9e3779b9u) ^
                (phase ? 0x5a5aa5a5u : 0xa5a55a5au);
            if (mem[i] != expected) {
                result[0] = BAD_DDR;
                result[5] = phase;
                result[6] = i;
                result[7] = mem[i];
                dsb();
                return 0;
            }
        }
    }
    result[0] = PASS;
    dsb();
    return 0;
}
