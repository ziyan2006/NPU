/* Minimal SD boot diagnostic: no model payload is embedded in this image. */
#include <stdint.h>

#define UART0_BASE 0xe0000000u
#define UART_SR_OFFSET 0x2cu
#define UART_FIFO_OFFSET 0x30u
#define UART_SR_TXFULL (1u << 4)
#define NPU_ID_REGISTER 0x43c00000u
#define NPU_EXPECTED_ID 0x3155504eu

static inline void dsb(void)
{
    __asm__ volatile("dsb sy" ::: "memory");
}

static void mask_write(uintptr_t address, uint32_t mask, uint32_t value)
{
    volatile uint32_t *const reg = (volatile uint32_t *)address;
    *reg = (*reg & ~mask) | (value & mask);
}

static void release_npu_pl(void)
{
    *(volatile uint32_t *)0xf8000008u = 0x0000df0du;
    mask_write(0xf8000900u, 0x0000000fu, 0x0000000fu);
    mask_write(0xf8000240u, 0xffffffffu, 0x00000000u);
    *(volatile uint32_t *)0xf8000004u = 0x0000767bu;
    dsb();
}

static void uart_putc(char character)
{
    volatile uint32_t *const uart = (volatile uint32_t *)UART0_BASE;
    while ((uart[UART_SR_OFFSET >> 2] & UART_SR_TXFULL) != 0u) { }
    uart[UART_FIFO_OFFSET >> 2] = (uint32_t)(uint8_t)character;
}

static void uart_puts(const char *text)
{
    while (*text != '\0') {
        if (*text == '\n')
            uart_putc('\r');
        uart_putc(*text++);
    }
}

static void uart_hex32(uint32_t value)
{
    static const char digits[] = "0123456789ABCDEF";
    for (int shift = 28; shift >= 0; shift -= 4)
        uart_putc(digits[(value >> (uint32_t)shift) & 0xfu]);
}

__attribute__((noinline, used))
int navigator_sd_probe_npu_id_valid(uint32_t npu_id)
{
    return npu_id == NPU_EXPECTED_ID;
}

int main(void)
{
    uint32_t npu_id;

    uart_puts("STEM_NPU_SD_PROBE v1\n");
    uart_puts("[INFO] releasing PL post-config\n");
    release_npu_pl();
    npu_id = *(volatile uint32_t *)NPU_ID_REGISTER;
    uart_puts("[INFO] NPU_ID=0x");
    uart_hex32(npu_id);
    uart_puts("\n");
    if (!navigator_sd_probe_npu_id_valid(npu_id)) {
        uart_puts("[FAIL] invalid NPU ID\n");
        for (;;)
            __asm__ volatile("wfi");
    }
    uart_puts("[PASS] minimal SD boot probe\n");
    for (;;) {
        uart_puts("[HEARTBEAT] SD probe alive\n");
        for (volatile uint32_t delay = 0u; delay != 80000000u; ++delay)
            __asm__ volatile("nop");
    }
}
