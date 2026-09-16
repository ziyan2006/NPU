# 领航者 ZYNQ-7020 软件首测方案

这份方案以 JTAG + bare-metal 为第一目标。Linux 和永久启动放在 NPU/DDR 链路验证之后，
避免同时调试设备树、DMA 分配、启动介质和硬件。

## 工件边界

- CSR：`0x43C0_0000`，4 KiB；
- task image：64-byte 对齐、大小为 64-byte 整数倍；
- HP0：非 cache-coherent，提交前 flush，完成后 invalidate；
- 首测 watchdog：`4,000,000` cycle；
- 预期完整任务：1,869 command，约 3,254,220 cycle，输出 32,768 byte；
- IP ID：`0x3155504E`，RTL/ISA major 都为 1。

## XSA 导入后需要补的薄适配层

```c
static void cache_clean(void *p, size_t n, void *ctx) {
    (void)ctx;
    Xil_DCacheFlushRange((INTPTR)p, (u32)n);
}

static void cache_invalidate(void *p, size_t n, void *ctx) {
    (void)ctx;
    Xil_DCacheInvalidateRange((INTPTR)p, (u32)n);
}

static void memory_barrier(void *ctx) {
    (void)ctx;
    __asm__ volatile("dmb sy" ::: "memory");
}
```

把这三个函数填入 `npu_platform_ops_t`，CSR 基址优先使用 XSA 生成的 `xparameters.h`
宏，并用编译期断言核对其值是 `0x43C00000`。不要把 CPU 虚拟地址直接当成 DMA 地址。

## 首次串口程序步骤

1. 打印 XSA/软件版本、CSR base 和 task DDR 地址；
2. 读 `IP_ID`、RTL version、ISA version、capability；不匹配立即停止；
3. 先跑 256-byte 的错误路径/最小 task，确认寄存器和 watchdog 可恢复；
4. 将完整 task image 放入 DDR 的 64-byte 对齐缓冲区，flush 后提交；
5. 轮询完成，invalidate，再和嵌入的 golden output 做逐字节比对；
6. 打印 completed tag、error context、各 cycle 和 byte counter；
7. 循环 30 分钟，任何 mismatch、timeout 或 error count 增长都判失败。

## 后续 Linux 适配

Linux 阶段使用内核 DMA API 分配/映射 task buffer，并把 NPU 描述为一个 4 KiB MMIO
设备和单路 IRQ。实际 GIC SPI 号必须由最终 XSA/设备树生成结果确定，不能在拿板前写死。
HP0 仍是非一致性端口，因此驱动必须遵守 DMA 同步语义；UIO 只适合早期 CSR 观察，不能
替代正式 DMA/cache 管理。
