# 领航者 ZYNQ-7020 软件首测方案

这份方案以 JTAG + bare-metal 为第一目标。实物是无底板版本丝印的第三方复刻板，
卖家称参考正点原子 V3.7（WM8960）资料；这不是已验证的官方 PCB 修订号。
FPGA `-2` 速度级别同样仅由卖家提供，尚未经独立核验；当前产物只面向可恢复的
JTAG 首测，不得写入启动 Flash。
Linux 和永久启动放在 NPU/DDR 链路验证之后，避免同时调试设备树、DMA 分配、
启动介质和硬件。

## 工件边界

- CSR：`0x43C0_0000`，4 KiB；
- task image：64-byte 对齐、大小为 64-byte 整数倍；
- HP0：非 cache-coherent，提交前 flush，完成后 invalidate；
- 首测 watchdog：`4,000,000` cycle；
- 完整任务：1,869 command，输出 32,768 byte；实板两组输入测得约 3,346,2xx cycle（约 33.46 ms @ 100 MHz）；
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

## 第一阶段：DDR 与 CSR 冒烟测试

`bringup/navigator_jtag_smoke.c` 是可放入 Vitis standalone 应用的首测源码，先于完整
task 推理程序运行。使用**最终确认的 NPU XSA**创建平台，而不是资料盘里的音频例程 XSA；
将本目录的 `include/` 加入编译器头文件搜索路径。默认测试静态分配的 4 MiB DDR 缓冲区，
检查两轮不同数据图案，再读取 NPU `IP_ID`、版本和空闲状态。缓冲区由链接器分配，
不会直接写某个猜测的物理地址。首次通过后设置编译宏 `NPU_DDR_TEST_MIB=64`，重编译
执行 64 MiB 测试。每轮写入后进行 cache flush/invalidate，以免只测到 CPU cache。

2026-09-17 已用 JTAG DAP 对前 64 MiB 完成两种固定图案的直接物理读回测试，PS7
初始化和 NPU CSR 也已实板通过；这降低了 DDR/MIO 配置错误的风险，但不能代替本 bare-metal
测试，因为后者还会验证 CPU cache 维护、ELF 启动、串口日志和软件寄存器访问。

作为不依赖 Vitis 的执行链路预检，`bringup/minimal_a9/` 提供了更小的 A9 probe：它由
`scripts/50_build_navigator_a9_probe.ps1` 使用 GNU Arm Embedded Toolchain 构建，
`scripts/51_run_navigator_a9_probe.tcl` 通过 JTAG 将其易失下载。2026-09-17 实板结果为
`NPU_A9_PROBE PASS`；A9 从 DDR 执行、关闭 cache 后完成 16 KiB 双图案读回，并读到 NPU
CSR。它没有 UART、IRQ、cache flush/invalidate API 或完整 task，因此不能替代本节的 Vitis
 standalone 应用。

完整 NPU 数据面也已完成 JTAG 实测：全零输入与固定非零输入（seed `0x4e505531`）各运行
1,869 条命令，均输出 32,768 byte 且与 Python 固定点参考逐字节一致。JTAG 是非一致的临时
下载路径；正式 bare-metal 程序仍必须在提交前 clean cache、完成后 invalidate cache。

运行顺序：确认 JTAG 启动模式与 PS UART 跳线，先加载已签核的 NPU bitstream/PS 初始化，
再通过 JTAG 下载 ELF 并观察串口。若 DDR 测试失败，不继续读 NPU CSR；若 CSR
`IP_ID` 不等于 `0x3155504E`，不提交 task。当前这份依赖 Xilinx BSP 的源码尚未在实板或
最终 Vitis 平台编译运行；不要把前述独立 minimal probe 的通过误记为它已经通过。

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
