# NPU PS 软件接口

`include/npu_driver.h` 与 `src/npu_driver.c` 是不绑定操作系统的 CSR 驱动核心。它负责：

- 校验 IP/RTL/ISA 主版本；
- 配置 IRQ、watchdog、task 物理地址、长度和 tag；
- 在 `DOORBELL` 前执行可注入的 cache clean 与内存屏障；
- 轮询完成/错误，并在 PS 读取输出前执行 cache invalidate；
- 按 `HI-LO-HI` 顺序稳定读取 64-bit 性能计数器；
- 返回完整错误上下文和性能统计。

平台适配层只需提供 MMIO 映射地址，以及可选的 cache clean、invalidate、barrier
回调。对 Zynq bare-metal，回调应封装 `Xil_DCacheFlushRange`、
`Xil_DCacheInvalidateRange` 和 `dmb sy`；Linux 驱动必须使用 DMA API，不能用用户态
虚拟地址直接当作物理地址。非 cache-coherent HP 端口下，不能省略 cache 维护。

典型提交流程：

```c
npu_device_t npu;
npu_completion_t completion;

npu_device_init(&npu, mapped_csr_base, &platform_ops);
npu_configure_interrupts(&npu, NPU_IRQ_DONE | NPU_IRQ_ERROR, 1);
npu_submit(&npu, task_dma_address, task_cpu_address, task_bytes,
           task_tag, 4000000);
if (npu_wait(&npu, maximum_polls, &completion) != NPU_OK) {
    /* inspect completion.error_code/error_pc/error_instruction_tag */
}
```

主机单元测试：

```sh
cc -std=c11 -Wall -Wextra -Werror -Isoftware/include \
  software/src/npu_driver.c software/tests/test_npu_driver.c \
  -o test_npu_driver
./test_npu_driver
```

当前 Windows 环境没有原生 host C 编译器，仓库提供的自动测试会核对 39 个 CSR 与
RTL read-decode，并用 Vivado 自带 MicroBlaze GCC 以 `-Wall -Wextra -Werror`
交叉编译驱动和测试源：

```powershell
python scripts/_test_npu_driver.py
```

`software/bringup/navigator_vitis/` 已提供 Zynq standalone 的 BSP 适配和完整 task
runner：使用 linker-owned、64-byte 对齐的 DDR staging buffer，提交前 clean、完成后
invalidate，并通过 payload generator 将本地 task / golden output 转换为 Vitis C 源。
它仍需在最终 XSA 的 Vitis BSP 中编译并上板执行；GIC IRQ 号将由最终 XSA 确认后再启用。
Linux 适配仍待实现，必须使用内核 DMA API 而不是直接复用 standalone 指针。
