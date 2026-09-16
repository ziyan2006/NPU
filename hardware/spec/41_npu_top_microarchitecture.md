# NPU v1 完整顶层微架构

文档版本：`1.0-implemented`

状态：网络 A 功能 RTL、整任务位精确验证和参考 XC7Z020 OOC post-route 已完成

## 1. 外部边界

`npu_top` 暴露一个 32-bit AXI4-Lite slave、一个 64-bit AXI4 master 和一个电平
IRQ。控制口由 PS 配置 task 物理地址、长度、tag、watchdog 和中断；数据口读取
self-contained task image 并写回激活。master 固定 ID=0、INCR burst、单共享读 burst
和单共享写 burst outstanding，简化错误恢复和 XC7Z020 资源开销。

Zynq-7000 PS 的 HP 口是 AXI3。最终 block design 必须经 AMD AXI SmartConnect 或
Protocol Converter 把本 IP 的 AXI4 master 适配到 HP AXI3；不能在 RTL 外直接按
同名信号硬连。所有逻辑当前在单个 `aclk` 域，CSR、core 和 master 共用
active-low `aresetn`，因此 IP 内部没有功能 CDC。

## 2. 任务生命周期

```text
IDLE --doorbell--> PREPARE --> LOAD --> RUN --> IDLE
                                  |       |
                                  +--error+--> RECOVER --> IDLE
```

- `IDLE`：接受 CSR doorbell；task 配置在此刻复制到 core；
- `PREPARE`：对所有执行单元发内部 reset，清前一任务状态；
- `LOAD`：校验 256-byte header、计算 section 地址并预载 4096 项 tanh LUT；
- `RUN`：顺序 fetch 128-bit 命令，两级寄存译码后向 DMA/compute/vector 分派；
- `RECOVER`：清 event、cache、FIFO 和执行单元，再返回 idle。

END 只有在命令区末尾、全部执行单元和写事务空闲后完成。CONV2D completion 也只在
post 和 O-bank 写回全部排空后置 event，从而保证软件观察到完成时输出已提交。

## 3. 内部数据通路

四客户端 `npu_memory_arbiter4` 把 task loader、command fetch、DMA descriptor 和
execution descriptor 的 64-byte block request 合并给 `npu_axi_block_reader`。其读口
再与 DMA data read、UPSAMPLE read 进入三客户端 AXI read arbiter。DMA write 和
UPSAMPLE write 进入两客户端 AXI write arbiter。arbiter 在地址握手后保持 owner，
直到 RLAST 或 B response，因而 response 不会投递给错误客户端。

当前网络执行单元为：

- `npu_dma_subsystem`：DDR burst、六个 A/W/O BRAM bank、CONV2D 与 post；
- `npu_vec_add`：两个已重定标的残差向量饱和相加并原位写 O bank；
- `npu_upsample2x`：DDR Tensor 间最近邻 2×，使用 1 KiB 行缓冲；
- `npu_tensor_mac_8x8`：64 个 INT16×INT8 lane、INT32 累加；
- `npu_requant_post`：bias、Q31 multiplier/shift、signed RNE、clamp 和 activation。

## 4. 控制、错误和计数器

CSR offset 和 bit 定义见 `21_control_registers.md`。busy 时重复 doorbell 和外部
soft reset 分别报告 `0xf001`、`0xf002`，不会打断活动 AXI transaction。loader、
fetch、descriptor、DMA、CONV2D 和 watchdog 错误最终统一为 code/PC/tag；CSR 保存
sticky error/IRQ，软件 W1C 清中断。core 统计总周期、compute busy、AXI 等待、bank
冲突、实际字节数和最大 burst beat 数。

## 5. 验证与实现基线

`scripts/npu_task_reference.py` 独立解释相同 command、descriptor、DMA plan、O8I8
weight、bias 和 quant 参数。`scripts/_test_npu_top_task.py` 用固定非零 INT12 输入同时
运行解释器与 XSim RTL：当前 1,869 条命令的 32,768-byte 最终输出全部一致，RTL
用 3,254,220 cycle，即 32.54 ms @100 MHz。

Vivado 2026.1、`xc7z020clg400-1`、10 ns OOC post-route 基线；约束包含
0.2 ns 时钟不确定度和 2 ns 同步接口预算：

| Slice LUT | FF | BRAM36 | DSP48E1 | WNS | TNS | WHS |
|---:|---:|---:|---:|---:|---:|---:|
| 25,645 | 24,894 | 61 | 72 | +0.225 ns | 0 ns | +0.007 ns |

所有可路由网络均已布通，critical DRC 为 0。该结果仍是无 PS/interconnect 和真实
part-pin 位置的 PL core OOC 数据；发布上板 bitstream 前必须在具体板卡的完整 block
design 中重新完成 post-route timing、DRC、CDC、地址映射和 PS 软件验收。
