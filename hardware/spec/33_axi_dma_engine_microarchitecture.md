# AXI DMA Engine 微架构规格

文档版本：`0.1-draft`

状态：P4 单 outstanding 可综合正确性原型；尚未接入真实 Zynq HP 端口或完成 Vivado 时序

## 1. 目标与边界

`npu_dma_engine.sv` 消费 AGU 产生的 `npu_dma_request_t`，负责：

- 可选的 activation scratchpad 清零；
- 展开 `x_bytes × y_count × z_count`；
- 生成 64-bit AXI4 INCR read/write burst；
- 按 4 KiB 边界和 256-beat 上限拆分；
- 处理 4/2/1-byte 窄尾部，不越界补读；
- 适配 DDR 与 scratchpad 双向 back-pressure；
- 最终完成时设置 event，错误时上报 reason/tag。

首版一次只允许一个 AXI transaction outstanding。该选择用于先关闭协议与数据正确性，不代表最终性能配置。

## 2. Scratchpad 接口

Load/clear 使用一个 64-bit write ready-valid 端口：kind、bank、8-byte 对齐地址、data 和 byte strobe。Store 使用解耦的 read request/response 端口，以适配同步 BRAM 的至少一周期读延迟。

地址是 bank 内 byte address。窄传输使用地址低三位选择 byte lane，但提供给 scratchpad wrapper 的地址按 8-byte 对齐；byte strobe 决定真正更新的 lane。

## 3. Burst 生成

每一逻辑行独立处理：

```text
remaining >= 8 : 8-byte beat，最多 min(floor(remaining/8), 256, 4KiB边界)
remaining >= 4 : 一个 4-byte beat
remaining >= 2 : 一个 2-byte beat
otherwise      : 一个 1-byte beat
```

因此 `x_bytes=6` 会生成 4 B + 2 B，而不是发一个 8 B read 后丢弃两字节。每个 burst 的 `ARLEN/AWLEN` 在 valid 被 back-pressure 时保持不变。

当前请求要求每一行 DDR/SP 起点 8-byte 对齐，且多行 stride 是 8 的倍数。现有网络 A 的全部请求满足约束。未来若网络 B 需要任意 2-byte 起点，可在不改 ISA 的前提下扩展 lane realignment。

## 4. Read 路径

```text
AR handshake → R beats → scratchpad writes → 下一 burst/row/plane
```

正常 beat 只有在 scratchpad 接受写入时才拉高 `RREADY`。`RRESP` 错误后停止写 scratchpad，但继续接收直到 `RLAST`，再报告错误。提前或缺失 `RLAST` 报 protocol error；缺失时继续 drain，避免遗留半个 AXI transaction。

## 5. Write 路径

```text
AW handshake → SP read request/response → W beat(s) → B response
```

每个 W beat 的 data/strobe/last 在 `WREADY=0` 时保持稳定。收到非 OKAY `BRESP` 后报告 write error；因为 B 只在全部 W beat 完成后返回，所以不会遗留未发送的数据 beat。

## 6. 完成、错误和复位

- 最后一行最后一个 R beat，或最后一个 write response 成功后，产生单周期 `done_pulse` 和请求携带的 `event_set`；
- 错误产生单周期 `error_pulse`、reason 和原请求 tag，不产生 completion event；
- 计数器报告本请求实际完成的 read/write payload byte；
- busy 时的 soft reset 不会抛弃已接受请求。外层 reset controller 先停止新命令、等待 engine idle，再保持/重发 soft reset 清状态。

## 7. 网络 A burst 审计

`scripts/31_plan_axi_dma.py` 使用与 RTL 相同的拆分规则审计全部请求：

| 项目 | 数值 |
|---|---:|
| normalized requests | 1,078 |
| 逻辑 DMA 行 | 44,940 |
| AXI bursts | 45,664 |
| AXI beats | 378,696 |
| read / write bursts | 13,920 / 31,744 |
| narrow bursts | 2,272 |
| 4 KiB splits | 374 |
| payload | 3,020,480 B，精确一致 |
| 对齐/进度检查 | 全部通过 |

按每 burst 16-cycle 保守开销估算，read/write 分别为 539,976/569,344 cycle。把真实逐行 transaction 和 1,246,656 B activation clear 纳入调度后，网络 A 总估算更新为 2,485,596 cycle：12.43 ms@200 MHz、24.86 ms@100 MHz，仍低于 4,000,000-cycle 目标。

write burst 数量较大，是因为每个 8-output-channel tile 写回完整 Tensor 中的一段通道。它是明确的性能优化点：后续可比较更大 `tile_cout`、write combining 或多个 outstanding，但必须同时满足 W/P/O bank 和时序预算。

## 8. 当前验证

Icarus testbench 使用可重复停顿的 AXI/scratchpad memory model，已覆盖：

- read 和 write 的 4 KiB 拆分；
- 256-beat 最大 burst；
- 8/4/2-byte 数据与 byte strobe；
- 2D/3D 独立 stride 和 activation clear；
- AR/AW/W back-pressure 下 payload 稳定；
- RRESP、BRESP、提前 RLAST 错误；
- busy 时 soft reset 不丢事务；
- completion event、tag 和 byte counter。

尚未覆盖真实 AXI interconnect、多个 outstanding、CDC、Vivado AXI protocol checker、综合资源和 post-route Fmax。

## 9. 下一步

实现 scratchpad BRAM wrapper 与 DMA unit sequencer，把 Descriptor Cache、AGU、DMA Engine 接成一条可运行的 `DMA_LOAD/STORE` 通路；随后开始 8×8 Tensor MAC loop controller 和 BRAM 读端口设计。
