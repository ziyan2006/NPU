# NPU 内部硬件框图与 I/O

本页描述当前 `hardware/rtl/npu_top.sv` 和 `npu_core.sv` 的实现。

![NPU 内部硬件框图与 I/O](03_npu_internal_io.png)

[SVG](03_npu_internal_io.svg) · [PDF](03_npu_internal_io.pdf) · [可编辑 drawio](03_npu_internal_io.drawio)

## 顶层接口

输入 / 输出方向均以 NPU 为参照。以下表格由顶层 RTL 端口声明直接生成。

模型输入、权重和输出均位于 DDR；NPU 经 AXI4 主接口读写，不暴露独立的 PCM 或 AXI-Stream 音频接口。

AXI4-Lite 是从接口，接收 PS 的控制请求；AXI4 是主接口，主动发起 DDR 访问。

所有 AXI 通道都通过 VALID/READY 握手；READY 的方向与该通道有效负载和 VALID 相反。

## 时钟、复位、中断

| 信号 | 方向 | 位宽 |
|---|---|---:|
| `aclk` | 输入 | 1 |
| `aresetn` | 输入 | 1 |
| `irq_o` | 输出 | 1 |

## AXI4-Lite 控制口

| 信号 | 方向 | 位宽 |
|---|---|---:|
| `s_axi_ctrl_awaddr` | 输入 | 12 |
| `s_axi_ctrl_awprot` | 输入 | 3 |
| `s_axi_ctrl_awvalid` | 输入 | 1 |
| `s_axi_ctrl_awready` | 输出 | 1 |
| `s_axi_ctrl_wdata` | 输入 | 32 |
| `s_axi_ctrl_wstrb` | 输入 | 4 |
| `s_axi_ctrl_wvalid` | 输入 | 1 |
| `s_axi_ctrl_wready` | 输出 | 1 |
| `s_axi_ctrl_bresp` | 输出 | 2 |
| `s_axi_ctrl_bvalid` | 输出 | 1 |
| `s_axi_ctrl_bready` | 输入 | 1 |
| `s_axi_ctrl_araddr` | 输入 | 12 |
| `s_axi_ctrl_arprot` | 输入 | 3 |
| `s_axi_ctrl_arvalid` | 输入 | 1 |
| `s_axi_ctrl_arready` | 输出 | 1 |
| `s_axi_ctrl_rdata` | 输出 | 32 |
| `s_axi_ctrl_rresp` | 输出 | 2 |
| `s_axi_ctrl_rvalid` | 输出 | 1 |
| `s_axi_ctrl_rready` | 输入 | 1 |

## AXI4 内存读口 AR / R

| 信号 | 方向 | 位宽 |
|---|---|---:|
| `m_axi_mem_arid` | 输出 | 4 |
| `m_axi_mem_araddr` | 输出 | 64 |
| `m_axi_mem_arlen` | 输出 | 8 |
| `m_axi_mem_arsize` | 输出 | 3 |
| `m_axi_mem_arburst` | 输出 | 2 |
| `m_axi_mem_arlock` | 输出 | 1 |
| `m_axi_mem_arcache` | 输出 | 4 |
| `m_axi_mem_arprot` | 输出 | 3 |
| `m_axi_mem_arqos` | 输出 | 4 |
| `m_axi_mem_arvalid` | 输出 | 1 |
| `m_axi_mem_arready` | 输入 | 1 |
| `m_axi_mem_rid` | 输入 | 4 |
| `m_axi_mem_rdata` | 输入 | 64 |
| `m_axi_mem_rresp` | 输入 | 2 |
| `m_axi_mem_rlast` | 输入 | 1 |
| `m_axi_mem_rvalid` | 输入 | 1 |
| `m_axi_mem_rready` | 输出 | 1 |

## AXI4 内存写口 AW / W / B

| 信号 | 方向 | 位宽 |
|---|---|---:|
| `m_axi_mem_awid` | 输出 | 4 |
| `m_axi_mem_awaddr` | 输出 | 64 |
| `m_axi_mem_awlen` | 输出 | 8 |
| `m_axi_mem_awsize` | 输出 | 3 |
| `m_axi_mem_awburst` | 输出 | 2 |
| `m_axi_mem_awlock` | 输出 | 1 |
| `m_axi_mem_awcache` | 输出 | 4 |
| `m_axi_mem_awprot` | 输出 | 3 |
| `m_axi_mem_awqos` | 输出 | 4 |
| `m_axi_mem_awvalid` | 输出 | 1 |
| `m_axi_mem_awready` | 输入 | 1 |
| `m_axi_mem_wdata` | 输出 | 64 |
| `m_axi_mem_wstrb` | 输出 | 8 |
| `m_axi_mem_wlast` | 输出 | 1 |
| `m_axi_mem_wvalid` | 输出 | 1 |
| `m_axi_mem_wready` | 输入 | 1 |
| `m_axi_mem_bid` | 输入 | 4 |
| `m_axi_mem_bresp` | 输入 | 2 |
| `m_axi_mem_bvalid` | 输入 | 1 |
| `m_axi_mem_bready` | 输出 | 1 |

## 内部连接与位宽

| 连接 | 载荷 / 行宽 | 说明 |
|---|---|---|
| CSR → Core | task base 64；bytes/tag/watchdog 各 32 bit | start / soft reset 为 1-bit 脉冲 |
| Fetch → Command Processor | command 128；PC 32 bit | 经 core 内单级指令寄存 |
| Command Processor → DMA / Execution Frontend | command 128 bit | DMA 与 compute/vector 为独立分派接口 |
| Loader / Fetch / DMA descriptor / Execution descriptor → Memory Arbiter | address 64；bytes 7；response 512 bit | 四个客户端共享 Block Reader |
| Execution Frontend → CONV / VEC | operator descriptor 512 bit | CONV 还含 A/W/O bank 选择和 event 8 bit |
| Execution Frontend → UPSAMPLE | source / destination descriptor 各 512 bit | 上采样直接访问 DDR |
| DMA Engine ↔ Scratchpad | data 64；address 32；write strobe 8 bit | A/W/O bank 读写 |
| Scratchpad → CONV | A 128；W 512 bit | 独立 A/W 端口；W bank 也提供 bias/quant |
| MAC → Requant/Post | 256 bit = 8 × INT32 | 64 个 INT16×INT8 乘法 lane |
| Post → FIFO → O bank | 128 bit = 8 × INT16 | 2-entry 输出 FIFO，写回排空后产生完成事件 |
| Scratchpad ↔ VEC_ADD | read/write data 各 512 bit；write strobe 64 bit | 有效向量可能只使用部分 byte lane |
| Loader → CONV tanh LUT [L] | data 16；address 12 bit | 4096 项 LUT 预载；图中同名连接省略长连线 |

## 读写仲裁

- 读仲裁器有三个客户端：Block Reader、DMA、UPSAMPLE。
- 写仲裁器有两个客户端：DMA、UPSAMPLE。
- descriptor cache 分别位于 DMA Frontend 和 Execution Frontend，不是只有一个全局 cache。
- busy/done/error/event 返回 core 控制逻辑；图中为避免遮挡省略部分反馈和 section 基址分发连线。

## 数值与外部适配

- INT12 是激活有效精度；实际 bank 存储和 MAC 输入使用 16 bit。
- 数据端口地址是 64 bit；接 Zynq-7000 HP0 时经 AXI4→AXI3 适配。
- ARID/AWID 固定为 0；RID/BID 被接受但不参与 core 响应匹配。
- 控制口 AWPROT/ARPROT 被接受但不参与 CSR 权限译码。
- aclk 当前为 100 MHz；aresetn 低有效，所有 NPU 功能逻辑同一时钟域。

## 重新生成

```sh
python 框图/render_npu_internal.py
```
