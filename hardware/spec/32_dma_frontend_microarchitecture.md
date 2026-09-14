# Descriptor Cache 与 DMA AGU 微架构规格

文档版本：`0.1-draft`

状态：P4 可综合多周期原型；网络 A 的 1,078 条 DMA 请求已与软件 reference 逐 bit 对齐，并通过 Vivado 2026.1 OOC 综合

## 1. 模块边界

DMA 前端把粗粒度 `DMA_LOAD/DMA_STORE` 转换为不再依赖 Tensor 语义的三维搬运请求：

```text
DMA command ─┬─► Descriptor Cache ─► Tensor/Operator/Quant/Segment descriptor
             │
             └──────────────────────► DMA AGU
                                           │
                                           ▼
                              x_bytes × y_count × z_count
                              + DDR/SP 独立 y/z stride
                                           │
                                           ▼
                              后续 AXI burst / SP engine
```

`npu_descriptor_cache.sv` 和 `npu_dma_agu.sv` 已可综合并通过 Icarus directed test。`npu_dma_frontend.sv` 已按命令自动读取所需 Operator/Tensor/Quant/Segment 描述符；网络 A 全部 1,078 条 DMA command 经该前端后的请求与软件 reference 逐 bit 一致。AGU 不再把地址公式展开为单周期并行乘法，而是复用一个 16-cycle radix-4 shift/add 乘法器；这保留了接口语义，并把控制路径 DSP 占用从早期综合的 138 个降为 0。

## 2. 任务地址空间

描述符里的地址都是 section-relative offset。任务提交时由 CSR/任务上下文提供独立的 64-bit 物理基址：

| Section | 基址 | offset 来源 |
|---|---|---|
| mutable activation | `activation_base` | Tensor `base_offset` |
| O8I8 weights | `weight_base` | constant Tensor `base_offset` |
| INT32 bias | `bias_base` | constant Tensor `base_offset` |
| quant parameters | `quant_param_base` | Quant descriptor `param_offset` |

Tensor、Operator、Quant descriptor 和 Segment record 也分别有 table base，供 descriptor cache 使用。这样同一个编译包可以由驱动重定位，不需要改写每条命令或描述符。

## 3. Descriptor Cache

当前原型是 4-line direct-mapped read-only cache：key 为 `{kind,index}`，一条 line 保存最多 64 B。支持：

| kind | record 大小 |
|---|---:|
| Tensor | 64 B |
| Operator | 64 B |
| Quant | 32 B |
| Segment | 8 B |

前端一次只允许一个 miss outstanding。hit 直接返回缓存记录；miss 通过与 AXI 无关的 `memory_request/response` ready-valid 接口取回记录；soft reset 使全部 line 失效，已经握手的后端 read 仍会接收并丢弃 response 后再回到 idle。`0xffff` 索引和后端错误不会填充 cache。

4-line 只是正确性原型，不是性能冻结值。后续应根据真实命令流的 hit/miss 统计决定 line 数和是否拆成 Operator/Tensor 两组，不能凭直觉占用更多 BRAM。

## 4. 统一内部 DMA 请求

`npu_dma_request_t` 描述：

```text
for z in [0, z_count):
  for y in [0, y_count):
    copy x_bytes from
      external_address + z*external_z_stride + y*external_y_stride
    to/from
      scratchpad_offset + z*scratchpad_z_stride + y*scratchpad_y_stride
```

请求还携带 load/store、section、A/W/O scratchpad、bank、completion event 和激活 tile 清零信息。下一阶段的 burst engine 只处理这个标准格式，不再解析卷积、segment 或 Tensor layout。

## 5. 地址规则

### 5.1 普通激活输入

本地 receptive field 为：

```text
local_h = (tile_h-1)*stride_h + dilation_h*(kh-1) + 1
local_w = (tile_w-1)*stride_w + dilation_w*(kw-1) + 1
local_c = align8(input_channel_count)
```

DMA 先把 `local_h*local_w*local_c*2` bytes 清零，再只从 DDR 搬有效空间和逻辑通道。这样卷积 padding 和 NHWC8 尾通道都由明确的零值产生，不要求 DDR 的 padded lane 已初始化。

### 5.2 Segmented/concat 输入

每个 segment 是独立的 3D 请求，`x_bytes=segment.channel_count*2`，外部像素 stride 来自实际源 Tensor，本地像素 stride 来自 concat 后逻辑通道数，并在 `dst_channel*2` 处写入。只有 segment 0 设置 `clear_before`，否则后一个 segment 会清掉前一个 segment。

这条规则是 decoder 正确性的关键：不能把两个源 Tensor 各自整块连续复制到 A bank，因为那会得到“按 Tensor 分块”，而 MAC 需要每个像素内按通道拼接。

### 5.3 权重、bias 和 quant

O8I8 权重按 `tile_origin_cout/8` 选择连续输出 block。W bank 固定为：

```text
weight            @ 0
bias              @ align64(weight_bytes)
convolution quant @ align64(bias_offset + bias_bytes)
vector quant      @ align64(conv_quant_offset + conv_quant_bytes)
```

`DMA_LOAD` 的 `imm[10]` 在 DMA 上下文中定义为 `DMA_VECTOR_QUANT`；在 `CONV2D` 上下文中仍是 accumulator bank。该位消除了依靠 `param_count==1` 猜测 quant 用途的歧义，因而也允许未来出现单输出通道卷积。

### 5.4 输出回写

O bank 仍按 `align8(tile_cout)` 保存，但 DDR store 的 `x_bytes` 只包含 `tile_cout` 个有效通道。以当前最终 4 通道输出为例，每个像素写 8 B、外部像素 stride 为 16 B，不覆盖 padded lane。

## 6. 边界检查

AGU 在发出请求前检查：

- opcode、必填 descriptor 和非零 tile/kernel/stride/dilation；
- command 与 Operator 的 weight/bias/input 索引契约；
- dtype/layout、segment 索引、目标通道范围；
- Tensor allocation 的最后访问 byte；
- A/W/O bank 的最后访问 byte和激活清零范围；
- section base 加 offset 的 64-bit 地址回绕。

失败时不给 burst engine `request_valid`，并产生 `error_reason`。系统集成时由 DMA front-end 把原因映射为统一 execution error，并附上 command PC/tag 上报 Command Processor。

## 7. 网络 A 证据

`scripts/30_plan_npu_dma.py` 对 1,869 条命令中的全部 DMA 做了展开：

| 项目 | 数值 |
|---|---:|
| DMA 请求 | 1,078（830 load + 248 store） |
| activation / weight / bias / quant | 302 / 248 / 248 / 280 |
| DMA payload | 3,020,480 B |
| upsample 单独估算流量 | 491,520 B |
| 总 DDR 流量 | 3,512,000 B |
| A bank 峰值 | 57,600 / 65,536 B |
| W bank 峰值 | 16,320 / 32,768 B |
| O bank 峰值 | 4,096 / 16,384 B |
| 地址/容量检查 | 全部通过，RTL 与 1,078 条 reference 逐 bit 一致 |

生成的 `dma_plan.json` 是逐命令可审计 reference；`dma_analysis.json` 是摘要。当前结果证明地址语义自洽。多周期 AGU 的 activation 最坏约 370 cycle/request；即使对全部 1,078 个请求都按此上界计，也小于 0.4M cycle，仍在 4M task budget 内。

## 8. 后续集成

上述 `npu_dma_request_t` 已由 `npu_dma_engine.sv` 消费并实现 scratchpad clear、z/y 展开、256-beat/4 KiB 拆分、尾部 strobe、completion event 和 AXI error 收尾，详见 `33_axi_dma_engine_microarchitecture.md`。

上述模块已经由 `npu_dma_subsystem.sv` 与 AXI Engine、A/W/O Scratchpad 接通，形成命令到 event/error 的 DMA 垂直链路，详见 `34_dma_subsystem_scratchpad_microarchitecture.md`。下一步是接入 Command Processor，并用 Tensor MAC 小原型确定最终计算读端口宽度。
