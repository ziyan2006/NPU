# DMA 子系统与 Scratchpad 微架构规格

文档版本：`0.1-draft`

状态：P4 可综合集成原型；已通过 Vivado 2026.1 OOC 资源与 100 MHz 时序验证，尚未经过真实 Zynq HP 端口和 post-route 验证

## 1. 已打通的数据路径

当前 DMA 垂直链路为：

```text
DMA command + PC
        |
        v
Descriptor fetch sequencer -> 4-line Descriptor Cache
        |
        v
DMA AGU -> npu_dma_request_t -> AXI DMA Engine
                                    |
                          +---------+---------+
                          v                   v
                    64-bit AXI4          A/W/O Scratchpad
```

`npu_dma_frontend.sv`、`npu_dma_engine.sv` 和 `npu_scratchpad.sv` 由
`npu_dma_subsystem.sv` 组合。Command Processor 发来一条 `DMA_LOAD` 或
`DMA_STORE` 后，子系统保留该命令的 PC/tag，直到请求完成或报错。

描述符读取目前使用独立的 request/response memory port，Tensor payload 使用
AXI4 master port。SoC 集成时可把描述符端口接独立小型 AXI reader，也可与 payload
AXI 经过仲裁器共享 HP 端口；本版尚未冻结该选择。

## 2. 描述符取数顺序

前端最多保存一条等待进入 Engine 的命令，按需依次读取：

```text
Operator -> source Tensor -> destination Tensor -> Quant -> Segment
```

- 相同 source/destination Tensor 只读取一次；
- `src0_td/dst_td == 0xffff` 时跳过对应 Tensor；
- Quant DMA 不读取 Tensor；
- segmented activation 从目标 Tensor 的 `segment_offset/8 + segment index`
  计算 Segment record 索引；
- cache miss/error、非法 segment 索引和 AGU 拒绝分别返回独立的 front-end
  reason，AGU 原始 reason 保存在 detail 字段。

前端拿到完整描述符后保持 `npu_dma_request_t`，直到 Engine ready。Engine 忙时，
前端可以预取下一条命令的描述符，但不会越过它再接收第三条命令。

## 3. Scratchpad 组织

当前原型实现 6 个逻辑 bank，物理槽均为 64 bit：

| bank | 数量 | 单 bank | 64-bit word | 理想 BRAM36 数量 |
|---|---:|---:|---:|---:|
| `A0/A1` | 2 | 64 KiB | 8,192 | 16/个 |
| `W0/W1` | 2 | 32 KiB | 4,096 | 8/个 |
| `O0/O1` | 2 | 16 KiB | 2,048 | 4/个 |
| **合计** | **6** | **224 KiB** | - | **56** |

每个 bank 使用 true-dual-port 风格：port A 接 DMA，port B 接 accelerator。两侧均支持
同步一周期 read response、ready/valid back-pressure，以及逐 byte write strobe。
同一物理端口读写同时请求时写优先；同周期跨端口访问相同 word 且至少一侧写入时，
DMA 优先并暂停 accelerator，产生 `collision_stall`。不同 bank 可并行访问。

Vivado 2026.1 在参考 `xc7z020clg400-1` 上把 6 个 bank 全部识别为 true dual-port
RAM，并实际映射为 56 个 RAMB36E1。该结果只适用于当前 64-bit accelerator 端口；
MAC 宽端口结构仍可能改变最终 BRAM 数。

## 4. 计算端口边界

当前 accelerator 端口为统一 64-bit read/write 接口，足够完成 DMA 数据落地、bank
隔离、Vector/debug 接入和端到端控制验证，但**不足以直接喂满 8x8 MAC**。64-lane
阵列每周期需要更宽的 activation/weight 供数，P4 MAC 设计必须选择以下一种方案并
通过综合比较：

1. A/W bank 进一步按 lane striping，并导出多组窄读口；
2. 从 BRAM 预取到寄存器 row buffer，再由 row buffer 每周期广播 8 个 activation
   和 64 个 weight；
3. 降低阵列发射率，以较少端口换取更低 BRAM/LUT 压力。

因此本模块关闭了“DMA 无处落地”和“bank 选择/冲突语义不明确”的风险，没有关闭
MAC 带宽、BRAM 复制和 200 MHz 时序风险。

## 5. 完成、错误与复位

- Engine 成功完成时把命令的 event mask 输出给 Command Processor；
- front-end error 来源编号为 1，Engine error 来源编号为 2；
- 错误同时返回 reason/detail、原始 command PC 和 tag；
- 子系统 `busy` 是 front-end 与 Engine busy 的或；
- `soft_reset` 一旦有效，子系统立即禁止接收新命令；若已有事务则继续排空，整体
  `busy=0` 后才把 reset 传入内部状态机。因此外层应保持或在 idle 后重发 reset，不能
  用 soft reset 取消已经越过 AXI ready/valid 边界的事务。

## 6. 当前验证

Icarus 回归已覆盖：

- 网络 A 全部 1,078 条 DMA command 经真实 fetch sequencer/cache/AGU 后，与软件
  `dma_plan.json` 的请求逐 bit 一致；
- A/W/O 六个 bank 的路由、byte strobe、同步读和 response back-pressure；
- 不同 bank 并行访问、同 word 冲突时 DMA 优先且 accelerator 后续恢复；
- 一条端到端 activation load 从命令和描述符开始，经 AXI read 写入 A0，再从
  accelerator port 读回；
- AXI read error 返回正确的 source/reason/PC/tag。

尚未覆盖真实 HP interconnect、描述符/payload AXI 仲裁、MAC 宽读端口、CDC 和
post-route Fmax。

## 7. Vivado OOC 结果

| 项目 | `npu_scratchpad` | `npu_dma_subsystem` |
|---|---:|---:|
| Slice LUT | 703 | 7,546 |
| Slice Register | 12 | 7,144 |
| RAMB36E1 | 56 | 56 |
| DSP48E1 | 0 | 0 |
| 100 MHz WNS | 未单独重跑 | `+1.888 ns` |
| 200 MHz WNS | `+0.041 ns` | `-3.112 ns` |

这是综合后 OOC 结果：没有最终 clock source、input/output delay、PS interconnect 和
布局布线。Vivado 同时提示 BRAM 输出寄存器未合并，以及 8Kx64 activation bank 的
byte-write 结构未采用原生 byte-wide enable；两项都保留为 MAC/row-buffer 阶段的
时序和资源优化输入。实际板卡型号未确认，`xc7z020clg400-1` 只是临时参考 part。

## 8. 下一步

实现 8x8 Tensor MAC 的 loop controller 与位精确 datapath 小原型，先验证单个
`CONV2D` tile 的 A/W 读地址、64 个 INT8xINT12 乘法、INT32 累加、bias/requant/RNE
和 O bank 写回，再决定最终 Scratchpad lane striping/row-buffer 组织。
