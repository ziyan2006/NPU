# DMA 子系统与 Scratchpad 微架构规格

文档版本：`0.2-draft`

状态：P4 可综合集成原型；lane-striped 计算宽口已通过 Vivado 2026.1 OOC
资源与 200 MHz 时序验证，尚未经过完整计算集成、真实 Zynq HP 端口和 post-route 验证

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

当前原型实现 6 个逻辑 bank。存储由 64-bit lane 组成，DMA 每次访问一条 lane，
计算端一次访问整行：

| bank | 数量 | 单 bank | lane 组织 | 计算行宽 | BRAM36/个 |
|---|---:|---:|---:|---:|---:|
| `A0/A1` | 2 | 64 KiB | `2 × 4096 × 64` | 128 bit | 16 |
| `W0/W1` | 2 | 32 KiB | `8 × 512 × 64` | 512 bit | 8 |
| `O0/O1` | 2 | 16 KiB | `2 × 1024 × 64` | 128 bit | 4 |
| **合计** | **6** | **224 KiB** | - | - | **56** |

每条 lane 使用 simple-dual-port BRAM：一个物理方向写、另一个物理方向读。仲裁器允许
主路径中的 DMA write 与 compute read、compute write 与 DMA read 并行；同一 bank
的双写或双读由 DMA 优先，同一行的交叉读写也暂停 compute 并产生
`collision_stall`。不同 bank 可并行访问。两侧均保留逐 byte write strobe。

计算侧使用统一 512-bit 接口，访问 A/O 时只有低 128 bit 有效。计算地址必须按行宽
对齐：A/O 为 16 byte，W 为 64 byte；lane 0 位于总线最低位并对应最低字节地址。
读请求经过 bank BRAM 输出与顶层 response register 两级，稳态在 consumer ready 时
仍可每拍接收一个请求并返回一个响应。顶层 pending selector 明确记录响应 bank，
避免通过六 bank valid OR 推断响应归属。

Vivado 2026.1 在参考 `xc7z020clg400-1` 上把 24 条 lane 全部识别为
simple-dual-port block RAM，并实际映射为 56 个 RAMB36E1。W bank 的 16 条
`512 × 64` lane 各使用一个 RAMB36；宽口没有复制 tensor 容量。

## 4. 计算端口边界

lane-striped 端口每拍提供 128-bit activation 和 512-bit weight，已经满足 8x8 MAC
单 token 的供数宽度。该结构关闭了 BRAM 复制与 Scratchpad 单体 200 MHz 风险；
CONV2D 控制器仍需证明 A/W 双请求的调度、地址对应关系和整条计算路径时序。

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
- A/W/O 六个 bank 的路由、byte strobe、两级同步读和 response back-pressure；
- A/O 128-bit 与 W 512-bit 行映射、W 全行读写及连续周期计算读吞吐；
- 不同 bank 并行访问、同 word 冲突时 DMA 优先且 accelerator 后续恢复；
- 一条端到端 activation load 从命令和描述符开始，经 AXI read 写入 A0，再从
  accelerator port 读回；
- AXI read error 返回正确的 source/reason/PC/tag。

尚未覆盖真实 HP interconnect、描述符/payload AXI 仲裁、MAC 与 A/W 双宽口的
完整集成、CDC 和 post-route Fmax。

## 7. Vivado OOC 结果

| 项目 | `npu_scratchpad` | `npu_dma_subsystem`（旧 64-bit 口基线） |
|---|---:|---:|
| Slice LUT | 3,147 | 7,546 |
| Slice Register | 596 | 7,144 |
| RAMB36E1 | 56 | 56 |
| DSP48E1 | 0 | 0 |
| 100 MHz WNS | `+5.360 ns` | `+1.888 ns` |
| 200 MHz WNS | `+0.360 ns` | `-3.112 ns` |

这是综合后 OOC 结果：没有最终 clock source、input/output delay、PS interconnect 和
布局布线。Vivado 仍提示部分 BRAM 输出寄存器未合并，完整集成后需结合布局继续检查。
实际板卡型号未确认，`xc7z020clg400-1` 只是临时参考 part。

## 8. 下一步

实现 8x8 Tensor MAC 的 CONV2D loop controller，先验证单个真实 `1×1` tile 的
A/W 宽读地址、lane mask、`first/last` 和 INT32 结果，再扩展 `1×3/3×3`、stride、
padding、bias/requant/RNE 与 O bank 写回。完整顶层 place/route 前不把 OOC WNS
当作最终 Fmax。
