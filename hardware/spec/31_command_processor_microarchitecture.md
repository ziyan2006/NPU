# Command Processor 微架构规格

文档版本：`0.1-draft`

状态：P4 首个可综合 RTL 原型；当前网络命令流已通过，接口尚未冻结

## 1. 模块边界

`hardware/rtl/npu_command_processor.sv` 负责顺序控制，不直接访问 DDR、描述符或 scratchpad：

```text
Command fetch ready/valid
          │
          ▼
  validate + PC + FSM ─────► DMA command port
          │                ► Compute command port
          │                ► Vector command port
          ▼
 event scoreboard ◄──────── async completion event
 WAIT / END / watchdog ◄──── unit done / idle / error
```

Command fetch 与 AXI command cache 留给后续模块。当前接口接收已经按 little-endian 组装好的 `command_bits_i[127:0]`，其中 bits `[7:0]` 是 opcode。

## 2. 接口分组

| 分组 | 信号 | 语义 |
|---|---|---|
| 任务控制 | `start_i`, `soft_reset_i` | 从 PC=0 开始；soft reset 清任务与首错状态 |
| 取指 | `command_valid_i/ready_o/bits_i` | 标准 ready/valid，握手后取指源前进 16 B |
| DMA | `dma_command_valid_o/ready_i/bits_o` | `DMA_LOAD/STORE` 完整命令旁路到 DMA |
| Compute | `compute_command_valid_o/ready_i/bits_o` | `CONV2D` 旁路到 MAC 控制器 |
| Vector | `vector_command_valid_o/ready_i/bits_o` | `VEC_ADD/ACT/REQUANT/UPSAMPLE2X` 旁路到 Vector/Post |
| 完成 | `event_set_i[7:0]`, `vector_sync_done_i`, `units_idle_i` | 异步 event、同步 Vector 完成、END 全局 idle 条件 |
| 异常 | `execution_error_*` | 执行单元上报 code/PC/tag，Command Processor 锁存首错 |
| 状态 | `busy/done/irq/error`, PC、retired、cycles、events | 连接后续 CSR/perf 模块 |

执行单元只有在 `*_command_valid_o && *_command_ready_i` 时接收命令。它们必须保存完成所需的 `imm` event mask、PC/tag 或由上层同时提供等价上下文；不能依赖随后变化的 command input。

## 3. 状态机

| 状态 | 行为 | 退出条件 |
|---|---|---|
| `IDLE` | 不接收命令，保留上次计数 | `start_i` |
| `RUN` | 校验并分派一条命令 | dispatch、WAIT、END 或错误 |
| `WAIT_EVENT` | 等待 event mask 全部置位 | mask 满足或 watchdog |
| `WAIT_VECTOR` | 等待同步 Vector/Post 指令完成 | `vector_sync_done_i` 或 watchdog |
| `WAIT_END` | 等待所有执行单元 idle | `units_idle_i` 或 watchdog |
| `ERROR` | 锁存首错，不再接收命令 | soft reset |

异步 `DMA_LOAD/STORE/CONV2D` 在执行端接受后即 retired；其数据依赖由后续 `WAIT` 保证。同步 Vector/Post 命令在 `vector_sync_done_i` 后 retired。`END` 在所有执行单元 idle 后 retired，并产生单周期 done；带 `IRQ` flag 时同时产生单周期 IRQ。

## 4. Event scoreboard

`event_state_o` 是 8-bit sticky 状态。执行端用 `event_set_i` 提交一个或多个完成事件；带 event mask 的异步命令握手时清相同 event，以免下一轮误用旧完成状态：

```text
event_next = (event_state OR event_set_i) AND NOT issued_event_clear
```

同周期 set 与复用 clear 指向同一 bit 时 clear 优先。执行端不得让新命令在握手同周期完成；至少一个周期后才能 set event。`WAIT` 不清事件。

## 5. 校验与首错

当前 RTL 接受网络 A 所需 opcode，以及预留的同步 `ACT/REQUANT` 路由；P1 opcode 返回 `ILLEGAL_OPCODE`。校验分为：

| code | 条件 |
|---:|---|
| `0x0001 ILLEGAL_OPCODE` | 未实现/保留 opcode |
| `0x0002 ILLEGAL_FLAGS` | opcode 已知但 flag 组合不合法 |
| `0x0003 ILLEGAL_FIELDS` | 描述符索引、保留位或必填字段不合法 |
| `0x0004 EXECUTION_ERROR` | 执行端未提供更具体 code 时的兜底 |
| `0x0005 WATCHDOG` | 从 start 起忙周期达到非零 limit 仍未结束 |

首个 fatal error 锁存 `error_code_o/error_pc_o/error_inst_tag_o` 并进入 `ERROR`。后续错误不能覆盖，soft reset 才清除。`watchdog_limit_i=0` 禁用 watchdog；非零值表示允许的最大任务忙周期，超时计数停在 limit。

## 6. 当前验证证据

`scripts/_test_npu_rtl.py` 使用 Icarus Verilog 进行两轮测试：

1. directed test：back-pressure、event set/clear、WAIT、同步 Vector、END/IRQ、soft reset、非法 opcode/flag/field、watchdog 和执行端首错；
2. stream test：把 `tile_commands.bin` 的 1,869 条真实命令全部输入 RTL，使用有延迟的 DMA/compute/vector stub，要求无非法译码、无死锁、全部 retired、最终 PC 与命令字节数一致。

这证明控制语义和现有编译器一致，不证明 Vivado 综合资源、200 MHz 时序或执行单元数值正确。进入下一步前应接入 command fetch/descriptor cache 接口，随后实现 DMA request/AGU 或 MAC loop controller 的其中一个垂直切片。
