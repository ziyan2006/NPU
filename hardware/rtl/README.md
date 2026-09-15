# RTL ISA 交付入口

状态：`P3 draft`，可用于 RTL 原型和联调，不代表 ISA 已冻结。

当前 P4 RTL 包含：

- `npu_command_processor.sv`：命令校验、分派、PC、event、WAIT/END、watchdog 和首错锁存；
- `npu_axi_block_reader.sv` / `npu_memory_arbiter4.sv`：为 task header、命令和描述符提供共享的单 outstanding AXI block-read 边界；
- `npu_task_loader.sv`：校验 256-byte task header、生成各 section 基址并从真实镜像预载 4096 项 tanh LUT；
- `npu_command_fetch.sv`：有界顺序取出 128-bit 命令，并检查 END 必须位于命令区末尾；
- `npu_descriptor_cache.sv`：四类固定描述符的 4-line read-only cache；
- `npu_execution_frontend.sv`：为 CONV2D/VEC_ADD/UPSAMPLE2X 取 descriptor，驱动异步/同步执行完成协议；
- `npu_u32_mul_iter.sv`：供控制路径复用的 16-cycle radix-4 无 DSP 乘法器；
- `npu_dma_agu.sv`：把 DMA command/descriptor 转换为带独立 DDR/SP stride 的三维请求，并检查边界；
- `npu_dma_frontend.sv`：按命令取回所需描述符并驱动 AGU；
- `npu_dma_engine.sv`：执行清零和 AXI4 read/write burst，处理 4 KiB、窄尾部、back-pressure 与错误；
- `npu_scratchpad_bank.sv` / `npu_scratchpad.sv`：A/W/O 六 bank 的 lane-striped simple-dual-port BRAM、独立 128-bit A/512-bit W 读口、128-bit O 写口与冲突仲裁；
- `npu_dma_subsystem.sv`：把 fetch/cache/AGU/Engine/Scratchpad/CONV2D/post 接成数据链路，并用 2-entry FIFO 隔离 O 写回反压；
- `npu_tensor_mac_8x8.sv`：64-lane signed INT16xINT8 DSP 阵列、三级加法树、INT32 累加和结果反压；
- `npu_requant_post.sv`：按 lane 复用的 bias、Q31 requant、signed RNE、clamp、ReLU/LeakyReLU/tanh LUT；
- `npu_conv2d_controller.sv`：解析 Operator descriptor，生成真实 tile 的 A/W 地址、首尾/尾 lane，并驱动 8x8 MAC；
- `npu_conv2d_pipeline.sv`：从 W bank 预取 bias/quant 参数，串接 CONV2D、requant/post 并生成 128-bit O-bank 结果；
- `npu_vec_add.sv`：读取 O/A/W bank，执行 residual scalar requant、INT12 饱和相加并原位写回 O；
- `npu_upsample2x.sv`：使用 1 KiB BRAM 行缓冲在 DDR Tensor 间执行最近邻 2×，带完整 AXI 错误检查；
- `include/npu_dma_pkg.sv`：DMA 内部接口类型。

逐接口语义见 `../spec/31_command_processor_microarchitecture.md`、`../spec/32_dma_frontend_microarchitecture.md`、`../spec/33_axi_dma_engine_microarchitecture.md`、`../spec/34_dma_subsystem_scratchpad_microarchitecture.md` 和 `../spec/35_tensor_mac_microarchitecture.md`。

从仓库根目录读取 `npu_rtl.f` 可获得当前可综合源文件顺序；package 必须先于使用它的模块编译。

硬件侧首先引用 `include/npu_isa_pkg.sv`，DMA 模块随后引用 `include/npu_dma_pkg.sv`。ISA package 由 `scripts/29_generate_npu_isa_headers.py` 从 `scripts/npu_isa.py` 生成，包含：

- opcode、flag、dtype、layout、post-op 和 event 编码；
- 128-bit command packed struct；
- Tensor、Operator、Quant、Segment 描述符 packed struct；
- tile immediate 的 event 和 bank 位定义。

译码器的基本用法：

```systemverilog
import npu_isa_pkg::*;

npu_command_t command;
assign command = npu_command_t'(command_bits);

always_comb begin
  case (npu_opcode_e'(command.opcode))
    NPU_OP_DMA_LOAD:  /* issue read DMA */;
    NPU_OP_CONV2D:    /* issue Tensor MAC */;
    NPU_OP_VEC_ADD:   /* issue residual add */;
    NPU_OP_DMA_STORE: /* issue write DMA */;
    default:          /* reject unsupported opcode */;
  endcase
end
```

命令存储为 little-endian，任务镜像 byte 0 是 `opcode`。Command fetch 把 16 个连续 byte 组装为 `command_bits[127:0]` 时，必须保证 byte 0 对应 bits `[7:0]`；若 AXI 数据拼接方向不同，应在 fetch 边界处理，不能在各执行单元分别交换。

算子到指令序列、字段约束和完成条件见 `../spec/22_operator_instruction_contract.md`。PS/驱动侧使用 `software/include/npu_isa.h`，不能复制一套独立常量。

重新生成并检查：

```powershell
python scripts/29_generate_npu_isa_headers.py
python scripts/29_generate_npu_isa_headers.py --check
python scripts/_test_npu_isa_headers.py
python scripts/_test_npu_rtl.py
```

修改编码时只编辑 `scripts/npu_isa.py` 和生成器；直接编辑生成的 `.sv`/`.h` 会被 `--check` 回归拒绝。
