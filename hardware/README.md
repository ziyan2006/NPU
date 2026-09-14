# STEM NPU

本目录是 XC7Z020 轻量通用 CNN NPU 的硬件设计入口。当前已进入 P4 微架构原型，ISA 仍未冻结，已有 Command Processor、Descriptor Fetch/Cache、DMA AGU、AXI DMA Engine、A/W/O Scratchpad 和集成 DMA Subsystem 可综合模块。

建议按以下顺序阅读：

1. `spec/00_spec_index.md`：设计目标、系统边界、专业开发流程和文档索引；
2. `spec/10_system_requirements.md`：带编号、状态和验证方法的需求基线；
3. `spec/11_workload_profile.md`：当前模型逐层画像、存储压力和第二验证网络候选；
4. `spec/20_programming_model_isa.md`：命令流、描述符、算子和数值语义；
5. `spec/21_control_registers.md`：通用 NPU AXI4-Lite 控制接口草案；
6. `spec/30_microarchitecture_budget.md`：MAC、DMA、BRAM、周期与资源预算；
7. `spec/31_command_processor_microarchitecture.md`：命令控制核心的接口、状态机和异常；
8. `spec/32_dma_frontend_microarchitecture.md`：描述符缓存、三维 DMA 请求和地址规则；
9. `spec/33_axi_dma_engine_microarchitecture.md`：AXI burst、4 KiB 拆分和错误行为；
10. `spec/34_dma_subsystem_scratchpad_microarchitecture.md`：DMA 集成、BRAM bank 和计算端口边界；
11. `spec/40_verification_plan.md`：位精确模型、RTL、实现与上板验证；
12. `spec/50_p2_executable_spec.md`：当前模型的可执行指令、整数语义和周期结果；
13. `spec/90_decision_log.md`：已接受方向、待批准提案和开放问题。

RTL 开发者从 `rtl/README.md` 和 `spec/22_operator_instruction_contract.md` 开始；前者说明如何引用生成的 SystemVerilog package，后者给出算子到指令序列及逐指令执行契约。

`spec/01_npu_architecture.md` 与 `spec/02_register_map.md` 保留为早期固定人声消除加速器基线，不再作为通用 NPU 顶层规格。

`generated/bott2_mir1k_v1/` 是由当前候选检查点生成的算法参考包。它的 OIHW 权重和 float scale 还需经过布局重排与整数 requant 编译，不能由 RTL 直接消费。

`generated/bott2_mir1k_v1_program/` 是第一版通用 ISA 架构包，已经包含层级与逐 tile 的 128-bit 命令、固定大小描述符、O8I8 权重、整数 requant 参数、4096 项 INT12 tanh LUT、bank/周期分析，以及全部 DMA command 的地址计划。它尚未封装为带 header/CRC 的任务镜像，也未经过完整数据通路 RTL 执行，因此不能直接上板。

重新导出硬件包：

```powershell
python scripts/25_export_npu_package.py
```

最终 RTL/HLS 不读取 PyTorch/ONNX/JSON，而读取编译后的二进制任务包。满足 v1 ISA 和资源边界的不同静态 CNN 应能在不重新综合 bitstream 的情况下切换。

生成并验证架构包：

```powershell
python scripts/26_compile_npu_program.py
python scripts/28_schedule_npu_tiles.py
python scripts/30_plan_npu_dma.py
python scripts/31_plan_axi_dma.py
python scripts/29_generate_npu_isa_headers.py --check
python scripts/_test_npu_isa.py
python scripts/_test_npu_tile_schedule.py
python scripts/_test_npu_dma.py
python scripts/_test_npu_axi_dma.py
python scripts/_test_npu_isa_headers.py
python scripts/_test_npu_rtl.py
```

Vivado 2026.1 OOC 综合当前 DMA 子系统：

```powershell
vivado -mode batch -nojournal -nolog `
  -source scripts/32_synthesize_npu_rtl.tcl `
  -tclargs npu_dma_subsystem xc7z020clg400-1 5.000
```

参考器件上的当前结果为 7,546 Slice LUT、7,144 FF、56 RAMB36、0 DSP；
100 MHz WNS `+1.888 ns`，200 MHz WNS `-3.112 ns`。这是综合后 OOC 证据，
不是最终板卡的 post-route 结论；详细报告见 `reports/vivado_2026_1/`。
