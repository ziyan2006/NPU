# STEM NPU

本目录是 XC7Z020 轻量通用 CNN NPU 的硬件设计入口。当前处于 P1 需求基线与 P2/P3 架构探索阶段，尚未冻结 ISA 或开始完整 RTL。

建议按以下顺序阅读：

1. `spec/00_spec_index.md`：设计目标、系统边界、专业开发流程和文档索引；
2. `spec/10_system_requirements.md`：带编号、状态和验证方法的需求基线；
3. `spec/11_workload_profile.md`：当前模型逐层画像、存储压力和第二验证网络候选；
4. `spec/20_programming_model_isa.md`：命令流、描述符、算子和数值语义；
5. `spec/21_control_registers.md`：通用 NPU AXI4-Lite 控制接口草案；
6. `spec/30_microarchitecture_budget.md`：MAC、DMA、BRAM、周期与资源预算；
7. `spec/40_verification_plan.md`：位精确模型、RTL、实现与上板验证；
8. `spec/90_decision_log.md`：已接受方向、待批准提案和开放问题。

`spec/01_npu_architecture.md` 与 `spec/02_register_map.md` 保留为早期固定人声消除加速器基线，不再作为通用 NPU 顶层规格。

`generated/bott2_mir1k_v1/` 是由当前候选检查点生成的算法参考包。它的 OIHW 权重和 float scale 还需经过布局重排与整数 requant 编译，不能由 RTL 直接消费。

重新导出硬件包：

```powershell
python scripts/25_export_npu_package.py
```

最终 RTL/HLS 不读取 PyTorch/ONNX/JSON，而读取编译后的二进制任务包。满足 v1 ISA 和资源边界的不同静态 CNN 应能在不重新综合 bitstream 的情况下切换。
