# Vivado 实现证据

`vivado_2026_1/` 保存由 `scripts/32_synthesize_npu_rtl.tcl` 生成的 OOC 综合报告。
当前使用临时参考 part `xc7z020clg400-1`；实际板卡型号确认前，这些报告不能用来
冻结封装、speed grade、时钟管脚或最终 Fmax。

每个 top 目录包含利用率、目标 200 MHz 时序、100 MHz 基线时序、methodology、
timing check 和机器可读摘要。OOC 报告不是 post-route 结论，SoC 顶层约束和 PS AXI
互连接入后必须重新综合、实现并检查 CDC。

当前保存 `npu_dma_subsystem`、`npu_scratchpad` 和 `npu_tensor_mac_8x8` 三个 top。
MAC 报告只覆盖算术 slice，不包含 row buffer、loop controller 或 requant/post。
