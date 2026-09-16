# Vivado 实现证据

`vivado_2026_1/` 保存由 `scripts/32_synthesize_npu_rtl.tcl` 生成的 OOC 综合报告。
当前使用临时参考 part `xc7z020clg400-1`；实际板卡型号确认前，这些报告不能用来
冻结封装、speed grade、时钟管脚或最终 Fmax。`npu_top_impl_fast/` 是完整 PL core
的 OOC post-route 报告，使用 10 ns 时钟、0.2 ns uncertainty 和 2 ns 接口预算。

每个 top 目录包含利用率、目标 200 MHz 时序、100 MHz 基线时序、methodology、
timing check 和机器可读摘要。OOC 报告不是 post-route 结论，SoC 顶层约束和 PS AXI
互连接入后必须重新综合、实现并检查 CDC。

当前保存 `npu_dma_subsystem`、`npu_scratchpad`、`npu_tensor_mac_8x8`、
`npu_conv2d_controller`、`npu_conv2d_pipeline`、`npu_requant_post`、`npu_vec_add`
和 `npu_upsample2x` 八个 top。Controller 报告包含 loop 与 MAC，但不包含
Scratchpad、bias、requant 或 post-op；它在 100 MHz 闭合，在 200 MHz 尚未闭合。
当前 `npu_dma_subsystem` 报告已包含 Scratchpad、CONV2D controller、MAC 和 O 写回
FIFO 及 post：19,035 LUT、13,856 FF、58 RAMB36、68 DSP，100 MHz 综合后 OOC WNS
`+0.197 ns`。`npu_scratchpad` 的 128-bit O 行版本为 5,668 LUT、1,244 FF、
56 RAMB36，100 MHz WNS `+4.568 ns`。
`npu_requant_post` 为 2,702 LUT、1,639 FF、2 RAMB36、4 DSP，100 MHz WNS
`+0.197 ns`。`npu_conv2d_pipeline` 包含 W bank 参数预取、MAC 和 post，为
6,398 LUT、5,521 FF、2 RAMB36、68 DSP，100 MHz WNS `+0.197 ns`。
`npu_vec_add` 为 2,716 LUT、1,525 FF、4 DSP，100 MHz WNS `+1.426 ns`；
`npu_upsample2x` 为 1,626 LUT、798 FF、1 RAMB36，100 MHz WNS `+0.425 ns`。

完整 `npu_top` OOC post-route 使用 25,645 Slice LUT、24,894 FF、61 RAMB36、
72 DSP；WNS `+0.225 ns`、TNS `0`、WHS `+0.007 ns`，全部网络布通且 critical
DRC 为 0。该结果关闭 PL core 的 100 MHz 实现风险，但不替代具体 Zynq PS block
design 的最终实现签核。
