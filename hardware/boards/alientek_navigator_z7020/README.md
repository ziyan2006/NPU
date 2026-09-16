# 正点原子领航者 ZYNQ-7020 上板配置

状态：`provisional`。这个目录把公开资料中可以交叉确认的配置做成可复现的候选，
但在看到核心板和底板丝印、并拿到对应版本的官方 PS7 preset 前，不把它称为发布配置。

## 已确认度较高的硬件

| 项目 | 候选值 | 用途 |
|---|---|---|
| Zynq | `XC7Z020CLG400-2` | Vivado part |
| DDR3 | 2 × `NT5CC256M16`，共 1 GiB、32-bit | task/权重/激活 |
| Vivado DDR 条目 | `MT41J256M16 RE-125` | 官方教程使用的兼容条目 |
| PS 晶振 | 33.333333 MHz | PS7 时钟配置 |
| UART | UART0，MIO14/15，115200 | 首次启动日志 |
| PL 晶振 | 50 MHz | 当前 NPU 不使用 |
| QSPI / eMMC | W25Q256 32 MiB / KLM8G1GETF 8 GiB | 首测阶段禁用 |

当前 NPU 由 PS 的 `FCLK_CLK0` 提供 100 MHz，因此不需要给 PL 50 MHz 晶振或底板
LED 添加 XDC。DDR 和 MIO 是 Zynq 的专用 PS 管脚，也不通过普通 PL XDC 分配。

`minimal_jtag_profile.tcl` 只打开 DDR、UART0 和 NPU 所需的 PS/PL 接口，用于拿板后的
第一轮 JTAG 下载。它采用公开教程的通用 DDR part 条目，但没有经过特定 PCB 版本的
DDR 走线延时校准，不能用于量产或写入启动 Flash。

## 离线构建

```powershell
vivado -mode batch -nolog -nojournal `
  -source scripts/39_create_reference_zynq_soc.tcl -- `
  xc7z020clg400-2 hardware/build/ip_repo `
  hardware/build/navigator_z7020_candidate `
  hardware/boards/alientek_navigator_z7020/minimal_jtag_profile.tcl `
  hardware/reports/vivado_2026_1/navigator_z7020_candidate

vivado -mode batch -nolog -nojournal `
  -source scripts/40_implement_reference_zynq_soc.tcl -- `
  hardware/build/navigator_z7020_candidate `
  hardware/reports/vivado_2026_1/navigator_z7020_candidate_impl

python scripts/42_check_navigator_z7020.py
```

当前离线结果：25,436 LUT、25,293 FF、61 BRAM36、72 DSP；100 MHz post-route
WNS `+0.461 ns`、WHS `+0.032 ns`，0 failing path、0 unrouted、0 critical DRC，CDC
全部 safely timed。该结果证明器件和 PL 结构有余量，不证明未确认的 DDR 参数能在实板
稳定工作。

在拿到板前只运行到实现和检查，不执行 `43_export_navigator_candidate.tcl`。拿到板并完成
下方身份确认后，才导出候选 bitstream/XSA。

## 拿板后先拍照/抄录的项目

将 `board_confirmation.template.json` 复制为不提交仓库的
`board_confirmation.local.json`，填写：

1. Zynq 芯片完整丝印，必须确认 `XC7Z020CLG400-2`；
2. 核心板版本和底板版本；
3. 两颗 DDR、以太网 PHY 的芯片丝印；
4. 官方资料包中的 PS7 preset 和 XDC 对应版本；
5. 首测跳线处于 JTAG 模式，串口连接的是 UART0。

若实际是 7010、part/speed grade 不同、DDR 容量不同，立即停止使用本目录的产物并重新
实现。PHY 的公开资料存在 `YT8521S` 与 `RTL8211E-VL` 两种记录，不影响当前最小 NPU
首测，但会影响后续 Linux/网络配置。

## 首次上板顺序

1. JTAG 下载 bitstream，仅检查 DONE、UART、CSR `IP_ID=0x3155504E`；
2. PS DDR 做 64 MiB walking-bit/伪随机测试，再扩大到可用内存；
3. 提交最小 task，核对完成 tag、错误码和 cycle；
4. 提交完整 1,869-command task，逐字节比对 32,768-byte golden；
5. 连续运行 30 分钟并记录 deadline miss、DMA error、watchdog；
6. 上述全部通过后，才启用 QSPI/eMMC 启动和 WM8960/音频通路。

## 资料来源

- 正点原子教程转载：器件、UART0 MIO14/15、PS 时钟和 Vivado DDR 兼容条目；
- 正点原子 FPGA 指南转载：7020/DDR/存储器/晶振和新版 PHY 信息；
- AMD UG1165/UG585：HP0 是 PL 到 PS DDR 的高吞吐非一致性路径；
- 第三方资料包目录仅用于确认官方文件名和板卡版本存在，不作为引脚值来源。
