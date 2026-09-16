# Zynq SoC 集成与上板前验收

文档版本：`1.0-preboard`

状态：通用 XC7Z020 参考系统已完成 IP 封装、综合和布局布线；真实板卡配置待输入

## 1. 参考系统结构

```text
ARM PS
  M_AXI_GP0 ── AXI interconnect/protocol converter ── S_AXI_CTRL
                                                        STEM NPU
  S_AXI_HP0 ◄─ AXI interconnect/protocol converter ── M_AXI_MEM
  IRQ_F2P   ◄──────────────────────────────────────── IRQ
  FCLK_CLK0 100 MHz ───────────────────────────────── ACLK
  FCLK_RESET0_N ── proc_sys_reset ─────────────────── ARESETN
```

- CSR aperture：PS 物理地址 `0x43C0_0000`，大小 4 KiB；
- 数据路径：64-bit NPU AXI4 master 经 Vivado 自动插入的转换器连接 64-bit PS HP0 AXI3；
- 中断：单路 done/error/watchdog 电平 IRQ 接 `IRQ_F2P`；
- 时钟：参考设计全 PL 单时钟域 100 MHz；
- 一致性：HP0 非 cache-coherent，软件提交前 clean、完成后 invalidate。

参考设计只证明 PL/PS 接口结构可综合实现。它不包含任何开发板 preset，因此不能直接
生成并下载 bitstream；真实 DDR 时序、MIO、启动设备和外设必须来自目标板卡。

## 2. 可复现构建

```powershell
vivado -mode batch -nolog -nojournal `
  -source scripts/38_package_npu_ip.tcl
vivado -mode batch -nolog -nojournal `
  -source scripts/39_create_reference_zynq_soc.tcl
vivado -mode batch -nolog -nojournal `
  -source scripts/40_implement_reference_zynq_soc.tcl
python scripts/_test_npu_driver.py
python scripts/41_check_preboard.py
```

`38` 生成 `ziyan2006.github.io:npu:stem_npu:1.0` IP-XACT，公开 `S_AXI_CTRL`、
`M_AXI_MEM`、`ACLK`、`ARESETN` 和 `IRQ`。`39` 生成并综合参考 Block Design；`40`
执行到 post-route physical optimization，但明确不写 bitstream。

## 3. 当前证据

工具：Vivado 2026.1；临时器件：`xc7z020clg400-1`。

| 验证项 | 结果 |
|---|---:|
| 真实 task RTL / 整数 golden | 1,869 command，32,768 byte 全一致 |
| 任务周期 | 3,254,220 cycle，32.5422 ms @100 MHz |
| 完整 SoC post-route LUT | 25,634 / 53,200（48.18%） |
| 完整 SoC post-route FF | 25,293 / 106,400（23.77%） |
| 完整 SoC BRAM36 | 61 / 140（43.57%） |
| 完整 SoC DSP48E1 | 72 / 220（32.73%） |
| post-route setup | WNS `+0.005 ns`，0 failing path |
| post-route hold | WHS `+0.015 ns`，0 failing path |
| route / DRC | 0 unrouted，0 critical DRC |
| CDC | 单 PL 时钟域；参考系统报告全部 safely timed |
| 驱动 | 39 个 CSR 与 RTL 自动核对，Vivado GCC `-Werror` 交叉编译通过 |

100 MHz 已满足签核定义，但 setup 余量很薄。板卡 preset 会改变 PS 配置和全局布局，
所以真实设计必须重跑 implementation，不能沿用本参考 checkpoint。若目标板在 100 MHz
无法稳定闭合，首选回退为 90 MHz；当前任务此时约 36.16 ms，仍低于 46 ms 门槛。

当前非阻塞告警主要为 DSP 输入/输出寄存建议、BRAM 地址相关寄存器使用异步复位、
以及 Vivado AXI3/AXI4 converter 内部的异步复位方法学提示；均不是 Error/Critical
Warning。板级复位必须只在 NPU idle 或 PS 初始化阶段动作，不能把 reset 当作任务取消。

## 4. 上板前已关闭项目

- [x] ISA、task image、CSR、错误码和量化语义已形成可执行规格；
- [x] CONV2D、requant/post、VEC_ADD、UPSAMPLE2X、DMA 和 task 控制已集成；
- [x] 完整真实任务通过位精确 RTL 回归；
- [x] NPU 单核 XC7Z020 OOC post-route 通过 100 MHz；
- [x] Vivado IP-XACT 封装完整性检查通过；
- [x] GP0 控制、HP0 数据、FCLK/reset 和 IRQ 的参考 Block Design 已综合；
- [x] 完整参考 SoC post-route 通过 100 MHz、route 和 critical DRC；
- [x] OS 无关 CSR 驱动核心和 cache/barrier hook 已实现并交叉编译；
- [x] 提交、轮询、中断、watchdog、错误上下文和性能计数 API 已定义。

## 5. 只有拿到真实板卡后才能关闭的项目

- [ ] 记录开发板型号、精确 part/speed grade、DDR 型号和原理图版本；
- [ ] 导入厂商 PS7 board preset，核对 DDR/MIO/QSPI/SD/UART 配置；
- [ ] 确认运行环境是 bare-metal 还是 Linux，并实现对应 DMA allocation/cache adapter；
- [ ] 重新综合、布局布线，要求 WNS/WHS 非负、0 unrouted、0 critical DRC；
- [ ] 生成 XSA/bitstream，建立 Vitis BSP 或 Linux device-tree、UIO/platform driver；
- [ ] 用 CSR 读回验证 IP ID `0x3155504E`、RTL/ISA 版本和 capability；
- [ ] DDR 中提交对齐的真实 task image，核对输出、tag、cycle 和错误计数；
- [ ] 做 AXI back-pressure/异常注入、软复位和 watchdog 恢复；
- [ ] 连续运行至少 30 分钟，确认 0 deadline miss、0 DMA 错误；
- [ ] 接入实际 STFT/iSTFT、音频 codec、按键渐变后做端到端试听和延迟测试。

## 6. 板级签核判据

只有同时满足以下条件才允许称为“已上板”：板卡专用 bitstream 生成成功，100 MHz 或
经批准的降频点 post-route 时序通过；软件可读 CSR 并完成 task；输出与 golden 一致；
30 分钟长稳无错误；音频端到端实时且无 FIFO 溢出。参考设计通过不等价于这些板级结论。
