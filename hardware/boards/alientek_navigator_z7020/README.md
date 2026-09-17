# 领航者兼容 ZYNQ-7020 复刻板上板配置

状态：`provisional`。实物是第三方复刻板，底板没有版本丝印；卖家称应按正点原子
V3.7（WM8960）资料配置。这里的“V3.7”仅指**参考资料/兼容性主张**，不是已确认的
实物 PCB 版本。厂商资料的 PS7 DDR/MIO 参数已接入 NPU 离线工程并完成布局布线；
两颗 DDR 型号已由用户确认；FPGA `-2` 速度级别由卖家提供、用户转述，尚未经
AMD 查询或原厂包装标签独立核验。已通过 JTAG 对前 64 MiB DDR 做了双图案读回验证；
独立 GNU Arm A9 裸机驱动已完成一次完整 NPU task 的实测；全容量稳定性和正式
Vitis/BSP 软件仍待验证。完整 NPU 任务已完成两组 JTAG/DDR 逐字节输出验证，
详见本页的真实板记录。

## 已确认度较高的硬件

| 项目 | 候选值 | 用途 |
|---|---|---|
| Zynq | 照片中芯片顶面为 `XC7Z020` / `CLG400ABX2317`；卖家提供 `-2`，用户转述 | 器件系列与 CLG400 封装可由照片确认；`-2` 尚未经独立核验 |
| DDR3L | 用户确认两颗均为 `NT5CC256M16EP-EK`；原理图简写 `NT5CC256M16` | 2 × 4 Gb、合计 1 GiB、32-bit；前 64 MiB 已通过 JTAG 双图案读回 |
| Vivado DDR 条目 | V3.7 集成候选 `MT41K256M16 RE-125`；旧通用候选 `MT41J256M16 RE-125` | 均不是物理 DDR 丝印，须以板测确认 |
| PS 晶振 | 33.333333 MHz | PS7 时钟配置 |
| UART | UART0，MIO14/15，115200 | 首次启动日志 |
| PL 晶振 | 50 MHz | 当前 NPU 不使用 |
| QSPI / eMMC | W25Q256 32 MiB / KLM8G1GETF 8 GiB（资料候选） | 首测阶段禁用 |

当前 NPU 由 PS 的 `FCLK_CLK0` 提供 100 MHz，因此不需要给 PL 50 MHz 晶振或底板
LED 添加 XDC。DDR 和 MIO 是 Zynq 的专用 PS 管脚，也不通过普通 PL XDC 分配。

`vendor_v37_jtag_profile.tcl` 是首轮 JTAG 候选：采用厂商 PS7 示例的 DDR 参数，
只打开 UART0 和 NPU 所需的 PS/PL 接口。旧 `minimal_jtag_profile.tcl` 保留作通用
对照，不能作为 V3.7 上板配置。两个配置均未在当前复刻板上验证，不能用于量产或写入启动 Flash。

## 本地厂商参考资料

原始文件复制在 `vendor_reference_local/`，共 18 个、约 4.6 MB。该目录被 `.gitignore`
排除，不会进入公开仓库。资料来自资料盘的
`9_领航者ZYNQ底板V3.7版本（音频芯片为WM8960）`，只供设计核对；未确认再分发授权，
不要用 `git add -f` 上传。可从原资料盘恢复本地副本。

| 本地路径 | 原资料盘位置与用途 |
|---|---|
| `vendor_reference_local/board/baseboard_v3.6_wm8960.pdf` | `3_开发板原理图/领航者ZYNQ底板原理图_V3.6.pdf`；WM8960、启动拨码及接口电路。注意：V3.7 资料包内的图纸标题仍为 V3.6，不能替代实物版本确认。 |
| `vendor_reference_local/board/coreboard_v2.5.pdf` | `3_开发板原理图/ZYNQ_CORE_2V5_user.pdf`；DDR、MIO、时钟、电源和 PS 接线。 |
| `vendor_reference_local/board/navigator_io_v3.7.xlsx`、`navigator_io_v3.7.xdc` | 同目录的 IO 引脚总表、`NAVIGATOR_ZYNQ_IO.xdc`；核对 PL 管脚。XDC 含多个例程的重名端口，不可整体导入 NPU。 |
| `vendor_reference_local/ps7/system_processing_system7_0_0.xci`、`system.bd` | `4_Source_Code/2_Embedded_Vitis/ZYNQ_Vitis_7020.rar` 中 `21_audio_loopback` 的 PS7 IP 参数与 block design；作为 DDR/MIO 配置的主要来源。 |
| `vendor_reference_local/ps7/ps7_init.c`、`ps7_init.h`、`ps7_init.tcl`、`system_wrapper.xsa` | 同一例程生成的初始化代码和硬件平台；仅作配置对照，不是 NPU 的 XSA/bitstream。 |
| `vendor_reference_local/audio/` | `4_Source_Code/1_FPGA_Design/ZYNQ_7020_FPGA.rar` 中 `37_audio_loopback/rtl` 的 8 个文件；WM8960 寄存器配置、I2S 和音频引脚参考。 |

V3.7 例程的 PS7 XCI 记录了 32-bit DDR3L、533.333 MHz、`MT41K256M16 RE-125`、
UART0 MIO14/15 等参数。它证明厂商示例如何配置 PS，不证明手中板卡与示例完全相同；
尤其物理 DDR 原理图标为 `NT5CC256M16`。用户报告的完整型号
`NT5CC256M16EP-EK`，按[南亚原厂数据手册](https://www.verical.com/datasheet/nanya-dram-nt5cc256m16ep-ek-6992005.pdf)
是单颗 4 Gb、×16 的 DDR3L，`-EK` 额定 1866 MT/s；当前厂商 PS7 例程采用
533.333 MHz DDR 时钟（约 1066 MT/s），容量、位宽、电压类别和额定速率层面
没有明显冲突。这是**规格推断**；两颗芯片型号虽已由用户确认，
替代料的全部时序/PCB 延时仍须上板验证。本次移植仅取板级 DDR/MIO 参数，保留 NPU
所需 GP0/HP0/IRQ 和 100 MHz FCLK；离线布局布线已完成，前 64 MiB DDR 的实板测试
记录见下文，不能将其外推为全容量 DDR 验收。

## V3.7 厂商配置的离线构建

以下构建要求本地 `vendor_reference_local/ps7/system_processing_system7_0_0.xci`
存在。`vendor_v37_jtag_profile.tcl` 从中读取全部 72 项 DDR 配置，同时只启用首测所需
UART0，保持 QSPI/SD/以太网关闭。PS 的 GP0/HP0/IRQ 和 PL 100 MHz 时钟由 NPU
集成脚本设置。Vivado 2026.1 传参使用 `-tclargs`。

```powershell
vivado -mode batch -nolog -nojournal `
  -source scripts/39_create_reference_zynq_soc.tcl -tclargs `
  xc7z020clg400-2 hardware/build/ip_repo `
  hardware/build/navigator_z7020_vendor_v37 `
  hardware/boards/alientek_navigator_z7020/vendor_v37_jtag_profile.tcl `
  hardware/reports/vivado_2026_1/navigator_z7020_vendor_v37_synth

vivado -mode batch -nolog -nojournal `
  -source scripts/40_implement_reference_zynq_soc.tcl -tclargs `
  hardware/build/navigator_z7020_vendor_v37 `
  hardware/reports/vivado_2026_1/navigator_z7020_vendor_v37_impl

python scripts/44_check_navigator_v37.py
```

Vivado 2026.1 离线结果：25,436 LUT、25,293 FF、61 BRAM36、72 DSP；
100 MHz post-route WNS `+0.461 ns`、WHS `+0.032 ns`，无 setup/hold 失败路径、
无未布线网络、无 Critical DRC。`scripts/44_check_navigator_v37.py` 已核对
厂商 XCI 的 SHA-256、Vivado 生成 PS7 XCI 中全部 72/72 个 DDR 字段、
BD 中 58/72 个显式字段、MIO 和时序报告，结果 `PASS`。
另有普通 DRC/methodology warnings（包括 DSP 流水线和 AXI IP 复位提示），
不等同于已完成实物 DDR 信号完整性验证。

旧的 `minimal_jtag_profile.tcl` 仍保留为可对照的通用候选，旧结果为 25,436 LUT、
25,293 FF、61 BRAM36、72 DSP；100 MHz post-route WNS `+0.461 ns`、WHS
`+0.032 ns`。PS7 是硬核，两个候选的 PL 资源/时序数字相同不代表 DDR 配置相同。

在填齐完整实物及首测记录前，只运行到实现和检查，
不执行 `43_export_navigator_candidate.tcl`。
导出脚本会核对 V3.7 DDR/MIO 参数，并通过 `45_validate_navigator_confirmation.py`
核对实物记录。复刻板**不需要虚构底板版本号**；只有以“无版本丝印的第三方复刻板、
卖家称参考 V3.7”如实登记，且如实填写卖家提供的速度级别及 DDR/JTAG 首测条件后，才允许导出
**JTAG-only 候选** bitstream/XSA。该门不代表 DDR 已通过板测，更不允许烧写启动 Flash。
`fpga_speed_grade_basis=seller_statement` 仅允许此可恢复的复刻板 JTAG 原型流程，
不等于器件真实性或 `-2` 的独立证明；正式验收仍应使用 AMD 查询、原厂包装标签等
更直接的证据并完成实板测试。

填齐并通过本地记录校验后，用以下命令导出只用于 JTAG 的候选文件；输出到忽略提交的
`hardware/build/navigator_z7020_vendor_v37_export/`。导出只在电脑上生成文件，
不会自动下载到板卡，也不会写入启动 Flash。

```powershell
python scripts/45_validate_navigator_confirmation.py `
  hardware/boards/alientek_navigator_z7020/board_confirmation.local.json
vivado -mode batch -nolog -nojournal `
  -source scripts/43_export_navigator_candidate.tcl -tclargs `
  hardware/build/navigator_z7020_vendor_v37 `
  hardware/build/navigator_z7020_vendor_v37_export
```

## 复刻板首测前需核对的项目

将 `board_confirmation.template.json` 复制为不提交仓库的
`board_confirmation.local.json`，按实际观察填写。`board_origin=third_party_clone`、
`reference_profile=alientek_navigator_v3.7_wm8960` 和
`reference_basis=seller_statement` 只记录来源，不证明电路完全一致。

1. 芯片顶面的 `XC7Z020`、`CLG400ABX2317` 可确认器件系列及 CLG400 封装，
   **不能从这一行推断 `-2`**。AMD 的 [UG865 封装标识说明](https://docs.amd.com/v/u/en-US/ug865-Zynq-7000-Pkg-Pinout)
   指出速度/温度级别在另一行或二维条码信息中；[XCN16014](https://docs.amd.com/v/u/en-US/xcn16014)
   说明 2017 年起 Zynq-7000 可省略第四行，需通过二维条码查验。PCB 印字不是替代证据；
   若可扫描芯片二维条码，使用 [AMD Device Lookup](https://www.amd.com/en/support/utilities/amd-device-lookup-support.html)
   或原厂包装标签核对 `-2`。当前只有卖家提供的型号信息，已如实预填在模板中；
   它可支持受限 JTAG 首测，但不能作为正式板卡验收依据；
2. 底板无版本丝印就填写 `base_board_revision=null`、
   `base_board_revision_marking=absent`；核心板若也无丝印，保持 `null`，不要填“V3.7”；
3. 两颗 DDR 已由用户确认为 `NT5CC256M16EP-EK`，在 `ddr_markings` 中分别填写；
4. 确认 JTAG 启动模式、串口连接和只做可恢复的 JTAG 下载；
5. WM8960/音频、PHY 与外接 IO 留待基础 NPU/DDR 首测通过后再核对。

若实际是 7010、part/speed grade 不同、DDR 容量不同，立即停止使用本目录的产物并重新
实现。复刻板的 PCB 走线、时钟和电源也可能与官方 V3.7 不同，必须通过 JTAG DDR
读写及后续稳定性测试。PHY 的公开资料存在 `YT8521S` 与 `RTL8211E-VL` 两种记录，
不影响当前最小 NPU 首测，但会影响后续 Linux/网络配置。

## 首次上板顺序

1. JTAG 下载 bitstream，检查 DONE、CSR `IP_ID=0x3155504E`；已完成。UART 日志仍待验证；
2. 用 `software/bringup/navigator_jtag_smoke.c` 先测 4 MiB DDR，再以
   `NPU_DDR_TEST_MIB=64` 重编译测 64 MiB，并核对 NPU CSR；JTAG 的直接 DAP 读写
   已完成等价的前 64 MiB 双图案测试；独立 A9 bare-metal 最小探针也已通过，
   Vitis/BSP 版本仍待执行；
3. 提交最小 task，核对完成 tag、错误码和 cycle；**已完成**；
4. 提交完整 1,869-command task，逐字节比对 32,768-byte golden；**已完成零输入与固定非零输入两组**；
5. 用 A9 裸机驱动提交同一完整 task，并核对 completed tag、错误码、retired、cycle 和输出校验值；**已用固定非零 task 通过**；
6. 连续运行长时间压测并记录 deadline miss、DMA error、watchdog；**已完成 18,000 轮（18.65 分钟）连续高压测试，0 error、0 watchdog、输出 FNV-1a 逐轮 100% 吻合，周期抖动仅 0.061%，贴散热片工况稳定**；
7. 上述全部通过后，才启用 QSPI/eMMC 启动和 WM8960/音频通路。

## MicroSD 冷启动串口诊断

2026-09-17 检查到的 `F:\BOOT.BIN` 含有 `zynq_fsbl.elf`、NPU bitstream 和加载到
`0x1000_0000` 的 `navigator_a9_sd_runner.elf`。该应用应从 **UART0 / MIO14/15**
以 **115200, 8N1, 无硬件或软件流控** 输出启动信息，并在成功后每秒输出一条
`[HEARTBEAT]`。因此，若串口始终没有任何文本，不能仅以绿灯判断 A9 应用已正常
handoff：绿灯可能只表示 PL 配置成功。

先在电脑上确认 CH340 出现的 COM 口（此前实测为 `COM3`）并以以上参数打开终端，
再给板卡完整断电/上电。即使错过起始日志，正常应用也会持续打印 heartbeat。

若仍无数据，在板卡上电、启动拨码保持 SD 模式且 JTAG 已接入时运行：

```powershell
xsdb scripts/70_probe_navigator_sd_boot.tcl
```

该脚本只会短暂暂停 A9#0、打印 PC 和 `0x1000_0000` 的 8 个 DDR 字，再立即继续
运行；它不会配置 PL，也绝不写入 SD、QSPI 或 eMMC。PC 落在
`0x1000_0000` 附近说明 A9 已进入 SD runner，此时应优先检查 CH340/终端链路；
PC 落在其他区域或出现异常入口，则继续检查 FSBL 到应用的 handoff 与 DDR 初始化。

## 2026-09-17 真实板 JTAG 记录

本节是实测事实，不替代上述仍待完成的软件、全容量与长期稳定性验收。

- Vivado Hardware Manager 识别到物理 `xc7z020` 与 `arm_dap_0`；候选 bitstream 通过
  JTAG 易失下载后，NPU CSR 读回 `IP_ID=0x3155504E`、RTL/ISA 均为 `0x00010000`、
  status 为 idle (`0x00000001`)。
- 从候选 XSA 导出的 `ps7_init.tcl` 初始化 PS 时钟、MIO 与 DDR 后，
  `scripts/46_probe_navigator_jtag.tcl` 对 `0x1000_0000`、`0x1000_1000` 各做两种
  32-bit 图案的写/读/恢复，均通过。
- `scripts/48_stress_navigator_ddr.tcl` 以 DAP AP0 直接物理访问
  `0x1000_0000`–`0x13FF_FFFF`。将测试拆为 8 个独立的 8 MiB 区段，每段重新配置
  PL、重新执行 PS7 初始化，并分别完整读回 `0xA5A55A5A` 与 `0x5A5AA5A5`，均通过；
  总覆盖 64 MiB。
- 64 MiB 单次连续任务及 16 MiB 任务均曾在未发生数据 mismatch 前出现
  `FT_Write io error`，即 JTAG/USB 传输中断。缩小到 8 MiB、每段重新连接后稳定通过。
  这不是 DDR 错误证据，但后续 30 分钟稳定性测试应使用裸机程序而非慢速 JTAG DAP。
- `software/bringup/minimal_a9/navigator_a9_probe.c` 使用独立 GNU Arm 工具链构建为
  768-byte ARM32 ELF，载入 DDR `0x1220_0000` 后由 A9#0 实际执行。它禁用 I/D cache，
  在 DDR `0x1201_0000` 做 16 KiB、两图案读回，再读取 NPU CSR；结果区返回
  `0x4E5052A5`（PASS）、`IP_ID=0x3155504E`、RTL/ISA=`0x00010000`，idle bit 为 1。
- 最小 `END` task 已在 HP0/DDR 上实测通过；在**同一次** PS/DDR 初始化中，激活区全零和
  单字节 `0xA5` 的两个镜像均成功完成，表明 task loader/LUT 预取不依赖激活区数据为零。
- 完整任务（`1,869` 条 command）已在易失 DDR `0x0100_0000` 运行：全零输入用时
  `3,346,185` cycles（约 `33.46 ms @ 100 MHz`），输出 `32,768` bytes 与软件参考逐字节一致，
  SHA-256 为 `f1ce7318e5236f7f27a5c32acb875704d6886a3ba9909c0793c0b2d6752f31ba`。
  固定非零输入（seed `0x4e505531`）用时 `3,346,190` cycles，输出也逐字节一致，SHA-256 为
  `d2264856e34e53cc1a0107f9b14a54ab041c11cc6c3450e5e975e81b05650efa`。
- `scripts/67_run_navigator_a9_task.tcl` 通过 `software/src/npu_driver.c` 的 A9#0
  freestanding 路径提交同一固定非零完整 task：提交后 CSR 读回
  `TASK_BASE=0x01000000`、`TASK_BYTES=0x001C4140`、busy 状态；完成 tag
  `0x41395053`、错误码 `0`、retired `1,869`、总计 `3,355,380` cycles
  （约 `33.55 ms @100 MHz`）。A9 对 32,768-byte 输出计算的 FNV-1a 为
  `0x4DB54515`，与 host golden 一致。该路径仅使用 JTAG 易失下载和 DDR，
  不使用 Vitis BSP，也不写入非易失存储。
- 完整非零任务长循环压测（`scripts/69_stress_navigator_a9_task.tcl`）：A9#0 连续循环执行
  18,000 轮完整 NPU 任务（每轮 1,869 条指令），耗时 1,119 秒（约 18.65 分钟，实际吞吐
  16.1 轮/秒，总搬移 DDR 数据逾 33.3 GB）。全部 18,000 轮每轮逐次实时计算 32,768 字节输出的
  FNV-1a 校验值均为 `0x4DB54515`（100% 吻合），全程错误数 0、看门狗触发 0；最小周期
  3,356,412、最大周期 3,358,451，周期抖动仅 2,039 周期（0.061%）。实物芯片贴附被动散热片，
  实测表面温烫但系统稳定无热衰减降速。
- 原始 JTAG 链路在连续大块下载或反复 PS7 初始化后曾出现 AP/USB 超时。稳定复现路径是：
  用 `scripts/61_run_navigator_end_task_pair.tcl` 完成一次初始化和最小任务检查，再用
  `scripts/62_run_navigator_resident_task.tcl` 提交完整分块任务；两者均只使用 JTAG 与易失 DDR。
- 所有上述操作仅使用 `fpga -file` 和易失 DDR 读写；没有对 QSPI、eMMC、SD 或其他
  非易失存储发出写入命令。

可复现单段测试（`base` 必须是 4 KiB 对齐的 DDR 地址）：

```powershell
xsdb scripts/48_stress_navigator_ddr.tcl `
  hardware/build/navigator_z7020_jtag_probe/ps7_init.tcl `
  hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit `
  8 0x10000000
```

构建并运行 A9 最小探针（同样仅 JTAG 易失操作）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/50_build_navigator_a9_probe.ps1
xsdb scripts/51_run_navigator_a9_probe.tcl `
  hardware/build/navigator_z7020_jtag_probe/ps7_init.tcl `
  hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit `
  hardware/build/navigator_a9_probe/navigator_a9_probe.elf
```

## 不依赖 Vitis 的 A9 软件任务验证

`scripts/66_build_navigator_a9_task_runner.ps1` 用已安装的 GNU Arm
Embedded Toolchain 构建一个 freestanding ARM32 ELF，不依赖 BSP、Flash 或
Vitis 平台服务。它复用 `software/src/npu_driver.c`：主机经 JTAG 把 task
分块放入易失 DDR 后，A9#0 读取一小段配置、提交 task、轮询完成、写回
completed tag/错误码/retired/cycle，并对 NPU 写出的输出做 FNV-1a 校验。

当前完整非零任务的本地元数据可这样取得：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/66_build_navigator_a9_task_runner.ps1
python scripts/68_describe_navigator_a9_task.py `
  --image hardware/build/navigator_seeded_task/task_image.bin `
  --golden hardware/build/navigator_seeded_task/expected_output.bin `
  --program-dir hardware/generated/bott2_mir1k_v1_program
```

当前输出会给出 `task_bytes=1851712`、`output_offset=966784`、
`output_bytes=32768`、`expected_fnv1a=0x4DB54515`。仅当上述本地文件仍对应
同一 task 时，使用以下命令（全部为 JTAG/易失 DDR 操作）：

```powershell
xsdb scripts/67_run_navigator_a9_task.tcl `
  hardware/build/navigator_z7020_jtag_probe/ps7_init.tcl `
  hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit `
  hardware/build/navigator_a9_task_runner/navigator_a9_task_runner.elf `
  hardware/build/navigator_seeded_task/jtag_chunks `
  1851712 966784 32768 0x4DB54515
```

不要复用这些数字给另一份 task：先运行 `scripts/68_describe_navigator_a9_task.py`
取得其元数据。

运行多轮长时间高压测试（以 18,000 轮 / 约 18.65 分钟为例）：

```powershell
xsdb scripts/69_stress_navigator_a9_task.tcl `
  hardware/build/navigator_z7020_jtag_probe/ps7_init.tcl `
  hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit `
  hardware/build/navigator_a9_task_runner/navigator_a9_stress_runner.elf `
  hardware/build/navigator_seeded_task/jtag_chunks `
  1851712 966784 32768 0x4DB54515 18000
```


Vitis 版 runner 仍保留在 `software/bringup/navigator_vitis/`。本机的 Vitis
2026.1 CLI 目前连其官方 `zc702` standalone 示例都无法生成 Processor List，
而 AMD `sdtgen` 能正确解析 NPU XSA；因此这是本机 Vitis Embedded 组件/服务问题，
不是板卡或 NPU XSA 问题。修复该安装后可直接运行
`scripts/65_build_navigator_vitis.ps1 -Clean` 重新尝试生成 BSP ELF。

## 其他设计依据

- 正点原子教程转载：器件、UART0 MIO14/15、PS 时钟和 Vivado DDR 兼容条目；
- 正点原子 FPGA 指南转载：7020/DDR/存储器/晶振和新版 PHY 信息；
- AMD UG1165/UG585：HP0 是 PL 到 PS DDR 的高吞吐非一致性路径；
- 本地厂商参考资料的版本、路径和适用边界见上节。
