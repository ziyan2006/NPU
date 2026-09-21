# 领航者 ZYNQ-7020 软件首测方案

这份方案以 JTAG + bare-metal 为第一目标。实物是无底板版本丝印的第三方复刻板，
卖家称参考正点原子 V3.7（WM8960）资料；这不是已验证的官方 PCB 修订号。
FPGA `-2` 速度级别同样仅由卖家提供，尚未经独立核验；当前产物只面向可恢复的
JTAG 首测，不得写入启动 Flash。
Linux 和永久启动放在 NPU/DDR 链路验证之后，避免同时调试设备树、DMA 分配、
启动介质和硬件。

## 音频 standalone 分阶段应用

NPU 与 WM8960 的联合 XSA 可通过 `scripts/101_build_audio_player.ps1` 构建四个独立
应用：`-Mode Tone` 只验证 codec/I2S/扬声器，`-Mode Wav` 从 FAT 分区读取
`0:/test.wav`。WAV 模式只接受 44.1 kHz、16-bit、双声道 PCM，并在 FIFO 预填满
8192 帧后才开始播放；文件不足 8192 帧会报告 `WAV_TOO_SHORT` 且不启用输出。
`-Mode Mp3Bypass` 从 `0:/music.mp3` 流式解码，只接受 MPEG-1 Layer III、
44.1 kHz、双声道，并将 vocal 通道保持为零，用于先验证 SD 到扬声器的完整旁路。
`-Mode FullStem` 使用同一个 MP3 输入，连续执行 STFT、NPU、iSTFT 和对齐混音；NPU
故障会锁存旁路，codec/SD/解码故障会静音。四个构建都只在
`hardware/build/navigator_audio_player/` 生成本地 ELF，不复制 SD 文件，也不连接板卡。

## 音频 MicroSD 分阶段镜像

运行以下命令会重建四个应用，并在忽略提交的
`hardware/build/navigator_audio_boot/` 中生成四套 `BOOT.BIN`、Bootgen readback 和
SHA-256 manifest。脚本只生成本地文件，**不会查找或写入可移动磁盘**。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/102_build_navigator_audio_boot.ps1 -Clean
powershell -ExecutionPolicy Bypass -File scripts/103_test_navigator_audio_boot.ps1
```

| 目录 | 应用 | SD 根目录输入 | 首测目标 |
|---|---|---|---|
| `tone/` | Tone | 无 | WM8960 配置、I2S 时钟、板载扬声器 |
| `wav/` | Wav | `test.wav` | FAT 读取与无压缩 PCM 播放 |
| `mp3_bypass/` | Mp3Bypass | `music.mp3` | MP3 解码、FIFO 与扬声器旁路 |
| `stem/` | FullStem | `music.mp3` | 连续 NPU STEM、KEY0 切换与降级路径 |

每个 manifest 锁定修复后的 handoff FSBL、音频 bitstream、对应 ELF、BIF 和 BOOT.BIN
哈希，并记录 Git commit、Vivado/Vitis 版本和构建模式。测试脚本要求 Bootgen 中只有
FSBL、音频 bitstream、对应 ELF 三个逻辑镜像且顺序固定，同时复跑已有 SD 冷启动回归。
上板时每次只人工复制一种 `BOOT.BIN`，保留上一份已知正常镜像，并按
Tone、Wav、Mp3Bypass、FullStem 顺序逐级验收；前一级失败时不要继续后一级。
当前四套镜像仅完成离线构建和静态审计，不能记为扬声器或 FullStem 实板通过。

## 音频实板验收与串口监控

2026-09-21 FullStem 欠载修复候选（`AUDIO_RUNTIME=sparse-timefix-v1`）：
此前用户日志中 `played` 增加 182079、`uf` 增加 75583，FIFO 水位相同，
实际有效输出为 106496 帧（26 个 4096 帧块），墙钟约 4.129 秒；持续供音约
25.8 kframe/s，低于 44.1 kframe/s。扩大 FIFO 仅增加启动余量，未解决吞吐不足。
发现 BSP 的 `COUNTS_PER_SECOND` 展开为无括号的 `CPU_HZ/2`，原有时间除法
因此低报 4 倍：`blk_us_avg=39508` 应约为 158032 us，旧 `deadline_miss=0`
不能作为实时通过证据。现通过带类型的换算函数修正毫秒/微秒，并避免长时间
运行时先乘单位造成溢出。CPU 实际频率仍须核对启动时只读的 `[CLOCK]` 日志。

前后处理现在初始化时缓存滤波器非零支持区间，每帧只遍历这些区间，保留原系数、
累加顺序、量化和受保护频带；每个方向每帧的滤波器项数上限为 1026。
`fifo_min` 仅在启用播放后采样，避免预填前的空 FIFO 污染统计；它仍是软件采样
的最低水位，真正欠载以硬件 `uf` 为准。该候选须重新实板验收，不能仅凭主机
数值回归测试声称音频实时通过。冷启动先保存 `[CLOCK]` 与连续 `[AUDIO]` 行，
确认秒数接近墙钟、`uf=0`、所有错误为零、`blk_us_max <= 92880`，再测试 KEY0。

对原始 handoff FSBL 的 ELF 初始化表进行只读检查，发现 IO PLL 倍频值为
48、FCLK0 两级分频为 8 和 4（`0xF8000108` 写入 `0x00030000`，
`0xF8000170` 写入 `0x00400800`），按 33.333333 MHz 晶振推算为 50 MHz。
实板日志随后确认完整任务的 NPU 时间约 65.84 ms，整块处理约 93.06 ms，超过
4096 sample/44.1 kHz 的 92.88 ms 截止时间并导致 FIFO 欠载。当前生成脚本会在
三个芯片版本初始化表中仅把 FPGA0_CLK_CTRL 第二级分频从 4 改为 2
（写入 `0x00200800`），使 FCLK0/NPU/AXI 达到 XSA 已签核的 100 MHz；IO PLL、CPU、
DDR、SD、UART 和独立的 50 MHz 音频参考时钟均不改变。补丁和镜像须通过静态门禁，
并用 `[CLOCK]`、`npu_us_avg`、`deadline_miss`、`uf` 完成实板验收。

完整验收记录模板位于 `hardware/reports/audio_board_acceptance.md`。先安装一次串口依赖，
再由 Windows 查找实际端口；不要默认照抄示例中的 COM 号：

```powershell
python -m pip install pyserial
Get-PnpDevice -Class Ports | Format-Table FriendlyName,InstanceId
$env:STEM_UART_PORT = 'COM7'  # 改成上一步识别出的开发板 UART
```

每一级均应断电后把对应目录里的 `BOOT.BIN` **人工**复制到 FAT32 SD 根目录，
JTAG 保持断开，然后冷启动。顺序固定为 Tone、WAV、MP3 Bypass、FullStem；分别准备
无媒体文件、`test.wav`、`music.mp3`、`music.mp3`。某一级失败时保留串口日志和当前
SD 内容，不继续后一级。

FullStem 进入 `PLAY` 后启动 30 分钟监控：

```powershell
python scripts/104_monitor_audio_uart.py `
  --port $env:STEM_UART_PORT `
  --baud 115200 `
  --seconds 1800 `
  --output hardware/build/navigator_audio_acceptance
```

监控器把所有串口行写为带 UTC 时间的 JSONL，并生成 summary JSON。验收期间至少按
KEY0 两次。只有连续 1800 个 `PLAY` 秒、至少两次 `stem` 目标变化、串口无秒数缺口、
`uf/of/dec_err/npu_err/codec_err/deadline_miss` 全为零且
`blk_us_max <= 92880` 时才返回 0 并打印 `AUDIO_BOARD_ACCEPTANCE: PASS`。
日志同时记录 decode+frontend、NPU、backend+sink 的平均/峰值耗时，便于定位超时阶段。
本地捕获目录在 `.gitignore` 覆盖的 `hardware/build/` 下；只把审核后的摘要和必要证据
抄入验收报告，不提交音乐或含版权的原始录音。

## 工件边界

- CSR：`0x43C0_0000`，4 KiB；
- task image：64-byte 对齐、大小为 64-byte 整数倍；
- HP0：非 cache-coherent，提交前 flush，完成后 invalidate；
- 首测 watchdog：`4,000,000` cycle；
- 完整任务：1,869 command，输出 32,768 byte；实板两组输入测得约 3,346,2xx cycle（约 33.46 ms @ 100 MHz）；
- IP ID：`0x3155504E`，RTL/ISA major 都为 1。

## 已实现的 Vitis standalone 适配层

`bringup/navigator_vitis/` 现已提供可直接加入 Vitis standalone 应用的实现：

- `npu_vitis_platform.[ch]`：将 `Xil_DCacheFlushRange`、
  `Xil_DCacheInvalidateRange` 和 `dmb sy` 绑定到通用 `npu_driver`；
- `navigator_npu_runner.c`：将完整 task 拷贝到 2 MiB、64-byte 对齐的 linker-owned
  DDR 缓冲区，提交、轮询、读取计数器，并与 golden output 比较；
- `scripts/63_generate_vitis_task_payload.py`：从本地 task image / golden output
  生成不进入仓库的 C payload。完整任务含模型权重，不能以空模板替代。

该第一版有意使用轮询，避免在尚未由最终 XSA 确认 GIC ID 时写死中断号。编译并运行
`navigator_npu_runner.c` 后，才把它作为 Vitis/BSP、A9 cache 维护和 DDR 数据面的验收结果。

## XSA 导入后需要补的薄适配层（接口说明）

```c
static void cache_clean(void *p, size_t n, void *ctx) {
    (void)ctx;
    Xil_DCacheFlushRange((INTPTR)p, (u32)n);
}

static void cache_invalidate(void *p, size_t n, void *ctx) {
    (void)ctx;
    Xil_DCacheInvalidateRange((INTPTR)p, (u32)n);
}

static void memory_barrier(void *ctx) {
    (void)ctx;
    __asm__ volatile("dmb sy" ::: "memory");
}
```

把这三个函数填入 `npu_platform_ops_t`，CSR 基址优先使用 XSA 生成的 `xparameters.h`
宏，并用编译期断言核对其值是 `0x43C00000`。不要把 CPU 虚拟地址直接当成 DMA 地址。

## 第一阶段：DDR 与 CSR 冒烟测试

`bringup/navigator_jtag_smoke.c` 是可放入 Vitis standalone 应用的首测源码，先于完整
task 推理程序运行。使用**最终确认的 NPU XSA**创建平台，而不是资料盘里的音频例程 XSA；
将本目录的 `include/` 加入编译器头文件搜索路径。默认测试静态分配的 4 MiB DDR 缓冲区，
检查两轮不同数据图案，再读取 NPU `IP_ID`、版本和空闲状态。缓冲区由链接器分配，
不会直接写某个猜测的物理地址。首次通过后设置编译宏 `NPU_DDR_TEST_MIB=64`，重编译
执行 64 MiB 测试。每轮写入后进行 cache flush/invalidate，以免只测到 CPU cache。

2026-09-17 已用 JTAG DAP 对前 64 MiB 完成两种固定图案的直接物理读回测试，PS7
初始化和 NPU CSR 也已实板通过；这降低了 DDR/MIO 配置错误的风险，但不能代替本 bare-metal
测试，因为后者还会验证 CPU cache 维护、ELF 启动、串口日志和软件寄存器访问。

作为不依赖 Vitis 的执行链路预检，`bringup/minimal_a9/` 提供了更小的 A9 probe：它由
`scripts/50_build_navigator_a9_probe.ps1` 使用 GNU Arm Embedded Toolchain 构建，
`scripts/51_run_navigator_a9_probe.tcl` 通过 JTAG 将其易失下载。2026-09-17 实板结果为
`NPU_A9_PROBE PASS`；A9 从 DDR 执行、关闭 cache 后完成 16 KiB 双图案读回，并读到 NPU
CSR。它没有 UART、IRQ、cache flush/invalidate API 或完整 task，因此不能替代本节的 Vitis
 standalone 应用。

完整 NPU 数据面也已完成 JTAG 实测：全零输入与固定非零输入（seed `0x4e505531`）各运行
1,869 条命令，均输出 32,768 byte 且与 Python 固定点参考逐字节一致。JTAG 是非一致的临时
下载路径；正式 bare-metal 程序仍必须在提交前 clean cache、完成后 invalidate cache。

运行顺序：确认 JTAG 启动模式与 PS UART 跳线，先加载已签核的 NPU bitstream/PS 初始化，
再通过 JTAG 下载 ELF 并观察串口。若 DDR 测试失败，不继续读 NPU CSR；若 CSR
`IP_ID` 不等于 `0x3155504E`，不提交 task。当前这份依赖 Xilinx BSP 的源码尚未在实板或
最终 Vitis 平台编译运行；不要把前述独立 minimal probe 的通过误记为它已经通过。

## 首次串口程序步骤

1. 打印 XSA/软件版本、CSR base 和 task DDR 地址；
2. 读 `IP_ID`、RTL version、ISA version、capability；不匹配立即停止；
3. 先跑 256-byte 的错误路径/最小 task，确认寄存器和 watchdog 可恢复；
4. 将完整 task image 放入 DDR 的 64-byte 对齐缓冲区，flush 后提交；
5. 轮询完成，invalidate，再和嵌入的 golden output 做逐字节比对；
6. 打印 completed tag、error context、各 cycle 和 byte counter；
7. 循环 30 分钟，任何 mismatch、timeout 或 error count 增长都判失败。

## 后续 Linux 适配

Linux 阶段使用内核 DMA API 分配/映射 task buffer，并把 NPU 描述为一个 4 KiB MMIO
设备和单路 IRQ。实际 GIC SPI 号必须由最终 XSA/设备树生成结果确定，不能在拿板前写死。
HP0 仍是非一致性端口，因此驱动必须遵守 DMA 同步语义；UIO 只适合早期 CSR 观察，不能
替代正式 DMA/cache 管理。
