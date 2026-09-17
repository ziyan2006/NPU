# SD MP3 STEM 扬声器验证设计

日期：2026-09-17

状态：设计已确认，等待实施计划

目标分支：`codex/audio-stem-bringup`

基线提交：`92155fc3e198a01432e8007f832d0cf4d959832b`

## 1. 目标

本阶段只验证产品主链路能否在现有第三方复刻领航者 Zynq-7020 板上自主运行：

1. 板卡从 MicroSD 冷启动，无需 JTAG；
2. 从 FAT 分区读取固定文件 `/music.mp3`；
3. 在 Cortex-A9 上解码 MP3，并完成模型前后处理；
4. 使用现有 PL NPU 执行人声掩码网络；
5. 通过 WM8960 的扬声器功放驱动板载扬声器；
6. 每次按下 KEY0，在原音和去人声伴奏之间切换；
7. 切换不中断解码或 NPU，且不产生明显爆音。

最终产品将把 MP3 解码后的 PCM 源替换为 WM8960 ADC/I2S 输入。本设计要求从 PCM 环形缓冲开始的 STFT、NPU、iSTFT、按钮和播放通路可以直接复用。

## 2. 首版范围

### 2.1 输入限制

- 路径固定为 `/music.mp3`；
- MPEG-1 Layer III；
- 44.1 kHz；
- 双声道；
- CBR 和 VBR 均可；
- 解码输出统一为有符号 16-bit 交错 PCM。

不满足采样率、声道数或编码层级时，应用通过 UART 报告原因，不开始播放。首版不做实时重采样，不做目录浏览、播放列表、暂停、快进或音量 UI。

### 2.2 输出和控制

- 输出设备：WM8960 的 SPKL/SPKR 扬声器功放和板载扬声器；
- 输出格式：44.1 kHz、双声道、24-bit I2S、每声道 32-bit slot；
- KEY0：PL 管脚 `L14`，低电平有效，20 ms 消抖；
- 上电默认 STEM 关闭，即输出延迟对齐后的原音；
- 每次 KEY0 有效按下切换一次目标状态；
- 状态切换使用 30 ms 逐采样增益斜坡。

### 2.3 明确不在本阶段内

- WM8960 ADC/Line-in/麦克风实时输入；
- Linux、ALSA 和设备树；
- 多文件播放器功能；
- 48 kHz 或其他采样率；
- 最终音质签核；
- 为连续模型上下文新增状态化 NPU 指令或硬件 FFT。

## 3. 已验证基线与新增风险

当前 `main` 已在实板完成：MicroSD 冷启动、FSBL handoff、DDR、NPU bitstream、1,869 条指令任务、32,768-byte 输出逐字节校验和 UART heartbeat。现有 NPU CSR 位于 `0x43C0_0000`。

本阶段新增且尚未实板验证的部分是：

- WM8960 44.1 kHz 扬声器配置；
- 音频时钟和 I2S TX；
- A9 到 PL 的 PCM FIFO；
- FatFs/xilffs 文件读取；
- MP3 流式解码；
- A9 上的连续 STFT/iSTFT；
- 每块动态替换 NPU 输入；
- 按钮消抖和低延迟 STEM 混音。

这些部分必须逐层验收，不能用最终“有声音”替代各层的独立检查。

## 4. 总体架构

```text
MicroSD /music.mp3
        |
        v
FatFs -> MP3 decoder -> stereo PCM ring
                           |
                 +---------+----------+
                 |                    |
                 v                    v
          delayed mixture       STFT 1024/256
                                      |
                               513 -> 128 bands
                                      |
                           INT12 task input patch
                                      |
                                  PL NPU
                                      |
                              vocal mask Q1.11
                                      |
                           band expand + iSTFT
                                      |
                              vocal estimate
                 |                    |
                 +---------+----------+
                           |
             64-bit {mix LR, vocal LR} FIFO
                           |
                   PL sample mixer <- KEY0
                           |
                        I2S TX
                           |
               WM8960 SPKL/SPKR -> speaker
```

处理分工如下：

- Cortex-A9：文件系统、MP3 解码、STFT、频带矩阵、量化、NPU 提交、掩码展开、iSTFT 和 PCM 缓冲调度；
- 现有 NPU：执行冻结的 16-frame 频谱掩码网络；
- 新增音频 PL：PCM FIFO、按钮消抖、逐采样 STEM 混音、I2S TX、WM8960 配置和计数器；
- DDR：任务镜像、MP3/PCM 环形缓冲和前后处理工作区；
- BRAM：8192 个 64-bit 音频帧的异步 FIFO。

## 5. 时钟与 WM8960

### 5.1 音频主从关系

PL 作为音频时钟主设备，WM8960 配置为 I2S 从设备。这样采样率由 PL 固定控制，不依赖原厂示例中面向 48 kHz 的 codec-master 配置。

从板载 50 MHz 时钟生成：

- MCLK：11.289602856 MHz，Clocking Wizard 参数 `D=3`、`M=63.25`、`O=93.375`，相对 11.2896 MHz 误差约 0.253 ppm；
- BCLK：MCLK / 4，约 2.822400714 MHz；
- DAC LRCLK：BCLK / 64，约 44,100.011 Hz。

I2S 每声道使用 32-bit slot，其中有效样本为高 24 bit。A9 提供的 16-bit PCM 饱和后左移 8 bit 输出。

### 5.2 Codec 配置

复用厂商 WM8960 I2C 写时序，但用声明式寄存器表替换仅支持固定 48 kHz 的硬编码序列。配置行为必须满足：

- WM8960 为 slave；
- I2S 格式、24-bit word length；
- SYSCLK 直接取 11.2896 MHz MCLK，不启用 codec 内部 PLL；
- DAC、左右 output mixer、SPKL/SPKR 和 class-D 扬声器级启用；
- ADC、耳机和 line-out 不是首版验收路径；
- 上电先静音，时钟稳定且全部 I2C 写入 ACK 后再做短渐入；
- 任一 I2C NACK 保持静音，并在状态寄存器中锁存错误。

具体寄存器值由 WM8960 数据手册字段生成，RTL 测试必须检查写入顺序、slave 位、采样率和扬声器使能位，禁止直接沿用原厂 `R4=48 kHz` 配置。

## 6. 音频 PL 模块

### 6.1 模块组成

新增独立 `audio_out_axi` 子系统，不修改 `npu_top` 内部计算逻辑：

- AXI4-Lite CSR/数据写接口；
- 64-bit、8192-depth 异步 FIFO；
- KEY0 两级同步、20 ms 消抖和单次按下事件；
- Q1.15 STEM 增益斜坡；
- 饱和混音器；
- 24-bit I2S TX；
- WM8960 I2C 初始化器；
- 测试音发生器；
- 欠载、溢出、写入和播放计数器。

FIFO 的一个条目对应同一时刻的四个 `int16` 样本：

```text
bits 15:0   delayed_mix_left
bits 31:16  delayed_mix_right
bits 47:32  vocal_estimate_left
bits 63:48  vocal_estimate_right
```

8192 帧约等于 185.8 ms 音频，用于吸收 MP3 解码和 A9 前后处理抖动。它不能掩盖平均处理速度慢于实时的问题。

### 6.2 AXI 地址

保留 NPU `0x43C0_0000` 的 4 KiB aperture。音频子系统使用独立的 4 KiB aperture `0x43C1_0000`，最终地址必须由 block design 强制分配，并由生成的 `xparameters.h` 和编译期断言共同检查。

建议寄存器契约：

| Offset | 名称 | 行为 |
|---:|---|---|
| `0x00` | `IP_ID` | 固定为 `0x31445541`，ASCII `AUD1` |
| `0x04` | `VERSION` | major/minor |
| `0x08` | `CONTROL` | enable、soft reset、codec reinit、tone enable |
| `0x0C` | `STATUS` | ready、muted、codec done/error、FIFO full/empty |
| `0x10` | `MIX_FRAME` | 暂存 `{right,left}` |
| `0x14` | `VOCAL_FRAME` | 写 `{right,left}` 并原子提交完整 64-bit 条目 |
| `0x18` | `FIFO_LEVEL` | 当前帧数 |
| `0x1C` | `FIFO_CAPACITY` | 固定 8192 |
| `0x20` | `UNDERFLOW_COUNT` | 饱和计数器，W1C |
| `0x24` | `OVERFLOW_COUNT` | 饱和计数器，W1C |
| `0x28` | `PLAYED_FRAMES` | 已消费帧数 |
| `0x2C` | `STEM_STATE` | target、ramping、当前 Q1.15 gain、KEY0 状态 |
| `0x30` | `CODEC_STATUS` | 配置索引、最后 ACK/NACK |
| `0x34` | `TONE_CONTROL` | 频率步进和幅度，仅用于 bring-up |

当 FIFO 已满时，`VOCAL_FRAME` 不提交半条数据并递增 overflow。软件必须先检查可用空间，再以 `MIX_FRAME`、`VOCAL_FRAME` 顺序成对写入。

### 6.3 STEM 按钮路径

NPU 无论按钮状态如何都持续运行。PL 对 FIFO 头部样本执行：

```text
output = saturate16(delayed_mix - round(gain * vocal_estimate))
```

- STEM 关闭：`gain=0`；
- STEM 开启：`gain=32767`；
- 30 ms 斜坡：1323 个采样逐步更新 gain；
- 在斜坡期间再次按键时，从当前 gain 向新目标反向，不产生不连续跳变。

按钮响应不受 185.8 ms FIFO 排队延迟影响，因为混音发生在 FIFO 消费端、紧邻 I2S TX。

## 7. A9 软件模块

第一版使用 Vitis standalone BSP，不引入 Linux：

- `xilffs`/FatFs 访问启动 SD 卡的 FAT 分区；
- 固定版本的 `minimp3` 负责 MPEG-1 Layer III 帧解码；
- 固定版本的 KissFFT float32 负责 1024 点 FFT/IFFT；
- 两个第三方组件保留各自许可证和版本哈希；
- 单个 Cortex-A9 核执行 cooperative loop，首版不依赖中断和 RTOS。

建议软件边界：

- `sd_mp3_source`：挂载、ID3 跳过、帧同步、顺序读取和格式拒绝；
- `pcm_ring`：解码 PCM 的生产者/消费者缓冲；
- `stem_frontend`：连续 Hann STFT、513-bin magnitude、128-band analysis matrix 和 INT12 打包；
- `stem_npu_session`：任务常驻、局部 cache 维护、提交、轮询和计数器；
- `stem_backend`：Q1.11 掩码解析、低频保护、513-bin synthesis matrix、复数乘法和连续 overlap-add；
- `audio_sink`：将对齐的原音和人声估计成对写入音频 FIFO；
- `player_main`：状态机、预填充、UART 遥测和错误降级。

MP3 解码器输出的 `int16` PCM 在进入频谱前转换为 `float32`，缩放规则固定为 `sample / 32768.0f`。原音延迟线保留同一份归一化 PCM，并在写入 PL FIFO 前统一量化回 `int16`，禁止前处理和旁路采用不同的电平标尺。

## 8. 模型数据契约

冻结模型参数：

- 44.1 kHz；
- `N_FFT=1024`；
- `hop=256`；
- 513 个单边频点；
- `legacy_log` 128-band 矩阵；
- linear magnitude 前端；
- 每块 16 帧，即 4096 个新 PCM 采样、约 92.88 ms；
- 250 Hz 低频保护，对应前 44 个 band 不减人声；
- `mask_mode=independent`。

FFT 数值约定必须与 PyTorch 参考一致：periodic Hann 窗、forward FFT 不做 `1/N` 归一化、inverse FFT 做 `1/N` 归一化、单边频谱含 DC 和 Nyquist。513-to-128 analysis matrix 与 128-to-513 synthesis matrix 由现有 Python 参考生成 `float32 little-endian` 常量，生成脚本同时记录尺寸和 SHA-256；C 实现不重新推导频带中心。

### 8.1 输入

当前任务输入 tensor 为 `[C=2,F=128,T=16]`、`INT12_IN_INT16`、NHWC8，大小 32,768 byte。第一层输入步长来自导出 manifest，为 `0.07046897899364925`。每个 magnitude 值按该步长量化并饱和到 signed INT12。

当前 task image 中 activation section offset 为 49,280，输入 tensor base offset 为 0；实现不得硬编码这两个数字，必须从生成的任务元数据头文件取得并做范围检查。

NHWC8 中每个 `(band,time)` 占 16 byte：lane 0/1 为左右声道，lane 2..7 清零。

### 8.2 输出

输出 tensor 为 `[C=4,F=128,T=16]`、Q1.11 signed INT16、NHWC8，分配 32,768 byte。当前输出绝对 offset 为 966,784，但同样只能由生成元数据导出。

- lane 0/1 是左右 vocal mask；
- lane 2/3 是独立 accompaniment head，首版不使用；
- vocal mask 为 `clamp(abs(q) / 2047, 0, 1)`；
- 前 44 个 band 强制为 0，再通过 128-to-513 synthesis matrix 展开；
- vocal estimate 为 `iSTFT(mixture_spectrum * vocal_mask)`；
- 最终伴奏由 PL 在播放时计算 `delayed_mix - vocal_estimate`。

### 8.3 流式任务提交

1. 启动时复制并检查完整 1.85 MiB task image，校验静态 header/CRC；
2. 完整 clean cache 一次；
3. 每块只覆盖并 clean 32 KiB 输入 tensor；
4. 提交同一个 task physical address；
5. 完成后只 invalidate 32 KiB 输出 tensor；
6. 检查 completion tag、error、retired command count 和 deadline；
7. 不在每块重新复制权重、指令或整个 activation arena。

现有 `npu_submit()` 会维护整个 task buffer 的 cache，需增加显式的 resident/streaming session API。旧接口行为保持不变，避免破坏已通过的冷启动回归。

## 9. 连续 STFT/iSTFT 与延迟

STFT 使用 1024 点 Hann 窗和 256 hop。为了与训练路径的 `center=True` 对齐，分析端保留 512-sample look-ahead；启动边界使用与主机参考一致的反射填充。iSTFT 使用持续 overlap-add 和窗口平方归一化，不能把每个 16-frame 块独立 iSTFT 后直接拼接。

原音必须经过同样的固定延迟线，使 `delayed_mix` 与 `vocal_estimate` 逐采样对齐。主机参考测试通过互相关和 impulse index 验证该对齐，而不是只靠听感判断。

每个 92.88 ms block 的稳态预算：

| 阶段 | 预算 |
|---|---:|
| MP3 解码、STFT、频带映射和输入打包 | 25 ms |
| NPU（实板历史值约 33.55 ms @ 100 MHz） | 40 ms |
| 掩码展开、iSTFT、FIFO 写入 | 20 ms |
| 调度余量 | 7.88 ms |
| 总截止时间 | 92.88 ms |

启动时先产生至少两个完整 block，再 enable I2S 消费。稳态平均处理时间必须低于 block 时长；增加 FIFO 只能吸收抖动，不能修复吞吐不足。

## 10. 当前 16-frame 模型限制

现有 task 每次从同一个 16-frame 输入 tensor 开始执行，卷积左侧 padding 会在块边界重置。当前硬件任务没有跨块隐藏状态，也没有重叠窗口的上下文保存。

因此首版可以验证“MP3 -> 动态 NPU -> 扬声器 -> 按钮”的完整可行性，但可能每 92.88 ms 出现模型边界伪影。首版验收不把该现象误判为 I2S/FIFO 故障，也不把演示通过描述为最终连续音质完成。

进入真实音频输入阶段前，必须单独选择并验证以下方案之一：

- 扩展任务接口，显式保存各层因果历史；或
- 使用重叠推理并只输出稳定区域，同时证明新的实时预算成立。

## 11. 状态机与故障降级

应用状态：

```text
BOOT -> SELF_TEST -> MOUNT -> OPEN -> PREFILL -> PLAY
  |         |          |       |        |        |
  +---------+----------+-------+--------+------> FAIL/MUTE
                                            NPU error -> BYPASS
```

规则：

- codec I2C NACK、SD 失败、MP3 同步丢失且无法恢复：静音并锁存 fatal error；
- NPU timeout、error 或不正确 retired count：本块 vocal estimate 清零，继续输出原音并进入 BYPASS；
- NPU 后续自检恢复前不自动重新开启 STEM；
- FIFO underflow：输出零样本并递增计数；
- FIFO overflow：拒绝整条 64-bit frame 并递增计数；
- UART 输出必须限频，不能在实时热路径逐样本打印；
- 上电、fatal error 和文件结束均使用短渐变后静音；
- 文件正常结束后不自动循环，输出渐隐和最终统计。

UART 至少每秒报告：播放秒数、NPU block 最大/平均周期、FIFO 最小/当前水位、underflow、overflow、decoder error、NPU error 和 STEM target。

## 12. 验证策略

### 12.1 主机测试

- WM8960 寄存器表字段和写入顺序测试；
- 音频时钟 divider 和 I2S bit-exact testbench；
- FIFO 原子提交、full/empty、跨时钟和计数器测试；
- KEY0 bounce 波形、单次 toggle 和斜坡反向测试；
- 饱和混音的边界向量测试；
- MP3 格式接受/拒绝和损坏帧测试；
- C STFT/频带/量化输出与 Python reference 逐元素比较；
- 动态 task 输入和 NPU output 与 Python fixed-point reference 逐字节比较；
- C 掩码展开/iSTFT 与 Python reference 的误差及对齐测试；
- 实际 MP3 文件的离线端到端 PCM checksum 测试；
- 所有现有 NPU、FSBL 和 SD coldboot 回归保持通过。

### 12.2 分阶段实板验收

1. **Codec/tone**：PL 固定正弦波从板载扬声器输出；测得 LRCLK 约 44.1 kHz；codec 无 NACK。
2. **FIFO/PCM**：A9 写入确定性 PCM；扬声器连续输出；checksum、played count 和 FIFO level 符合预期。
3. **FatFs/WAV**：从 SD 读取已知 WAV 并播放，用来隔离文件系统和 MP3 解码问题。
4. **MP3 bypass**：`/music.mp3` 原音连续播放，格式日志正确，无 underflow/overflow。
5. **Dynamic NPU**：已知频谱 block 的 32 KiB 输出与主机 golden 逐字节相同。
6. **Full pipeline**：连续 STFT/NPU/iSTFT 运行，默认输出原音，UART deadline 全部满足。
7. **KEY0 STEM**：反复切换，约 20 ms 消抖加 30 ms 斜坡，无明显 click，状态与串口一致。
8. **Cold boot**：重新构建 FSBL + 新 bitstream + player ELF 的 `BOOT.BIN`，SD 模式断电冷启动，无 JTAG 自动播放。
9. **Stability**：连续播放至少 30 分钟，`NPU error=0`、`underflow=0`、`overflow=0`、codec error=0。

### 12.3 首版通过标准

只有同时满足以下条件，才能称“链路跑起来”：

- 从 MicroSD 自主冷启动且无需 JTAG；
- 自动读取 `/music.mp3`；
- 板载扬声器持续播放；
- KEY0 可重复切换原音和去人声；
- 任一 NPU 故障会旁路原音而不是输出失控数据；
- 30 分钟内无 NPU、FIFO、codec 错误；
- 日志记录实际每阶段耗时和当前 16-frame 上下文限制。

该标准只证明产品数据通路和算力调度可行，不代表最终实时输入、最终音质或量产稳定性已经完成。

## 13. 预期实施顺序

1. 音频 CSR/FIFO/按钮/mixer/I2S 的 RTL 和 testbench；
2. 44.1 kHz Clocking Wizard、WM8960 寄存器表和约束；
3. 集成到现有 Zynq block design，综合、实现和静态时序检查；
4. tone-only SD 镜像及实板扬声器验证；
5. standalone FatFs、PCM/WAV 和 MP3 bypass 播放；
6. 主机可验证的 C STFT/iSTFT 和模型数据契约；
7. resident NPU streaming API 和动态 block golden 测试；
8. 全链路播放器、按钮和故障降级；
9. 冷启动与 30 分钟稳定性验收；
10. 基于实测结果决定是否进入状态化模型和 WM8960 ADC 实时输入阶段。
