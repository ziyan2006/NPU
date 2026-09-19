# Navigator Z7020 音频上板验收记录

状态：**PENDING — 尚未执行四阶段实板验收，不得标记 PASS。**

本记录用于第三方复刻的正点原子领航者 V3.7 兼容板、`xc7z020clg400-2`、
双颗 `NT5CC256M16EP-EK` 和板载 WM8960。最终验收必须从 MicroSD 冷启动，
JTAG 断开；串口仅用于观察，不参与启动。

## 候选版本

在实际测试当天填写，所有值必须来自同一次干净构建：

| 项目 | 实测值 |
|---|---|
| Git commit | `PENDING` |
| Tone `BOOT.BIN` SHA-256 | `PENDING` |
| WAV `BOOT.BIN` SHA-256 | `PENDING` |
| MP3 Bypass `BOOT.BIN` SHA-256 | `PENDING` |
| FullStem `BOOT.BIN` SHA-256 | `PENDING` |
| 开发板/底板识别 | `PENDING` |
| UART 端口 | `PENDING` |
| 测试日期和操作者 | `PENDING` |

从各模式的 `hardware/build/navigator_audio_boot/<mode>/manifest.json` 复制 Git commit，
从同目录 `BOOT.BIN` 重新计算 SHA-256，不手工猜测或沿用旧构建值。

## 分阶段结果

必须严格按顺序执行；任何一级失败后停止，不测试后续镜像。

### A. Tone / codec / I2S

- [ ] SD 根目录只放 Tone 版 `BOOT.BIN`，断开 JTAG 后冷启动。
- [ ] 串口出现 `BOOT`、`AUDIO_BUILD_MODE=TONE`、`CODEC_READY`、`PLAY`。
- [ ] 板载扬声器连续播放测试音，无爆音、明显失真或左右通道异常。
- [ ] 示波器/逻辑分析仪实测 LRCLK：`PENDING Hz`（目标约 44.1 kHz）。
- [ ] WM8960 初始化无 NACK/`FAIL CODEC`。

结论：`PENDING`

### B. WAV / SD / PCM

- [ ] SD 根目录放 WAV 版 `BOOT.BIN` 与 `test.wav`。
- [ ] `test.wav` 为 44.1 kHz、16-bit、双声道 PCM，长度足够完成 FIFO 预填充。
- [ ] 冷启动后完整播放，串口无 `FAIL`，听感速度和声道正确。
- [ ] 记录确定性测试文件 SHA-256：`PENDING`。

结论：`PENDING`

### C. MP3 Bypass

- [ ] SD 根目录放 MP3 Bypass 版 `BOOT.BIN` 与 `music.mp3`。
- [ ] MP3 为 MPEG-1 Layer III、44.1 kHz、双声道；记录文件 SHA-256：`PENDING`。
- [ ] 冷启动后持续播放，串口无解码、FIFO、codec 错误。
- [ ] 听感无周期性断音，曲速和声道正确。

结论：`PENDING`

### D. FullStem / KEY0 / 30 分钟

- [ ] SD 根目录放 FullStem 版 `BOOT.BIN` 与同一份 `music.mp3`。
- [ ] JTAG 断开，电源完全断开后重新上电，串口进入 `PLAY`。
- [ ] 运行 `scripts/104_monitor_audio_uart.py`；连续 1800 个 PLAY 秒无日志缺口。
- [ ] 运行期间至少按 KEY0 两次，日志观察到至少两次 `stem` 目标变化。
- [ ] STEM 开关切换使用 30 ms ramp，主观听感无明显 click/pop。
- [ ] `uf=0`、`of=0`、`dec_err=0`、`npu_err=0`、`codec_err=0`。
- [ ] `deadline_miss=0` 且 `blk_us_max <= 92880`。
- [ ] 保存 JSONL 和 summary JSON；summary 输出 `AUDIO_BOARD_ACCEPTANCE: PASS`。

| 指标 | 最终值 |
|---|---|
| 连续 PLAY 秒数 | `PENDING` |
| STEM target transitions | `PENDING` |
| FIFO minimum | `PENDING` |
| block average / maximum us | `PENDING` |
| decode+frontend average / maximum us | `PENDING` |
| NPU average / maximum us | `PENDING` |
| backend+sink average / maximum us | `PENDING` |
| JSONL 路径 | `PENDING` |
| summary JSON 路径 | `PENDING` |

结论：`PENDING`

## 最终结论

只有 A、B、C、D 全部通过，才能把本节改成 `PASS`。离线仿真、静态镜像门禁、
绿灯或单次心跳均不能替代扬声器听音、KEY0 切换和 30 分钟实板验收。
