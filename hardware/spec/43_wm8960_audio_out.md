# WM8960 44.1 kHz Speaker Output Contract

状态：P5 实现输入。本文只覆盖首版 PL-I2S 到板载扬声器的播放路径，不启用
ADC、耳机、line-out 或 codec PLL。

## Clock Contract

- PL 输出 MCLK `11.289602856 MHz`，WM8960 使用 slave mode；
- `R4.CLKSEL=0`、`SYSCLKDIV=00`、`DACDIV=000`，因此
  `SYSCLK=MCLK`、DAC sample rate=`SYSCLK/256`，约 `44.100011 kHz`；
- PL 输出 64 BCLK/sample，BCLK 约 `2.822400714 MHz`；slave mode 下
  `R8.BCLKDIV` 不参与生成 BCLK；
- `R8.DCLKDIV=111`，class-D switching clock 为 `SYSCLK/16`，约
  `705.600179 kHz`，位于数据手册建议的 700-800 kHz 范围；
- `R26.PLL_EN=0`，R52-R55 不写入。

依据为 WM8960 Rev 4.4 的 Figure 36、Tables 39-41。厂商例程的
`R4=0x005` 是 48 kHz/PLL 配置，不得用于此路径。

## Register Sequence

I2C 7-bit address 为 `0x1A`，每个 control word 编码为
`{register[6:0], data[8:0]}`。任一地址或数据字节 NACK 都立即停止序列、保持
DAC mute，并锁存当前表索引。

| Index | Write | Fields and purpose |
|---:|---:|---|
| 0 | `R15=0x000` | software reset |
| 1 | `R25=0x1C0` | `VMIDSEL=11`, `VREF=1`; 5k fast start, input/ADC/mic off |
| - | wait 50 ms | VMID settle |
| 2 | `R25=0x0C0` | `VMIDSEL=01`, `VREF=1`; playback divider |
| 3 | `R5=0x008` | `DACMU=1`; setup remains muted |
| 4 | `R6=0x008` | `DACSMM=1`, `DACMR=0`; gradual fast unmute |
| 5 | `R4=0x000` | MCLK direct, SYSCLK/1, DACDIV/1 |
| 6 | `R7=0x00A` | `MS=0`, normal polarity, 24-bit I2S |
| 7 | `R8=0x1C0` | DCLK=SYSCLK/16; BCLK divider ignored in slave mode |
| 8 | `R34=0x100` | left DAC into left output mixer |
| 9 | `R37=0x100` | right DAC into right output mixer |
| 10 | `R47=0x00C` | left/right output mixers enabled; mic mixers off |
| 11 | `R40=0x079` | left speaker volume 0 dB, update deferred |
| 12 | `R41=0x179` | right speaker volume 0 dB, update both channels |
| 13 | `R51=0x080` | required reserved bit, DC/AC boost 0 dB |
| 14 | `R26=0x198` | DAC L/R and SPKL/SPKR buffers on; PLL/HP/OUT3 off |
| 15 | `R49=0x0F7` | both class-D outputs on after valid SYSCLK/DCLK |
| - | wait 10 ms | speaker output stage settle |
| 16 | `R5=0x000` | clear DAC mute using configured soft-unmute |

The power and speaker-enable ordering follows WM8960 Rev 4.4 Tables 26 and 28:
`SPK_OP_EN` is not asserted until SYSCLK and DCLKDIV are valid. The external PL
sample mixer starts at gain zero, so the codec unmute does not bypass the PL's
30 ms transition control.

## Failure Contract

- `codec_done=1` only after index 16 ACKs;
- `codec_error=1` latches the first failed index until reset or explicit reinit;
- `codec_muted=1` from reset through the complete sequence and on any failure;
- no transaction is retried automatically; software can request an explicit reinit;
- SCL and SDA outputs are open-drain enables and must use board pull-ups.
