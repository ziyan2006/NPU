# SD MP3 STEM Speaker Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在领航者 Zynq-7020 复刻板上实现无需 JTAG 的 MicroSD 冷启动，读取 `/music.mp3`，经 A9 解码和现有 PL NPU 分离后从 WM8960 板载扬声器播放，并用 KEY0 在原音与去人声之间切换。

**Architecture:** Cortex-A9 standalone 应用负责 FatFs、MP3 解码、连续 STFT/iSTFT、任务 tensor 更新和调度；现有 `0x43C00000` NPU 只执行冻结的 16-frame mask 网络。新增 `0x43C10000` 音频 PL 外设负责 8192x64-bit 异步 FIFO、KEY0 消抖、消费端 Q1.15 混音、I2S TX、WM8960 初始化和测试音。

**Tech Stack:** SystemVerilog, AXI4-Lite, Xilinx XPM/BRAM, Vivado 2026.1 Tcl, Vitis 2026.1 standalone BSP, C11, xilffs/FatFs, pinned minimp3, pinned KissFFT float32, Python 3/NumPy/PyTorch, Icarus Verilog, GNU Arm A9 toolchain, Bootgen.

**Spec:** `docs/superpowers/specs/2026-09-17-sd-mp3-stem-speaker-bringup-design.md`

## Global Constraints

- Target is `xc7z020clg400-2`, third-party Navigator clone, using the validated V3.7 PS/DDRx32 profile and two `NT5CC256M16EP-EK` devices.
- Preserve the verified NPU CSR aperture at `0x43C00000`/4 KiB and allocate audio CSR at `0x43C10000`/4 KiB.
- Input is exactly `/music.mp3`, MPEG-1 Layer III, 44.1 kHz stereo, CBR or VBR; reject every other sample rate, channel count, or layer without starting playback.
- Audio output is 44.1 kHz stereo, 24-bit I2S in 32-bit slots; PL is clock master and WM8960 is slave.
- KEY0 is `L14`, active-low, two-flop synchronized, debounced for 20 ms; reset state is STEM off.
- STEM gain is Q1.15 and ramps over exactly 1323 samples (30 ms at 44.1 kHz); a press during a ramp reverses from the current gain.
- FIFO entries are `{vocal_right[15:0], vocal_left[15:0], mix_right[15:0], mix_left[15:0]}` and depth is exactly 8192 frames.
- Model contract is `N_FFT=1024`, `hop=256`, periodic Hann, 513 bins, `legacy_log` 128 bands, 16 frames/block, INT12 input in NHWC8, Q1.11 output in NHWC8, and 250 Hz/44-band low-frequency protection.
- Task offsets, sizes, command count and tensor addresses come from generated metadata; software must not embed current offsets `49280` or `966784` as literals.
- Existing `npu_submit()` semantics and all current NPU/FSBL/SD cold-boot regressions remain unchanged.
- NPU errors produce a zero vocal estimate and latch bypass; SD/decoder/codec fatal errors ramp to mute; FIFO underflow outputs zero and overflow rejects the complete 64-bit frame.
- Generated build products, vendor reference material, model task payloads and licensed test music remain ignored by Git; pinned third-party source licenses are committed.
- The current 16-frame task resets causal model context at each block. This milestone proves the path, not final live-input quality; stateful history or overlap-and-discard remains a later project.

---

## File Map

### PL audio subsystem

- `hardware/rtl/audio/audio_regs_pkg.sv`: canonical register offsets, ID, status/control bits and FIFO frame packing.
- `hardware/rtl/audio/audio_axi_csr.sv`: AXI4-Lite slave, atomic two-write frame staging, W1C counters and control/status crossing endpoints.
- `hardware/rtl/audio/audio_async_fifo.sv`: 8192x64 dual-clock FIFO using inferred block RAM and Gray-coded pointers.
- `hardware/rtl/audio/audio_key_debounce.sv`: synchronizer, 20 ms stable-level filter and one-cycle press event.
- `hardware/rtl/audio/audio_stem_mixer.sv`: 1323-sample Q1.15 ramp, reversible target and saturated stereo subtraction.
- `hardware/rtl/audio/audio_i2s_tx.sv`: 64-BCLK stereo frame serializer with 24 valid MSBs per 32-bit slot.
- `hardware/rtl/audio/audio_test_tone.sv`: deterministic phase-accumulator tone for board bring-up only.
- `hardware/rtl/audio/wm8960_i2c_master.sv`: open-drain write-only I2C transaction engine with ACK result.
- `hardware/rtl/audio/wm8960_init.sv`: declarative 44.1 kHz slave-mode register table, mute/unmute sequencing and latched NACK.
- `hardware/rtl/audio/audio_out_axi.sv`: integration top joining AXI, FIFO, button, mixer, tone, codec and I2S domains.
- `hardware/rtl/audio/audio_rtl.f`: ordered RTL source list.
- `hardware/rtl/tb/tb_audio_*.sv`: focused self-checking simulations.
- `hardware/boards/alientek_navigator_z7020/audio_out_v37.xdc`: only the 50 MHz, KEY0 and WM8960 pins used by this design.

### A9 application

- `software/audio_player/include/audio_hw.h` and `src/audio_hw.c`: typed CSR driver and atomic frame writer.
- `software/audio_player/include/pcm_ring.h` and `src/pcm_ring.c`: bounded stereo `int16_t` PCM ring.
- `software/audio_player/include/wav_source.h` and `src/wav_source.c`: 44.1 kHz/16-bit/stereo RIFF parser used for staged validation.
- `software/audio_player/include/sd_mp3_source.h` and `src/sd_mp3_source.c`: FatFs-backed stream, ID3 skip, minimp3 frame sync and strict format check.
- `software/audio_player/include/stem_frontend.h` and `src/stem_frontend.c`: continuous STFT, filterbank analysis and NHWC8 INT12 packing.
- `software/audio_player/include/stem_backend.h` and `src/stem_backend.c`: Q1.11 mask parsing, synthesis mapping, complex mask application and continuous OLA.
- `software/audio_player/include/stem_npu_session.h` and `src/stem_npu_session.c`: resident task validation, range-specific cache maintenance and block execution.
- `software/audio_player/include/player.h`, `src/player.c`, `src/player_main.c`: cooperative state machine, prefill, degradation, telemetry and EOF fade.
- `software/audio_player/third_party/minimp3/` and `third_party/kissfft/`: pinned upstream sources, license and SHA-256 provenance.
- `software/audio_player/generated/`: ignored task payload, task metadata and deterministic filterbank constants.
- `software/audio_player/tests/`: native C unit/integration tests and generated public-domain signal fixtures.

### Build and verification

- `scripts/_test_audio_rtl.py`: Icarus regression runner for all audio testbenches.
- `scripts/_test_audio_player.py`: host C build/test runner.
- `scripts/93_package_audio_out_ip.tcl`: package `audio_out_axi` as reusable AXI IP.
- `scripts/94_create_navigator_audio_soc.tcl`: recreate the Zynq/NPU/audio block design and force both CSR address segments.
- `scripts/95_implement_navigator_audio_soc.tcl`: synthesize, implement, check DRC/timing and export bit/XSA.
- `scripts/96_check_navigator_audio_soc.py`: audit part, pins, clocks, address map, utilization, route and timing reports.
- `scripts/97_generate_audio_vectors.py`: generate WAV/PCM/FFT/OLA public-domain golden vectors.
- `scripts/98_generate_stem_constants.py`: generate filterbank C constants and hashes from the Python reference.
- `scripts/99_generate_stem_task_payload.py`: generate resident task C payload and metadata header from task manifests.
- `scripts/100_create_audio_vitis_workspace.py`: create standalone domain with `xilffs` and application source imports.
- `scripts/101_build_audio_player.ps1`: regenerate assets and build WAV, MP3-bypass and full-STEM ELFs.
- `scripts/102_build_navigator_audio_boot.ps1`: build/read back staged and final `BOOT.BIN` images.
- `scripts/103_test_navigator_audio_boot.ps1`: static partition, hash, XSA address and legacy regression gate.
- `scripts/104_monitor_audio_uart.py`: parse UART telemetry and enforce the 30-minute acceptance counters.

---

## Milestone 1: PL Audio Subsystem

### Task 1: Freeze the Audio CSR Contract and Atomic Frame Commit

**Files:**
- Create: `hardware/rtl/audio/audio_regs_pkg.sv`
- Create: `hardware/rtl/audio/audio_axi_csr.sv`
- Create: `hardware/rtl/tb/tb_audio_axi_csr.sv`
- Create: `scripts/_test_audio_rtl.py`
- Modify: `hardware/rtl/README.md`

**Interfaces:**
- Consumes: AXI4-Lite 32-bit single-beat reads/writes on `s_axi_aclk`; synchronized FIFO status/counters from later tasks.
- Produces: `frame_wr_valid`, `frame_wr_data[63:0]`, `control_enable`, `control_soft_reset`, `control_codec_reinit`, `control_tone_enable`; registers `IP_ID=32'h31445541`, `VERSION=32'h00010000`, offsets `0x00..0x34` exactly as the spec.

- [x] **Step 1: Write the failing CSR simulation**

  Test reset ID/version/capacity, independent AW/W arrival, byte strobes, read back, MIX staging, atomic commit only on `VOCAL_FRAME`, full-FIFO rejection, one overflow increment, and W1C underflow/overflow. The central assertion is:

  ```systemverilog
  axi_write(12'h010, 32'h2222_1111, 4'hf);
  assert (!frame_wr_valid);
  axi_write(12'h014, 32'h4444_3333, 4'hf);
  assert (frame_wr_valid && frame_wr_data == 64'h4444_3333_2222_1111);
  ```

- [x] **Step 2: Run the focused test and record the red state**

  ```powershell
  python scripts/_test_audio_rtl.py --case csr
  ```

  Expected: nonzero exit because `audio_regs_pkg.sv` and `audio_axi_csr.sv` do not exist.

- [x] **Step 3: Implement the package and AXI slave**

  Use registered AXI ready/response channels, honor `WSTRB`, stage `MIX_FRAME`, and make a `VOCAL_FRAME` write the only commit point. A full FIFO must emit no `frame_wr_valid`; it increments the saturating overflow counter once. Snapshot synchronized counters for readback and encode `STEM_STATE` as `{gain_q15[15:0], 13'b0, key_pressed, ramping, target}`.

- [x] **Step 4: Run CSR and legacy RTL regressions**

  ```powershell
  python scripts/_test_audio_rtl.py --case csr
  python scripts/_test_npu_csr.py
  ```

  Expected: both print `PASS`; no existing NPU register definition changes.

- [x] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio/audio_regs_pkg.sv hardware/rtl/audio/audio_axi_csr.sv hardware/rtl/tb/tb_audio_axi_csr.sv hardware/rtl/README.md scripts/_test_audio_rtl.py
  git commit -m "feat(audio): define AXI audio register contract"
  ```

### Task 2: Implement the 8192-Frame Asynchronous FIFO

**Files:**
- Create: `hardware/rtl/audio/audio_async_fifo.sv`
- Create: `hardware/rtl/tb/tb_audio_async_fifo.sv`
- Modify: `scripts/_test_audio_rtl.py`

**Interfaces:**
- Consumes: `wr_clk`, `wr_reset_n`, `wr_valid`, `wr_data[63:0]`, `rd_clk`, `rd_reset_n`, `rd_ready`.
- Produces: `wr_full`, write-domain `wr_level[13:0]`, `rd_valid`, `rd_data[63:0]`, read-domain `rd_empty`, `rd_level[13:0]`; capacity is exactly 8192 and accepted transfers are `wr_valid && !wr_full` / `rd_ready && rd_valid`.

- [x] **Step 1: Add a failing dual-clock FIFO test**

  Drive 100 MHz writes and 11.2896 MHz reads with unrelated reset release. Verify ordering across pointer wrap, exact full/empty transitions, rejected full writes, stable head data under backpressure, and at least 20 randomized clock-phase seeds.

- [x] **Step 2: Confirm the red test**

  ```powershell
  python scripts/_test_audio_rtl.py --case fifo
  ```

  Expected: compile failure naming missing `audio_async_fifo`.

- [x] **Step 3: Implement the FIFO**

  Use 14-bit binary pointers with an extra wrap bit, Gray-code synchronization through two flops in each direction, and `(* ram_style = "block" *) logic [63:0] memory [0:8191]`. Compute full by comparing the next write Gray pointer with the synchronized read pointer whose top two bits are inverted; compute levels from synchronized binary pointers. Reset each domain locally and never combine asynchronous reset state combinationally.

- [x] **Step 4: Run focused and complete audio RTL tests**

  ```powershell
  python scripts/_test_audio_rtl.py --case fifo
  python scripts/_test_audio_rtl.py
  ```

  Expected: `audio_async_fifo: PASS`, no assertion or Icarus warning promoted to failure.

- [x] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio/audio_async_fifo.sv hardware/rtl/tb/tb_audio_async_fifo.sv scripts/_test_audio_rtl.py
  git commit -m "feat(audio): add asynchronous frame FIFO"
  ```

### Task 3: Add KEY0 Debounce and the Consumption-Time STEM Mixer

**Files:**
- Create: `hardware/rtl/audio/audio_key_debounce.sv`
- Create: `hardware/rtl/audio/audio_stem_mixer.sv`
- Create: `hardware/rtl/tb/tb_audio_key_mixer.sv`
- Modify: `scripts/_test_audio_rtl.py`

**Interfaces:**
- Consumes: KEY0 active-low asynchronous input, `clk_audio`, one `sample_tick` per stereo frame, and FIFO `{mix_l,mix_r,vocal_l,vocal_r}` values.
- Produces: one-cycle `press_event`, `target_stem`, `ramping`, `gain_q15[15:0]`, saturated `out_left/right[15:0]`.

- [x] **Step 1: Write the failing bounce/ramp/saturation test**

  Replay a press waveform with 1-10 ms chatter, 21 ms stable low, held-low time and release chatter; demand one toggle. Check gain reaches 32767 on sample 1323, reverses continuously at sample 400, and these vectors: `32767-(-32768)` saturates to `32767`, `-32768-32767` saturates to `-32768`.

- [x] **Step 2: Confirm the red test**

  ```powershell
  python scripts/_test_audio_rtl.py --case key_mixer
  ```

  Expected: compile failure naming the two missing modules.

- [x] **Step 3: Implement debounce and exact ramp accumulation**

  Count `ceil(clk_audio_hz/50)` identical synchronized samples before accepting a level. For the ramp, use a fractional accumulator so 1323 ticks cover the full 0..32767 range without a final jump: each tick adds/subtracts quotient 24 and distributes remainder 1015; clamp at the target. Compute signed 32-bit `mix - ((vocal*gain + 16384) >>> 15)` and saturate to int16.

- [x] **Step 4: Run the test suite**

  ```powershell
  python scripts/_test_audio_rtl.py --case key_mixer
  python scripts/_test_audio_rtl.py
  ```

  Expected: all bounce, reversal and arithmetic assertions pass.

- [x] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio/audio_key_debounce.sv hardware/rtl/audio/audio_stem_mixer.sv hardware/rtl/tb/tb_audio_key_mixer.sv scripts/_test_audio_rtl.py
  git commit -m "feat(audio): add debounced STEM mixer control"
  ```

### Task 4: Build the I2S Transmitter and Deterministic Test Tone

**Files:**
- Create: `hardware/rtl/audio/audio_i2s_tx.sv`
- Create: `hardware/rtl/audio/audio_test_tone.sv`
- Create: `hardware/rtl/tb/tb_audio_i2s_tx.sv`
- Modify: `scripts/_test_audio_rtl.py`

**Interfaces:**
- Consumes: 11.2896 MHz `mclk`, reset, signed stereo int16 frame, `frame_valid`, and tone enable/amplitude/phase step.
- Produces: `aud_mclk`, BCLK=MCLK/4, DAC LRCLK=BCLK/64, I2S `aud_dacdat`, `frame_ready` at a stereo boundary and `sample_tick` once per consumed frame.

- [ ] **Step 1: Add a bit-exact failing I2S test**

  Queue `left=16'h1234`, `right=16'hfedc` and assert the wire has one-bit I2S delay, then `24'h123400` and `24'hfedc00` in the MSBs of the two 32-bit slots. Count exactly 64 BCLK rising edges and four MCLK periods per BCLK period. Verify underflow serializes zero without changing frame timing.

- [ ] **Step 2: Confirm the red test**

  ```powershell
  python scripts/_test_audio_rtl.py --case i2s
  ```

  Expected: compile failure naming `audio_i2s_tx`.

- [ ] **Step 3: Implement serializer and phase-accumulator tone**

  Generate BCLK and LRCLK only from MCLK counters, latch one stereo frame at the frame boundary, left-shift int16 samples by eight, and serialize MSB first. Tone mode uses a 32-bit phase accumulator and a committed 256-entry signed sine ROM; default phase step is `round(1000*2^32/44100)=97391549`, amplitude defaults to 4096.

- [ ] **Step 4: Run bit-exact tests**

  ```powershell
  python scripts/_test_audio_rtl.py --case i2s
  python scripts/_test_audio_rtl.py
  ```

  Expected: serialized frame bits, clock ratios, zero-underflow and tone period all pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio/audio_i2s_tx.sv hardware/rtl/audio/audio_test_tone.sv hardware/rtl/tb/tb_audio_i2s_tx.sv scripts/_test_audio_rtl.py
  git commit -m "feat(audio): add 44.1 kHz I2S transmit path"
  ```

### Task 5: Implement and Verify the WM8960 Register Sequence

**Files:**
- Create: `hardware/rtl/audio/wm8960_i2c_master.sv`
- Create: `hardware/rtl/audio/wm8960_init.sv`
- Create: `hardware/rtl/tb/tb_wm8960_init.sv`
- Create: `hardware/spec/43_wm8960_audio_out.md`
- Modify: `scripts/_test_audio_rtl.py`

**Interfaces:**
- Consumes: 100 MHz AXI clock/reset, `codec_reinit`, open-drain SDA input, and the generated MCLK-stable indication.
- Produces: open-drain `scl_drive_low`/`sda_drive_low`, `codec_done`, `codec_error`, `codec_muted`, `config_index[7:0]`, `last_ack`; address is fixed at `7'h1a`.

- [ ] **Step 1: Encode the expected transaction test before RTL**

  The testbench I2C slave records `{register[6:0],data[8:0]}` and compares it against this exact array. `WAIT` entries are sequencer delays rather than I2C writes. Inject a NACK at every write index in turn and assert `codec_error=1`, `codec_done=0`, `codec_muted=1`, with no later writes.

  | Step | Register/value | Required effect |
  |---:|---:|---|
  | 0 | `R15=0x000` | reset all registers |
  | 1 | `R25=0x1C0` | VREF on, VMID 2x5k fast start, ADC/input/mic off |
  | 2 | `WAIT=50 ms` | settle VMID before changing divider |
  | 3 | `R25=0x0C0` | VREF on, VMID 2x50k playback mode |
  | 4 | `R5=0x008` | explicit DAC soft mute |
  | 5 | `R6=0x008` | gradual DAC soft-unmute mode, fast ramp |
  | 6 | `R4=0x000` | direct MCLK, SYSCLK/1, DACDIV=1 for 44.1 kHz |
  | 7 | `R7=0x00A` | slave, normal polarity, 24-bit I2S |
  | 8 | `R8=0x1C0` | class-D clock SYSCLK/16 = 705.6 kHz; BCLKDIV ignored in slave mode |
  | 9 | `R34=0x100` | left DAC to left output mixer |
  | 10 | `R37=0x100` | right DAC to right output mixer |
  | 11 | `R47=0x00C` | left and right output mixers on; mic paths off |
  | 12 | `R40=0x079` | stage left speaker PGA at 0 dB without update |
  | 13 | `R41=0x179` | right speaker PGA 0 dB and atomically update both channels |
  | 14 | `R51=0x080` | reserved field retained; DC/AC boost both 0 dB |
  | 15 | `R26=0x198` | DAC L/R and speaker L/R buffers on; headphones/OUT3/PLL off |
  | 16 | `R49=0x0F7` | enable both class-D outputs after valid SYSCLK/DCLK |
  | 17 | `WAIT=10 ms` | output stage settling while DAC remains muted |
  | 18 | `R5=0x000` | DAC soft-unmute; PL gain ramp still starts at zero |

- [ ] **Step 2: Confirm the red test**

  ```powershell
  python scripts/_test_audio_rtl.py --case wm8960
  ```

  Expected: compile failure naming `wm8960_init` and `wm8960_i2c_master`.

- [ ] **Step 3: Derive and implement the table from datasheet fields**

  Document every table row as register name, field values, 9-bit value and purpose in `43_wm8960_audio_out.md`, citing WM8960 Rev 4.4 Tables 26, 28, 39 and 40. Use a <=250 kHz open-drain I2C engine, sample ACK on the ninth clock, retry no transaction automatically, and latch the first NACK index until `codec_reinit` or reset. The direct-clock values `R4=0x000` and `R8=0x1C0` intentionally replace the vendor reference's 48-kHz PLL configuration.

- [ ] **Step 4: Run the sequence and full RTL regressions**

  ```powershell
  python scripts/_test_audio_rtl.py --case wm8960
  python scripts/_test_audio_rtl.py
  ```

  Expected: the exact table and every injected-NACK case pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio/wm8960_i2c_master.sv hardware/rtl/audio/wm8960_init.sv hardware/rtl/tb/tb_wm8960_init.sv hardware/spec/43_wm8960_audio_out.md scripts/_test_audio_rtl.py
  git commit -m "feat(audio): configure WM8960 speaker output"
  ```

### Task 6: Integrate, Package and Implement the Audio PL

**Files:**
- Create: `hardware/rtl/audio/audio_out_axi.sv`
- Create: `hardware/rtl/audio/audio_rtl.f`
- Create: `hardware/rtl/tb/tb_audio_out_axi.sv`
- Create: `hardware/boards/alientek_navigator_z7020/audio_out_v37.xdc`
- Create: `scripts/93_package_audio_out_ip.tcl`
- Create: `scripts/94_create_navigator_audio_soc.tcl`
- Create: `scripts/95_implement_navigator_audio_soc.tcl`
- Create: `scripts/96_check_navigator_audio_soc.py`
- Modify: `hardware/boards/alientek_navigator_z7020/README.md`

**Interfaces:**
- Consumes: Zynq GP0 AXI at 100 MHz, board 50 MHz clock on `U18`, reset, KEY0 `L14`, WM8960 SDA input; packaged NPU IP and validated V3.7 PS configuration.
- Produces: audio AXI segment `0x43C10000..0x43C10FFF`, MCLK `E19`, BCLK `M18`, DAC LRCLK `G18`, DACDAT `G17`, I2C SCL `E18`, SDA `F17`, plus bitstream/XSA under ignored `hardware/build/navigator_z7020_audio_export/`.

- [ ] **Step 1: Write the failing integration test and audit**

  The top test writes two FIFO frames, enables playback, toggles KEY0 and checks register counters against serialized samples. The Python audit must fail unless `xc7z020clg400-2`, both address segments, exact pins, 8192-depth BRAM inference, routed status, no critical DRC and nonnegative WNS are present.

- [ ] **Step 2: Confirm simulation and audit are red**

  ```powershell
  python scripts/_test_audio_rtl.py --case top
  python scripts/96_check_navigator_audio_soc.py
  ```

  Expected: missing top and missing exported reports cause nonzero exits.

- [ ] **Step 3: Wire the top and recreate the block design**

  Instantiate the existing `stem_npu_1_0` unchanged and the new audio IP behind GP0 SmartConnect. Create the 11.289602856 MHz clock with Clocking Wizard `D=3`, `M=63.25`, `O=93.375` from the 50 MHz pin, connect AXI reset through `proc_sys_reset`, and force address assignments with Tcl assertions. The XDC contains only these seven audio/control pins and creates the 20 ns input clock; it must not import the vendor-wide XDC.

- [ ] **Step 4: Simulate, implement and audit**

  ```powershell
  python scripts/_test_audio_rtl.py
  vivado -mode batch -source scripts/93_package_audio_out_ip.tcl
  vivado -mode batch -source scripts/94_create_navigator_audio_soc.tcl
  vivado -mode batch -source scripts/95_implement_navigator_audio_soc.tcl
  python scripts/96_check_navigator_audio_soc.py
  python scripts/44_check_navigator_v37.py
  ```

  Expected: all commands pass; audit reports audio CSR at `0x43C10000`, NPU CSR unchanged, routed design and timing closure.

- [ ] **Step 5: Commit**

  ```powershell
  git add hardware/rtl/audio hardware/rtl/tb/tb_audio_out_axi.sv hardware/boards/alientek_navigator_z7020/audio_out_v37.xdc hardware/boards/alientek_navigator_z7020/README.md scripts/93_package_audio_out_ip.tcl scripts/94_create_navigator_audio_soc.tcl scripts/95_implement_navigator_audio_soc.tcl scripts/96_check_navigator_audio_soc.py
  git commit -m "build(audio): integrate WM8960 output subsystem"
  ```

## Milestone 2: Standalone Bypass Player

### Task 7: Add the Host-Testable Audio Driver, PCM Ring and WAV Parser

**Files:**
- Create: `software/audio_player/include/audio_hw.h`
- Create: `software/audio_player/src/audio_hw.c`
- Create: `software/audio_player/include/pcm_ring.h`
- Create: `software/audio_player/src/pcm_ring.c`
- Create: `software/audio_player/include/wav_source.h`
- Create: `software/audio_player/src/wav_source.c`
- Create: `software/audio_player/tests/test_audio_foundation.c`
- Create: `scripts/97_generate_audio_vectors.py`
- Create: `scripts/_test_audio_player.py`

**Interfaces:**
- Produces: `audio_hw_init(audio_hw_t*, uintptr_t)`, `audio_hw_space(const audio_hw_t*)`, `audio_hw_write_frame(audio_hw_t*, const audio_frame_t*)`, `pcm_ring_push/peek/drop`, and `wav_source_open/read`; `audio_frame_t` contains four `int16_t` members in mix-L/R, vocal-L/R order.
- Consumes: memory-mapped register words, byte buffer/file callback, and generated stereo WAV fixtures.

- [ ] **Step 1: Write foundation tests**

  Test IP/address validation, compile-time `_Static_assert(XPAR_AUDIO_OUT_AXI_0_BASEADDR == 0x43C10000U)` in the target build, MIX-before-VOCAL write ordering, no write when full, ring wrap/backpressure, RIFF chunks with odd padding, and rejection of mono/48 kHz/24-bit WAV.

- [ ] **Step 2: Confirm the red host build**

  ```powershell
  python scripts/97_generate_audio_vectors.py --check
  python scripts/_test_audio_player.py --case foundation
  ```

  Expected: vector check and C compile fail because generators/modules are absent.

- [ ] **Step 3: Implement minimal portable modules and generator**

  Keep OS/BSP calls behind `wav_read_fn(void *ctx, void *dst, size_t bytes)`. `audio_hw_write_frame` first checks `FIFO_LEVEL < FIFO_CAPACITY`, packs `{right,left}` into two 32-bit writes, and only then updates software submitted count. Use power-of-two ring capacity with monotonically increasing 32-bit producer/consumer counters.

- [ ] **Step 4: Run native tests**

  ```powershell
  python scripts/97_generate_audio_vectors.py
  python scripts/97_generate_audio_vectors.py --check
  python scripts/_test_audio_player.py --case foundation
  ```

  Expected: generated fixture hashes are stable and `audio foundation: PASS`.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/include/audio_hw.h software/audio_player/src/audio_hw.c software/audio_player/include/pcm_ring.h software/audio_player/src/pcm_ring.c software/audio_player/include/wav_source.h software/audio_player/src/wav_source.c software/audio_player/tests/test_audio_foundation.c scripts/97_generate_audio_vectors.py scripts/_test_audio_player.py
  git commit -m "feat(player): add audio sink ring and WAV parser"
  ```

### Task 8: Create the Vitis Standalone WAV/Tone Player

**Files:**
- Create: `software/audio_player/include/player_platform.h`
- Create: `software/audio_player/src/player_platform_vitis.c`
- Create: `software/audio_player/src/player_main.c`
- Create: `software/audio_player/README.md`
- Create: `scripts/100_create_audio_vitis_workspace.py`
- Create: `scripts/101_build_audio_player.ps1`
- Modify: `software/NAVIGATOR_BRINGUP.md`

**Interfaces:**
- Consumes: exported audio XSA, standalone Cortex-A9 BSP with `xilffs`, `/test.wav`, UART0, `audio_hw` and `wav_source`.
- Produces: ignored `hardware/build/navigator_audio_player/tone_player.elf` and `wav_player.elf`; UART states `BOOT`, `CODEC_READY`, `MOUNT`, `OPEN_WAV`, `PREFILL`, `PLAY`, `EOF`, `FAIL`.

- [ ] **Step 1: Add a failing workspace/build contract test**

  Extend `_test_audio_player.py --case vitis_layout` to require `xilffs` in the generated domain configuration, require all source imports, reject XSA without the audio base address, and inspect the ELF map for `player_main`, `f_mount`, `audio_hw_write_frame` and no unresolved symbols.

- [ ] **Step 2: Confirm the build contract is red**

  ```powershell
  python scripts/_test_audio_player.py --case vitis_layout
  powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 -Mode Wav
  ```

  Expected: missing workspace script/app sources produce nonzero exits.

- [ ] **Step 3: Build the cooperative WAV playback loop**

  `Tone` mode initializes UART/audio, waits for codec success, sets `TONE_CONTROL`, enables tone and reports counters without mounting SD. `Wav` mode mounts `0:/`, opens `0:/test.wav`, fills exactly 8192 FIFO frames before enabling audio, and reports `WAV_TOO_SHORT` without playback if EOF occurs first. Feed vocal samples as zero, poll FIFO space in batches, fade at EOF, and print one rate-limited status line per second with played frames, FIFO level/minimum, underflow, overflow and codec status.

- [ ] **Step 4: Build and statically inspect the ELF**

  ```powershell
  python scripts/_test_audio_player.py --case vitis_layout
  powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 -Mode Tone -Clean
  powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 -Mode Wav -Clean
  arm-none-eabi-nm hardware/build/navigator_audio_player/wav_player.elf | Select-String 'player_main|f_mount|audio_hw_write_frame'
  ```

  Expected: build passes and each required symbol is present.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/include/player_platform.h software/audio_player/src/player_platform_vitis.c software/audio_player/src/player_main.c software/audio_player/README.md software/NAVIGATOR_BRINGUP.md scripts/100_create_audio_vitis_workspace.py scripts/101_build_audio_player.ps1 scripts/_test_audio_player.py
  git commit -m "feat(player): build standalone WAV playback image"
  ```

### Task 9: Pin minimp3 and Implement Strict MP3 Bypass Playback

**Files:**
- Create: `software/audio_player/third_party/minimp3/minimp3.h`
- Create: `software/audio_player/third_party/minimp3/LICENSE`
- Create: `software/audio_player/third_party/minimp3/UPSTREAM.json`
- Create: `software/audio_player/tests/fixtures/test_44100_stereo_cbr.mp3`
- Create: `software/audio_player/tests/fixtures/test_44100_stereo_vbr.mp3`
- Create: `software/audio_player/include/sd_mp3_source.h`
- Create: `software/audio_player/src/sd_mp3_source.c`
- Create: `software/audio_player/tests/test_mp3_source.c`
- Modify: `software/audio_player/src/player_main.c`
- Modify: `scripts/97_generate_audio_vectors.py`
- Modify: `scripts/_test_audio_player.py`
- Modify: `scripts/101_build_audio_player.ps1`

**Interfaces:**
- Produces: `mp3_source_open(mp3_source_t*, const mp3_io_t*)`, `mp3_source_decode(mp3_source_t*, int16_t *interleaved, size_t frame_capacity, size_t *frames_out)`, and explicit errors `MP3_E_LAYER`, `MP3_E_RATE`, `MP3_E_CHANNELS`, `MP3_E_SYNC`, `MP3_E_IO`.
- Consumes: sequential FatFs callback for `0:/music.mp3`; ID3v2 synchsafe size; pinned minimp3 frame decoder.

- [ ] **Step 1: Add failing format and corruption tests**

  Generate two 3-second synthetic fixtures with FFmpeg 9.0.1: left is 1 kHz and right is 2 kHz at -12 dBFS. Use `-ar 44100 -ac 2 -codec:a libmp3lame -b:a 128k` for CBR and `-q:a 4` for VBR, commit both files because their inputs are generated tones, and record their SHA-256. Test CBR, VBR, ID3 prefix, buffer-split sync words, recoverable corrupted frame, and rejection of MPEG-2/mono/48 kHz.

- [ ] **Step 2: Confirm the decoder tests are red**

  ```powershell
  python scripts/_test_audio_player.py --case mp3
  ```

  Expected: missing minimp3 and `sd_mp3_source` produce compile failure.

- [ ] **Step 3: Vendor exact upstream and implement bounded streaming**

  Pin `https://github.com/lieff/minimp3.git` commit `ea99364f61c14656440e8d77e9c233ccf3124633`; record URL, commit, vendored file SHA-256 and retrieval date in `UPSTREAM.json`, and preserve its CC0 license. Maintain a compacting 16 KiB byte window, skip ID3v2 before sync search, require the first valid frame to be MPEG-1 Layer III/stereo/44.1 kHz, and reject any later format change. Limit resync search to 64 KiB before `MP3_E_SYNC`.

- [ ] **Step 4: Test and build bypass ELF**

  ```powershell
  python scripts/_test_audio_player.py --case mp3
  powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 -Mode Mp3Bypass -Clean
  ```

  Expected: host decoder PCM checksum matches the generator and `mp3_bypass_player.elf` builds with `xilffs`.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/third_party/minimp3 software/audio_player/tests/fixtures/test_44100_stereo_cbr.mp3 software/audio_player/tests/fixtures/test_44100_stereo_vbr.mp3 software/audio_player/include/sd_mp3_source.h software/audio_player/src/sd_mp3_source.c software/audio_player/tests/test_mp3_source.c software/audio_player/src/player_main.c scripts/97_generate_audio_vectors.py scripts/_test_audio_player.py scripts/101_build_audio_player.ps1
  git commit -m "feat(player): stream MP3 to audio bypass path"
  ```

## Milestone 3: STEM Streaming Pipeline

### Task 10: Generate Filterbank and Task Metadata Constants

**Files:**
- Create: `scripts/98_generate_stem_constants.py`
- Create: `scripts/99_generate_stem_task_payload.py`
- Create: `software/audio_player/include/stem_contract.h`
- Create: `software/audio_player/tests/test_stem_generators.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: ignored `software/audio_player/generated/stem_filterbank.{h,c}`, `stem_task_metadata.h`, `stem_task_payload.{h,c}` with symbols `stem_analysis[513][128]`, `stem_synthesis[128][513]`, `STEM_TASK_INPUT_OFFSET`, `STEM_TASK_OUTPUT_OFFSET`, byte sizes, command count, task length and SHA-256 strings.
- Consumes: `scripts/09_target_model.py` `make_analysis_matrix(..., layout="legacy_log")`, `hardware/generated/bott2_mir1k_v1_program/{program.json,task_image.json,task_image.bin}`, tensor descriptors and entry tensor IDs.

- [ ] **Step 1: Write failing generator tests**

  Parse generated C with NumPy and require shapes 513x128 and 128x513, little-endian float32, no dead analysis band, synthesis row sums within `1e-7`, matching SHA-256 comments, input 32768 bytes, output 32768 bytes, and all derived ranges within the activation/task bounds. Assert generated source contains neither literal `49280` nor `966784` in generator logic.

- [ ] **Step 2: Confirm the generator tests are red**

  ```powershell
  python software/audio_player/tests/test_stem_generators.py
  ```

  Expected: imports/files for both generators are missing.

- [ ] **Step 3: Implement deterministic generation and checks**

  Resolve entry tensor offsets by parsing task tensor descriptors plus the activation section, never by copying current output locations. Emit `_Static_assert` checks for 64-byte task alignment, tensor extents, `C=2/4`, `F=128`, `T=16`, NHWC8 storage, and the block constants `4096`, `1024`, `256`.

- [ ] **Step 4: Generate twice and compare**

  ```powershell
  python scripts/98_generate_stem_constants.py
  python scripts/99_generate_stem_task_payload.py
  python scripts/98_generate_stem_constants.py --check
  python scripts/99_generate_stem_task_payload.py --check
  python software/audio_player/tests/test_stem_generators.py
  ```

  Expected: second generation is byte-identical and all metadata range checks pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add .gitignore scripts/98_generate_stem_constants.py scripts/99_generate_stem_task_payload.py software/audio_player/include/stem_contract.h software/audio_player/tests/test_stem_generators.py
  git commit -m "build(stem): generate filterbank and task contracts"
  ```

### Task 11: Pin KissFFT and Implement the Continuous Frontend

**Files:**
- Create: `software/audio_player/third_party/kissfft/kiss_fft.c`
- Create: `software/audio_player/third_party/kissfft/kiss_fft.h`
- Create: `software/audio_player/third_party/kissfft/kiss_fftr.c`
- Create: `software/audio_player/third_party/kissfft/kiss_fftr.h`
- Create: `software/audio_player/third_party/kissfft/LICENSES/BSD-3-Clause`
- Create: `software/audio_player/third_party/kissfft/UPSTREAM.json`
- Create: `software/audio_player/include/stem_frontend.h`
- Create: `software/audio_player/src/stem_frontend.c`
- Create: `software/audio_player/tests/test_stem_frontend.c`
- Modify: `scripts/97_generate_audio_vectors.py`
- Modify: `scripts/_test_audio_player.py`

**Interfaces:**
- Produces: `stem_spectrum_block_t { kiss_fft_cpx bins[2][16][513]; int64_t first_sample; }`, `stem_frontend_init(stem_frontend_t*)`, `stem_frontend_push(stem_frontend_t*, const int16_t *lr, size_t frames)`, `stem_frontend_block_ready`, and `stem_frontend_pack(stem_frontend_t*, int16_t nhwc8[16384], stem_spectrum_block_t*)`.
- Consumes: normalized `sample/32768.0f`, periodic Hann, 512-sample reflect-lookahead start policy, generated `stem_analysis`, input scale `0.07046897899364925f`.

- [ ] **Step 1: Generate goldens and write failing numerical tests**

  Use impulse, DC, alternating Nyquist, seeded stereo noise and a 20-block chirp. Compare each 1024-point complex spectrum, 513-bin magnitude, 128 bands and packed NHWC8 bytes to Python. Require lanes 2..7 zero and INT12 clipping to `[-2048,2047]`.

- [ ] **Step 2: Confirm the frontend test is red**

  ```powershell
  python scripts/97_generate_audio_vectors.py --stem
  python scripts/_test_audio_player.py --case frontend
  ```

  Expected: C compile fails because KissFFT and frontend are absent.

- [ ] **Step 3: Vendor KissFFT and implement streaming analysis**

  Pin KissFFT tag `131.2.0`, commit `7bce4153c6bc8aba2db0e889e576f9d00505cbe1`, preserve BSD-3-Clause and record hashes. Preallocate all FFT configurations/workspaces at init; perform no heap allocation in `push`/`pack`. Logical frame `k` is centered at sample `k*256` and reads `[k*256-512, k*256+511]`; reflect negative startup indices. Produce 16 frames per 4096 new samples while retaining overlap and 512 samples of lookahead. Quantize with `lrintf(value/scale)` then saturate to signed 12-bit.

- [ ] **Step 4: Run numerical comparison**

  ```powershell
  python scripts/_test_audio_player.py --case frontend
  python scripts/_test_audio_player.py --case foundation
  ```

  Expected: maximum complex FFT error <=`2e-4`, band error <=`2e-5`, and packed tensor is byte-identical.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/third_party/kissfft software/audio_player/include/stem_frontend.h software/audio_player/src/stem_frontend.c software/audio_player/tests/test_stem_frontend.c scripts/97_generate_audio_vectors.py scripts/_test_audio_player.py
  git commit -m "feat(stem): implement continuous STFT frontend"
  ```

### Task 12: Add the Resident NPU Streaming Session API

**Files:**
- Modify: `software/include/npu_driver.h`
- Modify: `software/src/npu_driver.c`
- Modify: `software/tests/test_npu_driver.c`
- Create: `software/audio_player/include/stem_npu_session.h`
- Create: `software/audio_player/src/stem_npu_session.c`
- Create: `software/audio_player/tests/test_stem_npu_session.c`
- Modify: `scripts/_test_npu_driver.py`
- Modify: `scripts/_test_audio_player.py`

**Interfaces:**
- Produces: `npu_range_t { void *cpu_address; uint64_t physical_address; size_t bytes; }`, `npu_resident_prepare(npu_resident_t*, npu_device_t*, uint64_t, void*, size_t)`, `npu_resident_submit(npu_resident_t*, const npu_range_t *clean, const npu_range_t *invalidate, uint32_t tag, uint32_t watchdog)`, and `stem_npu_run_block(stem_npu_session_t*, const int16_t input[16384], int16_t output[16384], stem_npu_stats_t*)`.
- Consumes: generated task payload/metadata, platform cache callbacks, NPU completion tag/error/retired count and dynamic input/output ranges.

- [ ] **Step 1: Extend driver tests to fail on full-cache maintenance**

  Require `prepare` to clean the full 1.85 MiB task once, every block to clean only the 32 KiB input, and completion to invalidate only the 32 KiB output. Test out-of-task ranges, overlap overflow, wrong completion tag, error flag, retired count not equal to generated command count and timeout.

- [ ] **Step 2: Confirm the new API is red**

  ```powershell
  python scripts/_test_npu_driver.py
  python scripts/_test_audio_player.py --case npu_session
  ```

  Expected: compile errors because the resident API/types are absent.

- [ ] **Step 3: Implement range-specific cache lifecycle**

  Keep `npu_submit()` untouched. `prepare` validates/records task base and performs one full clean; `resident_submit` validates that clean/invalidate CPU and DMA ranges fall inside the resident task, cleans before doorbell, invalidates only after DONE/ERROR, and preserves barrier ordering. `stem_npu_run_block` copies exact input bytes, zeroes no other activation region, submits a monotonically increasing nonzero tag, validates completion and copies output.

- [ ] **Step 4: Run driver and session regressions**

  ```powershell
  python scripts/_test_npu_driver.py
  python scripts/_test_audio_player.py --case npu_session
  powershell -ExecutionPolicy Bypass -File scripts/86_test_navigator_sd_coldboot_image.ps1
  ```

  Expected: old submit tests remain unchanged; range logs show one full clean then 32 KiB clean/invalidate pairs; SD cold-boot static regression passes.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/include/npu_driver.h software/src/npu_driver.c software/tests/test_npu_driver.c software/audio_player/include/stem_npu_session.h software/audio_player/src/stem_npu_session.c software/audio_player/tests/test_stem_npu_session.c scripts/_test_npu_driver.py scripts/_test_audio_player.py
  git commit -m "feat(npu): add resident streaming task API"
  ```

### Task 13: Implement Mask Expansion, Continuous iSTFT and Alignment

**Files:**
- Create: `software/audio_player/include/stem_backend.h`
- Create: `software/audio_player/src/stem_backend.c`
- Create: `software/audio_player/tests/test_stem_backend.c`
- Modify: `scripts/97_generate_audio_vectors.py`
- Modify: `scripts/_test_audio_player.py`

**Interfaces:**
- Produces: `stem_backend_init(stem_backend_t*)`, `stem_backend_process(stem_backend_t*, const stem_spectrum_block_t*, const int16_t output_nhwc8[16384], float vocal_lr[4096][2])`, and `stem_delay_mix(stem_backend_t*, const float in_lr[][2], float delayed_lr[][2], size_t frames)`.
- Consumes: saved 16x513 complex mixture spectra, Q1.11 lane 0/1 masks, generated 128-to-513 synthesis matrix and continuous Hann OLA state.

- [ ] **Step 1: Write failing backend and alignment tests**

  Compare zero/full/random masks over 20 consecutive blocks with Python. Require `mask=abs(q)/2047`, clamp 0..1, first 44 bands forced zero, inverse FFT normalized by 1/1024, OLA divided by accumulated window-square, block boundaries continuous, and impulse peak of delayed mix equal to vocal estimate peak.

- [ ] **Step 2: Confirm the backend test is red**

  ```powershell
  python scripts/_test_audio_player.py --case backend
  ```

  Expected: C compile fails because backend functions are absent.

- [ ] **Step 3: Implement persistent synthesis and delay state**

  Expand each channel mask by matrix multiply, multiply the original complex spectrum, reconstruct conjugate bins through KissFFT real inverse, and maintain overlap samples plus window-square weights across calls. A frame centered at `k*256` contributes to output indices `[k*256-512,k*256+511]`; after block 0, indices `[0,3583]` are final, and every later 16-frame block finalizes exactly 4096 more samples. Pair those absolute indices with the retained original PCM, so mixture and vocal impulse indices are identical. Quantize neither branch until `audio_hw_write_frame`; use one shared float-to-int16 saturating function for delayed mix and vocal estimate.

- [ ] **Step 4: Run backend and end-to-end numerical tests**

  ```powershell
  python scripts/_test_audio_player.py --case backend
  python scripts/_test_audio_player.py --case frontend
  ```

  Expected: RMS C/Python error <=`2e-4`, impulse index exact, no 4096-sample boundary spike above `5e-4`.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/include/stem_backend.h software/audio_player/src/stem_backend.c software/audio_player/tests/test_stem_backend.c scripts/97_generate_audio_vectors.py scripts/_test_audio_player.py
  git commit -m "feat(stem): add continuous mask synthesis backend"
  ```

### Task 14: Assemble the Player State Machine and Failure Degradation

**Files:**
- Create: `software/audio_player/include/player.h`
- Create: `software/audio_player/src/player.c`
- Create: `software/audio_player/tests/test_player.c`
- Modify: `software/audio_player/src/player_main.c`
- Modify: `scripts/_test_audio_player.py`
- Modify: `scripts/101_build_audio_player.ps1`

**Interfaces:**
- Produces: `player_init(player_t*, const player_deps_t*)`, `player_step(player_t*)`, states `BOOT/SELF_TEST/MOUNT/OPEN/PREFILL/PLAY/BYPASS/FADE/FAIL_MUTE/DONE`, and one-second `player_telemetry_t` snapshots serialized as `[AUDIO] sec=<u32> state=<name> stem=<0|1> ramp=<0|1> fifo=<u16> fifo_min=<u16> uf=<u32> of=<u32> dec_err=<u32> npu_err=<u32> codec_err=<u32> deadline_miss=<u32> blk_us_avg=<u32> blk_us_max=<u32>`.
- Consumes: MP3 PCM, frontend blocks, resident NPU, backend output and audio FIFO; dependency vtable provides filesystem, time, UART and hardware calls for host fault injection.

- [ ] **Step 1: Write state/fault/deadline tests**

  Run a fake-clock 60-second playback. Verify two-block prefill before enable; MP3/SD/codec errors end muted; one NPU timeout clears the current vocal block and enters latched BYPASS while PCM continues; FIFO writes remain paired; UART emits at most one telemetry line/second; EOF fades and does not loop. Inject stage durations 25/40/20 ms and assert 92.88 ms deadline acceptance, then inject 93 ms and assert a deadline violation counter.

- [ ] **Step 2: Confirm the player test is red**

  ```powershell
  python scripts/_test_audio_player.py --case player
  ```

  Expected: compile failure for missing `player.h/player.c`.

- [ ] **Step 3: Implement bounded cooperative scheduling**

  Decode into `pcm_ring`, feed exactly 4096 new frames per model block, run NPU regardless of STEM target, and use absolute sample indices to join finalized vocal frames to the retained original PCM. After two initial NPU blocks, 7680 aligned frames are available, which satisfies prefill without filling the 8192-frame FIFO completely. Write delayed-mix/vocal pairs whenever FIFO space exists, and stop starting work when output backpressure would overflow bounded rings. Track maximum/average cycles for decode+frontend, NPU and backend+sink; track FIFO minimum/current level and all error counters.

- [ ] **Step 4: Run all host software tests and build full ELF**

  ```powershell
  python scripts/_test_audio_player.py
  python scripts/_test_npu_driver.py
  powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 -Mode FullStem -Clean
  ```

  Expected: all fault scenarios pass and `stem_player.elf` links without heap calls in the PLAY path.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/include/player.h software/audio_player/src/player.c software/audio_player/tests/test_player.c software/audio_player/src/player_main.c scripts/_test_audio_player.py scripts/101_build_audio_player.ps1
  git commit -m "feat(player): integrate continuous STEM playback"
  ```

## Milestone 4: Integrated Cold Boot and Board Acceptance

### Task 15: Build Auditable Tone, WAV, Bypass and Full-STEM SD Images

**Files:**
- Create: `software/audio_player/boot/navigator_audio_tone.bif`
- Create: `software/audio_player/boot/navigator_audio_wav.bif`
- Create: `software/audio_player/boot/navigator_audio_mp3_bypass.bif`
- Create: `software/audio_player/boot/navigator_audio_stem.bif`
- Create: `scripts/102_build_navigator_audio_boot.ps1`
- Create: `scripts/103_test_navigator_audio_boot.ps1`
- Modify: `software/NAVIGATOR_BRINGUP.md`
- Modify: `hardware/boards/alientek_navigator_z7020/README.md`

**Interfaces:**
- Consumes: validated handoff FSBL, audio bitstream, stage-specific ELF and Bootgen 2026.1.
- Produces: ignored `hardware/build/navigator_audio_boot/{tone,wav,mp3_bypass,stem}/BOOT.BIN`, readback text and SHA-256 manifest; never writes an SD card automatically.

- [ ] **Step 1: Write a failing static image gate**

  The gate must parse each Bootgen readback and require exactly FSBL, bitstream and matching ELF in that order; verify bitstream ID/address map, FSBL validated hash, ELF build mode string and no JTAG dependency. It must also run every existing script numbered 79, 82, 85, 86, 89 and 92 that is applicable without hardware.

- [ ] **Step 2: Confirm the image gate is red**

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/103_test_navigator_audio_boot.ps1
  ```

  Expected: nonzero exit because staged BIFs and images are absent.

- [ ] **Step 3: Implement deterministic staged image build**

  Reuse `scripts/91_patch_navigator_sd_fsbl_handoff.ps1`; reject an FSBL hash mismatch. Build all application ELFs through `101`, invoke Bootgen with explicit BIF/output paths, read every image back, and write a JSON manifest containing input/output SHA-256, Vivado/Vitis version, Git commit and build mode. Do not copy to removable media in this script.

- [ ] **Step 4: Build and run all static regressions**

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/102_build_navigator_audio_boot.ps1 -Clean
  powershell -ExecutionPolicy Bypass -File scripts/103_test_navigator_audio_boot.ps1
  python scripts/41_check_preboard.py
  python scripts/96_check_navigator_audio_soc.py
  ```

  Expected: four images pass partition/hash inspection, audio/NPU gates pass, and the previous autonomous NPU image remains reproducible.

- [ ] **Step 5: Commit**

  ```powershell
  git add software/audio_player/boot scripts/102_build_navigator_audio_boot.ps1 scripts/103_test_navigator_audio_boot.ps1 software/NAVIGATOR_BRINGUP.md hardware/boards/alientek_navigator_z7020/README.md
  git commit -m "build(board): assemble staged audio SD images"
  ```

### Task 16: Execute Staged Board Tests and the 30-Minute Gate

**Files:**
- Create: `scripts/104_monitor_audio_uart.py`
- Create: `software/audio_player/tests/test_uart_monitor.py`
- Create: `hardware/reports/audio_board_acceptance.md`
- Modify: `software/NAVIGATOR_BRINGUP.md`

**Interfaces:**
- Consumes: a user-selected UART COM port at the board's documented baud rate and newline-delimited telemetry containing `seconds`, stage cycle metrics, FIFO min/current, underflow, overflow, decoder errors, NPU errors, codec errors and STEM target.
- Produces: timestamped JSONL capture plus final JSON summary; exits zero only after 1800 consecutive PLAY seconds with all four hardware/error counters zero and no missed model deadline.

- [ ] **Step 1: Write failing log-parser/acceptance tests**

  Feed recorded synthetic logs for success, heartbeat gap, underflow, overflow, NPU error, codec NACK, deadline miss, KEY0 target toggles and early EOF. Require a nonzero exit for every failure log and zero only for a generated 1800-second success log.

- [ ] **Step 2: Confirm monitor tests are red**

  ```powershell
  python software/audio_player/tests/test_uart_monitor.py
  ```

  Expected: import failure because `104_monitor_audio_uart.py` is absent.

- [ ] **Step 3: Implement monitor and board runbook**

  Discover ports with `Get-PnpDevice -Class Ports`, then set `$env:STEM_UART_PORT` to the actual board port. The runbook records: tone LRCLK measured near 44.1 kHz and no NACK; deterministic PCM played count; WAV playback; MP3 bypass; dynamic NPU golden; full pipeline deadlines; repeated KEY0 transitions with no click; power-cycle cold boot with JTAG disconnected; final 30-minute counters. Record exact BOOT.BIN SHA-256 and Git commit in the report.

- [ ] **Step 4: Run staged hardware acceptance**

  ```powershell
  Get-PnpDevice -Class Ports | Format-Table FriendlyName,InstanceId
  $env:STEM_UART_PORT = 'COM7'
  python scripts/104_monitor_audio_uart.py --port $env:STEM_UART_PORT --baud 115200 --seconds 1800 --output hardware/build/navigator_audio_acceptance
  ```

  Expected after manually selecting the listed board port and testing images in tone/WAV/MP3-bypass/full-STEM order: monitor prints `AUDIO_BOARD_ACCEPTANCE: PASS`, duration >=1800 s, `NPU error=0`, `underflow=0`, `overflow=0`, `codec error=0`, deadline misses=0, and at least two observed STEM target transitions. Replace `COM7` with the discovered board entry if Windows assigns another number.

- [ ] **Step 5: Run final regression and commit evidence**

  ```powershell
  python software/audio_player/tests/test_uart_monitor.py
  python scripts/_test_audio_rtl.py
  python scripts/_test_audio_player.py
  python scripts/_test_npu_driver.py
  powershell -ExecutionPolicy Bypass -File scripts/103_test_navigator_audio_boot.ps1
  git add scripts/104_monitor_audio_uart.py software/audio_player/tests/test_uart_monitor.py hardware/reports/audio_board_acceptance.md software/NAVIGATOR_BRINGUP.md
  git commit -m "test(board): verify SD MP3 STEM speaker path"
  ```

---

## Milestone Gates

- **Gate A, PL audio:** all audio RTL simulations pass; implementation is routed with nonnegative WNS; CSR is `0x43C10000`; tone image produces 44.1 kHz speaker output without codec NACK.
- **Gate B, bypass player:** WAV and MP3 bypass images play continuously from SD with zero FIFO underflow/overflow; unsupported formats fail muted with a specific UART reason.
- **Gate C, STEM pipeline:** C/Python numerical tolerances pass; each model block uses only partial cache maintenance; measured steady-state block time is below 92.88 ms; NPU failure demonstrably falls back to original audio.
- **Gate D, cold boot:** with JTAG disconnected, the full image starts from MicroSD, plays `/music.mp3`, KEY0 repeatedly changes the audible mode through a 30 ms ramp, and the 30-minute monitor reports no NPU/FIFO/codec/deadline error.

Do not start the next gate's board test when the preceding gate is red. A periodic artifact at 92.88 ms with otherwise clean counters is recorded as the known model-context limitation, not misclassified as FIFO or I2S failure.
