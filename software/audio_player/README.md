# Navigator standalone audio player

This directory contains the staged Cortex-A9 bare-metal player for the
Navigator-compatible Zynq-7020 board. It uses the audio CSR at `0x43C10000`
and the WM8960/I2S PL subsystem built by scripts 93-96.

Two initial build modes are available:

- `Tone`: waits for WM8960 initialization, then emits the PL test tone. It
  does not mount or read the SD card.
- `Wav`: mounts `0:/`, opens `0:/test.wav`, accepts only stereo 44.1 kHz
  16-bit PCM, preloads all 8192 FIFO frames before enabling playback, and
  applies a 1323-frame fade at EOF.
- `Mp3Bypass`: mounts `0:/`, opens `0:/music.mp3`, accepts only MPEG-1 Layer
  III stereo at 44.1 kHz, and sends decoded PCM to the original-audio path
  with the vocal path held at zero. This staged decoder exposes complete MPEG
  frames, including encoder delay/padding; later player integration owns
  gapless trimming and the final EOF fade.

Build with Vitis 2026.1:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 `
  -Mode Tone -Clean
powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 `
  -Mode Wav -Clean
powershell -ExecutionPolicy Bypass -File scripts/101_build_audio_player.ps1 `
  -Mode Mp3Bypass -Clean
```

Outputs are local and ignored by Git:

```text
hardware/build/navigator_audio_player/tone_player.elf
hardware/build/navigator_audio_player/wav_player.elf
hardware/build/navigator_audio_player/mp3_bypass_player.elf
```

The build validates the XSA audio address, enables `xilffs`, and rejects an
ELF missing `player_main`, `f_mount`, or `audio_hw_write_frame`. It does not
create boot media, copy files to an SD card, or connect to hardware.
