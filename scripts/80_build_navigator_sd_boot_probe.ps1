$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$gcc = Join-Path $toolchain 'arm-none-eabi-gcc.exe'
$objcopy = Join-Path $toolchain 'arm-none-eabi-objcopy.exe'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$out = 'hardware/build/navigator_sd_boot_probe'
$fsblElf = 'hardware/build/navigator_sd_fsbl/zynq_fsbl.elf'
$expectedFsblSha256 = 'B9C732AEF65EFA97AE89B8D5EFBD0A2A3265F2275DF9C91B66ADEAE31A7AB584'
$bif = 'software/bringup/minimal_a9/navigator_sd_boot_probe.bif'
& powershell -ExecutionPolicy Bypass -File scripts/91_patch_navigator_sd_fsbl_handoff.ps1
if ($LASTEXITCODE -ne 0) { throw 'handoff-capable FSBL generation failed' }
foreach ($file in @($gcc, $objcopy, $bootgen, $fsblElf, $bif, 'software/bringup/minimal_a9/start.S', 'software/bringup/minimal_a9/linker_sd_boot_probe.ld', 'software/bringup/minimal_a9/navigator_sd_boot_probe.c', 'hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit')) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $fsblElf).Hash -ne $expectedFsblSha256) {
  throw 'handoff-capable zynq_fsbl.elf SHA-256 does not match the validated reference'
}
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $gcc -mcpu=cortex-a9 -marm -mfloat-abi=soft -O2 -ffreestanding -fno-builtin -fdata-sections -ffunction-sections -nostdlib `
  '-Wl,-T,software/bringup/minimal_a9/linker_sd_boot_probe.ld' '-Wl,--gc-sections' `
  "-Wl,-Map,$out/navigator_sd_boot_probe.map" `
  software/bringup/minimal_a9/start.S software/bringup/minimal_a9/navigator_sd_boot_probe.c `
  -lgcc -o $out/navigator_sd_boot_probe.elf
if ($LASTEXITCODE -ne 0) { throw 'SD boot probe link failed.' }
& $objcopy -O binary $out/navigator_sd_boot_probe.elf $out/navigator_sd_boot_probe.bin
if ($LASTEXITCODE -ne 0) { throw 'SD boot probe binary conversion failed.' }
& $bootgen -arch zynq -image $bif -o "$out/BOOT.BIN" -w
if ($LASTEXITCODE -ne 0) { throw 'Bootgen failed.' }
Write-Output "NPU_SD_BOOT_PROBE_IMAGE $(Resolve-Path $out/BOOT.BIN)"
