$ErrorActionPreference = 'Stop'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$out = 'hardware/build/navigator_sd_coldboot_runner'
$fsblElf = 'hardware/build/navigator_sd_fsbl/zynq_fsbl.elf'
$expectedFsblSha256 = '7B3AD97C0ED47C94533A80B2FB5A3C46F316962C8CB9D9E9AB14483FA56B377E'
$runner = "$out/navigator_sd_coldboot_runner.elf"
$bif = 'software/bringup/minimal_a9/navigator_sd_coldboot.bif'
& powershell -ExecutionPolicy Bypass -File scripts/91_patch_navigator_sd_fsbl_handoff.ps1
if ($LASTEXITCODE -ne 0) { throw 'handoff-capable FSBL generation failed' }
foreach ($file in @($bootgen, $fsblElf, $runner, $bif, 'hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit')) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $fsblElf).Hash -ne $expectedFsblSha256) {
  throw 'handoff-capable zynq_fsbl.elf SHA-256 does not match the validated reference'
}
& $bootgen -arch zynq -image $bif -o "$out/BOOT.BIN" -w
if ($LASTEXITCODE -ne 0) { throw 'Bootgen failed.' }
& $bootgen -arch zynq -read "$out/BOOT.BIN" | Tee-Object -FilePath "$out/BOOT.read.txt"
if ($LASTEXITCODE -ne 0) { throw 'Bootgen readback failed.' }
Write-Output "NPU_SD_BOOT_IMAGE $(Resolve-Path "$out/BOOT.BIN")"
