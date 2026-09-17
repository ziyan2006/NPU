$ErrorActionPreference = 'Stop'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$out = 'hardware/build/navigator_sd_npu_bit_only'
$fsblElf = 'hardware/boards/alientek_navigator_z7020/vendor_reference_local/boot/zynq_fsbl.elf'
$expectedFsblSha256 = '499A4050469D66CD4EC2823F15EBA50E0898C62D598A429B4720FD78605D6818'
$boot = "$out/BOOT.BIN"
$bif = 'software/bringup/minimal_a9/navigator_sd_npu_bit_only.bif'
$bitstream = 'hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit'
foreach ($file in @($bootgen, $fsblElf, $bif, $bitstream)) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $fsblElf).Hash -ne $expectedFsblSha256) {
  throw 'vendor zynq_fsbl.elf SHA-256 does not match the validated reference'
}
New-Item -ItemType Directory -Force -Path $out | Out-Null
if (Test-Path -LiteralPath $boot) { Remove-Item -LiteralPath $boot -Force }
& $bootgen -arch zynq -image $bif -o $boot -w
if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $boot)) { throw 'Bootgen failed.' }
Write-Output "NPU_SD_NPU_BIT_ONLY_IMAGE $(Resolve-Path $boot)"
