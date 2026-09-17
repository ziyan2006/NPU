$ErrorActionPreference = 'Stop'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$boot = 'hardware/build/navigator_sd_npu_bit_only/BOOT.BIN'
if (!(Test-Path -LiteralPath $boot)) { throw "missing NPU bit-only image: $boot" }
if (!(Test-Path -LiteralPath $bootgen)) { throw "missing Bootgen: $bootgen" }
$report = & $bootgen -arch zynq -read $boot 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Bootgen could not inspect NPU bit-only image.' }
$text = $report -join "`n"
if ($text -notmatch 'total_images \(0x04\) : 0x00000002') {
  throw 'NPU bit-only image must contain exactly two images'
}
if ($text -notmatch 'zynq_fsbl\.elf') {
  throw 'bit-only image must use the original vendor zynq_fsbl.elf'
}
if ($text -match 'fsbl_rewrapped\.elf') {
  throw 'bit-only image must not use a rewrapped OCM snapshot as its FSBL'
}
if ($text -notmatch 'stem_npu_navigator_z7020_candidate\.bit') {
  throw 'NPU bitstream missing from bit-only image'
}
if ($text -match 'navigator_sd_boot_probe|navigator_sd_coldboot_runner') {
  throw 'bit-only image must not contain a PS application partition'
}
Write-Output 'NPU_SD_NPU_BIT_ONLY_TEST PASS'
