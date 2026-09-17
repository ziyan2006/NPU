$ErrorActionPreference = 'Stop'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$boot = 'hardware/build/navigator_sd_boot_probe/BOOT.BIN'
if (!(Test-Path -LiteralPath $boot)) { throw "missing probe image: $boot" }
if (!(Test-Path -LiteralPath $bootgen)) { throw "missing Bootgen: $bootgen" }
$report = & $bootgen -arch zynq -read $boot 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Bootgen could not inspect probe image.' }
$text = $report -join "`n"
if ($text -notmatch 'total_images \(0x04\) : 0x00000003') {
  throw 'probe image must contain exactly three images'
}
if ($text -notmatch 'zynq_fsbl\.elf') {
  throw 'probe image must use the validated handoff-capable zynq_fsbl.elf'
}
if ($text -match 'fsbl_rewrapped\.elf') {
  throw 'probe image must not use a rewrapped OCM snapshot as its FSBL'
}
if ($text -notmatch 'navigator_sd_boot_probe\.elf') {
  throw 'probe application partition missing from BOOT.BIN'
}
if ($text -notmatch 'stem_npu_navigator_z7020_candidate\.bit') {
  throw 'NPU bitstream partition missing from BOOT.BIN'
}
$match = [regex]::Match($text, 'PARTITION HEADER TABLE \(navigator_sd_boot_probe\.elf\.0\)[\s\S]*?total_length \(0x08\) : 0x([0-9a-fA-F]+)\s+load_addr \(0x0c\) : 0x([0-9a-fA-F]+)\s+exec_addr \(0x10\) : 0x([0-9a-fA-F]+)[\s\S]*?attributes \(0x18\) : 0x([0-9a-fA-F]+)')
if (!$match.Success) { throw 'could not determine probe application partition length' }
$words = [Convert]::ToUInt32($match.Groups[1].Value, 16)
$loadAddress = [Convert]::ToUInt32($match.Groups[2].Value, 16)
$execAddress = [Convert]::ToUInt32($match.Groups[3].Value, 16)
$attributes = [Convert]::ToUInt32($match.Groups[4].Value, 16)
if ($words -ge 0x4000) { throw "probe application must be below 64 KiB; got 0x$($words.ToString('X')) words" }
if ($loadAddress -ne 0x10000000) { throw ('unexpected probe load address: 0x{0:X8}' -f $loadAddress) }
$partitionEnd = [uint64]$loadAddress + 4L * $words
if (($execAddress -band 3) -ne 0 -or $execAddress -lt $loadAddress -or $execAddress -ge $partitionEnd) {
  throw ('probe entry 0x{0:X8} is outside its application partition' -f $execAddress)
}
if ($attributes -ne 0x10) { throw ('unexpected probe partition attributes: 0x{0:X8}' -f $attributes) }
Write-Output "NPU_SD_BOOT_PROBE_IMAGE_TEST PASS app_words=0x$($words.ToString('X'))"
