$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$objdump = Join-Path $toolchain 'arm-none-eabi-objdump.exe'
$elf = 'hardware/build/navigator_sd_boot_probe/navigator_sd_boot_probe.elf'
foreach ($file in @($objdump, $elf)) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}

$symbols = (& $objdump -t $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'objdump could not inspect probe symbols' }
if ($symbols -notmatch '\bnavigator_sd_probe_npu_id_valid\b') {
  throw 'probe does not contain an independently testable NPU ID guard'
}

$main = (& $objdump -d --disassemble=main $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'objdump could not disassemble probe main' }
if ($main -notmatch '\bbl\s+[0-9a-f]+\s+<navigator_sd_probe_npu_id_valid>') {
  throw 'probe main does not call the NPU ID guard'
}

$guard = (& $objdump -d --disassemble=navigator_sd_probe_npu_id_valid $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'objdump could not disassemble NPU ID guard' }
if ($guard -notmatch '504e' -or $guard -notmatch '3155' -or
    $guard -notmatch '\b(cmp|sub)\b') {
  throw 'compiled NPU ID guard does not compare against 0x3155504E'
}

$ascii = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes((Resolve-Path $elf).Path))
if ($ascii -notmatch '\[FAIL\] invalid NPU ID') {
  throw 'probe image does not contain an invalid-NPU-ID failure path'
}

Write-Output 'NPU_SD_BOOT_PROBE_ID_GUARD_TEST PASS'
