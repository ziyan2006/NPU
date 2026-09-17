$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$objdump = Join-Path $toolchain 'arm-none-eabi-objdump.exe'
$readelf = Join-Path $toolchain 'arm-none-eabi-readelf.exe'
$elf = 'hardware/build/navigator_sd_fsbl/zynq_fsbl.elf'

foreach ($file in @($objdump, $readelf, $elf)) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}

$headers = (& $readelf -h -l $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'readelf could not inspect generated FSBL' }
if ($headers -notmatch 'Machine:\s+ARM') { throw 'generated FSBL is not an ARM ELF' }

$disassembly = (& $objdump -d --start-address=0xe6dc --stop-address=0xe6ec $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'objdump could not disassemble generated FSBL' }
if ($disassembly -notmatch '(?m)^\s*e6e0:\s+ebffc8ff\s+bl\s+ae4\s+<LoadBootImage>') {
  throw 'generated FSBL no longer calls LoadBootImage immediately before handoff'
}
if ($disassembly -notmatch '(?m)^\s*e6e4:\s+ebffcb25\s+bl\s+1380\s+<FsblHandoff>') {
  throw 'generated FSBL main path does not call FsblHandoff at 0xE6E4'
}
$symbols = (& $objdump -t $elf 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'objdump could not inspect generated FSBL symbols' }
if ($symbols -notmatch '\bXSdPs_CfgInitialize\b') {
  throw 'generated FSBL does not contain the SD controller driver'
}

Write-Output 'NPU_SD_FSBL_TEST PASS'
