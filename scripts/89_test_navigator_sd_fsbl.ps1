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

$bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $elf).Path)
function Count-Pattern([byte[]]$Data, [byte[]]$Pattern) {
  $count = 0
  for ($offset = 0; $offset -le $Data.Length - $Pattern.Length; ++$offset) {
    $matches = $true
    for ($index = 0; $index -lt $Pattern.Length; ++$index) {
      if ($Data[$offset + $index] -ne $Pattern[$index]) {
        $matches = $false
        break
      }
    }
    if ($matches) { ++$count }
  }
  return $count
}
$fclkPrefix = [byte[]](0x70,0x01,0x00,0xf8,0x30,0x3f,0xf0,0x03)
$fclk50 = [byte[]]($fclkPrefix + [byte[]](0x00,0x08,0x40,0x00))
$fclk100 = [byte[]]($fclkPrefix + [byte[]](0x00,0x08,0x20,0x00))
if ((Count-Pattern $bytes $fclk50) -ne 0) {
  throw 'generated FSBL still contains a 50 MHz FCLK0 clock table'
}
if ((Count-Pattern $bytes $fclk100) -ne 3) {
  throw 'generated FSBL does not patch all three FCLK0 tables to 100 MHz'
}

Write-Output 'NPU_SD_FSBL_TEST PASS fclk0_mhz=100 clock_tables=3'
