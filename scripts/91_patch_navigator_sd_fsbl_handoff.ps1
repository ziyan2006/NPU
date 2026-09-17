$ErrorActionPreference = 'Stop'
$source = 'hardware/boards/alientek_navigator_z7020/vendor_reference_local/boot/zynq_fsbl.elf'
$outputDir = 'hardware/build/navigator_sd_fsbl'
$output = Join-Path $outputDir 'zynq_fsbl.elf'
$expectedSourceSha256 = '499A4050469D66CD4EC2823F15EBA50E0898C62D598A429B4720FD78605D6818'
$textFileOffset = 0x10000
$patchAddress = 0x0000e6e4
$handoffAddress = 0x00001380

if (!(Test-Path -LiteralPath $source)) { throw "missing vendor FSBL: $source" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash -ne $expectedSourceSha256) {
  throw 'vendor FSBL SHA-256 does not match the validated pure-PL reference'
}

$bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $source).Path)
$patchOffset = $textFileOffset + $patchAddress
$expectedLoop = [byte[]](0xfe, 0xff, 0xff, 0xea)
$expectedLoadCall = [byte[]](0xff, 0xc8, 0xff, 0xeb)
for ($index = 0; $index -lt 4; ++$index) {
  if ($bytes[$patchOffset - 4 + $index] -ne $expectedLoadCall[$index]) {
    throw 'instruction before patch site is not BL LoadBootImage'
  }
  if ($bytes[$patchOffset + $index] -ne $expectedLoop[$index]) {
    throw 'patch site is not the validated infinite-loop instruction'
  }
}

$delta = [int64]$handoffAddress - ([int64]$patchAddress + 8)
if (($delta % 4) -ne 0) { throw 'handoff branch target is not word aligned' }
$immediate = (($delta -shr 2) -band 0x00ffffff)
$branchWithLink = [Convert]::ToUInt64('EB000000', 16)
$instruction = [uint32]($branchWithLink -bor ([uint64]$immediate))
$expectedInstruction = [Convert]::ToUInt32('EBFFCB25', 16)
if ($instruction -ne $expectedInstruction) { throw ('unexpected BL encoding: 0x{0:X8}' -f $instruction) }
$encoded = [BitConverter]::GetBytes($instruction)
for ($index = 0; $index -lt 4; ++$index) {
  $bytes[$patchOffset + $index] = $encoded[$index]
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
[IO.File]::WriteAllBytes((Join-Path (Resolve-Path -LiteralPath $outputDir).Path 'zynq_fsbl.elf'), $bytes)
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash
Write-Output "NPU_SD_FSBL_PATCH PASS instruction=0x$($instruction.ToString('X8')) sha256=$hash output=$((Resolve-Path $output).Path)"
