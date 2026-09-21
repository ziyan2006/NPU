$ErrorActionPreference = 'Stop'
$source = 'hardware/boards/alientek_navigator_z7020/vendor_reference_local/boot/zynq_fsbl.elf'
$outputDir = 'hardware/build/navigator_sd_fsbl'
$output = Join-Path $outputDir 'zynq_fsbl.elf'
$expectedSourceSha256 = '499A4050469D66CD4EC2823F15EBA50E0898C62D598A429B4720FD78605D6818'
$textFileOffset = 0x10000
$patchAddress = 0x0000e6e4
$handoffAddress = 0x00001380
$fclk50Pattern = [byte[]](
  0x70, 0x01, 0x00, 0xf8, # FPGA0_CLK_CTRL
  0x30, 0x3f, 0xf0, 0x03, # register mask
  0x00, 0x08, 0x40, 0x00  # IO PLL / 8 / 4 = 50 MHz
)
$fclk100Value = [byte[]](0x00, 0x08, 0x20, 0x00) # IO PLL / 8 / 2
$expectedClockTables = 3

function Find-PatternOffsets([byte[]]$Data, [byte[]]$Pattern) {
  $offsets = @()
  for ($offset = 0; $offset -le $Data.Length - $Pattern.Length; ++$offset) {
    $matches = $true
    for ($index = 0; $index -lt $Pattern.Length; ++$index) {
      if ($Data[$offset + $index] -ne $Pattern[$index]) {
        $matches = $false
        break
      }
    }
    if ($matches) { $offsets += $offset }
  }
  return @($offsets)
}

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

# The validated vendor FSBL initializes all three silicon-revision clock
# tables with IO PLL / 8 / 4.  The NPU/audio XSA closes timing at 100 MHz and
# the board log proves the old 50 MHz setting misses the 92.88 ms deadline.
# Change only FPGA0_CLK_CTRL's second divisor; CPU, DDR, SD and UART clocks are
# untouched, and the PL is configured later in the FSBL boot sequence.
$clockOffsets = @(Find-PatternOffsets $bytes $fclk50Pattern)
if ($clockOffsets.Count -ne $expectedClockTables) {
  throw "expected $expectedClockTables validated 50 MHz FCLK0 tables, found $($clockOffsets.Count)"
}
foreach ($clockOffset in $clockOffsets) {
  for ($index = 0; $index -lt $fclk100Value.Length; ++$index) {
    $bytes[$clockOffset + 8 + $index] = $fclk100Value[$index]
  }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
[IO.File]::WriteAllBytes((Join-Path (Resolve-Path -LiteralPath $outputDir).Path 'zynq_fsbl.elf'), $bytes)
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash
Write-Output "NPU_SD_FSBL_PATCH PASS instruction=0x$($instruction.ToString('X8')) fclk0_mhz=100 clock_tables=$($clockOffsets.Count) sha256=$hash output=$((Resolve-Path $output).Path)"
