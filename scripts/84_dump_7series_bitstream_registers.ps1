param([Parameter(Mandatory = $true)][string]$Bitstream)
$ErrorActionPreference = 'Stop'

$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Bitstream))
$sync = [byte[]](0xAA, 0x99, 0x55, 0x66)
$syncOffset = -1
for ($offset = 0; $offset -le $bytes.Length - 4; ++$offset) {
  if ($bytes[$offset] -eq $sync[0] -and $bytes[$offset + 1] -eq $sync[1] -and
      $bytes[$offset + 2] -eq $sync[2] -and $bytes[$offset + 3] -eq $sync[3]) {
    $syncOffset = $offset
    break
  }
}
if ($syncOffset -lt 0) { throw '7-series sync word not found' }

$registerNames = @{
  0='CRC'; 1='FAR'; 2='FDRI'; 3='FDRO'; 4='CMD'; 5='CTL0'; 6='MASK';
  7='STAT'; 8='LOUT'; 9='COR0'; 10='MFWR'; 11='CBC'; 12='IDCODE';
  13='AXSS'; 14='COR1'; 16='WBSTAR'; 17='TIMER'; 22='BOOTSTS'; 24='CTL1'
}
function Read-BigEndianWord([int]$Offset) {
  return (([uint32]$bytes[$Offset] -shl 24) -bor
          ([uint32]$bytes[$Offset + 1] -shl 16) -bor
          ([uint32]$bytes[$Offset + 2] -shl 8) -bor
          [uint32]$bytes[$Offset + 3])
}

Write-Output "BITSTREAM_PACKET_REPORT file=$(Resolve-Path -LiteralPath $Bitstream) sync=0x$($syncOffset.ToString('X'))"
$cursor = $syncOffset + 4
while ($cursor -le $bytes.Length - 4) {
  $header = Read-BigEndianWord $cursor
  $cursor += 4
  $type = ($header -shr 29) -band 0x7
  if ($type -eq 1) {
    $opcode = ($header -shr 27) -band 0x3
    $address = ($header -shr 13) -band 0x3fff
    $count = $header -band 0x7ff
    if ($cursor + (4 * $count) -gt $bytes.Length) { break }
    if ($opcode -eq 2 -and $count -gt 0 -and $address -ne 2) {
      $name = if ($registerNames.ContainsKey([int]$address)) { $registerNames[[int]$address] } else { "REG_$address" }
      $values = for ($i = 0; $i -lt $count; ++$i) { '0x{0:X8}' -f (Read-BigEndianWord ($cursor + 4 * $i)) }
      Write-Output ("{0,-8} {1}" -f $name, ($values -join ','))
    }
    $cursor += 4 * $count
  } elseif ($type -eq 2) {
    $count = $header -band 0x07ffffff
    if ($cursor + (4L * $count) -gt $bytes.Length) { break }
    $cursor += 4 * $count
  }
}
