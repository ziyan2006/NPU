$ErrorActionPreference = 'Stop'
$text = Get-Content -Raw scripts/78_probe_navigator_sd_fsbl_state.tcl
foreach ($name in @('SDIO0_NORMAL_INT_STATUS', 'SDIO0_ERROR_INT_STATUS', 'SDIO0_AUTO_CMD12_ERROR_STATUS')) {
  if ($text -notmatch "\{$name 0x[0-9a-f]+ h\}") {
    throw "$name is not declared as a 16-bit register"
  }
}
if ($text -notmatch '\{SDIO0_ADMA_ERROR_STATUS 0x[0-9a-f]+ b\}') {
  throw 'SDIO0_ADMA_ERROR_STATUS is not declared as an 8-bit register'
}
if ($text -notmatch 'mrd[^\r\n]*-size \$size') {
  throw 'probe does not pass the declared access width to mrd'
}
Write-Output 'NPU_SD_FSBL_PROBE_WIDTH_TEST PASS'
