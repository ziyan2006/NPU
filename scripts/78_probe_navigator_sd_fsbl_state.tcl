# Read-only register snapshot for a Zynq FSBL stalled during SD boot.
# Usage: xsdb scripts/78_probe_navigator_sd_fsbl_state.tcl
if {$argc != 0} { error "usage: xsdb scripts/78_probe_navigator_sd_fsbl_state.tcl" }
connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
foreach register {
  {BOOT_MODE 0xf800025c w}
  {REBOOT_STATUS 0xf8000258 w}
  {DEVCFG_STATUS 0xf8007014 w}
  {DEVCFG_INT_STS 0xf800700c w}
  {DEVCFG_CTRL 0xf8007000 w}
  {SDIO0_BLOCK_SIZE_COUNT 0xe0100004 w}
  {SDIO0_ARGUMENT 0xe0100008 w}
  {SDIO0_TRANSFER_MODE_COMMAND 0xe010000c w}
  {SDIO0_PRESENT_STATE 0xe0100024 w}
  {SDIO0_NORMAL_INT_STATUS 0xe0100030 h}
  {SDIO0_ERROR_INT_STATUS 0xe0100032 h}
  {SDIO0_AUTO_CMD12_ERROR_STATUS 0xe010003c h}
  {SDIO0_ADMA_ERROR_STATUS 0xe0100054 b}
  {SDIO0_ADMA_SYSTEM_ADDRESS 0xe0100058 w}
  {SDIO0_BUFFER_DATA_PORT 0xe0100020 w}
} {
  lassign $register name address size
  switch $size { b {set digits 2} h {set digits 4} default {set digits 8} }
  if {[catch {set value [lindex [mrd -address-space AP0 -force -size $size -value $address] 0]} issue]} {
    puts "NPU_SD_FSBL_STATE $name READ_ERROR: $issue"
  } else {
    puts [format "NPU_SD_FSBL_STATE %s=0x%0*X" $name $digits $value]
  }
}
puts "NPU_SD_FSBL_STATE PASS"
