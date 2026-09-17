# Read NPU CSRs after an SD cold boot without halting Cortex-A9.
#
# Usage: xsdb scripts/72_probe_navigator_sd_npu_csr.tcl
#
# Uses the DAP AP0 physical address space only.  It neither changes the CPU
# state nor writes PL, DDR, SD, QSPI, or eMMC.
if {$argc != 0} {
  error "usage: xsdb scripts/72_probe_navigator_sd_npu_csr.tcl"
}
set csr_base 0x43c00000
connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
foreach pair {
  {IP_ID 0x00}
  {RTL_VERSION 0x04}
  {ISA_VERSION 0x08}
  {CONTROL 0x10}
  {STATUS 0x18}
  {ERROR 0x1c}
} {
  lassign $pair name offset
  set address [expr {$csr_base + $offset}]
  if {[catch {set value [lindex [mrd -address-space AP0 -force -value $address] 0]} issue]} {
    puts "NPU_SD_CSR_PROBE $name at [format 0x%08X $address]: READ_ERROR $issue"
  } else {
    puts "NPU_SD_CSR_PROBE $name at [format 0x%08X $address]: [format 0x%08X $value]"
  }
}
puts "NPU_SD_CSR_PROBE PASS"
