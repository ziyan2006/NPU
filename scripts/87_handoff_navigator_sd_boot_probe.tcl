# Volatile handoff to the minimal probe already loaded from SD into DDR.
# This does not program PL and does not write SD, QSPI, or eMMC.

if {$argc != 0} {
  error "usage: xsdb scripts/87_handoff_navigator_sd_boot_probe.tcl"
}

set probe_entry 0x100000f4

puts "NPU_SD_PROBE_HANDOFF connecting to localhost hw_server"
connect -url tcp:localhost:3121
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
stop
puts "NPU_SD_PROBE_HANDOFF PC before: [rrd pc]"
rwr pc $probe_entry
con
puts "NPU_SD_PROBE_HANDOFF resumed at [format 0x%08X $probe_entry]"
puts "NPU_SD_PROBE_HANDOFF PASS; inspect UART0"
