# Start an SD-loaded runner already present in volatile DDR.
#
# Usage: xsdb scripts/71_start_navigator_sd_runner_jtag.tcl
#
# This is a diagnostic recovery path for a system halted in the FSBL fallback
# loop.  It sets only the live A9 program counter to the application's existing
# DDR entry point and resumes it.  A power cycle restores normal boot behavior;
# no boot medium, flash, or PL configuration is written.

if {$argc != 0} {
  error "usage: xsdb scripts/71_start_navigator_sd_runner_jtag.tcl"
}

set app_entry 0x10000000
puts "NPU_SD_RUNNER_START connecting to localhost hw_server"
connect -url tcp:localhost:3121
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
if {[catch {stop} stop_result]} {
  error "could not suspend Cortex-A9 #0: $stop_result"
}
puts "NPU_SD_RUNNER_START PC before: [rrd pc]"
rwr pc $app_entry
puts "NPU_SD_RUNNER_START PC set to $app_entry; check UART0 terminal now"
con
puts "NPU_SD_RUNNER_START Cortex-A9 #0 resumed"
