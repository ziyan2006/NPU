# Inspect a Zynq A9 that was booted from a MicroSD BOOT.BIN.
#
# Usage: xsdb scripts/70_probe_navigator_sd_boot.tcl
#
# The probe briefly halts Cortex-A9 #0 to print its program counter and reads
# the first words of the expected DDR application image.  It resumes the core
# before exiting.  It never programs PL or writes any boot medium.

if {$argc != 0} {
  error "usage: xsdb scripts/70_probe_navigator_sd_boot.tcl"
}

set app_base 0x10000000

puts "NPU_SD_BOOT_PROBE connecting to localhost hw_server"
connect -url tcp:localhost:3121
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}

if {[catch {stop} stop_result]} {
  error "could not suspend Cortex-A9 #0: $stop_result"
}

puts "NPU_SD_BOOT_PROBE halted Cortex-A9 #0"
puts "NPU_SD_BOOT_PROBE PC: [rrd pc]"
puts "NPU_SD_BOOT_PROBE CPSR: [rrd cpsr]"

# A BOOT.BIN inspected on 2026-09-17 loads navigator_a9_sd_runner at this
# address.  These ARM-vector words show whether that payload reached DDR; they
# are observations only, not an expected checksum for future images.
targets -set -filter {name == "APU"}
set app_words [mrd -address-space AP0 -force -value $app_base 8]
puts "NPU_SD_BOOT_PROBE DDR[0x10000000..0x1000001C]=$app_words"

targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
con
puts "NPU_SD_BOOT_PROBE resumed Cortex-A9 #0"
puts "NPU_SD_BOOT_PROBE PASS"
