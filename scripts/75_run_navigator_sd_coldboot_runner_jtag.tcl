# Volatile JTAG validation of the self-recovering SD cold-boot runner.
# Usage: xsdb scripts/75_run_navigator_sd_coldboot_runner_jtag.tcl <elf>
if {$argc != 1} {
  error "usage: xsdb scripts/75_run_navigator_sd_coldboot_runner_jtag.tcl <elf>"
}
set elf_file [file normalize [lindex $argv 0]]
if {![file isfile $elf_file]} { error "missing ELF: $elf_file" }
connect -url tcp:localhost:3121
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
if {[catch {stop} issue]} { error "could not halt Cortex-A9 #0: $issue" }
puts "NPU_SD_COLDBOOT_JTAG loading runner to volatile DDR"
dow $elf_file
con
puts "NPU_SD_COLDBOOT_JTAG resumed; inspect UART0 for PASS and heartbeat"
