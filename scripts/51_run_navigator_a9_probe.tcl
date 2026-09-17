# JTAG-only: program volatile PL, initialize PS/DDR, run A9 ELF, read results.
# Usage: xsdb scripts/51_run_navigator_a9_probe.tcl <ps7_init.tcl> <bit> <elf>
if {$argc != 3} { error "usage: <ps7_init.tcl> <bit> <elf>" }
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set elf_file [file normalize [lindex $argv 2]]
foreach file [list $init_file $bit_file $elf_file] { if {![file isfile $file]} { error "missing $file" } }
connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}
dow $elf_file
con
after 250
stop
targets -set -filter {name == "APU"}
set values [mrd -address-space AP0 -force -value 0x12000000 8]
puts "NPU_A9_PROBE results=$values"
if {[lindex $values 0] != 0x4e5052a5} { error "A9 probe failed; code=[lindex $values 0]" }
if {[lindex $values 1] != 0x3155504e} { error "A9 probe IP_ID mismatch" }
puts "NPU_A9_PROBE PASS A9 executed DDR test and read NPU CSR"
