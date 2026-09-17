# Run a chunked DDR task through the freestanding A9 NPU driver.
# Usage: xsdb scripts/67_run_navigator_a9_task.tcl <ps7_init.tcl> <bit> <elf> <chunk_dir> <task_bytes> <output_offset> <output_bytes> <expected_fnv1a>
# The script programs only volatile PL/DDR through JTAG.  It never writes boot media.
if {$argc != 8} {
  error "usage: <ps7_init.tcl> <bit> <elf> <chunk_dir> <task_bytes> <output_offset> <output_bytes> <expected_fnv1a>"
}
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set elf_file [file normalize [lindex $argv 2]]
set chunk_dir [file normalize [lindex $argv 3]]
foreach file [list $init_file $bit_file $elf_file] {
  if {![file isfile $file]} { error "missing $file" }
}
if {![file isdirectory $chunk_dir]} { error "chunk directory missing: $chunk_dir" }
set task_bytes [expr {[lindex $argv 4]}]
set output_offset [expr {[lindex $argv 5]}]
set output_bytes [expr {[lindex $argv 6]}]
set expected_fnv1a [expr {[lindex $argv 7]}]
set task_base 0x01000000
set result_base 0x12000000
set csr_base 0x43c00000
set task_tag 0x41395053
set watchdog 10000000
set maximum_polls 100000000
if {$task_bytes < 256 || ($task_bytes & 63) != 0 || $output_offset < 0 || $output_bytes <= 0} {
  error "invalid task or output geometry"
}

connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}
puts "NPU_A9_TASK loading task chunks to volatile DDR"
set chunks [lsort [glob -nocomplain -directory $chunk_dir chunk_*.bin]]
if {[llength $chunks] == 0} { error "no task chunks found" }
foreach chunk $chunks {
  set tail [file tail $chunk]
  if {![regexp {^chunk_([0-9]+)\.bin$} $tail -> offset]} { error "invalid chunk: $tail" }
  set offset [string trimleft $offset 0]
  if {$offset eq ""} { set offset 0 }
  dow -data $chunk [expr {$task_base + $offset}]
}
targets -set -filter {name == "APU"}
set header [mrd -address-space AP0 -force -value $task_base 8]
if {[lindex $header 0] != 0x4d455453 || [lindex $header 1] != 0x0055504e || [lindex $header 3] != $task_bytes} {
  error "task header DDR readback failed"
}
mwr -address-space AP0 -force [expr {$result_base + 0x40}] 0x4e505443
mwr -address-space AP0 -force [expr {$result_base + 0x44}] $task_base
mwr -address-space AP0 -force [expr {$result_base + 0x48}] $task_bytes
mwr -address-space AP0 -force [expr {$result_base + 0x4c}] $task_tag
mwr -address-space AP0 -force [expr {$result_base + 0x50}] $watchdog
mwr -address-space AP0 -force [expr {$result_base + 0x54}] [expr {$task_base + $output_offset}]
mwr -address-space AP0 -force [expr {$result_base + 0x58}] $output_bytes
mwr -address-space AP0 -force [expr {$result_base + 0x5c}] $expected_fnv1a
mwr -address-space AP0 -force [expr {$result_base + 0x60}] $maximum_polls
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
dow $elf_file
con
for {set attempt 0} {$attempt < 200} {incr attempt} {
  after 25
  targets -set -filter {name == "APU"}
  set result [mrd -address-space AP0 -force -value $result_base 11]
  if {[lindex $result 0] != 0x4e505401} { break }
}
catch {stop}
puts "NPU_A9_TASK results=$result"
if {[lindex $result 0] != 0x4e5054a5} { error "A9 task runner failed; code=[lindex $result 0]" }
if {[lindex $result 4] != $task_tag || [lindex $result 5] != 0 || [lindex $result 10] != $expected_fnv1a} {
  error "A9 task completion or output checksum mismatch"
}
puts "NPU_A9_TASK PASS A9 driver submitted DDR task and verified output FNV-1a"
