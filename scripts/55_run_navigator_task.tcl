# JTAG-only full task lifecycle smoke for Navigator Z7020.
# Usage: xsdb scripts/55_run_navigator_task.tcl <ps7_init.tcl> <bit> <task.bin> ?task_base? ?watchdog_cycles?
# It downloads only volatile PL configuration and volatile DDR task data.
if {$argc < 3 || $argc > 5} {
  error "usage: <ps7_init.tcl> <bit> <task.bin> ?task_base? ?watchdog_cycles?"
}
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set task_file [file normalize [lindex $argv 2]]
foreach file [list $init_file $bit_file] {
  if {![file isfile $file]} { error "missing $file" }
}
if {![file isfile $task_file] && ![file isdirectory $task_file]} {
  error "task source is neither an image file nor a chunk directory: $task_file"
}
set task_base [expr {$argc >= 4 ? [lindex $argv 3] : 0x01000000}]
set watchdog [expr {$argc >= 5 ? [lindex $argv 4] : 10000000}]
if {($task_base & 63) != 0} {
  error "task base must be 64-byte aligned"
}
set csr_base 0x43c00000

connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
puts "NPU_TASK volatile PL program (JTAG only; no flash)"
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}
if {[file isdirectory $task_file]} {
  set chunks [lsort [glob -nocomplain -directory $task_file chunk_*.bin]]
  if {[llength $chunks] == 0} { error "task chunk directory has no chunk_*.bin files" }
  puts [format "NPU_TASK loading %d JTAG chunks to volatile DDR at 0x%08X" [llength $chunks] $task_base]
  foreach chunk $chunks {
    set tail [file tail $chunk]
    if {![regexp {^chunk_([0-9]+)\.bin$} $tail -> offset]} {
      error "invalid chunk filename: $tail"
    }
    # Tcl treats a leading-zero integer as octal; filenames are deliberately
    # zero-padded for lexical ordering, so normalize to unambiguous decimal.
    set offset [string trimleft $offset 0]
    if {$offset eq ""} { set offset 0 }
    puts [format "NPU_TASK chunk %s (%d bytes)" $tail [file size $chunk]]
    dow -data $chunk [expr {$task_base + $offset}]
  }
} else {
  puts [format "NPU_TASK loading %d volatile DDR bytes at 0x%08X" [file size $task_file] $task_base]
  dow -data $task_file $task_base
}

targets -set -filter {name == "APU"}
set header [mrd -address-space AP0 -force -value $task_base 8]
if {[lindex $header 0] != 0x4d455453 || [lindex $header 1] != 0x0055504e} {
  error "task header does not read back from DDR"
}
set command_count [lindex $header 7]
if {$command_count <= 0} { error "task header command_count is invalid" }
set task_bytes [lindex $header 3]
if {$task_bytes < 256 || ($task_bytes & 63) != 0} { error "task header total_bytes is invalid" }
set ip_id [lindex [mrd -address-space AP0 -force -value $csr_base] 0]
if {$ip_id != 0x3155504e} { error "NPU IP_ID mismatch" }

mwr -address-space AP0 -force [expr {$csr_base + 0x14}] 0x1
mwr -address-space AP0 -force [expr {$csr_base + 0x1c}] 0x7
mwr -address-space AP0 -force [expr {$csr_base + 0x24}] $task_base
mwr -address-space AP0 -force [expr {$csr_base + 0x28}] 0
mwr -address-space AP0 -force [expr {$csr_base + 0x2c}] $task_bytes
mwr -address-space AP0 -force [expr {$csr_base + 0x30}] 0x46554c4c
mwr -address-space AP0 -force [expr {$csr_base + 0x48}] $watchdog
mwr -address-space AP0 -force [expr {$csr_base + 0x34}] 1

set completed 0
for {set attempt 0} {$attempt < 1000} {incr attempt} {
  after 10
  set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
  if {($status & 0x18) != 0} { set completed 1; break }
}
if {!$completed} { error "NPU task timeout after 10 s" }
set tag [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x38}]] 0]
set error_code [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x3c}]] 0]
set error_pc [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x40}]] 0]
set retired [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x4c}]] 0]
set cycles [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x50}]] 0]
puts [format "NPU_TASK status=0x%08X tag=0x%08X error=0x%08X pc=0x%08X retired=%d/%d cycles=%d" \
  $status $tag $error_code $error_pc $retired $command_count $cycles]
if {($status & 0x10) != 0 || ($status & 0x09) != 0x09 || $tag != 0x46554c4c || $retired != $command_count} {
  error "NPU task did not complete cleanly"
}
puts "NPU_TASK PASS complete command stream and output-DMA lifecycle"
