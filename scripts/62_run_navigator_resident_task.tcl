# Submit a chunked task to an already initialized, already configured NPU.
# Usage: xsdb scripts/62_run_navigator_resident_task.tcl <chunk_dir> ?task_base? ?watchdog_cycles?
# This script does not configure PL, initialize PS, or write boot media.
if {$argc < 1 || $argc > 3} { error "usage: <chunk_dir> ?task_base? ?watchdog_cycles?" }
set chunk_dir [file normalize [lindex $argv 0]]
if {![file isdirectory $chunk_dir]} { error "chunk directory missing: $chunk_dir" }
set task_base [expr {$argc >= 2 ? [lindex $argv 1] : 0x01000000}]
set watchdog [expr {$argc >= 3 ? [lindex $argv 2] : 10000000}]
set csr_base 0x43c00000
if {($task_base & 63) != 0} { error "task base must be 64-byte aligned" }
set chunks [lsort [glob -nocomplain -directory $chunk_dir chunk_*.bin]]
if {[llength $chunks] == 0} { error "no chunk_*.bin files" }

connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
set ip_id [lindex [mrd -address-space AP0 -force -value $csr_base] 0]
set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
if {$ip_id != 0x3155504e || ($status & 1) == 0} { error "resident NPU is not healthy/idle" }
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
puts "NPU_RESIDENT loading [llength $chunks] chunks to volatile DDR"
foreach chunk $chunks {
  set tail [file tail $chunk]
  if {![regexp {^chunk_([0-9]+)\.bin$} $tail -> offset]} { error "invalid chunk: $tail" }
  set offset [string trimleft $offset 0]
  if {$offset eq ""} { set offset 0 }
  dow -data $chunk [expr {$task_base + $offset}]
}
targets -set -filter {name == "APU"}
set header [mrd -address-space AP0 -force -value $task_base 8]
if {[lindex $header 0] != 0x4d455453 || [lindex $header 1] != 0x0055504e} {
  error "task header DDR readback failed"
}
set task_bytes [lindex $header 3]
set command_count [lindex $header 7]
if {$task_bytes < 256 || $command_count <= 0} { error "invalid task header" }
mwr -address-space AP0 -force [expr {$csr_base + 0x14}] 1
mwr -address-space AP0 -force [expr {$csr_base + 0x1c}] 7
mwr -address-space AP0 -force [expr {$csr_base + 0x24}] $task_base
mwr -address-space AP0 -force [expr {$csr_base + 0x28}] 0
mwr -address-space AP0 -force [expr {$csr_base + 0x2c}] $task_bytes
mwr -address-space AP0 -force [expr {$csr_base + 0x30}] 0x53454544
mwr -address-space AP0 -force [expr {$csr_base + 0x48}] $watchdog
mwr -address-space AP0 -force [expr {$csr_base + 0x34}] 1
for {set attempt 0} {$attempt < 1000} {incr attempt} {
  after 10
  set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
  if {($status & 0x18) != 0} { break }
}
set tag [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x38}]] 0]
set error_code [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x3c}]] 0]
set retired [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x4c}]] 0]
set cycles [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x50}]] 0]
puts [format "NPU_RESIDENT status=0x%08X tag=0x%08X error=0x%08X retired=%d/%d cycles=%d" \
  $status $tag $error_code $retired $command_count $cycles]
if {($status & 0x19) != 0x09 || $tag != 0x53454544 || $error_code != 0 || $retired != $command_count} {
  error "resident task failed"
}
puts "NPU_RESIDENT PASS complete task lifecycle"
