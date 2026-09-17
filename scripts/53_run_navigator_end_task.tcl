# JTAG-only execution of the minimal NPU END task.
# Usage: xsdb scripts/53_run_navigator_end_task.tcl <ps7_init.tcl> <bit> <task.bin> ?task_base?
# The task is copied only to volatile DDR; this script never writes boot media.
if {$argc != 3 && $argc != 4} { error "usage: <ps7_init.tcl> <bit> <task.bin> ?task_base?" }
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set task_file [file normalize [lindex $argv 2]]
foreach file [list $init_file $bit_file $task_file] {
  if {![file isfile $file]} { error "missing $file" }
}

set csr_base 0x43c00000
set task_base [expr {$argc == 4 ? [lindex $argv 3] : 0x11000000}]
set task_bytes [file size $task_file]
if {$task_bytes <= 0 || ($task_bytes & 63) != 0 || ($task_base & 63) != 0} {
  error "task image must be nonempty and 64-byte aligned in size"
}

puts "NPU_END_TASK connecting to local hw_server"
connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
puts "NPU_END_TASK volatile PL program (JTAG only; no flash)"
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}
set hp0_rdctrl [lindex [mrd -force -value 0xF8008000] 0]
set hp0_rdissue [lindex [mrd -force -value 0xF8008004] 0]
set hp0_wrctrl [lindex [mrd -force -value 0xF8008014] 0]
set level_shifters [lindex [mrd -force -value 0xF8000900] 0]
puts [format "NPU_END_TASK HP0 rdctrl=0x%08X rdissue=0x%08X wrctrl=0x%08X lvlshft=0x%08X" \
  $hp0_rdctrl $hp0_rdissue $hp0_wrctrl $level_shifters]

puts [format "NPU_END_TASK loading %d bytes to volatile DDR at 0x%08X" $task_bytes $task_base]
dow -data $task_file $task_base

# Use the APU DAP physical address space; this is independent of CPU cache/MMU.
targets -set -filter {name == "APU"}
set task_magic [mrd -address-space AP0 -force -value $task_base 2]
puts "NPU_END_TASK DAP DDR header words=$task_magic"
if {[lindex $task_magic 0] != 0x4d455453 || [lindex $task_magic 1] != 0x0055504e} {
  error "task image did not read back correctly from DDR"
}
set ip_id [lindex [mrd -address-space AP0 -force -value $csr_base] 0]
if {$ip_id != 0x3155504e} {
  error [format "NPU IP_ID mismatch: expected 0x3155504E, got 0x%08X" $ip_id]
}
set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
if {($status & 1) == 0} { error [format "NPU not idle before submit: 0x%08X" $status] }

# Clear any stale sticky status, set a bounded watchdog, then atomically submit.
mwr -address-space AP0 -force [expr {$csr_base + 0x14}] 0x1
mwr -address-space AP0 -force [expr {$csr_base + 0x1c}] 0x7
mwr -address-space AP0 -force [expr {$csr_base + 0x24}] $task_base
mwr -address-space AP0 -force [expr {$csr_base + 0x28}] 0
mwr -address-space AP0 -force [expr {$csr_base + 0x2c}] $task_bytes
mwr -address-space AP0 -force [expr {$csr_base + 0x30}] 0x454e4431
mwr -address-space AP0 -force [expr {$csr_base + 0x48}] 100000
mwr -address-space AP0 -force [expr {$csr_base + 0x34}] 1

set completed 0
for {set attempt 0} {$attempt < 100} {incr attempt} {
  after 10
  set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
  if {($status & 0x18) != 0} { set completed 1; break }
}
if {!$completed} { error "NPU END task timeout after 1 s" }
set tag [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x38}]] 0]
set error_code [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x3c}]] 0]
set retired [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x4c}]] 0]
set cycles [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x50}]] 0]
puts [format "NPU_END_TASK status=0x%08X tag=0x%08X error=0x%08X retired=%d cycles=%d" $status $tag $error_code $retired $cycles]
if {($status & 0x10) != 0 || ($status & 0x09) != 0x09 || $tag != 0x454e4431 || $retired != 1} {
  error "NPU END task did not complete cleanly"
}
puts "NPU_END_TASK PASS DDR task fetch, LUT preload, command fetch, END, and CSR completion"
