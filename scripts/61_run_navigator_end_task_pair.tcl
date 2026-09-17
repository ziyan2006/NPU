# Run zero/nonzero minimal tasks after one PS/DDR initialization.
# Usage: xsdb scripts/61_run_navigator_end_task_pair.tcl <ps7_init.tcl> <bit> <zero.bin> <nonzero.bin>
# JTAG and DDR only; no persistent boot-media access.
if {$argc != 4} { error "usage: <ps7_init.tcl> <bit> <zero.bin> <nonzero.bin>" }
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set zero_file [file normalize [lindex $argv 2]]
set nonzero_file [file normalize [lindex $argv 3]]
foreach file [list $init_file $bit_file $zero_file $nonzero_file] {
  if {![file isfile $file]} { error "missing $file" }
}
set csr_base 0x43c00000

proc run_end_task {label image base} {
  global csr_base
  set bytes [file size $image]
  targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
  puts "NPU_END_PAIR loading $label ($bytes bytes) at $base"
  dow -data $image $base
  targets -set -filter {name == "APU"}
  set header [mrd -address-space AP0 -force -value $base 2]
  if {[lindex $header 0] != 0x4d455453 || [lindex $header 1] != 0x0055504e} {
    error "$label header DDR readback failed"
  }
  mwr -address-space AP0 -force [expr {$csr_base + 0x14}] 1
  mwr -address-space AP0 -force [expr {$csr_base + 0x1c}] 7
  mwr -address-space AP0 -force [expr {$csr_base + 0x24}] $base
  mwr -address-space AP0 -force [expr {$csr_base + 0x28}] 0
  mwr -address-space AP0 -force [expr {$csr_base + 0x2c}] $bytes
  mwr -address-space AP0 -force [expr {$csr_base + 0x30}] 0x50414952
  mwr -address-space AP0 -force [expr {$csr_base + 0x48}] 100000
  mwr -address-space AP0 -force [expr {$csr_base + 0x34}] 1
  for {set attempt 0} {$attempt < 100} {incr attempt} {
    after 10
    set status [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x18}]] 0]
    if {($status & 0x18) != 0} { break }
  }
  set error_code [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x3c}]] 0]
  set retired [lindex [mrd -address-space AP0 -force -value [expr {$csr_base + 0x4c}]] 0]
  puts [format "NPU_END_PAIR %s status=0x%08X error=0x%08X retired=%d" $label $status $error_code $retired]
  if {($status & 0x19) != 0x09 || $error_code != 0 || $retired != 1} {
    error "$label did not complete cleanly"
  }
}

connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
puts "NPU_END_PAIR volatile PL program (JTAG only; no flash)"
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}
targets -set -filter {name == "APU"}
set ip_id [lindex [mrd -address-space AP0 -force -value $csr_base] 0]
if {$ip_id != 0x3155504e} { error "NPU IP_ID mismatch" }
run_end_task zero $zero_file 0x00100000
run_end_task nonzero $nonzero_file 0x00110000
puts "NPU_END_PAIR PASS zero/nonzero task-loader comparison"
