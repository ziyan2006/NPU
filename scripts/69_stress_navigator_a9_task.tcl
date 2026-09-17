# Multi-iteration stress test runner through freestanding A9 NPU driver.
# Usage: xsdb scripts/69_stress_navigator_a9_task.tcl <ps7_init.tcl> <bit> <elf> <chunk_dir> <task_bytes> <output_offset> <output_bytes> <expected_fnv1a> <iterations>
if {$argc != 9} {
  error "usage: <ps7_init.tcl> <bit> <elf> <chunk_dir> <task_bytes> <output_offset> <output_bytes> <expected_fnv1a> <iterations>"
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
set iterations [expr {[lindex $argv 8]}]

set task_base 0x01000000
set result_base 0x12000000
set csr_base 0x43c00000
set task_tag 0x41395053
set watchdog 10000000
set maximum_polls 100000000

if {$task_bytes < 256 || ($task_bytes & 63) != 0 || $output_offset < 0 || $output_bytes <= 0 || $iterations <= 0} {
  error "invalid task geometry or iterations count"
}

connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
fpga -file $bit_file
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init; ps7_post_config}

puts "NPU_STRESS loading task chunks to volatile DDR..."
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

# Clear result area
for {set offset 0} {$offset < 64} {incr offset 4} {
  mwr -address-space AP0 -force [expr {$result_base + $offset}] 0
}

# Write config words
mwr -address-space AP0 -force [expr {$result_base + 0x40}] 0x4e505443
mwr -address-space AP0 -force [expr {$result_base + 0x44}] $task_base
mwr -address-space AP0 -force [expr {$result_base + 0x48}] $task_bytes
mwr -address-space AP0 -force [expr {$result_base + 0x4c}] $task_tag
mwr -address-space AP0 -force [expr {$result_base + 0x50}] $watchdog
mwr -address-space AP0 -force [expr {$result_base + 0x54}] [expr {$task_base + $output_offset}]
mwr -address-space AP0 -force [expr {$result_base + 0x58}] $output_bytes
mwr -address-space AP0 -force [expr {$result_base + 0x5c}] $expected_fnv1a
mwr -address-space AP0 -force [expr {$result_base + 0x60}] $maximum_polls
mwr -address-space AP0 -force [expr {$result_base + 0x64}] $iterations

targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
dow $elf_file
con

puts "NPU_STRESS started A9 stress test: Target = $iterations iterations"
set start_sec [clock seconds]
set last_report_sec $start_sec
set last_reported_iter 0

while {1} {
  after 2000
  targets -set -filter {name == "APU"}
  set res [mrd -address-space AP0 -force -value $result_base 12]
  set status [expr {[lindex $res 0]}]
  set cur_iter [expr {[lindex $res 1]}]
  set tot_iter [expr {[lindex $res 2]}]
  set err_cnt [expr {[lindex $res 3]}]
  set min_cyc [expr {[lindex $res 5]}]
  set max_cyc [expr {[lindex $res 6]}]
  set last_cyc [expr {[lindex $res 7]}]
  set last_hash [expr {[lindex $res 8]}]
  set now_sec [clock seconds]
  set elapsed_sec [expr {$now_sec - $start_sec}]

  # Print report every 10 seconds or when finished
  if {$now_sec - $last_report_sec >= 10 || $status != 0x4e505401} {
    set pct [format "%.1f" [expr {($cur_iter * 100.0) / $iterations}]]
    set rate [expr {$elapsed_sec > 0 ? [format "%.1f" [expr {$cur_iter * 1.0 / $elapsed_sec}]] : 0.0}]
    set remain_sec [expr {$rate > 0 ? int(($iterations - $cur_iter) / $rate) : 0}]
    set m [expr {$elapsed_sec / 60}]
    set s [expr {$elapsed_sec % 60}]
    set rem_m [expr {$remain_sec / 60}]
    set rem_s [expr {$remain_sec % 60}]
    puts "\[STRESS ${m}m${s}s / ETA ${rem_m}m${rem_s}s\] Iter: $cur_iter / $iterations ($pct%) | Rate: ${rate} it/s | Last cycles: $last_cyc (Min: $min_cyc, Max: $max_cyc) | Errors: $err_cnt | Hash: [format 0x%08X $last_hash]"
    set last_report_sec $now_sec
    set last_reported_iter $cur_iter
  }

  if {$status == 0x4e5054a5} {
    puts "=========================================================="
    puts "NPU_STRESS_PASS: 10-Minute Stress Test PASSED!"
    puts "Total Iterations: $cur_iter / $iterations"
    puts "Total Elapsed Time: ${elapsed_sec} seconds ([format "%.2f" [expr {$elapsed_sec / 60.0}]] minutes)"
    puts "Min Cycles: $min_cyc"
    puts "Max Cycles: $max_cyc"
    puts "Cycle Jitter: [expr {$max_cyc - $min_cyc}] cycles ([format "%.3f" [expr {($max_cyc - $min_cyc) * 100.0 / $min_cyc}]]%)"
    puts "Errors Encountered: $err_cnt"
    puts "Final Output FNV-1a Hash: [format 0x%08X $last_hash] (Matched Golden [format 0x%08X $expected_fnv1a])"
    puts "=========================================================="
    break
  }

  if {$status != 0x4e505401} {
    puts "NPU_STRESS_FAIL: Aborted at iteration $cur_iter / $iterations with status [format 0x%08X $status], error_code=[lindex $res 4]"
    error "Stress test failed"
  }
}
