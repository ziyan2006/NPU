# Read the 32 KiB network output left in volatile DDR by scripts/55.
# Usage: xsdb scripts/58_dump_navigator_output.tcl <output.bin> ?task_base?
if {$argc != 1 && $argc != 2} { error "usage: <output.bin> ?task_base?" }
set output_file [file normalize [lindex $argv 0]]
set task_base [expr {$argc == 2 ? [lindex $argv 1] : 0x01000000}]
set output_offset 966784
set output_bytes 32768
connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
set address [expr {$task_base + $output_offset}]
puts [format "NPU_OUTPUT_DUMP reading %d bytes from volatile DDR 0x%08X" $output_bytes $address]
# mrd's count is 32-bit words by default, not bytes.
mrd -address-space AP0 -force -bin -file $output_file $address [expr {$output_bytes / 4}]
if {![file isfile $output_file] || [file size $output_file] != $output_bytes} {
  error "DDR output dump has wrong size"
}
puts "NPU_OUTPUT_DUMP PASS wrote $output_file"
