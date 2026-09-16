# Export a *candidate* JTAG bitstream/XSA after the exact physical board has
# been confirmed.  This script refuses a generic part or an unimplemented run.
# Usage:
#   vivado -mode batch -source scripts/43_export_navigator_candidate.tcl -- \
#     ?project_directory? ?output_directory?

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_dir [expr {$argc >= 1
  ? [file normalize [lindex $argv 0]]
  : [file join $repo_root hardware build navigator_z7020_candidate]}]
set output_dir [expr {$argc >= 2
  ? [file normalize [lindex $argv 1]]
  : [file join $repo_root hardware build navigator_z7020_export]}]
set confirmation [file join $repo_root hardware boards \
  alientek_navigator_z7020 board_confirmation.local.json]
set project_file [file join $project_dir reference_zynq_soc.xpr]

if {![file exists $confirmation]} {
  error "board identity is not confirmed; create board_confirmation.local.json first"
}
set confirmation_file [open $confirmation r]
set confirmation_text [read $confirmation_file]
close $confirmation_file
if {![regexp -nocase {"fpga_marking"\s*:\s*"[^"]*XC7Z020CLG400-2[^"]*"} \
      $confirmation_text]} {
  error "board confirmation does not contain XC7Z020CLG400-2 marking"
}
foreach field {core_board_revision base_board_revision ddr_marking} {
  if {![regexp "\\\"$field\\\"\\s*:\\s*\\\"[^\\\"]+\\\"" \
        $confirmation_text]} {
    error "board confirmation field is missing or null: $field"
  }
}
if {![file exists $project_file]} {
  error "Navigator candidate project is missing: $project_file"
}

open_project $project_file
if {[string tolower [get_property PART [current_project]]] ne "xc7z020clg400-2"} {
  error "refusing to export a non-Navigator-Z7020 part"
}
set run [get_runs impl_1]
if {![string match "*Complete!*" [get_property STATUS $run]]} {
  error "implementation is not complete"
}
open_run $run
set setup [get_timing_paths -quiet -delay_type max -max_paths 1]
set hold [get_timing_paths -quiet -delay_type min -max_paths 1]
if {![llength $setup] || [get_property SLACK [lindex $setup 0]] < 0.0} {
  error "setup timing is not closed"
}
if {![llength $hold] || [get_property SLACK [lindex $hold 0]] < 0.0} {
  error "hold timing is not closed"
}

file mkdir $output_dir
set bit_file [file join $output_dir stem_npu_navigator_z7020_candidate.bit]
set xsa_file [file join $output_dir stem_npu_navigator_z7020_candidate.xsa]
write_bitstream -force $bit_file
write_hw_platform -fixed -include_bit -force -file $xsa_file
puts "NPU_NAVIGATOR_EXPORT_RESULT bit=$bit_file xsa=$xsa_file"
close_project
