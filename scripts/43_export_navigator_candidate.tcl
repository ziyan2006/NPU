# Export a *candidate* JTAG bitstream/XSA after the exact physical board has
# been confirmed.  This script refuses a generic part or an unimplemented run.
# Usage:
#   vivado -mode batch -source scripts/43_export_navigator_candidate.tcl -tclargs \
#     ?project_directory? ?output_directory?

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_dir [expr {$argc >= 1
  ? [file normalize [lindex $argv 0]]
  : [file join $repo_root hardware build navigator_z7020_vendor_v37]}]
set output_dir [expr {$argc >= 2
  ? [file normalize [lindex $argv 1]]
  : [file join $repo_root hardware build navigator_z7020_vendor_v37_export]}]
set confirmation [file join $repo_root hardware boards \
  alientek_navigator_z7020 board_confirmation.local.json]
set project_file [file join $project_dir reference_zynq_soc.xpr]

if {![file exists $confirmation]} {
  error "board identity is not confirmed; create board_confirmation.local.json first"
}
set validator [file join $script_dir 45_validate_navigator_confirmation.py]
set python_exe [auto_execok python]
if {$python_exe eq ""} {
  error "Python is required to validate the physical board confirmation"
}
if {[catch {exec {*}$python_exe $validator $confirmation} validation_output]} {
  error "board confirmation rejected: $validation_output"
}
puts $validation_output
if {![file exists $project_file]} {
  error "Navigator candidate project is missing: $project_file"
}

open_project $project_file
if {[string tolower [get_property PART [current_project]]] ne "xc7z020clg400-2"} {
  error "refusing to export a non-Navigator-Z7020 part"
}
# The earlier minimal profile used a generic DDR entry.  Verify the actual
# implemented project carries the V3.7 vendor-derived PS7 configuration;
# a filled-in confirmation JSON alone must never release the old candidate.
set bd_file [file join $project_dir reference_zynq_soc.srcs \
  sources_1 bd npu_soc npu_soc.bd]
if {![file isfile $bd_file]} {
  error "NPU block design is missing: $bd_file"
}
open_bd_design $bd_file
set ps [get_bd_cells -quiet /processing_system7_0]
if {[llength $ps] != 1} {
  error "expected one PS7 cell, found $ps"
}
foreach {property expected} {
  PCW_CRYSTAL_PERIPHERAL_FREQMHZ 33.333333
  PCW_UIPARAM_DDR_MEMORY_TYPE {DDR 3 (Low Voltage)}
  PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125}
  PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit}
  PCW_UIPARAM_DDR_FREQ_MHZ 533.333333
  PCW_DDR_RAM_HIGHADDR 0x3FFFFFFF
  PCW_UART0_PERIPHERAL_ENABLE 1
  PCW_UART0_UART0_IO {MIO 14 .. 15}
  PCW_USE_M_AXI_GP0 1
  PCW_USE_S_AXI_HP0 1
  PCW_S_AXI_HP0_DATA_WIDTH 64
  PCW_FPGA0_PERIPHERAL_FREQMHZ 100.000000
  PCW_QSPI_PERIPHERAL_ENABLE 0
  PCW_SD0_PERIPHERAL_ENABLE 0
} {
  set actual [get_property CONFIG.$property $ps]
  if {$actual ne $expected} {
    error "refusing non-V3.7 PS7 configuration: $property expected '$expected', got '$actual'"
  }
}
foreach index {0 1 2 3} {
  set actual [get_property CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY$index $ps]
  if {$actual ne "0.25"} {
    error "DDR board delay lane $index does not match vendor V3.7 reference"
  }
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
# In project mode, -include_bit reads the implementation run's own bitstream.
# A standalone write_bitstream to $output_dir leaves that run at route_design
# and makes write_hw_platform fail even though the external .bit is valid.
set run_bit_file [file join $project_dir reference_zynq_soc.runs impl_1 \
  "[get_property TOP [get_filesets sources_1]].bit"]
close_design
if {![file isfile $run_bit_file]} {
  launch_runs $run -to_step write_bitstream
  wait_on_run $run
}
if {![file isfile $run_bit_file]} {
  error "implementation run did not produce its bitstream: $run_bit_file"
}
file copy -force $run_bit_file $bit_file
open_run $run
# This XSA is consumed by a Zynq standalone application, not an accelerator
# shell.  Preserve that intent in the handoff metadata so downstream tools do
# not classify it as a data-centre accelerator platform.
set_property platform.design_intent.embedded true [current_project]
write_hw_platform -fixed -include_bit -force -file $xsa_file
puts "NPU_NAVIGATOR_EXPORT_RESULT bit=$bit_file xsa=$xsa_file"
close_project
