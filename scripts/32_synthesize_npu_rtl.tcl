# Vivado out-of-context synthesis for the current P4 RTL prototypes.
# Usage:
#   vivado -mode batch -source scripts/32_synthesize_npu_rtl.tcl \
#     -tclargs <top> ?part? ?clock_period_ns? ?output_dir?

if {$argc < 1} {
  puts stderr "usage: <top> ?part? ?clock_period_ns? ?output_dir?"
  exit 2
}

set top [lindex $argv 0]
set part [expr {$argc >= 2 ? [lindex $argv 1] : "xc7z020clg400-1"}]
set clock_period [expr {$argc >= 3 ? [lindex $argv 2] : "5.000"}]
set baseline_clock_period "10.000"
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set output_dir [expr {$argc >= 4
  ? [file normalize [lindex $argv 3]]
  : [file join $repo_root hardware reports vivado_2026_1 $top]}]

set valid_tops {
  npu_scratchpad
  npu_axi_block_reader
  npu_task_loader
  npu_command_fetch
  npu_execution_frontend
  npu_dma_subsystem
  npu_tensor_mac_8x8
  npu_requant_post
  npu_conv2d_controller
  npu_conv2d_pipeline
  npu_vec_add
  npu_upsample2x
}
if {[lsearch -exact $valid_tops $top] < 0} {
  puts stderr "unsupported top '$top'; choose one of: $valid_tops"
  exit 2
}

file mkdir $output_dir
set rtl_dir [file join $repo_root hardware rtl]
set source_files [list \
  [file join $rtl_dir include npu_isa_pkg.sv] \
  [file join $rtl_dir include npu_dma_pkg.sv] \
  [file join $rtl_dir npu_axi_block_reader.sv] \
  [file join $rtl_dir npu_task_loader.sv] \
  [file join $rtl_dir npu_memory_arbiter4.sv] \
  [file join $rtl_dir npu_command_fetch.sv] \
  [file join $rtl_dir npu_descriptor_cache.sv] \
  [file join $rtl_dir npu_execution_frontend.sv] \
  [file join $rtl_dir npu_u32_mul_iter.sv] \
  [file join $rtl_dir npu_dma_agu.sv] \
  [file join $rtl_dir npu_dma_frontend.sv] \
  [file join $rtl_dir npu_dma_engine.sv] \
  [file join $rtl_dir npu_scratchpad_bank.sv] \
  [file join $rtl_dir npu_scratchpad.sv] \
  [file join $rtl_dir npu_dma_subsystem.sv] \
  [file join $rtl_dir npu_tensor_mac_8x8.sv] \
  [file join $rtl_dir npu_requant_post.sv] \
  [file join $rtl_dir npu_conv2d_controller.sv] \
  [file join $rtl_dir npu_conv2d_pipeline.sv] \
  [file join $rtl_dir npu_vec_add.sv] \
  [file join $rtl_dir npu_upsample2x.sv]]

read_verilog -sv $source_files
synth_design -top $top -part $part -mode out_of_context \
  -flatten_hierarchy rebuilt
create_clock -name core_clk -period $clock_period [get_ports clk_i]

report_utilization -hierarchical -hierarchical_depth 4 \
  -file [file join $output_dir utilization_hierarchical.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
  -file [file join $output_dir timing_summary.rpt]
set target_timing_paths [get_timing_paths -delay_type max -max_paths 1]
set target_worst_slack "NA"
if {[llength $target_timing_paths] > 0} {
  set target_worst_slack \
    [get_property SLACK [lindex $target_timing_paths 0]]
}
report_methodology -file [file join $output_dir methodology.rpt]
check_timing -verbose -file [file join $output_dir check_timing.rpt]

create_clock -name core_clk -period $baseline_clock_period [get_ports clk_i]
report_timing_summary -delay_type max -max_paths 20 \
  -file [file join $output_dir timing_summary_100mhz.rpt]
set baseline_timing_paths [get_timing_paths -delay_type max -max_paths 1]
set baseline_worst_slack "NA"
if {[llength $baseline_timing_paths] > 0} {
  set baseline_worst_slack \
    [get_property SLACK [lindex $baseline_timing_paths 0]]
}

set cell_count [llength [get_cells -hierarchical]]
set lut_count [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
set ff_count [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
set bram36_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]

set summary_path [file join $output_dir synthesis_summary.txt]
set summary [open $summary_path w]
puts $summary "tool=Vivado [version -short]"
puts $summary "top=$top"
puts $summary "part=$part"
puts $summary "clock_period_ns=$clock_period"
puts $summary "cells=$cell_count"
puts $summary "luts=$lut_count"
puts $summary "flip_flops=$ff_count"
puts $summary "ramb36e1=$bram36_count"
puts $summary "ramb18e1=$bram18_count"
puts $summary "dsp48e1=$dsp_count"
puts $summary "target_post_synth_wns_ns=$target_worst_slack"
puts $summary "baseline_100mhz_post_synth_wns_ns=$baseline_worst_slack"
close $summary

puts "NPU_SYNTHESIS_RESULT top=$top part=$part LUT=$lut_count FF=$ff_count BRAM36=$bram36_count BRAM18=$bram18_count DSP=$dsp_count TARGET_WNS=$target_worst_slack BASELINE_100MHZ_WNS=$baseline_worst_slack"
exit 0
