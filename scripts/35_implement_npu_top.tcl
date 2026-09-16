# Board-independent out-of-context implementation of the complete NPU PL core.
# Usage:
#   vivado -mode batch -source scripts/35_implement_npu_top.tcl -- \
#     ?part? ?clock_period_ns? ?report_dir? ?build_dir? ?synth_only?
#
# The 2 ns synchronous interface delay and 0.2 ns clock uncertainty reserve
# budget for the surrounding SmartConnect/protocol-converter block design.
# Final sign-off must still be repeated in the actual board block design.

set part [expr {$argc >= 1 ? [lindex $argv 0] : "xc7z020clg400-1"}]
set clock_period [expr {$argc >= 2 ? [lindex $argv 1] : "10.000"}]
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set report_dir [expr {$argc >= 3
  ? [file normalize [lindex $argv 2]]
  : [file join $repo_root hardware reports vivado_2026_1 npu_top_impl]}]
set build_dir [expr {$argc >= 4
  ? [file normalize [lindex $argv 3]]
  : [file join $repo_root hardware build npu_top_impl]}]
set synth_only [expr {$argc >= 5 ? [lindex $argv 4] : 0}]
file mkdir $report_dir
file mkdir $build_dir

set file_list_path [file join $repo_root hardware rtl npu_rtl.f]
set file_list [open $file_list_path r]
set source_files {}
while {[gets $file_list line] >= 0} {
  set line [string trim $line]
  if {$line eq "" || [string index $line 0] eq "#"} {
    continue
  }
  lappend source_files [file join $repo_root $line]
}
close $file_list

read_verilog -sv $source_files
synth_design -top npu_top -part $part -mode out_of_context \
  -flatten_hierarchy rebuilt -directive PerformanceOptimized
write_checkpoint -force [file join $build_dir synthesized_unconstrained.dcp]
create_clock -name core_clk -period $clock_period [get_ports aclk]
set_clock_uncertainty 0.200 [get_clocks core_clk]

# OOC clock ports have no concrete PS clock buffer site.  The virtual source
# gives the placer a representative Zynq-7020 global clock insertion delay.
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports aclk]
set_false_path -from [get_ports aresetn]
set synchronous_inputs [get_ports -filter \
  {DIRECTION == IN && NAME != aclk && NAME != aresetn}]
if {[llength $synchronous_inputs] > 0} {
  set_input_delay 2.000 -clock core_clk $synchronous_inputs
}
set synchronous_outputs [all_outputs]
if {[llength $synchronous_outputs] > 0} {
  set_output_delay 2.000 -clock core_clk $synchronous_outputs
}

write_checkpoint -force [file join $build_dir post_synth.dcp]
if {$synth_only} {
  puts "NPU_SYNTH_CHECKPOINT_RESULT top=npu_top part=$part checkpoint=[file join $build_dir post_synth.dcp]"
  exit 0
}
opt_design -directive Explore
place_design -directive ExtraNetDelay_high
phys_opt_design -directive AggressiveExplore
write_checkpoint -force [file join $build_dir post_place.dcp]
route_design -directive Explore

set routed_paths [get_timing_paths -delay_type max -max_paths 1]
set routed_wns "NA"
if {[llength $routed_paths] > 0} {
  set routed_wns [get_property SLACK [lindex $routed_paths 0]]
}
if {$routed_wns ne "NA" && $routed_wns < 0.000} {
  puts "NPU_IMPLEMENTATION_INFO initial routed WNS=$routed_wns; running post-route phys_opt"
  phys_opt_design -post_route -directive AggressiveExplore
}
write_checkpoint -force [file join $build_dir post_route.dcp]

report_utilization -file [file join $report_dir utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 4 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 50 \
  -report_unconstrained -file [file join $report_dir timing_summary.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -ruledeck default -file [file join $report_dir drc.rpt]
report_methodology -file [file join $report_dir methodology.rpt]
check_timing -verbose -file [file join $report_dir check_timing.rpt]
report_clock_utilization -file [file join $report_dir clock_utilization.rpt]
report_power -file [file join $report_dir power_estimate.rpt]

set final_paths [get_timing_paths -delay_type max -max_paths 1]
set final_wns "NA"
set final_tns 0.000
if {[llength $final_paths] > 0} {
  set final_wns [get_property SLACK [lindex $final_paths 0]]
}
set failing_setup_paths [get_timing_paths -quiet -delay_type max \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]
foreach path $failing_setup_paths {
  set final_tns [expr {$final_tns + [get_property SLACK $path]}]
}
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set final_whs "NA"
if {[llength $hold_paths] > 0} {
  set final_whs [get_property SLACK [lindex $hold_paths 0]]
}

set cell_count [llength [get_cells -hierarchical]]
set lut_cells [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]
set lut_count [llength [get_bels -quiet -of_objects $lut_cells]]
set ff_count [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
set bram36_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set unrouted_count [llength [get_nets -hierarchical -filter \
  {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]
set critical_drc_count [llength [get_drc_violations -quiet -filter \
  {SEVERITY == Error || SEVERITY == {Critical Warning}}]]

set summary_path [file join $report_dir implementation_summary.txt]
set summary [open $summary_path w]
puts $summary "tool=Vivado [version -short]"
puts $summary "top=npu_top"
puts $summary "part=$part"
puts $summary "clock_period_ns=$clock_period"
puts $summary "clock_uncertainty_ns=0.200"
puts $summary "interface_delay_ns=2.000"
puts $summary "cells=$cell_count"
puts $summary "slice_luts=$lut_count"
puts $summary "flip_flops=$ff_count"
puts $summary "ramb36e1=$bram36_count"
puts $summary "ramb18e1=$bram18_count"
puts $summary "dsp48e1=$dsp_count"
puts $summary "post_route_wns_ns=$final_wns"
puts $summary "post_route_tns_ns=$final_tns"
puts $summary "post_route_whs_ns=$final_whs"
puts $summary "unrouted_nets=$unrouted_count"
puts $summary "critical_drc=$critical_drc_count"
close $summary

puts "NPU_IMPLEMENTATION_RESULT top=npu_top part=$part LUT=$lut_count FF=$ff_count BRAM36=$bram36_count BRAM18=$bram18_count DSP=$dsp_count WNS=$final_wns TNS=$final_tns WHS=$final_whs UNROUTED=$unrouted_count CRITICAL_DRC=$critical_drc_count"
if {$final_wns eq "NA" || $final_wns < 0.000 || $final_whs eq "NA" \
    || $final_whs < 0.000 || $unrouted_count != 0 || $critical_drc_count != 0} {
  exit 1
}
exit 0
