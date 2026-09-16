# Fast path-finding implementation run from scripts/35 post-synthesis DCP.
# This deliberately avoids the very slow ExtraNetDelay_high post-place pass.

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set input_dcp [expr {$argc >= 1 ? [file normalize [lindex $argv 0]] :
  [file join $repo_root hardware build npu_top_impl post_synth.dcp]}]
set report_dir [expr {$argc >= 2 ? [file normalize [lindex $argv 1]] :
  [file join $repo_root hardware reports vivado_2026_1 npu_top_impl_fast]}]
set build_dir [expr {$argc >= 3 ? [file normalize [lindex $argv 2]] :
  [file join $repo_root hardware build npu_top_impl_fast]}]
if {![file exists $input_dcp]} {
  puts stderr "missing synthesis checkpoint: $input_dcp"
  exit 2
}
file mkdir $report_dir
file mkdir $build_dir

open_checkpoint $input_dcp
opt_design -directive Explore
place_design -directive Explore
write_checkpoint -force [file join $build_dir post_place.dcp]
report_timing_summary -delay_type max -max_paths 20 \
  -file [file join $report_dir timing_post_place.rpt]
route_design -directive Explore
write_checkpoint -force [file join $build_dir post_route.dcp]

report_utilization -file [file join $report_dir utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 4 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 100 \
  -report_unconstrained -file [file join $report_dir timing_summary.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -ruledeck default -file [file join $report_dir drc.rpt]
check_timing -verbose -file [file join $report_dir check_timing.rpt]

set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set final_wns [expr {[llength $setup_paths] ?
  [get_property SLACK [lindex $setup_paths 0]] : "NA"}]
set final_whs [expr {[llength $hold_paths] ?
  [get_property SLACK [lindex $hold_paths 0]] : "NA"}]
set final_tns 0.000
set failing_setup_paths [get_timing_paths -quiet -delay_type max \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]
foreach path $failing_setup_paths {
  set final_tns [expr {$final_tns + [get_property SLACK $path]}]
}
set lut_cells [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]
set lut_count [llength [get_bels -quiet -of_objects $lut_cells]]
set ff_count [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
set bram36_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
set unrouted_count [llength [get_nets -hierarchical -filter \
  {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]
set critical_drc_count [llength [get_drc_violations -quiet -filter \
  {SEVERITY == Error || SEVERITY == {Critical Warning}}]]

set summary [open [file join $report_dir implementation_summary.txt] w]
puts $summary "tool=Vivado [version -short]"
puts $summary "top=npu_top"
puts $summary "part=[get_property PART [current_design]]"
puts $summary "clock_period_ns=10.000"
puts $summary "clock_uncertainty_ns=0.200"
puts $summary "interface_delay_ns=2.000"
puts $summary "slice_luts=$lut_count"
puts $summary "flip_flops=$ff_count"
puts $summary "ramb36e1=$bram36_count"
puts $summary "dsp48e1=$dsp_count"
puts $summary "post_route_wns_ns=$final_wns"
puts $summary "post_route_tns_ns=$final_tns"
puts $summary "post_route_whs_ns=$final_whs"
puts $summary "unrouted_nets=$unrouted_count"
puts $summary "critical_drc=$critical_drc_count"
close $summary

puts "NPU_ROUTE_RESULT LUT=$lut_count FF=$ff_count BRAM36=$bram36_count DSP=$dsp_count WNS=$final_wns TNS=$final_tns WHS=$final_whs UNROUTED=$unrouted_count CRITICAL_DRC=$critical_drc_count"
exit 0
