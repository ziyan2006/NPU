# Report routed reg-to-reg timing independently of unlocated OOC interface pins.
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set input_dcp [expr {$argc >= 1 ? [file normalize [lindex $argv 0]] :
  [file join $repo_root hardware build npu_top_impl_fast post_route.dcp]}]
set report_dir [expr {$argc >= 2 ? [file normalize [lindex $argv 1]] :
  [file join $repo_root hardware reports vivado_2026_1 npu_top_impl_fast]}]
file mkdir $report_dir
open_checkpoint $input_dcp
set registers [all_registers]
report_timing -delay_type max -from $registers -to $registers \
  -max_paths 100 -nworst 1 -path_type full_clock_expanded \
  -file [file join $report_dir timing_internal_setup.rpt]
report_timing -delay_type min -from $registers -to $registers \
  -max_paths 100 -nworst 1 -path_type full_clock_expanded \
  -file [file join $report_dir timing_internal_hold.rpt]
set setup_path [get_timing_paths -delay_type max -from $registers \
  -to $registers -max_paths 1]
set hold_path [get_timing_paths -delay_type min -from $registers \
  -to $registers -max_paths 1]
set internal_wns [expr {[llength $setup_path] ?
  [get_property SLACK [lindex $setup_path 0]] : "NA"}]
set internal_whs [expr {[llength $hold_path] ?
  [get_property SLACK [lindex $hold_path 0]] : "NA"}]
set summary [open [file join $report_dir internal_timing_summary.txt] w]
puts $summary "post_route_internal_wns_ns=$internal_wns"
puts $summary "post_route_internal_whs_ns=$internal_whs"
close $summary
puts "NPU_INTERNAL_TIMING_RESULT WNS=$internal_wns WHS=$internal_whs"
exit 0
