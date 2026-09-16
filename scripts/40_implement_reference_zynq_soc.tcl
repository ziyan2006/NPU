# Place and route the generic XC7Z020 reference SoC without generating a
# bitstream.  The preceding reference project is intentionally board-neutral;
# actual DDR/MIO configuration still has to replace the generic PS settings.
# Usage:
#   vivado -mode batch -source scripts/40_implement_reference_zynq_soc.tcl -- \
#     ?project_directory? ?report_directory?

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_dir [expr {$argc >= 1
  ? [file normalize [lindex $argv 0]]
  : [file join $repo_root hardware build reference_zynq_soc]}]
set report_dir [expr {$argc >= 2
  ? [file normalize [lindex $argv 1]]
  : [file join $repo_root hardware reports vivado_2026_1 \
      reference_zynq_soc_impl]}]
set project_file [file join $project_dir reference_zynq_soc.xpr]
if {![file exists $project_file]} {
  error "reference project missing; run scripts/39_create_reference_zynq_soc.tcl first"
}
file mkdir $report_dir

open_project $project_file
set run [get_runs impl_1]
set run_status [get_property STATUS $run]
if {$run_status eq "Not started phys_opt_design (Post-Route)"} {
  puts "NPU_REFERENCE_IMPL_INFO continuing existing routed checkpoint"
} elseif {$run_status ne "Not started"} {
  reset_run $run
}
set_property strategy Performance_ExplorePostRoutePhysOpt $run
launch_runs $run -to_step {phys_opt_design (Post-Route)} -jobs 4
wait_on_run $run
set run_status [get_property STATUS $run]
if {![string match "*Complete!*" $run_status]} {
  error "reference SoC implementation failed: $run_status"
}

open_run $run
report_utilization -hierarchical -hierarchical_depth 4 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 50 \
  -report_unconstrained -file [file join $report_dir timing_summary.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -ruledeck default -file [file join $report_dir drc.rpt]
report_methodology -file [file join $report_dir methodology.rpt]
report_clock_interaction -file [file join $report_dir clock_interaction.rpt]
report_cdc -details -file [file join $report_dir cdc.rpt]
report_power -file [file join $report_dir power_estimate.rpt]
write_checkpoint -force [file join $project_dir npu_soc_post_route.dcp]

set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set wns [expr {[llength $setup_paths] ? [get_property SLACK \
  [lindex $setup_paths 0]] : "NA"}]
set whs [expr {[llength $hold_paths] ? [get_property SLACK \
  [lindex $hold_paths 0]] : "NA"}]
set failing_setup [llength [get_timing_paths -quiet -delay_type max \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]]
set failing_hold [llength [get_timing_paths -quiet -delay_type min \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]]
set unrouted [llength [get_nets -hierarchical -filter \
  {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]
set critical_drc [llength [get_drc_violations -quiet -filter \
  {SEVERITY == Error || SEVERITY == {Critical Warning}}]]

set summary [open [file join $report_dir implementation_summary.txt] w]
puts $summary "part=[get_property PART [current_project]]"
puts $summary "strategy=[get_property STRATEGY $run]"
puts $summary "status=$run_status"
puts $summary "wns_ns=$wns"
puts $summary "whs_ns=$whs"
puts $summary "failing_setup_paths=$failing_setup"
puts $summary "failing_hold_paths=$failing_hold"
puts $summary "unrouted_nets=$unrouted"
puts $summary "critical_drc=$critical_drc"
close $summary

puts "NPU_REFERENCE_IMPL_RESULT status=$run_status wns=$wns whs=$whs failing_setup=$failing_setup failing_hold=$failing_hold unrouted=$unrouted critical_drc=$critical_drc reports=$report_dir"
if {$wns ne "NA" && $wns < 0.000} {
  error "reference SoC setup timing failed: WNS=$wns ns"
}
if {$whs ne "NA" && $whs < 0.000} {
  error "reference SoC hold timing failed: WHS=$whs ns"
}
if {$unrouted != 0 || $critical_drc != 0} {
  error "reference SoC routing/DRC sign-off failed"
}
close_project
