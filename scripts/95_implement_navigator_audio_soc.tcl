# Implement, report and export the Navigator NPU plus audio SoC.
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_dir [file join $repo_root hardware build navigator_z7020_audio]
set report_dir [file join $repo_root hardware reports vivado_2026_1 \
  navigator_z7020_audio_impl]
set export_dir [file join $repo_root hardware build navigator_z7020_audio_export]
set project_file [file join $project_dir navigator_audio_soc.xpr]
if {![file isfile $project_file]} {
  error "audio project missing; run scripts/94_create_navigator_audio_soc.tcl"
}
file mkdir $report_dir
file mkdir $export_dir
open_project $project_file
set run [get_runs impl_1]
if {[get_property STATUS $run] ne "Not started"} { reset_run $run }
set_property strategy Performance_ExplorePostRoutePhysOpt $run
launch_runs $run -to_step write_bitstream -jobs 4
wait_on_run $run
set status [get_property STATUS $run]
if {![string match "*Complete!*" $status]} {
  error "audio SoC implementation failed: $status"
}
open_run $run
report_utilization -hierarchical -hierarchical_depth 5 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 50 -report_unconstrained \
  -file [file join $report_dir timing_summary.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -ruledeck default -file [file join $report_dir drc.rpt]
report_cdc -details -file [file join $report_dir cdc.rpt]

set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
set wns [expr {[llength $setup] ? [get_property SLACK [lindex $setup 0]] : "NA"}]
set whs [expr {[llength $hold] ? [get_property SLACK [lindex $hold 0]] : "NA"}]
set failing_setup [llength [get_timing_paths -quiet -delay_type max \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]]
set failing_hold [llength [get_timing_paths -quiet -delay_type min \
  -slack_lesser_than 0.000 -max_paths 100000 -nworst 1]]
set unrouted [llength [get_nets -hierarchical -filter \
  {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]]
set critical_drc [llength [get_drc_violations -quiet -filter \
  {SEVERITY == Error || SEVERITY == {Critical Warning}}]]
# Vivado expands each logical IOBUF into one IBUF plus one OBUFT when the
# linked design is opened. Exactly two OBUFTs therefore proves that SCL and
# SDA retained their open-drain tri-state paths through OOC synthesis.
set iobuf_count [llength [get_cells -hierarchical -filter {REF_NAME == OBUFT}]]

set summary [open [file join $report_dir implementation_summary.txt] w]
puts $summary "part=[get_property PART [current_project]]"
puts $summary "status=$status"
puts $summary "wns_ns=$wns"
puts $summary "whs_ns=$whs"
puts $summary "failing_setup_paths=$failing_setup"
puts $summary "failing_hold_paths=$failing_hold"
puts $summary "unrouted_nets=$unrouted"
puts $summary "critical_drc=$critical_drc"
puts $summary "iobuf_count=$iobuf_count"
close $summary

open_bd_design [get_files */audio_soc.bd]
set address_file [open [file join $report_dir address_map.txt] w]
foreach segment [get_bd_addr_segs -of_objects \
  [get_bd_addr_spaces processing_system7_0/Data]] {
  puts $address_file "[get_property NAME $segment] \
    [format 0x%08X [get_property OFFSET $segment]] \
    [format 0x%08X [get_property RANGE $segment]]"
}
close $address_file
close_bd_design [current_bd_design]

if {$wns eq "NA" || $wns < 0 || $whs eq "NA" || $whs < 0 \
    || $unrouted != 0 || $critical_drc != 0 || $iobuf_count != 2} {
  error "audio implementation sign-off failed"
}
set run_bit [file join $project_dir navigator_audio_soc.runs impl_1 \
  audio_soc_wrapper.bit]
if {![file isfile $run_bit]} { error "bitstream missing: $run_bit" }
file copy -force $run_bit [file join $export_dir stem_npu_audio_navigator_z7020.bit]
set_property platform.design_intent.embedded true [current_project]
write_hw_platform -fixed -include_bit -force \
  -file [file join $export_dir stem_npu_audio_navigator_z7020.xsa]
puts "NAVIGATOR_AUDIO_IMPL_RESULT status=$status wns=$wns whs=$whs"
close_project
