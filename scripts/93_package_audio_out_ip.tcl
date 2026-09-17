# Package the WM8960 audio output subsystem as AXI4-Lite IP.
set part [expr {$argc >= 1 ? [lindex $argv 0] : "xc7z020clg400-2"}]
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set ip_root [expr {$argc >= 2 ? [file normalize [lindex $argv 1]] : \
  [file join $repo_root hardware build ip_repo audio_out_1_0]}]
set project_dir [expr {$argc >= 3 ? [file normalize [lindex $argv 2]] : \
  [file join $repo_root hardware build package_audio_out_ip]}]

file mkdir [file dirname $ip_root]
file mkdir $project_dir
create_project -force package_audio_out_ip $project_dir -part $part

set list_path [file join $repo_root hardware rtl audio audio_rtl.f]
set input [open $list_path r]
set sources {}
while {[gets $input line] >= 0} {
  set line [string trim $line]
  if {$line eq "" || [string index $line 0] eq "#"} { continue }
  lappend sources [file join $repo_root $line]
}
close $input
add_files -norecurse $sources
set_property file_type SystemVerilog [get_files $sources]
set_property top audio_out_axi [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -force -root_dir $ip_root -vendor ziyan2006.github.io \
  -library audio -taxonomy /Audio -import_files -set_current true
set core [ipx::current_core]
set_property name audio_out $core
set_property display_name {STEM WM8960 Audio Output} $core
set_property description \
  {AXI audio FIFO, STEM mixer, WM8960 setup and 44.1 kHz I2S output} $core
set_property version 1.0 $core
set_property core_revision 1 $core
set_property supported_families {zynq Production} $core

foreach {inferred normalized} {
  s_axi_audio S_AXI_AUDIO
  aclk ACLK
  aresetn ARESETN
} {
  set bus [ipx::get_bus_interfaces -quiet $inferred -of_objects $core]
  if {[llength $bus] != 1} {
    error "audio IP packaging did not infer $inferred: $bus"
  }
  set_property name $normalized $bus
}
set clock_bus [ipx::get_bus_interfaces ACLK -of_objects $core]
set reset_bus [ipx::get_bus_interfaces ARESETN -of_objects $core]
set associated_busif [ipx::get_bus_parameters -quiet ASSOCIATED_BUSIF \
  -of_objects $clock_bus]
if {[llength $associated_busif] == 0} {
  set associated_busif [ipx::add_bus_parameter ASSOCIATED_BUSIF $clock_bus]
}
set_property value {S_AXI_AUDIO} $associated_busif
set associated_reset [ipx::get_bus_parameters -quiet ASSOCIATED_RESET \
  -of_objects $clock_bus]
if {[llength $associated_reset] == 0} {
  set associated_reset [ipx::add_bus_parameter ASSOCIATED_RESET $clock_bus]
}
set_property value {ARESETN} $associated_reset
set reset_polarity [ipx::get_bus_parameters -quiet POLARITY \
  -of_objects $reset_bus]
if {[llength $reset_polarity] == 0} {
  set reset_polarity [ipx::add_bus_parameter POLARITY $reset_bus]
}
set_property value ACTIVE_LOW $reset_polarity

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
if {![ipx::check_integrity -quiet $core]} {
  error "packaged audio IP failed integrity check"
}
puts "AUDIO_IP_PACKAGE_RESULT vlnv=[get_property VLNV $core] root=$ip_root"
close_project
