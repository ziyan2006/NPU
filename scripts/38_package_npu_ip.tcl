# Package the synthesizable NPU top as a reusable Vivado IP-XACT component.
# Usage:
#   vivado -mode batch -source scripts/38_package_npu_ip.tcl -- \
#     ?part? ?output_ip_directory? ?project_directory?

set part [expr {$argc >= 1 ? [lindex $argv 0] : "xc7z020clg400-1"}]
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set ip_root [expr {$argc >= 2
  ? [file normalize [lindex $argv 1]]
  : [file join $repo_root hardware build ip_repo stem_npu_1_0]}]
set project_dir [expr {$argc >= 3
  ? [file normalize [lindex $argv 2]]
  : [file join $repo_root hardware build package_npu_ip]}]

file mkdir [file dirname $ip_root]
file mkdir $project_dir
create_project -force package_npu_ip $project_dir -part $part

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

add_files -norecurse $source_files
set_property file_type SystemVerilog [get_files $source_files]
set_property top npu_top [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -force -root_dir $ip_root -vendor ziyan2006.github.io \
  -library npu -taxonomy /Accelerator -import_files -set_current true
set core [ipx::current_core]
set_property name stem_npu $core
set_property display_name {STEM General NPU} $core
set_property description \
  {INT8 general NPU with AXI4 memory master and AXI4-Lite control} $core
set_property vendor_display_name {ziyan2006} $core
set_property company_url {https://github.com/ziyan2006/NPU} $core
set_property version 1.0 $core
set_property core_revision 1 $core
set_property supported_families {zynq Production} $core

# Port naming follows Vivado's AXI inference convention.  Treat missing or
# misnamed interfaces as a packaging failure instead of producing a subtly
# unusable catalog entry.
foreach {inferred_name normalized_name} {
  m_axi_mem M_AXI_MEM
  s_axi_ctrl S_AXI_CTRL
  aclk ACLK
  aresetn ARESETN
} {
  set inferred_bus [ipx::get_bus_interfaces -quiet $inferred_name \
    -of_objects $core]
  if {[llength $inferred_bus] != 1} {
    error "IP packaging did not infer required interface $inferred_name"
  }
  set_property name $normalized_name $inferred_bus
}

set clock_bus ""
set reset_bus ""
set interrupt_bus ""
foreach bus [ipx::get_bus_interfaces -of_objects $core] {
  set bus_type [get_property BUS_TYPE_VLNV $bus]
  if {$bus_type eq "xilinx.com:signal:clock:1.0"} {
    set clock_bus $bus
  } elseif {$bus_type eq "xilinx.com:signal:reset:1.0"} {
    set reset_bus $bus
  } elseif {$bus_type eq "xilinx.com:signal:interrupt:1.0"} {
    set interrupt_bus $bus
  }
}
if {$interrupt_bus eq ""} {
  set interrupt_bus [ipx::add_bus_interface IRQ $core]
  set_property interface_mode master $interrupt_bus
  set_property bus_type_vlnv xilinx.com:signal:interrupt:1.0 $interrupt_bus
  set_property abstraction_type_vlnv \
    xilinx.com:signal:interrupt_rtl:1.0 $interrupt_bus
  set interrupt_map [ipx::add_port_map INTERRUPT $interrupt_bus]
  set_property physical_name irq_o $interrupt_map
}
if {$clock_bus eq "" || $reset_bus eq ""} {
  error "clock/reset inference incomplete"
}

set associated_busif [ipx::get_bus_parameters ASSOCIATED_BUSIF \
  -of_objects $clock_bus]
if {[llength $associated_busif] == 0} {
  set associated_busif [ipx::add_bus_parameter ASSOCIATED_BUSIF $clock_bus]
}
set_property value {S_AXI_CTRL:M_AXI_MEM} $associated_busif
set associated_reset [ipx::get_bus_parameters ASSOCIATED_RESET \
  -of_objects $clock_bus]
if {[llength $associated_reset] == 0} {
  set associated_reset [ipx::add_bus_parameter ASSOCIATED_RESET $clock_bus]
}
set_property value {ARESETN} $associated_reset

set reset_polarity [ipx::get_bus_parameters POLARITY -of_objects $reset_bus]
if {[llength $reset_polarity] == 0} {
  set reset_polarity [ipx::add_bus_parameter POLARITY $reset_bus]
}
set_property value ACTIVE_LOW $reset_polarity

set clock_frequency [ipx::get_bus_parameters FREQ_HZ -of_objects $clock_bus]
if {[llength $clock_frequency] == 0} {
  set clock_frequency [ipx::add_bus_parameter FREQ_HZ $clock_bus]
}
set_property value 100000000 $clock_frequency

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core
set integrity [ipx::check_integrity -quiet $core]
if {!$integrity} {
  error "packaged NPU IP failed the IP-XACT integrity check"
}
set bus_names [lsort [get_property NAME [ipx::get_bus_interfaces -of_objects $core]]]
puts "NPU_IP_PACKAGE_RESULT vlnv=[get_property VLNV $core] part=$part root=$ip_root buses=$bus_names"
close_project
