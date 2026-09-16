# Build and synthesize a board-independent Zynq-7000 integration reference.
# This proves AXI, clock, reset and interrupt connectivity.  It deliberately
# does not generate a bitstream: PS DDR/MIO settings must come from the actual
# board preset before implementation.
# Usage:
#   vivado -mode batch -source scripts/39_create_reference_zynq_soc.tcl -- \
#     ?part? ?ip_repository? ?project_directory? ?board_profile? ?report_directory?

set part [expr {$argc >= 1 ? [lindex $argv 0] : "xc7z020clg400-1"}]
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set ip_repository [expr {$argc >= 2
  ? [file normalize [lindex $argv 1]]
  : [file join $repo_root hardware build ip_repo]}]
set project_dir [expr {$argc >= 3
  ? [file normalize [lindex $argv 2]]
  : [file join $repo_root hardware build reference_zynq_soc]}]
set board_profile [expr {$argc >= 4 && [lindex $argv 3] ne ""
  ? [file normalize [lindex $argv 3]]
  : ""}]
set report_dir [expr {$argc >= 5
  ? [file normalize [lindex $argv 4]]
  : [file join $repo_root hardware reports vivado_2026_1 \
      reference_zynq_soc]}]
set component_xml [file join $ip_repository stem_npu_1_0 component.xml]
if {![file exists $component_xml]} {
  error "packaged NPU IP missing; run scripts/38_package_npu_ip.tcl first"
}

file mkdir $project_dir
create_project -force reference_zynq_soc $project_dir -part $part
set_property ip_repo_paths $ip_repository [current_project]
update_ip_catalog

create_bd_design npu_soc
set ps [create_bd_cell -type ip -vlnv \
  xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} $ps
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
  CONFIG.PCW_EN_CLK0_PORT {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
  CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
  CONFIG.PCW_IRQ_F2P_INTR {1} \
] $ps

# A board profile may add PS DDR/MIO settings after the common NPU ports have
# been enabled.  Profiles are intentionally explicit Tcl files so that a
# vendor-exported preset can replace the provisional profile without editing
# the integration script.
set profile_name "board_independent"
if {$board_profile ne ""} {
  if {![file exists $board_profile]} {
    error "board profile does not exist: $board_profile"
  }
  source $board_profile
  if {![llength [info procs npu_apply_board_ps]]} {
    error "board profile must define npu_apply_board_ps"
  }
  npu_apply_board_ps $ps
  if {[info exists ::npu_board_profile_name]} {
    set profile_name $::npu_board_profile_name
  } else {
    set profile_name [file tail $board_profile]
  }
}

set npu [create_bd_cell -type ip -vlnv \
  ziyan2006.github.io:npu:stem_npu:1.0 stem_npu_0]

# GP0 gives software access to the 4 KiB CSR aperture.  HP0 is the NPU's
# non-coherent 64-bit path to task images and tensor storage in PS DDR.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
  Master {/processing_system7_0/M_AXI_GP0} \
  Clk {/processing_system7_0/FCLK_CLK0} \
] [get_bd_intf_pins $npu/S_AXI_CTRL]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
  Master {/stem_npu_0/M_AXI_MEM} \
  Clk {/processing_system7_0/FCLK_CLK0} \
] [get_bd_intf_pins $ps/S_AXI_HP0]

connect_bd_net [get_bd_pins $npu/irq_o] [get_bd_pins $ps/IRQ_F2P]

assign_bd_address
set csr_segments [get_bd_addr_segs -quiet $npu/S_AXI_CTRL/*]
if {[llength $csr_segments] != 1} {
  error "expected one NPU CSR address segment, got $csr_segments"
}
assign_bd_address -force -offset 0x43C00000 -range 0x00001000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] $csr_segments

validate_bd_design
if {$board_profile ne "" && [llength [info procs npu_validate_board_ps]]} {
  npu_validate_board_ps $ps
}
save_bd_design
set bd_file [get_files [file join $project_dir reference_zynq_soc.srcs \
  sources_1 bd npu_soc npu_soc.bd]]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top npu_soc_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
if {$synth_status ne "synth_design Complete!"} {
  error "reference SoC synthesis failed: $synth_status"
}

open_run synth_1
file mkdir $report_dir
report_utilization -file [file join $report_dir utilization_synth.rpt]
report_clock_interaction -file [file join $report_dir clock_interaction.rpt]
report_cdc -details -file [file join $report_dir cdc_synth.rpt]
report_drc -ruledeck default -file [file join $report_dir drc_synth.rpt]

set ctrl_interconnects [get_bd_cells -quiet -hierarchical -filter \
  {VLNV =~ "xilinx.com:ip:*connect*:*"}]
set csr_mapping [get_bd_addr_segs -of_objects [get_bd_addr_spaces $ps/Data] \
  -filter {OFFSET == 0x43C00000}]
puts "NPU_REFERENCE_SOC_RESULT part=$part profile=$profile_name status=$synth_status csr=$csr_mapping interconnects=$ctrl_interconnects project=$project_dir reports=$report_dir"
close_project
