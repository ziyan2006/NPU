# Create and synthesize the Navigator V3.7 NPU plus WM8960 audio SoC.
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set part "xc7z020clg400-2"
set ip_repository [file join $repo_root hardware build ip_repo]
set project_dir [file join $repo_root hardware build navigator_z7020_audio]
set report_dir [file join $repo_root hardware reports vivado_2026_1 \
  navigator_z7020_audio_synth]
set board_profile [file join $repo_root hardware boards \
  alientek_navigator_z7020 vendor_v37_jtag_profile.tcl]
set xdc [file join $repo_root hardware boards alientek_navigator_z7020 \
  audio_out_v37.xdc]

foreach required [list \
  [file join $ip_repository stem_npu_1_0 component.xml] \
  [file join $ip_repository audio_out_1_0 component.xml] \
  $board_profile $xdc] {
  if {![file isfile $required]} { error "missing audio SoC input: $required" }
}

file mkdir $project_dir
create_project -force navigator_audio_soc $project_dir -part $part
set_property ip_repo_paths $ip_repository [current_project]
update_ip_catalog
add_files -fileset constrs_1 -norecurse $xdc

create_bd_design audio_soc
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
source $board_profile
npu_apply_board_ps $ps

set npu [create_bd_cell -type ip -vlnv \
  ziyan2006.github.io:npu:stem_npu:1.0 stem_npu_0]
set audio [create_bd_cell -type ip -vlnv \
  ziyan2006.github.io:audio:audio_out:1.0 audio_out_0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
  Master {/processing_system7_0/M_AXI_GP0} \
  Clk {/processing_system7_0/FCLK_CLK0} \
] [get_bd_intf_pins $npu/S_AXI_CTRL]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
  Master {/processing_system7_0/M_AXI_GP0} \
  Clk {/processing_system7_0/FCLK_CLK0} \
] [get_bd_intf_pins $audio/S_AXI_AUDIO]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
  Master {/stem_npu_0/M_AXI_MEM} \
  Clk {/processing_system7_0/FCLK_CLK0} \
] [get_bd_intf_pins $ps/S_AXI_HP0]
connect_bd_net [get_bd_pins $npu/irq_o] [get_bd_pins $ps/IRQ_F2P]

set audio_clock [create_bd_cell -type ip -vlnv \
  xilinx.com:ip:clk_wiz:6.0 audio_clock]
set_property -dict [list \
  CONFIG.PRIMITIVE {MMCM} \
  CONFIG.PRIM_IN_FREQ {50.000} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {11.289602856} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {false} \
] $audio_clock
set sys_clk [create_bd_port -dir I -type clk -freq_hz 50000000 sys_clk]
connect_bd_net $sys_clk [get_bd_pins $audio_clock/clk_in1]
connect_bd_net [get_bd_pins $audio_clock/clk_out1] \
  [get_bd_pins $audio/audio_mclk_i]
connect_bd_net [get_bd_pins $audio_clock/locked] \
  [get_bd_pins $audio/audio_clock_locked_i]

foreach {name direction pin} {
  key_n I key_n_i
  aud_mclk O aud_mclk_o
  aud_bclk O aud_bclk_o
  aud_dac_lrclk O aud_dac_lrclk_o
  aud_dacdat O aud_dacdat_o
  aud_scl IO aud_scl_io
  aud_sda IO aud_sda_io
} {
  set port [create_bd_port -dir $direction $name]
  connect_bd_net $port [get_bd_pins $audio/$pin]
}

assign_bd_address
set npu_seg [get_bd_addr_segs -quiet $npu/S_AXI_CTRL/*]
set audio_seg [get_bd_addr_segs -quiet $audio/S_AXI_AUDIO/*]
if {[llength $npu_seg] != 1 || [llength $audio_seg] != 1} {
  error "unexpected CSR segments npu=$npu_seg audio=$audio_seg"
}
assign_bd_address -force -offset 0x43C00000 -range 0x00001000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] $npu_seg
assign_bd_address -force -offset 0x43C10000 -range 0x00001000 \
  -target_address_space [get_bd_addr_spaces $ps/Data] $audio_seg

validate_bd_design
npu_validate_board_ps $ps
save_bd_design
set bd_file [get_files [file join $project_dir navigator_audio_soc.srcs \
  sources_1 bd audio_soc audio_soc.bd]]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top audio_soc_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
set status [get_property STATUS [get_runs synth_1]]
if {$status ne "synth_design Complete!"} {
  error "audio SoC synthesis failed: $status"
}
open_run synth_1
file mkdir $report_dir
report_utilization -hierarchical -hierarchical_depth 4 \
  -file [file join $report_dir utilization_hierarchical.rpt]
report_timing_summary -report_unconstrained \
  -file [file join $report_dir timing_summary.rpt]
report_cdc -details -file [file join $report_dir cdc.rpt]
puts "NAVIGATOR_AUDIO_SYNTH_RESULT status=$status project=$project_dir"
close_project
