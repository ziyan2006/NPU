# Read-only inspection of the implemented Navigator candidate's PL-to-DDR map.
# Usage: vivado -mode batch -source scripts/54_inspect_navigator_hp0_addressing.tcl
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set project_file [file join $repo_root hardware build navigator_z7020_vendor_v37 \
  reference_zynq_soc.xpr]
if {![file isfile $project_file]} { error "candidate project is missing" }
open_project $project_file
set bd_file [file join [file dirname $project_file] reference_zynq_soc.srcs \
  sources_1 bd npu_soc npu_soc.bd]
open_bd_design $bd_file
puts "NPU_HP0_ADDR_SPACES [get_bd_addr_spaces -quiet]"
puts "NPU_HP0_ADDR_SEGS [get_bd_addr_segs -quiet]"
puts "NPU_HP0_NPU_MASTER_SPACE [get_bd_addr_spaces -quiet -of_objects \
  [get_bd_intf_pins /stem_npu_0/M_AXI_MEM]]"
set hp0_segs [get_bd_addr_segs -quiet -of_objects \
  [get_bd_intf_pins /stem_npu_0/M_AXI_MEM]]
puts "NPU_HP0_NPU_MASTER_SEGS $hp0_segs"
foreach segment $hp0_segs {
  puts "NPU_HP0_SEGMENT name=$segment offset=[get_property OFFSET $segment] range=[get_property RANGE $segment]"
}
foreach pin [list /processing_system7_0/S_AXI_HP0_ACLK \
                  /processing_system7_0/S_AXI_HP0_ARESETN \
                  /stem_npu_0/aclk /stem_npu_0/aresetn] {
  set object [get_bd_pins -quiet $pin]
  puts "NPU_HP0_PIN pin=$pin net=[get_bd_nets -quiet -of_objects $object]"
}
foreach pin [get_bd_pins -quiet -hier -filter {NAME =~ "*aresetn*"}] {
  set reset_net [get_bd_nets -quiet -of_objects $pin]
  puts "NPU_HP0_RESET_PIN pin=$pin net=$reset_net pins=[get_bd_pins -quiet -of_objects $reset_net]"
}
close_project
