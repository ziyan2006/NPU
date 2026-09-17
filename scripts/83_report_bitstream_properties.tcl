# Read-only report of bitstream-generation properties from a routed checkpoint.
# Usage: vivado -mode batch -source scripts/83_report_bitstream_properties.tcl -tclargs <checkpoint>
if {$argc != 1} {
  error "usage: 83_report_bitstream_properties.tcl <checkpoint>"
}
set checkpoint [file normalize [lindex $argv 0]]
if {![file isfile $checkpoint]} {
  error "missing checkpoint: $checkpoint"
}
open_checkpoint $checkpoint
puts "BITSTREAM_PROPERTY_REPORT checkpoint=$checkpoint"
foreach property [lsort [list_property [current_design]]] {
  if {[string match "BITSTREAM.*" $property] || $property in {
      CFGBVS CONFIG_MODE CONFIG_VOLTAGE PART}} {
    puts "$property=[get_property $property [current_design]]"
  }
}
close_design
