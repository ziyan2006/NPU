# Apply only the candidate PS7 post-configuration after an SD boot.
#
# Usage: xsdb scripts/73_recover_navigator_sd_pl_post_config.tcl <ps7_init.tcl>
#
# This diagnostic deliberately does not call ps7_init: reinitializing DDR
# while an application is executing from DDR is unsafe.  It only uses the
# generated ps7_post_config sequence (PL level shifters and FPGA reset release)
# and reads the NPU ID through DAP afterwards.  It never writes boot media.
if {$argc != 1} {
  error "usage: xsdb scripts/73_recover_navigator_sd_pl_post_config.tcl <ps7_init.tcl>"
}
set init_file [file normalize [lindex $argv 0]]
if {![file isfile $init_file]} { error "missing ps7_init.tcl: $init_file" }
set csr_base 0x43c00000
connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
namespace eval xsdb [list source $init_file]
puts "NPU_SD_POST_CONFIG applying candidate ps7_post_config only"
namespace eval xsdb {ps7_post_config}
if {[catch {set id [lindex [mrd -address-space AP0 -force -value $csr_base] 0]} issue]} {
  error "NPU CSR still unreadable after ps7_post_config: $issue"
}
puts [format "NPU_SD_POST_CONFIG IP_ID=0x%08X" $id]
if {$id != 0x3155504E} { error "unexpected NPU IP_ID" }
puts "NPU_SD_POST_CONFIG PASS"
