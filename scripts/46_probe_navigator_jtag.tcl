# One-time, JTAG-only PL download, PS initialization, and NPU ID readback.
# Optional third argument "ddr" tests and restores two DDR words via DAP.
# Does not access boot flash.

if {$argc != 2 && $argc != 3} {
  error "usage: xsdb scripts/46_probe_navigator_jtag.tcl <NPU XSA ps7_init.tcl> <NPU bitstream> ?ddr?"
}
if {$argc == 3 && [lindex $argv 2] ne "ddr"} {
  error "the only optional third argument is ddr"
}
set init_file [file normalize [lindex $argv 0]]
if {![file isfile $init_file]} {
  error "PS7 initialization file is missing: $init_file"
}
set bit_file [file normalize [lindex $argv 1]]
if {![file isfile $bit_file]} {
  error "NPU bitstream is missing: $bit_file"
}

puts "NPU_JTAG_PROBE connecting to localhost hw_server"
connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
puts "NPU_JTAG_PROBE programming PL through volatile JTAG (not flash)"
fpga -file $bit_file
puts "NPU_JTAG_PROBE [fpga -state]"
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
if {[catch {stop} stop_result]} {
  # Re-running the probe on an already suspended JTAG-booted core is fine.
  set stop_message [string tolower $stop_result]
  if {![string match "*already stopped*" $stop_message] &&
      ![string match "*already suspended*" $stop_message]} {
    error "could not suspend Cortex-A9 #0: $stop_result"
  }
}

puts "NPU_JTAG_PROBE initializing PS clocks, MIO, and DDR from exported NPU XSA"
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init}
namespace eval xsdb {ps7_post_config}

# XSDB has no platform memory map loaded in this standalone probe, so it
# blocks PL AXI addresses by policy. The address is checked against our RTL
# CSR map above; -force overrides only that debugger-side access filter.
set id [lindex [mrd -force -value 0x43c00000] 0]
puts [format "NPU_JTAG_PROBE ip_id=0x%08X" $id]
if {$id != 0x3155504E} {
  error "NPU IP_ID mismatch; expected 0x3155504E, got $id"
}
set rtl_version [lindex [mrd -force -value 0x43c00004] 0]
set isa_version [lindex [mrd -force -value 0x43c00008] 0]
set status [lindex [mrd -force -value 0x43c00018] 0]
puts [format "NPU_JTAG_PROBE rtl=0x%08X isa=0x%08X status=0x%08X" \
  $rtl_version $isa_version $status]
if {$rtl_version != 0x00010000 || $isa_version != 0x00010000 ||
    ($status & 1) == 0 || ($status & 0x12) != 0} {
  error "NPU version/idle status mismatch"
}
puts "NPU_JTAG_PROBE PASS PS initialization and NPU CSR readback"

if {$argc == 3} {
  # Use the DAP AHB-AP physical address space, bypassing CPU MMU/caches.
  # Save and restore each word; this is a small connectivity check, not a
  # 1 GiB DDR stability or bandwidth test.
  targets -set -filter {name == "APU"}
  foreach address {0x10000000 0x10001000} {
    set saved [lindex [mrd -address-space AP0 -force -value $address] 0]
    set failure ""
    foreach pattern {0xA5A55A5A 0x5A5AA5A5} {
      if {[catch {mwr -address-space AP0 -force $address $pattern} issue]} {
        set failure "write $address failed: $issue"
        break
      }
      if {[catch {set actual [lindex [mrd -address-space AP0 -force -value $address] 0]} issue]} {
        set failure "read $address failed: $issue"
        break
      }
      if {$actual != $pattern} {
        set failure "DDR mismatch at $address: expected $pattern, got $actual"
        break
      }
    }
    if {[catch {mwr -address-space AP0 -force $address $saved} issue]} {
      error "DDR original word restore failed at $address: $issue"
    }
    if {$failure ne ""} {
      error $failure
    }
    puts "NPU_JTAG_PROBE DDR word PASS at $address (two patterns, original restored)"
  }
  puts "NPU_JTAG_PROBE PASS two-word DDR connectivity test; bulk DDR still untested"
}
