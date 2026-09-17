# JTAG-only contiguous DDR read/write test for the Navigator NPU candidate.
# Usage:
#   xsdb scripts/48_stress_navigator_ddr.tcl <ps7_init.tcl> <npu.bit> <MiB> ?base?
# The optional base is a 4 KiB-aligned physical DDR address, defaulting to
# 0x1000_0000. The test programs PL
# through volatile JTAG, initializes PS/DDR from the supplied NPU XSA, and
# never accesses QSPI, eMMC, SD, or other persistent storage.

if {$argc != 3 && $argc != 4} {
  error "usage: xsdb scripts/48_stress_navigator_ddr.tcl <ps7_init.tcl> <npu.bit> <MiB 1..64> ?base?"
}
set init_file [file normalize [lindex $argv 0]]
set bit_file [file normalize [lindex $argv 1]]
set mib [lindex $argv 2]
if {![file isfile $init_file] || ![file isfile $bit_file]} {
  error "PS7 initialization file or JTAG bitstream is missing"
}
if {![string is integer -strict $mib] || $mib < 1 || $mib > 64} {
  error "MiB must be an integer in 1..64"
}

set base 0x10000000
if {$argc == 4} {
  set base [lindex $argv 3]
}
if {[catch {expr {$base}} base_value] || $base_value < 0x10000000 || ($base_value % 4096) != 0} {
  error "base must be a 4 KiB-aligned DDR address at or above 0x10000000"
}
set base $base_value
set total_words [expr {$mib * 1024 * 1024 / 4}]
set block_words 1024
set block_count [expr {$total_words / $block_words}]

puts "NPU_DDR_STRESS connecting; range=[format 0x%08X $base] + ${mib}MiB"
connect -url tcp:localhost:3121
targets -set -filter {name == "xc7z020"}
fpga -file $bit_file
if {[fpga -state] ne "FPGA is configured"} {
  error "JTAG bitstream configuration did not complete"
}
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
catch {stop}
namespace eval xsdb [list source $init_file]
namespace eval xsdb {ps7_init}
namespace eval xsdb {ps7_post_config}
set ip_id [lindex [mrd -force -value 0x43c00000] 0]
if {$ip_id != 0x3155504E} {
  error "NPU IP_ID mismatch before DDR test"
}

# DAP AP0 bypasses Cortex-A9 cache/MMU.  Values are intentionally repeated
# within each block, allowing XSDB's count argument to issue contiguous writes.
targets -set -filter {name == "APU"}
foreach pattern {0xA5A55A5A 0x5A5AA5A5} {
  puts [format "NPU_DDR_STRESS phase pattern=0x%08X" $pattern]
  for {set block 0} {$block < $block_count} {incr block} {
    set address [expr {$base + $block * $block_words * 4}]
    mwr -address-space AP0 -force $address $pattern $block_words
    set words [mrd -address-space AP0 -force -value $address $block_words]
    set offset 0
    foreach actual $words {
      if {$actual != $pattern} {
        set failing_address [expr {$address + $offset * 4}]
        error [format "DDR mismatch address=0x%08X pattern=0x%08X actual=0x%08X" \
          $failing_address $pattern $actual]
      }
      incr offset
    }
    if {($block + 1) % 64 == 0 || $block + 1 == $block_count} {
      puts [format "NPU_DDR_STRESS progress pattern=0x%08X blocks=%u/%u" \
        $pattern [expr {$block + 1}] $block_count]
    }
  }
}
puts "NPU_DDR_STRESS PASS ${mib}MiB, two patterns, direct physical DAP access"
