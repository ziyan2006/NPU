# Read the FSBL image currently resident in OCM into a local build artifact.
# Usage: xsdb scripts/76_dump_navigator_sd_fsbl.tcl <output.bin>
#
# BOOT.BIN inspection on 2026-09-17 reported the FSBL partition length as
# 0x18008 bytes.  This script is read-only with respect to the target.
if {$argc != 1} { error "usage: xsdb scripts/76_dump_navigator_sd_fsbl.tcl <output.bin>" }
set output [file normalize [lindex $argv 0]]
file mkdir [file dirname $output]
set fsbl_bytes 0x18008
set fsbl_words [expr {$fsbl_bytes / 4}]
connect -url tcp:localhost:3121
targets -set -filter {name == "APU"}
set words [mrd -address-space AP0 -force -value 0x00000000 $fsbl_words]
if {[llength $words] != $fsbl_words} {
  error "expected $fsbl_words FSBL words, read [llength $words]"
}
set channel [open $output wb]
fconfigure $channel -translation binary -encoding binary
foreach word $words {
  set value [expr {$word & 0xffffffff}]
  puts -nonewline $channel [binary format cccc \
    [expr {$value & 0xff}] [expr {($value >> 8) & 0xff}] \
    [expr {($value >> 16) & 0xff}] [expr {($value >> 24) & 0xff}]]
}
close $channel
puts "NPU_SD_FSBL_DUMP PASS bytes=$fsbl_bytes output=$output"
