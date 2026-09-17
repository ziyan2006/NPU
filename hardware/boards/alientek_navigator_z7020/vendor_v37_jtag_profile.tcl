# Board-specific PS7 settings for the ALIENTEK Navigator Z7020 V3.7
# (WM8960) reference package.  The proprietary XCI is kept locally in
# vendor_reference_local/ and is never committed to the public repository.
# This profile is for a first JTAG/DDR smoke test, not a verified boot image.

set ::npu_board_profile_name "alientek_navigator_z7020_vendor_v37_jtag"
set ::npu_vendor_ps7_xci [file join [file dirname [file normalize [info script]]] \
  vendor_reference_local ps7 system_processing_system7_0_0.xci]

proc npu_vendor_ps7_values {} {
  if {![file isfile $::npu_vendor_ps7_xci]} {
    error "vendor V3.7 PS7 XCI is missing: $::npu_vendor_ps7_xci"
  }
  set input [open $::npu_vendor_ps7_xci r]
  set xml [read $input]
  close $input

  set values [dict create]
  set matches [regexp -all -inline \
    {<spirit:configurableElementValue spirit:referenceId="PARAM_VALUE\.([^"]+)">([^<]*)</spirit:configurableElementValue>} \
    $xml]
  foreach {tag name value} $matches {
    dict set values $name $value
  }
  foreach {key expected} {
    PCW_CRYSTAL_PERIPHERAL_FREQMHZ 33.333333
    PCW_UIPARAM_DDR_MEMORY_TYPE {DDR 3 (Low Voltage)}
    PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125}
    PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit}
    PCW_UIPARAM_DDR_FREQ_MHZ 533.333333
    PCW_UART0_PERIPHERAL_ENABLE 1
    PCW_UART0_UART0_IO {MIO 14 .. 15}
  } {
    if {![dict exists $values $key] || [dict get $values $key] ne $expected} {
      error "unexpected V3.7 vendor XCI value for $key"
    }
  }
  set count 0
  foreach key [dict keys $values] {
    if {[string match "PCW_UIPARAM_DDR_*" $key]} { incr count }
  }
  if {$count != 72} {
    error "vendor V3.7 PS7 XCI expected 72 DDR fields, found $count"
  }
  return $values
}

proc npu_apply_board_ps {ps} {
  set part [string tolower [get_property PART [current_project]]]
  if {$part ne "xc7z020clg400-2"} {
    error "Navigator Z7020 V3.7 reference requires xc7z020clg400-2, got $part"
  }
  set values [npu_vendor_ps7_values]

  # Set the DDR type and catalog entry before applying the remaining timing
  # and board-delay fields; changing these two can reset dependent PS7 fields.
  set_property -dict [list \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ [dict get $values PCW_CRYSTAL_PERIPHERAL_FREQMHZ] \
    CONFIG.PCW_UIPARAM_DDR_MEMORY_TYPE [dict get $values PCW_UIPARAM_DDR_MEMORY_TYPE] \
    CONFIG.PCW_UIPARAM_DDR_PARTNO [dict get $values PCW_UIPARAM_DDR_PARTNO] \
  ] $ps
  set ddr_settings [list]
  foreach key [lsort [dict keys $values]] {
    if {![string match "PCW_UIPARAM_DDR_*" $key]} { continue }
    if {$key in {PCW_UIPARAM_DDR_MEMORY_TYPE PCW_UIPARAM_DDR_PARTNO}} { continue }
    lappend ddr_settings CONFIG.$key [dict get $values $key]
  }
  set_property -dict $ddr_settings $ps
  set_property -dict [list \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_UART0_IO {MIO 14 .. 15} \
    CONFIG.PCW_UART0_BAUD_RATE {115200} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_ENET1_PERIPHERAL_ENABLE {0} \
  ] $ps
}

proc npu_expect_vendor_ps {ps property expected} {
  set actual [get_property $property $ps]
  if {$actual ne $expected} {
    error "V3.7 PS7 mismatch: $property expected '$expected', got '$actual'"
  }
}

proc npu_validate_board_ps {ps} {
  foreach {property expected} {
    PCW_CRYSTAL_PERIPHERAL_FREQMHZ 33.333333
    PCW_UIPARAM_DDR_MEMORY_TYPE {DDR 3 (Low Voltage)}
    PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125}
    PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit}
    PCW_UIPARAM_DDR_FREQ_MHZ 533.333333
    PCW_DDR_RAM_HIGHADDR 0x3FFFFFFF
    PCW_UART0_PERIPHERAL_ENABLE 1
    PCW_UART0_UART0_IO {MIO 14 .. 15}
    PCW_UART0_BAUD_RATE 115200
    PCW_USE_M_AXI_GP0 1
    PCW_USE_S_AXI_HP0 1
    PCW_S_AXI_HP0_DATA_WIDTH 64
    PCW_FPGA0_PERIPHERAL_FREQMHZ 100.000000
    PCW_QSPI_PERIPHERAL_ENABLE 0
    PCW_SD0_PERIPHERAL_ENABLE 0
    PCW_ENET0_PERIPHERAL_ENABLE 0
    PCW_ENET1_PERIPHERAL_ENABLE 0
  } {
    npu_expect_vendor_ps $ps CONFIG.$property $expected
  }
  foreach index {0 1 2 3} {
    npu_expect_vendor_ps $ps CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY$index 0.25
  }
}
