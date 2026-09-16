# Provisional minimal PS7 profile for the ALIENTEK Navigator ZYNQ-7020.
#
# Intended use: first JTAG/UART hardware smoke test only.  It deliberately
# leaves QSPI, eMMC, Ethernet, USB and audio disabled.  Replace this profile
# with a preset exported from the exact board revision before making a boot
# image or a release bitstream.

set ::npu_board_profile_name "alientek_navigator_z7020_minimal_jtag_provisional"

proc npu_apply_board_ps {ps} {
  set part [string tolower [get_property PART [current_project]]]
  if {$part ne "xc7z020clg400-2"} {
    error "Navigator Z7020 candidate requires xc7z020clg400-2, got $part"
  }

  # Public ALIENTEK guide values: 33.333 MHz PS crystal, UART0 on MIO 14/15,
  # and the Vivado-compatible MT41J256M16 RE-125 entry for the two physical
  # NT5CC256M16 devices (32-bit total, 1 GiB).
  set_property -dict [list \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {33.333333} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41J256M16 RE-125} \
    CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_UART0_IO {MIO 14 .. 15} \
    CONFIG.PCW_UART0_BAUD_RATE {115200} \
  ] $ps
}

proc npu_expect_ps_property {ps property expected} {
  set actual [get_property $property $ps]
  if {$actual ne $expected} {
    error "PS7 profile mismatch: $property expected '$expected', got '$actual'"
  }
}

proc npu_validate_board_ps {ps} {
  npu_expect_ps_property $ps CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ 33.333333
  npu_expect_ps_property $ps CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41J256M16 RE-125}
  npu_expect_ps_property $ps CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH {32 Bit}
  npu_expect_ps_property $ps CONFIG.PCW_DDR_RAM_HIGHADDR 0x3FFFFFFF
  npu_expect_ps_property $ps CONFIG.PCW_UART0_PERIPHERAL_ENABLE 1
  npu_expect_ps_property $ps CONFIG.PCW_UART0_UART0_IO {MIO 14 .. 15}
  npu_expect_ps_property $ps CONFIG.PCW_UART0_BAUD_RATE 115200
  npu_expect_ps_property $ps CONFIG.PCW_USE_M_AXI_GP0 1
  npu_expect_ps_property $ps CONFIG.PCW_USE_S_AXI_HP0 1
  npu_expect_ps_property $ps CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64
  npu_expect_ps_property $ps CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100.000000

  foreach disabled {
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE
    CONFIG.PCW_ENET1_PERIPHERAL_ENABLE
  } {
    npu_expect_ps_property $ps $disabled 0
  }
}
