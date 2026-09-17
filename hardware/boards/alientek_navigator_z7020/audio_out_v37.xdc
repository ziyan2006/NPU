set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports sys_clk]
create_clock -period 20.000 -name sys_clk [get_ports sys_clk]

# FCLK_CLK0 and the WM8960 master clock come from independent oscillators.
# Their crossings are implemented with Gray pointers/two-flop synchronizers
# and are audited separately with report_cdc.
set_clock_groups -asynchronous \
  -group [get_clocks clk_fpga_0] \
  -group [get_clocks clk_out1_audio_soc_audio_clock_0]

set_property -dict {PACKAGE_PIN L14 IOSTANDARD LVCMOS33 PULLUP true} [get_ports key_n]
set_property -dict {PACKAGE_PIN E18 IOSTANDARD LVCMOS33 PULLUP true} [get_ports aud_scl]
set_property -dict {PACKAGE_PIN F17 IOSTANDARD LVCMOS33 PULLUP true} [get_ports aud_sda]
set_property -dict {PACKAGE_PIN E19 IOSTANDARD LVCMOS33} [get_ports aud_mclk]
set_property -dict {PACKAGE_PIN M18 IOSTANDARD LVCMOS33} [get_ports aud_bclk]
set_property -dict {PACKAGE_PIN G18 IOSTANDARD LVCMOS33} [get_ports aud_dac_lrclk]
set_property -dict {PACKAGE_PIN G17 IOSTANDARD LVCMOS33} [get_ports aud_dacdat]
