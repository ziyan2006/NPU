`timescale 1ns/1ps

module audio_key_debounce #(
  parameter integer CLOCK_HZ = 11_289_600,
  parameter integer DEBOUNCE_MS = 20
) (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic key_ni,
  output logic key_down_o,
  output logic press_event_o
);
  localparam integer DEBOUNCE_CYCLES =
    (CLOCK_HZ * DEBOUNCE_MS + 999) / 1000;
  localparam integer COUNTER_WIDTH = $clog2(DEBOUNCE_CYCLES + 1);

  (* async_reg = "true" *) logic key_sync1_q;
  (* async_reg = "true" *) logic key_sync2_q;
  logic stable_key_q;
  logic [COUNTER_WIDTH-1:0] stable_count_q;

  initial begin
    if (CLOCK_HZ <= 0 || DEBOUNCE_MS <= 0)
      $error("audio_key_debounce parameters must be positive");
  end

  assign key_down_o = !stable_key_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      key_sync1_q <= 1'b1;
      key_sync2_q <= 1'b1;
      stable_key_q <= 1'b1;
      stable_count_q <= '0;
      press_event_o <= 1'b0;
    end else begin
      key_sync1_q <= key_ni;
      key_sync2_q <= key_sync1_q;
      press_event_o <= 1'b0;
      if (key_sync2_q == stable_key_q) begin
        stable_count_q <= '0;
      end else if (stable_count_q == DEBOUNCE_CYCLES - 1) begin
        stable_key_q <= key_sync2_q;
        stable_count_q <= '0;
        if (!key_sync2_q)
          press_event_o <= 1'b1;
      end else begin
        stable_count_q <= stable_count_q + 1'b1;
      end
    end
  end
endmodule
