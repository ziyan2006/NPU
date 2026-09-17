`timescale 1ns/1ps

module wm8960_init #(
  parameter integer CLOCK_HZ = 100_000_000,
  parameter integer I2C_HZ = 250_000,
  parameter integer FAST_START_CYCLES = CLOCK_HZ / 20,
  parameter integer OUTPUT_SETTLE_CYCLES = CLOCK_HZ / 100
) (
  input  logic       clk_i,
  input  logic       rst_ni,
  input  logic       reinit_i,
  input  logic       mclk_stable_i,
  input  logic       sda_i,
  output logic       scl_drive_low_o,
  output logic       sda_drive_low_o,
  output logic       codec_done_o,
  output logic       codec_error_o,
  output logic       codec_muted_o,
  output logic [7:0] config_index_o,
  output logic       last_ack_o
);
  localparam integer WRITE_COUNT = 17;
  localparam integer MAX_WAIT = (FAST_START_CYCLES > OUTPUT_SETTLE_CYCLES)
    ? FAST_START_CYCLES : OUTPUT_SETTLE_CYCLES;
  localparam integer WAIT_WIDTH = (MAX_WAIT <= 1) ? 1 : $clog2(MAX_WAIT + 1);

  typedef enum logic [2:0] {
    INIT_WAIT_MCLK, INIT_LAUNCH, INIT_WAIT_WRITE, INIT_FAST_WAIT,
    INIT_OUTPUT_WAIT, INIT_DONE, INIT_ERROR
  } init_state_t;

  init_state_t state_q;
  logic i2c_start_q;
  logic [15:0] i2c_word;
  logic i2c_busy;
  logic i2c_done;
  logic i2c_ack;
  logic [7:0] index_q;
  logic [WAIT_WIDTH-1:0] wait_count_q;

  function automatic logic [15:0] config_word(input logic [7:0] index);
    begin
      case (index)
        8'd0: config_word = {7'd15, 9'h000};
        8'd1: config_word = {7'd25, 9'h1c0};
        8'd2: config_word = {7'd25, 9'h0c0};
        8'd3: config_word = {7'd5, 9'h008};
        8'd4: config_word = {7'd6, 9'h008};
        8'd5: config_word = {7'd4, 9'h000};
        8'd6: config_word = {7'd7, 9'h00a};
        8'd7: config_word = {7'd8, 9'h1c0};
        8'd8: config_word = {7'd34, 9'h100};
        8'd9: config_word = {7'd37, 9'h100};
        8'd10: config_word = {7'd47, 9'h00c};
        8'd11: config_word = {7'd40, 9'h079};
        8'd12: config_word = {7'd41, 9'h179};
        8'd13: config_word = {7'd51, 9'h080};
        8'd14: config_word = {7'd26, 9'h198};
        8'd15: config_word = {7'd49, 9'h0f7};
        8'd16: config_word = {7'd5, 9'h000};
        default: config_word = 16'd0;
      endcase
    end
  endfunction

  assign i2c_word = config_word(index_q);
  assign config_index_o = index_q;

  wm8960_i2c_master #(
    .CLOCK_HZ(CLOCK_HZ), .I2C_HZ(I2C_HZ), .SLAVE_ADDRESS(7'h1a)
  ) master (
    .clk_i(clk_i), .rst_ni(rst_ni), .start_i(i2c_start_q),
    .word_i(i2c_word), .sda_i(sda_i), .busy_o(i2c_busy),
    .done_o(i2c_done), .ack_o(i2c_ack),
    .scl_drive_low_o(scl_drive_low_o), .sda_drive_low_o(sda_drive_low_o)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= INIT_WAIT_MCLK;
      i2c_start_q <= 1'b0;
      index_q <= 8'd0;
      wait_count_q <= '0;
      codec_done_o <= 1'b0;
      codec_error_o <= 1'b0;
      codec_muted_o <= 1'b1;
      last_ack_o <= 1'b0;
    end else begin
      i2c_start_q <= 1'b0;
      if (reinit_i) begin
        state_q <= INIT_WAIT_MCLK;
        index_q <= 8'd0;
        wait_count_q <= '0;
        codec_done_o <= 1'b0;
        codec_error_o <= 1'b0;
        codec_muted_o <= 1'b1;
        last_ack_o <= 1'b0;
      end else begin
        case (state_q)
          INIT_WAIT_MCLK: begin
            if (mclk_stable_i)
              state_q <= INIT_LAUNCH;
          end
          INIT_LAUNCH: begin
            if (!i2c_busy) begin
              i2c_start_q <= 1'b1;
              state_q <= INIT_WAIT_WRITE;
            end
          end
          INIT_WAIT_WRITE: begin
            if (i2c_done) begin
              last_ack_o <= i2c_ack;
              if (!i2c_ack) begin
                codec_error_o <= 1'b1;
                codec_muted_o <= 1'b1;
                state_q <= INIT_ERROR;
              end else if (index_q == 1) begin
                wait_count_q <= '0;
                state_q <= INIT_FAST_WAIT;
              end else if (index_q == 15) begin
                wait_count_q <= '0;
                state_q <= INIT_OUTPUT_WAIT;
              end else if (index_q == WRITE_COUNT - 1) begin
                codec_done_o <= 1'b1;
                codec_muted_o <= 1'b0;
                state_q <= INIT_DONE;
              end else begin
                index_q <= index_q + 1'b1;
                state_q <= INIT_LAUNCH;
              end
            end
          end
          INIT_FAST_WAIT: begin
            if (wait_count_q == FAST_START_CYCLES - 1) begin
              index_q <= index_q + 1'b1;
              wait_count_q <= '0;
              state_q <= INIT_LAUNCH;
            end else begin
              wait_count_q <= wait_count_q + 1'b1;
            end
          end
          INIT_OUTPUT_WAIT: begin
            if (wait_count_q == OUTPUT_SETTLE_CYCLES - 1) begin
              index_q <= index_q + 1'b1;
              wait_count_q <= '0;
              state_q <= INIT_LAUNCH;
            end else begin
              wait_count_q <= wait_count_q + 1'b1;
            end
          end
          INIT_DONE: begin end
          INIT_ERROR: begin end
          default: state_q <= INIT_WAIT_MCLK;
        endcase
      end
    end
  end
endmodule
