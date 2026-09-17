`timescale 1ns/1ps

module wm8960_i2c_master #(
  parameter integer CLOCK_HZ = 100_000_000,
  parameter integer I2C_HZ = 250_000,
  parameter logic [6:0] SLAVE_ADDRESS = 7'h1a
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        start_i,
  input  logic [15:0] word_i,
  input  logic        sda_i,
  output logic        busy_o,
  output logic        done_o,
  output logic        ack_o,
  output logic        scl_drive_low_o,
  output logic        sda_drive_low_o
);
  localparam integer QUARTER_CYCLES = CLOCK_HZ / (I2C_HZ * 4);
  localparam integer DIV_WIDTH = (QUARTER_CYCLES <= 1)
    ? 1 : $clog2(QUARTER_CYCLES);

  typedef enum logic [3:0] {
    ST_IDLE, ST_START, ST_BIT_LOW, ST_BIT_HIGH, ST_ACK_LOW, ST_ACK_HIGH,
    ST_STOP_LOW, ST_STOP_HIGH, ST_STOP_RELEASE
  } state_t;

  state_t state_q;
  logic [DIV_WIDTH-1:0] divider_q;
  logic [23:0] transaction_q;
  logic [1:0] byte_index_q;
  logic [2:0] bit_index_q;
  logic ack_all_q;
  logic [7:0] current_byte;
  logic tick;

  initial begin
    if (CLOCK_HZ <= 0 || I2C_HZ <= 0 || QUARTER_CYCLES <= 0)
      $error("wm8960_i2c_master clock parameters are invalid");
  end

  always_comb begin
    case (byte_index_q)
      2'd0: current_byte = transaction_q[23:16];
      2'd1: current_byte = transaction_q[15:8];
      default: current_byte = transaction_q[7:0];
    endcase
    tick = divider_q == QUARTER_CYCLES - 1;
    scl_drive_low_o = 1'b0;
    sda_drive_low_o = 1'b0;
    case (state_q)
      ST_START: sda_drive_low_o = 1'b1;
      ST_BIT_LOW: begin
        scl_drive_low_o = 1'b1;
        sda_drive_low_o = !current_byte[bit_index_q];
      end
      ST_BIT_HIGH: sda_drive_low_o = !current_byte[bit_index_q];
      ST_ACK_LOW: scl_drive_low_o = 1'b1;
      ST_STOP_LOW: begin
        scl_drive_low_o = 1'b1;
        sda_drive_low_o = 1'b1;
      end
      ST_STOP_HIGH: sda_drive_low_o = 1'b1;
      default: begin end
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= ST_IDLE;
      divider_q <= '0;
      transaction_q <= '0;
      byte_index_q <= '0;
      bit_index_q <= 3'd7;
      ack_all_q <= 1'b1;
      busy_o <= 1'b0;
      done_o <= 1'b0;
      ack_o <= 1'b0;
    end else begin
      done_o <= 1'b0;
      if (state_q == ST_IDLE) begin
        divider_q <= '0;
        if (start_i) begin
          transaction_q <= {{SLAVE_ADDRESS, 1'b0}, word_i};
          byte_index_q <= 2'd0;
          bit_index_q <= 3'd7;
          ack_all_q <= 1'b1;
          ack_o <= 1'b0;
          busy_o <= 1'b1;
          state_q <= ST_START;
        end
      end else if (tick) begin
        divider_q <= '0;
        case (state_q)
          ST_START: state_q <= ST_BIT_LOW;
          ST_BIT_LOW: state_q <= ST_BIT_HIGH;
          ST_BIT_HIGH: begin
            if (bit_index_q == 0)
              state_q <= ST_ACK_LOW;
            else begin
              bit_index_q <= bit_index_q - 1'b1;
              state_q <= ST_BIT_LOW;
            end
          end
          ST_ACK_LOW: state_q <= ST_ACK_HIGH;
          ST_ACK_HIGH: begin
            if (sda_i) begin
              ack_all_q <= 1'b0;
              state_q <= ST_STOP_LOW;
            end else if (byte_index_q == 2) begin
              state_q <= ST_STOP_LOW;
            end else begin
              byte_index_q <= byte_index_q + 1'b1;
              bit_index_q <= 3'd7;
              state_q <= ST_BIT_LOW;
            end
          end
          ST_STOP_LOW: state_q <= ST_STOP_HIGH;
          ST_STOP_HIGH: state_q <= ST_STOP_RELEASE;
          ST_STOP_RELEASE: begin
            state_q <= ST_IDLE;
            busy_o <= 1'b0;
            done_o <= 1'b1;
            ack_o <= ack_all_q;
          end
          default: state_q <= ST_IDLE;
        endcase
      end else begin
        divider_q <= divider_q + 1'b1;
      end
    end
  end
endmodule
