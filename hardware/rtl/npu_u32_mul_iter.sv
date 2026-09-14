`timescale 1ns/1ps

// Shared radix-4 shift/add multiplier for control-path address arithmetic.
// It intentionally uses no DSP48 resources; one product completes in 16 cycles.
module npu_u32_mul_iter (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        soft_reset_i,
  input  logic        start_i,
  input  logic [31:0] operand_a_i,
  input  logic [31:0] operand_b_i,
  output logic        busy_o,
  output logic        done_o,
  output logic [63:0] result_o
);
  logic [63:0] accumulator_q;
  logic [63:0] multiplicand_q;
  logic [31:0] multiplier_q;
  logic [3:0]  step_q;
  logic [63:0] addend;
  logic [63:0] sum;

  always @* begin
    case (multiplier_q[1:0])
      2'd1: addend = multiplicand_q;
      2'd2: addend = multiplicand_q << 1;
      2'd3: addend = multiplicand_q + (multiplicand_q << 1);
      default: addend = 64'd0;
    endcase
    sum = accumulator_q + addend;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      accumulator_q <= '0;
      multiplicand_q <= '0;
      multiplier_q <= '0;
      step_q <= '0;
      busy_o <= 1'b0;
      done_o <= 1'b0;
      result_o <= '0;
    end else if (soft_reset_i) begin
      accumulator_q <= '0;
      multiplicand_q <= '0;
      multiplier_q <= '0;
      step_q <= '0;
      busy_o <= 1'b0;
      done_o <= 1'b0;
      result_o <= '0;
    end else begin
      done_o <= 1'b0;
      if (start_i && !busy_o) begin
        accumulator_q <= '0;
        multiplicand_q <= {32'd0, operand_a_i};
        multiplier_q <= operand_b_i;
        step_q <= '0;
        busy_o <= 1'b1;
      end else if (busy_o) begin
        accumulator_q <= sum;
        multiplicand_q <= multiplicand_q << 2;
        multiplier_q <= multiplier_q >> 2;
        if (step_q == 4'd15) begin
          result_o <= sum;
          busy_o <= 1'b0;
          done_o <= 1'b1;
        end else begin
          step_q <= step_q + 1'b1;
        end
      end
    end
  end

endmodule
