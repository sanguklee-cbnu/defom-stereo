`timescale 1ns/1ps
`default_nettype none
(* use_dsp = "yes" *)
// INT8 * INT8 -> INT32 MAC (signed), 1-cycle prod + sync-accumulate
module pe_int8_mac #(
  parameter integer OUTW = 32
)(
  input  wire                   clk,
  input  wire                   rstn,        // active-low sync reset

  input  wire                   in_valid,    // ??? MAC ???
  input  wire                   first,       // ??? MAC?? ??? ? MAC
  input  wire                   tile_last,   // ??? MAC?? ??? ?????? MAC
  input  wire                   a_is_qlog,

  input  wire      signed [7:0] a,
  input  wire      signed [7:0] b,

  output reg                    out_valid,
  input  wire                   out_ready,
  output reg signed [OUTW-1:0]  out_data
);

  // ---------------- MUL ----------------
  // ---------------- MUL ----------------
  // a_is_qlog=0: signed INT8 'a' (original)
  // a_is_qlog=1: 'a[3:0]' encodes qlog where a_real = 2^{-qlog} (always positive)
  wire [3:0] qlog = a[3:0];
  wire [16:0] a_q16_u17 = (17'd65536 >> qlog);       // Q0.16

  wire signed [17:0] a18_int8 = {{10{a[7]}}, a};
  wire signed [17:0] a18_q16  = $signed({1'b0, a_q16_u17});
  wire signed [17:0] a18      = a_is_qlog ? a18_q16 : a18_int8;
  wire signed [17:0] b18 = {{10{b[7]}}, b};
  (* use_dsp = "yes" *)
  wire signed [35:0] prod36 = a18 * b18;
  wire signed [47:0] mul48  = {{12{prod36[35]}}, prod36};

  // ---------------- ACC ----------------
  reg  signed [47:0] acc48;
  wire signed [47:0] acc_src  = first ? 48'sd0 : acc48;
  wire signed [47:0] acc_next = acc_src + mul48;   // DSP adder 1??

  always @(posedge clk) begin
    if (!rstn) begin
      acc48     <= 0;
      out_data  <= 0;
      out_valid <= 1'b0;
    end else begin
      // ???? ???????? ????
      if (in_valid) begin
        acc48 <= acc_next;
      end
      // out_data / out_valid ??? ????
      if (in_valid &&tile_last) begin
        // ???? ??????? ??? ????? ???? ???
        out_data  <= acc_next;
        out_valid <= 1'b1;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end
    end
  end

endmodule

`default_nettype wire
