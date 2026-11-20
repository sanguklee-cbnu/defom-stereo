// Simulation-only replacement for the correlation block's single-clock RAM.
// A simultaneous read and write at one address returns the pre-write value.
`timescale 1ns/1ps
module SimpleDP_Block_RAM #(
    parameter DW = 1,
    parameter AW = 1,
    parameter DEPTH = 1
) (
    input wire clk,
    input wire c_en_a,
    input wire w_en_a,
    input wire [AW-1:0] addr_a,
    input wire [DW-1:0] d_in_a,
    input wire c_en_b,
    input wire [AW-1:0] addr_b,
    output reg [DW-1:0] d_out_b
);
    reg [DW-1:0] words [0:DEPTH-1];

    always @(posedge clk) begin
        if (c_en_b)
            d_out_b <= words[addr_b];
        if (c_en_a && w_en_a)
            words[addr_a] <= d_in_a;
    end
endmodule
