`timescale 1ns/1ps
`default_nettype none

// 32x32 PE array (INT8 GEMM tile)
// - in_valid ????: C[r,c] += A_row[r] * B_col[c]
// - first = ??? ????? ? MAC ??????????? 1
// - tile_last = ??? ????? ?????? MAC ????????? 1
// - PE?? tile_last ??????? ???? ????? out_data?? ????,
//   ???? ????????? out_valid=1 ?? ????? drain ????
// - ????? col-major drain (WRITEBACK/GOUT col-packed ????):
//   out_col_idx = col index (0..31)
//   out_col_data_flat = [row0..31] (32 lanes)
module pe_array #(
  parameter integer OUTW = 32
)(
  input  wire                   clk,
  input  wire                   nRst,

  input  wire                   in_valid,
  input  wire                   first,
  input  wire                   tile_last,


  input  wire                   a_is_qlog,
  input  wire [32*8-1:0]        a_rows_flat,    // [row*8 +: 8]
  input  wire [32*8-1:0]        b_cols_flat,    // [col*8 +: 8]

  output reg                    out_col_valid,      // ???? ??? (col vector)
  input  wire                   out_col_ready,      // ????? ready
  output reg  [4:0]             out_col_idx,        // col index (0..31)
  output reg  [32*OUTW-1:0]     out_col_data_flat,  // col vector data (row0..31)

  output reg                    busy,
  output reg                    done
);

  localparam integer M   = 32;
  localparam integer N   = 32;
  localparam integer NUM = 1024;
  localparam integer GROUP_SIZE = 8;
  localparam integer GROUPS     = 4;

  wire [NUM-1:0]          pe_v;
  wire [NUM*OUTW-1:0]     pe_d_flat;

  // --------------------------
  // input bundle pipeline
  // - Keep PE-consumed controls/data aligned by registering the full bundle
  //   together at the array boundary.
  // - This targets feeder/memory-controller -> systolic_array route-heavy paths
  //   without changing the external handshake contract.
  // --------------------------
  reg                    in_valid_q;
  reg                    first_q;
  reg                    tile_last_q;
  // Four regional copies keep each PE-data register close to an 8-lane
  // row/column group. They capture the original array input on the same edge,
  // preserving latency while avoiding one 32-PE broadcast per data register.
  (* keep = "true", dont_touch = "true" *) reg       a_is_qlog_group_q [0:31][0:GROUPS-1];
  (* keep = "true", dont_touch = "true" *) reg [7:0] a_row_group_q     [0:31][0:GROUPS-1];
  (* keep = "true", dont_touch = "true" *) reg [7:0] b_col_group_q     [0:31][0:GROUPS-1];
  integer qrow;
  integer qcol;
  integer qgroup;

  always @(posedge clk) begin
    if (!nRst) begin
      in_valid_q    <= 1'b0;
      first_q       <= 1'b0;
      tile_last_q   <= 1'b0;
    end else begin
      in_valid_q    <= in_valid;
      first_q       <= first;
      tile_last_q   <= tile_last;
    end
  end

  // Datapath copies need no reset: in_valid_q is held low during reset and
  // every valid PE cycle uses values captured on the immediately prior edge.
  always @(posedge clk) begin
    for (qrow = 0; qrow < 32; qrow = qrow + 1) begin
      for (qgroup = 0; qgroup < GROUPS; qgroup = qgroup + 1) begin
        if (a_is_qlog && a_rows_flat[qrow*8 + 4]) begin
          a_is_qlog_group_q[qrow][qgroup] <= 1'b0;
          a_row_group_q[qrow][qgroup] <= 8'd0;
        end else if (a_is_qlog) begin
          a_is_qlog_group_q[qrow][qgroup] <= 1'b1;
          a_row_group_q[qrow][qgroup] <=
              {4'd0, (4'd15 - a_rows_flat[qrow*8 +: 4])};
        end else begin
          a_is_qlog_group_q[qrow][qgroup] <= 1'b0;
          a_row_group_q[qrow][qgroup] <= a_rows_flat[qrow*8 +: 8];
        end
      end
    end
    for (qcol = 0; qcol < 32; qcol = qcol + 1) begin
      for (qgroup = 0; qgroup < GROUPS; qgroup = qgroup + 1) begin
        b_col_group_q[qcol][qgroup] <= b_cols_flat[qcol*8 +: 8];
      end
    end
  end

  // tile_last 1-cycle delay:
  reg tile_last_d1;
  always @(posedge clk) begin
    if (!nRst) tile_last_d1 <= 1'b0;
    else       tile_last_d1 <= tile_last_q;
  end

  // --------------------------
  // drain state (COL-major)
  // --------------------------
  reg        streaming;
  reg [4:0]  cur_col;     // 0..31: ???? col index

  wire fire     = streaming && out_col_valid && out_col_ready;
  wire last_vec = (cur_col == 5'd31);

  wire start_drain = (!streaming) && tile_last_d1;

  // pop one-hot: ???? cur_col ????? ???? ???? 1
  reg [31:0] pop_oh_col;
  integer t;
  always @* begin
    pop_oh_col = 0;
    if (fire) pop_oh_col[cur_col] = 1'b1;
  end

  // --------------------------
  // PE instances
  // - col-major drain????,
  //   ???? col?? PE?? pop(out_ready) ????
  // --------------------------
  genvar r, c;
  generate
    for (r = 0; r < M; r = r + 1) begin : GEN_ROW
      for (c = 0; c < N; c = c + 1) begin : GEN_COL
        localparam integer IDX = r*N + c;
        localparam integer ROW_GROUP = r / GROUP_SIZE;
        localparam integer COL_GROUP = c / GROUP_SIZE;

        wire pe_out_ready = pop_oh_col[c];
        wire       a_is_qlog_local = a_is_qlog_group_q[r][COL_GROUP];
        wire [7:0] a_row_local = a_row_group_q[r][COL_GROUP];
        wire [7:0] b_col_local = b_col_group_q[c][ROW_GROUP];

        pe_int8_mac #(.OUTW(OUTW)) u_pe (
          .clk       (clk),
          .rstn      (nRst),
          .in_valid  (in_valid_q),
          .first     (first_q),
          .tile_last (tile_last_q),
          .a_is_qlog(a_is_qlog_local),
          .a         (a_row_local),
          .b         (b_col_local),
          .out_valid (pe_v[IDX]),
          .out_ready (pe_out_ready),
          .out_data  (pe_d_flat[IDX*OUTW +: OUTW])
        );
      end
    end
  endgenerate

  // --------------------------
  // packer data select
  // - Separate current-col and next-col packing so out_col_ready/fire
  //   does not feed the wide vec_data_sel mux directly.
  // - Functional behavior is unchanged:
  //   * start_drain loads col 0
  //   * during streaming, current payload stays held when fire=0
  //   * on fire && !last_vec, next payload loads col(cur_col+1)
  // --------------------------
  wire [4:0] col_for_packer_cur  = streaming ? cur_col : 5'd0;
  wire [4:0] col_for_packer_next = cur_col + 5'd1;

  reg [32*OUTW-1:0] vec_data_cur;
  reg               vec_valid_cur;
  reg [32*OUTW-1:0] vec_data_next;
  reg               vec_valid_next;

  integer i;
  always @* begin
    vec_data_cur  = 0;
    vec_valid_cur = 1'b1;
    vec_data_next = 0;
    vec_valid_next= 1'b1;

    // col vector: rows 0..31
    for (i = 0; i < 32; i = i + 1) begin
      // current column = cur_col (or 0 before streaming starts)
      vec_valid_cur = vec_valid_cur & pe_v[(i*32) + col_for_packer_cur];
      vec_data_cur[i*OUTW +: OUTW] =
        pe_d_flat[((i*32 + col_for_packer_cur)*OUTW) +: OUTW];

      // next column = cur_col + 1 (used only on fire && !last_vec)
      vec_valid_next = vec_valid_next & pe_v[(i*32) + col_for_packer_next];
      vec_data_next[i*OUTW +: OUTW] =
        pe_d_flat[((i*32 + col_for_packer_next)*OUTW) +: OUTW];
    end
  end

  // --------------------------
  // output registers
  // --------------------------
  always @(posedge clk) begin
    if (!nRst) begin
      streaming         <= 1'b0;
      cur_col           <= 5'd0;

      out_col_valid     <= 1'b0;
      out_col_idx       <= 5'd0;
      out_col_data_flat <= 0;

      busy              <= 1'b0;
      done              <= 1'b0;
    end else begin
      done <= 1'b0;

      if (!streaming) begin
        busy          <= 1'b0;
        out_col_valid <= 1'b0;
        out_col_idx   <= 5'd0;

        if (start_drain) begin
          streaming         <= 1'b1;
          busy              <= 1'b1;
          cur_col           <= 5'd0;

          out_col_idx       <= 5'd0;
          out_col_data_flat <= vec_data_cur;   // col 0 at drain start
          out_col_valid     <= vec_valid_cur;
        end

      end else begin
        busy <= 1'b1;

        if (fire) begin
          if (last_vec) begin
            streaming     <= 1'b0;
            busy          <= 1'b0;
            out_col_valid <= 1'b0;
            done          <= 1'b1;
          end else begin
            cur_col           <= cur_col + 5'd1;
            out_col_idx       <= cur_col + 5'd1;
            out_col_data_flat <= vec_data_next; // next column prepacked
            out_col_valid     <= vec_valid_next;
          end
        end
        // fire=0??? ???? payload ????
      end
    end
  end

endmodule

`default_nettype wire
