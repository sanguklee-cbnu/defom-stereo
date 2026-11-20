`timescale 1ns/1ps
`default_nettype none

module writeback_controller #(
    parameter integer M_BLK_NUM   = 7,
    parameter integer M_TOTAL     = 197,
    parameter integer OUT_STRIDE  = 384,
    parameter integer GOUT_AW     = 12,
    parameter integer SHIFT_PART  = 8,
    parameter integer BIAS_LSHIFT = 0,
    parameter integer SCALE_W     = 16,
    parameter signed [SCALE_W-1:0] SCALE_OUT = 16'sd1,
    parameter integer SHIFT_OUT   = 8,

    parameter integer CLS_ALT_EN  = 1,
    parameter signed [SCALE_W-1:0] SCALE_CLS = 16'sd1,
    parameter integer SHIFT_CLS   = 0,

    // SCORE softmax i_exp/log_round parameter
    parameter integer SM_HW_SCALE = 4,

    // IndexSoftmaxShift parameters (must match LUT generation)
    parameter integer SM_LOG2_PREC = 4,
    parameter integer SM_LOG2_FRAC_LUT_SIZE = 128,
    parameter integer SM_QLOG_MAX = 13,
    // Python delivery-contract defaults and per-block constant ROMs.
    // The low nibble stored in SCOREBUF is the delivered qlog code
    // (15-distance). DEFOM 401 compatibility retains the delivered 4-bit
    // code even when IExp underflows. The old Softmax_Array has no exact-zero
    // flag: log2_fixed(0)=0 still participates in the normalized log code.
    parameter integer SM_BLOCKS = 12,
    parameter integer SM_LIS_H_SCALE = 9,
    parameter integer SM_LIS_H_SCALE_POLY = 528,
    parameter integer SM_ROW_SUM_BITS = 26,
    parameter SM_ADAPTER_MUL_MEM = "sm_adapter_mul_s16.mem",
    parameter SM_ADAPTER_SHIFT_MEM = "sm_adapter_shift_u6.mem",
    parameter SM_H_MEM = "sm_h_u4.mem",
    parameter SM_B_MEM = "sm_b_u22.mem",
    parameter SM_EXP_U15_MEM = "softmax_exp_u15.mem",

    // The delivery model uses one 256-entry signed INT8 GELU LUT per block.
    parameter integer FC1_GELU_LUT_ENABLE = 1,
    parameter FC1_GELU_MEM = "gelu_lut_i8.mem",
    // out_proj and ffn2 use signed per-output-channel requant multipliers.
    parameter OPROJ_MUL_MEM = "oproj_mul_s16.mem",
    parameter FC2_MUL_MEM = "fc2_mul_s16.mem"
)(


    input  wire clk,
    input  wire nRst,

    // NEW: layer final ktile (2b)
    input  wire [5:0] layer_final_ktile,
    input  wire       cls_alt_apply, // 1: apply CLS special handling (Patch Embedding only)
    input  wire [11:0] active_token_total,
    input  wire [6:0]  active_mblk_num,
    input  wire [4:0]  active_last_valid_lane,
    input  wire [3:0] layer_id,
    input  wire [3:0] enc_blk_idx,

    // meta (pe_out_valid )
    input  wire [5:0] ktile_idx,
    input  wire [6:0] nblk_idx,
    input  wire [6:0] mblk_idx,
    input  wire [4:0] col_in_blk,

    // final-tile complete pulse + meta
    output reg          final_tile_pulse,
    output reg  [6:0]   final_nblk_idx,
    output reg  [6:0]   final_mblk_idx,

    input  wire             pe_out_valid,
    input  wire             pe_out_ready,
    input  wire [32*32-1:0] pe_out_i32_flat,

    // GOUT read (prev partial)
    output wire         gout_rd_en,
    output wire [GOUT_AW-1:0]  gout_rd_addr,
    input  wire [255:0] gout_rd_data,
    
    // Global IN read (FC2 residual add) - uses IN bank (layer_inbuf_sel)
    output wire         gin_rd_en,
    output wire [GOUT_AW-1:0]  gin_rd_addr,
    input  wire [255:0] gin_rd_data,

    // Bias read (final ktile )
    output wire         bias_rd_en,
    output wire [6:0]   bias_rd_addr,
    input  wire [255:0] bias_rd_data,


    // quant constants (async read LUTRAM)
    output wire        quant_rd_en,
    output wire [5:0]  quant_rd_addr,
    input  wire [255:0] quant_rd_data,
    // GOUT write
    output wire         gout_wr_valid,
    output wire [GOUT_AW-1:0]  gout_wr_addr,
    output wire [255:0] gout_wr_data,
    output wire         gout_wr_final_ktile,
    output wire [3:0]   gout_wr_layer_id,
    output wire         gout_wr_bank_ovr_en,
    output wire         gout_wr_bank_ovr,

    // FC2 no-SQPS INT32 partial-sum buffer.
    // Non-final FC2 K-tiles store full INT32 partial sums here; the final
    // K-tile restores them directly without shift quantization.
    output wire          psum_rd_en,
    output wire [GOUT_AW-1:0]   psum_rd_addr,
    input  wire [1023:0] psum_rd_data,

    output wire          psum_wr_valid,
    output wire [GOUT_AW-1:0]   psum_wr_addr,
    output wire [1023:0] psum_wr_data,

    // SCOREBUF raw-score buffer (ping-pong) R/W for SCORE softmax
    output wire         score_rd_en,
    output wire         score_rd_bank,
    output wire [11:0]  score_rd_addr,
    input  wire [255:0] score_rd_data,

    output wire         score_wr_valid,
    output wire         score_wr_bank,
    output wire [11:0]  score_wr_addr,
    output wire [255:0] score_wr_data,

    // Legacy softmax-output buffer port. The row-interleaved SCORE -> AV path
    // writes qlog softmax results back into SCOREBUF instead.
    output wire         attn_wr_valid,
    output wire [GOUT_AW-1:0] attn_wr_addr,
    output wire [255:0] attn_wr_data,

    output wire         wb_ready,
    output wire         wb_busy
);

    // local input shadow for long-lived stream meta
    reg        in_meta_valid_q;
    reg [5:0]  layer_final_ktile_q;
    reg        cls_alt_apply_q;
    reg [3:0]  layer_id_q;
    reg [5:0]  ktile_idx_q;
    reg [6:0]  nblk_idx_q;
    reg [6:0]  mblk_idx_q;
    reg        quant_word_valid_q;
    reg [255:0] quant_word_q;

    wire [6:0] M_BLK_LAST = active_mblk_num - 7'd1;
    wire [4:0] VALID_LANES_LAST = active_last_valid_lane;
    wire [11:0] TOKEN_LAST_IDX = active_token_total - 12'd1;

    function [M_BLK_NUM-1:0] f_mblk_mask;
        input [6:0] n;
        integer mi;
        begin
            f_mblk_mask = {M_BLK_NUM{1'b0}};
            for (mi = 0; mi < M_BLK_NUM; mi = mi + 1) begin
                if (mi < n) f_mblk_mask[mi] = 1'b1;
            end
        end
    endfunction

    wire [M_BLK_NUM-1:0] active_mblk_mask = f_mblk_mask(active_mblk_num);

    function [GOUT_AW-1:0] f_mblk_stride;
        input [6:0] mblk;
        begin
            f_mblk_stride = mblk * OUT_STRIDE;
        end
    endfunction

    function [GOUT_AW-1:0] calc_word_addr_rowmajor;
        input [6:0] nblk;
        input [4:0] col;
        input [6:0] mblk;
        reg   [11:0] n_global;
        begin
            n_global = {nblk, 5'd0} + col; // 0..191
            calc_word_addr_rowmajor = f_mblk_stride(mblk) + n_global;
        end
    endfunction

    function signed [7:0] sat8;
        input signed [31:0] x;
        begin
            if (x > 32'sd127)       sat8 = 8'sd127;
            else if (x < -32'sd128) sat8 = -8'sd128;
            else                    sat8 = x[7:0];
        end
    endfunction

    // ------------------------------------------------------------
    // Banker's rounding right-shift for signed 64-bit values
    // - rounds to nearest, ties to even (matches torch.round / numpy.rint)
    // ------------------------------------------------------------
    function signed [31:0] round_shift_even_s64;
        input signed [63:0] x;
        input [5:0]         sh;
        reg   signed [63:0] q;
        reg   signed [63:0] rem;
        reg          [63:0] abs_rem;
        reg          [63:0] half;
        reg   signed [63:0] sign_adj;
        begin
            if (sh == 0) begin
                round_shift_even_s64 = x[31:0];
            end else begin
                q   = x >>> sh;
                rem = x - (q <<< sh);
                abs_rem = rem[63] ? (~rem + 64'd1) : rem;
                half    = 64'd1 <<< (sh - 1);
                sign_adj = x[63] ? -64'sd1 : 64'sd1;

                if (abs_rem > half) begin
                    q = q + sign_adj;
                end else if (abs_rem == half) begin
                    if (q[0]) q = q + sign_adj;
                end
                round_shift_even_s64 = q[31:0];
            end
        end
    endfunction






// --------------------------------------------------------------------
// SCORE softmax helpers (ilog/qlog4)
// - i_exp/log_round as in golden Python (integer-only approximation)
// --------------------------------------------------------------------
function [7:0] log_round_u32;
    input [31:0] x;
    integer b;
    reg [5:0] msb;
    reg [5:0] big;
    begin
        if (x == 0) begin
            log_round_u32 = 8'd0;
        end else begin
            msb = 0;
            for (b = 0; b < 32; b = b + 1) begin
                if (x[b]) msb = b[5:0];
            end
            if (msb == 0) big = 0;
            else          big = msb - 1;

            log_round_u32 = {2'd0, msb}; // <= 31
            if (x[big]) log_round_u32 = log_round_u32 + 1;
        end
    end
endfunction

function [31:0] i_exp_int_from_diff;
    input signed [8:0] diff9; // score - row_max (<=0)
    input        [3:0] sm_scale;
    reg signed [15:0] x;
    reg signed [15:0] x_ln2_s;
    reg [15:0]        x_ln2_u;
    reg [5:0]         q6;
    reg [15:0]        q_scaled16;
    reg signed [15:0] r_s;
    reg signed [31:0] z_s;
    reg [31:0]        exp_u;
    integer sh;
    begin
        x = {{7{diff9[8]}}, diff9}; // sign extend to 16
        // x_int_ln2 = -(x + x//2 - x//16)  (floor div for negative via >>>)
        x_ln2_s = x + (x >>> 1) - (x >>> 4);
        x_ln2_s = -x_ln2_s;

        if (x_ln2_s[15]) x_ln2_u = 0;
        else             x_ln2_u = x_ln2_s[15:0];

        // clamp to (2^sm_scale)*13
        if (x_ln2_u > (16'd13 << sm_scale)) x_ln2_u = (16'd13 << sm_scale);

        q6  = x_ln2_u >> sm_scale;

        // IMPORTANT: widen before shifting.
        // In plain Verilog, (q6 << sm_scale) keeps q6's width (6b) and truncates,
        // which corrupts the remainder r and inflates exp_sum (logsum mismatch).
        q_scaled16 = ({10'd0, q6} << sm_scale);
        r_s = -$signed({1'b0, x_ln2_u} - {1'b0, q_scaled16});

        // z = (r + sm_scale)*r + sm_scale
        z_s = ($signed(r_s) + $signed(sm_scale)) * $signed(r_s) + $signed(sm_scale);
        if (z_s[31]) z_s = 0; // clamp min=0

        // exp_int = floor( z * 2^(13 - q - 4) ) = z * 2^(9 - q)
        if (q6 <= 6'd9) begin
            sh    = 9 - q6;
            exp_u = z_s <<< sh;
        end else begin
            sh    = q6 - 9;
            exp_u = z_s >>> sh;
        end

        i_exp_int_from_diff = exp_u;
    end
endfunction

// ------------------------------------------------------------
// IndexSoftmaxShift helper: ilog2 floor (x>=1)
// ------------------------------------------------------------
function [5:0] ilog2_floor_u32;
    input [31:0] x;
    integer b;
    begin
        ilog2_floor_u32 = 0;
        for (b = 0; b < 32; b = b + 1) begin
            if (x[b]) begin
                ilog2_floor_u32 = b[5:0];
            end
        end
    end
endfunction

function log2_next_bit_u32;
    input [31:0] x;
    input [5:0]  log2_q;
    begin
        if ((x == 32'd0) || (log2_q == 6'd0)) begin
            log2_next_bit_u32 = 1'b0;
        end else begin
            log2_next_bit_u32 = x[log2_q - 6'd1];
        end
    end
endfunction

    reg signed [15:0] sm_adapter_mul_rom [0:SM_BLOCKS-1];
    reg        [5:0]  sm_adapter_shift_rom [0:SM_BLOCKS-1];
    reg        [3:0]  sm_h_rom [0:SM_BLOCKS-1];
    reg        [21:0] sm_b_rom [0:SM_BLOCKS-1];
    integer sm_cfg_i;

    initial begin
        for (sm_cfg_i = 0; sm_cfg_i < SM_BLOCKS; sm_cfg_i = sm_cfg_i + 1) begin
            sm_adapter_mul_rom[sm_cfg_i] = 16'sd1;
            sm_adapter_shift_rom[sm_cfg_i] = 6'd0;
            sm_h_rom[sm_cfg_i] = SM_LIS_H_SCALE[3:0];
            sm_b_rom[sm_cfg_i] = SM_LIS_H_SCALE_POLY[21:0];
        end
        $readmemh(SM_ADAPTER_MUL_MEM, sm_adapter_mul_rom);
        $readmemh(SM_ADAPTER_SHIFT_MEM, sm_adapter_shift_rom);
        $readmemh(SM_H_MEM, sm_h_rom);
        $readmemh(SM_B_MEM, sm_b_rom);
    end

    wire [3:0] sm_cfg_idx = (enc_blk_idx < SM_BLOCKS) ? enc_blk_idx : 4'd0;
    wire signed [15:0] sm_adapter_mul = sm_adapter_mul_rom[sm_cfg_idx];
    wire [5:0] sm_adapter_shift = sm_adapter_shift_rom[sm_cfg_idx];
    wire [3:0] sm_lis_h = sm_h_rom[sm_cfg_idx];
    wire [21:0] sm_lis_b = sm_b_rom[sm_cfg_idx];

function [14:0] lis_exp_u15_from_posdiff;
    input signed [8:0] posdiff9; // row_max - score (>=0)
    input signed [15:0] adapter_mul;
    input [5:0] adapter_shift;
    input [3:0] h_scale;
    input [21:0] h_scale_poly;
    reg signed [47:0] centered;
    reg signed [47:0] adapter_product;
    reg signed [47:0] adapter_first_floor;
    reg signed [47:0] x_fixed;
    reg signed [47:0] t_s;
    reg signed [47:0] t_max;
    reg [47:0]        t_u;
    reg [5:0]         q6;
    reg [47:0]        q_scaled;
    reg signed [47:0] r_s;
    reg signed [47:0] z_s;
    reg signed [63:0] exp_s;
    integer sh;
    begin
        // centred = score - row_max, followed by the signed dyadic adapter.
        centered = -$signed({{39{posdiff9[8]}}, posdiff9});
        adapter_product = centered * adapter_mul;
        if (adapter_shift == 0) begin
            x_fixed = adapter_product;
        end
        else begin
            // Python rounded_shift_int64(): nested floor for both signs.
            adapter_first_floor = adapter_product >>> (adapter_shift - 1'b1);
            x_fixed = (adapter_first_floor + 48'sd1) >>> 1;
        end

        t_s = -(x_fixed + (x_fixed >>> 1) - (x_fixed >>> 4));
        t_max = 48'sd13 <<< h_scale;
        if (t_s < 0) t_u = 48'd0;
        else if (t_s > t_max) t_u = t_max[47:0];
        else t_u = t_s[47:0];

        q6 = t_u >> h_scale;
        q_scaled = ({42'd0, q6} << h_scale);
        r_s = -$signed(t_u - q_scaled);
        z_s = r_s + $signed({1'b0, h_scale_poly});

        // Linear IExp: z * 2^(9-q), then zero-underflow and U15 clamp.
        if (q6 <= 6'd9) begin
            sh    = 9 - q6;
            exp_s = z_s <<< sh;
        end else begin
            sh    = q6 - 9;
            exp_s = z_s >>> sh;
        end

        if (exp_s <= 0) begin
            lis_exp_u15_from_posdiff = 15'd0;
        end else if (exp_s > 64'sd32767) begin
            lis_exp_u15_from_posdiff = 15'd32767;
        end else begin
            lis_exp_u15_from_posdiff = exp_s[14:0];
        end
    end
endfunction


        localparam [3:0] L_PATCH = 4'd0;
localparam [3:0] L_QKV_V = 4'd2;
    localparam [3:0] L_SCORE = 4'd4;
    localparam [3:0] L_AV    = 4'd5;
    localparam [3:0] L_O_PROJ = 4'd6;
    localparam [3:0] L_MLP_FC1 = 4'd7;
localparam [3:0] L_MLP_FC2 = 4'd8;
    
localparam [3:0] L_CLS0   = 4'd9;
localparam [3:0] L_CLS1   = 4'd10;
localparam [3:0] L_CLS2   = 4'd11;
localparam [3:0] L_CLS3   = 4'd12;
localparam [3:0] L_CLS4   = 4'd13;
localparam [3:0] L_CLS5   = 4'd14;

localparam [5:0] QIDX_MLP_FC2_AUX0 = 6'd24; // 16 + L_MLP_FC2, shifts 22..44
localparam [5:0] QIDX_MLP_FC2_AUX1 = 6'd40; // 32 + L_MLP_FC2, shifts 44..47
wire meta_capture = pe_out_valid && pe_out_ready && (col_in_blk == 5'd0);
wire meta_release = pe_out_valid && pe_out_ready && (col_in_blk == 5'd31);
wire [5:0] layer_final_ktile_eff = in_meta_valid_q ? layer_final_ktile_q : layer_final_ktile;
wire       cls_alt_apply_eff     = in_meta_valid_q ? cls_alt_apply_q     : cls_alt_apply;
wire [3:0] layer_id_eff          = in_meta_valid_q ? layer_id_q          : layer_id;
wire [5:0] ktile_idx_eff         = in_meta_valid_q ? ktile_idx_q         : ktile_idx;
wire [6:0] nblk_idx_eff          = in_meta_valid_q ? nblk_idx_q          : nblk_idx;
wire [6:0] mblk_idx_eff          = in_meta_valid_q ? mblk_idx_q          : mblk_idx;
wire [255:0] quant_word_eff      = quant_word_valid_q ? quant_word_q : quant_rd_data;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        in_meta_valid_q    <= 1'b0;
        layer_final_ktile_q<= 5'd0;
        cls_alt_apply_q    <= 1'b0;
        layer_id_q         <= 4'd0;
        ktile_idx_q        <= 5'd0;
        nblk_idx_q         <= 7'd0;
        mblk_idx_q         <= 7'd0;
        quant_word_valid_q <= 1'b0;
        quant_word_q       <= 256'd0;
    end else begin
        // Meta cache is per accepted 32-beat output stream. Clear it at the
        // end of the current stream so the next layer/tile/block can recapture
        // fresh layer_id / ktile / nblk / mblk even when pe_out_valid stays high.
        if (!pe_out_valid || meta_release) begin
            in_meta_valid_q    <= 1'b0;
            quant_word_valid_q <= 1'b0;
        end
        if (meta_capture) begin
            in_meta_valid_q     <= 1'b1;
            layer_final_ktile_q <= layer_final_ktile;
            cls_alt_apply_q     <= cls_alt_apply;
            layer_id_q          <= layer_id;
            ktile_idx_q         <= ktile_idx;
            nblk_idx_q          <= nblk_idx;
            mblk_idx_q          <= mblk_idx;
            quant_word_valid_q  <= 1'b1;
            quant_word_q        <= quant_rd_data;
        end
    end
end

wire mode_v = (layer_id_eff == L_QKV_V);
wire mode_patch = (layer_id_eff == L_PATCH);

    wire mode_score = (layer_id_eff == L_SCORE);

    wire mode_av    = (layer_id_eff == L_AV);
    wire mode_oproj = (layer_id_eff == L_O_PROJ);
    wire mode_fc1  = (layer_id_eff == L_MLP_FC1);
    wire mode_fc2  = (layer_id_eff == L_MLP_FC2);


wire mode_cls = ((layer_id_eff >= L_CLS0) && (layer_id_eff <= L_CLS5));
// ------------------------------------------------------------
// Quant constants from quant_rd_data (256b)
// Bit layout (LSB..MSB) must match gen_memhex_quantbuf.py:
//   [15:0]   OUT_MUL   (s16)
//   [31:16]  OUT_OFF   (u16)
//   [36:32]  OUT_SH    (u5)
//   [52:37]  CLS_MUL   (s16)
//   [68:53]  CLS_OFF   (u16)
//   [73:69]  CLS_SH    (u5)
//   [77:74]  SM_SCALE  (u4)
//   [..]     ACC_SH[i] (u5 each) starting at bit 78
//   [208:193] RES_ALIGN_MUL   (s16, dyadic multiplier for residual align, shift_bits=15)
//   [212:209] RES_ALIGN_LSH   (u4, extra left shift before /2^15)
//   [228:213] RES_POST_MUL    (s16, dyadic multiplier for post-residual requant, shift_bits=15)
//   [232:229] RES_POST_LSH    (u4, extra left shift before /2^15)
//   [248:233] RES_POST_OFF    (u16, rounding offset for post-residual requant; recommend 16384)
// ------------------------------------------------------------
wire signed [15:0] q_out_mul = $signed(quant_word_eff[15:0]);
wire [15:0]        q_out_off_u16 = quant_word_eff[31:16];
wire [4:0]         q_out_sh  = quant_word_eff[36:32];

wire signed [15:0] q_cls_mul = $signed(quant_word_eff[52:37]);
wire [15:0]        q_cls_off_u16 = quant_word_eff[68:53];
wire [4:0]         q_cls_sh  = quant_word_eff[73:69];

wire [3:0]         sm_hw_scale_eff = quant_word_eff[77:74];
// Residual align/post-requant params for model_data_HW matching
// These fields are only used in OPROJ residual and FC2 final residual paths.
//  - Align: banker's rounding (torch.round) on dyadic ratio
//      aligned_i32 = round_even( (x_i8 * MUL) << LSH  / 2^15 )
//  - Post: half-up rounding (fixed-point multiply) with programmable offset
//      y_i8 = sat8( (((x_i32 * MUL) << LSH) + OFF) >>> 15 )
// Recommended: OFF = 1<<(15-1) = 16384
localparam integer RES_SHIFT_BITS = 15;

wire signed [15:0] q_res_align_mul    = $signed(quant_word_eff[208:193]);
wire [3:0]         q_res_align_divsh  =        quant_word_eff[212:209];
wire signed [15:0] q_res_post_mul     = $signed(quant_word_eff[228:213]);
wire [3:0]         q_res_post_divsh   =        quant_word_eff[232:229];
wire [15:0]        q_res_post_off_u16 =        quant_word_eff[248:233];
wire signed [47:0] q_res_post_off_s48 = $signed({32'd0, q_res_post_off_u16});

// default offset for shift_bits=15 (half-up)
localparam signed [47:0] RES_OFF_DYADIC_S48 = 48'sd16384;
wire signed [47:0] q_res_post_off_eff_s48 = (q_res_post_off_u16 == 16'd0) ? RES_OFF_DYADIC_S48 : q_res_post_off_s48;







// model_data_HW clamp: if mul is negative, treat as +32767 (matches _clamp_multiplier_modeldata)
wire signed [15:0] q_res_align_mul_eff = q_res_align_mul[15] ? 16'sd32767 : q_res_align_mul;
wire signed [15:0] q_res_post_mul_eff  = q_res_post_mul[15]  ? 16'sd32767 : q_res_post_mul;
// helper: extract 5b shift entry (0..22) from a quant word snapshot.
function [4:0] f_acc_sh_qw;
    input [255:0] qw;
    input [5:0] idx;
    integer bi;
    begin
        if (idx >= 6'd23) begin
            f_acc_sh_qw = 5'd0;
        end else begin
            bi = 78 + (idx << 2) + idx;
            f_acc_sh_qw = qw[bi +: 5];
        end
    end
endfunction

// Small FC2 uses three quant words. The local shift index depends on the
// current ktile because prev-restore can overlap the previous quant word range.
//   base: current ktile 0..22,  shift idx 0..22
//   aux0: current ktile 23..44, shift idx 22..44 -> local 0..22
//   aux1: current ktile 45..47, shift idx 44..46 -> local 0..2
function [5:0] f_acc_sh_local_idx_for_kt;
    input [5:0] cur_kt;
    input [5:0] shift_idx;
    begin
        if (cur_kt >= 6'd45) begin
            f_acc_sh_local_idx_for_kt = shift_idx - 6'd44;
        end else if (cur_kt >= 6'd23) begin
            f_acc_sh_local_idx_for_kt = shift_idx - 6'd22;
        end else begin
            f_acc_sh_local_idx_for_kt = shift_idx;
        end
    end
endfunction

function [4:0] f_acc_sh;
    input [5:0] idx;
    begin
        f_acc_sh = f_acc_sh_qw(quant_word_eff, idx);
    end
endfunction

wire [4:0]         rq_sh  = q_out_sh;
wire [15:0]        rq_off_u16 = q_out_off_u16;
wire signed [31:0]  rq_off_s32 = $signed({16'd0, rq_off_u16});
wire signed [31:0]  cls_off_s32 = $signed({16'd0, q_cls_off_u16});
wire signed [47:0] rq_round_auto_s48 =
    (q_out_sh == 0) ? 48'sd0 : (48'sd1 <<< (q_out_sh - 1'b1));
wire signed [47:0] rq_off_s48  =
    (rq_off_u16 == 0) ? rq_round_auto_s48 : $signed({32'd0, rq_off_u16});
wire signed [47:0] cls_off_s48 = $signed({32'd0, q_cls_off_u16});
    // V-bias cache: 32 x int16 per channel-block (2 x 256b words)
    localparam [1:0] BIAS_IDLE  = 2'd0;
    localparam [1:0] BIAS_LOAD0 = 2'd1;
    localparam [1:0] BIAS_LOAD1 = 2'd2;
    reg  [1:0]   bias_load_state;
    reg  [3:0]   bias_target_blk;
    reg  [3:0]   bias_cached_blk;
    reg          bias_cache_valid;
    reg  [255:0] bias_word_lo;
    reg  [255:0] bias_word_hi;

    // SCORE softmax per-bank bookkeeping
    reg        sm_ctx_vld   [0:1];
    reg        sm_ctx_ready [0:1];
    reg [6:0]  sm_ctx_mblk  [0:1];

    // per-mblk completion bookkeeping (PASS3 done for this mblk)
    reg sm_done [0:M_BLK_NUM-1];

    // SCOREBUF write bank ping-pong
    reg scorebuf_wr_bank_r;
    reg prev_mode_score;

    wire cur_is_final_ktile = (ktile_idx_eff == layer_final_ktile_eff);
    wire cur_is_final_eff = cur_is_final_ktile || mode_fc1;

    // Output word validity for edge token blocks.
// - V proj: last token-block has only VALID_LANES_LAST+1 valid rows
// - SCORE:  last col-block has only VALID_LANES_LAST+1 valid cols
wire out_word_valid = (mode_v) ? !((mblk_idx_eff == M_BLK_LAST) && (col_in_blk > VALID_LANES_LAST)) :
                      (mode_score)        ? !((nblk_idx_eff == M_BLK_LAST) && (col_in_blk > VALID_LANES_LAST)) :
                                          1'b1;

    // Backpressure:
    // - V final-ktile bias load requirement
    wire wb_ready_bias = !(mode_v && cur_is_final_ktile && pe_out_valid && out_word_valid &&
                           ((bias_load_state != BIAS_IDLE) || (!bias_cache_valid) || (bias_cached_blk != nblk_idx_eff)));

    // - SCORE softmax context overflow protection:
    //   do not allow overwriting a SCOREBUF bank that still holds an older mblk context
    wire sm_score_bank_conflict = (mode_score && pe_out_valid && out_word_valid &&
                                   ((scorebuf_wr_bank_r == 1'b0 && sm_ctx_vld[0] && (sm_ctx_mblk[0] != mblk_idx_eff)) ||
                                    (scorebuf_wr_bank_r == 1'b1 && sm_ctx_vld[1] && (sm_ctx_mblk[1] != mblk_idx_eff))));

    assign wb_ready = wb_ready_bias && !sm_score_bank_conflict;

    // Handshake
    // - pe_out_valid is the raw column-stream valid from PE
    // - wb_ready applies backpressure
    wire accept     = pe_out_valid && wb_ready && pe_out_ready;   // (= fire)
    wire accept_eff = accept && out_word_valid;

    wire        core_gout_rd_en;
    wire [GOUT_AW-1:0] core_gout_rd_addr;
    wire        core_gin_rd_en;
    wire [11:0] core_gin_rd_addr_compact;
    wire        core_psum_rd_en;
    wire [GOUT_AW-1:0] core_psum_rd_addr;
    assign core_gout_rd_en   = accept_eff && (ktile_idx_eff != 6'd0) && (!mode_fc1) && (!mode_fc2) && (!mode_patch);
    assign core_gout_rd_addr = calc_word_addr_rowmajor(mode_v ? mblk_idx_eff : nblk_idx_eff,
                                                       col_in_blk,
                                                       mode_v ? nblk_idx_eff : mblk_idx_eff);

    assign core_psum_rd_en   = accept_eff && (mode_fc2 || mode_patch) && (ktile_idx_eff != 6'd0);
    assign core_psum_rd_addr = core_gout_rd_addr;

    // FC2 residual add: read residual (int8) from IN bank on final ktile.
    // Register the global read requests once at the WB boundary.
    // This cuts WB -> mem_ctrl global-bank control routes, but adds one
    // cycle on GOUT/GIN request issue, so stage0b capture is shifted to
    // s0 -> s0a -> s0b.
    assign core_gin_rd_en = accept_eff && mode_fc2 && cur_is_final_ktile;
    // FC2 residuals are staged one output-channel block at a time in
    // SCOREBUF bank1, so only the mblk/column coordinates are retained.
    assign core_gin_rd_addr_compact = {mblk_idx_eff, 5'd0} + {7'd0, col_in_blk};

    reg        gout_rd_en_q;
    reg [GOUT_AW-1:0] gout_rd_addr_q;
    reg        psum_rd_en_q;
    reg [GOUT_AW-1:0] psum_rd_addr_q;
    reg        gin_rd_en_q;
    reg [GOUT_AW-1:0] gin_rd_addr_q;

    assign gout_rd_en   = gout_rd_en_q;
    assign gout_rd_addr = gout_rd_addr_q;
    assign psum_rd_en   = psum_rd_en_q;
    assign psum_rd_addr = psum_rd_addr_q;
    assign gin_rd_en    = gin_rd_en_q;
    assign gin_rd_addr  = gin_rd_addr_q;

    // stage0 regs
    reg        s0_valid;
    // Core writeback pipeline valids (2-cycle) for timing closure
    reg        s1_valid_core;
    reg        s2_valid_core;

    // StageQ8 pipeline (1-cycle): register packed_q8 + meta for SCOREBUF write & softmax PASS1 timing
    reg        s1_valid_q8;
    reg [255:0] s1_packed_q8;
    reg [6:0]  s1_mb_q8;
    reg [6:0]  s1_nb_q8;
    reg [4:0]  s1_col_q8;
    reg [5:0]  s1_kt_q8;
    reg        s1_mode_score_q8;
    reg        s1_mode_fc1_q8;
    reg        s1_is_final_ktile_q8;

    // ------------------------------------------------------------
    // Base Q8 pipeline for 300MHz timing closure
    //  - P1: accumulate + bias into int32 (per lane)
    //  - P2: final requant multiply (registered)
    //  - P3: pack/sat8 (comb) -> StageQ8 regs capture
    // ------------------------------------------------------------
    reg         p1_valid;
    reg  [GOUT_AW-1:0] p1_addr;
    reg  [6:0]  p1_mb, p1_nb;
    reg  [5:0]  p1_col, p1_kt;
    reg  [3:0]  p1_layer_id;
    reg         p1_is_final_ktile;
    reg         p1_is_final_eff;
    reg         p1_is_cls_block;
    reg  [4:0]  p1_shift_store_eff;
    // Preserve the intended writeback pipeline boundaries. Without these
    // attributes Vivado can retime the wide add/shift logic through P1/P2A
    // and recreate a single-cycle s0b-to-DSP path at 200 MHz.
    (* keep = "true", dont_touch = "true" *) reg [32*32-1:0] p1_sum32_flat;
    (* keep = "true", dont_touch = "true" *) reg [32*32-1:0] p1_sum32b_flat;
    // delay residual payload for core pipeline
    reg  [255:0] p1_score_res_data;
    reg  [255:0] p1_gin_res_data;
    reg  signed [15:0] p1_q_res_align_mul_eff;
    reg  [3:0]         p1_q_res_align_divsh;
    reg  signed [15:0] p1_q_res_post_mul_eff;
    reg  [3:0]         p1_q_res_post_divsh;
    reg  signed [47:0] p1_q_res_post_off_eff_s48;

    // Quant constants pipelined with BaseQ8 datapath
    // (cuts s0_layer_id -> quant_rd_data decode -> DSP path, and fixes layer-boundary misalignment)
    reg  signed [15:0] p1_rq_mul;
    reg  [4:0]         p1_rq_sh;
    reg  signed [47:0] p1_rq_off_s48;
    reg  signed [15:0] p1_q_cls_mul;
    reg  [4:0]         p1_q_cls_sh;
    reg  signed [47:0] p1_cls_off_s48;


    // P2A pipeline stage: register operands/meta before DSP multiply (cuts P1->DSP->P2 critical paths)
    reg         p2a_valid;
    reg  [GOUT_AW-1:0] p2a_addr;
    reg  [6:0]  p2a_mb, p2a_nb;
    reg  [5:0]  p2a_col, p2a_kt;
    reg  [3:0]  p2a_layer_id;
    reg         p2a_is_final_ktile;
    reg         p2a_is_final_eff;
    reg         p2a_is_cls_block;
    reg  [4:0]  p2a_shift_store_eff;
    (* keep = "true", dont_touch = "true" *) reg [32*32-1:0] p2a_sum32_flat;
    (* keep = "true", dont_touch = "true" *) reg [32*32-1:0] p2a_sum32b_flat;
    // delay residual payload for core pipeline
    reg  [255:0] p2a_score_res_data;
    reg  [255:0] p2a_gin_res_data;
    reg  signed [15:0] p2a_q_res_align_mul_eff;
    reg  [3:0]         p2a_q_res_align_divsh;
    reg  signed [15:0] p2a_q_res_post_mul_eff;
    reg  [3:0]         p2a_q_res_post_divsh;
    reg  signed [47:0] p2a_q_res_post_off_eff_s48;

    // Quant constants aligned with p2a_* stage
    reg  signed [15:0] p2a_rq_mul;
    reg  [4:0]         p2a_rq_sh;
    reg  signed [47:0] p2a_rq_off_s48;
    reg  signed [15:0] p2a_q_cls_mul;
    reg  [4:0]         p2a_q_cls_sh;
    reg  signed [47:0] p2a_cls_off_s48;

    // Regional copies keep the P2A-to-lane quantization routes local.  The
    // register stage and arithmetic are unchanged; each copy serves 4 lanes.
    localparam integer WB_GROUP_SIZE = 4;
    localparam integer WB_GROUPS     = 8;
    (* keep = "true", dont_touch = "true" *) reg signed [47:0] p2a_rq_off_group [0:WB_GROUPS-1];
    (* keep = "true", dont_touch = "true" *) reg [4:0]         p2a_rq_sh_group  [0:WB_GROUPS-1];
    (* keep = "true", dont_touch = "true" *) reg signed [47:0] p2a_cls_off_group[0:WB_GROUPS-1];
    (* keep = "true", dont_touch = "true" *) reg [4:0]         p2a_cls_sh_group [0:WB_GROUPS-1];

    reg         p2_valid;
    reg  [GOUT_AW-1:0] p2_addr;
    reg  [6:0]  p2_mb, p2_nb;
    reg  [5:0]  p2_col, p2_kt;
    reg  [3:0]  p2_layer_id;
    reg         p2_is_final_ktile;
    reg         p2_is_final_eff;
    reg         p2_is_cls_block;
    reg  [4:0]  p2_shift_store_eff;
    reg  [32*32-1:0] p2_sum32_flat;
    reg  [32*32-1:0] p2_sum32b_flat;
    reg  [32*48-1:0] p2_mul48_flat;
    // delay residual payload for core pipeline
    reg  [255:0] p2_score_res_data;
    reg  [255:0] p2_gin_res_data;
    reg  signed [15:0] p2_q_res_align_mul_eff;
    reg  [3:0]         p2_q_res_align_divsh;
    reg  signed [15:0] p2_q_res_post_mul_eff;
    reg  [3:0]         p2_q_res_post_divsh;
    reg  signed [47:0] p2_q_res_post_off_eff_s48;

    // Quant constants aligned with p2_* (used in P3)
    reg  signed [15:0] p2_rq_mul;
    reg  [4:0]         p2_rq_sh;
    reg  signed [47:0] p2_rq_off_s48;
    reg  signed [15:0] p2_q_cls_mul;
    reg  [4:0]         p2_q_cls_sh;
    reg  signed [47:0] p2_cls_off_s48;

    // Lane-local replicated quant constants for P3 packer (cuts route-heavy scalar fanout into s1_packed_q8)
    (* keep = "true" *) reg signed [47:0] p2_rq_off_lane [0:31];
    (* keep = "true" *) reg [4:0]         p2_rq_sh_lane  [0:31];
    (* keep = "true" *) reg signed [47:0] p2_cls_off_lane[0:31];
    (* keep = "true" *) reg [4:0]         p2_q_cls_sh_lane [0:31];

    // StageQ8 meta extension for core pipeline alignment
    reg  [GOUT_AW-1:0]  s1_addr_q8;
    reg  [3:0]   s1_layer_id_q8;
    reg          s1_is_final_eff_q8;
    reg          s1_is_cls_block_q8;
    reg  [4:0]   s1_shift_store_eff_q8;
    reg  [255:0] s1_score_res_data_q8;
    reg  [255:0] s1_gin_res_data_q8;
    reg  signed [15:0] s1_q_res_align_mul_eff_q8;
    reg  [3:0]         s1_q_res_align_divsh_q8;
    reg  signed [15:0] s1_q_res_post_mul_eff_q8;
    reg  [3:0]         s1_q_res_post_divsh_q8;
    reg  signed [47:0] s1_q_res_post_off_eff_s48_q8;

    // Regional copies for the residual-align multiplier.  A single q8
    // register feeding all 32 lane DSPs becomes route-dominated at 200 MHz.
    localparam integer RES_ALIGN_GROUP_SIZE = 4;
    localparam integer RES_ALIGN_GROUPS     = 8;
    (* keep = "true", dont_touch = "true" *)
    reg signed [15:0] s1_q_res_align_mul_group_q8 [0:RES_ALIGN_GROUPS-1];
    integer res_align_group_i;



    // WB busy: prevent layer advance while PE still streaming or a write is in-flight (1-cycle pipe)
    // + hold busy across SCORE softmax post-processing (3-pass)
    wire sm_busy;
    reg [5:0]  s0_kt;
    reg [6:0]  s0_nb, s0_mb;
    reg [4:0]  s0_col;
    reg [GOUT_AW-1:0] s0_addr;
    reg [32*32-1:0] s0_cur_flat;
    reg [255:0] s0_quant_word;

    // stage0a regs: align metadata with registered GOUT/GIN read requests
    reg        s0a_valid;
    reg [5:0]  s0a_kt;
    reg [6:0]  s0a_nb, s0a_mb;
    reg [4:0]  s0a_col;
    reg [GOUT_AW-1:0] s0a_addr;
    reg [32*32-1:0] s0a_cur_flat;
    reg [255:0] s0a_quant_word;
    reg [3:0]  s0a_layer_id;
    reg [5:0]  s0a_layer_final_ktile;
    reg        s0a_cls_alt_apply;

    // One-cycle delay taps for SCORE/BIAS sync-read payloads.
    reg [255:0] score_rd_data_d1;
    reg [255:0] bias_rd_data_d1;

    // stage0b regs: register BRAM/LUTRAM read data to cut long routes into P1 adder/mul
    reg        s0b_valid;
    reg [5:0]  s0b_kt;
    reg [6:0]  s0b_nb, s0b_mb;
    reg [4:0]  s0b_col;
    reg [GOUT_AW-1:0] s0b_addr;
    reg [32*32-1:0] s0b_cur_flat;
    reg [3:0]  s0b_layer_id;
    reg [5:0]  s0b_layer_final_ktile;
    reg        s0b_cls_alt_apply;
    reg [4:0]  s0b_shift_prev_eff;
    reg [4:0]  s0b_shift_store_eff;
    reg [255:0] s0b_gout_rd_data;
    reg [1023:0] s0b_psum_rd_data;
    reg [255:0] s0b_gin_rd_data;
    reg [255:0] s0b_score_rd_data;
    reg [255:0] s0b_bias_rd_data;

    // stage0 layer/meta regs (cuts layer_id_eff fanout into datapath)
    reg [3:0]  s0_layer_id;
    reg [5:0]  s0_layer_final_ktile;
    reg        s0_cls_alt_apply;

    // stage0 precomputed SQPS shifts (cuts quant->barrel-shift critical paths)
    reg [4:0]  s0_shift_prev_eff;
    reg [4:0]  s0_shift_store_eff;

    // stage0 layer decode (use s0_* only in datapath calculations)
    wire s0_mode_v     = (s0_layer_id == L_QKV_V);
    wire s0_mode_score = (s0_layer_id == L_SCORE);
    wire s0_mode_av    = (s0_layer_id == L_AV);
    wire s0_mode_oproj = (s0_layer_id == L_O_PROJ);
    wire s0_mode_fc1   = (s0_layer_id == L_MLP_FC1);
    wire s0_mode_fc2   = (s0_layer_id == L_MLP_FC2);
    wire s0_mode_cls   = ((s0_layer_id >= L_CLS0) && (s0_layer_id <= L_CLS5));

    // stage0b layer decode (used in P1 computations)
    wire s0b_mode_v     = (s0b_layer_id == L_QKV_V);
    wire s0b_mode_patch = (s0b_layer_id == L_PATCH);
    wire s0b_mode_score = (s0b_layer_id == L_SCORE);
    wire s0b_mode_av    = (s0b_layer_id == L_AV);
    wire s0b_mode_oproj = (s0b_layer_id == L_O_PROJ);
    wire s0b_mode_fc1   = (s0b_layer_id == L_MLP_FC1);
    wire s0b_mode_fc2   = (s0b_layer_id == L_MLP_FC2);
    wire s0b_mode_cls   = ((s0b_layer_id >= L_CLS0) && (s0b_layer_id <= L_CLS5));
    wire s0b_is_final_ktile = (s0b_kt == s0b_layer_final_ktile);
    wire s0b_is_final_eff   = s0b_is_final_ktile || s0b_mode_fc1;
    wire s0b_is_cls_block = s0b_cls_alt_apply && s0b_is_final_ktile && (s0b_mb == 3'd0);
    wire [3:0] bias_lane_idx_b = s0b_col[3:0];

    // Requant constants. The delivered model uses channel-wise multipliers
    // only for out_proj and ffn2; all other layers use the scalar quant word.
    reg signed [15:0] oproj_mul_rom [0:(SM_BLOCKS*384)-1];
    reg signed [15:0] fc2_mul_rom   [0:(SM_BLOCKS*384)-1];
    integer channel_mul_init_i;
    initial begin
        for (channel_mul_init_i = 0; channel_mul_init_i < (SM_BLOCKS*384);
             channel_mul_init_i = channel_mul_init_i + 1) begin
            oproj_mul_rom[channel_mul_init_i] = 16'sd1;
            fc2_mul_rom[channel_mul_init_i] = 16'sd1;
        end
        $readmemh(OPROJ_MUL_MEM, oproj_mul_rom);
        $readmemh(FC2_MUL_MEM, fc2_mul_rom);
    end

    wire [12:0] rq_channel_index =
        ({9'd0, enc_blk_idx} * 13'd384) + {1'b0, s0b_nb, 5'd0} + s0b_col;
    wire signed [15:0] rq_mul =
        (s0b_layer_id == L_O_PROJ) ? oproj_mul_rom[rq_channel_index] :
        (s0b_layer_id == L_MLP_FC2) ? fc2_mul_rom[rq_channel_index] :
                                      q_out_mul;

    // Bias read addr
    wire [6:0] bias_word_idx = s0_mode_fc1 ? {s0_kt, s0_col[4]} :
                               (s0_mode_fc2 ? (7'd96 + {2'd0, s0_nb, s0_col[4]}) :
                                               {2'd0, s0_nb, s0_col[4]});
    wire [3:0] bias_lane_idx = s0_col[3:0];

    wire is_final_ktile = (s0_kt == s0_layer_final_ktile);
    wire is_final_eff = is_final_ktile || s0_mode_fc1;

    
    wire use_bias_q = (!s0_mode_v) && (!s0_mode_score) && (!s0_mode_av);
wire use_bias_q_b = (!s0b_mode_v) && (!s0b_mode_score) && (!s0b_mode_av);
wire bias_rd_en_q   = (s0_valid && is_final_eff && use_bias_q);
wire [6:0] bias_rd_addr_q = bias_word_idx;

    wire       bias_rd_en_v   = (bias_load_state != BIAS_IDLE);
    wire [6:0] bias_rd_addr_v = {2'd0, bias_target_blk, (bias_load_state == BIAS_LOAD1)};

    assign bias_rd_en   = mode_v ? bias_rd_en_v : bias_rd_en_q;
    assign bias_rd_addr = mode_v ? bias_rd_addr_v : bias_rd_addr_q;
// Quant constants: address by s0_layer_id (0..15) to cut layer_id_eff fanout into datapath
// NOTE: ram_1w1r_bias uses async read, so quant_rd_en is informational only.
assign quant_rd_en   = 1'b1;
wire [5:0] quant_rd_addr_fc2 =
    (ktile_idx_eff >= 6'd45) ? QIDX_MLP_FC2_AUX1 :
    (ktile_idx_eff >= 6'd23) ? QIDX_MLP_FC2_AUX0 :
                               {2'd0, layer_id_eff};
assign quant_rd_addr = (layer_id_eff == L_MLP_FC2) ? quant_rd_addr_fc2 : {2'd0, layer_id_eff};
    integer i;
    integer j;
    integer p2r_i;
    integer p2ag_i;
    integer row_base;
    integer valid_lanes;
    reg [255:0] packed_q8;
    reg signed [31:0] cur32, prev32, sum32, sum32b;
    reg signed [7:0]  prev_q8;
    reg signed [15:0] bias16_col;
    reg signed [15:0] bias16_lane;
    reg signed [31:0] qtmp32;
    integer shift_prev_eff;
    integer shift_store_eff;
    reg signed [47:0] mul48;
    reg signed [63:0] mul64;
    reg signed [31:0] res_align32;

    wire is_cls_block = (s0_cls_alt_apply && is_final_ktile && (s0_mb == 3'd0));






    // Bias prefetch/cache for V projection: load 32 biases (2 words) per channel-block
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            bias_load_state <= BIAS_IDLE;
            bias_target_blk <= 0;
            bias_cached_blk <= 0;
            bias_cache_valid <= 1'b0;
        end else if (!mode_v) begin
            // Only meaningful in V mode
            bias_load_state <= BIAS_IDLE;
            bias_cache_valid <= 1'b0;
        end else begin
            case (bias_load_state)
                BIAS_IDLE: begin
                    if (pe_out_valid && out_word_valid && cur_is_final_ktile &&
                        (!bias_cache_valid || (bias_cached_blk != nblk_idx_eff))) begin
                        bias_target_blk <= nblk_idx_eff;
                        bias_cache_valid <= 1'b0;
                        bias_load_state <= BIAS_LOAD0;
                    end
                end
                BIAS_LOAD0: begin
                    bias_word_lo <= bias_rd_data;
                    bias_load_state <= BIAS_LOAD1;
                end
                BIAS_LOAD1: begin
                    bias_word_hi <= bias_rd_data;
                    bias_cached_blk <= bias_target_blk;
                    bias_cache_valid <= 1'b1;
                    bias_load_state <= BIAS_IDLE;
                end
                default: begin
                    bias_load_state <= BIAS_IDLE;
                end
            endcase
        end
    end

    // ------------------------------------------------------------
    // BaseQ8 pipeline (P1/P2/P3): cuts BRAM->DSP->FF critical paths @300MHz
    //   P1: sum32/sum32b accumulation + bias into int32 vectors
    //   P2: final requant multiplier (registered, DSP-friendly)
    //   P3: pack/sat8 (comb) -> StageQ8 regs capture
    // ------------------------------------------------------------
    reg [32*32-1:0] p1_sum32_flat_next;
    reg [32*32-1:0] p1_sum32b_flat_next;
    reg [4:0]       p1_shift_store_eff_next;
    reg [32*48-1:0] p2_mul48_flat_next;

    always @(*) begin
        // defaults
        p1_sum32_flat_next       = 0;
        p1_sum32b_flat_next      = 0;
        p1_shift_store_eff_next  = SHIFT_PART;
        p2_mul48_flat_next       = 0;
        packed_q8                = 0;

        bias16_col = 16'sd0;
        bias16_lane = 16'sd0;
        cur32  = 0;
        prev32 = 0;
        sum32  = 0;
        sum32b = 0;
        prev_q8 = 0;
        qtmp32 = 0;
        mul48  = 0;
        row_base = 0;
        valid_lanes = 0;

        // -----------------------------
        // P1 (from s0_* + gout_rd_data)
        // -----------------------------
        // Q-mode: 1 bias per output channel (col)
        if (use_bias_q_b && s0b_is_final_eff) begin
            bias16_col = $signed(s0b_bias_rd_data[bias_lane_idx_b*16 +: 16]);
        end

        // effective accumulation shift for partial-sum storage/restore
        // - precomputed in stage0 regs to cut quant->barrel-shift critical paths
        p1_shift_store_eff_next = s0b_shift_store_eff;

        for (i = 0; i < 32; i = i + 1) begin
            cur32 = $signed(s0b_cur_flat[i*32 +: 32]);

            prev_q8 = $signed(s0b_gout_rd_data[i*8 +: 8]);
            if (s0b_mode_fc1 || (s0b_kt == 5'd0)) begin
                prev32 = 32'sd0;
            end else if (s0b_mode_fc2 || s0b_mode_patch) begin
                prev32 = $signed(s0b_psum_rd_data[i*32 +: 32]);
            end else if (CLS_ALT_EN && s0b_is_cls_block && (i == 0)) begin
                prev32 = 32'sd0;
            end else begin
                prev32 = ($signed(prev_q8) <<< s0b_shift_prev_eff);
            end

            sum32 = prev32 + cur32;
            p1_sum32_flat_next[i*32 +: 32] = sum32;

            if (s0b_is_final_eff) begin
                // V-mode: 1 bias per lane (channel). Q-mode: 1 bias per column.
                if (s0b_mode_v) begin
                    if (i < 16) bias16_lane = $signed(bias_word_lo[i*16 +: 16]);
                    else        bias16_lane = $signed(bias_word_hi[(i-16)*16 +: 16]);
                end else begin
                    bias16_lane = use_bias_q_b ? bias16_col : 16'sd0;
                end
                sum32b = sum32 + ($signed(bias16_lane) <<< BIAS_LSHIFT);
            end else begin
                sum32b = sum32;
            end
            p1_sum32b_flat_next[i*32 +: 32] = sum32b;
        end

        // -----------------------------------------
        // P2 next (from p2a regs): registered mul48
        // -----------------------------------------
        if (p2a_valid && p2a_is_final_eff) begin
            for (i = 0; i < 32; i = i + 1) begin
                mul48 = $signed(p2a_sum32b_flat[i*32 +: 32]) * $signed(p2a_rq_mul);
                p2_mul48_flat_next[i*48 +: 48] = mul48;
            end
            // CLS special-case only affects lane0
            if (CLS_ALT_EN && p2a_is_cls_block) begin
                mul48 = $signed(p2a_sum32_flat[0*32 +: 32]) * $signed(p2a_q_cls_mul);
                p2_mul48_flat_next[0*48 +: 48] = mul48;
            end
        end

// -----------------------------------------
        // P3 pack (from p2 regs): shift/add/sat8
        // -----------------------------------------
        if (p2_valid) begin
            for (i = 0; i < 32; i = i + 1) begin
                if (p2_is_final_eff) begin
                    mul48 = $signed(p2_mul48_flat[i*48 +: 48]);
                    if (CLS_ALT_EN && p2_is_cls_block && (i == 0)) begin
                        qtmp32 = $signed(($signed(mul48) + $signed(p2_cls_off_lane[i])) >>> p2_q_cls_sh_lane[i]);
                    end else begin
                        qtmp32 = $signed(($signed(mul48) + $signed(p2_rq_off_lane[i])) >>> p2_rq_sh_lane[i]);
                    end
                end else begin
                    sum32  = $signed(p2_sum32_flat[i*32 +: 32]);
                    qtmp32 = $signed(sum32 >>> p2_shift_store_eff);
                end
                packed_q8[i*8 +: 8] = sat8(qtmp32);
            end

            // AV padding mask: rows beyond the active token count are padded and must be written as 0.
            // OUT_STRIDE is the embedding/global-buffer column stride; active_token_total is runtime-sized.
            if (p2_layer_id == L_AV) begin
                row_base = (p2_mb << 5);
                if (row_base >= active_token_total) begin
                    packed_q8 = 0;
                end else if ((row_base + 31) >= active_token_total) begin
                    valid_lanes = active_token_total - row_base; // 0..32
                    for (j = 0; j < 32; j = j + 1) begin
                        if (j >= valid_lanes) begin
                            packed_q8[j*8 +: 8] = 8'd0;
                        end
                    end
                end
            end

            // CLS output mask for golden-compare
            if ((p2_layer_id >= L_CLS0) && (p2_layer_id <= L_CLS5)) begin
                for (j = 1; j < 32; j = j + 1) begin
                    packed_q8[j*8 +: 8] = 8'd0;
                end
                if (p2_layer_id == L_CLS5) begin
                    if ({p2_nb, p2_col[4:0]} >= 9'd40) begin
                        packed_q8[7:0] = 8'd0;
                    end
                end
            end
        end
    end

    // P1 regs: capture int32 accum + bias + meta + residual payload
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            p1_valid <= 0;
            p1_addr  <= 0;
            p1_mb    <= 0;
            p1_nb    <= 0;
            p1_col   <= 0;
            p1_kt    <= 0;
            p1_layer_id <= 0;
            p1_is_final_ktile <= 0;
            p1_is_final_eff   <= 0;
            p1_is_cls_block   <= 0;
            p1_shift_store_eff <= 0;

            p1_rq_mul <= 0;
            p1_rq_sh  <= 0;
            p1_rq_off_s48 <= 0;
            p1_q_cls_mul <= 0;
            p1_q_cls_sh  <= 0;
            p1_cls_off_s48 <= 0;

        end else begin
            p1_valid <= s0b_valid;
            if (s0b_valid) begin
                p1_addr  <= s0b_addr;
                p1_mb    <= s0b_mb;
                p1_nb    <= s0b_nb;
                p1_col   <= s0b_col;
                p1_kt    <= s0b_kt;
                p1_layer_id <= s0b_layer_id;
                p1_is_final_ktile <= s0b_is_final_ktile;
                p1_is_final_eff   <= s0b_is_final_eff;
                p1_is_cls_block   <= s0b_is_cls_block;
                p1_shift_store_eff <= p1_shift_store_eff_next;
                // latch quant constants with the same beat as p1_* regs
                p1_rq_mul      <= rq_mul;
                p1_rq_sh       <= rq_sh;
                p1_rq_off_s48  <= rq_off_s48;
                p1_q_cls_mul   <= q_cls_mul;
                p1_q_cls_sh    <= q_cls_sh;
                p1_cls_off_s48 <= cls_off_s48;
                p1_score_res_data <= s0b_score_rd_data;
                p1_gin_res_data   <= s0b_gin_rd_data;
                p1_q_res_align_mul_eff <= q_res_align_mul_eff;
                p1_q_res_align_divsh   <= q_res_align_divsh;
                p1_q_res_post_mul_eff  <= q_res_post_mul_eff;
                p1_q_res_post_divsh    <= q_res_post_divsh;
                p1_q_res_post_off_eff_s48 <= q_res_post_off_eff_s48;
            end
        end
    end

    // Wide P1 datapath registers deliberately have no reset. Keeping them in
    // the asynchronous-reset process above can infer FDCPE cells when the
    // pipeline boundary is preserved, which prevents accurate timing analysis.
    // p1_valid qualifies every use, so their reset value is irrelevant.
    always @(posedge clk) begin
        if (s0b_valid) begin
            p1_sum32_flat  <= p1_sum32_flat_next;
            p1_sum32b_flat <= p1_sum32b_flat_next;
        end
    end

    
    // P2A regs: capture mul operands/meta before DSP multiply (extra 1-cycle pipe for 300MHz)
    // - Only reset valids/meta; large datapath regs are not reset (safe because p2a_valid=0 during reset)
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            p2a_valid <= 0;
        end else begin
            p2a_valid <= p1_valid;
            if (p1_valid) begin
                p2a_addr  <= p1_addr;
                p2a_mb    <= p1_mb;
                p2a_nb    <= p1_nb;
                p2a_col   <= p1_col;
                p2a_kt    <= p1_kt;
                p2a_layer_id <= p1_layer_id;
                p2a_is_final_ktile <= p1_is_final_ktile;
                p2a_is_final_eff   <= p1_is_final_eff;
                p2a_is_cls_block   <= p1_is_cls_block;
                p2a_shift_store_eff <= p1_shift_store_eff;

                // forward quant constants (aligned)
                p2a_rq_mul      <= p1_rq_mul;
                p2a_rq_sh       <= p1_rq_sh;
                p2a_rq_off_s48  <= p1_rq_off_s48;
                p2a_q_cls_mul   <= p1_q_cls_mul;
                p2a_q_cls_sh    <= p1_q_cls_sh;
                p2a_cls_off_s48 <= p1_cls_off_s48;

                // forward residual payload for core pipeline alignment
                p2a_score_res_data <= p1_score_res_data;
                p2a_gin_res_data   <= p1_gin_res_data;
                p2a_q_res_align_mul_eff <= p1_q_res_align_mul_eff;
                p2a_q_res_align_divsh   <= p1_q_res_align_divsh;
                p2a_q_res_post_mul_eff  <= p1_q_res_post_mul_eff;
                p2a_q_res_post_divsh    <= p1_q_res_post_divsh;
                p2a_q_res_post_off_eff_s48 <= p1_q_res_post_off_eff_s48;
            end
        end
    end

    // Preserve the second wide datapath boundary as reset-free FDRE storage.
    // p2a_valid provides the corresponding validity/reset qualification.
    always @(posedge clk) begin
        if (p1_valid) begin
            p2a_sum32_flat  <= p1_sum32_flat;
            p2a_sum32b_flat <= p1_sum32b_flat;
            for (p2ag_i = 0; p2ag_i < WB_GROUPS; p2ag_i = p2ag_i + 1) begin
                p2a_rq_off_group[p2ag_i]  <= p1_rq_off_s48;
                p2a_rq_sh_group[p2ag_i]   <= p1_rq_sh;
                p2a_cls_off_group[p2ag_i] <= p1_cls_off_s48;
                p2a_cls_sh_group[p2ag_i]  <= p1_q_cls_sh;
            end
        end
    end

// P2 regs: capture mul48 + delayed meta/payload
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            p2_valid <= 0;
            p2_addr  <= 0;
            p2_mb    <= 0;
            p2_nb    <= 0;
            p2_col   <= 0;
            p2_kt    <= 0;
            p2_layer_id <= 0;
            p2_is_final_ktile <= 0;
            p2_is_final_eff   <= 0;
            p2_is_cls_block   <= 0;
            p2_shift_store_eff <= 0;

            p2_rq_mul <= 0;
            p2_rq_sh  <= 0;
            p2_rq_off_s48 <= 0;
            p2_q_cls_mul <= 0;
            p2_q_cls_sh  <= 0;
            p2_cls_off_s48 <= 0;
            for (p2r_i = 0; p2r_i < 32; p2r_i = p2r_i + 1) begin
                p2_rq_off_lane[p2r_i]  <= 0;
                p2_rq_sh_lane[p2r_i]   <= 0;
                p2_cls_off_lane[p2r_i] <= 0;
                p2_q_cls_sh_lane[p2r_i] <= 0;
            end

        end else begin
            p2_valid <= p2a_valid;
            if (p2a_valid) begin
                p2_addr  <= p2a_addr;
                p2_mb    <= p2a_mb;
                p2_nb    <= p2a_nb;
                p2_col   <= p2a_col;
                p2_kt    <= p2a_kt;
                p2_layer_id <= p2a_layer_id;
                p2_is_final_ktile <= p2a_is_final_ktile;
                p2_is_final_eff   <= p2a_is_final_eff;
                p2_is_cls_block   <= p2a_is_cls_block;
                p2_shift_store_eff <= p2a_shift_store_eff;
                p2_sum32_flat  <= p2a_sum32_flat;
                p2_sum32b_flat <= p2a_sum32b_flat;
                p2_mul48_flat  <= p2_mul48_flat_next;

                // forward quant constants into p2_* stage for P3 alignment
                p2_rq_mul      <= p2a_rq_mul;
                p2_rq_sh       <= p2a_rq_sh;
                p2_rq_off_s48  <= p2a_rq_off_s48;
                p2_q_cls_mul   <= p2a_q_cls_mul;
                p2_q_cls_sh    <= p2a_q_cls_sh;
                p2_cls_off_s48 <= p2a_cls_off_s48;
                for (p2r_i = 0; p2r_i < 32; p2r_i = p2r_i + 1) begin
                    p2_rq_off_lane[p2r_i]   <= p2a_rq_off_group[p2r_i / WB_GROUP_SIZE];
                    p2_rq_sh_lane[p2r_i]    <= p2a_rq_sh_group[p2r_i / WB_GROUP_SIZE];
                    p2_cls_off_lane[p2r_i]  <= p2a_cls_off_group[p2r_i / WB_GROUP_SIZE];
                    p2_q_cls_sh_lane[p2r_i] <= p2a_cls_sh_group[p2r_i / WB_GROUP_SIZE];
                end

                p2_score_res_data <= p2a_score_res_data;
                p2_gin_res_data   <= p2a_gin_res_data;
                p2_q_res_align_mul_eff <= p2a_q_res_align_mul_eff;
                p2_q_res_align_divsh   <= p2a_q_res_align_divsh;
                p2_q_res_post_mul_eff  <= p2a_q_res_post_mul_eff;
                p2_q_res_post_divsh    <= p2a_q_res_post_divsh;
                p2_q_res_post_off_eff_s48 <= p2a_q_res_post_off_eff_s48;
            end
        end
    end

    // ------------------------------------------------------------
    // StageQ8 regs: capture packed_q8 + meta (cuts BRAM->DSP/LUT->BRAM critical paths)
    // ------------------------------------------------------------
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            s1_valid_q8          <= 0;
            s1_mb_q8             <= 0;
            s1_nb_q8             <= 0;
            s1_col_q8            <= 0;
            s1_kt_q8             <= 0;
            s1_mode_score_q8     <= 0;
            s1_mode_fc1_q8       <= 0;
            s1_is_final_ktile_q8 <= 0;

            s1_addr_q8           <= 0;
            s1_layer_id_q8       <= 0;
            s1_is_final_eff_q8   <= 0;
            s1_is_cls_block_q8   <= 0;
            s1_shift_store_eff_q8 <= 0;
        end else begin
            s1_valid_q8 <= p2_valid;
            if (p2_valid) begin
                s1_packed_q8         <= packed_q8;
                s1_mb_q8             <= p2_mb;
                s1_nb_q8             <= p2_nb;
                s1_col_q8            <= p2_col;
                s1_kt_q8             <= p2_kt;
                s1_mode_score_q8     <= (p2_layer_id == L_SCORE);
                s1_mode_fc1_q8       <= (p2_layer_id == L_MLP_FC1);
                s1_is_final_ktile_q8 <= p2_is_final_ktile;

                s1_addr_q8           <= p2_addr;
                s1_layer_id_q8       <= p2_layer_id;
                s1_is_final_eff_q8   <= p2_is_final_eff;
                s1_is_cls_block_q8   <= p2_is_cls_block;
                s1_shift_store_eff_q8 <= p2_shift_store_eff;
                s1_score_res_data_q8 <= p2_score_res_data;
                s1_gin_res_data_q8   <= p2_gin_res_data;
                s1_q_res_align_mul_eff_q8 <= p2_q_res_align_mul_eff;
                s1_q_res_align_divsh_q8   <= p2_q_res_align_divsh;
                s1_q_res_post_mul_eff_q8  <= p2_q_res_post_mul_eff;
                s1_q_res_post_divsh_q8    <= p2_q_res_post_divsh;
                s1_q_res_post_off_eff_s48_q8 <= p2_q_res_post_off_eff_s48;
            end
        end
    end

    // Datapath-only regional copies.  They are qualified by s1_valid_q8 in
    // the following stage, so reset values are not functionally observable.
    always @(posedge clk) begin
        if (p2_valid) begin
            for (res_align_group_i = 0;
                 res_align_group_i < RES_ALIGN_GROUPS;
                 res_align_group_i = res_align_group_i + 1)
                s1_q_res_align_mul_group_q8[res_align_group_i] <= p2_q_res_align_mul_eff;
        end
    end

    wire         core_gout_wr_valid;
    wire [GOUT_AW-1:0]  core_gout_wr_addr;
    wire [255:0] core_gout_wr_data;

    assign psum_wr_valid = p2_valid && ((p2_layer_id == L_MLP_FC2) || (p2_layer_id == L_PATCH)) && !p2_is_final_ktile;
    assign psum_wr_addr  = p2_addr;
    assign psum_wr_data  = p2_sum32_flat;

    // ------------------------------------------------------------
    // Core GOUT writeback pipeline (2-cycle) for timing closure
    //  - Stage1: register base quant output (packed_q8) + residual data + constants
    //  - Stage2: residual-align + add (int32) for OPROJ/FC2 final
    //  - Stage3 (comb): post-requant + sat8, or pass-through base_q8
    // ------------------------------------------------------------
    wire s1_mode_oproj_q8 = (s1_layer_id_q8 == L_O_PROJ);
    wire s1_mode_fc2_q8   = (s1_layer_id_q8 == L_MLP_FC2);
    wire s1_mode_patch_q8 = (s1_layer_id_q8 == L_PATCH);
    wire core_en_s0;
    assign core_en_s0 = s1_valid_q8 && !s1_mode_score_q8 && !s1_mode_fc1_q8 &&
                        !((s1_mode_fc2_q8 || s1_mode_patch_q8) && !s1_is_final_ktile_q8);
    wire do_res_oproj_s0;
    wire do_res_fc2_s0;
    wire do_res_patch_s0;
    wire do_res_any_s0;
    assign do_res_oproj_s0 = core_en_s0 && s1_mode_oproj_q8 && s1_is_final_ktile_q8;
    assign do_res_fc2_s0   = core_en_s0 && s1_mode_fc2_q8   && s1_is_final_ktile_q8;
    assign do_res_patch_s0 = core_en_s0 && s1_mode_patch_q8 && s1_is_final_ktile_q8;
    assign do_res_any_s0   = do_res_oproj_s0 || do_res_fc2_s0 || do_res_patch_s0;

    // Metadata aligned with core_gout_wr_* after the registered writeback pipe.
    reg [10:0] core_final_ktile_pipe;
    reg [3:0]  core_layer_id_pipe [0:10];
    integer core_meta_i;

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            core_final_ktile_pipe <= 11'd0;
            for (core_meta_i = 0; core_meta_i < 11; core_meta_i = core_meta_i + 1)
                core_layer_id_pipe[core_meta_i] <= 4'd0;
        end else begin
            core_final_ktile_pipe[0] <= core_en_s0 && s1_is_final_ktile_q8;
            core_layer_id_pipe[0]    <= s1_layer_id_q8;
            for (core_meta_i = 1; core_meta_i < 11; core_meta_i = core_meta_i + 1) begin
                core_final_ktile_pipe[core_meta_i] <= core_final_ktile_pipe[core_meta_i-1];
                core_layer_id_pipe[core_meta_i]    <= core_layer_id_pipe[core_meta_i-1];
            end
        end
    end

    // stage1 regs
    reg [GOUT_AW-1:0]  s1_addr;
    reg [255:0] s1_base_q8;
    reg         s1_do_res;
    reg         s1_wr_state_bank;
    reg [255:0] s1_res_q8_sel;
    reg signed [15:0] s1_q_res_align_mul_eff;
    reg [3:0]         s1_q_res_align_divsh;
    reg signed [15:0] s1_q_res_post_mul_eff;
    reg [3:0]         s1_q_res_post_divsh;
    reg signed [47:0] s1_q_res_post_off_eff_s48;
    (* keep = "true", dont_touch = "true" *)
    reg signed [15:0] s1_q_res_align_mul_lane [0:31];

    // stage2 regs
    reg [GOUT_AW-1:0]  s2_addr;
    reg [255:0] s2_base_q8;
    reg         s2_do_res;
    reg         s2_wr_state_bank;
    reg [32*32-1:0] s2_sum32_flat;
    reg signed [15:0] s2_q_res_post_mul_eff;
    reg [3:0]         s2_q_res_post_divsh;
    reg signed [47:0] s2_q_res_post_off_eff_s48;

    // stage2m regs: extra bundle stage before DSP multiply (cuts DSP->s3m logic-heavy path without touching commit banking)
    reg         s2m_valid_core;
    reg [GOUT_AW-1:0]  s2m_addr;
    reg [255:0] s2m_base_q8;
    reg         s2m_do_res;
    reg         s2m_wr_state_bank;
    reg [32*32-1:0] s2m_sum32_flat;
    // Keep one local multiplier copy per lane. A single shared copy placed
    // far from the 32 DSPs becomes a route-dominated 200-MHz path.
    (* keep = "yes", dont_touch = "yes" *)
    reg signed [15:0] s2m_q_res_post_mul_eff_lane [0:31];
    reg [3:0]         s2m_q_res_post_divsh;
    reg signed [47:0] s2m_q_res_post_off_eff_s48;


    // stage1m regs: residual-align multiply/shift result
    reg         s1m_valid_core;
    reg [GOUT_AW-1:0]  s1m_addr;
    reg [255:0] s1m_base_q8;
    reg         s1m_do_res;
    reg         s1m_wr_state_bank;
    reg signed [63:0] s1m_mul64_lane [0:31];
    reg signed [15:0] s1m_q_res_post_mul_eff;
    reg [3:0]         s1m_q_res_post_divsh;
    reg signed [47:0] s1m_q_res_post_off_eff_s48;

    // stage1a regs: rounded/aligned residual in i32 domain
    reg         s1a_valid_core;
    reg [GOUT_AW-1:0]  s1a_addr;
    reg [255:0] s1a_base_q8;
    reg         s1a_do_res;
    reg         s1a_wr_state_bank;
    reg [32*32-1:0] s1a_res_align32_flat;
    reg signed [15:0] s1a_q_res_post_mul_eff;
    reg [3:0]         s1a_q_res_post_divsh;
    reg signed [47:0] s1a_q_res_post_off_eff_s48;

    // stage1m comb (residual-align multiply + left-shift only)
    reg signed [63:0] mul64_shift_lane_next [0:31];
    reg signed [63:0] mul64_p1m;
    reg signed [7:0]  res_i8_p1m;
    integer k1m;
    always @(*) begin
        mul64_p1m  = 0;
        res_i8_p1m = 0;
        for (k1m = 0; k1m < 32; k1m = k1m + 1) begin
            res_i8_p1m = $signed(s1_res_q8_sel[k1m*8 +: 8]);
            mul64_p1m  = $signed({{56{res_i8_p1m[7]}}, res_i8_p1m}) * $signed(s1_q_res_align_mul_lane[k1m]);
            mul64_p1m  = $signed(mul64_p1m <<< s1_q_res_align_divsh);
            mul64_shift_lane_next[k1m] = mul64_p1m;
        end
    end

    // stage1a comb (banker round only)
    reg [32*32-1:0] res_align32_flat_next;
    reg signed [63:0] mul64_sel_p1a;
    reg signed [31:0] res_align32_p1a;
    integer k1a;
    always @(*) begin
        res_align32_flat_next = 0;
        mul64_sel_p1a         = 0;
        res_align32_p1a       = 0;
        for (k1a = 0; k1a < 32; k1a = k1a + 1) begin
            mul64_sel_p1a   = s1m_mul64_lane[k1a];
            res_align32_p1a = round_shift_even_s64(mul64_sel_p1a, 6'd15);
            res_align32_flat_next[k1a*32 +: 32] = res_align32_p1a;
        end
    end

    // stage2 comb (base + aligned residual add only)
    reg [32*32-1:0] sum32_flat_next;
    reg signed [31:0] sum32_p1;
    reg signed [7:0]  base_i8_p1;
    reg signed [31:0] res_align32_i32_p1;
    integer k1;
    always @(*) begin
        sum32_flat_next    = 0;
        sum32_p1           = 0;
        base_i8_p1         = 0;
        res_align32_i32_p1 = 0;

        for (k1 = 0; k1 < 32; k1 = k1 + 1) begin
            base_i8_p1         = $signed(s1a_base_q8[k1*8 +: 8]);
            res_align32_i32_p1 = $signed(s1a_res_align32_flat[k1*32 +: 32]);
            sum32_p1 = $signed({{24{base_i8_p1[7]}}, base_i8_p1}) + $signed(res_align32_i32_p1);
            sum32_flat_next[k1*32 +: 32] = sum32_p1;
        end
    end

    // stage1/stage1m/stage1a/stage2 regs update
    // - only valid/control regs are explicitly reset
    // - wide datapath regs intentionally keep no async-reset for timing/fanout
    integer s1m_i;
    integer res_align_lane_i;

    // Second-level lane copies preserve the existing q8-to-stage1 boundary
    // while limiting each regional register to four lane registers.
    always @(posedge clk) begin
        if (core_en_s0) begin
            for (res_align_lane_i = 0;
                 res_align_lane_i < 32;
                 res_align_lane_i = res_align_lane_i + 1)
                s1_q_res_align_mul_lane[res_align_lane_i] <=
                    s1_q_res_align_mul_group_q8[res_align_lane_i / RES_ALIGN_GROUP_SIZE];
        end
    end

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            s1_valid_core <= 0;
            s1_do_res     <= 0;
            s1_wr_state_bank <= 0;

            s1m_valid_core <= 0;
            s1m_do_res     <= 0;
            s1m_wr_state_bank <= 0;
            s1a_valid_core <= 0;
            s1a_do_res     <= 0;
            s1a_wr_state_bank <= 0;

            s2_valid_core <= 0;
            s2_do_res     <= 0;
            s2_wr_state_bank <= 0;
            s2m_valid_core <= 0;
            s2m_do_res     <= 0;
            s2m_wr_state_bank <= 0;
        end else begin
            // stage1 capture
            s1_valid_core <= core_en_s0;
            if (core_en_s0) begin
                s1_addr      <= s1_addr_q8;
                s1_base_q8   <= s1_packed_q8;
                s1_do_res    <= do_res_any_s0;
                // AV is produced in GB1 and remains the OPROJ input until all
                // output-channel blocks finish. Store OPROJ add1 in the now-free
                // GB0 so an early output block cannot overwrite AV channels
                // needed by later output blocks. FC2 add2 remains in GB1.
                s1_wr_state_bank <= do_res_fc2_s0 ||
                                    (s1_layer_id_q8 == L_AV);
                s1_res_q8_sel <= (do_res_oproj_s0 || do_res_patch_s0) ? s1_score_res_data_q8 : s1_gin_res_data_q8;
                s1_q_res_align_mul_eff <= s1_q_res_align_mul_eff_q8;
                s1_q_res_align_divsh   <= s1_q_res_align_divsh_q8;
                s1_q_res_post_mul_eff  <= s1_q_res_post_mul_eff_q8;
                s1_q_res_post_divsh    <= s1_q_res_post_divsh_q8;
                s1_q_res_post_off_eff_s48 <= s1_q_res_post_off_eff_s48_q8;
            end else begin
                s1_do_res <= 0;
                s1_wr_state_bank <= 0;
            end

            // stage1m capture: DSP mul/shift only
            s1m_valid_core <= s1_valid_core;
            if (s1_valid_core) begin
                s1m_addr      <= s1_addr;
                s1m_base_q8   <= s1_base_q8;
                s1m_do_res    <= s1_do_res;
                s1m_wr_state_bank <= s1_wr_state_bank;
                for (s1m_i = 0; s1m_i < 32; s1m_i = s1m_i + 1)
                    s1m_mul64_lane[s1m_i] <= mul64_shift_lane_next[s1m_i];
                s1m_q_res_post_mul_eff <= s1_q_res_post_mul_eff;
                s1m_q_res_post_divsh   <= s1_q_res_post_divsh;
                s1m_q_res_post_off_eff_s48 <= s1_q_res_post_off_eff_s48;
            end else begin
                s1m_do_res <= 0;
                s1m_wr_state_bank <= 0;
            end

            // stage1a capture: banker round only
            s1a_valid_core <= s1m_valid_core;
            if (s1m_valid_core) begin
                s1a_addr      <= s1m_addr;
                s1a_base_q8   <= s1m_base_q8;
                s1a_do_res    <= s1m_do_res;
                s1a_wr_state_bank <= s1m_wr_state_bank;
                s1a_res_align32_flat <= res_align32_flat_next;
                s1a_q_res_post_mul_eff <= s1m_q_res_post_mul_eff;
                s1a_q_res_post_divsh   <= s1m_q_res_post_divsh;
                s1a_q_res_post_off_eff_s48 <= s1m_q_res_post_off_eff_s48;
            end else begin
                s1a_do_res <= 0;
                s1a_wr_state_bank <= 0;
            end

            // stage2 capture: base + aligned residual add only
            s2_valid_core <= s1a_valid_core;
            if (s1a_valid_core) begin
                s2_addr      <= s1a_addr;
                s2_base_q8   <= s1a_base_q8;
                s2_do_res    <= s1a_do_res;
                s2_wr_state_bank <= s1a_wr_state_bank;
                s2_sum32_flat <= sum32_flat_next;
                s2_q_res_post_mul_eff <= s1a_q_res_post_mul_eff;
                s2_q_res_post_divsh   <= s1a_q_res_post_divsh;
                s2_q_res_post_off_eff_s48 <= s1a_q_res_post_off_eff_s48;
            end else begin
                s2_do_res <= 0;
                s2_wr_state_bank <= 0;
            end

            // stage2m capture: bundle-align inputs right before stage3m multiply
            s2m_valid_core <= s2_valid_core;
            if (s2_valid_core) begin
                s2m_addr      <= s2_addr;
                s2m_base_q8   <= s2_base_q8;
                s2m_do_res    <= s2_do_res;
                s2m_wr_state_bank <= s2_wr_state_bank;
                s2m_sum32_flat <= s2_sum32_flat;
                s2m_q_res_post_divsh   <= s2_q_res_post_divsh;
                s2m_q_res_post_off_eff_s48 <= s2_q_res_post_off_eff_s48;
            end else begin
                s2m_do_res <= 0;
                s2m_wr_state_bank <= 0;
            end
        end
    end

    // Datapath-only copies deliberately have no asynchronous reset. Keeping
    // them out of the reset-sensitive pipeline block prevents FDCPE inference
    // while preserving one physically local source register per DSP lane.
    integer s2m_lane_i;
    always @(posedge clk) begin
        if (s2_valid_core) begin
            for (s2m_lane_i = 0; s2m_lane_i < 32; s2m_lane_i = s2m_lane_i + 1)
                s2m_q_res_post_mul_eff_lane[s2m_lane_i] <= s2_q_res_post_mul_eff;
        end
    end

    // stage3m comb (post-mul only)
    // - always computes multiplier products for timing
    // - s2_do_res is only consumed as downstream use/selection control
reg signed [47:0] prod48_p3;
reg signed [47:0] prod48_lane_next [0:31];
integer k3;
always @(*) begin
    prod48_p3 = 0;
    for (k3 = 0; k3 < 32; k3 = k3 + 1) begin
        prod48_p3 = $signed(s2m_sum32_flat[k3*32 +: 32]) *
                    $signed(s2m_q_res_post_mul_eff_lane[k3]);
        prod48_lane_next[k3] = prod48_p3;
    end
end

// stage3m regs (mul results + params)
// - only valid/control regs are explicitly reset
reg         s3m_valid_core;
reg [GOUT_AW-1:0]  s3m_addr;
reg [255:0] s3m_base_q8;
reg         s3m_do_res;
reg         s3m_wr_state_bank;
reg signed [47:0] s3m_prod48_lane [0:31];
reg [3:0]         s3m_q_res_post_divsh;
reg signed [47:0] s3m_q_res_post_off_eff_s48;
integer s3m_i;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3m_valid_core <= 0;
        s3m_do_res     <= 0;
        s3m_wr_state_bank <= 0;
        for (s3m_i = 0; s3m_i < 32; s3m_i = s3m_i + 1)
            s3m_prod48_lane[s3m_i] <= 0;
    end else begin
        s3m_valid_core <= s2m_valid_core;
        if (s2m_valid_core) begin
            s3m_addr       <= s2m_addr;
            s3m_base_q8    <= s2m_base_q8;
            s3m_do_res     <= s2m_do_res;
            s3m_wr_state_bank <= s2m_wr_state_bank;
            for (s3m_i = 0; s3m_i < 32; s3m_i = s3m_i + 1)
                s3m_prod48_lane[s3m_i] <= prod48_lane_next[s3m_i];
            s3m_q_res_post_divsh <= s2m_q_res_post_divsh;
            s3m_q_res_post_off_eff_s48 <= s2m_q_res_post_off_eff_s48;
        end else begin
            s3m_do_res <= 0;
            s3m_wr_state_bank <= 0;
        end
    end
end


function signed [63:0] sext48_to_s64;
    input signed [47:0] x;
    begin
        sext48_to_s64 = {{16{x[47]}}, x};
    end
endfunction

function signed [63:0] sll_s64_by4x;
    input signed [63:0] x;
    input [1:0] sh_hi;
    begin
        case (sh_hi)
            2'd0:    sll_s64_by4x = x;
            2'd1:    sll_s64_by4x = x <<< 4;
            2'd2:    sll_s64_by4x = x <<< 8;
            2'd3:    sll_s64_by4x = x <<< 12;
            default: sll_s64_by4x = x;
        endcase
    end
endfunction

function signed [63:0] sll_s64_by1x;
    input signed [63:0] x;
    input [1:0] sh_lo;
    begin
        case (sh_lo)
            2'd0:    sll_s64_by1x = x;
            2'd1:    sll_s64_by1x = x <<< 1;
            2'd2:    sll_s64_by1x = x <<< 2;
            2'd3:    sll_s64_by1x = x <<< 3;
            default: sll_s64_by1x = x;
        endcase
    end
endfunction

// stage3a comb (sign-extend only)
reg signed [63:0] x64_lane_next [0:31];
reg signed [47:0] prod48_sel_p3a;
integer k4;
always @(*) begin
    prod48_sel_p3a = 0;
    for (k4 = 0; k4 < 32; k4 = k4 + 1)
        x64_lane_next[k4] = 0;

    if (s3m_valid_core && s3m_do_res) begin
        for (k4 = 0; k4 < 32; k4 = k4 + 1) begin
            prod48_sel_p3a = $signed(s3m_prod48_lane[k4]);
            x64_lane_next[k4] = sext48_to_s64(prod48_sel_p3a);
        end
    end
end

// stage3a regs (sign-extended 64b lanes + shift/off meta)
reg               s3a_valid_core;
reg [GOUT_AW-1:0]        s3a_addr;
reg [255:0]       s3a_base_q8;
reg               s3a_do_res;
reg               s3a_wr_state_bank;
reg signed [63:0] s3a_x64_lane [0:31];
reg [3:0]         s3a_q_res_post_divsh;
reg signed [47:0] s3a_q_res_post_off_eff_s48;
integer s3a_i;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3a_valid_core <= 0;
        s3a_do_res     <= 0;
        s3a_wr_state_bank <= 0;
        s3a_addr       <= 0;
        s3a_base_q8    <= 0;
        s3a_q_res_post_divsh <= 0;
        s3a_q_res_post_off_eff_s48 <= 0;
        for (s3a_i = 0; s3a_i < 32; s3a_i = s3a_i + 1)
            s3a_x64_lane[s3a_i] <= 0;
    end else begin
        s3a_valid_core <= s3m_valid_core;
        if (s3m_valid_core) begin
            s3a_addr     <= s3m_addr;
            s3a_base_q8  <= s3m_base_q8;
            s3a_do_res   <= s3m_do_res;
            s3a_wr_state_bank <= s3m_wr_state_bank;
            for (s3a_i = 0; s3a_i < 32; s3a_i = s3a_i + 1)
                s3a_x64_lane[s3a_i] <= x64_lane_next[s3a_i];
            s3a_q_res_post_divsh <= s3m_q_res_post_divsh;
            s3a_q_res_post_off_eff_s48 <= s3m_q_res_post_off_eff_s48;
        end else begin
            s3a_do_res <= 0;
            s3a_wr_state_bank <= 0;
        end
    end
end

// stage3b comb (coarse shift: 0/4/8/12)
reg signed [63:0] coarse64_lane_next [0:31];
reg signed [63:0] x64_sel_p3b;
integer k5;
always @(*) begin
    x64_sel_p3b = 0;
    for (k5 = 0; k5 < 32; k5 = k5 + 1)
        coarse64_lane_next[k5] = 0;

    if (s3a_valid_core && s3a_do_res) begin
        for (k5 = 0; k5 < 32; k5 = k5 + 1) begin
            x64_sel_p3b = $signed(s3a_x64_lane[k5]);
            coarse64_lane_next[k5] = sll_s64_by4x(x64_sel_p3b, s3a_q_res_post_divsh[3:2]);
        end
    end
end

// stage3b regs (coarse-shifted lanes + low shift bits + off/meta)
reg               s3b_valid_core;
reg [GOUT_AW-1:0]        s3b_addr;
reg [255:0]       s3b_base_q8;
reg               s3b_do_res;
reg               s3b_wr_state_bank;
reg signed [63:0] s3b_sh64_lane [0:31];
reg [1:0]         s3b_q_res_post_divsh_lo;
reg signed [47:0] s3b_q_res_post_off_eff_s48;
integer s3b_i;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3b_valid_core <= 0;
        s3b_do_res     <= 0;
        s3b_wr_state_bank <= 0;
        s3b_addr       <= 0;
        s3b_base_q8    <= 0;
        s3b_q_res_post_divsh_lo <= 0;
        s3b_q_res_post_off_eff_s48 <= 0;
        for (s3b_i = 0; s3b_i < 32; s3b_i = s3b_i + 1)
            s3b_sh64_lane[s3b_i] <= 0;
    end else begin
        s3b_valid_core <= s3a_valid_core;
        if (s3a_valid_core) begin
            s3b_addr     <= s3a_addr;
            s3b_base_q8  <= s3a_base_q8;
            s3b_do_res   <= s3a_do_res;
            s3b_wr_state_bank <= s3a_wr_state_bank;
            for (s3b_i = 0; s3b_i < 32; s3b_i = s3b_i + 1)
                s3b_sh64_lane[s3b_i] <= coarse64_lane_next[s3b_i];
            s3b_q_res_post_divsh_lo <= s3a_q_res_post_divsh[1:0];
            s3b_q_res_post_off_eff_s48 <= s3a_q_res_post_off_eff_s48;
        end else begin
            s3b_do_res <= 0;
            s3b_wr_state_bank <= 0;
        end
    end
end

// stage3c comb (fine shift: 0/1/2/3)
reg signed [63:0] fine64_lane_next [0:31];
reg signed [63:0] sh64_sel_p3c;
integer k6;
always @(*) begin
    sh64_sel_p3c = 0;
    for (k6 = 0; k6 < 32; k6 = k6 + 1)
        fine64_lane_next[k6] = 0;

    if (s3b_valid_core && s3b_do_res) begin
        for (k6 = 0; k6 < 32; k6 = k6 + 1) begin
            sh64_sel_p3c = $signed(s3b_sh64_lane[k6]);
            fine64_lane_next[k6] = sll_s64_by1x(sh64_sel_p3c, s3b_q_res_post_divsh_lo);
        end
    end
end

// stage3c regs (fully shifted 64b lanes + off/meta)
reg               s3c_valid_core;
reg [GOUT_AW-1:0]        s3c_addr;
reg [255:0]       s3c_base_q8;
reg               s3c_do_res;
reg               s3c_wr_state_bank;
reg signed [63:0] s3c_sh64_lane [0:31];
reg signed [47:0] s3c_q_res_post_off_eff_s48;
integer s3c_i;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3c_valid_core <= 0;
        s3c_do_res     <= 0;
        s3c_wr_state_bank <= 0;
        s3c_addr       <= 0;
        s3c_base_q8    <= 0;
        s3c_q_res_post_off_eff_s48 <= 0;
        for (s3c_i = 0; s3c_i < 32; s3c_i = s3c_i + 1)
            s3c_sh64_lane[s3c_i] <= 0;
    end else begin
        s3c_valid_core <= s3b_valid_core;
        if (s3b_valid_core) begin
            s3c_addr     <= s3b_addr;
            s3c_base_q8  <= s3b_base_q8;
            s3c_do_res   <= s3b_do_res;
            s3c_wr_state_bank <= s3b_wr_state_bank;
            for (s3c_i = 0; s3c_i < 32; s3c_i = s3c_i + 1)
                s3c_sh64_lane[s3c_i] <= fine64_lane_next[s3c_i];
            s3c_q_res_post_off_eff_s48 <= s3b_q_res_post_off_eff_s48;
        end else begin
            s3c_do_res <= 0;
            s3c_wr_state_bank <= 0;
        end
    end
end

// stage3d comb (offset add + >>>15)
reg signed [31:0] q32_lane_next [0:31];
reg signed [63:0] tmp64_p3d;
reg signed [31:0] qtmp32_p3d;
integer k7;
always @(*) begin
    tmp64_p3d  = 0;
    qtmp32_p3d = 0;
    for (k7 = 0; k7 < 32; k7 = k7 + 1)
        q32_lane_next[k7] = 0;

    if (s3c_valid_core && s3c_do_res) begin
        for (k7 = 0; k7 < 32; k7 = k7 + 1) begin
            tmp64_p3d = $signed(s3c_sh64_lane[k7]) + $signed({{16{s3c_q_res_post_off_eff_s48[47]}}, s3c_q_res_post_off_eff_s48});
            qtmp32_p3d = $signed(tmp64_p3d >>> 15);
            q32_lane_next[k7] = qtmp32_p3d;
        end
    end
end

// stage3d regs (intermediate q32 lanes + meta)
reg               s3d_valid_core;
reg [GOUT_AW-1:0]        s3d_addr;
reg [255:0]       s3d_base_q8;
reg               s3d_do_res;
reg               s3d_wr_state_bank;
reg signed [31:0] s3d_q32_lane [0:31];
integer s3d_i;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3d_valid_core <= 0;
        s3d_do_res     <= 0;
        s3d_wr_state_bank <= 0;
        s3d_addr       <= 0;
        s3d_base_q8    <= 0;
        for (s3d_i = 0; s3d_i < 32; s3d_i = s3d_i + 1)
            s3d_q32_lane[s3d_i] <= 0;
    end else begin
        s3d_valid_core <= s3c_valid_core;
        if (s3c_valid_core) begin
            s3d_addr     <= s3c_addr;
            s3d_base_q8  <= s3c_base_q8;
            s3d_do_res   <= s3c_do_res;
            s3d_wr_state_bank <= s3c_wr_state_bank;
            for (s3d_i = 0; s3d_i < 32; s3d_i = s3d_i + 1)
                s3d_q32_lane[s3d_i] <= q32_lane_next[s3d_i];
        end else begin
            s3d_do_res <= 0;
            s3d_wr_state_bank <= 0;
        end
    end
end

// stage3e comb (sat8/pack only) - compute next value for final stage3 register
reg [255:0] packed_q8_core_next;
reg signed [31:0] q32_sel_p3e;
integer k8;
always @(*) begin
    packed_q8_core_next = s3d_base_q8;
    q32_sel_p3e         = 0;

    if (s3d_valid_core && s3d_do_res) begin
        for (k8 = 0; k8 < 32; k8 = k8 + 1) begin
            q32_sel_p3e = $signed(s3d_q32_lane[k8]);
            packed_q8_core_next[k8*8 +: 8] = sat8(q32_sel_p3e);
        end
    end
end

// stage3 regs (registered core GOUT write payload)
reg         s3_valid_core;
reg [GOUT_AW-1:0]  s3_addr;
reg [255:0] s3_packed_q8_core;
reg         s3_wr_state_bank;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        s3_valid_core      <= 0;
        s3_addr            <= 0;
        s3_packed_q8_core  <= 0;
        s3_wr_state_bank   <= 0;
    end else begin
        s3_valid_core <= s3d_valid_core;
        if (s3d_valid_core) begin
            s3_addr           <= s3d_addr;
            s3_packed_q8_core <= packed_q8_core_next;
            s3_wr_state_bank  <= s3d_wr_state_bank;
        end else begin
            s3_wr_state_bank <= 0;
        end
    end
end

assign core_gout_wr_valid = s3_valid_core;
assign core_gout_wr_addr  = s3_addr;
assign core_gout_wr_data  = s3_packed_q8_core;
assign gout_wr_final_ktile = core_final_ktile_pipe[10];
assign gout_wr_layer_id    = core_layer_id_pipe[10];
assign gout_wr_bank_ovr_en = core_gout_wr_valid && s3_wr_state_bank;
assign gout_wr_bank_ovr    = 1'b1;
// ============================================================
    // SCORE Softmax (row-wise) - 3 pass in writeback_controller
    // - Only active when layer_id_eff == L_SCORE (mode_score)
    // - Output format: uint8 where [3:0]=qlog (ilog/qlog4), overwrite in-place
    
    // ============================================================
    // SCORE Softmax (row-wise) with SCOREBUF ping-pong
    //  - raw score(int8) is written to SCOREBUF (not to GOUT)
    //  - PASS2 reads SCOREBUF -> computes inv per row
    //  - PASS3 reads SCOREBUF -> writes softmax(u8) to the attention buffer
    //  - PASS2+PASS3 run per-mblk as soon as that mblk is complete
    //
    // Assumption (current design): SCORE uses a single ktile (ktile_idx_eff==0),
    // so we don't need partial-sum accumulation in SCOREBUF.
    // ============================================================

    localparam signed [7:0] SM_MAX_INIT = -8'sd128;
    
    // per-bank (SCOREBUF ping-pong) softmax context:
    // - Because SCOREBUF is banked ping-pong, at most 2 row-blocks can be in-flight.
    // - We keep row-wise softmax state only for the two banks (instead of for all mblks).
    reg signed [7:0] sm_max_b0    [0:31];
    reg signed [7:0] sm_max_b1    [0:31];
    // LOD + next-bit log2(exp_sum) approximation per lane.
    reg [5:0]        sm_log0_q_b0 [0:31];
    reg [5:0]        sm_log0_q_b1 [0:31];
    reg              sm_log0_f_b0 [0:31];
    reg              sm_log0_f_b1 [0:31];

    reg [7:0]  fc1_gelu_lut_u8 [0:(SM_BLOCKS*256)-1];

    integer sm_init_i;
    initial begin
        for (sm_init_i = 0; sm_init_i < (SM_BLOCKS*256); sm_init_i = sm_init_i + 1) begin
            fc1_gelu_lut_u8[sm_init_i] = sm_init_i[7:0];
        end
        $readmemh(FC1_GELU_MEM, fc1_gelu_lut_u8);
    end
    
    // SCOREBUF write from core stream (raw score)
    wire        scorebuf_wr_valid_w;
    wire [11:0] s0_n_global;

    // FC1 output column-block -> SCOREBUF (bank0, addr = mblk*32 + col)
    wire [11:0] fc1_score_addr;
    wire        fc1_score_wr_valid_w;

    assign scorebuf_wr_valid_w   = s1_valid_q8 && s1_mode_score_q8;
    assign s0_n_global           = {s0_nb, 5'd0} + s0_col;

    assign fc1_score_wr_valid_w  = s1_valid_q8 && s1_mode_fc1_q8;
    assign fc1_score_addr        = {s1_mb_q8, 5'd0} + s1_col_q8;

    // FC1 activation after quantization.
    reg [255:0] packed_q8_fc1_act;
    integer gi;
    always @(*) begin
        packed_q8_fc1_act = s1_packed_q8;
        if (s1_mode_fc1_q8) begin
            for (gi = 0; gi < 32; gi = gi + 1) begin
                if (FC1_GELU_LUT_ENABLE != 0) begin
                    // Each 256-entry block slice is stored in signed-input order.
                    packed_q8_fc1_act[gi*8 +: 8] =
                        fc1_gelu_lut_u8[{enc_blk_idx, 8'd0} +
                            (s1_packed_q8[gi*8 +: 8] ^ 8'h80)];
                end else if (s1_packed_q8[gi*8+7]) begin
                    packed_q8_fc1_act[gi*8 +: 8] = 8'd0;
                end
            end
        end
    end

    wire        sm_score_wr_valid_w;
    wire        sm_score_wr_bank_w;
    wire [11:0] sm_score_wr_addr_w;
    wire [255:0] sm_score_wr_data_w;

    assign score_wr_valid = scorebuf_wr_valid_w || sm_score_wr_valid_w || fc1_score_wr_valid_w;
    assign score_wr_bank  = scorebuf_wr_valid_w ? scorebuf_wr_bank_r :
                            (sm_score_wr_valid_w ? sm_score_wr_bank_w : 1'b0);
    assign score_wr_addr  = scorebuf_wr_valid_w ? ({s1_nb_q8, 5'd0} + s1_col_q8) :
                            (sm_score_wr_valid_w ? sm_score_wr_addr_w : fc1_score_addr);
    assign score_wr_data  = scorebuf_wr_valid_w ? s1_packed_q8 :
                            (sm_score_wr_valid_w ? sm_score_wr_data_w : packed_q8_fc1_act);
// last valid word of one token row-block @ final ktile
    wire score_last_valid_word;
    assign score_last_valid_word =
        scorebuf_wr_valid_w && s1_is_final_ktile_q8 &&
        (s1_nb_q8 == M_BLK_LAST) && (s1_col_q8 == VALID_LANES_LAST);

// --------------------------------------------------------------------
// PASS1 softmax row-max (sm_max_b0/b1) next-state update
// - This is isolated from the main softmax FSM to avoid a shared, high-fanout
//   clock-enable path into sm_max_* regs (which was the WNS hot-spot).
// --------------------------------------------------------------------
wire score_entry_w;
assign score_entry_w = (!prev_mode_score && mode_score);

reg signed [7:0] sm_max_b0_next [0:31];
reg signed [7:0] sm_max_b1_next [0:31];

integer km;

always @(*) begin
    // default: hold
    for (km = 0; km < 32; km = km + 1) begin
        sm_max_b0_next[km] = sm_max_b0[km];
        sm_max_b1_next[km] = sm_max_b1[km];
    end

    // treat SCORE entry as "ctx invalid" for this cycle (safe even if first word overlaps)
    if (score_entry_w) begin
        for (km = 0; km < 32; km = km + 1) begin
            sm_max_b0_next[km] = SM_MAX_INIT;
            sm_max_b1_next[km] = SM_MAX_INIT;
        end
    end

    // PASS1 update on each accepted SCORE word
    if (scorebuf_wr_valid_w) begin
        if (scorebuf_wr_bank_r == 1'b0) begin
            if (!sm_ctx_vld[0] || score_entry_w) begin
                // init from first word
                for (km = 0; km < 32; km = km + 1) begin
                    if ((s1_mb_q8 != M_BLK_LAST) || (km <= VALID_LANES_LAST)) begin
                        sm_max_b0_next[km] = $signed(s1_packed_q8[km*8 +: 8]);
                    end else begin
                        sm_max_b0_next[km] = SM_MAX_INIT;
                    end
                end
            end else begin
                // update max
                for (km = 0; km < 32; km = km + 1) begin
                    if ((s1_mb_q8 != M_BLK_LAST) || (km <= VALID_LANES_LAST)) begin
                        if ($signed(s1_packed_q8[km*8 +: 8]) > sm_max_b0[km]) begin
                            sm_max_b0_next[km] = $signed(s1_packed_q8[km*8 +: 8]);
                        end
                    end
                end
            end
        end else begin
            if (!sm_ctx_vld[1] || score_entry_w) begin
                // init from first word
                for (km = 0; km < 32; km = km + 1) begin
                    if ((s1_mb_q8 != M_BLK_LAST) || (km <= VALID_LANES_LAST)) begin
                        sm_max_b1_next[km] = $signed(s1_packed_q8[km*8 +: 8]);
                    end else begin
                        sm_max_b1_next[km] = SM_MAX_INIT;
                    end
                end
            end else begin
                // update max
                for (km = 0; km < 32; km = km + 1) begin
                    if ((s1_mb_q8 != M_BLK_LAST) || (km <= VALID_LANES_LAST)) begin
                        if ($signed(s1_packed_q8[km*8 +: 8]) > sm_max_b1[km]) begin
                            sm_max_b1_next[km] = $signed(s1_packed_q8[km*8 +: 8]);
                        end
                    end
                end
            end
        end
    end
end

always @(posedge clk) begin
    if (!nRst) begin
        for (km = 0; km < 32; km = km + 1) begin
            sm_max_b0[km] <= SM_MAX_INIT;
            sm_max_b1[km] <= SM_MAX_INIT;
        end
    end else begin
        for (km = 0; km < 32; km = km + 1) begin
            sm_max_b0[km] <= sm_max_b0_next[km];
            sm_max_b1[km] <= sm_max_b1_next[km];
        end
    end
end


    // ------------------------------------------------------------
    // Softmax PASS2/PASS3 engine (one mblk at a time)
    // ------------------------------------------------------------
    localparam [1:0] SM_IDLE    = 2'd0;
    localparam [1:0] SM_PASS2   = 2'd1;
    localparam [1:0] SM_PASS3   = 2'd2;
    localparam [1:0] SM_P2FINAL = 2'd3;

    reg [1:0] sm_state;
    reg       sm_issuing;

    reg [6:0] sm_mblk;
    reg       sm_bank;  // SCOREBUF bank being processed by softmax engine
    reg [11:0] sm_n_issue; // 0..M_TOTAL-1

    // sum_work for PASS2 (Q0.16 accumulate)
    reg [31:0] sm_sum_work [0:31];

    // delayed read meta (SCOREBUF is 1-cycle synchronous read)
    reg       sm_rd_en_dly;
    reg       sm_rd_is_p3_dly;
    reg [6:0] sm_rd_mblk_dly;
    reg       sm_rd_bank_dly;
    reg [11:0] sm_rd_n_dly;

    // PASS2 consume pipeline (break SCOREBUF-read -> sm_sum_work timing)
    reg        sm_p2_valid_dly;
    reg [6:0]  sm_p2_mblk_dly;
    reg        sm_p2_bank_dly;
    reg [11:0]  sm_p2_n_dly;
    reg [255:0] sm_p2_score_rd_data_dly;

    // PASS3 consume pipeline (break SCOREBUF-read -> per-bank context select timing)
    reg        sm_p3_valid_dly;
    reg [6:0]  sm_p3_mblk_dly;
    reg        sm_p3_bank_dly;
    reg [11:0]  sm_p3_n_dly;
    reg [255:0] sm_p3_score_rd_data_dly;

    // PASS3a pipeline (selected row params + exp approximation)
    reg        sm_p3a_valid_dly;
    reg [6:0]  sm_p3a_mblk_dly;
    reg        sm_p3a_bank_dly;
    reg [11:0]  sm_p3a_n_dly;
    reg [GOUT_AW-1:0] sm_p3a_wr_addr_dly;
    reg [14:0] sm_p3a_exp_u15 [0:31];
    reg [5:0]  sm_p3a_log0_q [0:31];
    reg        sm_p3a_log0_f [0:31];

    // PASS2 finalize context (2-cycle stage before PASS3)
    reg [6:0] sm_p2final_mblk;
    reg       sm_p2final_bank;
    reg       sm_p2f0_valid;
    reg [6:0] sm_p2f0_mblk;
    reg       sm_p2f0_bank;
    reg [5:0] sm_p2f0_log0_q [0:31];
    reg       sm_p2f0_log0_f [0:31];

    // Registered block-specific softmax-exp ROM stage shared by PASS2/PASS3.
    reg        sm_rom_p2_valid_dly;
    reg [6:0]  sm_rom_p2_mblk_dly;
    reg        sm_rom_p2_bank_dly;
    reg [11:0] sm_rom_p2_n_dly;
    reg        sm_rom_p3_valid_dly;
    reg [6:0]  sm_rom_p3_mblk_dly;
    reg        sm_rom_p3_bank_dly;
    reg [11:0] sm_rom_p3_n_dly;

    reg signed [8:0] sm_exp_rom_diff [0:31];
    reg        [11:0] sm_exp_rom_addr [0:31];
    wire       [14:0] sm_exp_rom_data [0:31];
    wire              sm_exp_rom_en = sm_p2_valid_dly || sm_p3_valid_dly;
    integer sm_rom_lane;

    always @(*) begin
        for (sm_rom_lane = 0; sm_rom_lane < 32; sm_rom_lane = sm_rom_lane + 1) begin
            if (sm_p3_valid_dly) begin
                if (sm_p3_bank_dly == 1'b0) begin
                    sm_exp_rom_diff[sm_rom_lane] =
                        $signed({sm_max_b0[sm_rom_lane][7], sm_max_b0[sm_rom_lane]}) -
                        $signed({sm_p3_score_rd_data_dly[sm_rom_lane*8+7],
                                 sm_p3_score_rd_data_dly[sm_rom_lane*8 +: 8]});
                end else begin
                    sm_exp_rom_diff[sm_rom_lane] =
                        $signed({sm_max_b1[sm_rom_lane][7], sm_max_b1[sm_rom_lane]}) -
                        $signed({sm_p3_score_rd_data_dly[sm_rom_lane*8+7],
                                 sm_p3_score_rd_data_dly[sm_rom_lane*8 +: 8]});
                end
            end else if (sm_p2_valid_dly) begin
                if (sm_p2_bank_dly == 1'b0) begin
                    sm_exp_rom_diff[sm_rom_lane] =
                        $signed({sm_max_b0[sm_rom_lane][7], sm_max_b0[sm_rom_lane]}) -
                        $signed({sm_p2_score_rd_data_dly[sm_rom_lane*8+7],
                                 sm_p2_score_rd_data_dly[sm_rom_lane*8 +: 8]});
                end else begin
                    sm_exp_rom_diff[sm_rom_lane] =
                        $signed({sm_max_b1[sm_rom_lane][7], sm_max_b1[sm_rom_lane]}) -
                        $signed({sm_p2_score_rd_data_dly[sm_rom_lane*8+7],
                                 sm_p2_score_rd_data_dly[sm_rom_lane*8 +: 8]});
                end
            end else begin
                sm_exp_rom_diff[sm_rom_lane] = 9'sd0;
            end

            if (sm_exp_rom_diff[sm_rom_lane] < 0) begin
                sm_exp_rom_addr[sm_rom_lane] = {sm_cfg_idx, 8'd0};
            end else if (sm_exp_rom_diff[sm_rom_lane] > 9'sd255) begin
                sm_exp_rom_addr[sm_rom_lane] = {sm_cfg_idx, 8'd255};
            end else begin
                sm_exp_rom_addr[sm_rom_lane] =
                    {sm_cfg_idx, sm_exp_rom_diff[sm_rom_lane][7:0]};
            end
        end
    end

    genvar sm_rom_g;
    generate
        for (sm_rom_g = 0; sm_rom_g < 32; sm_rom_g = sm_rom_g + 1) begin : GEN_SM_EXP_ROM
            wb_softmax_exp_rom #(.MEM_FILE(SM_EXP_U15_MEM)) u_exp_rom (
                .clk(clk),
                .en(sm_exp_rom_en),
                .addr(sm_exp_rom_addr[sm_rom_g]),
                .dout(sm_exp_rom_data[sm_rom_g])
            );
        end
    endgenerate

    // pick next pending softmax job (bank context):
    // - choose among banks that have completed PASS1 (ctx_ready=1)
    // - if both banks ready, process lower mblk first (to preserve row order)
    reg       sm_pick_valid;
    reg       sm_pick_bank;
    reg [6:0] sm_pick_mblk;
    integer j_done;
    integer j_seq;

    always @(*) begin
        sm_pick_valid = 0;
        sm_pick_bank  = 0;
        sm_pick_mblk  = 0;

        if (sm_ctx_vld[0] && sm_ctx_ready[0] && sm_ctx_vld[1] && sm_ctx_ready[1]) begin
            if (sm_ctx_mblk[0] <= sm_ctx_mblk[1]) begin
                sm_pick_valid = 1;
                sm_pick_bank  = 1'b0;
                sm_pick_mblk  = sm_ctx_mblk[0];
            end else begin
                sm_pick_valid = 1;
                sm_pick_bank  = 1'b1;
                sm_pick_mblk  = sm_ctx_mblk[1];
            end
        end else if (sm_ctx_vld[0] && sm_ctx_ready[0]) begin
            sm_pick_valid = 1;
            sm_pick_bank  = 1'b0;
            sm_pick_mblk  = sm_ctx_mblk[0];
        end else if (sm_ctx_vld[1] && sm_ctx_ready[1]) begin
            sm_pick_valid = 1;
            sm_pick_bank  = 1'b1;
            sm_pick_mblk  = sm_ctx_mblk[1];
        end
    end

    // SCOREBUF read request
    wire       sm_sbuf_rd_en_now;
    wire       sm_sbuf_rd_is_p3_now;
    wire [6:0] sm_sbuf_rd_mblk_now;
    wire [11:0] sm_sbuf_rd_n_now;
    wire       sm_sbuf_rd_bank_now;

    assign sm_sbuf_rd_en_now    = mode_score && sm_issuing;
    assign sm_sbuf_rd_is_p3_now = (sm_state == SM_PASS3);
    assign sm_sbuf_rd_mblk_now  = sm_mblk;
    assign sm_sbuf_rd_n_now     = sm_n_issue;
    assign sm_sbuf_rd_bank_now  = sm_bank;

// SCOREBUF read mux:
// - SCORE softmax engine uses SCOREBUF during mode_score
// - OPROJ residual add reads SCOREBUF once per accepted output word (1-cycle sync read)
// - PATCH final positional add reuses the same residual layout as OPROJ
wire        score_rd_en_sm   = sm_sbuf_rd_en_now;
wire        score_rd_bank_sm = sm_sbuf_rd_bank_now;
wire [11:0] score_rd_addr_sm = sm_sbuf_rd_n_now;

// OPROJ/PATCH_POS: residual-like data is prefetched into SCOREBUF by read_controller:
//   bank = nblk_idx_eff[0] ping-pong (even/odd nblk)
//   addr = (mblk_idx_eff*32) + col_in_blk
wire        mode_score_residual = mode_oproj || (mode_patch && cur_is_final_ktile);
// Register residual SCOREBUF requests before the BRAM. This removes the
// scheduler/ready/accept chain from the BRAM enable path. The extra request
// cycle is compensated below by taking score_rd_data directly into s0b.
reg         score_rd_en_res_q;
reg         score_rd_bank_res_q;
reg  [11:0] score_rd_addr_res_q;
always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        score_rd_en_res_q   <= 1'b0;
        score_rd_bank_res_q <= 1'b0;
        score_rd_addr_res_q <= 12'd0;
    end else begin
        score_rd_en_res_q <= accept_eff && mode_score_residual;
        if (accept_eff && mode_score_residual) begin
            score_rd_bank_res_q <= nblk_idx_eff[0];
            score_rd_addr_res_q <= {mblk_idx_eff, 5'd0} + col_in_blk;
        end
    end
end

assign score_rd_en   = mode_score ? score_rd_en_sm   : score_rd_en_res_q;
assign score_rd_bank = mode_score ? score_rd_bank_sm : score_rd_bank_res_q;
assign score_rd_addr = mode_score ? score_rd_addr_sm : score_rd_addr_res_q;

    // Softmax write regs (to SCOREBUF row-local qlog overwrite)
    reg         sm_wr_valid_r;
    reg         sm_wr_bank_r;
    reg [GOUT_AW-1:0]  sm_wr_addr_r;
    reg [255:0] sm_wr_data_r;

    assign sm_score_wr_valid_w = sm_wr_valid_r;
    assign sm_score_wr_bank_w  = sm_wr_bank_r;
    assign sm_score_wr_addr_w  = sm_wr_addr_r[11:0];
    assign sm_score_wr_data_w  = sm_wr_data_r;

    assign attn_wr_valid = 1'b0;
    assign attn_wr_addr  = {GOUT_AW{1'b0}};
    assign attn_wr_data  = 256'd0;

    // Compatibility aliases for existing debug testbenches that monitor the
    // softmax output stream through hierarchical names.
    wire        sm_gout_wr_valid = sm_wr_valid_r;
    wire [GOUT_AW-1:0] sm_gout_wr_addr = sm_wr_addr_r;
    wire [255:0] sm_gout_wr_data = sm_wr_data_r;

    // sm_busy: SCORE mode + any pending or in-flight
    reg sm_all_done;
    always @(*) begin
        sm_all_done = 1;
        for (j_done = 0; j_done < M_BLK_NUM; j_done = j_done + 1) begin
            if (active_mblk_mask[j_done] && !sm_done[j_done]) sm_all_done = 0;
        end
    end

    assign sm_busy = mode_score && ((sm_state != SM_IDLE) || sm_rd_en_dly || sm_p2_valid_dly || sm_p3_valid_dly ||
                     sm_rom_p2_valid_dly || sm_rom_p3_valid_dly || sm_p3a_valid_dly || sm_p2f0_valid ||
                     sm_wr_valid_r || sm_ctx_ready[0] || sm_ctx_ready[1]);

    assign wb_busy = pe_out_valid || s0_valid || s0a_valid || s0b_valid || p1_valid || p2a_valid || p2_valid ||
                     s1_valid_q8 || s1_valid_core || s1m_valid_core || s1a_valid_core ||
                     s2_valid_core || s2m_valid_core ||
                     s3m_valid_core || s3a_valid_core || s3b_valid_core || s3c_valid_core ||
                     s3d_valid_core || s3_valid_core || sm_busy;

    // ------------------------------------------------------------
    // Softmax datapath (PASS2 consume / PASS3 consume)
    // ------------------------------------------------------------
    integer k;
    reg signed [7:0]  v_s8;
    reg signed [8:0]  diff9;
    reg [7:0]         diff_u8;

    reg [14:0]        exp_u15;
    reg [31:0]        sum_next;
    reg [31:0]        sum_norm;
    reg [5:0]         log0_q;
    reg [5:0]         log1_q;
    reg               log0_f;
    reg               log1_f;
    reg signed [7:0]  logdiff_half_s;
    reg signed [7:0]  shift_round_s;
    reg [3:0]         qlog_u4;

    // ------------------------------------------------------------
    // Sequential control
    // ------------------------------------------------------------
    always @(posedge clk) begin
        if (!nRst) begin
            scorebuf_wr_bank_r <= 0;
            prev_mode_score <= 0;

            sm_state   <= SM_IDLE;
            sm_issuing <= 0;
            sm_mblk    <= 0;
            sm_bank    <= 0;
            sm_n_issue <= 0;

            sm_rd_en_dly     <= 0;
            sm_rd_is_p3_dly  <= 0;
            sm_rd_mblk_dly   <= 0;
            sm_rd_bank_dly  <= 0;
            sm_rd_n_dly      <= 0;
            sm_p2_valid_dly  <= 0;
            sm_p2_mblk_dly   <= 0;
            sm_p2_bank_dly   <= 0;
            sm_p2_n_dly      <= 0;
            sm_p2_score_rd_data_dly <= 0;
            sm_p3_valid_dly  <= 0;
            sm_p3_mblk_dly   <= 0;
            sm_p3_bank_dly   <= 0;
            sm_p3_n_dly      <= 0;
            sm_p3_score_rd_data_dly <= 0;
            sm_p3a_valid_dly <= 0;
            sm_p3a_mblk_dly  <= 0;
            sm_p3a_bank_dly  <= 0;
            sm_p3a_n_dly     <= 0;
            sm_p3a_wr_addr_dly <= 0;
            sm_p2final_mblk <= 0;
            sm_p2final_bank <= 0;
            sm_p2f0_valid   <= 0;
            sm_p2f0_mblk    <= 0;
            sm_p2f0_bank    <= 0;
            sm_rom_p2_valid_dly <= 0;
            sm_rom_p2_mblk_dly  <= 0;
            sm_rom_p2_bank_dly  <= 0;
            sm_rom_p2_n_dly     <= 0;
            sm_rom_p3_valid_dly <= 0;
            sm_rom_p3_mblk_dly  <= 0;
            sm_rom_p3_bank_dly  <= 0;
            sm_rom_p3_n_dly     <= 0;

            sm_wr_valid_r <= 0;
            sm_wr_bank_r  <= 0;
            sm_wr_addr_r  <= 0;
            sm_wr_data_r  <= 0;

            // softmax bookkeeping reset
            sm_ctx_vld[0]   <= 0;
            sm_ctx_ready[0] <= 0;
            sm_ctx_mblk[0]  <= 0;
            sm_ctx_vld[1]   <= 0;
            sm_ctx_ready[1] <= 0;
            sm_ctx_mblk[1]  <= 0;

            for (j_seq = 0; j_seq < M_BLK_NUM; j_seq = j_seq + 1) begin
                sm_done[j_seq] <= 0;
            end
            for (j_seq = 0; j_seq < 32; j_seq = j_seq + 1) begin
                sm_log0_q_b0[j_seq] <= 0;
                sm_log0_q_b1[j_seq] <= 0;
                sm_log0_f_b0[j_seq] <= 0;
                sm_log0_f_b1[j_seq] <= 0;
            end
            for (k = 0; k < 32; k = k + 1) begin
                sm_sum_work[k] <= 0;
                sm_p3a_exp_u15[k] <= 0;
                sm_p3a_log0_q[k] <= 0;
                sm_p3a_log0_f[k] <= 0;
                sm_p2f0_log0_q[k] <= 0;
                sm_p2f0_log0_f[k] <= 0;
            end
        end else begin
            // default: no softmax write unless PASS3 consume fires
            sm_wr_valid_r <= 0;
            sm_p2_valid_dly <= 0;
            sm_p3_valid_dly <= 0;
            sm_p3a_valid_dly <= 0;
            sm_rom_p2_valid_dly <= sm_p2_valid_dly;
            sm_rom_p3_valid_dly <= sm_p3_valid_dly;
            prev_mode_score <= mode_score;

            if (sm_p2_valid_dly) begin
                sm_rom_p2_mblk_dly <= sm_p2_mblk_dly;
                sm_rom_p2_bank_dly <= sm_p2_bank_dly;
                sm_rom_p2_n_dly    <= sm_p2_n_dly;
            end
            if (sm_p3_valid_dly) begin
                sm_rom_p3_mblk_dly <= sm_p3_mblk_dly;
                sm_rom_p3_bank_dly <= sm_p3_bank_dly;
                sm_rom_p3_n_dly    <= sm_p3_n_dly;
            end

            // ----------------------------------------------------
            // SCORE mode entry: clear state
            // ----------------------------------------------------
            if (!prev_mode_score && mode_score) begin
                scorebuf_wr_bank_r <= 0;

                sm_state   <= SM_IDLE;
                sm_issuing <= 0;
                sm_mblk    <= 0;
                sm_bank    <= 0;
                sm_n_issue <= 0;

                sm_rd_en_dly     <= 0;
                sm_rd_is_p3_dly  <= 0;
                sm_rd_mblk_dly   <= 0;
                sm_rd_bank_dly   <= 0;
                sm_rd_n_dly      <= 0;
                sm_p2_valid_dly  <= 0;
                sm_p2_mblk_dly   <= 0;
                sm_p2_bank_dly   <= 0;
                sm_p2_n_dly      <= 0;
                sm_p2_score_rd_data_dly <= 0;
                sm_p3_valid_dly  <= 0;
                sm_p3_mblk_dly   <= 0;
                sm_p3_bank_dly   <= 0;
                sm_p3_n_dly      <= 0;
                sm_p3_score_rd_data_dly <= 0;
                sm_p3a_valid_dly <= 0;
                sm_p3a_mblk_dly  <= 0;
                sm_p3a_bank_dly  <= 0;
                sm_p3a_n_dly     <= 0;
                sm_p3a_wr_addr_dly <= 0;
            sm_p2final_mblk <= 0;
            sm_p2final_bank <= 0;
            sm_p2f0_valid   <= 0;
            sm_p2f0_mblk    <= 0;
            sm_p2f0_bank    <= 0;
            sm_rom_p2_valid_dly <= 0;
            sm_rom_p3_valid_dly <= 0;

                sm_ctx_vld[0]   <= 0;
                sm_ctx_ready[0] <= 0;
                sm_ctx_mblk[0]  <= 0;
                sm_ctx_vld[1]   <= 0;
                sm_ctx_ready[1] <= 0;
                sm_ctx_mblk[1]  <= 0;

                for (j_seq = 0; j_seq < M_BLK_NUM; j_seq = j_seq + 1) begin
                    sm_done[j_seq] <= 0;
                end
                for (j_seq = 0; j_seq < 32; j_seq = j_seq + 1) begin
                    sm_log0_q_b0[j_seq] <= 0;
                    sm_log0_q_b1[j_seq] <= 0;
                    sm_log0_f_b0[j_seq] <= 0;
                    sm_log0_f_b1[j_seq] <= 0;
                end
                for (k = 0; k < 32; k = k + 1) begin
                    sm_sum_work[k] <= 0;
                    sm_p3a_exp_u15[k] <= 0;
                    sm_p3a_log0_q[k] <= 0;
                    sm_p3a_log0_f[k] <= 0;
                end
            end

            // ----------------------------------------------------
            // PASS1 (max): update per accepted VALID score word
            // - NOTE: sm_max_b0/b1 are updated in a dedicated next-state block (avoid shared high-fanout CE).
            //   sm_log0_* are still updated in this sequential block (PASS1 init + PASS2 finalize).
            // ----------------------------------------------------
            if (scorebuf_wr_valid_w) begin
                if (scorebuf_wr_bank_r == 1'b0) begin
                    // bank0 context
                    if (!sm_ctx_vld[0]) begin
                        sm_ctx_vld[0]   <= 1'b1;
                        sm_ctx_ready[0] <= 1'b0;
                        sm_ctx_mblk[0]  <= s1_mb_q8;
                        for (k = 0; k < 32; k = k + 1) begin
                            sm_log0_q_b0[k] <= 0;
                            sm_log0_f_b0[k] <= 0;
                        end
                    end
                end else begin
                    // bank1 context
                    if (!sm_ctx_vld[1]) begin
                        sm_ctx_vld[1]   <= 1'b1;
                        sm_ctx_ready[1] <= 1'b0;
                        sm_ctx_mblk[1]  <= s1_mb_q8;
                        for (k = 0; k < 32; k = k + 1) begin
                            sm_log0_q_b1[k] <= 0;
                            sm_log0_f_b1[k] <= 0;
                        end
                    end
                end
            end


            // row-block done: mark this bank context as READY for PASS2 and toggle bank
            if (score_last_valid_word) begin
                if (scorebuf_wr_bank_r == 1'b0) begin
                    sm_ctx_ready[0] <= 1'b1;
                    sm_ctx_mblk[0]  <= s1_mb_q8;
                end else begin
                    sm_ctx_ready[1] <= 1'b1;
                    sm_ctx_mblk[1]  <= s1_mb_q8;
                end
                scorebuf_wr_bank_r <= ~scorebuf_wr_bank_r;
            end


            // ----------------------------------------------------
            // SCOREBUF read meta pipeline (1-cycle)
            // ----------------------------------------------------
            sm_rd_en_dly     <= sm_sbuf_rd_en_now;
            sm_rd_is_p3_dly  <= sm_sbuf_rd_is_p3_now;
            sm_rd_mblk_dly   <= sm_sbuf_rd_mblk_now;
            sm_rd_bank_dly   <= sm_sbuf_rd_bank_now;
            sm_rd_n_dly      <= sm_sbuf_rd_n_now;

            // ----------------------------------------------------
            // PASS2 consume pipeline capture (register SCOREBUF data/meta)
            // ----------------------------------------------------
            if (sm_rd_en_dly && !sm_rd_is_p3_dly) begin
                sm_p2_valid_dly <= 1'b1;
                sm_p2_mblk_dly  <= sm_rd_mblk_dly;
                sm_p2_bank_dly  <= sm_rd_bank_dly;
                sm_p2_n_dly     <= sm_rd_n_dly;
                sm_p2_score_rd_data_dly <= score_rd_data;
            end

            // ----------------------------------------------------
            // PASS3 consume pipeline capture (register SCOREBUF data/meta)
            // ----------------------------------------------------
            if (sm_rd_en_dly && sm_rd_is_p3_dly) begin
                sm_p3_valid_dly <= 1'b1;
                sm_p3_mblk_dly  <= sm_rd_mblk_dly;
                sm_p3_bank_dly  <= sm_rd_bank_dly;
                sm_p3_n_dly     <= sm_rd_n_dly;
                sm_p3_score_rd_data_dly <= score_rd_data;
            end

            // ----------------------------------------------------
            // PASS3a consume: capture the registered exp-ROM output.
            // ----------------------------------------------------
            if (sm_rom_p3_valid_dly) begin
                sm_p3a_valid_dly <= 1'b1;
                sm_p3a_mblk_dly  <= sm_rom_p3_mblk_dly;
                sm_p3a_bank_dly  <= sm_rom_p3_bank_dly;
                sm_p3a_n_dly     <= sm_rom_p3_n_dly;
                // SCORE softmax output is consumed by AV immediately for the
                // same mblk, so overwrite the row-local SCOREBUF addresses.
                sm_p3a_wr_addr_dly <= sm_rom_p3_n_dly;

                for (k = 0; k < 32; k = k + 1) begin
                    if ((sm_rom_p3_mblk_dly != M_BLK_LAST) || (k <= VALID_LANES_LAST)) begin
                        sm_p3a_exp_u15[k] <= sm_exp_rom_data[k];
                        if (sm_rom_p3_bank_dly == 1'b0) begin
                            log0_q = sm_log0_q_b0[k];
                            log0_f = sm_log0_f_b0[k];
                        end else begin
                            log0_q = sm_log0_q_b1[k];
                            log0_f = sm_log0_f_b1[k];
                        end
                        sm_p3a_log0_q[k] <= log0_q;
                        sm_p3a_log0_f[k] <= log0_f;
                    end else begin
                        sm_p3a_exp_u15[k] <= 0;
                        sm_p3a_log0_q[k] <= 0;
                        sm_p3a_log0_f[k] <= 0;
                    end
                end
            end

            // ----------------------------------------------------
// PASS2 consume: accumulate exp_sum (Wooyoung INT_Softmax)
//   - NOTE: LOD + next-bit finalize is moved to SM_P2FINAL.
// ----------------------------------------------------
if (sm_rom_p2_valid_dly) begin
    for (k = 0; k < 32; k = k + 1) begin
        if ((sm_rom_p2_mblk_dly != M_BLK_LAST) || (k <= VALID_LANES_LAST)) begin
            exp_u15  = sm_exp_rom_data[k];
            sum_next = sm_sum_work[k] + {17'd0, exp_u15};
            if (sum_next > ((32'd1 << SM_ROW_SUM_BITS) - 1'b1)) begin
                sm_sum_work[k] <= (32'd1 << SM_ROW_SUM_BITS) - 1'b1;
            end else begin
                sm_sum_work[k] <= sum_next;
            end
        end
    end

    if (sm_rom_p2_n_dly == TOKEN_LAST_IDX) begin
        // PASS2 done -> compute LOD + next-bit sum log in SM_P2FINAL, then start PASS3
        sm_state        <= SM_P2FINAL;
        sm_p2final_mblk <= sm_rom_p2_mblk_dly;
        sm_p2final_bank <= sm_rom_p2_bank_dly;
    end
end

// ----------------------------------------------------
// SM_P2FINAL: 2-cycle finalize
//   P2F0: ilog2 floor + next bit
//   P2F1: write row params, then start PASS3
// ----------------------------------------------------
if (sm_state == SM_P2FINAL) begin
    if (!sm_p2f0_valid) begin
        sm_p2f0_valid <= 1'b1;
        sm_p2f0_mblk  <= sm_p2final_mblk;
        sm_p2f0_bank  <= sm_p2final_bank;

        for (k = 0; k < 32; k = k + 1) begin
            if ((sm_p2final_mblk != M_BLK_LAST) || (k <= VALID_LANES_LAST)) begin
                sum_norm = sm_sum_work[k];
                if (sum_norm == 32'd0) begin
                    sum_norm = 32'd1;
                end

                log0_q = ilog2_floor_u32(sum_norm);
                log0_f = log2_next_bit_u32(sum_norm, log0_q);

                sm_p2f0_log0_q[k] <= log0_q;
                sm_p2f0_log0_f[k] <= log0_f;
            end else begin
                sm_p2f0_log0_q[k] <= 0;
                sm_p2f0_log0_f[k] <= 0;
            end
        end
    end else begin
        sm_p2f0_valid <= 1'b0;

        for (k = 0; k < 32; k = k + 1) begin
            if ((sm_p2f0_mblk != M_BLK_LAST) || (k <= VALID_LANES_LAST)) begin
                if (sm_p2f0_bank == 1'b0) begin
                    sm_log0_q_b0[k] <= sm_p2f0_log0_q[k];
                    sm_log0_f_b0[k] <= sm_p2f0_log0_f[k];
                end else begin
                    sm_log0_q_b1[k] <= sm_p2f0_log0_q[k];
                    sm_log0_f_b1[k] <= sm_p2f0_log0_f[k];
                end
            end
        end

        // start PASS3 (issue reads from the same mblk/bank)
        sm_state   <= SM_PASS3;
        sm_issuing <= 1;
        sm_mblk    <= sm_p2f0_mblk;
        sm_bank    <= sm_p2f0_bank;
        sm_n_issue <= 0;
    end
end

// PASS3b consume: write the legacy DEFOM 401 delivered 4-bit log code.
// ----------------------------------------------------
if (sm_p3a_valid_dly) begin
    sm_wr_valid_r <= 1;
    sm_wr_bank_r  <= sm_p3a_bank_dly;
    sm_wr_addr_r  <= sm_p3a_wr_addr_dly;

    // qlog code = 15-distance, including exp=0. Keep bit 4 clear so AV
    // receives the same finite log-coded contribution as the old RTL.
    for (k = 0; k < 32; k = k + 1) begin
        if ((sm_p3a_mblk_dly != M_BLK_LAST) || (k <= VALID_LANES_LAST)) begin
            exp_u15 = sm_p3a_exp_u15[k];
            log0_q = sm_p3a_log0_q[k];
            log0_f = sm_p3a_log0_f[k];

            if (exp_u15 == 15'd0) begin
                log1_q = 6'd0;
                log1_f = 1'b0;
            end else begin
                log1_q = ilog2_floor_u32({17'd0, exp_u15});
                log1_f = log2_next_bit_u32({17'd0, exp_u15}, log1_q);
            end

            // diff_q = 2*(log0_q - log1_q) + (log0_f - log1_f)
            // rounds = (diff_q + 1) >> 1
            // qlog shift = clamp(rounds, 0, 15)
            logdiff_half_s =
                ($signed({1'b0, log0_q}) - $signed({1'b0, log1_q})) <<< 1;
            logdiff_half_s = logdiff_half_s +
                ($signed({7'd0, log0_f}) - $signed({7'd0, log1_f}));
            shift_round_s = (logdiff_half_s + 8'sd1) >>> 1;

            if (shift_round_s < 0) begin
                qlog_u4 = 4'd0;
            end else if (shift_round_s > 8'sd15) begin
                qlog_u4 = 4'd15;
            end else begin
                qlog_u4 = shift_round_s[3:0];
            end

            sm_wr_data_r[k*8 +: 8] <= {4'd0, (4'd15 - qlog_u4)};
        end else begin
            sm_wr_data_r[k*8 +: 8] <= 8'd0;
        end
    end

    if (sm_p3a_n_dly == TOKEN_LAST_IDX) begin
        sm_done[sm_p3a_mblk_dly] <= 1;
        // free bank context
        if (sm_p3a_bank_dly == 1'b0) begin
            sm_ctx_vld[0]   <= 1'b0;
            sm_ctx_ready[0] <= 1'b0;
            sm_ctx_mblk[0]  <= 0;
        end else begin
            sm_ctx_vld[1]   <= 1'b0;
            sm_ctx_ready[1] <= 1'b0;
            sm_ctx_mblk[1]  <= 0;
        end
        sm_state   <= SM_IDLE;
        sm_issuing <= 0;
        sm_n_issue <= 0;
    end
end

// ----------------------------------------------------
// PASS2/PASS3 issue progression + IDLE->PASS2 start
            // ----------------------------------------------------
            if (mode_score) begin
                if (sm_state == SM_IDLE) begin
                    if (sm_pick_valid) begin
                        sm_state   <= SM_PASS2;
                        sm_issuing <= 1;
                        sm_mblk    <= sm_pick_mblk;
                        sm_bank    <= sm_pick_bank;
                        if (sm_pick_bank == 1'b0) begin
                            sm_ctx_ready[0] <= 1'b0;
                        end else begin
                            sm_ctx_ready[1] <= 1'b0;
                        end
                        sm_n_issue <= 0;
                        for (k = 0; k < 32; k = k + 1) begin
                            sm_sum_work[k] <= 0;
                        end
                    end
                end else begin
                    // issuing reads in PASS2 or PASS3
                    if (sm_issuing) begin
                        if (sm_n_issue == TOKEN_LAST_IDX) begin
                            sm_issuing <= 0; // stop issuing; wait last consume to advance/finish
                        end else begin
                            sm_n_issue <= sm_n_issue + 1;
                        end
                    end
                end
            end else begin
                // not SCORE: keep softmax engine idle
                sm_state   <= SM_IDLE;
                sm_issuing <= 0;
                sm_mblk    <= 0;
                sm_bank    <= 0;
                sm_n_issue <= 0;
                sm_p2f0_valid <= 0;
                sm_p2f0_mblk  <= 0;
                sm_p2f0_bank  <= 0;
                sm_rom_p2_valid_dly <= 0;
                sm_rom_p3_valid_dly <= 0;
            end
        end
    end

// ------------------------------------------------------------
    // GOUT write port: SCORE softmax no longer uses the global buffer.
    // ------------------------------------------------------------
    assign gout_wr_valid = core_gout_wr_valid;
    assign gout_wr_addr  = core_gout_wr_addr;
    assign gout_wr_data  = core_gout_wr_data;

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            s0_valid    <= 0;
            s0a_valid   <= 0;
            gout_rd_en_q   <= 0;
            gout_rd_addr_q <= 0;
            psum_rd_en_q   <= 0;
            psum_rd_addr_q <= 0;
            gin_rd_en_q    <= 0;
            gin_rd_addr_q  <= 0;
            score_rd_data_d1 <= 0;
            bias_rd_data_d1  <= 0;
            s0_kt       <= 0;
            s0_nb       <= 0;
            s0_mb       <= 0;
            s0_col      <= 0;
            s0_addr     <= 0;
            s0_quant_word <= 0;
            s0a_kt       <= 0;
            s0a_nb       <= 0;
            s0a_mb       <= 0;
            s0a_col      <= 0;
            s0a_addr     <= 0;
            s0a_cur_flat <= 0;
            s0a_quant_word <= 0;
            s0a_layer_id <= 0;
            s0a_layer_final_ktile <= 0;
            s0a_cls_alt_apply <= 0;

            s0_layer_id <= 0;
            s0_layer_final_ktile <= 0;
            s0_cls_alt_apply <= 0;
            s0_shift_prev_eff <= SHIFT_PART;
            s0_shift_store_eff <= SHIFT_PART;
            s0b_valid    <= 0;
            s0b_kt       <= 0;
            s0b_nb       <= 0;
            s0b_mb       <= 0;
            s0b_col      <= 0;
            s0b_addr     <= 0;
            s0b_cur_flat <= 0;
            s0b_layer_id <= 0;
            s0b_layer_final_ktile <= 0;
            s0b_cls_alt_apply <= 0;
            s0b_shift_prev_eff <= SHIFT_PART;
            s0b_shift_store_eff <= SHIFT_PART;
            s0b_gout_rd_data <= 0;
            s0b_psum_rd_data <= 0;
            s0b_gin_rd_data  <= 0;
            s0b_score_rd_data<= 0;
            s0b_bias_rd_data <= 0;
            final_tile_pulse <= 0;
            final_nblk_idx   <= 0;
            final_mblk_idx   <= 0;
        end else begin
            final_tile_pulse <= 0;

            // Register global read requests once before mem_ctrl.
            gout_rd_en_q   <= core_gout_rd_en;
            gout_rd_addr_q <= core_gout_rd_addr;
            psum_rd_en_q   <= core_psum_rd_en;
            psum_rd_addr_q <= core_psum_rd_addr;
            gin_rd_en_q    <= core_gin_rd_en;
            gin_rd_addr_q  <= {{(GOUT_AW-12){1'b0}}, core_gin_rd_addr_compact};

            // Decouple first-beat/live metadata control from the wide stage0 payload regs.
            // These payload regs update every cycle; stage0 validity still comes from accept_eff.
            s0_cur_flat   <= pe_out_i32_flat;
            s0_quant_word <= quant_word_eff;

            s0_valid <= accept_eff;
            if (accept_eff) begin
                s0_kt       <= ktile_idx_eff;
                s0_nb       <= nblk_idx_eff;
                s0_mb       <= mblk_idx_eff;
                s0_col      <= col_in_blk;
                s0_addr     <= calc_word_addr_rowmajor(mode_v ? mblk_idx_eff : nblk_idx_eff, col_in_blk, mode_v ? nblk_idx_eff : mblk_idx_eff);
                s0_layer_id <= layer_id_eff;
                s0_layer_final_ktile <= layer_final_ktile_eff;
                s0_cls_alt_apply <= cls_alt_apply_eff;

                // Keep defaults in stage0; actual SQPS shift extraction is deferred to stage0b
                // so feeder->writeback ktile routing does not sit on the critical path.
                s0_shift_prev_eff <= SHIFT_PART;
                s0_shift_store_eff <= SHIFT_PART;
            end

            // Delay taps for SCORE/BIAS synchronous-read payloads.
            score_rd_data_d1 <= score_rd_data;
            bias_rd_data_d1  <= bias_rd_data;

            // stage0a capture: align metadata with the registered GOUT/GIN requests.
            // Update wide payload every cycle so accept_eff does not directly drive their CE.
            s0a_cur_flat   <= s0_cur_flat;
            s0a_quant_word <= s0_quant_word;
            s0a_valid <= s0_valid;
            if (s0_valid) begin
                s0a_kt       <= s0_kt;
                s0a_nb       <= s0_nb;
                s0a_mb       <= s0_mb;
                s0a_col      <= s0_col;
                s0a_addr     <= s0_addr;
                s0a_layer_id <= s0_layer_id;
                s0a_layer_final_ktile <= s0_layer_final_ktile;
                s0a_cls_alt_apply <= s0_cls_alt_apply;
            end

            // stage0b capture: GOUT/GIN now arrive one cycle later.
            s0b_valid <= s0a_valid;
            if (s0a_valid) begin
                s0b_kt       <= s0a_kt;
                s0b_nb       <= s0a_nb;
                s0b_mb       <= s0a_mb;
                s0b_col      <= s0a_col;
                s0b_addr     <= s0a_addr;
                s0b_cur_flat <= s0a_cur_flat;
                s0b_layer_id <= s0a_layer_id;
                s0b_layer_final_ktile <= s0a_layer_final_ktile;
                s0b_cls_alt_apply <= s0a_cls_alt_apply;
                s0b_shift_prev_eff <= SHIFT_PART;
                s0b_shift_store_eff <= SHIFT_PART;
                if ((s0a_layer_id == L_PATCH) || (s0a_layer_id == L_MLP_FC2)) begin
                    if (s0a_kt != 6'd0) begin
                        s0b_shift_prev_eff <= f_acc_sh_qw(s0a_quant_word, f_acc_sh_local_idx_for_kt(s0a_kt, s0a_kt - 6'd1));
                    end
                    s0b_shift_store_eff <= f_acc_sh_qw(s0a_quant_word, f_acc_sh_local_idx_for_kt(s0a_kt, s0a_kt));
                end
                s0b_gout_rd_data <= gout_rd_data;
                s0b_psum_rd_data <= psum_rd_data;
                s0b_gin_rd_data  <= gin_rd_data;
                // Residual SCOREBUF reads have one registered request cycle,
                // so the direct BRAM output is aligned with s0a -> s0b.
                s0b_score_rd_data<= score_rd_data;
                s0b_bias_rd_data <= bias_rd_data_d1;
            end

            // End-of-tile (stream) detection:
            // - IMPORTANT: tile/loop advance must follow the PE's full 32-beat column stream (accept).
            //   Even if some beats are 'invalid' (out_word_valid=0) in the last partial block,
            //   we still must DRAIN those beats to avoid meta(nblk/mblk) advancing mid-stream.
            if (accept && cur_is_final_eff && (col_in_blk == 5'd31)) begin
                final_tile_pulse <= 1'b1;

                // For V dump we treat: mblk = channel_block (0..5), nblk = token_block (0..6)
                if (mode_v) begin
                    final_mblk_idx <= nblk_idx_eff; // channel block
                    final_nblk_idx <= mblk_idx_eff; // token block
                end else begin
                    final_nblk_idx <= nblk_idx_eff;
                    final_mblk_idx <= mblk_idx_eff;
                end
            end
        end
    end

endmodule

module wb_softmax_exp_rom #(
    parameter MEM_FILE = "softmax_exp_u15.mem"
)(
    input  wire        clk,
    input  wire        en,
    input  wire [11:0] addr,
    output reg  [14:0] dout
);
    (* rom_style = "distributed" *) reg [14:0] mem [0:3071];
    integer init_i;

    initial begin
        for (init_i = 0; init_i < 3072; init_i = init_i + 1) begin
            mem[init_i] = 15'd0;
        end
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        if (en) begin
            dout <= mem[addr];
        end
    end
endmodule

`default_nettype wire
