`timescale 1ns/1ps
`default_nettype none

// -----------------------------------------------------------------------------
// SCOREBUF-backed post-layer LayerNorm pass.
//
// The engine runs while vit_controller is held at layer end, so the existing
// GOUT port points to the just-written output bank. It performs a row-wise
// GOUT -> SCOREBUF -> GOUT pass for one full activation tensor:
//   - PASS_LOAD reads one m-block from GOUT and stores the 384 channel words
//     into the two physical SCOREBUF banks as a single scratch space.
//   - During PASS_LOAD, each byte lane accumulates sum and sumsq across all
//     channels. In this memory layout, each lane corresponds to one token row.
//   - PASS_STAT computes row statistics one lane at a time and maps each row
//     variance to a 64-entry reciprocal LUT.
//   - PASS_WRITE reads the scratch words back, applies the Wooyoung-style INT
//     LayerNorm datapath, and writes GOUT at the same addresses. By default,
//     gamma/beta are assumed to be folded into the following linear layers; an
//     optional ROM affine mode is retained for experiments.  In dual-affine
//     mode, the row statistics and SCOREBUF copy are shared while two affine
//     parameter sets are applied sequentially to different global-buffer banks.
//
// APPLY_LAYER_NORM=0 preserves the bring-up identity pass. APPLY_LAYER_NORM=1
// enables a Wooyoung-style integer LayerNorm core:
//     variance_core = C*sum(x^2) - sum(x)^2
//     variance_u12  = clamp(round((variance_core + epsilon)/2^VAR_SHIFT), 1, 4095)
//     reciprocal    = recip_lut[std_code(variance_u12)-1]
//     y_q           = sat8(((M_row * (x_q - mean_q0)) >>> N_row) + beta)
// where M/N are derived from reciprocal and either a shared folded-LN gain,
// layer/channel-wise ROM gamma_q16 when LN_AFFINE_MODE=1, or bias-buffer
// gamma/beta words when LN_AFFINE_MODE=2.
// -----------------------------------------------------------------------------
module layernorm_scratch_engine #(
    parameter integer MEM_WIDTH         = 256,
    parameter integer EMBED_DIM         = 384,
    parameter integer OUT_STRIDE        = 384,
    parameter integer M_BLK_NUM         = 7,
    parameter integer GOUT_AW           = 12,
    parameter integer SCORE_BANK_DEPTH  = 256,
    // Inclusive index of the final valid lane in the last m-block.
    parameter integer VALID_LANES_LAST  = 5,
    parameter integer APPLY_LAYER_NORM  = 0,
    parameter integer RSQRT_FRAC_BITS   = 12,
    parameter integer NORM_GAIN_SHIFT   = 5,
    parameter integer LN_VAR_SHIFT      = 18,
    parameter integer LN_GAIN_FRAC_BITS = 0,
    parameter integer LN_EPSILON_CORE   = 0,
    parameter integer LN_AFFINE_MODE    = 0,
    parameter integer LN_PARAM_SETS     = 29,
    parameter integer LN_BASE_GAIN_Q16  = 2097152,
    parameter LN_GAIN_MEM               = "ln_gain_q16.mem",
    parameter LN_BIAS_MEM               = "ln_bias_i8.mem",
    parameter LN_RECIP_MEM              = "ln_recip_lut_u16.mem",
    parameter LN_EPSILON_MEM            = "ln_epsilon_u15.mem"
)(
    input  wire                   clk,
    input  wire                   nRst,

    input  wire                   start,
    input  wire [5:0]             ln_param_sel,
    input  wire                   dual_affine,
    input  wire                   primary_out_bank,
    input  wire                   secondary_out_bank,
    input  wire [6:0]             active_mblk_num,
    input  wire [4:0]             active_last_valid_lane,
    output reg                    busy,
    output reg                    done,

    // Current GOUT bank, selected by memory_controller layer_inbuf_sel.
    output reg                    gout_rd_en,
    output reg  [GOUT_AW-1:0]     gout_rd_addr,
    input  wire [MEM_WIDTH-1:0]   gout_rd_data,

    output reg                    gout_wr_valid,
    output reg  [GOUT_AW-1:0]     gout_wr_addr,
    output reg  [MEM_WIDTH-1:0]   gout_wr_data,
    output reg                    gout_wr_bank,

    // SCOREBUF is used as one flat scratch buffer. For EMBED_DIM=384 and
    // SCORE_BANK_DEPTH=256, channel words 0..255 go to bank0 and 256..383
    // go to bank1.
    output reg                    score_rd_en,
    output reg                    score_rd_bank,
    output reg  [11:0]            score_rd_addr,
    input  wire [MEM_WIDTH-1:0]   score_rd_data,

    output reg                    score_wr_valid,
    output reg                    score_wr_bank,
    output reg  [11:0]            score_wr_addr,
    output reg  [MEM_WIDTH-1:0]   score_wr_data,

    // Optional affine parameter read port. LN_AFFINE_MODE=2 reuses the
    // existing bias buffer.  The primary set occupies gamma 0..47 and beta
    // 48..59.  The optional secondary set occupies gamma 60..107 and beta
    // 108..119 for EMBED_DIM=384.
    output reg                    aff_rd_en,
    output reg  [6:0]             aff_rd_addr,
    input  wire [MEM_WIDTH-1:0]   aff_rd_data,

    output reg  [6:0]             dbg_mblk,
    output reg  [2:0]             dbg_state
);

    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_LOAD       = 3'd1;
    localparam [2:0] ST_STAT       = 3'd2;
    localparam [2:0] ST_WRITE      = 3'd3;
    localparam [2:0] ST_WRITE_AFF_GAMMA = 3'd4;
    localparam [2:0] ST_WRITE_AFF_BETA  = 3'd5;
    localparam [2:0] ST_WRITE_PROC      = 3'd6;
    localparam [2:0] ST_DONE            = 3'd7;
    localparam [9:0] EMBED_DIM_10        = EMBED_DIM;
    localparam [15:0] EMBED_DIM_16       = EMBED_DIM;
    localparam [9:0] SCORE_BANK_DEPTH_10 = SCORE_BANK_DEPTH;
    localparam [6:0] M_BLK_LAST_MAX      = M_BLK_NUM - 1;
    localparam integer LN_PARAM_DEPTH    = LN_PARAM_SETS * EMBED_DIM;
    wire [6:0] active_mblk_last = active_mblk_num - 7'd1;

    reg [2:0] state;
    reg [9:0] ch_issue;
    reg [15:0] ln_param_base;
    reg [14:0] ln_epsilon_core;

    reg        rd_valid_d1;
    reg [9:0]  rd_ch_d1;
    reg [6:0]  rd_mblk_d1;
    reg        rd_valid_d2;
    reg [9:0]  rd_ch_d2;
    reg [6:0]  rd_mblk_d2;

    reg signed [31:0] sum_lane   [0:31];
    reg        [31:0] sumsq_lane [0:31];
    reg signed [15:0] mean_lane  [0:31];
    reg        [15:0] recip_lane [0:31];

    // Pipeline each 32-lane input word through input, square, and accumulator
    // stages. This keeps BRAM read, INT8 square, and 32-bit accumulation on
    // separate 200-MHz paths while retaining one 32-lane word per cycle.
    reg               load_word_valid;
    reg               load_stat_valid;
    reg signed [7:0]  load_lane_s8_q      [0:31];
    reg signed [7:0]  load_lane_s8_stat_q [0:31];
    reg        [15:0] load_lane_sq_u16_q  [0:31];
    reg        [31:0] load_word_lane_valid_q;
    reg        [31:0] load_lane_valid_q;

    reg [4:0]  stat_lane;
    reg        stat_issue_done;
    reg        stat_valid_s0;
    reg        stat_valid_s1;
    reg        stat_valid_s2;
    reg        stat_valid_s3;
    reg        stat_valid_s4;
    reg [4:0]  stat_lane_s0;
    reg [4:0]  stat_lane_s1;
    reg [4:0]  stat_lane_s2;
    reg [4:0]  stat_lane_s3;
    reg [4:0]  stat_lane_s4;
    reg signed [17:0] stat_sum_s0;
    reg        [22:0] stat_sumsq_s0;
    reg signed [35:0] stat_mean_product_s1;
    reg        [35:0] stat_sum_square_s1;
    reg        [32:0] stat_sumsq_scaled_s1;
    reg signed [15:0] stat_mean_s2;
    reg signed [36:0] stat_variance_core_s2;
    reg signed [15:0] stat_mean_s3;
    reg        [11:0] stat_variance_u12_s3;
    reg signed [15:0] stat_mean_s4;
    reg        [5:0]  stat_std_code_s4;
    reg [9:0]  proc_ch;
    reg [6:0]  proc_mblk;
    reg [MEM_WIDTH-1:0] proc_word_buf;
    reg [MEM_WIDTH-1:0] aff_gamma_word;
    reg [MEM_WIDTH-1:0] aff_beta_word;
    reg                 aff_read_wait;
    reg                 affine_pass_secondary;

    wire [6:0] aff_gamma_addr =
        (affine_pass_secondary ? 7'd60 : 7'd0) + {1'b0, proc_ch[8:3]};
    wire [6:0] aff_beta_addr =
        (affine_pass_secondary ? 7'd108 : 7'd48) + {2'd0, proc_ch[8:5]};

    // All 32 token lanes enter this four-stage pipeline together. The
    // arithmetic boundaries match the scalar implementation, while one
    // 256-bit channel word now drains in five cycles instead of 36 cycles.
    reg       proc_word_issued;
    reg       proc_valid_s1;
    reg       proc_valid_s2;
    reg       proc_valid_s3;
    reg       proc_valid_s4;
    reg signed [15:0] proc_centered_s1    [0:31];
    reg signed [63:0] proc_a_q_s1         [0:31];
    reg signed [7:0]  proc_bias_s1        [0:31];
    reg signed [15:0] proc_centered_s2    [0:31];
    reg [7:0]         proc_m_u8_s2        [0:31];
    reg [5:0]         proc_n_shift_s2     [0:31];
    reg signed [7:0]  proc_bias_s2        [0:31];
    reg signed [63:0] proc_scaled_prod_s3 [0:31];
    reg [5:0]         proc_n_shift_s3     [0:31];
    reg signed [7:0]  proc_bias_s3        [0:31];
    reg signed [31:0] proc_scaled_x_s4    [0:31];
    reg signed [7:0]  proc_bias_s4        [0:31];

    reg signed [31:0] gain_rom [0:LN_PARAM_DEPTH-1];
    reg signed [7:0]  bias_rom [0:LN_PARAM_DEPTH-1];
    reg        [15:0] recip_lut [0:63];
    reg        [14:0] epsilon_rom [0:LN_PARAM_SETS-1];

    integer i;
    integer norm_i;
    integer init_i;
    reg [MEM_WIDTH-1:0] norm_word;
    wire signed [7:0]  load_lane_s8_w     [0:31];
    wire signed [15:0] load_lane_sq_s16_w [0:31];

    genvar load_lane_gen;
    generate
        for (load_lane_gen = 0; load_lane_gen < 32; load_lane_gen = load_lane_gen + 1) begin : GEN_LOAD_STATS
            assign load_lane_s8_w[load_lane_gen] =
                $signed(gout_rd_data[load_lane_gen*8 +: 8]);
            (* use_dsp = "yes" *)
            assign load_lane_sq_s16_w[load_lane_gen] =
                load_lane_s8_q[load_lane_gen] * load_lane_s8_q[load_lane_gen];
        end
    endgenerate

    initial begin
        for (init_i = 0; init_i < LN_PARAM_DEPTH; init_i = init_i + 1) begin
            gain_rom[init_i] = LN_BASE_GAIN_Q16;
            bias_rom[init_i] = 8'sd0;
        end
        for (init_i = 0; init_i < 64; init_i = init_i + 1) begin
            recip_lut[init_i] = 16'd1;
        end
        for (init_i = 0; init_i < LN_PARAM_SETS; init_i = init_i + 1) begin
            epsilon_rom[init_i] = LN_EPSILON_CORE[14:0];
        end
        if (LN_AFFINE_MODE == 1) begin
            $readmemh(LN_GAIN_MEM, gain_rom);
            $readmemh(LN_BIAS_MEM, bias_rom);
        end
        $readmemh(LN_RECIP_MEM, recip_lut);
        $readmemh(LN_EPSILON_MEM, epsilon_rom);
    end

    function [GOUT_AW-1:0] f_mblk_stride;
        input [6:0] mblk;
        begin
            f_mblk_stride = mblk * OUT_STRIDE;
        end
    endfunction

    function [0:0] f_scratch_bank;
        input [9:0] ch;
        begin
            f_scratch_bank = (ch >= SCORE_BANK_DEPTH);
        end
    endfunction

    function [11:0] f_scratch_addr;
        input [9:0] ch;
        begin
            if (ch >= SCORE_BANK_DEPTH) begin
                f_scratch_addr = {2'd0, (ch - SCORE_BANK_DEPTH_10)};
            end else begin
                f_scratch_addr = {2'd0, ch};
            end
        end
    endfunction

    function signed [15:0] f_mean_q0_wooyoung;
        input signed [31:0] value;
        reg signed [63:0] product;
        reg signed [63:0] abs_product;
        reg signed [63:0] rounded;
        begin
            product = value * 64'sd21845;
            if (product >= 0) begin
                f_mean_q0_wooyoung = ((product + (64'sd1 << 22)) >>> 23);
            end else begin
                abs_product = -product;
                rounded = (abs_product + (64'sd1 << 22)) >>> 23;
                f_mean_q0_wooyoung = -rounded[15:0];
            end
        end
    endfunction

    function [11:0] f_variance_u12_wooyoung;
        input signed [31:0] sum_value;
        input        [31:0] sumsq_value;
        input        [14:0] epsilon_core;
        reg signed [63:0] sum64;
        reg signed [63:0] variance_core;
        reg        [63:0] variance_scaled;
        reg [LN_VAR_SHIFT:0] round_compare;
        begin
            sum64 = {{32{sum_value[31]}}, sum_value};
            variance_core = ($signed({1'b0, sumsq_value}) * EMBED_DIM) - (sum64 * sum64);
            if (variance_core <= 0) begin
                variance_scaled = 64'd0;
            end else begin
                // Exact decomposition of
                // (variance_core + epsilon + 2^(shift-1)) >> shift.
                // Only the low remainder participates in rounding, which avoids
                // a wide epsilon/rounding adder on the critical path.
                variance_scaled = variance_core >>> LN_VAR_SHIFT;
                round_compare =
                    {1'b0, variance_core[LN_VAR_SHIFT-1:0]} +
                    {{(LN_VAR_SHIFT+1-15){1'b0}}, epsilon_core};
                if (round_compare >= ({{LN_VAR_SHIFT{1'b0}}, 1'b1} << (LN_VAR_SHIFT - 1))) begin
                    variance_scaled = variance_scaled + 64'd1;
                end
            end

            if (variance_scaled == 0) begin
                f_variance_u12_wooyoung = 12'd1;
            end else if (variance_scaled > 64'd4095) begin
                f_variance_u12_wooyoung = 12'd4095;
            end else begin
                f_variance_u12_wooyoung = variance_scaled[11:0];
            end
        end
    endfunction

    function signed [15:0] f_mean_from_product;
        input signed [35:0] product;
        reg signed [35:0] abs_product;
        reg signed [35:0] rounded;
        begin
            if (product >= 0) begin
                f_mean_from_product = ((product + (36'sd1 << 22)) >>> 23);
            end else begin
                abs_product = -product;
                rounded = (abs_product + (36'sd1 << 22)) >>> 23;
                f_mean_from_product = -rounded[15:0];
            end
        end
    endfunction

    function [11:0] f_variance_from_core;
        input signed [36:0] variance_core;
        input        [14:0] epsilon_core;
        reg        [36:0] variance_scaled;
        reg [LN_VAR_SHIFT:0] round_compare;
        begin
            if (variance_core <= 0) begin
                variance_scaled = 37'd0;
            end else begin
                variance_scaled = variance_core >>> LN_VAR_SHIFT;
                round_compare =
                    {1'b0, variance_core[LN_VAR_SHIFT-1:0]} +
                    {{(LN_VAR_SHIFT+1-15){1'b0}}, epsilon_core};
                if (round_compare >= ({{LN_VAR_SHIFT{1'b0}}, 1'b1} << (LN_VAR_SHIFT - 1))) begin
                    variance_scaled = variance_scaled + 37'd1;
                end
            end

            if (variance_scaled == 0) begin
                f_variance_from_core = 12'd1;
            end else if (variance_scaled > 37'd4095) begin
                f_variance_from_core = 12'd4095;
            end else begin
                f_variance_from_core = variance_scaled[11:0];
            end
        end
    endfunction

    function [3:0] f_log2_u12;
        input [11:0] value;
        integer b;
        begin
            f_log2_u12 = 4'd0;
            for (b = 0; b < 12; b = b + 1) begin
                if (value[b]) begin
                    f_log2_u12 = b[3:0];
                end
            end
        end
    endfunction

    function [5:0] f_log2_u64;
        input [63:0] value;
        integer b;
        begin
            f_log2_u64 = 6'd0;
            for (b = 0; b < 64; b = b + 1) begin
                if (value[b]) begin
                    f_log2_u64 = b[5:0];
                end
            end
        end
    endfunction

    function [5:0] f_std_code_wooyoung;
        input [11:0] variance_u12;
        reg [3:0] variance_log2;
        reg [3:0] half_exponent;
        reg [11:0] sqrt_base;
        reg [11:0] sqrt_linear_term;
        reg [12:0] std_tmp;
        begin
            variance_log2 = f_log2_u12(variance_u12);
            half_exponent = (variance_log2 + 4'd1) >> 1;
            sqrt_base = (half_exponent > 0) ? (12'd1 << (half_exponent - 1)) : 12'd0;
            sqrt_linear_term = variance_u12 >> (half_exponent + 1);
            std_tmp = {1'b0, sqrt_base} + {1'b0, sqrt_linear_term};
            if (std_tmp == 0) begin
                f_std_code_wooyoung = 6'd1;
            end else if (std_tmp > 13'd63) begin
                f_std_code_wooyoung = 6'd63;
            end else begin
                f_std_code_wooyoung = std_tmp[5:0];
            end
        end
    endfunction

    function [7:0] f_sat8_u;
        input signed [31:0] value;
        begin
            if (value > 32'sd127) begin
                f_sat8_u = 8'h7F;
            end else if (value < -32'sd128) begin
                f_sat8_u = 8'h80;
            end else begin
                f_sat8_u = value[7:0];
            end
        end
    endfunction

    function [7:0] f_norm_byte_wooyoung;
        input [7:0]         value_u8;
        input signed [15:0] mean_value;
        input [15:0]        recip_value;
        input signed [31:0] gain_value;
        input signed [7:0]  bias_value;
        reg signed [15:0] centered;
        reg signed [63:0] a_q_signed;
        reg [63:0] a_work;
        reg [5:0] a_log2;
        reg [5:0] n_shift;
        reg [5:0] top_byte_shift;
        reg signed [63:0] top_byte_signed;
        reg [7:0] m_u8;
        reg signed [63:0] scaled_prod;
        reg signed [31:0] scaled_x;
        reg signed [31:0] with_bias;
        begin
            centered = {{8{value_u8[7]}}, value_u8} - mean_value;
            a_q_signed = $signed({1'b0, recip_value}) * gain_value;
            a_work = (a_q_signed <= 0) ? 64'd1 : a_q_signed[63:0];
            a_log2 = f_log2_u64(a_work);

            if ((23 + LN_GAIN_FRAC_BITS) <= a_log2) begin
                n_shift = 6'd0;
            end else if ((23 + LN_GAIN_FRAC_BITS - a_log2) > 63) begin
                n_shift = 6'd63;
            end else begin
                n_shift = 23 + LN_GAIN_FRAC_BITS - a_log2;
            end

            top_byte_shift = (a_log2 > 6'd7) ? (a_log2 - 6'd7) : 6'd0;
            top_byte_signed = a_q_signed >>> top_byte_shift;
            if (a_log2 > (LN_GAIN_FRAC_BITS + 7)) begin
                if (top_byte_signed < 0) begin
                    m_u8 = 8'd0;
                end else if (top_byte_signed > 64'sd255) begin
                    m_u8 = 8'd255;
                end else begin
                    m_u8 = top_byte_signed[7:0];
                end
            end else begin
                m_u8 = 8'd255;
            end

            scaled_prod = $signed({1'b0, m_u8}) * centered;
            scaled_x = (n_shift >= 6'd63) ? {32{scaled_prod[63]}} : (scaled_prod >>> n_shift);
            with_bias = scaled_x + {{24{bias_value[7]}}, bias_value};
            f_norm_byte_wooyoung = f_sat8_u(with_bias);
        end
    endfunction

    function [5:0] f_norm_n_shift;
        input [5:0] a_log2;
        integer shift_delta;
        begin
            if ((23 + LN_GAIN_FRAC_BITS) <= a_log2) begin
                f_norm_n_shift = 6'd0;
            end else begin
                shift_delta = 23 + LN_GAIN_FRAC_BITS - a_log2;
                if (shift_delta > 63) begin
                    f_norm_n_shift = 6'd63;
                end else begin
                    f_norm_n_shift = shift_delta[5:0];
                end
            end
        end
    endfunction

    function [7:0] f_norm_m_u8;
        input signed [63:0] a_q_signed;
        input        [5:0]  a_log2;
        reg [5:0] top_byte_shift;
        reg signed [63:0] top_byte_signed;
        begin
            top_byte_shift = (a_log2 > 6'd7) ? (a_log2 - 6'd7) : 6'd0;
            top_byte_signed = a_q_signed >>> top_byte_shift;
            if (a_log2 > (LN_GAIN_FRAC_BITS + 7)) begin
                if (top_byte_signed < 0) begin
                    f_norm_m_u8 = 8'd0;
                end else if (top_byte_signed > 64'sd255) begin
                    f_norm_m_u8 = 8'd255;
                end else begin
                    f_norm_m_u8 = top_byte_signed[7:0];
                end
            end else begin
                f_norm_m_u8 = 8'd255;
            end
        end
    endfunction

    wire issue_valid = (ch_issue < EMBED_DIM_10);
    wire load_done   = (!issue_valid) && !rd_valid_d1 && !rd_valid_d2 &&
                       !load_word_valid && !load_stat_valid;
    wire last_mblk   = (dbg_mblk == active_mblk_last);

    // Row-statistics pipeline. The accumulated ranges are bounded by 384
    // signed INT8 channels, so 18-bit sum and 23-bit sum-of-squares operands
    // retain the exact values while keeping sum^2 inside one 18x18 DSP.
    wire signed [35:0] stat_mean_product_calc = stat_sum_s0 * 18'sd21845;
    wire        [35:0] stat_sum_square_calc = stat_sum_s0 * stat_sum_s0;
    wire        [32:0] stat_sumsq_scaled_calc =
        ({10'd0, stat_sumsq_s0} << 8) + ({10'd0, stat_sumsq_s0} << 7);
    wire signed [36:0] stat_variance_core_calc =
        $signed({4'd0, stat_sumsq_scaled_s1}) -
        $signed({1'b0, stat_sum_square_s1});
    wire signed [15:0] stat_mean_pipe_calc =
        f_mean_from_product(stat_mean_product_s1);
    wire [11:0] stat_variance_u12_pipe_calc =
        f_variance_from_core(stat_variance_core_s2, ln_epsilon_core);
    wire [5:0] stat_std_code_pipe_calc =
        f_std_code_wooyoung(stat_variance_u12_s3);
    wire signed [31:0] proc_gain_q16 =
        (LN_AFFINE_MODE == 2) ? $signed(aff_gamma_word[{proc_ch[2:0], 5'b0} +: 32]) :
        ((LN_AFFINE_MODE == 1) ? gain_rom[ln_param_base + proc_ch] : LN_BASE_GAIN_Q16);
    wire signed [7:0] proc_bias_i8 =
        (LN_AFFINE_MODE == 2) ? $signed(aff_beta_word[{proc_ch[4:0], 3'b0} +: 8]) :
        ((LN_AFFINE_MODE == 1) ? bias_rom[ln_param_base + proc_ch] : 8'sd0);
    wire [63:0] proc_a_work_s1 [0:31];
    wire [5:0] proc_a_log2_s1 [0:31];
    wire [5:0] proc_n_shift_calc_s1 [0:31];
    wire [7:0] proc_m_u8_calc_s1 [0:31];
    wire signed [31:0] proc_with_bias_s4 [0:31];
    wire [7:0] proc_norm_byte_s4 [0:31];

    genvar proc_lane;
    generate
        for (proc_lane = 0; proc_lane < 32; proc_lane = proc_lane + 1) begin : GEN_PROC_LANE
            assign proc_a_work_s1[proc_lane] =
                (proc_a_q_s1[proc_lane] <= 0) ? 64'd1 : proc_a_q_s1[proc_lane][63:0];
            assign proc_a_log2_s1[proc_lane] = f_log2_u64(proc_a_work_s1[proc_lane]);
            assign proc_n_shift_calc_s1[proc_lane] = f_norm_n_shift(proc_a_log2_s1[proc_lane]);
            assign proc_m_u8_calc_s1[proc_lane] =
                f_norm_m_u8(proc_a_q_s1[proc_lane], proc_a_log2_s1[proc_lane]);
            assign proc_with_bias_s4[proc_lane] =
                proc_scaled_x_s4[proc_lane] +
                {{24{proc_bias_s4[proc_lane][7]}}, proc_bias_s4[proc_lane]};
            assign proc_norm_byte_s4[proc_lane] = f_sat8_u(proc_with_bias_s4[proc_lane]);
        end
    endgenerate

    always @* begin
        norm_word = proc_word_buf;
        for (norm_i = 0; norm_i < 32; norm_i = norm_i + 1) begin
            if ((proc_mblk != active_mblk_last) || (norm_i <= active_last_valid_lane)) begin
                norm_word[norm_i*8 +: 8] = proc_norm_byte_s4[norm_i];
            end
        end
    end

    task automatic t_reset_row_stats;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                sum_lane[i]   <= 32'sd0;
                sumsq_lane[i] <= 32'd0;
                mean_lane[i]  <= 16'sd0;
                recip_lane[i] <= 16'd0;
            end
        end
    endtask

    task automatic t_advance_after_word_write;
        begin
            if (proc_ch == (EMBED_DIM_10 - 1)) begin
                if (last_mblk) begin
                    state <= ST_DONE;
                end else begin
                    dbg_mblk  <= dbg_mblk + 7'd1;
                    ch_issue  <= 10'd0;
                    stat_lane <= 5'd0;
                    state     <= ST_LOAD;
                    t_reset_row_stats();
                end
            end else begin
                ch_issue <= proc_ch + 10'd1;
                state    <= ST_WRITE;
            end
        end
    endtask

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            state <= ST_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;

            gout_rd_en    <= 1'b0;
            gout_rd_addr  <= {GOUT_AW{1'b0}};
            gout_wr_valid <= 1'b0;
            gout_wr_addr  <= {GOUT_AW{1'b0}};
            gout_wr_data  <= {MEM_WIDTH{1'b0}};
            gout_wr_bank  <= 1'b1;

            score_rd_en    <= 1'b0;
            score_rd_bank  <= 1'b0;
            score_rd_addr  <= 12'd0;
            score_wr_valid <= 1'b0;
            score_wr_bank  <= 1'b0;
            score_wr_addr  <= 12'd0;
            score_wr_data  <= {MEM_WIDTH{1'b0}};
            aff_rd_en      <= 1'b0;
            aff_rd_addr    <= 7'd0;

            dbg_mblk  <= 7'd0;
            dbg_state <= ST_IDLE;
            ch_issue  <= 10'd0;
            ln_param_base <= 16'd0;
            ln_epsilon_core <= LN_EPSILON_CORE[14:0];

            rd_valid_d1 <= 1'b0;
            rd_ch_d1    <= 10'd0;
            rd_mblk_d1  <= 7'd0;
            rd_valid_d2 <= 1'b0;
            rd_ch_d2    <= 10'd0;
            rd_mblk_d2  <= 7'd0;

            load_word_valid <= 1'b0;
            load_stat_valid <= 1'b0;
            load_word_lane_valid_q <= 32'd0;
            load_lane_valid_q <= 32'd0;

            stat_lane    <= 5'd0;
            stat_issue_done <= 1'b0;
            stat_valid_s0 <= 1'b0;
            stat_valid_s1 <= 1'b0;
            stat_valid_s2 <= 1'b0;
            stat_valid_s3 <= 1'b0;
            stat_valid_s4 <= 1'b0;
            stat_lane_s0 <= 5'd0;
            stat_lane_s1 <= 5'd0;
            stat_lane_s2 <= 5'd0;
            stat_lane_s3 <= 5'd0;
            stat_lane_s4 <= 5'd0;
            stat_sum_s0 <= 18'sd0;
            stat_sumsq_s0 <= 23'd0;
            stat_mean_product_s1 <= 36'sd0;
            stat_sum_square_s1 <= 36'd0;
            stat_sumsq_scaled_s1 <= 33'd0;
            stat_mean_s2 <= 16'sd0;
            stat_variance_core_s2 <= 37'sd0;
            stat_mean_s3 <= 16'sd0;
            stat_variance_u12_s3 <= 12'd1;
            stat_mean_s4 <= 16'sd0;
            stat_std_code_s4 <= 6'd1;
            proc_ch      <= 10'd0;
            proc_mblk    <= 7'd0;
            proc_word_buf <= {MEM_WIDTH{1'b0}};
            aff_gamma_word <= {MEM_WIDTH{1'b0}};
            aff_beta_word  <= {MEM_WIDTH{1'b0}};
            aff_read_wait  <= 1'b0;
            affine_pass_secondary <= 1'b0;
            proc_word_issued    <= 1'b0;
            proc_valid_s1       <= 1'b0;
            proc_valid_s2       <= 1'b0;
            proc_valid_s3       <= 1'b0;
            proc_valid_s4       <= 1'b0;

            for (i = 0; i < 32; i = i + 1) begin
                sum_lane[i]   <= 32'sd0;
                sumsq_lane[i] <= 32'd0;
                mean_lane[i]  <= 16'sd0;
                recip_lane[i] <= 16'd0;
                proc_centered_s1[i]    <= 16'sd0;
                proc_a_q_s1[i]         <= 64'sd0;
                proc_bias_s1[i]        <= 8'sd0;
                proc_centered_s2[i]    <= 16'sd0;
                proc_m_u8_s2[i]        <= 8'd0;
                proc_n_shift_s2[i]     <= 6'd0;
                proc_bias_s2[i]        <= 8'sd0;
                proc_scaled_prod_s3[i] <= 64'sd0;
                proc_n_shift_s3[i]     <= 6'd0;
                proc_bias_s3[i]        <= 8'sd0;
                proc_scaled_x_s4[i]    <= 32'sd0;
                proc_bias_s4[i]        <= 8'sd0;
            end
        end else begin
            done <= 1'b0;

            gout_rd_en     <= 1'b0;
            gout_wr_valid  <= 1'b0;
            score_rd_en    <= 1'b0;
            score_wr_valid <= 1'b0;
            aff_rd_en      <= 1'b0;
            dbg_state      <= state;

            load_word_valid <= 1'b0;
            load_stat_valid <= load_word_valid;

            if (load_word_valid) begin
                for (i = 0; i < 32; i = i + 1) begin
                    load_lane_s8_stat_q[i] <= load_lane_s8_q[i];
                    load_lane_sq_u16_q[i] <= load_lane_sq_s16_w[i];
                    load_lane_valid_q[i] <= load_word_lane_valid_q[i];
                end
            end

            rd_valid_d2 <= rd_valid_d1;
            rd_ch_d2    <= rd_ch_d1;
            rd_mblk_d2  <= rd_mblk_d1;

            case (state)
                ST_IDLE: begin
                    rd_valid_d1 <= 1'b0;
                    rd_valid_d2 <= 1'b0;
                    load_word_valid <= 1'b0;
                    load_stat_valid <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        affine_pass_secondary <= 1'b0;
                        dbg_mblk  <= 7'd0;
                        ch_issue  <= 10'd0;
                        stat_lane <= 5'd0;
                        stat_issue_done <= 1'b0;
                        stat_valid_s0 <= 1'b0;
                        stat_valid_s1 <= 1'b0;
                        stat_valid_s2 <= 1'b0;
                        stat_valid_s3 <= 1'b0;
                        stat_valid_s4 <= 1'b0;
                        ln_param_base <= {10'd0, ln_param_sel} * EMBED_DIM_16;
                        if (ln_param_sel < LN_PARAM_SETS) begin
                            ln_epsilon_core <= epsilon_rom[ln_param_sel];
                        end else begin
                            ln_epsilon_core <= LN_EPSILON_CORE[14:0];
                        end
                        state     <= ST_LOAD;
                        t_reset_row_stats();
                    end
                end

                ST_LOAD: begin
                    if (load_stat_valid) begin
                        for (i = 0; i < 32; i = i + 1) begin
                            if (load_lane_valid_q[i]) begin
                                sum_lane[i] <= sum_lane[i] +
                                    {{24{load_lane_s8_stat_q[i][7]}}, load_lane_s8_stat_q[i]};
                                sumsq_lane[i] <= sumsq_lane[i] +
                                    {16'd0, load_lane_sq_u16_q[i]};
                            end
                        end
                    end

                    if (rd_valid_d2) begin
                        score_wr_valid <= 1'b1;
                        score_wr_bank  <= f_scratch_bank(rd_ch_d2);
                        score_wr_addr  <= f_scratch_addr(rd_ch_d2);
                        score_wr_data  <= gout_rd_data;

                        load_word_valid <= 1'b1;

                        for (i = 0; i < 32; i = i + 1) begin
                            load_lane_s8_q[i] <= load_lane_s8_w[i];
                            load_word_lane_valid_q[i] <=
                                (rd_mblk_d2 != active_mblk_last) ||
                                (i <= active_last_valid_lane);
                        end
                    end

                    if (issue_valid) begin
                        gout_rd_en   <= 1'b1;
                        gout_rd_addr <= f_mblk_stride(dbg_mblk) + {2'd0, ch_issue};
                        rd_valid_d1  <= 1'b1;
                        rd_ch_d1     <= ch_issue;
                        rd_mblk_d1   <= dbg_mblk;
                        ch_issue     <= ch_issue + 10'd1;
                    end else begin
                        rd_valid_d1 <= 1'b0;
                    end

                    if (load_done) begin
                        ch_issue <= 10'd0;
                        if (APPLY_LAYER_NORM != 0) begin
                            stat_lane <= 5'd0;
                            stat_issue_done <= 1'b0;
                            stat_valid_s0 <= 1'b0;
                            stat_valid_s1 <= 1'b0;
                            stat_valid_s2 <= 1'b0;
                            stat_valid_s3 <= 1'b0;
                            stat_valid_s4 <= 1'b0;
                            state     <= ST_STAT;
                        end else begin
                            state <= ST_WRITE;
                        end
                    end
                end

                ST_STAT: begin
                    rd_valid_d1 <= 1'b0;
                    rd_valid_d2 <= 1'b0;

                    stat_valid_s4 <= stat_valid_s3;
                    stat_valid_s3 <= stat_valid_s2;
                    stat_valid_s2 <= stat_valid_s1;
                    stat_valid_s1 <= stat_valid_s0;
                    stat_valid_s0 <= 1'b0;

                    if (!stat_issue_done) begin
                        stat_sum_s0   <= sum_lane[stat_lane][17:0];
                        stat_sumsq_s0 <= sumsq_lane[stat_lane][22:0];
                        stat_lane_s0  <= stat_lane;
                        stat_valid_s0 <= 1'b1;
                        if (stat_lane == 5'd31) begin
                            stat_issue_done <= 1'b1;
                        end else begin
                            stat_lane <= stat_lane + 5'd1;
                        end
                    end

                    if (stat_valid_s0) begin
                        stat_mean_product_s1 <= stat_mean_product_calc;
                        stat_sum_square_s1   <= stat_sum_square_calc;
                        stat_sumsq_scaled_s1 <= stat_sumsq_scaled_calc;
                        stat_lane_s1         <= stat_lane_s0;
                    end

                    if (stat_valid_s1) begin
                        stat_mean_s2          <= stat_mean_pipe_calc;
                        stat_variance_core_s2 <= stat_variance_core_calc;
                        stat_lane_s2          <= stat_lane_s1;
                    end

                    if (stat_valid_s2) begin
                        stat_mean_s3         <= stat_mean_s2;
                        stat_variance_u12_s3 <= stat_variance_u12_pipe_calc;
                        stat_lane_s3         <= stat_lane_s2;
                    end

                    if (stat_valid_s3) begin
                        stat_mean_s4     <= stat_mean_s3;
                        stat_std_code_s4 <= stat_std_code_pipe_calc;
                        stat_lane_s4     <= stat_lane_s3;
                    end

                    if (stat_valid_s4) begin
                        mean_lane[stat_lane_s4]  <= stat_mean_s4;
                        recip_lane[stat_lane_s4] <= recip_lut[stat_std_code_s4 - 6'd1];
                        if (stat_lane_s4 == 5'd31) begin
                            ch_issue <= 10'd0;
                            state    <= ST_WRITE;
                        end
                    end
                end

                ST_WRITE: begin
                    if (rd_valid_d2) begin
                        rd_valid_d1 <= 1'b0;
                        rd_valid_d2 <= 1'b0;
                        if (APPLY_LAYER_NORM != 0) begin
                            proc_word_buf <= score_rd_data;
                            proc_ch       <= rd_ch_d2;
                            proc_mblk     <= rd_mblk_d2;
                            proc_word_issued <= 1'b0;
                            proc_valid_s1 <= 1'b0;
                            proc_valid_s2 <= 1'b0;
                            proc_valid_s3 <= 1'b0;
                            proc_valid_s4 <= 1'b0;
                            if (LN_AFFINE_MODE == 2) begin
                                aff_rd_en   <= 1'b1;
                                aff_rd_addr <= rd_ch_d2[8:3];
                                aff_read_wait <= 1'b0;
                                state       <= ST_WRITE_AFF_GAMMA;
                            end else begin
                                state       <= ST_WRITE_PROC;
                            end
                        end else begin
                            gout_wr_valid <= 1'b1;
                            gout_wr_addr  <= f_mblk_stride(rd_mblk_d2) + {2'd0, rd_ch_d2};
                            gout_wr_data  <= score_rd_data;
                            gout_wr_bank  <= primary_out_bank;
                            proc_ch       <= rd_ch_d2;
                            proc_mblk     <= rd_mblk_d2;
                            t_advance_after_word_write();
                        end
                    end else if (!rd_valid_d1 && issue_valid) begin
                        score_rd_en   <= 1'b1;
                        score_rd_bank <= f_scratch_bank(ch_issue);
                        score_rd_addr <= f_scratch_addr(ch_issue);
                        rd_valid_d1   <= 1'b1;
                        rd_ch_d1      <= ch_issue;
                        rd_mblk_d1    <= dbg_mblk;
                    end else begin
                        rd_valid_d1 <= 1'b0;
                    end
                end

                ST_WRITE_AFF_GAMMA: begin
                    rd_valid_d1 <= 1'b0;
                    rd_valid_d2 <= 1'b0;
                    if (!aff_read_wait) begin
                        // The full accelerator supplies affine words from a
                        // synchronous 1-cycle RAM. Hold the request before
                        // sampling its registered output.
                        aff_rd_en      <= 1'b1;
                        aff_rd_addr    <= aff_gamma_addr;
                        aff_read_wait  <= 1'b1;
                    end else begin
                        aff_gamma_word <= aff_rd_data;
                        aff_rd_en      <= 1'b1;
                        aff_rd_addr    <= aff_beta_addr;
                        aff_read_wait  <= 1'b0;
                        state          <= ST_WRITE_AFF_BETA;
                    end
                end

                ST_WRITE_AFF_BETA: begin
                    rd_valid_d1 <= 1'b0;
                    rd_valid_d2 <= 1'b0;
                    if (!aff_read_wait) begin
                        aff_rd_en      <= 1'b1;
                        aff_rd_addr    <= aff_beta_addr;
                        aff_read_wait  <= 1'b1;
                    end else begin
                        aff_beta_word <= aff_rd_data;
                        aff_read_wait <= 1'b0;
                        state         <= ST_WRITE_PROC;
                    end
                end

                ST_WRITE_PROC: begin
                    rd_valid_d1 <= 1'b0;
                    rd_valid_d2 <= 1'b0;

                    // Shift one 32-lane vector through the four arithmetic stages.
                    proc_valid_s4 <= proc_valid_s3;
                    proc_valid_s3 <= proc_valid_s2;
                    proc_valid_s2 <= proc_valid_s1;
                    proc_valid_s1 <= 1'b0;

                    // Stage 1: issue all byte lanes from one channel word.
                    if (!proc_word_issued) begin
                        for (i = 0; i < 32; i = i + 1) begin
                            proc_centered_s1[i] <=
                                $signed({{8{proc_word_buf[i*8 + 7]}}, proc_word_buf[i*8 +: 8]}) -
                                mean_lane[i];
                            proc_a_q_s1[i] <=
                                $signed({1'b0, recip_lane[i]}) * proc_gain_q16;
                            proc_bias_s1[i] <= proc_bias_i8;
                        end
                        proc_valid_s1    <= 1'b1;
                        proc_word_issued <= 1'b1;
                    end

                    // Stage 2: decode the integer multiplier and shift.
                    if (proc_valid_s1) begin
                        for (i = 0; i < 32; i = i + 1) begin
                            proc_centered_s2[i] <= proc_centered_s1[i];
                            proc_m_u8_s2[i]     <= proc_m_u8_calc_s1[i];
                            proc_n_shift_s2[i]  <= proc_n_shift_calc_s1[i];
                            proc_bias_s2[i]     <= proc_bias_s1[i];
                        end
                    end

                    // Stage 3: multiply the centered value.
                    if (proc_valid_s2) begin
                        for (i = 0; i < 32; i = i + 1) begin
                            proc_scaled_prod_s3[i] <=
                                $signed({1'b0, proc_m_u8_s2[i]}) * proc_centered_s2[i];
                            proc_n_shift_s3[i] <= proc_n_shift_s2[i];
                            proc_bias_s3[i]    <= proc_bias_s2[i];
                        end
                    end

                    // Stage 4: apply the variable shifts in parallel.
                    if (proc_valid_s3) begin
                        for (i = 0; i < 32; i = i + 1) begin
                            proc_scaled_x_s4[i] <=
                                (proc_n_shift_s3[i] >= 6'd63) ?
                                {32{proc_scaled_prod_s3[i][63]}} :
                                (proc_scaled_prod_s3[i] >>> proc_n_shift_s3[i]);
                            proc_bias_s4[i] <= proc_bias_s3[i];
                        end
                    end

                    // Commit one complete 256-bit word after the pipeline drains.
                    if (proc_valid_s4) begin
                        gout_wr_valid <= 1'b1;
                        gout_wr_addr  <= f_mblk_stride(proc_mblk) + {2'd0, proc_ch};
                        gout_wr_data  <= norm_word;
                        gout_wr_bank  <= affine_pass_secondary ?
                                         secondary_out_bank : primary_out_bank;
                        if (dual_affine && !affine_pass_secondary &&
                            (LN_AFFINE_MODE == 2)) begin
                            // Preserve proc_word_buf and the row statistics.
                            // Re-run only the affine pipeline with set 1.
                            affine_pass_secondary <= 1'b1;
                            proc_word_issued <= 1'b0;
                            proc_valid_s1 <= 1'b0;
                            proc_valid_s2 <= 1'b0;
                            proc_valid_s3 <= 1'b0;
                            proc_valid_s4 <= 1'b0;
                            aff_rd_en <= 1'b1;
                            aff_rd_addr <= 7'd60 + {1'b0, proc_ch[8:3]};
                            aff_read_wait <= 1'b0;
                            state <= ST_WRITE_AFF_GAMMA;
                        end else begin
                            affine_pass_secondary <= 1'b0;
                            t_advance_after_word_write();
                        end
                    end
                end

                ST_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                    busy  <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
