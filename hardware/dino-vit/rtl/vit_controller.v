`timescale 1ns/1ps
`default_nettype none

module vit_controller #(
    parameter integer M_TOTAL   = 1873,
    parameter integer N_TOTAL   = 384,
    parameter integer K_TOTAL   = 589,

    parameter integer M_BLK_NUM = 59,
    parameter integer N_BLK_NUM = 12,

    parameter integer BASE_AW   = 27,

    parameter integer RUN_QKV   = 6,
    parameter integer RUN_ATTN_HEADS = 6,
    parameter integer RUN_ENC_BLK_NUM = 12,
    parameter integer RUN_CLASSIFIER  = 0,

    // Debug-only start override. Defaults preserve normal PATCH-first execution.
    parameter integer DEBUG_START_LAYER = 0,
    parameter integer DEBUG_START_ENC_BLK = 0
)(
    input  wire clk,
    input  wire nRst,

    input  wire start,
    output reg  done,

    // from read_controller
    input  wire        bias_ready,
    input  wire quant_ready,
    input  wire [M_BLK_NUM-1:0]  in_ready_cur,
    input  wire [11:0]  w_ready_cur,
    input  wire [11:0]  active_token_total,
    input  wire [6:0]   active_mblk_num,

    // from read_controller: next-layer W0 prefetch completion (latched valid)
    input  wire        nl_pf_valid,
    input  wire [3:0]  nl_pf_layer,
    input  wire [5:0]  nl_pf_ktile,
    input  wire [6:0]  nl_pf_nblk,
    input  wire        nl_pf_bank,

    // layer-end synchronization (controller_top -> vit_controller)
    // - layer_end_pulse: asserted when this layer's *compute* completes
    // - layer_adv_grant: controller_top asserts when DDR dump for this layer is finished
    output reg         layer_end_pulse,
    input  wire        layer_adv_grant,

    // ---------- layer broadcast ----------
    output reg  [3:0]  layer_id,
    output reg  [5:0]  layer_final_ktile,
    output reg         layer_inbuf_sel,
    output reg  [2:0]  layer_head_idx,

    // encoder block index (0..ENC_BLK_NUM-1)
    output reg  [3:0]  enc_blk_idx,

    output reg         layer_wgt_phase,
    // to read_controller
    output reg         layer_active,
    output reg         layer_start_pulse,
    output reg  [5:0]  ktile_idx_cur,
    output wire [6:0]  nblk_idx_cur_o,
    output wire [6:0]  mblk_idx_cur_o,
    output wire [2:0]  dbg_fsm,
    output reg  [M_BLK_NUM-1:0]  mblk_release_pulse,
    output reg  [1:0]  wgt_slot_release_pulse,


    // next-layer W0 prefetch request (e.g., PATCH -> QKV_Q)
    output reg         nl_wgt0_prefetch_pulse,
    output reg  [3:0]  nl_wgt0_layer_id,
    output reg  [5:0]  nl_wgt0_ktile,
    output reg  [6:0]  nl_wgt0_nblk,
    output reg         nl_wgt0_bank,
    output reg  [BASE_AW-1:0] nl_wgt0_base_w_word,

    // to PE
    input  wire        pe_ready,
    output reg         pe_start,
    output reg  [11:0] pe_klen,
    output reg  [5:0]  pe_ktile,
    output reg  [6:0]  pe_nblk,
    output reg  [6:0]  pe_mblk,
    output reg         pe_wbuf_sel,
    output reg         pe_inbuf_sel,
    input  wire        pe_done,

    // big base (word index) from top
    input  wire [BASE_AW-1:0] base_in_word,
    input  wire [BASE_AW-1:0] base_w_word,
    input  wire [BASE_AW-1:0] base_b_word,
    input  wire [BASE_AW-1:0] base_out_word,

    // per-layer base (word index) to read_controller
    output reg  [BASE_AW-1:0] rd_base_in_word,
    output reg  [BASE_AW-1:0] rd_base_w_word,
    output reg  [BASE_AW-1:0] rd_base_b_word,
    output reg  [BASE_AW-1:0] rd_base_out_word
);

        `undef ADDR_MAP_VH
        `include "addr_map.vh"

    wire [6:0] active_mblk_last = active_mblk_num - 7'd1;

// ---------------- layer IDs ----------------
    localparam [3:0] L_PATCH  = 4'd0;
    localparam [3:0] L_QKV_Q  = 4'd1;
    localparam [3:0] L_QKV_V  = 4'd2;
    localparam [3:0] L_QKV_K  = 4'd3;
    localparam [3:0] L_SCORE  = 4'd4;
    localparam [3:0] L_AV     = 4'd5;
    localparam [3:0] L_O_PROJ = 4'd6;
    localparam [3:0] L_MLP_FC1 = 4'd7;
    localparam [3:0] L_MLP_FC2 = 4'd8;


localparam [3:0] L_CLS0    = 4'd9;
localparam [3:0] L_CLS1    = 4'd10;
localparam [3:0] L_CLS2    = 4'd11;
localparam [3:0] L_CLS3    = 4'd12;
localparam [3:0] L_CLS4    = 4'd13;
localparam [3:0] L_CLS5    = 4'd14;
// ---------------- DDR layout (bring-up constants) ----------------
    localparam integer KSTRIDE_PATCH = STRIDE_PATCH_K;
    localparam integer KSTRIDE_Q     = 384;

    localparam integer WWORDS_PATCH  = (N_BLK_NUM * KSTRIDE_PATCH); // 6*769
    localparam integer WWORDS_Q      = (N_BLK_NUM * KSTRIDE_Q);     // 12*384

    localparam integer BWORDS_PER_LAYER  = 24;
    localparam integer BWORDS_CLS_GROUP  = 12;
    // Bias layout in parameter256.txt:
    //   [PATCH (24 words)] then for each encoder block:
    //     Q(24) V(24) K(24) O(24) FC1(96) FC2(24) = 216 words
    localparam integer BWORDS_FC1        = 96;
    localparam integer BWORDS_FC2        = 24;
    // Per-encoder-block bias footprint (EXCLUDES PATCH bias)
    localparam integer BWORDS_PER_BLOCK  = (4*BWORDS_PER_LAYER + BWORDS_FC1 + BWORDS_FC2);

    function [BASE_AW-1:0] f_w_off_word;
    input [3:0] layer;
    input [3:0] blk;
    reg [BASE_AW-1:0] blk_stride;
    begin
        // blk_stride = blk * SIZE_W_ENC_BLK_WGT (from addr_map.vh)
        blk_stride = blk * SIZE_W_ENC_BLK_WGT;

        case (layer)
            L_PATCH:   f_w_off_word = OFF_W_PATCH_WGT;

            // per-encoder-block weights
            L_QKV_Q:   f_w_off_word = OFF_W_Q_WGT   + blk_stride;
            L_QKV_V:   f_w_off_word = OFF_W_V_WGT   + blk_stride;
            L_QKV_K:   f_w_off_word = OFF_W_K_WGT   + blk_stride;
            L_O_PROJ:  f_w_off_word = OFF_W_O_WGT   + blk_stride;
            L_MLP_FC1: f_w_off_word = OFF_W_FC1_WGT + blk_stride;
            L_MLP_FC2: f_w_off_word = OFF_W_FC2_WGT + blk_stride;

            // Q/V outputs stored per block (for debug/dump)
            L_SCORE:   f_w_off_word = OFF_W_Q_OUT + blk_stride;
            L_AV   :   f_w_off_word = OFF_W_V_OUT + blk_stride;

            L_CLS0:   f_w_off_word = OFF_W_CLS0_WGT;
            L_CLS1:   f_w_off_word = OFF_W_CLS1_WGT;
            L_CLS2:   f_w_off_word = OFF_W_CLS2_WGT;
            L_CLS3:   f_w_off_word = OFF_W_CLS3_WGT;
            L_CLS4:   f_w_off_word = OFF_W_CLS4_WGT;
            L_CLS5:   f_w_off_word = OFF_W_CLS5_WGT;


            default:   f_w_off_word = OFF_W_PATCH_WGT;
        endcase
    end
endfunction


    function [BASE_AW-1:0] f_b_off_word;
    input [3:0] layer;
    begin
        case (layer)
            L_PATCH:    f_b_off_word = 0;
            L_QKV_Q:    f_b_off_word = (1*BWORDS_PER_LAYER);
            L_QKV_V:    f_b_off_word = (2*BWORDS_PER_LAYER);
            L_QKV_K:    f_b_off_word = (3*BWORDS_PER_LAYER);
            L_SCORE:    f_b_off_word = 0;
            L_AV   :    f_b_off_word = 0;
            L_O_PROJ:   f_b_off_word = (4*BWORDS_PER_LAYER);
            // MLP FC1+FC2 bias block starts right after OPROJ bias (5*24 words)
            L_MLP_FC1:  f_b_off_word = (5*BWORDS_PER_LAYER);
            L_MLP_FC2:  f_b_off_word = (5*BWORDS_PER_LAYER + BWORDS_FC1);
            default:    f_b_off_word = 0;
        endcase
    end
endfunction

    // ---------------- TOP state ----------------
    localparam [15:0] STATE_IDLE = 16'd0;
    localparam [15:0] STATE_RUN  = 16'd2;
    localparam [15:0] STATE_DONE = 16'd3;

    reg [15:0] state;

    // MLP tile index (FC1/FC2 interleaved over 24 tiles)
    reg [5:0] mlp_tile_idx;


    // MLP W0 prefetch bookkeeping (one-shot per layer start)
    reg mlp_pf_fc1_to_fc2_sent;
    reg mlp_pf_fc2_to_fc1_sent;
    reg mlp_fc2_compute_started;
    // ---------------- scheduler FSM ----------------
    localparam [2:0] S_IDLE      = 3'd0;
    localparam [2:0] S_WAIT_BIAS = 3'd1;
    localparam [2:0] S_WAIT_RDY  = 3'd2;
    localparam [2:0] S_WAIT_PE   = 3'd3;
    localparam [2:0] S_LAYER_END = 3'd4;

    reg [2:0] fsm;

    reg [6:0] nblk_idx_cur;
    reg [6:0] mblk_idx_cur;

    assign nblk_idx_cur_o = nblk_idx_cur;
    assign mblk_idx_cur_o = mblk_idx_cur;
    assign dbg_fsm = fsm;

    // PATCH internal ktile advance guard (prevents using previous-tile IN when DDR is slow)
    reg patch_tile_hold;  // 1-cycle holdoff after ktile++ (PATCH only)

    // ---------------- Next-layer W0 prefetch gating ----------------
    // If a next-layer W0 prefetch was requested, we require it to be completed
    // (nl_pf_valid matches the requested descriptor) before we advance to the next layer.
    // Also, layer_adv_grant from controller_top can be a 1-cycle pulse, so we latch it.
    reg        layer_adv_seen;
    reg        pf_wait;
    reg  [3:0] pf_layer;
    reg  [5:0] pf_ktile;
    reg  [6:0] pf_nblk;
    reg        pf_bank;

    wire pf_done = nl_pf_valid &&
                   (nl_pf_layer == pf_layer) &&
                   (nl_pf_ktile == pf_ktile) &&
                   (nl_pf_nblk  == pf_nblk) &&
                   (nl_pf_bank  == pf_bank);


    // tile length (klen)
    function [11:0] f_klen;
        input [3:0] layer;
        input [4:0] t;
        begin
            if (layer == L_PATCH) begin
                if (t == 5'd3) f_klen = KSTRIDE_PATCH - 12'd576;
                else           f_klen = 12'd192;
            end else if (layer == L_SCORE) begin
                f_klen = 12'd64;
            end else if (layer == L_AV) begin
                f_klen = active_token_total;
            end else if (layer == L_MLP_FC2) begin
                f_klen = 12'd32;
            end else begin
                f_klen = 12'd384;
            end
        end
    endfunction

    // final ktile for "internal ktile looping" (only PATCH uses ktile loop)
    wire [5:0] cfg_final_ktile = (layer_id == L_PATCH) ? 6'd3 : 6'd0;

    wire cfg_inbuf_sel_raw =
    (layer_id == L_PATCH)  ? 1'b1 :
    (layer_id == L_SCORE)  ? 1'b0 :
    (layer_id == L_AV)     ? 1'b1 :
    (layer_id == L_O_PROJ) ? 1'b1 :
    1'b1;

wire is_cls = ((layer_id >= L_CLS0) && (layer_id <= L_CLS5));


// Fixed-role global-buffer schedule:
// - GB1 keeps the INT8 block state/residual across all encoder blocks and
//   receives the chunked PATCH input stream.
// - GB0 is used as the temporary/global-output bank for PATCH/QKV/AV and
//   32-bit non-final partial sums.
// - FC2 final residual-add writes are redirected to GB1 in the writeback path.
wire cfg_inbuf_sel =
    (layer_id == L_PATCH) ? cfg_inbuf_sel_raw :
    (is_cls)              ? 1'b1 :
                            cfg_inbuf_sel_raw;

    // ready conditions
    wire cur_in_ready =
        (layer_id == L_PATCH) ? in_ready_cur[mblk_idx_cur] :
                                1'b1;

    wire cur_w_ready =
        (layer_id == L_SCORE) ? w_ready_cur[mblk_idx_cur[0] ^ layer_wgt_phase] :
        (layer_id == L_MLP_FC2) ? w_ready_cur[nblk_idx_cur] :
                                 w_ready_cur[nblk_idx_cur];

    // last flags
    wire is_av = (layer_id == L_AV);

    wire [6:0] av_nblk_base = {3'd0, layer_head_idx, 1'b0};
    wire [6:0] av_nblk_last = av_nblk_base + 7'd1;
    localparam [2:0] RUN_ATTN_LAST_HEAD = (RUN_ATTN_HEADS <= 1) ? 3'd0 : (RUN_ATTN_HEADS - 1);

    wire last_nblk  = (layer_id == L_SCORE)   ? (nblk_idx_cur == active_mblk_last) :
                      (layer_id == L_AV)      ? (nblk_idx_cur == av_nblk_last) :
                      (layer_id == L_MLP_FC1) ? (nblk_idx_cur == 7'd0) :
                      (layer_id == L_CLS5)    ? (nblk_idx_cur == 7'd1) :
                      (is_cls)                ? (nblk_idx_cur == 7'd5) :
                                               (nblk_idx_cur == (N_BLK_NUM[6:0] - 7'd1));

    wire last_mblk  = (is_cls) ? (mblk_idx_cur == 7'd0) : (mblk_idx_cur == active_mblk_last);
wire is_mlp = (layer_id == L_MLP_FC1) || (layer_id == L_MLP_FC2);
wire is_cls5 = (layer_id == L_CLS5);
    wire last_ktile = is_mlp ? 1'b1 : (ktile_idx_cur == cfg_final_ktile);
    wire [6:0] prefetch_nblk = is_cls ? 7'd4 : (N_BLK_NUM[6:0] - 7'd2);

    // end-of-layer marker (used to suppress useless slot releases on layer boundary)
    wire end_of_layer = last_ktile && last_nblk && last_mblk;

    // slot0 stream: even nblk values
    wire do_release_slot0 =
        (!is_av) && (!is_mlp) && (!is_cls5) && (!end_of_layer) && last_mblk && !nblk_idx_cur[0];
// slot1 stream: odd nblk values
    wire do_release_slot1 =
        (!is_av) && (!is_mlp) && (!is_cls5) && (!end_of_layer) && last_mblk && nblk_idx_cur[0];
// ---------------- base broadcast ----------------
    always @(*) begin
        rd_base_in_word  = base_in_word;
        rd_base_w_word   = base_w_word + f_w_off_word(layer_id, enc_blk_idx);

        // SCORE: Q output is stored with depth=384; select head slice by adding +64*head
        if (layer_id == L_SCORE) begin
                        rd_base_w_word = base_w_word + OFF_W_Q_OUT + (enc_blk_idx * SIZE_W_ENC_BLK_WGT) + ({{(BASE_AW-3){1'b0}}, layer_head_idx} * 27'd64);
end

        if (is_cls) begin
            rd_base_b_word = base_b_word + BWORDS_PER_LAYER + (ENC_BLK_NUM * BWORDS_PER_BLOCK) + ((layer_id - L_CLS0) * BWORDS_CLS_GROUP);
        end else begin
            rd_base_b_word = base_b_word + (enc_blk_idx * BWORDS_PER_BLOCK) + f_b_off_word(layer_id);
        end
        // DDR residual sources:
        // - OPROJ reads the block input (patch output or previous add2).
        // - FC2 reads this block's add1, written to the block output slot by
        //   OPROJ before the in-place norm2 pass. FC2 later overwrites that
        //   same slot with add2.
        if (layer_id == L_O_PROJ) begin
            rd_base_out_word = base_out_word + ((enc_blk_idx == 0) ? OFF_OUT_PATCH_OUT : (OFF_OUT_O_OUT + ((enc_blk_idx - 4'd1) * SIZE_OUT_PATCH_OUT)));
        end else if (layer_id == L_MLP_FC2) begin
            rd_base_out_word = base_out_word + OFF_OUT_O_OUT +
                               (enc_blk_idx * SIZE_OUT_PATCH_OUT);
        end else begin
            rd_base_out_word = base_out_word;
        end
    end

    // ---------------- sequential ----------------
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            state <= STATE_IDLE;
            fsm   <= S_IDLE;

            done <= 0;

            layer_active      <= 0;
            layer_start_pulse <= 0;
            layer_end_pulse   <= 0;

            layer_id          <= L_PATCH;
            layer_final_ktile <= 6'd3;
            layer_inbuf_sel   <= 0;
            layer_head_idx    <= 0;
            enc_blk_idx       <= 0;
            mlp_tile_idx      <= 0;
            mlp_pf_fc1_to_fc2_sent <= 0;
            mlp_pf_fc2_to_fc1_sent <= 0;
            mlp_fc2_compute_started <= 0;
            ktile_idx_cur <= 0;
            nblk_idx_cur  <= 0;
            mblk_idx_cur  <= 0;
            patch_tile_hold <= 0;

            layer_adv_seen <= 0;
            pf_wait        <= 0;
            pf_layer       <= 0;
            pf_ktile       <= 0;
            pf_nblk        <= 0;
            pf_bank        <= 0;

            mblk_release_pulse     <= 0;
            wgt_slot_release_pulse <= 0;


            nl_wgt0_prefetch_pulse <= 0;
            nl_wgt0_layer_id       <= 0;
            nl_wgt0_ktile          <= 0;
            nl_wgt0_nblk           <= 0;
            nl_wgt0_bank           <= 0;
            nl_wgt0_base_w_word    <= 0;
            pe_start     <= 0;

            pe_klen      <= 0;
            pe_ktile     <= 0;
            pe_nblk      <= 0;
            pe_mblk      <= 0;
            pe_wbuf_sel  <= 0;
            pe_inbuf_sel <= 0;

        end else begin
            // defaults
            layer_start_pulse      <= 0;
            layer_end_pulse        <= 0;
            mblk_release_pulse     <= 0;
            wgt_slot_release_pulse <= 0;
            pe_start               <= 0;

            // auto-clear PATCH ktile holdoff
            if (patch_tile_hold) patch_tile_hold <= 1'b0;


            nl_wgt0_prefetch_pulse <= 0;
            // broadcast cfg
            layer_inbuf_sel   <= cfg_inbuf_sel;
            pe_inbuf_sel      <= cfg_inbuf_sel;
            // layer_final_ktile is for WB final-tile decisions
            layer_final_ktile <= (layer_id == L_MLP_FC2) ? 6'd47 : ((layer_id == L_PATCH) ? 6'd3 : 6'd0);

            case (state)
                STATE_IDLE: begin
                    done <= 0;
                    layer_active <= 0;

                    // clear run-scoped bookkeeping while idle (prevents stale waveforms across back-to-back runs)
                    fsm            <= S_IDLE;
                    layer_id        <= DEBUG_START_LAYER[3:0];
                    layer_wgt_phase <= 1'b0;
                    ktile_idx_cur   <= 0;
                    nblk_idx_cur    <= 0;
                    mblk_idx_cur    <= 0;

                    layer_adv_seen  <= 1'b0;
                    pf_wait         <= 1'b0;
                    pf_layer        <= 0;
                    pf_ktile        <= 0;
                    pf_nblk         <= 0;
                    pf_bank         <= 1'b0;

                    layer_head_idx  <= 0;
                    enc_blk_idx     <= DEBUG_START_ENC_BLK[3:0];
                    mlp_tile_idx    <= 0;
                    mlp_pf_fc1_to_fc2_sent   <= 1'b0;
                    mlp_pf_fc2_to_fc1_sent   <= 1'b0;
                    mlp_fc2_compute_started  <= 1'b0;
                    patch_tile_hold <= 1'b0;

                    pe_klen     <= 0;
                    pe_ktile    <= 0;
                    pe_nblk     <= 0;
                    pe_mblk     <= 0;
                    pe_wbuf_sel <= 1'b0;


                    // clear next-layer W0 prefetch descriptor while idle (avoids stale values across runs)
                    nl_wgt0_layer_id    <= 0;
                    nl_wgt0_ktile       <= 0;
                    nl_wgt0_nblk        <= 0;
                    nl_wgt0_bank        <= 1'b0;
                    nl_wgt0_base_w_word <= 0;
                    if (start) begin
                        // ensure run-start starts from clean descriptors (waveform hygiene)
                        nl_wgt0_layer_id    <= 0;
                        nl_wgt0_ktile       <= 0;
                        nl_wgt0_nblk        <= 0;
                        nl_wgt0_bank        <= 1'b0;
                        nl_wgt0_base_w_word <= 0;
                        layer_adv_seen <= 1'b0;
                        pf_wait        <= 1'b0;
                        layer_wgt_phase <= 0;
                        state <= STATE_RUN;
                        fsm   <= S_WAIT_BIAS;
                        layer_head_idx <= 0;
                        enc_blk_idx    <= DEBUG_START_ENC_BLK[3:0];
                        mlp_tile_idx  <= 0;

                        mlp_pf_fc1_to_fc2_sent <= 0;
                        mlp_pf_fc2_to_fc1_sent <= 0;
                        layer_id <= DEBUG_START_LAYER[3:0];

                        layer_active      <= 1'b1;
                        layer_start_pulse <= 1'b1;

                        ktile_idx_cur <= 0;
                        nblk_idx_cur  <= 0;
                        mblk_idx_cur  <= 0;
                    end
                end

                STATE_RUN: begin
                    case (fsm)
                        S_WAIT_BIAS: begin
                            if (quant_ready && (bias_ready || (layer_id == L_MLP_FC2))) begin
                                fsm <= S_WAIT_RDY;
                            end
                        end

                        S_WAIT_RDY: begin
                            if (pe_ready && cur_in_ready && cur_w_ready && !((layer_id == L_PATCH) && patch_tile_hold)) begin
                                // MLP FC1: as soon as we are ready to start compute,
                                // kick FC2 full-weight prefetch (nblk0..5) for this ktile into the opposite bank.
                                // (FC1 W0 is already prefetched when entering FC1, so we can overlap FC2 prefetch with FC1 compute.)
                                if ((layer_id == L_MLP_FC1) && !mlp_pf_fc1_to_fc2_sent) begin
                                    nl_wgt0_prefetch_pulse <= 1'b1;
                                    nl_wgt0_layer_id       <= L_MLP_FC2;
                                    nl_wgt0_ktile          <= ktile_idx_cur;
                                    nl_wgt0_nblk           <= 7'd0;
                                    nl_wgt0_bank           <= ~layer_wgt_phase;
                                    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_MLP_FC2, enc_blk_idx);

                                    pf_wait  <= 1'b1;
                                    pf_layer <= L_MLP_FC2;
                                    pf_ktile <= ktile_idx_cur;
                                    pf_nblk  <= 7'd0;
                                    pf_bank  <= ~layer_wgt_phase;

                                    mlp_pf_fc1_to_fc2_sent <= 1'b1;
                                end

                                if (layer_id == L_MLP_FC2) mlp_fc2_compute_started <= 1'b1;
                                pe_start <= 1'b1;

                                pe_klen  <= f_klen(layer_id, ktile_idx_cur);
                                // IMPORTANT:
                                // - FC1 uses ktile_idx_cur for weight addressing, but should appear as ktile=0 to WB (no psum accumulate)
                                // - FC2 uses ktile_idx_cur for psum accumulate across tiles
                                pe_ktile <= ktile_idx_cur;
                                pe_nblk  <= nblk_idx_cur;
                                pe_mblk  <= mblk_idx_cur;

                                // 2-slot weight: even->slot0, odd->slot1
                                pe_wbuf_sel <= (layer_id == L_MLP_FC2) ? (~layer_wgt_phase) : ((layer_id == L_SCORE) ? (mblk_idx_cur[0] ^ layer_wgt_phase) : (nblk_idx_cur[0] ^ layer_wgt_phase));

                                fsm <= S_WAIT_PE;
                            end
                        end

                        S_WAIT_PE: begin
                            if (pe_done) begin
                                // 1) weight slot release
                                if (layer_id == L_SCORE) begin
                                    if (last_nblk && !last_mblk) wgt_slot_release_pulse[mblk_idx_cur[0] ^ layer_wgt_phase] <= 1'b1;
                                end else begin
                                    if (do_release_slot0) wgt_slot_release_pulse[0] <= 1'b1;
                                    if (do_release_slot1) wgt_slot_release_pulse[1] <= 1'b1;
                                end

                                // AV: after finishing even-nblk (nblk_base) on last mblk, release slot0
                                // so we can prefetch next SCORE(head+1) W0 into freed bank while odd-nblk runs.
                                if ((layer_id == L_AV) && last_mblk &&
                                    (nblk_idx_cur == {layer_head_idx, 1'b0})) begin
                                    wgt_slot_release_pulse[0] <= 1'b1;
                                end

                                // 1b) next-layer weight0 prefetch
// - PATCH/QKV chain: after nblk4@last_mblk releases slot0 (bank = phase)
// - SCORE->AV: after each SCORE row-block, use the cached V tiles for the matching AV row-block
// - AV->SCORE(next head): after AV even-nblk done, prefetch next SCORE head W0
if ((layer_id == L_PATCH) && (RUN_QKV != 0) && last_ktile && last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_QKV_Q;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_QKV_Q;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_QKV_Q, enc_blk_idx);
end else if ((layer_id == L_QKV_Q) && (RUN_QKV >= 2) && last_ktile && last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_QKV_V;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_QKV_V;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_QKV_V, enc_blk_idx);
end else if ((layer_id == L_QKV_V) && (RUN_QKV >= 3) && last_ktile && last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_QKV_K;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_QKV_K;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_QKV_K, enc_blk_idx);
end else if ((layer_id == L_QKV_K) && (RUN_QKV >= 4) && last_ktile && last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    // SCORE head0 W0 (mblk0) bank = phase
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_SCORE;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_SCORE;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_SCORE, enc_blk_idx);
end else if ((layer_id == L_SCORE) && last_nblk) begin
    // SCORE row-block completed -> expose cached AV V tiles for the same head.
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_AV;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= {3'd0, layer_head_idx, 1'b0};
    nl_wgt0_bank           <= ~layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_AV;
    pf_ktile <= 0;
    pf_nblk  <= {3'd0, layer_head_idx, 1'b0};
    pf_bank  <= ~layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_AV, enc_blk_idx);
end else if ((layer_id == L_AV) && last_mblk && (nblk_idx_cur == {layer_head_idx, 1'b0}) && (layer_head_idx != RUN_ATTN_LAST_HEAD)) begin
    // AV even-nblk completed -> prefetch next SCORE head W0 (mblk0)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_SCORE;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_SCORE;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_SCORE, enc_blk_idx) + ({1'b0, (layer_head_idx + 3'd1), 6'd0}); // +64*(head+1)
end

else if ((layer_id == L_AV) && last_mblk && (nblk_idx_cur == {layer_head_idx, 1'b0}) && (layer_head_idx == RUN_ATTN_LAST_HEAD)) begin
    // AV even-nblk completed (last enabled head) -> prefetch OPROJ W0 (nblk0)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_O_PROJ;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_O_PROJ;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_O_PROJ, enc_blk_idx);
end

else if ((layer_id == L_O_PROJ) && last_ktile && last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    // OPROJ nblk4@last_mblk completed -> prefetch MLP_FC1 W0 (tile0)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_MLP_FC1;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_MLP_FC1;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_MLP_FC1, enc_blk_idx);
end


else if ((layer_id == L_MLP_FC2) && (mlp_tile_idx == 6'd47) && (enc_blk_idx != (RUN_ENC_BLK_NUM-1)) &&
         last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    // FC2(last tile) nblk4@last_mblk completed -> prefetch NEXT BLOCK QKV_Q W0 (tile0) into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_QKV_Q;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_QKV_Q;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_QKV_Q, enc_blk_idx + 4'd1);
end



else if ((RUN_CLASSIFIER != 0) && (layer_id == L_MLP_FC2) && (mlp_tile_idx == 6'd47) && (enc_blk_idx == (RUN_ENC_BLK_NUM-1)) &&
         last_mblk && (nblk_idx_cur == prefetch_nblk)) begin
    // FC2(last tile) nblk4@last_mblk completed -> prefetch classifier CLS0 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS0;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS0;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS0, 4'd0);
end

else if ((layer_id == L_CLS0) && last_mblk && (nblk_idx_cur == 7'd4)) begin
    // CLS0 nblk4@last_mblk completed -> prefetch CLS1 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS1;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS1;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS1, 4'd0);
end

else if ((layer_id == L_CLS1) && last_mblk && (nblk_idx_cur == 7'd4)) begin
    // CLS1 nblk4@last_mblk completed -> prefetch CLS2 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS2;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS2;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS2, 4'd0);
end

else if ((layer_id == L_CLS2) && last_mblk && (nblk_idx_cur == 7'd4)) begin
    // CLS2 nblk4@last_mblk completed -> prefetch CLS3 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS3;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS3;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS3, 4'd0);
end

else if ((layer_id == L_CLS3) && last_mblk && (nblk_idx_cur == 7'd4)) begin
    // CLS3 nblk4@last_mblk completed -> prefetch CLS4 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS4;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS4;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS4, 4'd0);
end

else if ((layer_id == L_CLS4) && last_mblk && (nblk_idx_cur == 7'd4)) begin
    // CLS4 nblk4@last_mblk completed -> prefetch CLS5 W0 into free bank (=phase)
    nl_wgt0_prefetch_pulse <= 1'b1;
    nl_wgt0_layer_id       <= L_CLS5;
    nl_wgt0_ktile          <= 0;
    nl_wgt0_nblk           <= 0;
    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_CLS5;
    pf_ktile <= 0;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_CLS5, 4'd0);
end

// NOTE: MLP FC1/FC2 W0 prefetch is now triggered as soon as the current layer's
//       required weights are fully loaded (see MLP early-prefetch block below).

                                // MLP FC2->FC1 W0 prefetch fallback at end-of-compute
                                // (If early-prefetch didn't fire, still request next-tile FC1 W0 here.)
                                if ((layer_id == L_MLP_FC2) && !mlp_pf_fc2_to_fc1_sent && (ktile_idx_cur != 6'd47)) begin
                                    nl_wgt0_prefetch_pulse <= 1'b1;
                                    nl_wgt0_layer_id       <= L_MLP_FC1;
                                    nl_wgt0_ktile          <= ktile_idx_cur + 1'b1;
                                    nl_wgt0_nblk           <= 0;
                                    nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_MLP_FC1;
    pf_ktile <= ktile_idx_cur + 1'b1;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

                                    nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_MLP_FC1, enc_blk_idx);
                                    mlp_pf_fc2_to_fc1_sent <= 1'b1;
                                end

                                // 2) input mblk release for next ktile overwrite (patch only)
                                if ((layer_id == L_PATCH) && last_nblk && !last_ktile) begin
                                    mblk_release_pulse <= ({{(M_BLK_NUM-1){1'b0}}, 1'b1} << mblk_idx_cur);
                                end

                                // 3) advance indices
                                if (layer_id == L_SCORE) begin
                                    // Row-interleaved attention: one SCORE mblk sweeps
                                    // all score nblks, then immediately hands off to AV
                                    // for the same mblk.
                                    if (!last_nblk) begin
                                        nblk_idx_cur <= nblk_idx_cur + 1'b1;
                                        fsm <= S_WAIT_RDY;
                                    end else begin
                                        layer_end_pulse <= 1'b1;
                                        layer_adv_seen <= 1'b0;
                                        fsm <= S_LAYER_END;
                                    end
                                end else if (layer_id == L_AV) begin
                                    // AV consumes the just-softmaxed SCOREBUF row
                                    // for the same mblk and only sweeps this head's
                                    // two V output-channel nblks.
                                    if (!last_nblk) begin
                                        nblk_idx_cur <= nblk_idx_cur + 1'b1;
                                        fsm <= S_WAIT_RDY;
                                    end else begin
                                        layer_end_pulse <= 1'b1;
                                        layer_adv_seen <= 1'b0;
                                        fsm <= S_LAYER_END;
                                    end
                                end else begin
                                    // default order: mblk inner, then nblk, then ktile
                                    if (!last_mblk) begin
                                        mblk_idx_cur <= mblk_idx_cur + 1'b1;
                                        fsm <= S_WAIT_RDY;
                                    end else if (!last_nblk) begin
                                        mblk_idx_cur <= 0;
                                        nblk_idx_cur <= nblk_idx_cur + 1'b1;
                                        fsm <= S_WAIT_RDY;
                                    end else if (!last_ktile) begin
                                        mblk_idx_cur  <= 0;
                                        nblk_idx_cur  <= 0;
                                        ktile_idx_cur <= ktile_idx_cur + 1'b1;

                                        if (layer_id == L_PATCH) patch_tile_hold <= 1'b1;
                                        fsm <= S_WAIT_RDY;
                                    end else begin
                                        layer_end_pulse <= 1'b1;
                                        layer_adv_seen <= 1'b0;
                                        fsm <= S_LAYER_END;
                                    end
                                end
                            end
                        end

                                                S_LAYER_END: begin
                            // layer_adv_grant can be a 1-cycle pulse; latch it until we advance
                            if (layer_adv_grant) begin
                                layer_adv_seen <= 1'b1;
                            end

                            // Advance only when: (1) dump grant observed and (2) next-layer W0 prefetch done (if requested)
                            if (layer_adv_seen && (!pf_wait || pf_done)) begin
                                layer_adv_seen <= 1'b0;
                                pf_wait        <= 1'b0;

                                if ((RUN_QKV == 0) && (layer_id == L_PATCH)) begin
                                    layer_active <= 0;
                                    done         <= 1'b1;
                                    state        <= STATE_DONE;

                                end else if ((RUN_QKV != 0) && (layer_id == L_PATCH)) begin
                                    // Patch -> Q
                                    layer_id <= L_QKV_Q;

                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;

                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if ((RUN_QKV >= 2) && (layer_id == L_QKV_Q)) begin
                                    // Q -> V
                                    layer_id <= L_QKV_V;

                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;

                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if ((RUN_QKV >= 3) && (layer_id == L_QKV_V)) begin
                                    // V -> K
                                    layer_id <= L_QKV_K;

                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;

                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if ((RUN_QKV >= 4) && (layer_id == L_QKV_K)) begin
                                    // K -> SCORE
                                    layer_head_idx <= 3'd0;
                                    layer_wgt_phase <= 1'b0;
                                    layer_id <= L_SCORE;

                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;

                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if ((RUN_QKV >= 5) && (layer_id == L_SCORE)) begin
                                    // SCORE row-block -> AV row-block
                                    layer_wgt_phase <= ~layer_wgt_phase;
                                    layer_id <= L_AV;

                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= {layer_head_idx, 1'b0};

                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if ((RUN_QKV >= 5) && (layer_id == L_AV)) begin
                                    // AV row-block done -> next SCORE row-block in the
                                    // same head; after the last row, move to next head.
                                    if (!last_mblk) begin
                                        layer_id <= L_SCORE;

                                        ktile_idx_cur <= 0;
                                        nblk_idx_cur  <= 0;
                                        mblk_idx_cur  <= mblk_idx_cur + 1'b1;

                                        layer_start_pulse <= 1'b1;
                                        fsm <= S_WAIT_BIAS;
                                    end else if (layer_head_idx != RUN_ATTN_LAST_HEAD) begin
                                        layer_head_idx <= layer_head_idx + 3'd1;
                                        layer_id <= L_SCORE;

                                        ktile_idx_cur <= 0;
                                        nblk_idx_cur  <= 0;
                                        mblk_idx_cur  <= 0;

                                        layer_start_pulse <= 1'b1;
                                        fsm <= S_WAIT_BIAS;
                                    end else begin
                                        if (RUN_QKV >= 6) begin
                                            // AV(head5) done -> OPROJ
                                            layer_head_idx <= 0;
                                            layer_id <= L_O_PROJ;
                                            ktile_idx_cur <= 0;
                                            nblk_idx_cur  <= 0;
                                            mblk_idx_cur  <= 0;
                                            layer_start_pulse <= 1'b1;
                                            fsm <= S_WAIT_BIAS;
                                        end else begin
                                            layer_active <= 0;
                                            done         <= 1'b1;
                                            state        <= STATE_DONE;
                                        end
                                    end

                                end else if ((RUN_QKV >= 6) && (layer_id == L_O_PROJ)) begin
                                    // OPROJ -> MLP (FC1/FC2 interleaved over 24 tiles)
                                    mlp_tile_idx <= 0;
                                    layer_id <= L_MLP_FC1;
                                    mlp_pf_fc1_to_fc2_sent <= 0;
                                        mlp_fc2_compute_started <= 0;
                                    mlp_pf_fc2_to_fc1_sent <= 0;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_MLP_FC1) begin
                                    // MLP FC1(tile t) -> MLP FC2(tile t)
                                    layer_id <= L_MLP_FC2;
                                    mlp_pf_fc2_to_fc1_sent <= 0;
                                    mlp_fc2_compute_started <= 0;
                                    ktile_idx_cur <= mlp_tile_idx;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_MLP_FC2) begin
                                    if (mlp_tile_idx == 6'd47) begin
                                        if (enc_blk_idx == (RUN_ENC_BLK_NUM-1)) begin
                                            if (RUN_CLASSIFIER != 0) begin
                                                // Enter classifier head (cls0..cls5)
                                                layer_head_idx <= 0;
                                                layer_id <= L_CLS0;

                                                mlp_tile_idx <= 0;
                                                mlp_pf_fc1_to_fc2_sent <= 0;
                                                mlp_pf_fc2_to_fc1_sent <= 0;
                                                mlp_fc2_compute_started <= 0;

                                                ktile_idx_cur <= 0;
                                                nblk_idx_cur  <= 0;
                                                mblk_idx_cur  <= 0;

                                                layer_start_pulse <= 1'b1;
                                                fsm <= S_WAIT_BIAS;
                                            end else begin
                                                layer_active <= 0;
                                                done         <= 1'b1;
                                                state        <= STATE_DONE;
                                            end
                                        end else begin
                                            // Next encoder block
                                            enc_blk_idx <= enc_blk_idx + 1'b1;

                                            layer_head_idx <= 0;
                                            layer_id <= L_QKV_Q;

                                            mlp_tile_idx <= 0;
                                            mlp_pf_fc1_to_fc2_sent <= 0;
                                            mlp_pf_fc2_to_fc1_sent <= 0;
                                            mlp_fc2_compute_started <= 0;

                                            ktile_idx_cur <= 0;
                                            nblk_idx_cur  <= 0;
                                            mblk_idx_cur  <= 0;

                                            layer_start_pulse <= 1'b1;
                                            fsm <= S_WAIT_BIAS;
                                        end
                                    end else begin
                                        mlp_tile_idx <= mlp_tile_idx + 1'b1;
                                        layer_id <= L_MLP_FC1;
                                        mlp_pf_fc1_to_fc2_sent <= 0;
                                        ktile_idx_cur <= mlp_tile_idx + 1'b1;
                                        nblk_idx_cur  <= 0;
                                        mblk_idx_cur  <= 0;
                                        layer_start_pulse <= 1'b1;
                                        fsm <= S_WAIT_BIAS;
                                    end

                                end else if (layer_id == L_CLS0) begin
                                    layer_head_idx <= 0;
                                    layer_id <= L_CLS1;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_CLS1) begin
                                    layer_head_idx <= 0;
                                    layer_id <= L_CLS2;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_CLS2) begin
                                    layer_head_idx <= 0;
                                    layer_id <= L_CLS3;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_CLS3) begin
                                    layer_head_idx <= 0;
                                    layer_id <= L_CLS4;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_CLS4) begin
                                    layer_head_idx <= 0;
                                    layer_id <= L_CLS5;
                                    ktile_idx_cur <= 0;
                                    nblk_idx_cur  <= 0;
                                    mblk_idx_cur  <= 0;
                                    layer_start_pulse <= 1'b1;
                                    fsm <= S_WAIT_BIAS;

                                end else if (layer_id == L_CLS5) begin
                                    layer_active <= 0;
                                    done         <= 1'b1;
                                    state        <= STATE_DONE;

                                end else begin
                                    layer_active <= 0;
                                    done         <= 1'b1;
                                    state        <= STATE_DONE;
                                end
                            
                            end
                        end

default: fsm <= S_WAIT_BIAS;
                    endcase

                    // ------------------------------------------------------------
                    // MLP early W0 prefetch:
                    // - FC1: once W0 is loaded (w_ready_cur[0]==1), immediately prefetch FC2 W0 of the same tile
                    // - FC2: once all 12 weight blocks are loaded, immediately prefetch next-tile FC1 W0
                    // This does NOT wait for compute progress; it only depends on weight-load completion.
                    // ------------------------------------------------------------
                    if (layer_active) begin
                        // FC1(tile t) -> FC2(tile t) W0 prefetch (into ~phase bank)                        // (FC1->FC2 full prefetch is now triggered in S_WAIT_RDY when FC1 compute starts)

                        // FC2(tile t) -> FC1(tile t+1) W0 prefetch (into phase bank)
                        if ((layer_id == L_MLP_FC2) && mlp_fc2_compute_started && !mlp_pf_fc2_to_fc1_sent && w_ready_cur[N_BLK_NUM-1] && (ktile_idx_cur != 6'd47)) begin
                            nl_wgt0_prefetch_pulse <= 1'b1;
                            nl_wgt0_layer_id       <= L_MLP_FC1;
                            nl_wgt0_ktile          <= ktile_idx_cur + 1'b1;
                            nl_wgt0_nblk           <= 0;
                            nl_wgt0_bank           <= layer_wgt_phase;
    pf_wait  <= 1'b1;
    pf_layer <= L_MLP_FC1;
    pf_ktile <= ktile_idx_cur + 1'b1;
    pf_nblk  <= 0;
    pf_bank  <= layer_wgt_phase;

                            nl_wgt0_base_w_word    <= base_w_word + f_w_off_word(L_MLP_FC1, enc_blk_idx);
                            mlp_pf_fc2_to_fc1_sent <= 1'b1;
                        end
                    end
                end
                STATE_DONE: begin
                    if (!start) begin
                        state <= STATE_IDLE;
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
