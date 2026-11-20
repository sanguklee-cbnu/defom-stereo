`timescale 1ns/1ps
`default_nettype none

module controller_top #(
    parameter integer M_AXI_DATA_WIDTH = 256,
    parameter integer M_AXI_ADDR_WIDTH = 32,
    parameter integer ADDR_LSB         = $clog2(M_AXI_DATA_WIDTH/8),

    parameter integer M_BLK_NUM   = 59,
    parameter integer N_BLK_NUM   = 12,
    parameter integer OUT_STRIDE  = 384,
    parameter integer M_TOTAL     = 1873,
    parameter integer GOUT_AW     = 18,
 
    // vit_controller base word-index width
    parameter integer BASE_AW     = (M_AXI_ADDR_WIDTH-ADDR_LSB),

    // ---- Run control ----
        parameter integer RUN_QKV     = 6,      // 0: PATCH only, 1: PATCH->Q, 2: PATCH->Q->V, 3: PATCH->Q->V->K, 4: +SCORE(head0)
    parameter integer RUN_ATTN_HEADS = 6,
    parameter integer RUN_ENC_BLK_NUM = 12,
    parameter integer RUN_CLASSIFIER  = 0,
    parameter integer DUMP_ENABLE = 1,      // 0: disable DDR dump, 1: enable
    parameter [3:0]   DUMP_LAYER_ID = 4'd1, // default: dump Q layer (1). PATCH=0
    parameter integer FINAL_OUT_WRITE_ENABLE = 1, // 1: always write final MLP output to DDR
    parameter integer ENABLE_LAYER_NORM = 1,      // 1: run SCOREBUF-backed post-layer LN pass
    parameter integer LN_AFFINE_MODE = 2,
    parameter integer LN_PARAM_SETS = 29,
    parameter integer LN_AFFINE_BASE_OFFSET_WORD = 2688,
    parameter integer LN_AFFINE_SET_WORDS = 60,
    parameter LN_GAIN_MEM = "ln_gain_q16.mem",
    parameter LN_BIAS_MEM = "ln_bias_i8.mem",
    parameter LN_RECIP_MEM = "ln_recip_lut_u16.mem",
    parameter LN_EPSILON_MEM = "ln_epsilon_u15.mem",
    parameter SM_ADAPTER_MUL_MEM = "sm_adapter_mul_s16.mem",
    parameter SM_ADAPTER_SHIFT_MEM = "sm_adapter_shift_u6.mem",
    parameter SM_H_MEM = "sm_h_u4.mem",
    parameter SM_B_MEM = "sm_b_u22.mem",
    parameter SM_EXP_U15_MEM = "softmax_exp_u15.mem",
    parameter FC1_GELU_MEM = "gelu_lut_i8.mem",
    parameter OPROJ_MUL_MEM = "oproj_mul_s16.mem",
    parameter FC2_MUL_MEM = "fc2_mul_s16.mem",
    parameter integer RESIDUAL_REGION_BASE = 22656,
    parameter integer DEBUG_START_LAYER = 0,
    parameter integer DEBUG_START_ENC_BLK = 0
)(
    input  wire clk,
    input  wire nRst,

    input  wire start,
    output wire done,

    // AXI read/write to DDR (simple control IF)
    input  wire                        m_axi_read_done,
    output wire                        m_axi_read_init,
    output wire [M_AXI_ADDR_WIDTH-1:0] m_axi_araddress,
    output wire [16:0]                 m_axi_read_length,

    input  wire                        m_axi_write_done,
    output wire                        m_axi_write_init,
    output wire [M_AXI_ADDR_WIDTH-1:0] m_axi_awaddress,
    output wire [16:0]                 m_axi_write_length,

    // ---- Source-side WDATA stream to AXI_Master ----
    output wire [M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output wire                        m_axi_wvalid,
    input  wire                        m_axi_wready,

    // (optional) keep for debug
    output wire                        m_axi_wlast,

    // base addresses (word index)
    input  wire [BASE_AW-1:0] base_in_word,
    input  wire [BASE_AW-1:0] base_w_word,
    input  wire [BASE_AW-1:0] base_b_word,
    input  wire [BASE_AW-1:0] base_out_word,

    input  wire [BASE_AW-1:0] base_q_word,
    input  wire [11:0] cfg_token_total,
    input  wire [6:0]  cfg_mblk_num,
    input  wire [4:0]  cfg_last_valid_lane,
    // AXI R-channel tap
    input  wire                         axi_rvalid,
    input  wire [M_AXI_DATA_WIDTH-1:0]  axi_rdata,
    input  wire                         axi_rlast,
    input  wire                         axi_rready,

    // DDR->SRAM write port
    output wire                         mem_wr_valid,
    output wire [2:0]                   mem_wr_kind,   // 1:IN 2:WGT 3:BIAS
    output wire                         mem_wr_bank,
    output wire [GOUT_AW-1:0]           mem_wr_addr,
    output wire [M_AXI_DATA_WIDTH-1:0]  mem_wr_data,

    // LayerNorm affine parameters use a dedicated Bias-RAM write port. This
    // keeps the affine-loader mux and its control fanout off the global-buffer
    // BRAM address/data inputs.
    output wire                         ln_aff_wr_valid,
    output wire [6:0]                   ln_aff_wr_addr,
    output wire [M_AXI_DATA_WIDTH-1:0]  ln_aff_wr_data,

    // PE output (to writeback)
    input  wire                    pe_out_col_valid,
    output wire                    pe_out_col_ready,
    input  wire [4:0]              pe_out_col_idx,
    input  wire [32*32-1:0]        pe_out_col_data_flat,

    // Global OUT bank port (to memory_controller)
    output wire                    gout_rd_en,
    output wire [GOUT_AW-1:0]      gout_rd_addr,
    input  wire [255:0]            gout_rd_data,
    output wire                    gout_rd_bank_ovr_en,
    output wire                    gout_rd_bank_ovr,
    
    // Global IN read (FC2 residual add) from memory_controller IN bank
    output wire                    gin_rd_en,
    output wire [GOUT_AW-1:0]      gin_rd_addr,
    input  wire [255:0]            gin_rd_data,

    output wire                    gout_wr_valid,
    output wire [GOUT_AW-1:0]      gout_wr_addr,
    output wire [255:0]            gout_wr_data,
    output wire                    gout_wr_bank_ovr_en,
    output wire                    gout_wr_bank_ovr,

    output wire                    psum_rd_en,
    output wire [GOUT_AW-1:0]      psum_rd_addr,
    input  wire [1023:0]           psum_rd_data,
    output wire                    psum_wr_valid,
    output wire [GOUT_AW-1:0]      psum_wr_addr,
    output wire [1023:0]           psum_wr_data,

    // SCOREBUF raw-score buffer (ping-pong) R/W for SCORE softmax
    output wire                    score_rd_en,
    output wire                    score_rd_bank,
    output wire [11:0]             score_rd_addr,
    input  wire [255:0]            score_rd_data,

    output wire                    score_wr_valid,
    output wire                    score_wr_bank,
    output wire [11:0]             score_wr_addr,
    output wire [255:0]            score_wr_data,

    // Legacy softmax-output buffer port. The row-interleaved attention path
    // writes qlog softmax results back into SCOREBUF instead.
    output wire                    attn_wr_valid,
    output wire [GOUT_AW-1:0]      attn_wr_addr,
    output wire [255:0]            attn_wr_data,

    // Bias port
    output wire                    bias_rd_en,
    output wire [6:0]              bias_rd_addr,
    input  wire [255:0]            bias_rd_data,

    // quant constants read port
    output wire        quant_rd_en,
    output wire [5:0]  quant_rd_addr,
    input  wire [255:0] quant_rd_data,
    // Compute handshake
    input  wire pe_ready,
    input  wire pe_done,
    output wire pe_start,

    // Job descriptor export
    output wire [11:0] pe_klen,
    output wire [5:0] pe_ktile,
    output wire [6:0] pe_nblk,
    output wire [6:0] pe_mblk,

    input  wire [5:0] wb_ktile_idx,
    input  wire [6:0] wb_nblk_idx,
    input  wire [6:0] wb_mblk_idx,

    output wire       pe_wbuf_sel,
    output wire       pe_inbuf_sel, // (memory_controller bank sel)
    output wire       pe_swap_ab,
    output wire [9:0] pe_in_koff,
    output wire       pe_in_base_is_nblk,
    output wire       pe_a_is_qlog,

    output wire [31:0] dbg_status0,
    output wire [31:0] dbg_status1,
    output wire [31:0] dbg_status2,
    output wire [31:0] dbg_status3,
    output wire [31:0] dbg_status4
);
    `include "addr_map.vh"

    // Runtime image-size configuration. Physical buffers and DDR regions keep
    // the 59-mblk maximum layout; these values only limit active token loops.
    wire [11:0] active_token_total =
        ((cfg_token_total == 12'd0) || (cfg_token_total > M_TOTAL[11:0])) ?
        M_TOTAL[11:0] : cfg_token_total;
    wire [6:0] active_mblk_num =
        ((cfg_mblk_num == 7'd0) || (cfg_mblk_num > M_BLK_NUM[6:0])) ?
        M_BLK_NUM[6:0] : cfg_mblk_num;
    wire [6:0] active_mblk_last = active_mblk_num - 7'd1;
    wire [4:0] active_last_valid_lane = cfg_last_valid_lane;
    wire [5:0] active_last_tile_words = {1'b0, active_last_valid_lane} + 6'd1;


// ---------------- layer id constants (keep consistent with vit_controller) ----------------
localparam [3:0] L_PATCH  = 4'd0;
localparam [3:0] L_QKV_Q  = 4'd1;
localparam [3:0] L_QKV_V  = 4'd2;
localparam [3:0] L_O_PROJ = 4'd6;

localparam [3:0] L_MLP_FC2 = 4'd8;
    
localparam [3:0] L_CLS0   = 4'd9;
localparam [3:0] L_CLS1   = 4'd10;
localparam [3:0] L_CLS2   = 4'd11;
localparam [3:0] L_CLS3   = 4'd12;
localparam [3:0] L_CLS4   = 4'd13;
localparam [3:0] L_CLS5   = 4'd14;
// ---------------- vit<->read interconnect ----------------
    wire layer_active, layer_start_pulse;
    wire [5:0] ktile_idx_cur;
    wire [6:0] sched_nblk_idx_cur;
    wire [6:0] sched_mblk_idx_cur;
    wire [M_BLK_NUM-1:0] mblk_release_pulse;
    wire [1:0] wgt_slot_release_pulse;


    // next-layer W0 prefetch (PATCH->QKV_Q)
    wire        nl_wgt0_prefetch_pulse;
    wire [3:0]  nl_wgt0_layer_id;
    wire [5:0]  nl_wgt0_ktile;
    wire [6:0]  nl_wgt0_nblk;
    wire        nl_wgt0_bank;
    wire [BASE_AW-1:0] nl_wgt0_base_w_word;

    wire vit_done;

    wire bias_ready;
    wire quant_ready;
    wire [M_BLK_NUM-1:0] in_ready_cur;
    wire [11:0] w_ready_cur;
wire [11:0] res_ready_cur;

    wire [6:0] res_upto0, res_upto1, res_upto2, res_upto3, res_upto4, res_upto5;
    wire [6:0] res_upto6, res_upto7, res_upto8, res_upto9, res_upto10, res_upto11;
    // layer broadcast
    wire [3:0] layer_id;
    wire [2:0]       layer_head_idx;
wire [3:0]       enc_blk_idx;
    wire             layer_wgt_phase;
    wire [5:0] layer_final_ktile;
    wire       layer_inbuf_sel;
    wire [2:0] dbg_fsm;
    reg  [3:0]       layer_id_pe_q;
    reg  [2:0]       layer_head_idx_pe_q;
    reg  [3:0]       layer_id_ctrl_q;
wire [3:0]       layer_id_pe_eff       = layer_start_pulse ? layer_id       : layer_id_pe_q;
    wire [2:0]       layer_head_idx_pe_eff = layer_start_pulse ? layer_head_idx : layer_head_idx_pe_q;
wire [11:0] w_ready_vit;
wire patch_pos_phase = (layer_id_ctrl_q == L_PATCH) && (ktile_idx_cur == layer_final_ktile);
wire needs_score_res_ready =
    (layer_id_ctrl_q == L_O_PROJ) || patch_pos_phase ||
    ((layer_id_ctrl_q == L_MLP_FC2) &&
     (ktile_idx_cur == layer_final_ktile));
assign w_ready_vit = needs_score_res_ready ? (w_ready_cur & res_ready_cur) : w_ready_cur;
    // Swap A/B inputs for V projection (layer_id==2) and SCORE (layer_id==4)
    assign pe_swap_ab = (layer_id_pe_eff == 4'd2) || (layer_id_pe_eff == 4'd4);
    // SCORE: input base uses nblk (K block) rather than mblk
    assign pe_in_base_is_nblk = (layer_id_pe_eff == 4'd4);
    assign pe_in_koff      = (layer_id_pe_eff == 4'd4) ? ({1'b0, layer_head_idx_pe_eff, 6'd0}) : 10'd0;
assign pe_a_is_qlog      = (layer_id_pe_eff == 4'd5);

    // layer-end synchronization
    wire layer_end_pulse;
    wire layer_adv_grant;

    // next-layer W0 prefetch completion (from read_controller)
    wire        nl_pf_valid;
    wire [3:0]  nl_pf_layer;
    wire [5:0]  nl_pf_ktile;
    wire [6:0]  nl_pf_nblk;
    wire        nl_pf_bank;

    // With layer-end wait enabled (vit_controller_wait), layer_id is held
    // stable until all DDR dumps for the layer are completed.
    wire [3:0] layer_id_dump = layer_id;

    // per-layer base (from vit_controller)
    wire [BASE_AW-1:0] rd_base_in_word;
    wire [BASE_AW-1:0] rd_base_w_word;
    wire [BASE_AW-1:0] rd_base_b_word;
    wire [BASE_AW-1:0] rd_base_out_word;
    reg  [BASE_AW-1:0] base_q_word_rd_q;

    wire                        rd_m_axi_read_init;
    wire [M_AXI_ADDR_WIDTH-1:0] rd_m_axi_araddress;
    wire [16:0]                 rd_m_axi_read_length;
    wire                        rd_mem_wr_valid;
    wire [2:0]                  rd_mem_wr_kind;
    wire                        rd_mem_wr_bank;
    wire [GOUT_AW-1:0]          rd_mem_wr_addr;
    wire [M_AXI_DATA_WIDTH-1:0] rd_mem_wr_data;

    // ---------------- vit_controller ----------------
    vit_controller #(
        .BASE_AW (BASE_AW),
        .M_TOTAL(M_TOTAL),
        .M_BLK_NUM(M_BLK_NUM),
        .N_BLK_NUM(N_BLK_NUM),
        .RUN_QKV (RUN_QKV),
        .RUN_ATTN_HEADS(RUN_ATTN_HEADS),
        .RUN_ENC_BLK_NUM(RUN_ENC_BLK_NUM),
        .RUN_CLASSIFIER (RUN_CLASSIFIER),
        .DEBUG_START_LAYER(DEBUG_START_LAYER),
        .DEBUG_START_ENC_BLK(DEBUG_START_ENC_BLK)
    ) u_vit (
        .clk(clk),
        .nRst(nRst),

        .start(start),
        .done(vit_done),

        .bias_ready(bias_ready),
        .quant_ready(quant_ready),
        .in_ready_cur(in_ready_cur),
        .w_ready_cur(w_ready_vit),
        .active_token_total(active_token_total),
        .active_mblk_num(active_mblk_num),

        .nl_pf_valid(nl_pf_valid),
        .nl_pf_layer(nl_pf_layer),
        .nl_pf_ktile(nl_pf_ktile),
        .nl_pf_nblk (nl_pf_nblk),
        .nl_pf_bank(nl_pf_bank),

        .layer_end_pulse(layer_end_pulse),
        .layer_adv_grant(layer_adv_grant),

        .layer_id(layer_id),
        .layer_final_ktile(layer_final_ktile),
        .layer_inbuf_sel(layer_inbuf_sel),
        .layer_head_idx(layer_head_idx),
        .enc_blk_idx (enc_blk_idx),
        .layer_wgt_phase(layer_wgt_phase),

        .layer_active(layer_active),
        .layer_start_pulse(layer_start_pulse),
        .ktile_idx_cur(ktile_idx_cur),
        .nblk_idx_cur_o(sched_nblk_idx_cur),
        .mblk_idx_cur_o(sched_mblk_idx_cur),
        .dbg_fsm(dbg_fsm),
        .mblk_release_pulse(mblk_release_pulse),
        .wgt_slot_release_pulse(wgt_slot_release_pulse),


        .nl_wgt0_prefetch_pulse(nl_wgt0_prefetch_pulse),
        .nl_wgt0_layer_id      (nl_wgt0_layer_id),
        .nl_wgt0_ktile         (nl_wgt0_ktile),
        .nl_wgt0_nblk          (nl_wgt0_nblk),
        .nl_wgt0_bank          (nl_wgt0_bank),
        .nl_wgt0_base_w_word   (nl_wgt0_base_w_word),

        .pe_ready(pe_ready),
        .pe_start(pe_start),
        .pe_done(pe_done),

        .pe_klen     (pe_klen),
        .pe_ktile    (pe_ktile),
        .pe_nblk     (pe_nblk),
        .pe_mblk     (pe_mblk),
        .pe_wbuf_sel (pe_wbuf_sel),
        .pe_inbuf_sel(pe_inbuf_sel),

        .base_in_word (base_in_word),
        .base_w_word  (base_w_word),
        .base_b_word  (base_b_word),
        .base_out_word(base_out_word),

        .rd_base_in_word (rd_base_in_word),
        .rd_base_w_word  (rd_base_w_word),
        .rd_base_b_word  (rd_base_b_word),
        .rd_base_out_word(rd_base_out_word)
    );

    // Quantization constants are stored per encoder block (word index):
    // - Normal encoder layers use enc_blk_idx (0..ENC_BLK_NUM-1).
    // - Classifier head (L_CLS0..L_CLS5) uses the dedicated quant block at index ENC_BLK_NUM,
    //   because gen_memhex.py appends classifier quant words after all encoder blocks.
    // - Small FC2 needs two auxiliary quant words inside each block, so the block stride is 48 words:
    //   normal layer words at +0..15, FC2 aux shift words at +24 and +40.
    //   (Keep enc_blk_idx unchanged so global-buffer ping-pong still matches the last encoder block output.)
    wire is_cls_layer = ((layer_id >= L_CLS0) && (layer_id <= L_CLS5));
    wire [3:0] q_blk_idx_sel = is_cls_layer ? ENC_BLK_NUM[3:0] : enc_blk_idx;
    // base_q_word_eff = base_q_word + q_blk_idx_sel*48  (read_controller uses local indices within the block)
    wire [BASE_AW-1:0] enc_blk_q_idx_ext = { {(BASE_AW-4){1'b0}}, q_blk_idx_sel };
    wire [BASE_AW-1:0] enc_blk_q_off    = (enc_blk_q_idx_ext << 5) + (enc_blk_q_idx_ext << 4);
    wire [BASE_AW-1:0] base_q_word_eff  = base_q_word + enc_blk_q_off;
    wire [BASE_AW-1:0] base_q_word_rd_eff = layer_start_pulse ? base_q_word_eff : base_q_word_rd_q;


// ---------------- read_controller ----------------
    read_controller #(
        .M_TOTAL(M_TOTAL),
        .M_BLK_NUM(M_BLK_NUM),
        .N_BLK_NUM(N_BLK_NUM)
    ) u_read (
        .clk(clk),
        .nRst(nRst),

        .layer_active(layer_active),
        .layer_start_pulse(layer_start_pulse),
        .ktile_idx_cur(ktile_idx_cur),

        .layer_id(layer_id),
        .layer_final_ktile(layer_final_ktile),
        .layer_inbuf_sel(layer_inbuf_sel),
        .layer_head_idx(layer_head_idx),
        .layer_wgt_phase(layer_wgt_phase),
        .mblk_idx_cur(sched_mblk_idx_cur),
        .nblk_idx_cur(sched_nblk_idx_cur),
        .active_token_total(active_token_total),
        .active_mblk_num(active_mblk_num),

        .mblk_release_pulse(mblk_release_pulse),
        .wgt_slot_release_pulse(wgt_slot_release_pulse),


        .nl_wgt0_prefetch_pulse(nl_wgt0_prefetch_pulse),
        .nl_wgt0_layer_id      (nl_wgt0_layer_id),
        .nl_wgt0_ktile         (nl_wgt0_ktile),
        .nl_wgt0_nblk          (nl_wgt0_nblk),
        .nl_wgt0_bank          (nl_wgt0_bank),
        .nl_wgt0_base_w_word   (nl_wgt0_base_w_word),

        .base_in_word (rd_base_in_word),
        .base_w_word  (rd_base_w_word),
        .base_b_word  (rd_base_b_word),
        .base_out_word(rd_base_out_word),
        .base_q_word(base_q_word_rd_eff),

        .m_axi_read_done(m_axi_read_done),
        .m_axi_read_init(rd_m_axi_read_init),
        .m_axi_araddress(rd_m_axi_araddress),
        .m_axi_read_length(rd_m_axi_read_length),

        .bias_ready(bias_ready),
        .quant_ready(quant_ready),
        .in_ready_cur(in_ready_cur),
        .w_ready_cur(w_ready_cur),
        .res_ready_cur(res_ready_cur),

        .res_upto0(res_upto0),
        .res_upto1(res_upto1),
        .res_upto2(res_upto2),
        .res_upto3(res_upto3),
        .res_upto4(res_upto4),
        .res_upto5(res_upto5),
        .res_upto6(res_upto6),
        .res_upto7(res_upto7),
        .res_upto8(res_upto8),
        .res_upto9(res_upto9),
        .res_upto10(res_upto10),
        .res_upto11(res_upto11),

        .nl_pf_valid(nl_pf_valid),
        .nl_pf_layer(nl_pf_layer),
        .nl_pf_ktile(nl_pf_ktile),
        .nl_pf_nblk (nl_pf_nblk),
        .nl_pf_bank(nl_pf_bank),
        .axi_rvalid(axi_rvalid),
        .axi_rdata (axi_rdata),
        .axi_rlast (axi_rlast),
        .axi_rready(axi_rready),

        .mem_wr_valid(rd_mem_wr_valid),
        .mem_wr_kind (rd_mem_wr_kind),
        .mem_wr_bank (rd_mem_wr_bank),
        .mem_wr_addr (rd_mem_wr_addr),
        .mem_wr_data (rd_mem_wr_data)
    );


    // dump layer match (declared early for backpressure gating)
    wire dump_layer_match;
    // ---------------- writeback_controller ----------------
    wire wb_ready;
    wire wb_busy;
    // Fine-grain residual safety (OPROJ): allow PE output only when the corresponding residual tile is loaded.
    reg [6:0] res_upto_cur;
    always @(*) begin
    case (sched_nblk_idx_cur)
    4'd0: res_upto_cur = res_upto0;
    4'd1: res_upto_cur = res_upto1;
    4'd2: res_upto_cur = res_upto2;
    4'd3: res_upto_cur = res_upto3;
    4'd4: res_upto_cur = res_upto4;
    4'd5: res_upto_cur = res_upto5;
    4'd6: res_upto_cur = res_upto6;
    4'd7: res_upto_cur = res_upto7;
    4'd8: res_upto_cur = res_upto8;
    4'd9: res_upto_cur = res_upto9;
    4'd10: res_upto_cur = res_upto10;
    4'd11: res_upto_cur = res_upto11;
    default: res_upto_cur = 7'd0;
    endcase
    end

    wire res_tile_ok = res_ready_cur[sched_nblk_idx_cur] && (sched_mblk_idx_cur <= res_upto_cur);

    // Stream-dump pass: when dumping this layer AND we're on the final ktile.
    // Data for a 32-word tile reaches the dump FIFO before the tile descriptor
    // is pushed at the final column.  Therefore, apply dump backpressure only
    // at the first column of a dump tile; once a tile starts, let all 32 words
    // drain so the descriptor/data stream cannot be split.
    wire dump_stream_pass = dump_layer_match && (wb_ktile_idx == layer_final_ktile);
    wire dump_stream_ok;
    wire dump_stream_tile_start_gate = dump_stream_pass && pe_out_col_valid && (pe_out_col_idx == 5'd0);

    assign pe_out_col_ready = wb_ready && (!needs_score_res_ready || res_tile_ok) &&
                              (!dump_stream_tile_start_gate || dump_stream_ok);

    wire        wb_gout_rd_en;
    wire [GOUT_AW-1:0] wb_gout_rd_addr;
    wire        wb_gout_wr_valid;
    wire [GOUT_AW-1:0] wb_gout_wr_addr;
    wire [255:0] wb_gout_wr_data;
    wire        wb_gout_wr_final_ktile;
    wire [3:0]  wb_gout_wr_layer_id;
    wire        wb_gout_wr_bank_ovr_en;
    wire        wb_gout_wr_bank_ovr;

    wire        wb_psum_rd_en;
    wire [GOUT_AW-1:0] wb_psum_rd_addr;
    wire        wb_psum_wr_valid;
    wire [GOUT_AW-1:0] wb_psum_wr_addr;
    wire [1023:0] wb_psum_wr_data;

    wire        wb_score_rd_en;
    wire        wb_score_rd_bank;
    wire [11:0] wb_score_rd_addr;
    wire        wb_score_wr_valid;
    wire        wb_score_wr_bank;
    wire [11:0] wb_score_wr_addr;
    wire [255:0] wb_score_wr_data;

    wire        wb_bias_rd_en;
    wire [6:0] wb_bias_rd_addr;

    wire        wb_attn_wr_valid;
    wire [GOUT_AW-1:0] wb_attn_wr_addr;
    wire [255:0] wb_attn_wr_data;

    wire wb_final_tile_pulse;
    wire [6:0] wb_final_mblk;
    wire [6:0] wb_final_nblk;

    // Apply CLS special handling only for Patch Embedding layer (layer_id==0)
    wire wb_cls_alt_apply;
    assign wb_cls_alt_apply = (layer_id_ctrl_q == 4'd0);

    writeback_controller #(
        .M_BLK_NUM  (M_BLK_NUM),
        .M_TOTAL    (M_TOTAL),
        .OUT_STRIDE (OUT_STRIDE),
        .GOUT_AW    (GOUT_AW),
        .SHIFT_PART (8),
        .BIAS_LSHIFT(0),
        .SCALE_W    (16),
        .SCALE_OUT  (16'sd1),
        .SHIFT_OUT  (8),
        .SM_ADAPTER_MUL_MEM(SM_ADAPTER_MUL_MEM),
        .SM_ADAPTER_SHIFT_MEM(SM_ADAPTER_SHIFT_MEM),
        .SM_H_MEM(SM_H_MEM),
        .SM_B_MEM(SM_B_MEM),
        .SM_EXP_U15_MEM(SM_EXP_U15_MEM),
        .FC1_GELU_MEM(FC1_GELU_MEM),
        .OPROJ_MUL_MEM(OPROJ_MUL_MEM),
        .FC2_MUL_MEM(FC2_MUL_MEM)
    ) u_wb (
        .clk(clk),
        .nRst(nRst),

        .layer_id(layer_id_ctrl_q),
        .enc_blk_idx(enc_blk_idx),

        .layer_final_ktile(layer_final_ktile),
        .cls_alt_apply     (wb_cls_alt_apply),
        .active_token_total(active_token_total),
        .active_mblk_num(active_mblk_num),
        .active_last_valid_lane(active_last_valid_lane),

        .ktile_idx(wb_ktile_idx),
        .nblk_idx(wb_nblk_idx),
        .mblk_idx(wb_mblk_idx),
        .col_in_blk(pe_out_col_idx),

        .final_tile_pulse(wb_final_tile_pulse),
        .final_nblk_idx  (wb_final_nblk),
        .final_mblk_idx  (wb_final_mblk),

        .pe_out_valid(pe_out_col_valid),
        .pe_out_ready(pe_out_col_ready),
        .pe_out_i32_flat(pe_out_col_data_flat),

        .gout_rd_en  (wb_gout_rd_en),
        .gout_rd_addr(wb_gout_rd_addr),
        .gout_rd_data(gout_rd_data),

        .gin_rd_en   (gin_rd_en),
        .gin_rd_addr (gin_rd_addr),
        .gin_rd_data (gin_rd_data),

        .bias_rd_en  (wb_bias_rd_en),
        .bias_rd_addr(wb_bias_rd_addr),
        .bias_rd_data(bias_rd_data),

        .quant_rd_en  (quant_rd_en),
        .quant_rd_addr(quant_rd_addr),
        .quant_rd_data(quant_rd_data),

        .gout_wr_valid(wb_gout_wr_valid),
        .gout_wr_addr (wb_gout_wr_addr),
        .gout_wr_data (wb_gout_wr_data),
        .gout_wr_final_ktile(wb_gout_wr_final_ktile),
        .gout_wr_layer_id   (wb_gout_wr_layer_id),
        .gout_wr_bank_ovr_en(wb_gout_wr_bank_ovr_en),
        .gout_wr_bank_ovr   (wb_gout_wr_bank_ovr),

        .psum_rd_en   (wb_psum_rd_en),
        .psum_rd_addr (wb_psum_rd_addr),
        .psum_rd_data (psum_rd_data),
        .psum_wr_valid(wb_psum_wr_valid),
        .psum_wr_addr (wb_psum_wr_addr),
        .psum_wr_data (wb_psum_wr_data),

        .score_rd_en  (wb_score_rd_en),
        .score_rd_bank(wb_score_rd_bank),
        .score_rd_addr(wb_score_rd_addr),
        .score_rd_data(score_rd_data),

        .score_wr_valid(wb_score_wr_valid),
        .score_wr_bank (wb_score_wr_bank),
        .score_wr_addr (wb_score_wr_addr),
        .score_wr_data (wb_score_wr_data),

        .attn_wr_valid(wb_attn_wr_valid),
        .attn_wr_addr (wb_attn_wr_addr),
        .attn_wr_data (wb_attn_wr_data),

        .wb_ready(wb_ready),
        .wb_busy (wb_busy)
    );

    // ---------------- LayerNorm scratch pass ----------------
    wire        ln_start_pulse;
    wire        ln_busy;
    wire        ln_done;

    wire        ln_gout_rd_en;
    wire [GOUT_AW-1:0] ln_gout_rd_addr;
    wire        ln_gout_wr_valid;
    wire [GOUT_AW-1:0] ln_gout_wr_addr;
    wire [255:0] ln_gout_wr_data;
    wire        ln_gout_wr_bank;

    wire        ln_score_rd_en;
    wire        ln_score_rd_bank;
    wire [11:0] ln_score_rd_addr;
    wire        ln_score_wr_valid;
    wire        ln_score_wr_bank;
    wire [11:0] ln_score_wr_addr;
    wire [255:0] ln_score_wr_data;
    wire        ln_aff_rd_en;
    wire [6:0]  ln_aff_rd_addr;
    wire [255:0] ln_aff_rd_data;
    wire [5:0]  ln_blk2 = {1'b0, enc_blk_idx, 1'b0};
    wire        ln_fc2_has_next = (enc_blk_idx != (RUN_ENC_BLK_NUM-1));
    wire        ln_fc2_norm3_tap =
        (layer_id_dump == L_MLP_FC2) &&
        ((enc_blk_idx == 4'd2) || (enc_blk_idx == 4'd5) ||
         (enc_blk_idx == 4'd8) || (enc_blk_idx == 4'd11));
    wire        ln_dual_affine_req = ln_fc2_norm3_tap && ln_fc2_has_next;
    wire        ln_norm3_only_req  = ln_fc2_norm3_tap && !ln_fc2_has_next;
    wire [5:0]  ln_norm3_param_sel =
        (enc_blk_idx == 4'd2)  ? 6'd25 :
        (enc_blk_idx == 4'd5)  ? 6'd26 :
        (enc_blk_idx == 4'd8)  ? 6'd27 : 6'd28;
    wire [5:0]  ln_next_param_sel =
        (layer_id_dump == L_PATCH)  ? 6'd0 :
        (layer_id_dump == L_O_PROJ) ? (ln_blk2 + 6'd1) :
                                      (ln_blk2 + 6'd2);
    wire [5:0]  ln_param_sel = ln_norm3_only_req ?
                                    ln_norm3_param_sel : ln_next_param_sel;
    wire [5:0]  ln_secondary_param_sel = ln_norm3_param_sel;
    wire        ln_primary_out_bank = ln_norm3_only_req ? 1'b0 : 1'b1;

    layernorm_scratch_engine #(
        .MEM_WIDTH       (M_AXI_DATA_WIDTH),
        .EMBED_DIM       (OUT_STRIDE),
        .OUT_STRIDE      (OUT_STRIDE),
        .M_BLK_NUM       (M_BLK_NUM),
        .GOUT_AW         (GOUT_AW),
        .VALID_LANES_LAST(TOKEN_LAST_VALID),
        .SCORE_BANK_DEPTH(256),
        .APPLY_LAYER_NORM(ENABLE_LAYER_NORM),
        .LN_AFFINE_MODE  (LN_AFFINE_MODE),
        .LN_PARAM_SETS   (LN_PARAM_SETS),
        .LN_GAIN_MEM     (LN_GAIN_MEM),
        .LN_BIAS_MEM     (LN_BIAS_MEM),
        .LN_RECIP_MEM    (LN_RECIP_MEM),
        .LN_EPSILON_MEM  (LN_EPSILON_MEM)
    ) u_ln_scratch (
        .clk(clk),
        .nRst(nRst),

        .start(ln_start_pulse),
        .ln_param_sel(ln_param_sel),
        .dual_affine(ln_dual_affine_req),
        .primary_out_bank(ln_primary_out_bank),
        .secondary_out_bank(1'b0),
        .active_mblk_num(active_mblk_num),
        .active_last_valid_lane(active_last_valid_lane),
        .busy (ln_busy),
        .done (ln_done),

        .gout_rd_en  (ln_gout_rd_en),
        .gout_rd_addr(ln_gout_rd_addr),
        .gout_rd_data(gout_rd_data),

        .gout_wr_valid(ln_gout_wr_valid),
        .gout_wr_addr (ln_gout_wr_addr),
        .gout_wr_data (ln_gout_wr_data),
        .gout_wr_bank (ln_gout_wr_bank),

        .score_rd_en  (ln_score_rd_en),
        .score_rd_bank(ln_score_rd_bank),
        .score_rd_addr(ln_score_rd_addr),
        .score_rd_data(score_rd_data),

        .score_wr_valid(ln_score_wr_valid),
        .score_wr_bank (ln_score_wr_bank),
        .score_wr_addr (ln_score_wr_addr),
        .score_wr_data (ln_score_wr_data),

        .aff_rd_en  (ln_aff_rd_en),
        .aff_rd_addr(ln_aff_rd_addr),
        .aff_rd_data(ln_aff_rd_data),

        .dbg_mblk(),
        .dbg_state()
    );

    // ---------------- dump path: GOUT read mux (WB has priority) ----------------
    wire        dump_gout_rd_en;
    wire [GOUT_AW-1:0] dump_gout_rd_addr;

    // Post-compute GB1 dump state. OPROJ uses this pass to write add1 to DDR
    // before the in-place norm2 pass, avoiding write-stream backpressure in
    // the OPROJ PE schedule. The same walker remains available to FC2.
    reg        post_fc2_dump_active;
    reg        post_fc2_dump_pending;
    reg [6:0]  post_fc2_mb;
    reg [6:0]  post_fc2_nb;
    reg        post_fc2_issue_pulse;
    reg        post_ln_fc2_dump_pending;
    reg        pre_ln_oproj_dump_pending;
    reg        post_fc2_dump_is_norm3;

    assign gout_rd_en   = ln_busy ? ln_gout_rd_en :
                          (wb_gout_rd_en ? 1'b1 : dump_gout_rd_en);
    assign gout_rd_addr = ln_busy ? ln_gout_rd_addr :
                          (wb_gout_rd_en ? wb_gout_rd_addr : dump_gout_rd_addr);

    assign gout_wr_valid = ln_busy ? ln_gout_wr_valid : wb_gout_wr_valid;
    assign gout_wr_addr  = ln_busy ? ln_gout_wr_addr : wb_gout_wr_addr;
    assign gout_wr_data  = ln_busy ? ln_gout_wr_data  : wb_gout_wr_data;
    // OPROJ add1 is held in GB0 while AV remains in GB1. FC2 add2 is held in
    // GB1. The completed tensor is dumped to DDR before LayerNorm starts;
    // LayerNorm writes the normalized activation to GB1 for the next layer.
    reg layer_state_rd_bank_q;

    // The state-bank role is fixed for the entire layer and its end-service
    // window. Capture it at layer start: layer_end_pulse is observed one cycle
    // after vit_controller may advance layer_id, which can otherwise select
    // the next layer's bank for the current layer's dump/LayerNorm pass.
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            layer_state_rd_bank_q <= 1'b0;
        end else if (layer_start_pulse) begin
            layer_state_rd_bank_q <= (layer_id == L_MLP_FC2);
        end
    end

    assign gout_rd_bank_ovr_en = ln_busy || post_fc2_dump_active;
    assign gout_rd_bank_ovr    = post_fc2_dump_active && post_fc2_dump_is_norm3 ?
                                 1'b0 : layer_state_rd_bank_q;
    assign gout_wr_bank_ovr_en = ln_busy ? 1'b1 : wb_gout_wr_bank_ovr_en;
    assign gout_wr_bank_ovr    = ln_busy ? ln_gout_wr_bank : wb_gout_wr_bank_ovr;

    assign psum_rd_en    = wb_psum_rd_en;
    assign psum_rd_addr  = wb_psum_rd_addr;
    assign psum_wr_valid = wb_psum_wr_valid;
    assign psum_wr_addr  = wb_psum_wr_addr;
    assign psum_wr_data  = wb_psum_wr_data;

    assign score_rd_en   = ln_busy ? ln_score_rd_en   : wb_score_rd_en;
    assign score_rd_bank = ln_busy ? ln_score_rd_bank : wb_score_rd_bank;
    assign score_rd_addr = ln_busy ? ln_score_rd_addr : wb_score_rd_addr;

    assign score_wr_valid = ln_busy ? ln_score_wr_valid : wb_score_wr_valid;
    assign score_wr_bank  = ln_busy ? ln_score_wr_bank  : wb_score_wr_bank;
    assign score_wr_addr  = ln_busy ? ln_score_wr_addr  : wb_score_wr_addr;
    assign score_wr_data  = ln_busy ? ln_score_wr_data  : wb_score_wr_data;

    assign bias_rd_en    = ln_busy ? ln_aff_rd_en   : wb_bias_rd_en;
    assign bias_rd_addr  = ln_busy ? ln_aff_rd_addr : wb_bias_rd_addr;
    assign ln_aff_rd_data = bias_rd_data;

    assign attn_wr_valid = wb_attn_wr_valid;
    assign attn_wr_addr  = wb_attn_wr_addr;
    assign attn_wr_data  = wb_attn_wr_data;

    // AXI_Master WLAST ? 0 ? 
    assign m_axi_wlast = 1'b0;

    // ---------------- write_controller (dump) ----------------
    wire dump_done_pulse;
    wire [6:0] wc_cur_mb;
    wire [6:0] wc_cur_nb;

    // ---------------- write_controller (dump) ----------------
    // NOTE: For correctness, the tile's DDR destination descriptor is computed here
    //       and latched *per request* inside write_controller.
    // dump layer selection
    // - debug dump layers are controlled by DUMP_ENABLE
    // - final output (MLP FC2) writeback can be enabled even when DUMP_ENABLE=0
    wire debug_dump_layer_match = (layer_id_dump == L_PATCH) ||
                       ((RUN_QKV != 0) && (layer_id_dump == L_QKV_Q)) ||
                       ((RUN_QKV >= 2) && (layer_id_dump == L_QKV_V));

    wire is_cls_dump = ((layer_id_dump >= L_CLS0) && (layer_id_dump <= L_CLS5));
    wire final_out_layer_match = (layer_id_dump == L_MLP_FC2) || is_cls_dump;
    wire final_out_lastpass     = (wb_ktile_idx == layer_final_ktile);

    wire debug_dump_en = (DUMP_ENABLE != 0) && debug_dump_layer_match;
    // FC2 add2 is streamed directly. OPROJ add1 is dumped from GB1 only after
    // OPROJ compute completes, so DDR backpressure cannot perturb the PE loop.
    wire fc2_state_dump_en = (layer_id_dump == L_MLP_FC2) && final_out_lastpass;
    wire cls_final_out_en = (FINAL_OUT_WRITE_ENABLE != 0) && is_cls_dump && final_out_lastpass;
    wire final_out_en = fc2_state_dump_en || cls_final_out_en;
    wire is_fc2_final_out_layer = final_out_en; // (layer_id_dump == L_MLP_FC2) && FINAL_OUT_WRITE_ENABLE
    wire fc2_final_needs_post_dump = 1'b0;

    wire debug_dump_tile_start = debug_dump_en && wb_final_tile_pulse;
    wire final_out_tile_start  = final_out_en && wb_final_tile_pulse;
    wire post_fc2_tile_start   = post_fc2_issue_pulse;

    assign dump_layer_match = debug_dump_en || final_out_en;
    wire dump_tile_start  = debug_dump_tile_start || final_out_tile_start ||
                            post_fc2_tile_start;

    wire [6:0] dump_req_mblk = post_fc2_tile_start ? post_fc2_mb : wb_final_mblk;
    wire [6:0] dump_req_nblk = post_fc2_tile_start ? post_fc2_nb : wb_final_nblk;

    // this tile uses stream source iff it is FC2 final output tile
    wire dump_tile_src_stream = debug_dump_tile_start || final_out_tile_start;

    wire [6:0] post_fc2_mb_last = active_mblk_last;
    wire [6:0] post_fc2_nb_last = N_BLK_NUM[6:0] - 7'd1;
// destination mapping per layer
    reg  [1:0]              dump_dst_sel;
    reg  [BASE_AW-1:0]      dump_dst_off_word;
    reg  [11:0]             dump_dst_stride_words;

    always @(*) begin
        dump_dst_sel          = DST_SEL_OUT;
        dump_dst_off_word     = 0;
        dump_dst_stride_words = STRIDE_384;

        if (post_fc2_tile_start && post_fc2_dump_is_norm3) begin
            dump_dst_sel          = DST_SEL_OUT;
            dump_dst_stride_words = STRIDE_384;
            case (enc_blk_idx)
                4'd2:    dump_dst_off_word = OFF_OUT_NORM3_B3;
                4'd5:    dump_dst_off_word = OFF_OUT_NORM3_B6;
                4'd8:    dump_dst_off_word = OFF_OUT_NORM3_B9;
                default: dump_dst_off_word = OFF_OUT_NORM3_B12;
            endcase
        end else case (layer_id_dump)
            L_PATCH: begin
                // Patch Embedding output -> OUT region (residual read later)
                dump_dst_sel          = DST_SEL_OUT;
                dump_dst_off_word     = OFF_OUT_PATCH_OUT;
                dump_dst_stride_words = STRIDE_384;
            end

            L_QKV_Q: begin
                // Q projection output -> W region (Q_OUT scratch inside encoder block)
                dump_dst_sel          = DST_SEL_W;
                dump_dst_off_word     = OFF_W_Q_OUT + (enc_blk_idx * SIZE_W_ENC_BLK_WGT);
                dump_dst_stride_words = STRIDE_384;
            end


            L_QKV_V: begin
                // V projection output -> W region (V_OUT scratch inside encoder block)
                dump_dst_sel          = DST_SEL_W;
                dump_dst_off_word     = OFF_W_V_OUT + (enc_blk_idx * SIZE_W_ENC_BLK_WGT);
                dump_dst_stride_words = STRIDE_197;
            end
            L_O_PROJ: begin
                dump_dst_sel          = DST_SEL_OUT;
                dump_dst_off_word     = OFF_OUT_O_OUT + (enc_blk_idx * SIZE_OUT_PATCH_OUT);
                dump_dst_stride_words = STRIDE_384;
            end
            L_MLP_FC2: begin
                dump_dst_sel          = DST_SEL_OUT;
                dump_dst_off_word     = OFF_OUT_O_OUT + (enc_blk_idx * SIZE_OUT_PATCH_OUT);  // bring-up: reuse OPROJ region
                dump_dst_stride_words = STRIDE_384;
            end

            
                    L_CLS0: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS0_OUT; dump_dst_stride_words = STRIDE_192; end
                    L_CLS1: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS1_OUT; dump_dst_stride_words = STRIDE_192; end
                    L_CLS2: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS2_OUT; dump_dst_stride_words = STRIDE_192; end
                    L_CLS3: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS3_OUT; dump_dst_stride_words = STRIDE_192; end
                    L_CLS4: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS4_OUT; dump_dst_stride_words = STRIDE_192; end
                    L_CLS5: begin dump_dst_sel = DST_SEL_OUT; dump_dst_off_word = OFF_OUT_CLS5_OUT; dump_dst_stride_words = STRIDE_64;  end

                    default: begin
                dump_dst_sel          = DST_SEL_OUT;
                dump_dst_off_word     = 0;
                dump_dst_stride_words = STRIDE_384;
            end
        endcase
    end

    wire [BASE_AW-1:0] dump_dst_base_word_sel =
        (dump_dst_sel == DST_SEL_IN)  ? base_in_word :
        (dump_dst_sel == DST_SEL_W)   ? base_w_word  :
                                        base_out_word;

    wire [BASE_AW-1:0] dump_tile_dst_base_word = dump_dst_base_word_sel + dump_dst_off_word;

    // compute per-tile word count (supports long token strides such as 1873)
    wire [6:0] dump_nblk_num =
        (layer_id_dump == L_QKV_V) ? active_mblk_num :
                                     f_nblk_num(dump_dst_stride_words);
    wire [6:0] dump_last_nblk_idx = dump_nblk_num - 7'd1;
    wire       dump_is_last_nblk  = (dump_req_nblk == dump_last_nblk_idx);
    wire [5:0] dump_tile_words    =
        dump_is_last_nblk ?
        ((layer_id_dump == L_QKV_V) ? active_last_tile_words :
                                      f_last_tile_cols(dump_dst_stride_words)) :
        6'd32;

    // Stream push into write_controller internal DATA FIFO.
    // Use data-aligned writeback metadata; controller ktile/layer indices advance
    // before the registered wb_gout_wr_data reaches this point.
    wire wb_gout_wr_debug_dump_layer_match =
        (wb_gout_wr_layer_id == L_PATCH) ||
        ((RUN_QKV != 0) && (wb_gout_wr_layer_id == L_QKV_Q)) ||
        ((RUN_QKV >= 2) && (wb_gout_wr_layer_id == L_QKV_V));
    wire wb_gout_wr_is_cls_dump =
        ((wb_gout_wr_layer_id >= L_CLS0) && (wb_gout_wr_layer_id <= L_CLS5));
    wire wb_gout_wr_final_out_layer_match =
        (wb_gout_wr_layer_id == L_MLP_FC2) || wb_gout_wr_is_cls_dump;
    wire dump_stream_data_pass =
        wb_gout_wr_final_ktile &&
        (((DUMP_ENABLE != 0) && wb_gout_wr_debug_dump_layer_match) ||
         ((FINAL_OUT_WRITE_ENABLE != 0) && wb_gout_wr_final_out_layer_match &&
          (wb_gout_wr_layer_id != L_MLP_FC2)) ||
         (wb_gout_wr_layer_id == L_MLP_FC2));
    wire dump_stream_in_valid = wb_gout_wr_valid && dump_stream_data_pass;
    wire [255:0] dump_stream_in_data = wb_gout_wr_data;
    wire dump_stream_in_ready;

    // ---------------- write_controller instance ----------------
    write_controller #(
        .M_AXI_ADDR_WIDTH(M_AXI_ADDR_WIDTH),
        .DATA_W          (M_AXI_DATA_WIDTH),
        .ADDR_LSB        (ADDR_LSB),
        .M_BLK_NUM       (M_BLK_NUM),
        .GOUT_STRIDE     (OUT_STRIDE),
        .GOUT_AW         (GOUT_AW),
        .REQ_FIFO_HEADROOM  (16),
        .DATA_FIFO_HEADROOM (48)
    ) u_dump (
        .clk(clk),
        .nRst(nRst),

        .tile_start_pulse     (dump_tile_start),
        .tile_mblk_idx        (dump_req_mblk),
        .tile_nblk_idx        (dump_req_nblk),
        .tile_dst_base_word   (dump_tile_dst_base_word),
        .tile_dst_stride_words(dump_dst_stride_words),
        .tile_dst_words       (dump_tile_words),

        .tile_src_stream     (dump_tile_src_stream),
        .stream_in_valid     (dump_stream_in_valid),
        .stream_in_data      (dump_stream_in_data),
        .stream_in_ready     (dump_stream_in_ready),
        .stream_ok           (dump_stream_ok),

        .wb_gout_rd_en   (wb_gout_rd_en),

        .m_axi_write_done  (m_axi_write_done),
        .m_axi_write_init  (m_axi_write_init),
        .m_axi_awaddress   (m_axi_awaddress),
        .m_axi_write_length(m_axi_write_length),

        .S_WDATA  (m_axi_wdata),
        .S_WVALID (m_axi_wvalid),
        .S_WREADY (m_axi_wready),

        .gout_rd_en  (dump_gout_rd_en),
        .gout_rd_addr(dump_gout_rd_addr),
        .gout_rd_data(gout_rd_data),

        .dump_done_pulse(dump_done_pulse),

        .cur_mblk_idx_o(wc_cur_mb),
        .cur_nblk_idx_o(wc_cur_nb)
    );


    // ---------------- done generation ----------------
    localparam integer DUMP_TILES_PER_LAYER = (M_BLK_NUM * N_BLK_NUM);
    localparam integer DUMP_EXPECTED_TILES =
        (RUN_QKV == 0) ? (1*DUMP_TILES_PER_LAYER) :
        (RUN_QKV == 1) ? (2*DUMP_TILES_PER_LAYER) :
                         (3*DUMP_TILES_PER_LAYER);

    reg start_d1;
    wire start_pulse = start && !start_d1;

    reg [13:0] dump_done_cnt;
    reg       vit_done_latched;
    reg       final_out_last_tile_done; // asserted when LAST encoder-block FC2 last tile is written to DDR

    // --- per-layer wait for DDR dump completion (prevents early layer advance) ---
    reg [13:0] dump_done_cnt_layer;
    reg [13:0] dump_expected_layer;
    reg       wait_layer_dump;

    wire       layer_needs_dump         = dump_layer_match;
    wire [6:0] dump_mblk_num = (layer_id_dump == L_QKV_V) ? N_BLK_NUM[6:0] : (is_cls_dump ? 7'd1 : active_mblk_num);
    wire [13:0] layer_expected_tiles_calc = dump_mblk_num * dump_nblk_num;

    wire layer_needs_ln_calc =
        (ENABLE_LAYER_NORM != 0) &&
        ((layer_id_dump == L_PATCH) ||
         (layer_id_dump == L_O_PROJ) ||
         ((layer_id_dump == L_MLP_FC2) &&
          (ktile_idx_cur == layer_final_ktile) &&
          (ln_fc2_has_next || ln_fc2_norm3_tap)));
    wire layer_defer_dump_until_ln = 1'b0;
    wire layer_defer_fc2_dump = 1'b0;

    reg ln_pending;
    reg ln_done_seen;
    reg ln_aff_loaded;
    reg ln_aff_load_busy;
    reg ln_aff_read_init;
    reg [M_AXI_ADDR_WIDTH-1:0] ln_aff_araddr;
    reg [16:0] ln_aff_read_len;
    reg [6:0] ln_aff_beat_idx;
    reg       ln_aff_load_secondary;
    reg ln_aff_mem_wr_valid;
    reg [GOUT_AW-1:0] ln_aff_mem_wr_addr;
    reg [M_AXI_DATA_WIDTH-1:0] ln_aff_mem_wr_data;

    wire layer_dump_ready = wait_layer_dump &&
                            (dump_done_cnt_layer >= dump_expected_layer) &&
                            !wb_busy;

    wire ln_affine_buffer_mode = (LN_AFFINE_MODE == 2);
    wire [BASE_AW-1:0] ln_aff_base_offset_word = LN_AFFINE_BASE_OFFSET_WORD;
    wire [5:0] ln_aff_load_param_sel = ln_aff_load_secondary ?
                                          ln_secondary_param_sel : ln_param_sel;
    wire [BASE_AW-1:0] ln_aff_set_offset_word  =
        ln_aff_load_param_sel * LN_AFFINE_SET_WORDS;
    wire [BASE_AW-1:0] ln_aff_start_word       =
        base_b_word + ln_aff_base_offset_word + ln_aff_set_offset_word;
    wire post_state_dump_idle = !pre_ln_oproj_dump_pending &&
                                !post_fc2_dump_active &&
                                !post_fc2_dump_pending;
    wire ln_aff_load_start =
        ln_affine_buffer_mode && layer_dump_ready && ln_pending &&
        post_state_dump_idle &&
        // The same AXI master is used by the next-layer weight prefetch.
        // Wait for that read to complete before issuing the affine read;
        // otherwise a new init pulse aborts the in-flight preload.
        (nl_pf_valid || ln_norm3_only_req) &&
        !ln_busy && !ln_aff_loaded && !ln_aff_load_busy;
    wire ln_aff_read_mux_active = ln_aff_load_busy || ln_aff_read_init;

    assign ln_start_pulse = layer_dump_ready && ln_pending && !ln_busy &&
                            post_state_dump_idle &&
                            (!ln_affine_buffer_mode || ln_aff_loaded) &&
                            !ln_aff_load_busy;

    assign m_axi_read_init   = ln_aff_read_mux_active ? ln_aff_read_init   : rd_m_axi_read_init;
    assign m_axi_araddress   = ln_aff_read_mux_active ? ln_aff_araddr      : rd_m_axi_araddress;
    assign m_axi_read_length = ln_aff_read_mux_active ? ln_aff_read_len    : rd_m_axi_read_length;

    assign mem_wr_valid = rd_mem_wr_valid;
    assign mem_wr_kind  = rd_mem_wr_kind;
    assign mem_wr_bank  = rd_mem_wr_bank;
    assign mem_wr_addr  = rd_mem_wr_addr;
    assign mem_wr_data  = rd_mem_wr_data;

    assign ln_aff_wr_valid = ln_aff_mem_wr_valid;
    assign ln_aff_wr_addr  = ln_aff_mem_wr_addr[6:0];
    assign ln_aff_wr_data  = ln_aff_mem_wr_data;

    assign layer_adv_grant = layer_dump_ready &&
                             !ln_pending &&
                             !ln_busy &&
                             ln_done_seen &&
                             !pre_ln_oproj_dump_pending &&
                             !post_ln_fc2_dump_pending &&
                             !post_fc2_dump_active &&
                             !post_fc2_dump_pending;

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            start_d1        <= 0;
            dump_done_cnt   <= 0;
            vit_done_latched<= 0;
            final_out_last_tile_done <= 0;

            dump_done_cnt_layer <= 0;
            dump_expected_layer <= 0;
            wait_layer_dump     <= 0;
            ln_pending          <= 1'b0;
            ln_done_seen        <= 1'b1;
            ln_aff_loaded       <= 1'b1;
            ln_aff_load_busy    <= 1'b0;
            ln_aff_read_init    <= 1'b0;
            ln_aff_araddr       <= {M_AXI_ADDR_WIDTH{1'b0}};
            ln_aff_read_len     <= 17'd0;
            ln_aff_beat_idx     <= 7'd0;
            ln_aff_load_secondary <= 1'b0;
            ln_aff_mem_wr_valid <= 1'b0;
            ln_aff_mem_wr_addr  <= {GOUT_AW{1'b0}};
            ln_aff_mem_wr_data  <= {M_AXI_DATA_WIDTH{1'b0}};

            post_fc2_dump_active  <= 0;
            post_fc2_dump_pending <= 0;
            post_fc2_mb           <= 0;
            post_fc2_nb           <= 0;
            post_fc2_issue_pulse  <= 0;
            post_ln_fc2_dump_pending <= 0;
            pre_ln_oproj_dump_pending <= 0;
            post_fc2_dump_is_norm3 <= 1'b0;
            layer_id_pe_q         <= 0;
            layer_head_idx_pe_q   <= 0;
            layer_id_ctrl_q       <= 0;
            base_q_word_rd_q      <= 0;
        end else begin
            start_d1 <= start;
            ln_aff_read_init    <= 1'b0;
            ln_aff_mem_wr_valid <= 1'b0;

            if (layer_start_pulse) begin
                layer_id_pe_q       <= layer_id;
                layer_head_idx_pe_q <= layer_head_idx;
                layer_id_ctrl_q     <= layer_id;
                base_q_word_rd_q    <= base_q_word_eff;
            end

            post_fc2_issue_pulse <= 1'b0;
            if (start_pulse) begin
                dump_done_cnt    <= 0;
                vit_done_latched <= 0;
            final_out_last_tile_done <= 0;

                dump_done_cnt_layer <= 0;
                dump_expected_layer <= 0;
                wait_layer_dump     <= 0;
                ln_pending          <= 1'b0;
                ln_done_seen        <= 1'b1;
                ln_aff_loaded       <= 1'b1;
                ln_aff_load_busy    <= 1'b0;
                ln_aff_read_init    <= 1'b0;
                ln_aff_beat_idx     <= 7'd0;
                ln_aff_load_secondary <= 1'b0;
                ln_aff_mem_wr_valid <= 1'b0;

                post_fc2_dump_active  <= 0;
                post_fc2_dump_pending <= 0;
                post_fc2_mb           <= 0;
                post_fc2_nb           <= 0;
                post_fc2_issue_pulse  <= 0;
                post_ln_fc2_dump_pending <= 0;
                pre_ln_oproj_dump_pending <= 0;
                post_fc2_dump_is_norm3 <= 1'b0;
            end else begin
                if (dump_done_pulse) begin
                    // Latch final done when the LAST encoder block's FC2 last (mblk,nblk) tile write completes
                    if (final_out_en && (enc_blk_idx == (RUN_ENC_BLK_NUM-1)) &&
                        (wc_cur_mb == active_mblk_last) && (wc_cur_nb == (N_BLK_NUM[6:0]-7'd1))) begin
                        final_out_last_tile_done <= 1'b1;
                    end
                    dump_done_cnt <= dump_done_cnt + 14'd1;
                end

                // per-layer dump counter (reset on each layer start)
                if (layer_start_pulse) begin
                    dump_done_cnt_layer <= 0;
                end else if (dump_done_pulse) begin
                    dump_done_cnt_layer <= dump_done_cnt_layer + 14'd1;
                end

                // start waiting when vit_controller reports end-of-layer compute
                if (layer_end_pulse) begin
                    dump_expected_layer <= (layer_needs_dump && !layer_defer_fc2_dump) ?
                                           layer_expected_tiles_calc : 14'd0;
                    wait_layer_dump     <= 1'b1;
                    ln_pending          <= layer_needs_ln_calc;
                    ln_done_seen        <= !layer_needs_ln_calc;
                    ln_aff_loaded       <= !layer_needs_ln_calc || !ln_affine_buffer_mode;
                    ln_aff_load_secondary <= 1'b0;
                    post_ln_fc2_dump_pending <= layer_needs_ln_calc && ln_fc2_norm3_tap;
                    pre_ln_oproj_dump_pending <= (layer_id_dump == L_O_PROJ);
                end else if (ln_start_pulse) begin
                    ln_pending          <= 1'b0;
                    ln_done_seen        <= 1'b0;
                    ln_aff_loaded       <= 1'b0;
                end else if (ln_done) begin
                    ln_done_seen        <= 1'b1;
                    if (post_ln_fc2_dump_pending) begin
                        dump_done_cnt_layer <= 14'd0;
                        dump_expected_layer <= layer_expected_tiles_calc;
                        post_fc2_dump_active  <= 1'b1;
                        post_fc2_dump_pending <= 1'b1;
                        post_fc2_mb           <= 0;
                        post_fc2_nb           <= 0;
                        post_ln_fc2_dump_pending <= 1'b0;
                        post_fc2_dump_is_norm3 <= 1'b1;
                    end
                end else if (layer_adv_grant) begin
                    wait_layer_dump     <= 1'b0;
                    ln_pending          <= 1'b0;
                    ln_done_seen        <= 1'b1;
                    post_ln_fc2_dump_pending <= 1'b0;
                    pre_ln_oproj_dump_pending <= 1'b0;
                    post_fc2_dump_is_norm3 <= 1'b0;
                end

                // OPROJ add1 is already complete in GB1. Walk GB1 and commit
                // the tensor to its DDR block-output slot before norm2 starts.
                if (pre_ln_oproj_dump_pending && layer_dump_ready &&
                    !post_fc2_dump_active) begin
                    dump_done_cnt_layer <= 14'd0;
                    dump_expected_layer <= layer_expected_tiles_calc;
                    post_fc2_dump_active  <= 1'b1;
                    post_fc2_dump_pending <= 1'b1;
                    post_fc2_mb           <= 0;
                    post_fc2_nb           <= 0;
                    pre_ln_oproj_dump_pending <= 1'b0;
                    post_fc2_dump_is_norm3 <= 1'b0;
                end

                if (ln_aff_load_start) begin
                    ln_aff_load_busy <= 1'b1;
                    ln_aff_read_init <= 1'b1;
                    ln_aff_araddr[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= ln_aff_start_word;
                    ln_aff_araddr[ADDR_LSB-1:0] <= {ADDR_LSB{1'b0}};
                    ln_aff_read_len <= LN_AFFINE_SET_WORDS;
                    ln_aff_beat_idx <= 7'd0;
                    ln_aff_loaded   <= 1'b0;
                end else if (ln_aff_load_busy) begin
                    if (axi_rvalid && axi_rready) begin
                        ln_aff_mem_wr_valid <= 1'b1;
                        ln_aff_mem_wr_addr  <=
                            (ln_aff_load_secondary ? LN_AFFINE_SET_WORDS : 0) +
                            ln_aff_beat_idx;
                        ln_aff_mem_wr_data  <= axi_rdata;
                        ln_aff_beat_idx     <= ln_aff_beat_idx + 7'd1;
                    end
                    if (m_axi_read_done) begin
                        ln_aff_load_busy <= 1'b0;
                        if (ln_dual_affine_req && !ln_aff_load_secondary) begin
                            // Load norm3 into addresses 60..119 after the
                            // next-block norm1 set at addresses 0..59.
                            ln_aff_load_secondary <= 1'b1;
                            ln_aff_loaded <= 1'b0;
                        end else begin
                            ln_aff_load_secondary <= 1'b0;
                            ln_aff_loaded <= 1'b1;
                        end
                    end
                end

                // ---------------- Post-dump pass for final output (FC2) ----------------
                // After a norm3 tap completes, stream normalized GB0 tiles to
                // the dedicated DPT feature slot. OPROJ retains the legacy
                // pre-LayerNorm dump behavior through the same walker.
                if (post_fc2_dump_active) begin
                    // issue next tile when pending and WB is idle
                    if (post_fc2_dump_pending) begin
                        if (!wb_busy) begin
                            post_fc2_issue_pulse  <= 1'b1; // 1-cycle kick to write_controller
                            post_fc2_dump_pending <= 1'b0;
                        end
                    end else begin
                        // wait for write_controller to finish the issued tile
                        if (dump_done_pulse) begin
                            if ((post_fc2_mb == post_fc2_mb_last) && (post_fc2_nb == post_fc2_nb_last)) begin
                                post_fc2_dump_active <= 1'b0;
                                post_fc2_dump_is_norm3 <= 1'b0;
                            end else begin
                                // advance (nblk major)
                                if (post_fc2_nb == post_fc2_nb_last) begin
                                    post_fc2_nb <= 0;
                                    post_fc2_mb <= post_fc2_mb + 7'd1;
                                end else begin
                                    post_fc2_nb <= post_fc2_nb + 7'd1;
                                end
                                post_fc2_dump_pending <= 1'b1;
                            end
                        end
                    end
                end


                if (vit_done) begin
                    vit_done_latched <= 1'b1;
                end
            end
        end
    end

    assign dbg_status0 = {
        sched_mblk_idx_cur[4:0], // [31:27]
        sched_nblk_idx_cur,      // [26:20]
        ktile_idx_cur,           // [19:14]
        layer_head_idx,          // [13:11]
        enc_blk_idx,             // [10:7]
        dbg_fsm,                 // [6:4]
        layer_id                 // [3:0]
    };

    assign dbg_status1 = {
        4'd0,                    // [31:28]
        post_fc2_mb,             // [27:21]
        post_fc2_nb,             // [20:14]
        pe_ready,                // [13]
        wb_ready,                // [12]
        wb_busy,                 // [11]
        post_fc2_dump_pending,   // [10]
        post_fc2_dump_active,    // [9]
        post_ln_fc2_dump_pending,// [8]
        ln_done_seen,            // [7]
        ln_done,                 // [6]
        ln_busy,                 // [5]
        ln_pending,              // [4]
        wait_layer_dump,         // [3]
        layer_adv_grant,         // [2]
        layer_end_pulse,         // [1]
        layer_active             // [0]
    };

    assign dbg_status2 = {
        dump_done_cnt_layer,     // [31:18]
        dump_expected_layer,     // [17:4]
        dump_done_cnt[3:0]       // [3:0]
    };

    assign dbg_status3 = {
        m_axi_read_init,         // [31]
        m_axi_read_done,         // [30]
        m_axi_write_init,        // [29]
        m_axi_write_done,        // [28]
        wc_cur_mb,               // [27:21]
        wc_cur_nb,               // [20:14]
        wb_mblk_idx,             // [13:7]
        wb_nblk_idx              // [6:0]
    };

    assign dbg_status4 = {
        4'd0,                    // [31:28]
        sched_mblk_idx_cur,      // [27:21]
        sched_nblk_idx_cur,      // [20:14]
        wc_cur_mb,               // [13:7]
        wc_cur_nb                // [6:0]
    };

    wire [7:0] dump_expected = DUMP_EXPECTED_TILES[7:0];
    wire dump_all_done = vit_done_latched; // vit_done already implies all per-layer dumps completed via layer_adv_grant handshake

    wire vit_done_pulse = vit_done && !vit_done_latched; // 1-cycle pulse at end of run
    assign done = vit_done_pulse;

endmodule

`default_nettype wire
