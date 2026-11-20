`timescale 1ns/1ps
`default_nettype none

module read_controller #(
    parameter integer M_AXI_DATA_WIDTH = 256,
    parameter integer M_AXI_ADDR_WIDTH = 32,
    parameter integer ADDR_LSB = $clog2(M_AXI_DATA_WIDTH/8),

    parameter integer M_TOTAL   = 1873,
    parameter integer N_TOTAL   = 384,

    parameter integer M_BLK_NUM = 59,
    parameter integer N_BLK_NUM = 12,
    parameter integer MAX_KTILE = 384
)(
    input  wire clk,
    input  wire nRst,

    // from vit_controller
    input  wire       layer_active,
    input  wire       layer_start_pulse,   // 1-cycle
    input  wire [5:0] ktile_idx_cur,

    input  wire [3:0] layer_id,
    input  wire [5:0] layer_final_ktile,
    input  wire       layer_inbuf_sel,
    input  wire [2:0] layer_head_idx,


    input  wire       layer_wgt_phase,
    input  wire [6:0] mblk_idx_cur,
    input  wire [6:0] nblk_idx_cur,
    input  wire [11:0] active_token_total,
    input  wire [6:0]  active_mblk_num,
    input  wire [M_BLK_NUM-1:0] mblk_release_pulse,  // patch?????? ???(???? ktile overwrite)
    input  wire [1:0] wgt_slot_release_pulse,


        // next-layer W0 prefetch (from vit_controller)
        input  wire        nl_wgt0_prefetch_pulse,
        input  wire [3:0]  nl_wgt0_layer_id,
        input  wire [5:0]  nl_wgt0_ktile,
        input  wire [6:0]  nl_wgt0_nblk,
        input  wire        nl_wgt0_bank,
        input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] nl_wgt0_base_w_word,

    // per-layer base addresses (word index)
    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_in_word,
    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_w_word,
    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_b_word,
    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_out_word,

    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_q_word,
    // AXI read interface
    input  wire                        m_axi_read_done,
    output reg                         m_axi_read_init,     // 1-cycle pulse
    output reg  [M_AXI_ADDR_WIDTH-1:0] m_axi_araddress,
    output reg  [16:0]                 m_axi_read_length,

    // ready outputs to vit_controller
    output reg                         bias_ready,
    output reg  quant_ready,
    output reg  [M_BLK_NUM-1:0]        in_ready_cur,
    output reg  [11:0]                  w_ready_cur,
    output reg  [11:0]  res_ready_cur,

    output reg  [6:0] res_upto0,
    output reg  [6:0] res_upto1,
    output reg  [6:0] res_upto2,
    output reg  [6:0] res_upto3,
    output reg  [6:0] res_upto4,
    output reg  [6:0] res_upto5,
    output reg  [6:0] res_upto6,
    output reg  [6:0] res_upto7,
    output reg  [6:0] res_upto8,
    output reg  [6:0] res_upto9,
    output reg  [6:0] res_upto10,
    output reg  [6:0] res_upto11,
    // next-layer prefetch status (to vit_controller)
    output wire                        nl_pf_valid,
    output wire [3:0]                  nl_pf_layer,
    output wire [5:0]                  nl_pf_ktile,
    output wire [6:0]                  nl_pf_nblk,
    output wire                        nl_pf_bank,

    // AXI R-channel tap
    input  wire                        axi_rvalid,
    input  wire [M_AXI_DATA_WIDTH-1:0] axi_rdata,
    input  wire                        axi_rlast,
    input  wire                        axi_rready,

    // DDR -> SRAM write port
    output reg                         mem_wr_valid,
    output reg  [2:0]                  mem_wr_kind,   // 1:IN 2:WGT 3:BIAS 4:SCOREBUF(residual)
    output reg                         mem_wr_bank,
    output reg  [17:0]                 mem_wr_addr,
    output reg  [M_AXI_DATA_WIDTH-1:0] mem_wr_data
);
    `undef ADDR_MAP_VH
        `include "addr_map.vh"

    // layer IDs (vit_controller?? ????)
    localparam [3:0] L_PATCH = 4'd0;

localparam [3:0] L_SCORE = 4'd4;
    localparam [3:0] L_AV    = 4'd5;
        localparam [3:0] L_O_PROJ = 4'd6;
    localparam [3:0] L_MLP_FC1 = 4'd7;
    localparam [3:0] L_MLP_FC2 = 4'd8;
    localparam [3:0] L_CLS0    = 4'd9;
    localparam [3:0] L_CLS5    = 4'd14;

    localparam [5:0] QIDX_MLP_FC1      = 6'd7;
    localparam [5:0] QIDX_MLP_FC2      = 6'd8;
    localparam [5:0] QIDX_MLP_FC2_AUX0 = 6'd24; // 16 + L_MLP_FC2, shifts 22..44
    localparam [5:0] QIDX_MLP_FC2_AUX1 = 6'd40; // 32 + L_MLP_FC2, shifts 44..47
// kind
    localparam [2:0] LD_NONE = 3'd0;
    localparam [2:0] LD_IN   = 3'd1;
    localparam [2:0] LD_WGT  = 3'd2;
    localparam [2:0] LD_BIAS = 3'd3;

        localparam [2:0] LD_SCORE= 3'd4;
        localparam [2:0] LD_QUANT= 3'd5;

    // Weight-buffer address map used by the large-token attention schedule.
    // SCORE keeps the current Q row-block in the front 64 words, while the
    // following AV layer reuses V tiles cached behind this front slot.
    localparam [17:0] WBUF_Q_BASE = 18'd0;
    localparam [17:0] WBUF_V_BASE = 18'd128;
// FSM
    localparam [2:0] S_IDLE   = 3'd0;
    localparam [2:0] S_CHOOSE = 3'd1;
    localparam [2:0] S_WAIT   = 3'd2;
    localparam [2:0] S_PATCH_ADDR = 3'd3;
    localparam [2:0] S_NL_W_ADDR  = 3'd4;
    localparam [2:0] S_PATCH_LAUNCH = 3'd5;

    localparam [M_BLK_NUM-1:0] MBLK_ALL_READY = {M_BLK_NUM{1'b1}};

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

    wire [6:0] active_mblk_last_in = active_mblk_num - 7'd1;
    wire [M_BLK_NUM-1:0] active_mblk_mask_in = f_mblk_mask(active_mblk_num);

    // Runtime dimensions are stable for a layer. Register their decoded form
    // locally so the AXI address path does not start at the AXI-Lite config
    // registers and pass through the mblk-mask decoder in the same cycle.
    reg [6:0] active_mblk_last_q;
    reg [M_BLK_NUM-1:0] active_mblk_mask_q;

    reg [2:0] fsm;

    // Patch input address pipeline. The row and tile offsets are generated in
    // S_PATCH_ADDR, then added to the base and launched in S_PATCH_LAUNCH.
    // This removes two address multipliers from the AXI ARADDR register path.
    reg [17:0] patch_row_off_q;
    reg [17:0] patch_tile_off_q;
    reg [11:0] patch_len_q;

    // latched layer/base controls (use only registered copies in address generation)
    reg [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_in_l, base_w_l, base_b_l, base_out_l, base_q_l;
    reg [3:0] layer_id_q;
    reg [5:0] layer_final_ktile_q;
    reg       layer_inbuf_sel_q;
    reg [2:0] layer_head_idx_q;
    reg       layer_wgt_phase_q;
    reg       start_guard_q;

    wire [3:0] layer_id_eff = layer_id_q;

    // Address generation/control must see only registered copies, never live layer_start inputs.
    wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_in_eff  = base_in_l;
    wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_w_eff   = base_w_l;
    wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_b_eff   = base_b_l;
    wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_out_eff = base_out_l;
    wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base_q_eff   = base_q_l;
wire [11:0] w_ready_eff = w_ready_cur;
wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] score_head_q_off =
    {{(M_AXI_ADDR_WIDTH-ADDR_LSB-9){1'b0}}, layer_head_idx_q, 6'd0};
wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] score_v_base_w =
    base_w_eff - score_head_q_off + (OFF_W_V_OUT - OFF_W_Q_OUT);
wire [6:0] score_vcache_nblk0 = {3'd0, layer_head_idx_q, 1'b0};
wire [6:0] score_vcache_nblk1 = score_vcache_nblk0 + 7'd1;

    // stride(=blk?? word ??)
    function [15:0] f_kstride;
        input [3:0] layer;
        begin
            if (layer == L_PATCH)      f_kstride = STRIDE_PATCH_K;
            else if (layer == L_AV)      f_kstride = M_TOTAL;
            else if (layer == L_MLP_FC2) f_kstride = 16'd32;
            else                         f_kstride = 16'd384;
        end
    endfunction

    // DDR tile base in 256-bit words.
    // PATCH uses four chunks of a patch14 input/weight row.
    // FC2 uses 32-word output-channel blocks inside a 384-word hidden tile.
    // Other Small projection/MLP tiles use 384-word tile stride.
    function [17:0] f_kbase;
        input [5:0] t;
        input [15:0] kstride;
        begin
            if (kstride == STRIDE_PATCH_K) f_kbase = t * 18'd192;
            else if (kstride == 16'd32)  f_kbase = t * 18'd384;
            else                         f_kbase = t * {2'd0, kstride};
        end
    endfunction

    // tile length
    function [11:0] f_klen;
        input [3:0] layer;
        input [5:0] t;
        begin
            if (layer == L_PATCH) begin
                if (t == 6'd3) f_klen = STRIDE_PATCH_K - 12'd576;
                else           f_klen = 12'd192;
            end else if (layer == L_SCORE) begin
                f_klen = 12'd64; // SCORE head0: K=64
            end
        else if (layer == L_AV) begin
            f_klen = active_token_total; // AV: K=active token count
        end else if (layer == L_MLP_FC2) begin
            f_klen = 12'd32; // FC2 tile
        end else begin
                f_klen = 12'd384; // Small QKV/proj/FC1/CLS: K=384
            end
        end
    endfunction

    // priority encoder
    function [6:0] pick_first_set_mblk;
        input [M_BLK_NUM-1:0] v;
        integer pi;
        begin
            pick_first_set_mblk = M_BLK_NUM[6:0] - 7'd1;
            for (pi = M_BLK_NUM-1; pi >= 0; pi = pi - 1) begin
                if (v[pi]) pick_first_set_mblk = pi[6:0];
            end
        end
    endfunction

    // IN: base + mblk*KSTRIDE + k_base(tile)
    function [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] calc_in_addr;
        input [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base;
        input [6:0] mblk;
        input [5:0] tile;
        input [15:0] kstride;
        reg [17:0] off;
        begin
            off = (mblk * kstride) + f_kbase(tile, kstride);
            calc_in_addr = base + off;
        end
    endfunction

    // WGT: base + nblk*KSTRIDE + k_base(tile)
    function [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] calc_w_addr;
        input [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base;
        input [6:0] nblk;
        input [5:0] tile;
        input [15:0] kstride;
        reg [17:0] off;
        begin
            off = (nblk * kstride) + f_kbase(tile, kstride);
            calc_w_addr = base + off;
        end
    endfunction

    // Second half of the pipelined next-layer weight address calculation.
    // f_kbase() is registered in S_CHOOSE so this stage contains only one
    // nblk-by-stride multiply and the final base/offset addition.
    function [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] calc_w_addr_from_kbase;
        input [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base;
        input [6:0] nblk;
        input [17:0] kbase;
        input [15:0] kstride;
        reg [17:0] off;
        begin
            off = (nblk * kstride) + kbase;
            calc_w_addr_from_kbase = base + off;
        end
    endfunction

// OUT(residual): base_out + mblk*384 + nblk*32
function [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] calc_out_addr;
    input [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base;
    input [6:0] mblk;
    input [6:0] nblk;
    reg [17:0] off;
    begin
        off = (mblk * 16'd384) + (nblk * 16'd32);
        calc_out_addr = base + off;
    end
endfunction

// POS embedding table in W region:
//   base_w + OFF_W_POS_EMB + nblk*(M_BLK_NUM*32) + mblk*32
// Each burst loads 32 channels for one token-row block into SCOREBUF.
function [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] calc_pos_addr;
    input [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] base;
    input [6:0] mblk;
    input [6:0] nblk;
    reg [17:0] off;
    begin
        off = (nblk * (M_BLK_NUM * 32)) + (mblk * 18'd32);
        calc_pos_addr = base + OFF_W_POS_EMB + off;
    end
endfunction

// ktile tracking (buffer-side)
    reg [5:0] ktile_seen;

    // MLP bias is loaded once at tile0 (FC1) and reused for all other MLP tiles
    reg        mlp_bias_loaded;
    reg mlp_quant_fc1_loaded;
    reg mlp_quant_fc2_loaded;
    reg mlp_quant_fc2_aux0_loaded;
    reg mlp_quant_fc2_aux1_loaded;
    reg [5:0] act_quant_tgt_addr;
    reg        act_bias_is_mlp;

// Quant load target:
// - FC1 preloads FC2 base/aux quant words.
// - FC2 can also load missing words if entered directly during debug.
wire mlp_quant_layer_eff = (layer_id_eff == L_MLP_FC1) || (layer_id_eff == L_MLP_FC2);
wire [5:0] quant_tgt_addr_eff =
    ((layer_id_eff == L_MLP_FC1) && !mlp_quant_fc1_loaded) ? QIDX_MLP_FC1 :
    (mlp_quant_layer_eff && !mlp_quant_fc2_loaded)          ? QIDX_MLP_FC2 :
    (mlp_quant_layer_eff && !mlp_quant_fc2_aux0_loaded)     ? QIDX_MLP_FC2_AUX0 :
    (mlp_quant_layer_eff && !mlp_quant_fc2_aux1_loaded)     ? QIDX_MLP_FC2_AUX1 :
                                                              {2'd0, layer_id_eff};

wire is_cls_eff = (layer_id_eff >= L_CLS0) && (layer_id_eff <= L_CLS5);

    // ktile change pending
    reg        ktile_change_pending;
    reg [5:0] ktile_pending_val;

    wire ktile_change = (layer_id_eff == L_PATCH) && (ktile_idx_cur != ktile_seen);
    wire can_apply_tile_change = (fsm != S_WAIT);
    wire tile_switching_now = ktile_change_pending && can_apply_tile_change;
    wire patch_tile_boundary = (layer_id_eff == L_PATCH) && (ktile_change || ktile_change_pending);

    // next-tile input bookkeeping (patch only)
    reg [M_BLK_NUM-1:0] need_in_next;
    reg [M_BLK_NUM-1:0] in_ready_next;
    reg [5:0] next_tile_for_in;

    wire [6:0] m_pick_cur  = pick_first_set_mblk((~in_ready_cur) & active_mblk_mask_q);
    wire [6:0] m_pick_next = pick_first_set_mblk(need_in_next & (~in_ready_next) & active_mblk_mask_q);

    // 2-slot WEI schedule state
    // FC2 bring-up: sequential WGT preload pointer (nblk 0..11)
    reg [3:0] fc2_wgt_ptr;
    reg [3:0] wgt_slot0_nblk;  // 0->2->4->(next tile 0)
    reg [3:0] wgt_slot1_nblk;  // 1->3->5
    reg       wgt_slot0_valid;
    reg       wgt_slot1_valid;

    reg [1:0] wgt_slot_loading;
    reg [1:0] wgt_slot_release_pending;
    reg boot_wgt0_loaded;
    reg oproj_wgt0_seen;
    // SCORE(Q head0) reload bookkeeping
    reg  [6:0] score_q_mblk_next;
    reg        score_q_release_pending;
    reg act_wgt_slot;
    reg act_wgt_is_next;

    reg       slot0_is_next;
    reg [5:0] slot0_next_ktile;


    // next-layer W0 prefetch bookkeeping (slot0)
    reg        nl_w0_req;
    reg        nl_w0_valid;
    reg  [3:0] nl_w0_layer;
    reg  [5:0] nl_w0_ktile;
    reg  [6:0] nl_w0_nblk;
    reg        nl_w0_bank;
    reg  [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] nl_w0_base_w;
    reg        nl_w0_av_full;
    reg [1:0]  score_vcache_ready;



    // FC1->FC2 full weight prefetch sequencing (nblk 0..11)
    reg        nl_fc2_pf_active;
    reg [3:0]  nl_fc2_pf_ptr;
    reg        nl_w0_fc2_full;   // this nl_w0 descriptor represents "FC2 full preload"
// Export next-layer W0 prefetch completion (latched valid until consumed at layer_start_pulse)
assign nl_pf_valid = nl_w0_valid;
assign nl_pf_layer = nl_w0_layer;
assign nl_pf_ktile = nl_w0_ktile;
assign nl_pf_nblk  = nl_w0_nblk;
assign nl_pf_bank  = nl_w0_bank;

// Expected W0 descriptor at layer start / steady-state
// - Most layers start with (ktile=0, nblk=0, bank0)
// - AV starts with nblk={head,0}
// - FC2 (if prefetched later) starts on bank1
wire [6:0] nl_w0_expect_nblk = (layer_id_eff == L_AV) ? {3'd0, layer_head_idx_q, 1'b0} : 7'd0;
wire       nl_fc2_bank       = ~layer_wgt_phase_q;
wire       nl_w0_expect_bank = (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : layer_wgt_phase_q;
wire       nl_w0_match = nl_w0_valid &&
                         (nl_w0_layer == layer_id_eff) &&
                         (nl_w0_ktile == ktile_idx_cur) &&
                         (nl_w0_nblk  == nl_w0_expect_nblk) &&
                         (nl_w0_bank  == nl_w0_expect_bank);

// Raw-input match used only on the layer_start_pulse capture cycle.
wire [6:0] nl_w0_expect_nblk_live = (layer_id == L_AV) ? {3'd0, layer_head_idx, 1'b0} : 7'd0;
wire       nl_fc2_bank_live       = ~layer_wgt_phase;
wire       nl_w0_expect_bank_live = (layer_id == L_MLP_FC2) ? nl_fc2_bank_live : layer_wgt_phase;
wire       nl_w0_match_live = nl_w0_valid &&
                              (nl_w0_layer == layer_id) &&
                              (nl_w0_ktile == ktile_idx_cur) &&
                              (nl_w0_nblk  == nl_w0_expect_nblk_live) &&
                              (nl_w0_bank  == nl_w0_expect_bank_live);

wire [11:0] w_ready_eff_fc2 = w_ready_eff;
wire [3:0] fc2_ptr_eff_fc2 = fc2_wgt_ptr;
 
// MLP FC1/FC2 are run in prefetch-only mode: current-layer weight reads are suppressed.
wire mlp_prefetch_only = (layer_id_eff == L_MLP_FC1) || (layer_id_eff == L_MLP_FC2);



    reg        act_wgt_is_nextlayer;

    reg        act_wgt_is_nl_fc2pf;
    reg        act_wgt_is_score_vcache;
    // Active transaction latch
    wire r_fire = axi_rvalid && axi_rready;

    reg [2:0]  act_kind;
    reg [5:0] act_ktile;
    reg [6:0]  act_nblk;
    reg [6:0]  act_mblk;
    reg [17:0] act_base;
    reg        act_bank;
    reg [11:0] beat_idx;

    reg [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] nl_addr_base_q;
    reg [17:0] nl_addr_kbase_q;
    reg [15:0] nl_addr_kstride_q;
    reg [11:0] nl_addr_len_q;

    reg act_in_is_next;

    // OPROJ residual prefetch into SCOREBUF
    reg        res_inflight;
    reg [6:0]  res_nblk;
    reg [6:0]  res_mblk;
    reg        res_bank;

    wire [11:0] res_need     = w_ready_cur & ~res_ready_cur;
    wire       res_need_any = |res_need;
    wire [6:0] res_pick     = res_need[0]  ? 7'd0  :
                              res_need[1]  ? 7'd1  :
                              res_need[2]  ? 7'd2  :
                              res_need[3]  ? 7'd3  :
                              res_need[4]  ? 7'd4  :
                              res_need[5]  ? 7'd5  :
                              res_need[6]  ? 7'd6  :
                              res_need[7]  ? 7'd7  :
                              res_need[8]  ? 7'd8  :
                              res_need[9]  ? 7'd9  :
                              res_need[10] ? 7'd10 :
                              res_need[11] ? 7'd11 : 7'd0;

    wire [6:0] res_nblk_eff = res_inflight ? res_nblk : res_pick;
    wire [6:0] res_mblk_eff = res_inflight ? res_mblk : 7'd0;
    wire       res_bank_eff = res_inflight ? res_bank : res_pick[0];
    wire [6:0] fc2_res_nblk_eff = res_inflight ? res_nblk : nblk_idx_cur;
    wire [6:0] fc2_res_mblk_eff = res_inflight ? res_mblk : 7'd0;


    // stride latch per layer (combinational is fine too)
wire [15:0] kstride_cur = f_kstride(layer_id_eff);
wire [3:0] slot0_last_even = N_BLK_NUM[3:0] - 4'd2;
wire [3:0] slot1_last_odd  = N_BLK_NUM[3:0] - 4'd1;

    // ---------------- Sequential ----------------
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            fsm <= S_IDLE;

            base_in_l <= 0;
            base_w_l  <= 0;
            base_b_l  <= 0;
            base_out_l<= 0;
            base_q_l  <= 0;
            layer_id_q <= 0;
            layer_final_ktile_q <= 0;
            layer_inbuf_sel_q   <= 0;
            layer_head_idx_q    <= 0;
            layer_wgt_phase_q   <= 0;
            start_guard_q       <= 0;

            ktile_seen <= 0;
            ktile_change_pending <= 0;
            ktile_pending_val    <= 0;

            need_in_next     <= 0;
            in_ready_next    <= 0;
            next_tile_for_in <= 0;

            wgt_slot0_nblk  <= 0;
            wgt_slot1_nblk  <= 0;
            wgt_slot0_valid <= 0;
            wgt_slot1_valid <= 0;

            wgt_slot_loading         <= 0;
            wgt_slot_release_pending <= 0;

            boot_wgt0_loaded <= 0;
                    oproj_wgt0_seen <= 0;
            score_q_mblk_next       <= 0;
            score_q_release_pending <= 1'b0;

            slot0_is_next    <= 0;
            slot0_next_ktile <= 0;
            nl_w0_req   <= 0;
            nl_w0_valid <= 0;
            nl_w0_layer <= 0;
            nl_w0_ktile <= 0;
            nl_w0_nblk  <= 0;
            nl_w0_bank  <= 0;
            nl_w0_base_w<= 0;
            nl_w0_av_full <= 1'b0;
            score_vcache_ready <= 2'b00;
            nl_fc2_pf_active <= 1'b0;
            nl_fc2_pf_ptr    <= 4'd0;
            nl_w0_fc2_full   <= 1'b0;
            act_wgt_is_nextlayer <= 0;
            act_wgt_is_nl_fc2pf    <= 1'b0;
            act_wgt_is_score_vcache <= 1'b0;
            bias_ready   <= 0;
                mlp_bias_loaded <= 1'b0;
                mlp_quant_fc1_loaded <= 1'b0;
                mlp_quant_fc2_loaded <= 1'b0;
                mlp_quant_fc2_aux0_loaded <= 1'b0;
                mlp_quant_fc2_aux1_loaded <= 1'b0;
                act_quant_tgt_addr <= 6'd0;
                act_bias_is_mlp <= 1'b0;
            in_ready_cur <= 0;
            w_ready_cur  <= 0;
            active_mblk_last_q <= M_BLK_NUM[6:0] - 7'd1;
            active_mblk_mask_q <= {M_BLK_NUM{1'b1}};
            res_ready_cur <= 0;
            res_upto0 <= 0; res_upto1 <= 0; res_upto2 <= 0; res_upto3 <= 0; res_upto4 <= 0; res_upto5 <= 0;
            res_upto6 <= 0; res_upto7 <= 0; res_upto8 <= 0; res_upto9 <= 0; res_upto10 <= 0; res_upto11 <= 0;
                    fc2_wgt_ptr <= 0;

            m_axi_read_init   <= 0;
            m_axi_araddress   <= 0;
            m_axi_read_length <= 0;

            act_kind  <= LD_NONE;
            act_ktile <= 0;
            act_nblk  <= 0;
            act_mblk  <= 0;
            act_base  <= 0;
            act_bank  <= 0;
            beat_idx  <= 0;
            nl_addr_base_q    <= 0;
            nl_addr_kbase_q   <= 0;
            nl_addr_kstride_q <= 0;
            nl_addr_len_q     <= 0;

            act_in_is_next  <= 0;
            act_wgt_slot    <= 0;
            act_wgt_is_next <= 0;

            mem_wr_valid <= 0;
            mem_wr_kind  <= 0;
            mem_wr_bank  <= 0;
            mem_wr_addr  <= 0;
            mem_wr_data  <= 0;
        end else begin
            // defaults
            m_axi_read_init <= 0;
            mem_wr_valid    <= 0;

            if (start_guard_q && !layer_start_pulse) begin
                start_guard_q <= 1'b0;
            end

            if (!layer_active) begin
                fsm <= S_IDLE;

                bias_ready   <= 0;
            quant_ready <= 0;
                mlp_quant_fc1_loaded <= 1'b0;
                mlp_quant_fc2_loaded <= 1'b0;
                mlp_quant_fc2_aux0_loaded <= 1'b0;
                mlp_quant_fc2_aux1_loaded <= 1'b0;
                in_ready_cur <= 0;
                w_ready_cur  <= 0;
                res_ready_cur <= 0;
                    fc2_wgt_ptr <= 0;

                need_in_next  <= 0;
                in_ready_next <= 0;

                wgt_slot0_nblk  <= 0;
                wgt_slot1_nblk  <= 0;
                wgt_slot0_valid <= 0;
                wgt_slot1_valid <= 0;

                wgt_slot_loading         <= 0;
                wgt_slot_release_pending <= 0;

                boot_wgt0_loaded <= 0;

                slot0_is_next    <= 0;
                slot0_next_ktile <= 0;

                ktile_seen <= 0;
                ktile_change_pending <= 0;
                ktile_pending_val    <= 0;
                nl_w0_av_full <= 1'b0;
                score_vcache_ready <= 2'b00;

                act_kind <= LD_NONE;
                                    act_wgt_is_nextlayer <= 0;
                                    act_wgt_is_nl_fc2pf    <= 1'b0;
                                    act_wgt_is_score_vcache <= 1'b0;
                beat_idx <= 0;

                m_axi_read_length <= 0;
                layer_id_q <= 0;
                layer_final_ktile_q <= 0;
                layer_inbuf_sel_q   <= 0;
                layer_head_idx_q    <= 0;
                layer_wgt_phase_q   <= 0;
                start_guard_q       <= 0;
            end else begin
                // start init
                
                // latch next-layer prefetch request (one-shot)
// - Default: single W0 prefetch.
// - Special: FC1->FC2 uses "full preload" (nblk 0..5) sequence; nl_pf_valid asserts only after all 6 are loaded.
                if (nl_wgt0_prefetch_pulse) begin
                    nl_w0_layer  <= nl_wgt0_layer_id;
                    nl_w0_ktile  <= nl_wgt0_ktile;
                    nl_w0_bank   <= nl_wgt0_bank;
                    nl_w0_base_w <= nl_wgt0_base_w_word;

                    if (nl_wgt0_layer_id == L_MLP_FC2) begin
                        // FC2 full preload sequence
                        nl_fc2_pf_active <= 1'b1;
                        nl_fc2_pf_ptr    <= 3'd0;
                        nl_w0_fc2_full   <= 1'b1;
                        nl_w0_av_full    <= 1'b0;

                        nl_w0_req        <= 1'b0;   // sequence-driven; do not use legacy single-shot path
                        nl_w0_nblk       <= 3'd0;   // descriptor stays at W0 for pf_done matching
                        nl_w0_valid      <= 1'b0;   // clear any stale completion
                    end else if (nl_wgt0_layer_id == L_AV) begin
                        // SCORE->AV: V tile0/1 are cached behind the Q slot while SCORE runs.
                        // This descriptor becomes valid only after both cached V tiles are ready.
                        nl_w0_req      <= 1'b0;
                        nl_w0_nblk     <= nl_wgt0_nblk;
                        nl_w0_fc2_full <= 1'b0;
                        nl_w0_av_full  <= 1'b1;
                        nl_w0_valid    <= (score_vcache_ready == 2'b11);
                    end else begin
                        // legacy: single W0 prefetch
                        nl_w0_req      <= 1'b1;
                        nl_w0_nblk     <= nl_wgt0_nblk;
                        nl_w0_fc2_full <= 1'b0;
                        nl_w0_av_full  <= 1'b0;
                        nl_w0_valid    <= 1'b0; // (re)arm
                    end
                end

                if (nl_w0_av_full && (score_vcache_ready == 2'b11)) begin
                    nl_w0_valid <= 1'b1;
                end

                if (layer_start_pulse) begin
                    active_mblk_last_q <= active_mblk_last_in;
                    active_mblk_mask_q <= active_mblk_mask_in;
                    base_in_l <= base_in_word;
                    layer_id_q <= layer_id;
                    layer_final_ktile_q <= layer_final_ktile;
                    layer_inbuf_sel_q   <= layer_inbuf_sel;
                    layer_head_idx_q    <= layer_head_idx;
                    layer_wgt_phase_q   <= layer_wgt_phase;
                    start_guard_q       <= 1'b1;
                    base_w_l  <= base_w_word;
                    base_b_l  <= base_b_word;
                    base_out_l<= base_out_word;

            base_q_l   <= base_q_word;
                    ktile_seen <= ktile_idx_cur;

                    ktile_change_pending <= 0;
                    ktile_pending_val    <= 0;

// quant policy on new layer start:
// - default: quant not ready, will be loaded first
// - FC2: can skip reload if it was preloaded during FC1
if (layer_id == L_MLP_FC2) quant_ready <= (mlp_quant_fc2_loaded &&
                                           mlp_quant_fc2_aux0_loaded &&
                                           mlp_quant_fc2_aux1_loaded);
else                       quant_ready <= 1'b0;

// FC1 triggers (re)load of FC1 quant and all FC2 quant words
if (layer_id == L_MLP_FC1) begin
    mlp_quant_fc1_loaded <= 1'b0;
    mlp_quant_fc2_loaded <= 1'b0;
    mlp_quant_fc2_aux0_loaded <= 1'b0;
    mlp_quant_fc2_aux1_loaded <= 1'b0;
end

                    // bias policy on new layer start:
                    // - SCORE/AV/FC2: no bias fetch needed
                    // - MLP_FC1: fetch bias only once at tile0, then reuse for other tiles
                    // - others: bias fetch required
                    // IMPORTANT: MLP bias is per-encoder-block. Reset when FC1 tile0 starts.
                    if ((layer_id == L_MLP_FC1) && (ktile_idx_cur == 5'd0)) begin
                        mlp_bias_loaded <= 1'b0;
                    end
                    if ((layer_id == L_SCORE) || (layer_id == L_AV) || (layer_id == L_MLP_FC2)) begin
                        bias_ready <= 1'b1;
                    end else if (layer_id == L_MLP_FC1) begin
                        // For each encoder block, FC1 tile0 must reload bias (base_b_word changes per block).
                        bias_ready <= (ktile_idx_cur == 5'd0) ? 1'b0 : (mlp_bias_loaded ? 1'b1 : 1'b0);
end else begin
                        bias_ready <= 1'b0;
                    end

                    // Patch: DDR???? IN ???? ???
                    if (layer_id == L_PATCH) begin
                        in_ready_cur <= ~active_mblk_mask_in;
                        mlp_bias_loaded <= 1'b0; // new run / rewind
                    end else begin
                        // Q: IN?? resident(GB)?? ???? -> ??? ????
                        in_ready_cur <= MBLK_ALL_READY;
                    end

                    need_in_next     <= 0;
                    in_ready_next    <= 0;
                    next_tile_for_in <= ktile_idx_cur + 1'b1;

                    w_ready_cur <= 0;
                    fc2_wgt_ptr <= 0;
                    res_ready_cur <= 0;
                    res_upto0 <= 0; res_upto1 <= 0; res_upto2 <= 0; res_upto3 <= 0; res_upto4 <= 0; res_upto5 <= 0;
                    res_upto6 <= 0; res_upto7 <= 0; res_upto8 <= 0; res_upto9 <= 0; res_upto10 <= 0; res_upto11 <= 0;
                    res_inflight  <= 0;
                    res_nblk      <= 0;
                    res_mblk      <= 0;
                    res_bank      <= 0;

                    if (layer_id == L_AV) begin
                        wgt_slot0_nblk  <= {layer_head_idx, 1'b0};
                        wgt_slot1_nblk  <= {layer_head_idx, 1'b0} + 3'd1;
                    end else begin
                        wgt_slot0_nblk  <= 3'd0;
                        wgt_slot1_nblk  <= 3'd1;
                    end
                    wgt_slot0_valid <= 0;
                    wgt_slot1_valid <= 0;

                    wgt_slot_loading         <= 0;
                    wgt_slot_release_pending <= 0;

                    boot_wgt0_loaded <= 0;
                    oproj_wgt0_seen <= 1'b0;
                    score_q_mblk_next       <= (layer_id == L_SCORE) ? mblk_idx_cur : 7'd0;
                    score_q_release_pending <= 1'b0;
                    if (layer_id == L_SCORE) begin
                        score_vcache_ready <= 2'b00;
                        nl_w0_av_full <= 1'b0;
                    end

                    slot0_is_next    <= 0;
                    slot0_next_ktile <= 0;

                    nl_fc2_pf_active <= 1'b0;
                    nl_fc2_pf_ptr    <= 3'd0;


// Consume next-layer prefetched W0 (skip W0 boot read)
if (nl_w0_match_live) begin
    boot_wgt0_loaded <= 1'b1;

    

    if (layer_id == L_O_PROJ) oproj_wgt0_seen <= 1'b1;
// Mark W0 as ready immediately for this layer
    if (layer_id == L_SCORE) begin
        w_ready_cur[mblk_idx_cur[0] ^ layer_wgt_phase] <= 1'b1;
        score_q_mblk_next   <= mblk_idx_cur + 7'd1;
    end else if ((layer_id == L_AV) && nl_w0_av_full) begin
        // Both V tiles for this head were cached during SCORE.
        w_ready_cur[{3'd0, layer_head_idx, 1'b0}] <= 1'b1;
        w_ready_cur[{3'd0, layer_head_idx, 1'b0} + 7'd1] <= 1'b1;

        wgt_slot0_nblk              <= {layer_head_idx, 1'b0};
        wgt_slot1_nblk              <= {layer_head_idx, 1'b0} + 4'd1;
        wgt_slot0_valid             <= 1'b1;
        wgt_slot1_valid             <= 1'b1;
        wgt_slot_loading[0]         <= 1'b0;
        wgt_slot_loading[1]         <= 1'b0;
        wgt_slot_release_pending[0] <= 1'b0;
        wgt_slot_release_pending[1] <= 1'b0;
    end else if ((layer_id == L_MLP_FC2) && nl_w0_fc2_full) begin
        // FC2 full preload consumed: all nblk0..5 are already in the FC2 bank
        w_ready_cur   <= 12'hFFF;
        fc2_wgt_ptr   <= N_BLK_NUM[3:0];
    end else begin
        w_ready_cur[nl_w0_expect_nblk_live] <= 1'b1;
    end

    // Keep slot0 bookkeeping as valid W0 (common path)
// NOTE: SCORE layer uses slot-valids as "unused" (kept 0) so that SCORE->AV next-layer prefetch
// can always reuse slot0 during SCORE without being blocked by slot0_valid.
if ((layer_id == L_SCORE) || (layer_id == L_MLP_FC2)) begin
    wgt_slot0_nblk              <= nl_w0_expect_nblk_live;
    wgt_slot0_valid             <= 1'b0;
    wgt_slot_loading[0]         <= 1'b0;
    wgt_slot_release_pending[0] <= 1'b0;
end else begin
    wgt_slot0_nblk              <= nl_w0_expect_nblk_live;
    wgt_slot0_valid             <= 1'b1;
    wgt_slot_loading[0]         <= 1'b0;
    wgt_slot_release_pending[0] <= 1'b0;
end

    // One-shot consume
    nl_w0_valid <= 1'b0;
    nl_w0_fc2_full <= 1'b0;
    nl_w0_av_full <= 1'b0;
    if (layer_id == L_AV) begin
        score_vcache_ready <= 2'b00;
    end
end

                    act_kind <= LD_NONE;
                                    act_wgt_is_nextlayer <= 0;
                                    act_wgt_is_score_vcache <= 1'b0;
                    act_bias_is_mlp <= 1'b0;
                    beat_idx <= 0;

                    m_axi_read_length <= 0;
                
                    // Force read FSM to restart for the new layer
                    fsm <= S_CHOOSE;

                    // NEW: run-boundary cleanup (when a new run starts at PATCH)
                    // Clear any latched next-layer prefetch completion / OPROJ residual gate that could leak across runs.
                    if (layer_id == L_PATCH) begin
                        nl_w0_valid    <= 1'b0;
                        nl_w0_fc2_full <= 1'b0;
                        nl_w0_req      <= 1'b0;
                        nl_w0_layer    <= 0;
                        nl_w0_ktile    <= 0;
                        nl_w0_nblk     <= 0;
                        nl_w0_bank     <= 0;
                        nl_w0_base_w   <= 0;
                        nl_w0_av_full  <= 1'b0;
                        score_vcache_ready <= 2'b00;
                        oproj_wgt0_seen <= 1'b0;
                        mlp_quant_fc1_loaded <= 1'b0;
                        mlp_quant_fc2_loaded <= 1'b0;
                        mlp_quant_fc2_aux0_loaded <= 1'b0;
                        mlp_quant_fc2_aux1_loaded <= 1'b0;
                    end
end else begin

                // ktile change handling (PATCH only)
                // - ktile change detection arrives 1 cycle after vit_controller increments ktile_idx_cur.
                // - With slow DDR, we must not allow compute to reuse previous-ktile IN data.
                // - If no AXI read is inflight (fsm != S_WAIT), apply the tile promotion immediately (no extra bubble).
                // - Otherwise, latch pending and force in_ready_cur=0 until promotion is applied.
                if (ktile_change && can_apply_tile_change) begin
                    // apply immediately (no 1-cycle bubble)
                    ktile_seen <= ktile_idx_cur;
                
                    // Patch: promote next->cur
                    in_ready_cur  <= in_ready_next;
                    in_ready_next <= 0;
                    need_in_next  <= 0;
                
                    // reset weights for new tile, BUT keep prefetched next-W0 if it matches
                    w_ready_cur <= 0;
                    fc2_wgt_ptr <= 0;
                    res_ready_cur <= 0;
                    res_inflight  <= 0;
                    res_nblk      <= 0;
                    res_mblk      <= 0;
                    res_bank      <= 0;
                
                    wgt_slot1_nblk  <= 3'd1;
                    wgt_slot1_valid <= 0;
                    wgt_slot_loading[1] <= 0;
                    wgt_slot_release_pending[1] <= 0;
                
                    if (slot0_is_next && wgt_slot0_valid && (slot0_next_ktile == ktile_idx_cur) && (wgt_slot0_nblk == 3'd0)) begin
                        w_ready_cur[0]   <= 1'b1;
                        boot_wgt0_loaded <= 1'b1;
                
                        wgt_slot0_nblk   <= 3'd0;
                        wgt_slot0_valid  <= 1'b1;
                        wgt_slot_loading[0] <= 0;
                        wgt_slot_release_pending[0] <= 0;
                
                        slot0_is_next <= 0;
                    end else begin
                        boot_wgt0_loaded <= 0;
                
                        wgt_slot0_nblk   <= 3'd0;
                        wgt_slot0_valid  <= 0;
                        wgt_slot_loading[0] <= 0;
                        wgt_slot_release_pending[0] <= 0;
                
                        slot0_is_next    <= 0;
                        slot0_next_ktile <= 0;
                    end
                
                    next_tile_for_in <= ktile_idx_cur + 1'b1;
                
                    ktile_change_pending <= 0;
                end else begin
                    // ktile change -> pending
                    if (ktile_change) begin
                        // block compute on the new ktile until promotion is applied
                        in_ready_cur <= 0;
                
                        ktile_change_pending <= 1'b1;
                        ktile_pending_val    <= ktile_idx_cur;
                    end
                
                    // apply tile promote only when not inflight
                    if (ktile_change_pending && can_apply_tile_change) begin
                        ktile_seen <= ktile_pending_val;
                
                        // Patch: promote next->cur
                        in_ready_cur  <= in_ready_next;
                        in_ready_next <= 0;
                        need_in_next  <= 0;
                
                        // reset weights for new tile, BUT keep prefetched next-W0 if it matches
                        w_ready_cur <= 0;
                        fc2_wgt_ptr <= 0;
                        res_ready_cur <= 0;
                        res_inflight  <= 0;
                        res_nblk      <= 0;
                        res_mblk      <= 0;
                        res_bank      <= 0;
                
                        wgt_slot1_nblk  <= 3'd1;
                        wgt_slot1_valid <= 0;
                        wgt_slot_loading[1] <= 0;
                        wgt_slot_release_pending[1] <= 0;
                
                        if (slot0_is_next && wgt_slot0_valid && (slot0_next_ktile == ktile_pending_val) && (wgt_slot0_nblk == 3'd0)) begin
                            w_ready_cur[0]   <= 1'b1;
                            boot_wgt0_loaded <= 1'b1;
                
                            wgt_slot0_nblk   <= 3'd0;
                            wgt_slot0_valid  <= 1'b1;
                            wgt_slot_loading[0] <= 0;
                            wgt_slot_release_pending[0] <= 0;
                
                            slot0_is_next <= 0;
                        end else begin
                            boot_wgt0_loaded <= 0;
                
                            wgt_slot0_nblk   <= 3'd0;
                            wgt_slot0_valid  <= 0;
                            wgt_slot_loading[0] <= 0;
                            wgt_slot_release_pending[0] <= 0;
                
                            slot0_is_next    <= 0;
                            slot0_next_ktile <= 0;
                        end
                
                        next_tile_for_in <= ktile_pending_val + 1'b1;
                
                        ktile_change_pending <= 0;
                    end
                end

                // next-tile input overwrite request (patch only)
                if ((layer_id_eff == L_PATCH) && (mblk_release_pulse != 0)) begin
                    need_in_next     <= need_in_next | mblk_release_pulse;
                    next_tile_for_in <= ktile_seen + 1'b1;
                end

                // capture releases
                if (layer_id == L_SCORE) begin
                    // SCORE: free the bank that just finished (vit_controller toggles bank by mblk parity)
                    if (wgt_slot_release_pulse[0]) w_ready_cur[0] <= 1'b0;
                    if (wgt_slot_release_pulse[1]) w_ready_cur[1] <= 1'b0;
                end else if (patch_tile_boundary) begin
                    // A PATCH ktile boundary resets the weight-slot sequence for the new tile.
                    // Ignore one-cycle release pulses that belong to the previous tile.
                    wgt_slot_release_pending <= 0;
                end else begin
                    if (wgt_slot_release_pulse[0]) wgt_slot_release_pending[0] <= 1'b1;
                    if (wgt_slot_release_pulse[1]) wgt_slot_release_pending[1] <= 1'b1;
                end
                if ((layer_id_eff != L_SCORE) && !patch_tile_boundary && wgt_slot_release_pending[0] && !wgt_slot_loading[0]) begin
                    wgt_slot_release_pending[0] <= 0;

                    if (wgt_slot0_valid && !slot0_is_next) begin
                        w_ready_cur[wgt_slot0_nblk] <= 0;
                    end

                    if (wgt_slot0_nblk < slot0_last_even) begin
                        wgt_slot0_nblk  <= wgt_slot0_nblk + 4'd2;
                        wgt_slot0_valid <= 0;
                        slot0_is_next   <= 0;
                    end else if (wgt_slot0_nblk == slot0_last_even) begin
                        wgt_slot0_nblk  <= 4'd0;
                        wgt_slot0_valid <= 0;
                        slot0_is_next   <= 1'b1;
                        slot0_next_ktile<= ktile_seen + 1'b1;
                    end
                end

                if ((layer_id_eff != L_SCORE) && !patch_tile_boundary && wgt_slot_release_pending[1] && !wgt_slot_loading[1]) begin
                    wgt_slot_release_pending[1] <= 0;

                    if (wgt_slot1_valid) begin
                        w_ready_cur[wgt_slot1_nblk] <= 0;
                    end

                    if (wgt_slot1_nblk < slot1_last_odd) begin
                        wgt_slot1_nblk  <= wgt_slot1_nblk + 4'd2;
                        wgt_slot1_valid <= 0;
                    end
                end

                // R-channel -> SRAM write
                if (r_fire) begin
                    mem_wr_valid <= 1'b1;
                    mem_wr_kind  <= act_kind;
                    // BIAS is stored in a single (unbanked) RAM in memory_controller
                    mem_wr_bank  <= (act_kind == LD_BIAS) ? 1'b0 : act_bank;
mem_wr_addr  <= act_base + beat_idx;
                    mem_wr_data  <= axi_rdata;
                    beat_idx     <= beat_idx + 1'b1;
                end

                case (fsm)
                    S_IDLE: begin
                        fsm <= S_CHOOSE;
                    end

                    S_CHOOSE: begin
                        if (start_guard_q) begin
                            fsm <= S_CHOOSE;
                        end else if (tile_switching_now) begin
                            fsm <= S_CHOOSE;
                        end else begin
                            // priority:
// 0) quant constants (must be loaded before compute starts)
if (!quant_ready) begin
    act_kind       <= LD_QUANT;
    act_quant_tgt_addr <= quant_tgt_addr_eff;
    act_ktile      <= 0;
    act_nblk       <= 0;
    act_mblk       <= 0;
    act_bank       <= 0;
    act_base       <= {{(M_AXI_ADDR_WIDTH-ADDR_LSB-6){1'b0}}, quant_tgt_addr_eff};
    act_in_is_next <= 0;
    act_wgt_is_next<= 0;
    beat_idx       <= 0;

    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= base_q_eff + quant_tgt_addr_eff;
    m_axi_araddress[ADDR_LSB-1:0]               <= 0;
    m_axi_read_length <= 17'd1;
    m_axi_read_init   <= 1'b1;
    fsm <= S_WAIT;
end else

                            // 1) bias
                            // 2) W0 boot
                            // 3) IN (patch only)
                            // 4) W1 boot
                            // 5) next tile W0 prefetch
                            // 6) next-tile IN overwrite (patch only)
                            // 7) slot reload

                            // SCORE: prefetch Q(mblk) into WGT ping-pong banks (bank = mblk[0] ^ phase)
                            // FC2: sequential WGT preload (nblk 0..5) into FC2 bank.
                            // After preload completes (fc2_wgt_ptr > 11), fall through so other priorities
                            // (e.g., next-layer W0 prefetch for the next tile) can run during FC2.
                            if ((layer_id_eff == L_MLP_FC2) && (fc2_ptr_eff_fc2 < N_BLK_NUM[3:0]) && !mlp_prefetch_only) begin
                                if (w_ready_eff_fc2[fc2_ptr_eff_fc2]) begin
                                    fc2_wgt_ptr <= fc2_wgt_ptr + 3'd1;
                                    fsm <= S_CHOOSE;
                                end else if (!wgt_slot_loading[1]) begin
                                    act_wgt_slot   <= 1;

                                    act_kind       <= LD_WGT;
                                    act_ktile      <= ktile_idx_cur;
                                    act_nblk       <= fc2_ptr_eff_fc2;
                                    act_mblk       <= 0;
                                    act_bank       <= nl_fc2_bank;
                                    act_base       <= {fc2_ptr_eff_fc2, 5'd0};
                                    act_in_is_next <= 0;
                                    act_wgt_is_next<= 0;
                                    beat_idx       <= 0;

                                    wgt_slot_loading[1] <= 1'b1;

                                    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, fc2_ptr_eff_fc2, ktile_idx_cur, kstride_cur);
                                    m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                    m_axi_read_length <= {5'd0, f_klen(layer_id_eff, ktile_idx_cur)};
                                    m_axi_read_init   <= 1'b1;

                                    fsm <= S_WAIT;
                                end else begin
                                    fsm <= S_CHOOSE;
                                end
                             end else if (layer_id == L_SCORE) begin
                                // SCORE: prefetch Q(mblk) into ping-pong banks (bank = mblk[0] ^ phase)
                                // While the SCORE PE work consumes cached Q blocks, fill the back
                                // region of both weight banks with the V tiles needed by AV.
                                if (score_q_mblk_next <= active_mblk_last_q) begin
                                    if (!wgt_slot_loading[score_q_mblk_next[0]] && !w_ready_cur[score_q_mblk_next[0] ^ layer_wgt_phase_q]) begin
                                        act_wgt_slot   <= score_q_mblk_next[0];

                                        act_kind       <= LD_WGT;
                                        act_ktile      <= 5'd0;
                                        act_nblk       <= {5'd0, score_q_mblk_next[0]};
                                        act_mblk       <= score_q_mblk_next;
                                        act_bank       <= score_q_mblk_next[0] ^ layer_wgt_phase_q;
                                        act_base       <= WBUF_Q_BASE;
                                        act_in_is_next <= 0;
                                        act_wgt_is_next<= 0;
                                        act_wgt_is_score_vcache <= 1'b0;
                                        beat_idx       <= 0;

                                        wgt_slot_loading[score_q_mblk_next[0]] <= 1'b1;

                                        // Q is stored in DDR with depth=384 per mblk; selected head uses depth 0..63.
                                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_in_addr(base_w_eff, score_q_mblk_next, 6'd0, 16'd384);
                                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                        m_axi_read_length <= {5'd0, f_klen(layer_id_eff, 5'd0)};
                                        m_axi_read_init   <= 1'b1;

                                        score_q_mblk_next <= score_q_mblk_next + 1'b1;

                                        fsm <= S_WAIT;
                                    end else if (!score_vcache_ready[0] && !wgt_slot_loading[0]) begin
                                        act_wgt_slot   <= 0;

                                        act_kind       <= LD_WGT;
                                        act_ktile      <= 5'd0;
                                        act_nblk       <= score_vcache_nblk0;
                                        act_mblk       <= 0;
                                        act_bank       <= ~layer_wgt_phase_q;
                                        act_base       <= WBUF_V_BASE;
                                        act_in_is_next <= 0;
                                        act_wgt_is_next<= 0;
                                        act_wgt_is_score_vcache <= 1'b1;
                                        beat_idx       <= 0;

                                        wgt_slot_loading[0] <= 1'b1;

                                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                                            calc_w_addr(score_v_base_w, score_vcache_nblk0, 6'd0, f_kstride(L_AV));
                                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                        m_axi_read_length <= {5'd0, f_klen(L_AV, 6'd0)};
                                        m_axi_read_init   <= 1'b1;

                                        fsm <= S_WAIT;
                                    end else if (!score_vcache_ready[1] && !wgt_slot_loading[1]) begin
                                        act_wgt_slot   <= 1;

                                        act_kind       <= LD_WGT;
                                        act_ktile      <= 5'd0;
                                        act_nblk       <= score_vcache_nblk1;
                                        act_mblk       <= 0;
                                        act_bank       <= layer_wgt_phase_q;
                                        act_base       <= WBUF_V_BASE;
                                        act_in_is_next <= 0;
                                        act_wgt_is_next<= 0;
                                        act_wgt_is_score_vcache <= 1'b1;
                                        beat_idx       <= 0;

                                        wgt_slot_loading[1] <= 1'b1;

                                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                                            calc_w_addr(score_v_base_w, score_vcache_nblk1, 6'd0, f_kstride(L_AV));
                                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                        m_axi_read_length <= {5'd0, f_klen(L_AV, 6'd0)};
                                        m_axi_read_init   <= 1'b1;

                                        fsm <= S_WAIT;
                                    end else begin
                                        fsm <= S_CHOOSE;
                                    end
                                end else if (!score_vcache_ready[0] && !wgt_slot_loading[0]) begin
                                    act_wgt_slot   <= 0;

                                    act_kind       <= LD_WGT;
                                    act_ktile      <= 5'd0;
                                    act_nblk       <= score_vcache_nblk0;
                                    act_mblk       <= 0;
                                    act_bank       <= ~layer_wgt_phase_q;
                                    act_base       <= WBUF_V_BASE;
                                    act_in_is_next <= 0;
                                    act_wgt_is_next<= 0;
                                    act_wgt_is_score_vcache <= 1'b1;
                                    beat_idx       <= 0;

                                    wgt_slot_loading[0] <= 1'b1;

                                    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                                        calc_w_addr(score_v_base_w, score_vcache_nblk0, 6'd0, f_kstride(L_AV));
                                    m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                    m_axi_read_length <= {5'd0, f_klen(L_AV, 6'd0)};
                                    m_axi_read_init   <= 1'b1;

                                    fsm <= S_WAIT;
                                end else if (!score_vcache_ready[1] && !wgt_slot_loading[1]) begin
                                    act_wgt_slot   <= 1;

                                    act_kind       <= LD_WGT;
                                    act_ktile      <= 5'd0;
                                    act_nblk       <= score_vcache_nblk1;
                                    act_mblk       <= 0;
                                    act_bank       <= layer_wgt_phase_q;
                                    act_base       <= WBUF_V_BASE;
                                    act_in_is_next <= 0;
                                    act_wgt_is_next<= 0;
                                    act_wgt_is_score_vcache <= 1'b1;
                                    beat_idx       <= 0;

                                    wgt_slot_loading[1] <= 1'b1;

                                    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                                        calc_w_addr(score_v_base_w, score_vcache_nblk1, 6'd0, f_kstride(L_AV));
                                    m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                    m_axi_read_length <= {5'd0, f_klen(L_AV, 6'd0)};
                                    m_axi_read_init   <= 1'b1;

                                    fsm <= S_WAIT;
                                end else if (nl_w0_req && !nl_w0_av_full && !nl_w0_valid && !wgt_slot0_valid && !wgt_slot_loading[0]) begin
                                    // next-layer W0 prefetch during SCORE (use slot0 bookkeeping, bank from descriptor)
                                    act_wgt_slot        <= 0;

                                    act_kind            <= LD_WGT;
                                    act_ktile           <= nl_w0_ktile;
                                    act_nblk            <= nl_w0_nblk;
                                    act_mblk            <= 0;
                                    act_bank            <= nl_w0_bank;
                                    act_base            <= 0;
                                    act_in_is_next      <= 0;
                                    act_wgt_is_next     <= 1'b0;
                                    act_wgt_is_nextlayer<= 1'b1;
                                    act_wgt_is_score_vcache <= 1'b0;
                                    beat_idx            <= 0;

                                    // disable next-tile W0 prefetch when using this slot for next layer
                                    slot0_is_next       <= 0;
                                    slot0_next_ktile    <= 0;

                                    // consume request
                                    nl_w0_req           <= 1'b0;

                                    wgt_slot_loading[0] <= 1'b1;

                                    nl_addr_base_q    <= nl_w0_base_w;
                                    nl_addr_kbase_q   <= f_kbase(nl_w0_ktile, f_kstride(nl_w0_layer));
                                    nl_addr_kstride_q <= f_kstride(nl_w0_layer);
                                    nl_addr_len_q     <= f_klen(nl_w0_layer, nl_w0_ktile);
                                    fsm <= S_NL_W_ADDR;
                                end else begin
                                    fsm <= S_CHOOSE;
                                end
                            end
                             else if (nl_fc2_pf_active && bias_ready && boot_wgt0_loaded && !nl_w0_valid && !wgt_slot1_valid && !wgt_slot_loading[1]) begin
                                 // FC1->FC2 full preload: prefetch FC2 nblk0..5 sequentially into nl_w0_bank (slot1).
                                 act_wgt_slot        <= 1;
                                 wgt_slot_loading[1] <= 1'b1;

                                 act_kind            <= LD_WGT;
                                 act_ktile           <= nl_w0_ktile;
                                 act_nblk            <= nl_fc2_pf_ptr;
                                 act_mblk            <= 0;
                                 act_bank            <= nl_w0_bank;
                                 act_base            <= {nl_fc2_pf_ptr, 5'd0};
                                 act_in_is_next      <= 0;
                                 act_wgt_is_next     <= 1'b0;
                                 act_wgt_is_nextlayer<= 1'b1;
                                 act_wgt_is_nl_fc2pf <= 1'b1;
                                 beat_idx            <= 0;

                                 nl_addr_base_q    <= nl_w0_base_w;
                                 nl_addr_kbase_q   <= f_kbase(nl_w0_ktile, f_kstride(nl_w0_layer));
                                 nl_addr_kstride_q <= f_kstride(nl_w0_layer);
                                 nl_addr_len_q     <= f_klen(nl_w0_layer, nl_w0_ktile);
                                 fsm <= S_NL_W_ADDR;
                             end
    else if ((layer_id_eff == L_MLP_FC2) &&
             (ktile_idx_cur == 6'd47) && bias_ready &&
             (res_inflight || !res_ready_cur[nblk_idx_cur])) begin
        // FC2 final K-tile: fetch the current output-channel residual slice
        // (add1) from DDR into SCOREBUF bank1. FC2 consumes nblk-major with
        // mblk inner, so the next nblk does not overwrite bank1 until the
        // current nblk has completed.
        if (!res_inflight) begin
            res_inflight <= 1'b1;
            res_nblk     <= nblk_idx_cur;
            res_mblk     <= 7'd0;
            res_bank     <= 1'b1;
        end

        act_kind       <= LD_SCORE;
        act_ktile      <= ktile_idx_cur;
        act_nblk       <= fc2_res_nblk_eff;
        act_mblk       <= fc2_res_mblk_eff;
        act_bank       <= 1'b1;
        act_base       <= {fc2_res_mblk_eff, 5'd0};
        act_in_is_next <= 1'b0;
        act_wgt_is_next<= 1'b0;
        beat_idx       <= 0;

        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
            calc_out_addr(base_out_eff, fc2_res_mblk_eff, fc2_res_nblk_eff);
        m_axi_araddress[ADDR_LSB-1:0] <= 0;
        m_axi_read_length <= 17'd32;
        m_axi_read_init   <= 1'b1;

        fsm <= S_WAIT;
    end
    else if ((layer_id_eff == L_PATCH) && (ktile_idx_cur == layer_final_ktile_q) &&
             bias_ready && (res_inflight || res_need_any)) begin
        // PATCH final: preload positional-embedding slice into SCOREBUF.
        // Reuses the OPROJ residual bookkeeping:
        //   bank = nblk[0], addr = mblk*32 + col, length = 32 words.
        if (!res_inflight) begin
            res_inflight <= 1'b1;
            res_nblk     <= res_pick;
            res_mblk     <= 0;
            res_bank     <= res_pick[0];
        end

        act_kind       <= LD_SCORE;
        act_ktile      <= ktile_idx_cur;
        act_nblk       <= res_nblk_eff;
        act_mblk       <= res_mblk_eff;
        act_bank       <= res_bank_eff;
        act_base       <= {res_mblk_eff, 5'd0}; // mblk*32
        act_in_is_next <= 0;
        act_wgt_is_next<= 0;
        beat_idx       <= 0;

        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_pos_addr(base_w_eff, res_mblk_eff, res_nblk_eff);
        m_axi_araddress[ADDR_LSB-1:0] <= 0;
        m_axi_read_length <= 17'd32;
        m_axi_read_init   <= 1'b1;

        fsm <= S_WAIT;
    end
    else if ((layer_id_eff == L_O_PROJ) && bias_ready && (res_inflight || (oproj_wgt0_seen && res_need_any))) begin
        // OPROJ: prefetch residual from DDR OUT(PATCH_OUT) into SCOREBUF bank=nblk[0]
        if (!res_inflight) begin
            res_inflight <= 1'b1;
            res_nblk     <= res_pick;
            res_mblk     <= 0;
            res_bank     <= res_pick[0];
        end

        act_kind       <= LD_SCORE;
        act_ktile      <= ktile_idx_cur;
        act_nblk       <= res_nblk_eff;
        act_mblk       <= res_mblk_eff;
        act_bank       <= res_bank_eff;
        act_base       <= {res_mblk_eff, 5'd0}; // mblk*32
        act_in_is_next <= 0;
        act_wgt_is_next<= 0;
        beat_idx       <= 0;

        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_out_addr(base_out_eff, res_mblk_eff, res_nblk_eff);
        m_axi_araddress[ADDR_LSB-1:0] <= 0;
        m_axi_read_length <= 17'd32;
        m_axi_read_init   <= 1'b1;

        fsm <= S_WAIT;
    end
    else if (!bias_ready) begin

                                act_kind       <= LD_BIAS;
                                act_bias_is_mlp <= (layer_id_eff == L_MLP_FC1);
                                act_ktile      <= ktile_idx_cur;
                                act_nblk       <= 0;
                                act_mblk       <= 0;
                                act_bank       <= 1'b0;  // BIAS RAM is single-bank; force writes to bank0
                                act_base       <= 0;
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= base_b_eff;
                                m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                m_axi_read_length <= (layer_id_eff == L_MLP_FC1) ? 17'd120 :
                                                     (is_cls_eff ? 17'd12 : 17'd24);
                                m_axi_read_init   <= 1'b1;

                                fsm <= S_WAIT;
                            end
                            else if (!boot_wgt0_loaded) begin
                                if (mlp_prefetch_only) begin
                                    // MLP FC1/FC2: prefetch-only. Do not auto-read current-layer weights here.
                                    fsm <= S_CHOOSE;
                                end else if (layer_id == L_SCORE) begin
                                    // SCORE: load Q for the current mblk into the
                                    // parity-selected front slot, depth 0..63.
                                    if (!wgt_slot_loading[score_q_mblk_next[0]]) begin
                                        act_wgt_slot   <= score_q_mblk_next[0];

                                        act_kind       <= LD_WGT;
                                        act_ktile      <= 0;
                                        act_nblk       <= {5'd0, score_q_mblk_next[0]};
                                        act_mblk       <= score_q_mblk_next;
                                        act_bank       <= score_q_mblk_next[0] ^ layer_wgt_phase_q;
                                        act_base       <= WBUF_Q_BASE;
                                        act_in_is_next <= 0;
                                        act_wgt_is_next<= 0;
                                        beat_idx       <= 0;

                                        wgt_slot_loading[score_q_mblk_next[0]] <= 1'b1;

                                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_in_addr(base_w_eff, score_q_mblk_next, 6'd0, 16'd384);
                                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                        m_axi_read_length <= {5'd0, f_klen(layer_id_eff, 0)};
                                        m_axi_read_init   <= 1'b1;

                                        score_q_mblk_next <= score_q_mblk_next + 1'b1;

                                        fsm <= S_WAIT;
                                    end else begin
                                        fsm <= S_CHOOSE;
                                    end
                                end else begin
                                if (!wgt_slot0_valid && !wgt_slot_loading[0] && !slot0_is_next) begin
                                    act_wgt_slot   <= 0;

                                    act_kind       <= LD_WGT;
                                    act_ktile      <= ktile_idx_cur;
                                    act_nblk       <= wgt_slot0_nblk;
                                    act_mblk       <= 0;
                                    act_bank       <= (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : layer_wgt_phase_q; // slot0 -> wgt bank0 (FC2 forces bank1)
                                    act_base       <= (layer_id_eff == L_AV) ? WBUF_V_BASE : 18'd0;
                                    act_in_is_next <= 0;
                                    act_wgt_is_next<= 0;
                                    beat_idx       <= 0;

                                    wgt_slot_loading[0] <= 1'b1;

                                    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, wgt_slot0_nblk, ktile_idx_cur, kstride_cur);
                                    m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                    m_axi_read_length <= {5'd0, f_klen(layer_id_eff, ktile_idx_cur)};
                                    m_axi_read_init   <= 1'b1;

                                    fsm <= S_WAIT;
                                end else begin
                                    fsm <= S_CHOOSE;
                                end

                                end
                            end
                            else if (layer_id == L_SCORE) begin
                                // SCORE: after Q(mblk) is loaded, wait for release from vit_controller
                                fsm <= S_CHOOSE;
                            end
                            else if ((layer_id_eff == L_PATCH) && ((in_ready_cur & active_mblk_mask_q) != active_mblk_mask_q)) begin
                                act_kind       <= LD_IN;
                                act_ktile      <= ktile_idx_cur;
                                act_nblk       <= 0;
                                act_mblk       <= m_pick_cur;
                                act_bank       <= layer_inbuf_sel_q;             // IN -> layer input bank
                                act_base       <= (m_pick_cur * MAX_KTILE);    // mblk segment base
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                // Register the selected row block before address
                                // generation. This keeps the priority encoder and
                                // address multiplier in separate 200-MHz stages.
                                fsm <= S_PATCH_ADDR;
                            end
                            else if (!wgt_slot1_valid && !wgt_slot_loading[1] && (wgt_slot1_nblk == 3'd1) && (layer_id_eff != L_MLP_FC1) && (layer_id_eff != L_MLP_FC2)) begin
                                act_wgt_slot   <= 1;

                                act_kind       <= LD_WGT;
                                act_ktile      <= ktile_idx_cur;
                                act_nblk       <= 3'd1;
                                act_mblk       <= 0;
                                act_bank       <= (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : (1'b1 ^ layer_wgt_phase_q); // slot1 -> wgt bank1
                                act_base       <= (layer_id_eff == L_AV) ? WBUF_V_BASE : 18'd0;
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                wgt_slot_loading[1] <= 1'b1;

                                m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, 3'd1, ktile_idx_cur, kstride_cur);
                                m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                m_axi_read_length <= {5'd0, f_klen(layer_id_eff, ktile_idx_cur)};
                                m_axi_read_init   <= 1'b1;

                                fsm <= S_WAIT;
                            end
                                                        else if (nl_w0_req && !nl_w0_valid &&
                                     ((!wgt_slot0_valid && !wgt_slot_loading[0]) || (!wgt_slot1_valid && !wgt_slot_loading[1]))) begin
                                // next-layer W0 prefetch into a free slot (prefer slot0)
                                if (!wgt_slot0_valid && !wgt_slot_loading[0]) begin
                                    act_wgt_slot        <= 0;
                                    wgt_slot_loading[0] <= 1'b1;
                                end else begin
                                    act_wgt_slot        <= 1;
                                    wgt_slot_loading[1] <= 1'b1;
                                end

                                act_kind            <= LD_WGT;
                                act_ktile           <= nl_w0_ktile;
                                act_nblk            <= nl_w0_nblk;
                                act_mblk            <= 0;
                                act_bank            <= nl_w0_bank;
                                act_base            <= 0;
                                act_in_is_next      <= 0;
                                act_wgt_is_next     <= 1'b0;
                                // NOTE: next-layer W0 prefetch request
                                act_wgt_is_nextlayer<= 1'b1;
                                act_wgt_is_nl_fc2pf <= 1'b0;
                                beat_idx            <= 0;

                                // consume request; next-tile prefetch (slot0) is also disabled by the caller
                                nl_w0_req           <= 1'b0;

                                // Pipeline the tile-base multiply separately from
                                // the nblk stride and final AXI address addition.
                                nl_addr_base_q    <= nl_w0_base_w;
                                nl_addr_kbase_q   <= f_kbase(nl_w0_ktile, f_kstride(nl_w0_layer));
                                nl_addr_kstride_q <= f_kstride(nl_w0_layer);
                                nl_addr_len_q     <= f_klen(nl_w0_layer, nl_w0_ktile);
                                fsm <= S_NL_W_ADDR;
                            end
                            else if (slot0_is_next && !wgt_slot0_valid && !wgt_slot_loading[0]) begin
                                act_wgt_slot   <= 0;

                                act_kind       <= LD_WGT;
                                act_ktile      <= slot0_next_ktile;
                                act_nblk       <= 3'd0;
                                act_mblk       <= 0;
                                act_bank       <= (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : layer_wgt_phase_q;
                                act_base       <= (layer_id_eff == L_AV) ? WBUF_V_BASE : 18'd0;
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 1'b1;
                                beat_idx       <= 0;

                                wgt_slot_loading[0] <= 1'b1;

                                m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, 3'd0, slot0_next_ktile, kstride_cur);
                                m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                m_axi_read_length <= {5'd0, f_klen(layer_id_eff, slot0_next_ktile)};
                                m_axi_read_init   <= 1'b1;

                                fsm <= S_WAIT;
                            end
                            else if ((layer_id_eff == L_PATCH) && ((need_in_next & (~in_ready_next) & active_mblk_mask_q) != 0)) begin
                                act_kind       <= LD_IN;
                                act_ktile      <= next_tile_for_in;
                                act_nblk       <= 0;
                                act_mblk       <= m_pick_next;
                                act_bank       <= layer_inbuf_sel_q;
                                act_base       <= (m_pick_next * MAX_KTILE);
                                act_in_is_next <= 1'b1;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                fsm <= S_PATCH_ADDR;
                            end
                            else if (!slot0_is_next && !wgt_slot0_valid && !wgt_slot_loading[0] && !mlp_prefetch_only) begin
                                act_wgt_slot   <= 0;

                                act_kind       <= LD_WGT;
                                act_ktile      <= ktile_idx_cur;
                                act_nblk       <= wgt_slot0_nblk;
                                act_mblk       <= 0;
                                act_bank       <= (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : layer_wgt_phase_q;
                                act_base       <= (layer_id_eff == L_AV) ? WBUF_V_BASE : 18'd0;
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                wgt_slot_loading[0] <= 1'b1;

                                m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, wgt_slot0_nblk, ktile_idx_cur, kstride_cur);
                                m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                m_axi_read_length <= {5'd0, f_klen(layer_id_eff, ktile_idx_cur)};
                                m_axi_read_init   <= 1'b1;

                                fsm <= S_WAIT;
                            end
                            else if (!wgt_slot1_valid && !wgt_slot_loading[1] && (layer_id_eff != L_MLP_FC1) && (layer_id_eff != L_MLP_FC2)) begin
                                act_wgt_slot   <= 1;

                                act_kind       <= LD_WGT;
                                act_ktile      <= ktile_idx_cur;
                                act_nblk       <= wgt_slot1_nblk;
                                act_mblk       <= 0;
                                act_bank       <= (layer_id_eff == L_MLP_FC2) ? nl_fc2_bank : (1'b1 ^ layer_wgt_phase_q);
                                act_base       <= 0;
                                act_in_is_next <= 0;
                                act_wgt_is_next<= 0;
                                beat_idx       <= 0;

                                wgt_slot_loading[1] <= 1'b1;

                                m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= calc_w_addr(base_w_eff, wgt_slot1_nblk, ktile_idx_cur, kstride_cur);
                                m_axi_araddress[ADDR_LSB-1:0] <= 0;
                                m_axi_read_length <= {5'd0, f_klen(layer_id_eff, ktile_idx_cur)};
                                m_axi_read_init   <= 1'b1;

                                fsm <= S_WAIT;
                            end
                            else begin
                                fsm <= S_CHOOSE;
                            end
                        end
                    end

                    S_PATCH_ADDR: begin
                        patch_row_off_q  <= act_mblk * STRIDE_PATCH_K;
                        patch_tile_off_q <= f_kbase(act_ktile, STRIDE_PATCH_K);
                        patch_len_q      <= f_klen(L_PATCH, act_ktile);
                        fsm <= S_PATCH_LAUNCH;
                    end

                    S_PATCH_LAUNCH: begin
                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                            base_in_eff + patch_row_off_q + patch_tile_off_q;
                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                        m_axi_read_length <= {5'd0, patch_len_q};
                        m_axi_read_init   <= 1'b1;
                        fsm <= S_WAIT;
                    end

                    S_NL_W_ADDR: begin
                        m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
                            calc_w_addr_from_kbase(nl_addr_base_q, act_nblk,
                                                  nl_addr_kbase_q, nl_addr_kstride_q);
                        m_axi_araddress[ADDR_LSB-1:0] <= 0;
                        m_axi_read_length <= {5'd0, nl_addr_len_q};
                        m_axi_read_init   <= 1'b1;
                        fsm <= S_WAIT;
                    end

                    S_WAIT: begin
                        if (m_axi_read_done) begin
                            if (act_kind == LD_BIAS) begin
                                bias_ready <= 1'b1;
                                if (act_bias_is_mlp) mlp_bias_loaded <= 1'b1;
                                act_bias_is_mlp <= 1'b0;
                            end else if (act_kind == LD_QUANT) begin

                                if ((layer_id == L_MLP_FC1) || (layer_id == L_MLP_FC2)) begin

                                    if (act_quant_tgt_addr == QIDX_MLP_FC1)      mlp_quant_fc1_loaded <= 1'b1;

                                    if (act_quant_tgt_addr == QIDX_MLP_FC2)      mlp_quant_fc2_loaded <= 1'b1;
                                    if (act_quant_tgt_addr == QIDX_MLP_FC2_AUX0) mlp_quant_fc2_aux0_loaded <= 1'b1;
                                    if (act_quant_tgt_addr == QIDX_MLP_FC2_AUX1) mlp_quant_fc2_aux1_loaded <= 1'b1;


                                    if ( (layer_id == L_MLP_FC2 ||
                                          (act_quant_tgt_addr == QIDX_MLP_FC1) || mlp_quant_fc1_loaded) &&

                                         ((act_quant_tgt_addr == QIDX_MLP_FC2) || mlp_quant_fc2_loaded) &&
                                         ((act_quant_tgt_addr == QIDX_MLP_FC2_AUX0) || mlp_quant_fc2_aux0_loaded) &&
                                         ((act_quant_tgt_addr == QIDX_MLP_FC2_AUX1) || mlp_quant_fc2_aux1_loaded) ) begin

                                        quant_ready <= 1'b1;

                                    end else begin

                                        quant_ready <= 1'b0;

                                    end

                                end else begin

                                    quant_ready <= 1'b1;

                                end


                            end else if (act_kind == LD_IN) begin
                                if (act_in_is_next) begin
                                    in_ready_next[act_mblk] <= 1'b1;
                                    need_in_next[act_mblk]  <= 1'b0;
                                end else begin
                                    in_ready_cur[act_mblk] <= 1'b1;
                                end
                            end else if (act_kind == LD_WGT) begin
                                // -------------------------------
                                // Weight load complete
                                // -------------------------------
                                if (act_wgt_is_score_vcache) begin
                                    // SCORE-time V-cache load for the following AV layer.
                                    if (act_wgt_slot == 0) begin
                                        score_vcache_ready[0] <= 1'b1;
                                        wgt_slot_loading[0]   <= 1'b0;
                                    end else begin
                                        score_vcache_ready[1] <= 1'b1;
                                        wgt_slot_loading[1]   <= 1'b0;
                                    end

                                    if (nl_w0_av_full &&
                                        ((act_wgt_slot == 0 && score_vcache_ready[1]) ||
                                         (act_wgt_slot == 1 && score_vcache_ready[0]))) begin
                                        nl_w0_valid <= 1'b1;
                                    end

                                    act_wgt_is_score_vcache <= 1'b0;
                                end else if (act_wgt_is_nextlayer) begin
                                    // -------------------------------
                                    // Next-layer prefetch complete
                                    // -------------------------------
                                    if (act_wgt_is_nl_fc2pf) begin
                                        // FC1->FC2 full preload: advance pointer; assert nl_w0_valid only after nblk11 completes.
                                        wgt_slot_loading[1] <= 1'b0;
                                        wgt_slot1_valid     <= 1'b0;

                                        if (nl_fc2_pf_ptr == (N_BLK_NUM[3:0] - 4'd1)) begin
                                            nl_fc2_pf_active <= 1'b0;
                                            nl_fc2_pf_ptr    <= 4'd0;
                                            nl_w0_valid      <= 1'b1;   // pf_done for FC1->FC2 gate
                                        end else begin
                                            nl_fc2_pf_ptr    <= nl_fc2_pf_ptr + 4'd1;
                                            nl_w0_valid      <= 1'b0;
                                        end
                                    end else begin
                                        // Legacy next-layer W0 prefetch: keep it protected in slot0/1 and assert nl_w0_valid.
                                        if (act_wgt_slot == 0) begin
                                            wgt_slot0_valid     <= 1'b1;
                                            wgt_slot_loading[0] <= 1'b0;
                                            wgt_slot0_nblk      <= act_nblk[3:0];
                                        end else begin
                                            wgt_slot1_valid     <= 1'b1;
                                            wgt_slot_loading[1] <= 1'b0;
                                            wgt_slot1_nblk      <= act_nblk[3:0];
                                        end
                                        nl_w0_valid <= 1'b1;
                                    end

                                    act_wgt_is_nl_fc2pf <= 1'b0;
                                    // NOTE: do NOT touch boot_wgt0_loaded / w_ready_cur for the current layer.
                                end else if (layer_id == L_SCORE) begin
                                    // SCORE Q(mblk) load: track readiness per physical bank (act_bank).
                                    if (act_wgt_slot == 0) wgt_slot_loading[0] <= 1'b0;
                                    else                  wgt_slot_loading[1] <= 1'b0;

                                    w_ready_cur[act_bank] <= 1'b1;

                                    if (!act_wgt_is_next) begin
                                        boot_wgt0_loaded <= 1'b1;
                                    end
                                end else begin
                                    if (act_wgt_slot == 0) begin
                                        wgt_slot0_valid     <= 1'b1;
                                        wgt_slot_loading[0] <= 1'b0;

                                        w_ready_cur[act_nblk] <= 1'b1;

                                        if ((layer_id_eff == L_O_PROJ) && !act_wgt_is_next) oproj_wgt0_seen <= 1'b1;
                                        if (!act_wgt_is_next) begin
                                            boot_wgt0_loaded <= 1'b1;
                                        end
                                    end else begin
                                        wgt_slot1_valid     <= 1'b1;
                                        wgt_slot_loading[1] <= 1'b0;

                                        w_ready_cur[act_nblk] <= 1'b1;
                                    end
                                end
                            end else if (act_kind == LD_SCORE) begin
// Track loaded residual tiles per nblk (loaded in-order mblk=0..6)
case (act_nblk)
    4'd0:  res_upto0  <= act_mblk;
    4'd1:  res_upto1  <= act_mblk;
    4'd2:  res_upto2  <= act_mblk;
    4'd3:  res_upto3  <= act_mblk;
    4'd4:  res_upto4  <= act_mblk;
    4'd5:  res_upto5  <= act_mblk;
    4'd6:  res_upto6  <= act_mblk;
    4'd7:  res_upto7  <= act_mblk;
    4'd8:  res_upto8  <= act_mblk;
    4'd9:  res_upto9  <= act_mblk;
    4'd10: res_upto10 <= act_mblk;
    4'd11: res_upto11 <= act_mblk;
    default: ;
endcase

// Allow compute to start as soon as the first tile(mblk0) is loaded.
if (act_mblk == 7'd0) begin
    res_ready_cur[act_nblk] <= 1'b1;
end

// Continue prefetching remaining mblks in background.
if (act_mblk == active_mblk_last_q) begin
    res_inflight <= 0;
    res_mblk     <= 0;
end else begin
    res_mblk     <= act_mblk + 1'b1;
    res_inflight <= 1'b1;
end
end

m_axi_read_length <= 0;
fsm <= S_CHOOSE;
                        end
                    end

                    default: begin
                        fsm <= S_IDLE;
                    end
                endcase
end
            end
        end
    end

endmodule

`default_nettype wire

