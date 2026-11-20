`ifndef ADDR_MAP_VH
`define ADDR_MAP_VH
// ============================================================================
// addr_map.vh  (DINOv2-S/14 rectangular-token experiment)
// - DDR layout constants in 256-bit WORD INDEX units (word = 32 bytes)
// - Base addresses (BASE_*_WORD) are provided externally (controller_top / TB)
//
// Policy:
// - Patch weights appear once.
// - Encoder weights are stored per-block: block0 .. block11.
// - Q_OUT / V_OUT are stored per-encoder-block (block0..block11) for debug/dump.
// - OUT region is scratch by default (will be overwritten per block unless TB allocates per-block dump).
// ============================================================================

/* ---------- Common tiling ---------- */
localparam  integer ENC_BLK_NUM        = 12;

// Physical maximum layout. Runtime AXI-Lite configuration selects the active
// token count up to this limit. DeFoM pads KITTI images to multiples of 32
// before the /4 and 3.5x Depth-Anything resize, so raw 364x1218 maps to
// 336x1092 = 24x78 patches + CLS = 1873 tokens.
localparam  integer TOKEN_TOTAL         = 1873;
localparam  integer M_BLK_NUM_PATCH_Q  = 59;   // ceil(1873/32)
localparam  integer TOKEN_LAST_VALID    = 16;   // inclusive last valid lane index
localparam  integer EMBED_DIM          = 384;
localparam  integer EMBED_NBLK         = 12;  // for 384 channels -> 12 n-blocks
localparam  integer CLS_G192_NBLK      = 6;   // for 192 class logits
localparam  integer CLS_G64_NBLK       = 2;   // for 64 class logits
localparam  integer HEAD_NUM           = 6;
localparam  integer HEAD_DIM           = 64;

localparam  integer STRIDE_384 = 384;
localparam  integer STRIDE_192 = 192;
localparam  integer STRIDE_TOKEN = TOKEN_TOTAL;
localparam  integer STRIDE_197 = TOKEN_TOTAL;
localparam  integer STRIDE_PATCH_K = 589; // 3*14*14 + CLS-token special column
localparam  integer STRIDE_769 = STRIDE_PATCH_K;
localparam  integer STRIDE_64  = 64;

/* ---------- IN region (BASE_IN_WORD + OFF_*) ---------- */
// Patch-Embedding input is preloaded for the maximum token layout.
localparam integer OFF_IN_PATCH_INPUT  = 0;
localparam integer SIZE_IN_PATCH_INPUT = (M_BLK_NUM_PATCH_Q * STRIDE_PATCH_K);

// Optional externally supplied attention-score slots. The Small hardware path
// computes all six heads, but keeping these offsets reserved makes the DDR
// layout explicit for debug/backward tooling.
localparam integer OFF_IN_ATT_H1       = (OFF_IN_PATCH_INPUT + SIZE_IN_PATCH_INPUT);
localparam integer SIZE_IN_ATT_H1      = (M_BLK_NUM_PATCH_Q * STRIDE_TOKEN);
localparam integer OFF_IN_ATT_H2       = (OFF_IN_ATT_H1 + SIZE_IN_ATT_H1);
localparam integer SIZE_IN_ATT_H2      = (M_BLK_NUM_PATCH_Q * STRIDE_TOKEN);
localparam integer OFF_IN_ATT_H3       = (OFF_IN_ATT_H2 + SIZE_IN_ATT_H2);
localparam integer SIZE_IN_ATT_H3      = (M_BLK_NUM_PATCH_Q * STRIDE_197);
localparam integer OFF_IN_ATT_H4       = (OFF_IN_ATT_H3 + SIZE_IN_ATT_H3);
localparam integer SIZE_IN_ATT_H4      = (M_BLK_NUM_PATCH_Q * STRIDE_197);
localparam integer OFF_IN_ATT_H5       = (OFF_IN_ATT_H4 + SIZE_IN_ATT_H4);
localparam integer SIZE_IN_ATT_H5      = (M_BLK_NUM_PATCH_Q * STRIDE_197);

/* ---------- W region (BASE_W_WORD + OFF_*) ---------- */
// Patch-Embedding weights (global, once): 12 blk * 589 words
localparam integer OFF_W_PATCH_WGT     = 0;
localparam integer SIZE_W_PATCH_WGT    = (EMBED_NBLK * STRIDE_PATCH_K);

// -------- Encoder weights per-block (block0..block11) --------
// Q/V/K/O weights: each 12 blk * 384 words
localparam integer SIZE_W_Q_WGT        = (EMBED_NBLK * STRIDE_384);                            // 4608
localparam integer SIZE_W_V_WGT        = (EMBED_NBLK * STRIDE_384);                            // 4608
localparam integer SIZE_W_K_WGT        = (EMBED_NBLK * STRIDE_384);                            // 4608
localparam integer SIZE_W_O_WGT        = (EMBED_NBLK * STRIDE_384);                            // 4608

// MLP weights: FC1/FC2 each 48 tiles * 384 words (hidden=1536 -> 48*32)
localparam integer MLP_TILE_NUM        = 48;
localparam integer SIZE_W_FC1_WGT      = (MLP_TILE_NUM * STRIDE_384);                          // 18432
localparam integer SIZE_W_FC2_WGT      = (MLP_TILE_NUM * STRIDE_384);                          // 18432

// One encoder block weight footprint
// Layout per encoder block (matches gen_memhex.py):
//   Q_WGT -> V_WGT -> K_WGT -> Q_OUT(scratch) -> V_OUT(scratch) -> O_WGT -> FC1_WGT -> FC2_WGT
localparam integer SIZE_W_Q_OUT        = (M_BLK_NUM_PATCH_Q * STRIDE_384);
localparam integer SIZE_W_V_OUT        = (EMBED_NBLK * STRIDE_197);
localparam integer SIZE_W_ENC_BLK_WGT  = (SIZE_W_Q_WGT + SIZE_W_V_WGT + SIZE_W_K_WGT +
                                         SIZE_W_Q_OUT + SIZE_W_V_OUT +
                                         SIZE_W_O_WGT + SIZE_W_FC1_WGT + SIZE_W_FC2_WGT);

// Absolute offsets for block0 weights (block b base = OFF_W_ENC0 + b*SIZE_W_ENC_BLK_WGT)
localparam integer OFF_W_ENC0          = (OFF_W_PATCH_WGT + SIZE_W_PATCH_WGT);

localparam integer OFF_W_Q_WGT         = (OFF_W_ENC0 + 0);
localparam integer OFF_W_V_WGT         = (OFF_W_Q_WGT + SIZE_W_Q_WGT);
localparam integer OFF_W_K_WGT         = (OFF_W_V_WGT + SIZE_W_V_WGT);
// Q/V outputs are stored inside each encoder block footprint
localparam integer OFF_W_Q_OUT         = (OFF_W_K_WGT + SIZE_W_K_WGT);
localparam integer OFF_W_V_OUT         = (OFF_W_Q_OUT + SIZE_W_Q_OUT);
// OPROJ/MLP weights follow after Q_OUT/V_OUT scratch
localparam integer OFF_W_O_WGT         = (OFF_W_V_OUT + SIZE_W_V_OUT);
localparam integer OFF_W_FC1_WGT       = (OFF_W_O_WGT + SIZE_W_O_WGT);
localparam integer OFF_W_FC2_WGT       = (OFF_W_FC1_WGT + SIZE_W_FC1_WGT);

// End of all encoder blocks (weights + Q/V scratch)
localparam integer OFF_W_ENC_ALL_END   = (OFF_W_ENC0 + (ENC_BLK_NUM * SIZE_W_ENC_BLK_WGT));
// ---------- Classifier head weights (1000 classes, grouped) ----------
// Groups: cls0..cls4 each produce 192 logits, cls5 produces 64 logits.
// Each class-logit block consumes the 384-wide Small CLS vector.
localparam integer SIZE_W_CLS_G192_WGT = (CLS_G192_NBLK * STRIDE_384);                         // 6*384 = 2304
localparam integer SIZE_W_CLS_G64_WGT  = (CLS_G64_NBLK  * STRIDE_384);                         // 2*384 = 768

localparam integer OFF_W_CLS0_WGT      = OFF_W_ENC_ALL_END;
localparam integer OFF_W_CLS1_WGT      = (OFF_W_CLS0_WGT + SIZE_W_CLS_G192_WGT);
localparam integer OFF_W_CLS2_WGT      = (OFF_W_CLS1_WGT + SIZE_W_CLS_G192_WGT);
localparam integer OFF_W_CLS3_WGT      = (OFF_W_CLS2_WGT + SIZE_W_CLS_G192_WGT);
localparam integer OFF_W_CLS4_WGT      = (OFF_W_CLS3_WGT + SIZE_W_CLS_G192_WGT);
localparam integer OFF_W_CLS5_WGT      = (OFF_W_CLS4_WGT + SIZE_W_CLS_G192_WGT);

localparam integer OFF_W_CLS_ALL_END   = (OFF_W_CLS5_WGT + SIZE_W_CLS_G64_WGT);

// Positional embedding table appended after all regular weights.
// Layout: nblk-major, then token mblk, then 32 columns within the nblk:
//   OFF_W_POS_EMB + nblk*(M_BLK_NUM_PATCH_Q*32) + mblk*32 + col
// Each 256-bit word stores the 32 token lanes for one embedding channel.
localparam integer SIZE_W_POS_EMB      = (EMBED_NBLK * M_BLK_NUM_PATCH_Q * 32);
localparam integer OFF_W_POS_EMB       = OFF_W_CLS_ALL_END;
localparam integer OFF_W_END_ALL       = (OFF_W_POS_EMB + SIZE_W_POS_EMB);
/* ---------- OUT region (BASE_OUT_WORD + OFF_*) ---------- */
// Patch output (residual reuse): maximum token layout
localparam integer OFF_OUT_PATCH_OUT   = 0;
localparam integer SIZE_OUT_PATCH_OUT  = (M_BLK_NUM_PATCH_Q * STRIDE_384);

// Output projection output (option B / debug scratch)
localparam integer OFF_OUT_O_OUT       = (OFF_OUT_PATCH_OUT + SIZE_OUT_PATCH_OUT);

// ENC block final outputs (FC2 post-residual): base_out_word + OFF_OUT_O_OUT + enc_blk_idx*SIZE_OUT_PATCH_OUT (per-block)
localparam integer SIZE_OUT_O_OUT      = (M_BLK_NUM_PATCH_Q * STRIDE_384);

// ---------- Classifier head outputs (DDR dump destination) ----------
localparam integer OFF_OUT_ENC_ALL_END = (OFF_OUT_O_OUT + (ENC_BLK_NUM * SIZE_OUT_O_OUT));
localparam integer OFF_OUT_CLS0_OUT    = OFF_OUT_ENC_ALL_END;
localparam integer OFF_OUT_CLS1_OUT    = (OFF_OUT_CLS0_OUT + STRIDE_192);
localparam integer OFF_OUT_CLS2_OUT    = (OFF_OUT_CLS1_OUT + STRIDE_192);
localparam integer OFF_OUT_CLS3_OUT    = (OFF_OUT_CLS2_OUT + STRIDE_192);
localparam integer OFF_OUT_CLS4_OUT    = (OFF_OUT_CLS3_OUT + STRIDE_192);
localparam integer OFF_OUT_CLS5_OUT    = (OFF_OUT_CLS4_OUT + STRIDE_192);
localparam integer OFF_OUT_CLS_ALL_END = (OFF_OUT_CLS5_OUT + STRIDE_64);

// DPT feature taps. Encoder blocks 3/6/9/12 expose LayerNorm(Add2), while
// the original per-block FC2 slots above retain raw Add2 for the next
// encoder block's residual path.
localparam integer OFF_OUT_NORM3_B3     = OFF_OUT_CLS_ALL_END;
localparam integer OFF_OUT_NORM3_B6     = (OFF_OUT_NORM3_B3 + SIZE_OUT_PATCH_OUT);
localparam integer OFF_OUT_NORM3_B9     = (OFF_OUT_NORM3_B6 + SIZE_OUT_PATCH_OUT);
localparam integer OFF_OUT_NORM3_B12    = (OFF_OUT_NORM3_B9 + SIZE_OUT_PATCH_OUT);
localparam integer OFF_OUT_NORM3_ALL_END = (OFF_OUT_NORM3_B12 + SIZE_OUT_PATCH_OUT);

/* ---------- DDR dump destination selector ---------- */
localparam [1:0] DST_SEL_IN  = 2'd0;
localparam [1:0] DST_SEL_W   = 2'd1;
localparam [1:0] DST_SEL_OUT = 2'd2;

// ceil(stride_words / 32) -> number of n-blocks
function [6:0] f_nblk_num;
    input [11:0] stride_words;
    reg [11:0] tmp;
    begin
        tmp = stride_words + 12'd31;
        f_nblk_num = tmp[11:5];
    end
endfunction

function [5:0] f_last_tile_cols;
    input [11:0] stride_words;
    reg [4:0] rem;
    begin
        rem = stride_words[4:0];
        if (rem == 5'd0) f_last_tile_cols = 6'd32;
        else             f_last_tile_cols = {1'b0, rem};
    end
endfunction

`endif // ADDR_MAP_VH
