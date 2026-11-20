`timescale 1ns/1ps
`default_nettype none

module memory_controller #(
    parameter integer MEM_WIDTH     = 256,

    // WEI_MEM_DEPTH = "physical depth"
    // logical bank depth = WEI_MEM_DEPTH/2
    // ex) WEI_MEM_DEPTH=512 -> bank0:0..255, bank1:256..511
    parameter integer WEI_MEM_DEPTH = 1024,

    // SCORE_MEM_DEPTH = physical depth for SCOREBUF (2-bank ping-pong)
    parameter integer SCORE_MEM_DEPTH = 512,

    parameter integer GLO_MEM_DEPTH = 4096,
    // GB1 holds two full activation regions: normalized/intermediate data at
    // the low addresses and the pre-LayerNorm residual at the high addresses.
    parameter integer GLO1_MEM_DEPTH = 8192,
    parameter integer GLO_ADDR_W    = 18,
    parameter integer SCORE_ADDR_W  = 12,
    parameter integer ATTN_MEM_DEPTH = 262144,
    parameter integer BIAS_DEPTH    = 128,

    parameter integer QUANT_DEPTH   = 64
)(
    input  wire clk,
    input  wire nRst,

    // -------------------------------------------------
    // Dynamic global bank select
    // IN bank      = layer_inbuf_sel
    // default GOUT = ~layer_inbuf_sel
    // Optional GOUT overrides are used by the fixed-role schedule:
    //   FC2 final/LN output can be redirected to the state bank.
    // -------------------------------------------------
    input  wire layer_inbuf_sel,
    input  wire gout_rd_bank_ovr_en,
    input  wire gout_rd_bank_ovr,
    input  wire gout_wr_bank_ovr_en,
    input  wire gout_wr_bank_ovr,

    // -------------------------------------------------
    // Write side (from AXI RDATA sink)
    // -------------------------------------------------
    input  wire        wr_valid,
    input  wire [2:0]  wr_kind,     // 1:IN(global) 2:WGT 3:BIAS 4:SCOREBUF(loader) 5:QUANT
    input  wire        wr_bank,     // WGT bank select(0/1). IN?? ????(????), bias?? ????
    input  wire [GLO_ADDR_W-1:0] wr_addr,     // word addr (in-bank addr?? ???)
    input  wire [MEM_WIDTH-1:0] wr_data,

    // Dedicated LayerNorm affine-loader write port into Bias RAM.
    input  wire                 ln_aff_wr_valid,
    input  wire [6:0]           ln_aff_wr_addr,
    input  wire [MEM_WIDTH-1:0] ln_aff_wr_data,

    // -------------------------------------------------
    // Read side to PE (global input read)
    // -------------------------------------------------
    input  wire        in_rd_en,
    input  wire [GLO_ADDR_W-1:0] in_rd_addr,
    output reg  [MEM_WIDTH-1:0] in_rd_data,

    // -------------------------------------------------
    // Read side to PE (weight read) : logical bank + in-bank addr
    // -------------------------------------------------
    input  wire        w_rd_en,
    input  wire        w_rd_bank,
    input  wire [11:0] w_rd_addr,
    output reg  [MEM_WIDTH-1:0] w_rd_data,

    // -------------------------------------------------
    // Read side for bias
    // -------------------------------------------------
    input  wire        b_rd_en,
    input  wire [6:0]  b_rd_addr,
    output reg  [MEM_WIDTH-1:0] b_rd_data,


    // -------------------------------------------------
    // Read side for quant constants
    // -------------------------------------------------
    input  wire        q_rd_en,
    input  wire [5:0]  q_rd_addr,
    output reg  [MEM_WIDTH-1:0] q_rd_data,

    // -------------------------------------------------
    // Global OUT bank ports for writeback_controller / dump
    // -------------------------------------------------
    input  wire        gout_rd_en,
    input  wire [GLO_ADDR_W-1:0] gout_rd_addr,
    output reg  [MEM_WIDTH-1:0] gout_rd_data,

    input  wire        gout_wr_valid,
    input  wire [GLO_ADDR_W-1:0] gout_wr_addr,
    input  wire [MEM_WIDTH-1:0] gout_wr_data,

    // PATCH/FC2 INT32 partial-sum storage in fixed temp bank GB0.
    // Addressing matches the existing row-major GOUT address; each word stores
    // 32 lanes of INT32 partial sums instead of 32 lanes of INT8 activations.
    input  wire         psum_rd_en,
    input  wire [GLO_ADDR_W-1:0]  psum_rd_addr,
    output reg  [1023:0] psum_rd_data,

    input  wire         psum_wr_valid,
    input  wire [GLO_ADDR_W-1:0]  psum_wr_addr,
    input  wire [1023:0] psum_wr_data,

    // -------------------------------------------------
    // SCOREBUF raw-score buffer ports (1W1R, 2-bank ping-pong)
    // -------------------------------------------------
    input  wire        score_rd_en,
    input  wire        score_rd_bank,
    input  wire [SCORE_ADDR_W-1:0] score_rd_addr,
    output reg  [MEM_WIDTH-1:0] score_rd_data,

    // Independent read of SCOREBUF bank1 for the FC2 residual tile. During
    // FC2, the regular read port consumes GELU data from bank0 while this port
    // supplies the add1 tensor prefetched from DDR.
    input  wire        score_res_rd_en,
    input  wire [SCORE_ADDR_W-1:0] score_res_rd_addr,
    output reg  [MEM_WIDTH-1:0] score_res_rd_data,

    input  wire        score_wr_valid,
    input  wire        score_wr_bank,
    input  wire [SCORE_ADDR_W-1:0] score_wr_addr,
    input  wire [MEM_WIDTH-1:0] score_wr_data,

    // Softmax output attention matrix. This keeps SCORE -> AV traffic out of
    // the global buffers while preserving the existing SCOREBUF raw-score path.
    input  wire        attn_rd_en,
    input  wire [GLO_ADDR_W-1:0] attn_rd_addr,
    output reg  [MEM_WIDTH-1:0] attn_rd_data,

    input  wire        attn_wr_valid,
    input  wire [GLO_ADDR_W-1:0] attn_wr_addr,
    input  wire [MEM_WIDTH-1:0] attn_wr_data

);

    localparam integer WEI_AW   = $clog2(WEI_MEM_DEPTH);
    localparam integer SCORE_AW = $clog2(SCORE_MEM_DEPTH);
    localparam integer GLO0_AW = $clog2(GLO_MEM_DEPTH);
    localparam integer GLO1_AW = $clog2(GLO1_MEM_DEPTH);
    localparam integer ATTN_AW = $clog2(ATTN_MEM_DEPTH);
    localparam integer BIAS_AW = $clog2(BIAS_DEPTH);

    localparam integer QUANT_AW = $clog2(QUANT_DEPTH);

    // kind encoding
    localparam [2:0] K_NONE = 3'd0;
    localparam [2:0] K_IN   = 3'd1;
    localparam [2:0] K_WGT  = 3'd2;
    localparam [2:0] K_BIAS = 3'd3;
    localparam [2:0] K_SCORE = 3'd4; // loader -> SCOREBUF (residual cache)
    localparam [2:0] K_QUANT = 3'd5;

    // dynamic selects
    wire sel_in_bank        = layer_inbuf_sel;
    wire sel_gout_rd_bank   = gout_rd_bank_ovr_en ? gout_rd_bank_ovr : ~layer_inbuf_sel;
    wire sel_gout_wr_bank   = gout_wr_bank_ovr_en ? gout_wr_bank_ovr : ~layer_inbuf_sel;

    // IN write bank is forced to input bank (ignore wr_bank for IN)
    wire in_wr_bank_eff = layer_inbuf_sel;

    // ============================================================
    // Global banks: bank0/1 ???? 1W1R
    // - read : (??? bank?? GOUT bank??) gout_rd ???, ???? in_rd
    // - write: (??? bank?? GOUT bank??) gout_wr ???, ???? loader(K_IN)
    // ============================================================
    wire [1023:0] glo0_rdata_wide;
    wire [MEM_WIDTH-1:0] glo_rdata_b0 = glo0_rdata_wide[MEM_WIDTH-1:0];
    wire [MEM_WIDTH-1:0] glo_rdata_b1;

    // Local write-side stage for global-out bank writes.
    // This cuts WB -> gout_wr_addr/data/valid -> BRAM ADDR/WE long routes.
    reg                  gout_wr_valid_q;
    reg [GLO_ADDR_W-1:0] gout_wr_addr_q;
    reg [MEM_WIDTH-1:0]  gout_wr_data_q;
    reg                  sel_gout_wr_bank_q;

    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            gout_wr_valid_q  <= 1'b0;
            gout_wr_addr_q   <= 0;
            gout_wr_data_q   <= 0;
            sel_gout_wr_bank_q  <= 1'b0;
        end else begin
            gout_wr_valid_q <= gout_wr_valid;
            if (gout_wr_valid) begin
                gout_wr_addr_q <= gout_wr_addr;
                gout_wr_data_q <= gout_wr_data;
                sel_gout_wr_bank_q <= sel_gout_wr_bank;
            end
            
            if (!gout_wr_valid && !gout_wr_valid_q) begin
                sel_gout_wr_bank_q <= sel_gout_wr_bank;
            end
        end
    end

    // ---------------- bank0 arb ----------------
    wire bank0_is_gout = (sel_gout_rd_bank == 1'b0);
    wire bank0_is_gout_wr_q = (sel_gout_wr_bank_q == 1'b0);
    wire bank0_is_in   = (sel_in_bank   == 1'b0);

    wire        glo0_norm_we = (bank0_is_gout_wr_q && gout_wr_valid_q) ? 1'b1 :
                               (wr_valid && (wr_kind==K_IN) && (in_wr_bank_eff==1'b0));
    wire [GLO0_AW-1:0] glo0_norm_waddr = (bank0_is_gout_wr_q && gout_wr_valid_q) ? gout_wr_addr_q[GLO0_AW-1:0] :
                                         wr_addr[GLO0_AW-1:0];
    wire [MEM_WIDTH-1:0] glo0_norm_wdata = (bank0_is_gout_wr_q && gout_wr_valid_q) ? gout_wr_data_q :
                                           wr_data;

    wire        glo0_we = psum_wr_valid ? 1'b1 : glo0_norm_we;
    wire [GLO0_AW-1:0] glo0_waddr = psum_wr_valid ? psum_wr_addr[GLO0_AW-1:0] :
                                                    glo0_norm_waddr;
    wire [1023:0] glo0_wdata = psum_wr_valid ? psum_wr_data :
                                               {{(1024-MEM_WIDTH){1'b0}}, glo0_norm_wdata};

    wire        glo0_norm_re = (bank0_is_gout && gout_rd_en) ? 1'b1 :
                               (in_rd_en && bank0_is_in);
    wire [GLO0_AW-1:0] glo0_norm_raddr = (bank0_is_gout && gout_rd_en) ? gout_rd_addr[GLO0_AW-1:0] :
                                         in_rd_addr[GLO0_AW-1:0];

    wire        glo0_re = psum_rd_en ? 1'b1 : glo0_norm_re;
    wire [GLO0_AW-1:0] glo0_raddr = psum_rd_en ? psum_rd_addr[GLO0_AW-1:0] :
                                                  glo0_norm_raddr;

    ram_1w1r_global #(
        .DW(1024), .DEPTH(GLO_MEM_DEPTH), .AW(GLO0_AW)
    ) u_glo0 (
        .clk(clk),
        .we(glo0_we), .waddr(glo0_waddr), .wdata(glo0_wdata),
        .re(glo0_re), .raddr(glo0_raddr), .rdata(glo0_rdata_wide)
    );

    // ---------------- bank1 arb ----------------
    wire bank1_is_gout = (sel_gout_rd_bank == 1'b1);
    wire bank1_is_gout_wr_q = (sel_gout_wr_bank_q == 1'b1);
    wire bank1_is_in   = (sel_in_bank   == 1'b1);

    wire        glo1_we = (bank1_is_gout_wr_q && gout_wr_valid_q) ? 1'b1 :
                          (wr_valid && (wr_kind==K_IN) && (in_wr_bank_eff==1'b1));
    wire [GLO1_AW-1:0] glo1_waddr = (bank1_is_gout_wr_q && gout_wr_valid_q) ? gout_wr_addr_q[GLO1_AW-1:0] :
                                    wr_addr[GLO1_AW-1:0];
    wire [MEM_WIDTH-1:0] glo1_wdata = (bank1_is_gout_wr_q && gout_wr_valid_q) ? gout_wr_data_q :
                                      wr_data;

    wire        glo1_re = (bank1_is_gout && gout_rd_en) ? 1'b1 :
                          (in_rd_en && bank1_is_in);
    wire [GLO1_AW-1:0] glo1_raddr = (bank1_is_gout && gout_rd_en) ? gout_rd_addr[GLO1_AW-1:0] :
                                    in_rd_addr[GLO1_AW-1:0];

    ram_1w1r_global #(
        .DW(MEM_WIDTH), .DEPTH(GLO1_MEM_DEPTH), .AW(GLO1_AW)
    ) u_glo1 (
        .clk(clk),
        .we(glo1_we), .waddr(glo1_waddr), .wdata(glo1_wdata),
        .re(glo1_re), .raddr(glo1_raddr), .rdata(glo_rdata_b1)
    );

    // ============================================================
    // Weight: ???? RAM 1???? ???? 2-bank?? ??? (1W1R)
    // ============================================================
    localparam integer WEI_BANK_DEPTH = (WEI_MEM_DEPTH/2);
    localparam integer WEI_BANK_AW    = $clog2(WEI_BANK_DEPTH);

    wire [WEI_AW-1:0] wei_waddr_phys = {wr_bank,   wr_addr[WEI_BANK_AW-1:0]};
    wire [WEI_AW-1:0] wei_raddr_phys = {w_rd_bank, w_rd_addr[WEI_BANK_AW-1:0]};

    wire [MEM_WIDTH-1:0] wei_rdata;

    ram_1w1r #(
        .DW(MEM_WIDTH), .DEPTH(WEI_MEM_DEPTH), .AW(WEI_AW)
    ) u_wei (
        .clk(clk),
        .we( wr_valid && (wr_kind==K_WGT) ),
        .waddr( wei_waddr_phys ),
        .wdata( wr_data ),
        .re( w_rd_en ),
        .raddr( wei_raddr_phys ),
        .rdata( wei_rdata )
    );

    
    // ============================================================
    // SCOREBUF: two physical 1W1R banks.
    // Splitting the banks preserves the original total capacity while allowing
    // FC2 to read bank0 (GELU input) and bank1 (DDR-prefetched add1) together.
    // ============================================================
    localparam integer SCORE_BANK_DEPTH = (SCORE_MEM_DEPTH/2);
    localparam integer SCORE_BANK_AW    = $clog2(SCORE_BANK_DEPTH);

    // SCOREBUF write mux:
// - core softmax path uses score_wr_*
// - loader path (wr_valid + K_SCORE) can also write SCOREBUF (e.g. residual prefetch)
wire        score_we_mux   = score_wr_valid || (wr_valid && (wr_kind==K_SCORE));
wire        score_bank_mux = score_wr_valid ? score_wr_bank : wr_bank;
wire [SCORE_ADDR_W-1:0] score_addr_mux = score_wr_valid ? score_wr_addr : wr_addr[SCORE_ADDR_W-1:0];
wire [MEM_WIDTH-1:0] score_data_mux = score_wr_valid ? score_wr_data : wr_data;

// Local write-side stage for SCOREBUF.
// This cuts both WB-direct(score_wr_*) and loader(wr_valid/K_SCORE) paths before u_score.
reg        score_we_q;
reg        score_bank_q;
reg [SCORE_ADDR_W-1:0] score_addr_q;
reg [MEM_WIDTH-1:0] score_data_q;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        score_we_q   <= 1'b0;
        score_bank_q <= 1'b0;
        score_addr_q <= 0;
        score_data_q <= 0;
    end else begin
        score_we_q <= score_we_mux;
        if (score_we_mux) begin
            score_bank_q <= score_bank_mux;
            score_addr_q <= score_addr_mux;
            score_data_q <= score_data_mux;
        end
    end
end

wire score_bank0_we = score_we_q && !score_bank_q;
wire score_bank1_we = score_we_q &&  score_bank_q;

wire score_bank0_re = score_rd_en && !score_rd_bank;
wire score_bank1_re_normal = score_rd_en && score_rd_bank;
wire score_bank1_re = score_bank1_re_normal || score_res_rd_en;
wire [SCORE_BANK_AW-1:0] score_bank1_raddr =
    score_bank1_re_normal ? score_rd_addr[SCORE_BANK_AW-1:0] :
                            score_res_rd_addr[SCORE_BANK_AW-1:0];

wire [MEM_WIDTH-1:0] score_rdata_b0;
wire [MEM_WIDTH-1:0] score_rdata_b1;
reg                  score_rd_bank_qsel;

always @(posedge clk or negedge nRst) begin
    if (!nRst) begin
        score_rd_bank_qsel <= 1'b0;
    end else if (score_rd_en) begin
        score_rd_bank_qsel <= score_rd_bank;
    end
end

ram_1w1r #(
    .DW(MEM_WIDTH), .DEPTH(SCORE_BANK_DEPTH), .AW(SCORE_BANK_AW)
) u_score_b0 (
    .clk(clk),
    .we   (score_bank0_we),
    .waddr(score_addr_q[SCORE_BANK_AW-1:0]),
    .wdata(score_data_q),
    .re   (score_bank0_re),
    .raddr(score_rd_addr[SCORE_BANK_AW-1:0]),
    .rdata(score_rdata_b0)
);

ram_1w1r #(
    .DW(MEM_WIDTH), .DEPTH(SCORE_BANK_DEPTH), .AW(SCORE_BANK_AW)
) u_score_b1 (
    .clk(clk),
    .we   (score_bank1_we),
    .waddr(score_addr_q[SCORE_BANK_AW-1:0]),
    .wdata(score_data_q),
    .re   (score_bank1_re),
    .raddr(score_bank1_raddr),
    .rdata(score_rdata_b1)
);

// ============================================================
// Legacy attention-buffer ports
// - SCORE softmax now overwrites the current row-block in SCOREBUF and AV
//   consumes that row immediately, so no full-head attention SRAM is used.
// - Keep these ports for top-level compatibility with older debug builds.
// ============================================================
wire [MEM_WIDTH-1:0] attn_rdata = {MEM_WIDTH{1'b0}};

// ============================================================
    // Bias: LUTRAM(Distributed)
    // ============================================================
    wire [MEM_WIDTH-1:0] bias_rdata;
    wire bias_loader_we = wr_valid && (wr_kind==K_BIAS);
    wire bias_we = ln_aff_wr_valid || bias_loader_we;
    wire [BIAS_AW-1:0] bias_waddr = ln_aff_wr_valid ?
                                     ln_aff_wr_addr[BIAS_AW-1:0] :
                                     wr_addr[BIAS_AW-1:0];
    wire [MEM_WIDTH-1:0] bias_wdata = ln_aff_wr_valid ?
                                      ln_aff_wr_data : wr_data;

    ram_1w1r_bias #(
        .DW(MEM_WIDTH), .DEPTH(BIAS_DEPTH), .AW(BIAS_AW)
    ) u_bias (
        .clk(clk),
        .we( bias_we ),
        .waddr( bias_waddr ),
        .wdata( bias_wdata ),
        .re( b_rd_en ),
        .raddr( b_rd_addr[BIAS_AW-1:0] ),
        .rdata( bias_rdata )
    );

    

// ============================================================
// Quant constants: LUTRAM(Distributed)
// - 1 word (256b) per layer_id or per-entry index
// ============================================================
wire [MEM_WIDTH-1:0] quant_rdata;

ram_1w1r_bias #(
    .DW(MEM_WIDTH), .DEPTH(QUANT_DEPTH), .AW(QUANT_AW)
) u_quant (
    .clk(clk),
    .we( wr_valid && (wr_kind==K_QUANT) ),
    .waddr( wr_addr[QUANT_AW-1:0] ),
    .wdata( wr_data ),
    .re( q_rd_en ),
    .raddr( q_rd_addr[QUANT_AW-1:0] ),
    .rdata( quant_rdata )
);
// ============================================================
    // Output signals (combinational select)
    // ============================================================
    always @(*) begin
        in_rd_data   = (sel_in_bank   == 1'b0) ? glo_rdata_b0 : glo_rdata_b1;
        gout_rd_data = (sel_gout_rd_bank == 1'b0) ? glo_rdata_b0 : glo_rdata_b1;
        psum_rd_data = glo0_rdata_wide;
        w_rd_data     = wei_rdata;
        b_rd_data     = bias_rdata;
        score_rd_data = score_rd_bank_qsel ? score_rdata_b1 : score_rdata_b0;
        score_res_rd_data = score_rdata_b1;
        attn_rd_data  = attn_rdata;
        q_rd_data     = quant_rdata;
    end
endmodule


// ---------------------------------------------------------
// Deep global-buffer RAM. VCU128 UltraRAM keeps the large state
// buffers out of long block-RAM cascade chains at 200 MHz.
// ---------------------------------------------------------
module ram_1w1r_global #(
    parameter integer DW = 256,
    parameter integer DEPTH = 1024,
    parameter integer AW = 10
)(
    input  wire clk,
    input  wire we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire re,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);
    (* ram_style = "ultra" *)
    reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule


// ---------------------------------------------------------
// Simple 1W1R synchronous RAM (BRAM inference)
// ---------------------------------------------------------
module ram_1w1r #(
    parameter integer DW = 256,
    parameter integer DEPTH = 1024,
    parameter integer AW = 10
)(
    input  wire clk,
    input  wire we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire re,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);
    // synthesis ram_style = "block"
    reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule


// ---------------------------------------------------------
// Bias LUTRAM (Distributed RAM inference)
// ---------------------------------------------------------
module ram_1w1r_bias #(
    parameter integer DW = 256,
    parameter integer DEPTH = 16,
    parameter integer AW = 4
)(
    input  wire              clk,
    input  wire              we,
    input  wire [AW-1:0]      waddr,
    input  wire [DW-1:0]      wdata,
    input  wire               re,
    input  wire [AW-1:0]      raddr,
    output wire [DW-1:0]      rdata
);
    (* ram_style = "distributed" *) reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    assign rdata = mem[raddr];   // async read
endmodule

`default_nettype wire
