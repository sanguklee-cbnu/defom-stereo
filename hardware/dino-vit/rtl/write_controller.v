`timescale 1ns/1ps
`default_nettype none

module write_controller #(
    parameter integer M_AXI_ADDR_WIDTH = 32,
    parameter integer DATA_W           = 256,
    parameter integer ADDR_LSB         = 5,

    // ---- GOUT layout (BRAM) ----
    parameter integer M_BLK_NUM        = 7,
    parameter integer GOUT_STRIDE      = 384,

    // ---- GOUT address width ----
    parameter integer GOUT_AW          = 12,

    // ---- Internal FIFOs ----
    parameter integer REQ_FIFO_DEPTH      = 64,
    parameter integer REQ_FIFO_HEADROOM   = 8,
    parameter integer DATA_FIFO_DEPTH     = 64,
    parameter integer DATA_FIFO_HEADROOM  = 8
)(
    input  wire                         clk,
    input  wire                         nRst,

    // per-tile kick from WB (final ktile, col31)
    input  wire                         tile_start_pulse,
    input  wire [6:0]                   tile_mblk_idx,
    input  wire [6:0]                   tile_nblk_idx,

    // DDR destination descriptor (must be stable when tile_start_pulse=1)
    input  wire [M_AXI_ADDR_WIDTH-ADDR_LSB-1:0] tile_dst_base_word,     // absolute base (word index)
    input  wire [11:0]                  tile_dst_stride_words,          // columns per mblk (word count)
    input  wire [5:0]                   tile_dst_words,                 // words in this nblk (1..32)

    // NEW: for this tile, choose source
    // 0: read from GOUT SRAM (legacy dump)
    // 1: stream from internal DATA FIFO (final output overlap mode)
    input  wire                         tile_src_stream,

    // NEW: stream input (push into internal DATA FIFO)
    input  wire                         stream_in_valid,
    input  wire [DATA_W-1:0]            stream_in_data,
    output wire                         stream_in_ready,

    // WB is using GOUT read port for accumulation (block legacy dump reads)
    input  wire                         wb_gout_rd_en,

    // to AXI_Master (simple control IF)
    input  wire                         m_axi_write_done,
    output reg                          m_axi_write_init,
    output reg  [M_AXI_ADDR_WIDTH-1:0]  m_axi_awaddress,
    output reg  [16:0]                  m_axi_write_length,

    // to AXI_Master source-side stream
    output wire [DATA_W-1:0]            S_WDATA,
    output wire                         S_WVALID,
    input  wire                         S_WREADY,

    // GOUT SRAM read port (to memory_controller) - used only when tile_src_stream=0
    output reg                          gout_rd_en,
    output reg  [GOUT_AW-1:0]           gout_rd_addr,
    input  wire [DATA_W-1:0]            gout_rd_data,

    // burst done pulse (1 per tile)
    output reg                          dump_done_pulse,

    // current tile meta (for top-level done detect)
    output reg  [6:0]                   cur_mblk_idx_o,
    output reg  [6:0]                   cur_nblk_idx_o,

    // NEW: stream backpressure hint for top (data FIFO + req FIFO headroom)
    output wire                         stream_ok
);

    localparam integer WORD_AW = (M_AXI_ADDR_WIDTH-ADDR_LSB);

    // =========================================================================
    // DATA FIFO (stream words, 256b)
    // =========================================================================
    localparam integer DFIFO_AW = $clog2(DATA_FIFO_DEPTH);

    reg [DATA_W-1:0] dfifo_mem [0:DATA_FIFO_DEPTH-1];
    reg [DFIFO_AW-1:0] dfifo_wptr, dfifo_rptr;
    reg [DFIFO_AW:0]   dfifo_count;

    wire dfifo_full  = (dfifo_count == DATA_FIFO_DEPTH[DFIFO_AW:0]);
    wire dfifo_empty = (dfifo_count == 0);

    // "almost full" with headroom to cover WB pipeline and WREADY bubbles
    wire dfifo_afull = (dfifo_count >= (DATA_FIFO_DEPTH - DATA_FIFO_HEADROOM));

    assign stream_in_ready = !dfifo_full;

    // =========================================================================
    // REQ FIFO (tile descriptors)
    // =========================================================================
    localparam integer RFIFO_AW = $clog2(REQ_FIFO_DEPTH);
    localparam integer REQ_W = 1 + 7 + 7 + WORD_AW + 12 + 6; // src + mb + nb + base + stride + words

    reg [REQ_W-1:0] req_mem [0:REQ_FIFO_DEPTH-1];
    reg [RFIFO_AW-1:0] req_wptr, req_rptr;
    reg [RFIFO_AW:0]   req_count;

    wire req_full  = (req_count == REQ_FIFO_DEPTH[RFIFO_AW:0]);
    wire req_empty = (req_count == 0);
    wire req_afull = (req_count >= (REQ_FIFO_DEPTH - REQ_FIFO_HEADROOM));

    // stream_ok: use this to stall PE output during final stream pass
    // (prevents BOTH data FIFO overflow and req FIFO overflow)
    assign stream_ok = !dfifo_afull && !dfifo_full && !req_afull && !req_full;

    // =========================================================================
    // Request enqueue
    // =========================================================================
    wire [5:0] tile_words_norm = (tile_dst_words == 0) ? 6'd32 : tile_dst_words;

    wire req_push = tile_start_pulse && !req_full;

    wire [REQ_W-1:0] req_pack_in = {
        tile_src_stream,
        tile_mblk_idx,
        tile_nblk_idx,
        tile_dst_base_word,
        tile_dst_stride_words,
        tile_words_norm
    };

// =========================================================================
// FIFO memory writes (posedge-only) to improve LUTRAM/Distributed RAM inference
// - Keep pointers/counters in the main resettable always block
// - Avoid writing RAM arrays inside an async-reset always block
// =========================================================================
always @(posedge clk) begin
    if (nRst) begin
        if (stream_in_valid && stream_in_ready) begin
            dfifo_mem[dfifo_wptr] <= stream_in_data;
        end
        if (req_push) begin
            req_mem[req_wptr] <= req_pack_in;
        end
    end
end


    // head element (combinational view)
    wire [REQ_W-1:0] req_head = req_mem[req_rptr];

    // Unpack (LSB-first): words[5:0], stride[17:6], base[17+WORD_AW:18], nb[+7], mb[+7], src[MSB]
    wire [5:0]         head_words        = req_head[5:0];
    wire [11:0]        head_stride_words = req_head[17:6];
    wire [WORD_AW-1:0] head_base_word    = req_head[17+WORD_AW:18];
    wire [6:0]         head_nb           = req_head[17+WORD_AW+7:18+WORD_AW];
    wire [6:0]         head_mb           = req_head[17+WORD_AW+14:18+WORD_AW+7];
    wire               head_src_stream   = req_head[REQ_W-1];

    // =========================================================================
    // State / per-tile latched meta
    // =========================================================================
    reg active;
    reg wait_resp;
    wire busy = active || wait_resp;

    reg tile_src_stream_q;
    reg stream_start_hold; // 1-cycle bubble to avoid WVALID same-cycle as write_init in stream mode

    reg [5:0] sent_cnt;   // 0..31
    reg [5:0] fetch_idx;  // 0..31 (gout mode only)

    reg [GOUT_AW-1:0] tile_gout_base;
    reg [WORD_AW-1:0] tile_ddr_base_word;
    reg [11:0]        tile_ddr_stride_words;
    reg [5:0]         tile_words;

    // gout-mode read pipe
    reg rd_issue_d1;
    reg rd_issue_d2;
    reg rd_inflight;  // throttle outstanding GOUT reads (timing-friendly)
    reg [DATA_W-1:0] hold_data;
    reg              hold_valid;


    // pipelined skid-capture (break long CE path into 1 extra cycle)
    reg              stall_pipe_r;
    reg [DATA_W-1:0]  gout_rd_data_r;
    // =========================================================================
    // Source selection
    // =========================================================================
    // stream-mode: W channel directly pops DATA FIFO
    wire stream_mode = active && tile_src_stream_q;

    wire stream_wvalid = stream_mode && !stream_start_hold && (sent_cnt < tile_words) && !dfifo_empty;
    wire [DATA_W-1:0] stream_wdata = dfifo_mem[dfifo_rptr];

    // gout-mode: legacy pipe + skid (kept from original)
    wire pipe_valid = rd_issue_d2;
    wire gout_wvalid = hold_valid ? 1'b1 : pipe_valid;
    wire [DATA_W-1:0] gout_wdata  = hold_valid ? hold_data : gout_rd_data;

    assign S_WVALID = tile_src_stream_q ? stream_wvalid : gout_wvalid;
    assign S_WDATA  = tile_src_stream_q ? stream_wdata  : gout_wdata;

    wire beat_fire = S_WVALID && S_WREADY;
    wire last_fire = beat_fire && (sent_cnt == (tile_words - 1'b1));

    // pop from DATA FIFO only in stream mode
    wire dfifo_pop = beat_fire && tile_src_stream_q;

    // =========================================================================
    // GOUT read issue (only in gout-mode)
    // =========================================================================
    wire stall_pipe = (!hold_valid) && pipe_valid && (!S_WREADY);
    wire hold_pop   = hold_valid && S_WREADY;

    wire can_issue_rd = active &&
                        !tile_src_stream_q &&
                        (fetch_idx < tile_words) &&
                         (!rd_inflight) &&
                        (!stall_pipe) &&
                        (!stall_pipe_r) &&
                        (!(hold_valid && !hold_pop)) &&
                        (!wb_gout_rd_en);



    // [TIMING] Registered GOUT read enable/address (adds 1 extra cycle of latency in gout-mode)
    wire rd_issue = can_issue_rd;
    wire [GOUT_AW-1:0] gout_rd_addr_next = tile_gout_base + fetch_idx[4:0];


    // =========================================================================
    // Start condition (pop REQ FIFO head)
    // - stream tiles can start regardless of wb_gout_rd_en
    // - gout tiles must avoid conflict with wb_gout_rd_en
    // =========================================================================
    wire can_start = !busy && !req_empty && (head_src_stream || !wb_gout_rd_en);

    // =========================================================================
    // Sequential
    // =========================================================================
    always @(posedge clk or negedge nRst) begin
        if (!nRst) begin
            m_axi_write_init   <= 0;
            m_axi_awaddress    <= 0;
            m_axi_write_length <= 0;

            active    <= 0;
            wait_resp <= 0;

            tile_src_stream_q <= 0;
            stream_start_hold <= 0;

            sent_cnt  <= 0;
            fetch_idx <= 0;

            rd_issue_d1 <= 0;
                rd_issue_d2 <= 0;
                rd_inflight <= 0;

                gout_rd_en   <= 0;
                gout_rd_addr <= 0;
            hold_data   <= 0;
            hold_valid  <= 0;



                stall_pipe_r  <= 0;
                gout_rd_data_r <= 0;
            // FIFOs
            dfifo_wptr  <= 0;
            dfifo_rptr  <= 0;
            dfifo_count <= 0;

            req_wptr    <= 0;
            req_rptr    <= 0;
            req_count   <= 0;

            tile_gout_base        <= 0;
            tile_ddr_base_word    <= 0;
            tile_ddr_stride_words <= 0;
            tile_words            <= 0;

            cur_mblk_idx_o <= 0;
            cur_nblk_idx_o <= 0;

            dump_done_pulse <= 0;
        end else begin
            m_axi_write_init <= 0;
            dump_done_pulse  <= 0;

            // 1-cycle bubble clear (stream mode)
            if (stream_start_hold) stream_start_hold <= 1'b0;

            // ---------------------------
            // DATA FIFO push/pop
            // ---------------------------
            // push
            if (stream_in_valid && stream_in_ready) begin
                dfifo_wptr <= dfifo_wptr + 1'b1;
            end

            // pop
            if (dfifo_pop) begin
                dfifo_rptr <= dfifo_rptr + 1'b1;
            end

            // count update (handle simultaneous push+pop)
            case ({(stream_in_valid && stream_in_ready), dfifo_pop})
                2'b10: dfifo_count <= dfifo_count + 1'b1;
                2'b01: dfifo_count <= dfifo_count - 1'b1;
                default: dfifo_count <= dfifo_count;
            endcase

            // ---------------------------
            // REQ FIFO push
            // ---------------------------
            if (req_push) begin
                req_wptr <= req_wptr + 1'b1;
            end

            // ---------------------------
            // Start tile (pop REQ FIFO)
            // ---------------------------
            if (can_start) begin
                cur_mblk_idx_o <= head_mb;
                cur_nblk_idx_o <= head_nb;

                tile_src_stream_q <= head_src_stream;
                stream_start_hold <= head_src_stream; // bubble only for stream tiles

                // BRAM base: mblk*GOUT_STRIDE + nblk*32 (only used in gout mode)
                tile_gout_base <= (head_mb * GOUT_STRIDE) + {head_nb, 5'd0};

                // DDR base (word index): base + mblk*stride + nblk*32
                tile_ddr_base_word    <= head_base_word + (head_mb * head_stride_words) + {head_nb, 5'd0};
                tile_ddr_stride_words <= head_stride_words;
                tile_words            <= head_words;

                // setup AXI burst
                m_axi_awaddress    <= { (head_base_word + (head_mb * head_stride_words) + {head_nb, 5'd0}), {ADDR_LSB{1'b0}} };
                m_axi_write_length <= {11'd0, head_words};
                m_axi_write_init   <= 1'b1;

                active    <= 1'b1;
                wait_resp <= 1'b0;

                sent_cnt  <= 0;
                fetch_idx <= 0;

                rd_issue_d1 <= 0;
                rd_issue_d2 <= 0;
                rd_inflight <= 0;

                gout_rd_en   <= 0;
                gout_rd_addr <= 0;
                hold_data   <= 0;
                hold_valid  <= 0;


                stall_pipe_r  <= 0;
                gout_rd_data_r <= 0;
                // pop REQ FIFO head
                req_rptr  <= req_rptr + 1'b1;
            end

            // Request FIFO count update. A request enqueue and a request pop
            // can occur in the same cycle when a completed tile descriptor is
            // produced while the dump engine starts the next queued tile.
            // Treat that case as net-zero occupancy change.
            case ({req_push, can_start})
                2'b10: req_count <= req_count + 1'b1;
                2'b01: req_count <= req_count - 1'b1;
                default: req_count <= req_count;
            endcase

            // ---------------------------
            // gout-mode pipe marker / fetch
            // ---------------------------
            // gout-mode pipe marker (NOTE: rd_issue -> gout_rd_en (reg) -> RAM -> gout_rd_data)
            rd_issue_d1 <= rd_issue;
            rd_issue_d2 <= rd_issue_d1;

            // issue/read throttle bookkeeping
            if (rd_issue) begin
                rd_inflight <= 1'b1;
            end else if (pipe_valid) begin
                rd_inflight <= 1'b0;  // data has returned (either consumed or captured)
            end

            // registered read request to memory_controller (cuts comb path to BRAM EN)
            gout_rd_en <= rd_issue;
            if (rd_issue) begin
                gout_rd_addr <= gout_rd_addr_next;
            end else begin
                gout_rd_addr <= 0;
            end


            // register control/data for skid capture (adds 1-cycle latency only on stalls)
            stall_pipe_r  <= stall_pipe;
            gout_rd_data_r <= gout_rd_data;
            if (rd_issue) begin
                fetch_idx <= fetch_idx + 1'b1;
            end

            // hold capture / pop (gout mode only; harmless otherwise)
            if (stall_pipe_r) begin
                hold_data  <= gout_rd_data_r;
                hold_valid <= 1'b1;
            end else if (hold_pop) begin
                hold_valid <= 1'b0;
            end

            // ---------------------------
            // sent count (both modes)
            // ---------------------------
            if (beat_fire) begin
                if (sent_cnt < tile_words) begin
                    sent_cnt <= sent_cnt + 1'b1;
                end
            end

            // end of stream -> wait resp
            if (last_fire) begin
                active    <= 1'b0;
                wait_resp <= 1'b1;
            end

            // response done
            if (wait_resp && m_axi_write_done) begin
                dump_done_pulse <= 1'b1;
                wait_resp       <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
