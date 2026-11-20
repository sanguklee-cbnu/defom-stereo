`timescale 1ns/1ps
`default_nettype none

module pe_feeder #(
    parameter integer AXI_DATA_WIDTH = 256,
    parameter integer MAX_KTILE      = 384
)(
    input  wire                   clk,
    input  wire                   nRst,

    // job descriptor from vit_controller
    input  wire                   job_start,
    input  wire [11:0]            job_klen,
    input  wire [5:0]             job_ktile,
    input  wire [6:0]             job_nblk,
    input  wire [6:0]             job_mblk,
    input  wire                   job_in_base_is_nblk,
    input  wire [9:0]             job_in_koff,
    input  wire                   job_wbuf_sel,
    input  wire                   swap_ab,
    input  wire                   job_a_is_qlog,

    output wire                   job_ready,
    output reg                    job_done,      // pulse when tile_last MAC is issued

    // writeback meta (aligned to drain)
    output reg  [5:0]             wb_ktile_idx,
    output reg  [6:0]             wb_nblk_idx,
    output reg  [6:0]             wb_mblk_idx,

    // memory_controller read ports
    output reg                    in_rd_en,
    output reg  [17:0]            in_rd_addr,
    input  wire [AXI_DATA_WIDTH-1:0] in_rd_data,

    output reg                    w_rd_en,
    output reg                    w_rd_bank,
    output reg  [11:0]            w_rd_addr,
    input  wire [AXI_DATA_WIDTH-1:0] w_rd_data,

    // to pe_array
    output reg                    pe_in_valid,
    output reg                    pe_first,
    output reg                    pe_tile_last,
    output reg                    pe_a_is_qlog,
    output reg  [32*8-1:0]        pe_a_rows_flat,
    output reg  [32*8-1:0]        pe_b_cols_flat,

    // from pe_array
    input  wire                   pe_array_busy,
    input  wire                   pe_array_done
);
    // NOTE:
    // We register-slice A/B vectors to cut BRAM->DSP routing at 300MHz.
    // memory_controller's ram_1w1r has synchronous read: rdata updates on the
    // same posedge that observes in_rd_en/w_rd_en.
    // Since in_rd_en/w_rd_en are driven by THIS module's flops, the RAM sees them
    // one cycle later. So the first valid read data appears one cycle after job_start.
    // If we latch A/B on the same edge as issue_fire, the very first beat becomes X.
    // Fix: delay (valid/first/last) by 1 cycle and capture A/B on that delayed edge.

    reg        running;           // issuing/accumulating only
    reg        drain_inflight;    // previous tile drain occupies PE out_data/out_valid

    reg [11:0] klen_lat;
    reg [5:0]  ktile_lat;
    reg [6:0]  nblk_lat;
    reg [6:0]  mblk_lat;
    reg        in_base_is_nblk_lat;
    reg        wbank_lat;
    reg        swap_ab_lat;

    // one-cycle delayed issue flags (align with RAM read data availability)
    reg        issue_q;
    reg        first_q;
    reg        last_q;

    // One more cycle of completion/meta alignment for pe_array input-pipeline mode.
    // Original timing: job_done/meta pulse when last beat is PRESENTED to PE.
    // With pe_array input bundle pipelined by 1 cycle, actual drain start shifts by +1,
    // so keep the original relative relationship by delaying done/meta by 1 cycle here.
    reg        done_meta_q;
    reg [5:0]  wb_ktile_idx_q;
    reg [6:0]  wb_nblk_idx_q;
    reg [6:0]  wb_mblk_idx_q;

    // FC2 (K=32) and AV V-cache helpers.
    // During large-token attention, AV reuses V tiles cached behind the Q slot
    // in the weight buffer. Q still occupies the front 64 words.
    localparam [11:0] WBUF_V_BASE = 12'd128;
    reg        fc2_k32_lat;
    reg [11:0] w_base_lat;   // nblk*32 for FC2, WBUF_V_BASE for AV, 0 otherwise

    reg [17:0] in_base_lat;
    reg [11:0] k_issue;           // current MAC index (0..klen-1)

    localparam [17:0] MAX_KTILE_U = MAX_KTILE;
    wire [17:0] job_in_stride =
        (job_klen == 12'd32) ? 18'd32 : MAX_KTILE_U;

    wire [17:0] job_in_base_calc =
        job_a_is_qlog ? {8'd0, job_in_koff} :
        (((job_klen == 12'd32) ? job_mblk : (job_in_base_is_nblk ? job_nblk : job_mblk))
         * job_in_stride) + job_in_koff;

    wire [11:0] k_last   = (klen_lat == 0) ? 12'd0 : (klen_lat - 12'd1);
    wire       is_last_k = (k_issue == k_last);

    // stall ONLY the final MAC if previous tile is still draining
    wire stall_last = running && (k_issue < klen_lat) && is_last_k && drain_inflight;
    wire issue_fire = running && (k_issue < klen_lat) && !stall_last;

    // can accept a new job only when no issuing AND no pending beat
    assign job_ready = (!running) && (!issue_q);

    always @(posedge clk) begin
        if (!nRst) begin
            running        <= 0;
            drain_inflight <= 0;

            klen_lat  <= 0;
            ktile_lat <= 0;
            nblk_lat  <= 0;
            mblk_lat  <= 0;
            in_base_is_nblk_lat <= 0;
            wbank_lat <= 0;


            fc2_k32_lat <= 0;
            w_base_lat   <= 0;
            in_base_lat <= 0;
            k_issue     <= 0;

            job_done <= 0;

            wb_ktile_idx <= 0;
            wb_nblk_idx  <= 0;
            wb_mblk_idx  <= 0;

            in_rd_en   <= 0;
            in_rd_addr <= 0;

            w_rd_en   <= 0;
            w_rd_bank <= 0;
            w_rd_addr <= 0;

            pe_in_valid  <= 0;
            pe_first     <= 0;
            pe_tile_last <= 0;
            pe_a_is_qlog <= 0;

            pe_a_rows_flat <= 0;
            pe_b_cols_flat <= 0;
            swap_ab_lat    <= 0;

            issue_q <= 0;
            first_q <= 0;
            last_q  <= 0;

            done_meta_q   <= 0;
            wb_ktile_idx_q<= 0;
            wb_nblk_idx_q <= 0;
            wb_mblk_idx_q <= 0;

        end else begin
            // defaults
            job_done <= done_meta_q;

            // delayed presentation to PE
            pe_in_valid  <= issue_q;
            pe_first     <= first_q;
            pe_tile_last <= last_q;

            // emit 1-cycle delayed completion/meta (re-aligned to pe_array input pipeline)
            if (done_meta_q) begin
                wb_ktile_idx <= wb_ktile_idx_q;
                wb_nblk_idx  <= wb_nblk_idx_q;
                wb_mblk_idx  <= wb_mblk_idx_q;
            end
            done_meta_q <= 0;

            // when issue_q is 1 (from previous cycle's issue_fire), RAM outputs are
            // now stable for that beat. Capture A/B on this edge.
            if (issue_q) begin
                if (swap_ab_lat) begin
                    pe_a_rows_flat <= w_rd_data[32*8-1:0];
                    pe_b_cols_flat <= in_rd_data[32*8-1:0];
                end else begin
                    pe_a_rows_flat <= in_rd_data[32*8-1:0];
                    pe_b_cols_flat <= w_rd_data[32*8-1:0];
                end
            end

            // arm 1-cycle delayed job_done/meta aligned to pe_array input-pipelined drain
            if (issue_q && last_q) begin
                done_meta_q    <= 1;
                drain_inflight <= 1;

                // latch drain meta for writeback
                wb_ktile_idx_q <= ktile_lat;
                wb_nblk_idx_q  <= nblk_lat;
                wb_mblk_idx_q  <= mblk_lat;
            end

            // clear delayed flags by default
            issue_q <= 0;
            first_q <= 0;
            last_q  <= 0;

            in_rd_en <= 0;
            w_rd_en  <= 0;


            // drain completion (previous tile)
            if (pe_array_done) begin
                drain_inflight <= 0;
            end

            // accept new job
            if (job_start && job_ready) begin
                running   <= 1;

                klen_lat  <= job_klen;
                ktile_lat <= job_ktile;
                nblk_lat  <= job_nblk;
                mblk_lat <= job_mblk;
                in_base_is_nblk_lat <= job_in_base_is_nblk;
                wbank_lat <= job_wbuf_sel;
                swap_ab_lat <= swap_ab;
                pe_a_is_qlog <= job_a_is_qlog;

                fc2_k32_lat <= (job_klen == 12'd32);
                w_base_lat   <= job_a_is_qlog ? WBUF_V_BASE :
                                ((job_klen == 12'd32) ? {job_nblk, 5'd0} : 12'd0);

                // input base:
                // - normal:  (job_in_base_is_nblk ? nblk : mblk) * MAX_KTILE + koff
                // - AV qlog: row-local SCOREBUF address koff (0..M_TOTAL-1)
                // - FC2 K=32: mblk * 32 + koff   (SCOREBUF tile input)
                in_base_lat <= job_in_base_calc;
                k_issue     <= 0;

                // prime read ports for k=0 (data appears next cycle)
                in_rd_en   <= 1;
                in_rd_addr <= job_in_base_calc;

                w_rd_en    <= 1;
                w_rd_bank  <= job_wbuf_sel;
                w_rd_addr  <= job_a_is_qlog ? WBUF_V_BASE :
                              ((job_klen == 12'd32) ? {job_nblk, 5'd0} : 12'd0);

            end else if (running) begin
                if (issue_fire) begin
                    // arm delayed valid flags for NEXT cycle
                    issue_q <= 1;
                    first_q <= (k_issue == 0);
                    last_q  <= is_last_k;

                    if (is_last_k) begin
                        // last MAC executed -> commit tile to PE out_data
                        running        <= 0;
                        in_rd_en <= 0;
                        w_rd_en  <= 0;

                    end else begin
                        // advance to next k, keep reads continuous
                        k_issue <= k_issue + 12'd1;

                        in_rd_en   <= 1;
                        in_rd_addr <= in_base_lat + (k_issue + 12'd1);

                        w_rd_en    <= 1;
                        w_rd_bank  <= wbank_lat;
                        w_rd_addr  <= w_base_lat + (k_issue + 12'd1);
                    end

                end else if (stall_last) begin
                    // hold ONLY the final MAC until drain finishes
                    // keep re-reading the last word so data stays valid
                    in_rd_en <= 1;
                    w_rd_en  <= 1;

                end else begin
                    in_rd_en <= 0;
                    w_rd_en  <= 0;
                end
            end
        end
    end

endmodule

`default_nettype wire
