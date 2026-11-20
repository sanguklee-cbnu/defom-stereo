`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/08/31 09:32:08
// Design Name: 
// Module Name: corr_lookup
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module corr_lookup #(
    parameter OP3_DW = 11,
    parameter OFF_AW = 28,
    parameter DWIDTH = 32,
    parameter QWIDTH = 8,
    parameter M_AXI_DATA_WIDTH = 128,
    parameter M_AXI_ADDR_WIDTH = 32,
    parameter ADDR_LSB = $clog2(M_AXI_DATA_WIDTH/8),
    parameter MUL_WIDTH = 16,
    parameter SHT_WIDTH = 5
    )(
    input wire clk,
    input wire nRst,
    
    input  wire m_axi_read_done,
    output reg m_axi_read_init,
    output reg [M_AXI_ADDR_WIDTH-1:0] m_axi_araddress,
    output reg [16:0] m_axi_read_length,    
    
    input wire [DWIDTH-1:0] disp_in,
    input wire disp_in_valid,
    output wire disp_in_request,
    
    input wire [M_AXI_DATA_WIDTH-1:0] corr_in,
    input wire corr_in_valid,
    
    output wire [M_AXI_DATA_WIDTH-1:0] data_out,
    output wire data_out_valid,
    input wire data_out_ready,        
    
    input wire config_valid,
    input wire [OP3_DW-1:0] total_width,  
    input wire [OP3_DW-1:0] tile_width,  
    input wire [OP3_DW-1:0] tile_height,
    input wire [OFF_AW-1:0] offset_address_corr_level0,
    input wire [OFF_AW-1:0] offset_address_corr_level1,    
    input wire update_type,
    input wire [MUL_WIDTH+SHT_WIDTH-1:0] rescale_param0,
    input wire [MUL_WIDTH+SHT_WIDTH-1:0] rescale_param1,
    input wire finish,
    
    
    input wire compute_start,
    input wire [OP3_DW-1:0] current_height_iter,  
    input wire [OP3_DW-1:0] current_width_iter,
    output wire compute_done    
    );
    
    localparam CONCAT = M_AXI_DATA_WIDTH/DWIDTH;
    localparam C_CONCAT = $clog2(CONCAT);
    
    localparam SCALE_UPDATE = 0;
    localparam DELTA_UPDATE = 1;    
    
    localparam STATE_IDLE           = 0;
    localparam STATE_HEIGHT_ITER    = 1;
    localparam STATE_WIDTH_ITER     = 2;
    localparam STATE_REQ_DISP       = 3;
    localparam STATE_RD_DISP        = 4;
    localparam STATE_COMP_START     = 5;
    localparam STATE_SCALE_DISP     = 6;
    localparam STATE_SUB_COORDS     = 7;
    localparam STATE_LEVEL_SHIFT    = 8;
    localparam STATE_INDEX_CLIP     = 9;
    localparam STATE_RD_CORR_START  = 10;
    localparam STATE_RD_CORR        = 11;
    localparam STATE_RD_CORR_WAIT   = 12;
    localparam STATE_RD_CORR_DONE   = 13;
    localparam STATE_1D_INTERP      = 14;
    localparam STATE_DONE           = 15;
    localparam STATE_WIDTH = $clog2(STATE_DONE+1);
    
    reg update_type_;
    reg [OP3_DW-1:0] total_width_;
    reg [OP3_DW-1:0] tile_width_;
    reg [OP3_DW-1:0] tile_height_;
    reg [OFF_AW-1:0] offset_address_corr_level0_;
    reg [OFF_AW-1:0] offset_address_corr_level1_;
    reg signed [MUL_WIDTH-1:0] rescale_mul_, rescale_mul_0, rescale_mul_1;
    reg [SHT_WIDTH-1:0] rescale_shift_, rescale_shift_0, rescale_shift_1;
    
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            update_type_ <= 'd0;
            total_width_ <= 'd0;
            tile_width_ <= 'd0;
            tile_height_ <= 'd0;
            offset_address_corr_level0_ <= 'd0;
            offset_address_corr_level1_ <= 'd0;
            {rescale_shift_0, rescale_mul_0} <= 'd0;
            {rescale_shift_1, rescale_mul_1} <= 'd0;
        end
        else begin
            update_type_ <= update_type;
            {rescale_shift_0, rescale_mul_0} <= rescale_param0;
            {rescale_shift_1, rescale_mul_1} <= rescale_param1;
            
            if(config_valid) begin
                total_width_ <= total_width;
                tile_width_ <= tile_width;
                tile_height_ <= tile_height;
                offset_address_corr_level0_ <= offset_address_corr_level0;
                offset_address_corr_level1_ <= offset_address_corr_level1;
            end
            else if(finish) begin
                total_width_ <= 'd0;
                tile_width_ <= 'd0;
                tile_height_ <= 'd0;
                offset_address_corr_level0_ <= 'd0;
                offset_address_corr_level1_ <= 'd0;
            end
        end
    end
    
    
    reg [3:0] interp_cnt;
    reg [STATE_WIDTH-1:0] state;
    reg [OP3_DW-1:0] current_height_iter_;
    reg [OP3_DW-1:0] current_width_iter_;
    
    reg [OP3_DW-1:0] internal_height_iter;
    reg [OP3_DW-1:0] internal_width_iter;
    reg [OP3_DW-1:0] current_height_point;
    reg [OP3_DW-1:0] current_width_point;    
    reg [3:0] level_cnt;
    reg signed [DWIDTH-1:0] disp_in_;
    reg signed [DWIDTH:0] scaled_disp;    
    reg signed [DWIDTH-1:0] coords;
    reg signed [DWIDTH-1:0] coords_base;
    reg signed [QWIDTH+1:0] frac_low, frac_high;
    wire signed [DWIDTH-QWIDTH-1:0] coords_int;
    wire [QWIDTH-1:0] coords_frac;
    reg signed [DWIDTH-QWIDTH-1:0] coords_int_prev;
    
    wire [3:0] total_level;
    wire signed [3:0] dx_size;
    reg [DWIDTH-1:0] index_start, index_end;
    reg [DWIDTH-1:0] index_start_prev, index_end_prev, read_length_prev;
    wire [DWIDTH-1:0] index_end_wire, read_length_wire;
    reg [3:0] index_length;
    reg [3:0] clip_length;
    wire signed [3:0] index_down_bound;
    wire signed [OP3_DW-1:0] index_upper_bound;

    wire [OFF_AW-1:0] offset_address_corr;
    reg [(M_AXI_ADDR_WIDTH-ADDR_LSB)-1:0] start_addr;
    wire level_shift;
    wire corr_reuse;
    assign offset_address_corr = (update_type_ == SCALE_UPDATE) ? offset_address_corr_level0_ : 
                                (level_cnt == 1) ? offset_address_corr_level0_ : (level_cnt == 2) ? offset_address_corr_level1_ : 'd0;
    assign level_shift = (update_type_ == SCALE_UPDATE || level_cnt == 1) ? 1'b0 : (level_cnt == 2) ? 1'b1 : 1'b0;
//    assign corr_reuse = (update_type_ == SCALE_UPDATE && level_cnt != 1) && (coords_int_prev == coords_int);
//    assign corr_reuse = (update_type_ == SCALE_UPDATE && level_cnt != 1) 
//                        && (index_start_prev[DWIDTH-1:C_CONCAT] == index_start[DWIDTH-1:C_CONCAT]) && (index_end_prev[DWIDTH-1:C_CONCAT] == index_end_wire[DWIDTH-1:C_CONCAT]);
    assign corr_reuse = (update_type_ == SCALE_UPDATE && level_cnt != 1) 
                        && (index_start_prev[DWIDTH-1:C_CONCAT] == index_start[DWIDTH-1:C_CONCAT]) && (read_length_prev >= read_length_wire);
    assign index_end_wire = index_start + index_length - 1'b1;
    assign read_length_wire = index_end_wire[DWIDTH-1:C_CONCAT] - index_start[DWIDTH-1:C_CONCAT];
    
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            start_addr <= 'd0;
        end
        else begin
            if(state == STATE_INDEX_CLIP) begin
                start_addr <= current_height_point * total_width_ + current_width_point;
            end
            else if(state == STATE_RD_CORR_START) begin            
                if(update_type_ == SCALE_UPDATE || level_cnt == 1)
                    start_addr <= start_addr * ( (total_width_>>2) + |total_width_[1:0] );
                else if(level_cnt == 2)
                    start_addr <= start_addr * ( (total_width_>>3) + |total_width_[2:1] );
            end
        end
    end
    
    reg [DWIDTH-1:0] corr_in_temp [0:15];
    reg [1:0] corr_in_cnt;
    reg signed [DWIDTH-1:0] corr_data [0:9];
    wire [C_CONCAT-1:0] corr_addr;
    
    assign corr_addr = index_start[C_CONCAT-1:0];
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            {corr_in_temp[3],  corr_in_temp[2],  corr_in_temp[1],  corr_in_temp[0]} <= 'd0;
            {corr_in_temp[7],  corr_in_temp[6],  corr_in_temp[5],  corr_in_temp[4]} <= 'd0;
            {corr_in_temp[11], corr_in_temp[10], corr_in_temp[9],  corr_in_temp[8]} <= 'd0;
            {corr_in_temp[15], corr_in_temp[14], corr_in_temp[13], corr_in_temp[12]} <= 'd0;
            corr_in_cnt <= 'd0;
            corr_data[0] <= 'd0;
            corr_data[1] <= 'd0;
            corr_data[2] <= 'd0;
            corr_data[3] <= 'd0;
            corr_data[4] <= 'd0;
            corr_data[5] <= 'd0;
            corr_data[6] <= 'd0;
            corr_data[7] <= 'd0;
            corr_data[8] <= 'd0;
            corr_data[9] <= 'd0;
        end
        else begin
            if(state == STATE_RD_CORR_START) begin
                if(!corr_reuse) begin
                    {corr_in_temp[3],  corr_in_temp[2],  corr_in_temp[1],  corr_in_temp[0]} <= 'd0;
                    {corr_in_temp[7],  corr_in_temp[6],  corr_in_temp[5],  corr_in_temp[4]} <= 'd0;
                    {corr_in_temp[11], corr_in_temp[10], corr_in_temp[9],  corr_in_temp[8]} <= 'd0;
                    {corr_in_temp[15], corr_in_temp[14], corr_in_temp[13], corr_in_temp[12]} <= 'd0;
                end
                corr_in_cnt <= 'd0;
                corr_data[0] <= 'd0;
                corr_data[1] <= 'd0;
                corr_data[2] <= 'd0;
                corr_data[3] <= 'd0;
                corr_data[4] <= 'd0;
                corr_data[5] <= 'd0;
                corr_data[6] <= 'd0;
                corr_data[7] <= 'd0;
                corr_data[8] <= 'd0;
                corr_data[9] <= 'd0;
            end
            else if(state == STATE_RD_CORR_WAIT) begin
                if(corr_in_valid) begin
                    corr_in_cnt <= corr_in_cnt + 1'b1;
                    case(corr_in_cnt)
                        0: {corr_in_temp[3],  corr_in_temp[2],  corr_in_temp[1],  corr_in_temp[0]}  <= corr_in;
                        1: {corr_in_temp[7],  corr_in_temp[6],  corr_in_temp[5],  corr_in_temp[4]}  <= corr_in;
                        2: {corr_in_temp[11], corr_in_temp[10], corr_in_temp[9],  corr_in_temp[8]}  <= corr_in;
                        3: {corr_in_temp[15], corr_in_temp[14], corr_in_temp[13], corr_in_temp[12]} <= corr_in;
                    endcase
                end
            end
            else if(state == STATE_RD_CORR_DONE) begin
                if(coords_int < dx_size) begin
                    corr_data[0] <= 'd0;
                    corr_data[1] <= (clip_length<2) ? corr_in_temp[1-clip_length] : 'd0;
                    corr_data[2] <= (clip_length<3) ? corr_in_temp[2-clip_length] : 'd0;
                    corr_data[3] <= (clip_length<4) ? corr_in_temp[3-clip_length] : 'd0;
                    corr_data[4] <= (clip_length<5) ? corr_in_temp[4-clip_length] : 'd0;
                    corr_data[5] <= (clip_length<6) ? corr_in_temp[5-clip_length] : 'd0;
                    corr_data[6] <= (clip_length<7) ? corr_in_temp[6-clip_length] : 'd0;
                    corr_data[7] <= (clip_length<8) ? corr_in_temp[7-clip_length] : 'd0;
                    corr_data[8] <= (clip_length<9) ? corr_in_temp[8-clip_length] : 'd0;
                    corr_data[9] <= (clip_length<10) ? corr_in_temp[9-clip_length] : 'd0;
                end
                else if(coords_int > (total_width_>>level_shift) - dx_size - 2) begin
                    corr_data[0] <= (clip_length<10) ? corr_in_temp[corr_addr]     : 'd0; 
                    corr_data[1] <= (clip_length<9)  ? corr_in_temp[corr_addr+1]   : 'd0; 
                    corr_data[2] <= (clip_length<8)  ? corr_in_temp[corr_addr+2]   : 'd0; 
                    corr_data[3] <= (clip_length<7)  ? corr_in_temp[corr_addr+3]   : 'd0; 
                    corr_data[4] <= (clip_length<6)  ? corr_in_temp[corr_addr+4]   : 'd0; 
                    corr_data[5] <= (clip_length<5)  ? corr_in_temp[corr_addr+5]   : 'd0; 
                    corr_data[6] <= (clip_length<4)  ? corr_in_temp[corr_addr+6]   : 'd0; 
                    corr_data[7] <= (clip_length<3)  ? corr_in_temp[corr_addr+7]   : 'd0; 
                    corr_data[8] <= (clip_length<2)  ? corr_in_temp[corr_addr+8]   : 'd0;
                    corr_data[9] <= 'd0;
                end
                else begin
                    corr_data[0] <= corr_in_temp[corr_addr];
                    corr_data[1] <= corr_in_temp[corr_addr+1];
                    corr_data[2] <= corr_in_temp[corr_addr+2];
                    corr_data[3] <= corr_in_temp[corr_addr+3];
                    corr_data[4] <= corr_in_temp[corr_addr+4];
                    corr_data[5] <= corr_in_temp[corr_addr+5];
                    corr_data[6] <= corr_in_temp[corr_addr+6];
                    corr_data[7] <= corr_in_temp[corr_addr+7];
                    corr_data[8] <= corr_in_temp[corr_addr+8];
                    corr_data[9] <= corr_in_temp[corr_addr+9];
                end
            end
        end
    end
    
    assign coords_int = coords_base[DWIDTH-1:QWIDTH];
    assign coords_frac = coords_base[QWIDTH-1:0];
    assign total_level = (update_type_ == SCALE_UPDATE) ? 8 : 2;
    assign dx_size = (update_type_ == SCALE_UPDATE) ? $signed(2) : $signed(4);
    assign index_down_bound = $signed(-1) - dx_size;
    assign index_upper_bound = (total_width_>>level_shift) + dx_size - 1;
    assign disp_in_request = (state == STATE_WIDTH_ITER) && (tile_width_ > internal_width_iter);
    assign compute_done = (state == STATE_DONE);
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            disp_in_ <= 'd0;
            scaled_disp <= 'd0;
            coords <= 'd0;
            coords_base <= 'd0;
            frac_low <= 'd0;
            frac_high <= 'd0;
            
            index_start <= 'd0;
            index_length <= 'd0;
            clip_length <= 'd0;
            index_end <= 'd0;
            index_start_prev <= 'd0;
            index_end_prev <= 'd0;
            read_length_prev <= 'd0;

            current_height_iter_ <= 'd0;        
            current_width_iter_ <= 'd0;        
            internal_height_iter <= 'd0;        
            internal_width_iter <= 'd0;        
            current_height_point <= 'd0;        
            current_width_point <= 'd0;    
            {rescale_shift_, rescale_mul_} <= 'd0;    
                
            m_axi_read_init <= 'd0;        
            m_axi_araddress <= 'd0;        
            m_axi_read_length <= 'd0;        
    
            coords_int_prev <= 'd0;        
            level_cnt <= 'd0;        
            state <= 'd0;
        end
        else begin
            case(state)
                STATE_IDLE: begin
                    disp_in_ <= 'd0;
                    scaled_disp <= 'd0;
                    coords <= 'd0;
                    coords_base <= 'd0;
                    frac_low <= 'd0;
                    frac_high <= 'd0;
                    index_start <= 'd0;
                    index_length <= 'd0;
                    clip_length <= 'd0;
                    index_end <= 'd0;
                    
                    {rescale_shift_, rescale_mul_} <= 'd0;    
                    
                    index_start_prev <= 'd0;
                    index_end_prev <= 'd0;                    
                    read_length_prev <= 'd0;                    
                    coords_int_prev <= 'd0;        
                    level_cnt <= 'd0;        
                        
                    if(compute_start) begin
                        current_height_iter_ <= current_height_iter;    
                        current_width_iter_ <= current_width_iter;                        
                        state <= STATE_HEIGHT_ITER;
                    end
                    else begin
                        current_height_iter_ <= 'd0;        
                        current_width_iter_ <= 'd0;    
                    end
                end
                STATE_HEIGHT_ITER: begin
                    if(tile_height_ > internal_height_iter) begin
                        internal_height_iter <= internal_height_iter + 1'b1;
                        current_height_point <= current_height_iter_ + internal_height_iter;
                        state <= STATE_WIDTH_ITER;
                    end
                    else begin
                        internal_height_iter <= 'd0;
                        current_height_point <= 'd0;
                        state <= STATE_DONE;
                    end
                end
                STATE_WIDTH_ITER: begin
                    if(tile_width_ > internal_width_iter) begin
                        coords[DWIDTH-1:QWIDTH] <= current_width_iter_ + internal_width_iter;
                        internal_width_iter <= internal_width_iter + 1'b1;
                        current_width_point <= current_width_iter_ + internal_width_iter;
                        state <= STATE_RD_DISP;
                    end
                    else begin
                        coords <= 'd0;
                        internal_width_iter <= 'd0;
                        current_width_point <= 'd0;
                        state <= STATE_HEIGHT_ITER;
                    end
                end
//                STATE_REQ_DISP: begin
//                    state <= STATE_RD_DISP;
//                end
                STATE_RD_DISP: begin
                    disp_in_ <= disp_in;
                    coords_int_prev <= 'd0;
                    index_start_prev <= 'd0;
                    index_end_prev <= 'd0;
                    read_length_prev <= 'd0;
                    if(update_type_ == SCALE_UPDATE)
                        state <= STATE_SCALE_DISP;
                    else
                        state <= STATE_SUB_COORDS;
                end
//                STATE_COMP_START: begin
//                    if(update_type_ == SCALE_UPDATE)
//                        state <= STATE_SCALE_DISP;
//                    else
//                        state <= STATE_SUB_COORDS;
//                end
                STATE_SCALE_DISP: begin
                    case(level_cnt)
                        0: scaled_disp <= ((disp_in_ >>> 2) + $signed(1)) >>> 1; // 0.125
                        1: scaled_disp <= ((disp_in_ >>> 1) + $signed(1)) >>> 1; // 0.25
                        2: scaled_disp <= (disp_in_+ $signed(1)) >>> 1; // 0.5
                        3: scaled_disp <= (((disp_in_*$signed(3)) >>> 1) + $signed(1)) >>> 1; // 0.75
                        4: scaled_disp <= disp_in_; // 1
                        5: scaled_disp <= (((disp_in_*$signed(5)) >>> 1) + $signed(1)) >>> 1; // 1.25
                        6: scaled_disp <= ((disp_in_*$signed(3))+ $signed(1)) >>> 1; // 1.5
                        7: scaled_disp <= disp_in_ <<< 1;
                    endcase
                    level_cnt <= level_cnt + 1'b1;
                    state <= STATE_SUB_COORDS;
                end
                STATE_SUB_COORDS: begin
                    if(update_type_ == SCALE_UPDATE) begin
                        coords_base <= coords - scaled_disp;
                        state <= STATE_INDEX_CLIP;
                    end
                    else begin
                        coords_base <= coords - disp_in_;
                        state <= STATE_LEVEL_SHIFT;
                    end
                end
                STATE_LEVEL_SHIFT: begin
                    level_cnt <= level_cnt + 1'b1;
                    if(level_cnt == 1)
                        coords_base <= (coords_base < 0) ? (coords_base) >>> 1 : (coords_base + $signed(1)) >>> 1;
//                        coords_base <= (coords_base + $signed(1)) >>> 1;
                    state <= STATE_INDEX_CLIP;
                end
                STATE_INDEX_CLIP: begin
                    if(coords_int < index_down_bound || coords_int > index_upper_bound) begin
                        index_start <= 'd0;
                        index_length <= 'd0;
                    end
                    else if(coords_int < dx_size) begin
                        index_start <= 'd0;
                        index_length <= coords_int + dx_size + $signed(2);
                    end
                    else if(coords_int > (total_width_>>level_shift) - dx_size - 2) begin
                        index_start <= coords_int - dx_size;
                        index_length <= (total_width_>>level_shift) + dx_size - coords_int;
                    end
                    else begin
                        index_start <= coords_int - dx_size;
                        index_length <= dx_size*2 + 2;
                    end
                
                    state <= STATE_RD_CORR_START;
                end
                STATE_RD_CORR_START: begin
                    clip_length <= dx_size*2 + 2 - index_length;
                    index_end <= index_start + index_length - 1'b1;
                    coords_int_prev <= coords_int;
                    index_start_prev <= index_start;
                    index_end_prev <= index_end_wire;
                    read_length_prev <= read_length_wire;
                    if(corr_reuse) begin
                        state <= STATE_RD_CORR_DONE;
                    end
                    else begin
                        if(index_length != 'd0)
                            state <= STATE_RD_CORR;
                        else
                            state <= STATE_RD_CORR_DONE;
                    end
                end
                STATE_RD_CORR: begin
                    m_axi_araddress[M_AXI_ADDR_WIDTH-1:ADDR_LSB] <= {offset_address_corr, {(M_AXI_ADDR_WIDTH-(OFF_AW+ADDR_LSB)){1'b0}}} + start_addr + index_start[DWIDTH-1:C_CONCAT];
                    m_axi_araddress[ADDR_LSB-1:0] <= {ADDR_LSB{1'b0}};
                    m_axi_read_init <= 1'b1;
                    m_axi_read_length <= index_end[DWIDTH-1:C_CONCAT] - index_start[DWIDTH-1:C_CONCAT] + 1'b1;

                    state <= STATE_RD_CORR_WAIT;
                end
                STATE_RD_CORR_WAIT: begin
                    if(m_axi_read_done) begin
                        m_axi_araddress <= 'd0;
                        m_axi_read_init <= 1'b0;
                        m_axi_read_length <= 'd0;
                        state <= STATE_RD_CORR_DONE;
                    end
                end
                STATE_RD_CORR_DONE: begin
                    frac_low <= 256 - coords_frac;
                    frac_high <= coords_frac;
                    state <= STATE_1D_INTERP;
                end
                STATE_1D_INTERP: begin
                    if(interp_cnt == dx_size*2) begin
                        if(level_cnt == total_level) begin
                            level_cnt <= 'd0;
                            state <= STATE_WIDTH_ITER;
                        end
                        else begin
                            if(update_type_ == SCALE_UPDATE)
                                state <= STATE_SCALE_DISP;
                            else
                                state <= STATE_SUB_COORDS;
                        end
                    end
                    
                    if(update_type_ == SCALE_UPDATE || level_cnt==1)
                        {rescale_shift_, rescale_mul_} <= {rescale_shift_0, rescale_mul_0};
                    else if(level_cnt==2)
                        {rescale_shift_, rescale_mul_} <= {rescale_shift_1, rescale_mul_1};
                end
                STATE_DONE: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
    

    reg signed [DWIDTH-1:0] interp_weight_sum;
    reg signed [DWIDTH-1:0] interp_result;
    reg interp_weight_sum_valid;
    reg interp_result_valid;
    wire signed [DWIDTH-1:0] interp_round;
//    wire signed [QWIDTH-1:0] interp_clip;
    
    assign interp_round = ((interp_weight_sum >>> 7) + $signed(1)) >>> 1;
//    assign interp_clip = (interp_round > 127) ? $signed(127) : (interp_round < -128) ? $signed(-128) : interp_round[QWIDTH-1:0];
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            interp_cnt <= 'd0;
            interp_result_valid <= 'd0;
            interp_weight_sum_valid <= 'd0;
            interp_weight_sum <= 'd0;
            interp_result <= 'd0;
        end
        else begin
            if(state == STATE_1D_INTERP) begin
                interp_weight_sum <= frac_low * corr_data[interp_cnt] + frac_high * corr_data[interp_cnt+1];
                interp_cnt <= interp_cnt + 1'b1;
                interp_weight_sum_valid <= 1'b1;
            end
            else begin
                interp_weight_sum <= 'd0;
                interp_cnt <= 'd0;
                interp_weight_sum_valid <= 'd0;
            end
            
            if(interp_weight_sum_valid)
                interp_result <= interp_round;
            else
                interp_result <= 'd0;
                
            interp_result_valid <= interp_weight_sum_valid;
        end
    end
    
    reg signed [DWIDTH+MUL_WIDTH-1:0] mul_out;    
    reg signed [DWIDTH-1:0] shift_out;  
    reg [1:0] rescale_valid;  
        
    wire signed [DWIDTH-1:0] rescale_in;  
    wire signed [DWIDTH-2:0] rescale_out;   
    wire signed [QWIDTH-1:0] rescale_clip;
    
    assign rescale_in = interp_result;
    assign rescale_out = shift_out[DWIDTH-1:1];
    assign rescale_clip = (rescale_out > 127) ? $signed(127) : (rescale_out < -128) ? $signed(-128) : rescale_out[QWIDTH-1:0];
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            mul_out <= 'd0;
            shift_out <= 'd0;
            rescale_valid <= 'd0;
        end
        else begin
            mul_out <= rescale_in * rescale_mul_;
            shift_out <= (mul_out >>> rescale_shift_) + $signed(1);
            rescale_valid <= {rescale_valid[0], interp_result_valid};
        end
    end
    
    reg [5:0] rescale_data_cnt;
    reg [QWIDTH-1:0] rescale_data_temp [0:39];   
    reg rescale_data_push;
    integer i;
    
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            rescale_data_cnt <= 'd0;
            for (i=0; i<40; i=i+1) begin
                rescale_data_temp[i] <= 'd0;
            end
            rescale_data_push <= 'd0;
        end
        else begin
            if(rescale_valid[1]) begin
                if(rescale_data_cnt == (dx_size*2+1) * total_level - 1'b1) begin
                    rescale_data_cnt <= 'd0;
                    rescale_data_push <= 1'b1;
                end
                else
                    rescale_data_cnt <= rescale_data_cnt + 1'b1;
                rescale_data_temp[rescale_data_cnt] <= rescale_clip;
            end
            else if(rescale_data_push) begin
                rescale_data_push <= 1'b0;
                for (i=0; i<40; i=i+1) begin
                    rescale_data_temp[i] <= 'd0;
                end
            end
        end
    end
    
    genvar index;
    
    localparam MEM_WIDTH = M_AXI_DATA_WIDTH;
    localparam MEM_DEPTH = 2 ** 10;
    localparam MEM_AW = $clog2(MEM_DEPTH);
    localparam MEM_NUM = 4;
    
    wire [MEM_WIDTH-1:0] mem_d_in [0:MEM_NUM-1];
    wire mem_w_en;
    wire [MEM_NUM-1:0] mem_r_en;
    reg [MEM_AW-1:0] mem_waddr;
    reg [MEM_AW-1:0] mem_raddr [0:MEM_NUM-1];
    wire [MEM_WIDTH-1:0] mem_d_out [0:MEM_NUM-1];
    reg  [MEM_NUM-1:0] mem_d_out_valid;
    wire [MEM_NUM-1:0] mem_empty;

    reg  mem_d_out_valid_skid;
    reg  [MEM_WIDTH-1:0] mem_d_out_skid;


    assign mem_d_in[0] = {rescale_data_temp[15], rescale_data_temp[14], rescale_data_temp[13], rescale_data_temp[12], 
                          rescale_data_temp[11], rescale_data_temp[10], rescale_data_temp[9],  rescale_data_temp[8], 
                          rescale_data_temp[7],  rescale_data_temp[6],  rescale_data_temp[5],  rescale_data_temp[4], 
                          rescale_data_temp[3],  rescale_data_temp[2],  rescale_data_temp[1],  rescale_data_temp[0]};
                          
    assign mem_d_in[1] = {rescale_data_temp[31], rescale_data_temp[30], rescale_data_temp[29], rescale_data_temp[28], 
                          rescale_data_temp[27], rescale_data_temp[26], rescale_data_temp[25], rescale_data_temp[24], 
                          rescale_data_temp[23], rescale_data_temp[22], rescale_data_temp[21], rescale_data_temp[20], 
                          rescale_data_temp[19], rescale_data_temp[18], rescale_data_temp[17], rescale_data_temp[16]};

    assign mem_d_in[2] = {{(MEM_WIDTH-QWIDTH*8){1'b0}},
                          rescale_data_temp[39], rescale_data_temp[38], rescale_data_temp[37], rescale_data_temp[36], 
                          rescale_data_temp[35], rescale_data_temp[34], rescale_data_temp[33], rescale_data_temp[32]};
    
    assign mem_d_in[3] = {(MEM_WIDTH){1'b0}};

    assign mem_w_en = rescale_data_push;
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            mem_waddr <= 'd0;
        end
        else begin
            if(mem_w_en)
                mem_waddr <= mem_waddr + 1'b1;
            else if(compute_start)
                mem_waddr <= 'd0;
        end
    end

    assign mem_r_en[0] = (data_out_ready | !data_out_valid) & !mem_empty[0];
    assign mem_r_en[1] = (data_out_ready | !data_out_valid) & !mem_empty[1] & mem_empty[0];
    assign mem_r_en[2] = (data_out_ready | !data_out_valid) & !mem_empty[2] & mem_empty[1];
    assign mem_r_en[3] = (data_out_ready | !data_out_valid) & !mem_empty[3] & mem_empty[2];
    generate
    for (index=0; index<MEM_NUM; index=index+1) begin
        assign mem_empty[index] = mem_raddr[index] == mem_waddr;
        
        always @(posedge clk, negedge nRst) begin
            if(!nRst) begin
                mem_raddr[index] <= 'd0;
                mem_d_out_valid[index] <= 'd0;
            end
            else begin
                if(mem_r_en[index])
                    mem_raddr[index] <= mem_raddr[index] + 1'b1;
                else if(compute_start)
                    mem_raddr[index] <= 'd0;
                    
                mem_d_out_valid[index] <= mem_r_en[index];
            end
        end        
        
        SimpleDP_Block_RAM #(.DW(MEM_WIDTH), .AW(MEM_AW), .DEPTH(MEM_DEPTH))
        Buffer (
            .clk(clk), 
            .c_en_a(mem_w_en), 
            .w_en_a(mem_w_en), 
            .addr_a(mem_waddr), 
            .d_in_a(mem_d_in[index]), 
            .c_en_b(mem_r_en[index]), 
            .addr_b(mem_raddr[index]), 
            .d_out_b(mem_d_out[index])
        );    
    end
    endgenerate
    
    always @(posedge clk, negedge nRst) begin
        if(!nRst) begin
            mem_d_out_valid_skid <= 0;
            mem_d_out_skid <= 0;
        end
        else begin
            if(!data_out_ready && data_out_valid) begin
                mem_d_out_valid_skid <= 1'b1;
                mem_d_out_skid <= data_out;
            end
            else begin
                mem_d_out_valid_skid <= 1'b0;
                mem_d_out_skid <= 'd0;
            end
        end
    end
    
    assign data_out_valid = ((|mem_d_out_valid) | mem_d_out_valid_skid);
    assign data_out = mem_d_out_valid_skid ? mem_d_out_skid : 
                      mem_d_out_valid[0] ? mem_d_out[0] : mem_d_out_valid[1] ? mem_d_out[1] :
                      mem_d_out_valid[2] ? mem_d_out[2] : mem_d_out_valid[3] ? mem_d_out[3] : 'd0;
    
endmodule
