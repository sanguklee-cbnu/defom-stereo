`timescale 1ns / 1ps
module tb_corr_lookup_pixel;
  reg clk=0; always #5 clk=~clk;
  reg nRst=0;
  reg m_axi_read_done=0;
  wire m_axi_read_init;
  wire [31:0] m_axi_araddress;
  wire [16:0] m_axi_read_length;
  reg [31:0] disp_in=235;
  reg disp_in_valid=1;
  wire disp_in_request;
  reg [127:0] corr_in=0;
  reg corr_in_valid=0;
  wire [127:0] data_out;
  wire data_out_valid;
  reg data_out_ready=1;
  reg config_valid=0;
  reg [10:0] total_width=160,tile_width=1,tile_height=1;
  reg [27:0] offset_address_corr_level0=28'h0100000;
  reg [27:0] offset_address_corr_level1=28'h0200000;
  reg update_type=1;
  reg [20:0] rescale_param0=21'h1b4738;
  reg [20:0] rescale_param1=21'h1c7a9d;
  reg finish=0,compute_start=0;
  reg [10:0] current_height_iter=0,current_width_iter=41;
  wire compute_done;
  corr_lookup dut(.*);

  reg [127:0] row0[0:39];
  reg [127:0] row1[0:19];
  reg [7:0] expected[0:63];
  reg [7:0] frozen[0:63];
  integer request_count=0,return_count=0,output_beats=0,done_count=0;
  integer active=0,armed=1,level=0,index=0,remaining=0,gap=0;
  integer channel9_actual=-999;

  // A bounded transaction-level DDR responder, not a compute mock.
  // Addresses emitted by corr_lookup are offsets BEFORE the top master base.
  always @(negedge clk) begin
    corr_in_valid=0; m_axi_read_done=0;
    if (!nRst) begin active=0; armed=1; end
    else begin
      if (!m_axi_read_init) armed=1;
      if (!active && armed && m_axi_read_init) begin
        if (m_axi_araddress>=32'h01006680 && m_axi_araddress<32'h01006900) begin
          level=0; index=(m_axi_araddress-32'h01006680)>>4;
          if (index!=9 || m_axi_read_length!=3) $fatal(1,"Unexpected level0 DDR transaction");
        end else if (m_axi_araddress>=32'h02003340 && m_axi_araddress<32'h02003480) begin
          level=1; index=(m_axi_araddress-32'h02003340)>>4;
          if (index!=4 || m_axi_read_length!=3) $fatal(1,"Unexpected level1 DDR transaction");
        end else $fatal(1,"DDR address does not point to pixel41 original Corr row: %h",m_axi_araddress);
        remaining=m_axi_read_length; active=1; armed=0; gap=2;
        request_count=request_count+1;
        $display("DDR_REQUEST level=%0d address=%h beats=%0d",level,m_axi_araddress,m_axi_read_length);
      end
      if (active) begin
        if (gap>0) gap=gap-1;
        else begin
          corr_in=level ? row1[index] : row0[index]; corr_in_valid=1;
          return_count=return_count+1; index=index+1; remaining=remaining-1;
          if (remaining==0) begin m_axi_read_done=1; active=0; end
          else gap=1;
        end
      end
    end
  end

  always @(posedge clk) begin : compare_real_output
    integer lane,channel,got,want;
    #1;
    if (compute_done) done_count=done_count+1;
    if (nRst && data_out_valid && data_out_ready) begin
      if (output_beats>=4) $fatal(1,"Extra output beat");
      for(lane=0;lane<16;lane=lane+1) begin
        channel=output_beats*16+lane;
        got=$signed(data_out[lane*8 +: 8]);
        want=$signed(expected[channel]);
        if (got!==want) $fatal(1,"Channel mismatch ch=%0d actual=%0d python_RTL=%0d",channel,got,want);
        if (channel==9) begin
          channel9_actual=got;
          $display("FIRST_DU_DIFFERENCE ch=9 actual_RTL=%0d Python_RTL=%0d frozen_golden=%0d",got,want,$signed(frozen[9]));
        end
      end
      $display("OUTPUT_BEAT_PASS beat=%0d data=%h",output_beats,data_out);
      output_beats=output_beats+1;
    end
  end

  initial begin
    $readmemh("row0.hex",row0); $readmemh("row1.hex",row1);
    $readmemh("expected.hex",expected); $readmemh("frozen.hex",frozen);
    if ($signed(expected[9])!=21 || $signed(frozen[9])!=20) $fatal(1,"Frozen first-DU fixture differs");
    repeat(5) @(negedge clk); nRst=1;
    repeat(3) @(negedge clk); config_valid=1;
    @(negedge clk); config_valid=0;
    repeat(3) @(negedge clk); compute_start=1;
    @(negedge clk); compute_start=0;
    wait(output_beats==4 && done_count==1);
    repeat(10) @(negedge clk);
    if (request_count!=2 || return_count!=6 || channel9_actual!=21)
      $fatal(1,"Incorrect DDR/output coverage requests=%0d returns=%0d ch9=%0d",request_count,return_count,channel9_actual);
    $display("CORR_LOOKUP_PIXEL_PASS original_FSM=1 original_datapath=1 simulation_RAM_model=1 forced_internal_nets=0 pixels=1 valid_INT8_channels=18 padding_channels=46 DDR_requests=2 DDR_return_beats=6 output_beats=4 first_DU_ch9=21 frozen_ch9=20");
    $finish;
  end
  initial begin #200000; $fatal(1,"Simulation timed out outputs=%0d requests=%0d",output_beats,request_count); end
endmodule
