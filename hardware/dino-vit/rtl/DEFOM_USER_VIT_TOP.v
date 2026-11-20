`timescale 1ns/1ps
`default_nettype none
// The proven compute RTL is unchanged. The second DDR master is permanently
// idle so existing interconnect slave numbering and NPU/Corr wiring survive.
module DEFOM_USER_VIT_TOP #(
    parameter integer AXI_ID_WIDTH = 1,
    parameter integer AXI_ADDR_WIDTH = 32,
    parameter integer AXI_DATA_WIDTH = 256,
    parameter integer S_AXI_ADDR_WIDTH = 12,
    parameter integer S_AXI_DATA_WIDTH = 32,
    parameter integer IDLE_AXI_DATA_WIDTH = 128
)(
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 100000000, ASSOCIATED_BUSIF S_AXI:M00_AXI:M01_AXI, ASSOCIATED_RESET aresetn" *)
input wire  clk,
(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aresetn, POLARITY ACTIVE_LOW" *)
input wire  aresetn,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, FREQ_HZ 100000000" *)
input wire [S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
input wire  S_AXI_AWVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
output wire  S_AXI_AWREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
input wire [S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
input wire [S_AXI_DATA_WIDTH/8-1:0] S_AXI_WSTRB,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
input wire  S_AXI_WVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
output wire  S_AXI_WREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
output wire [1:0] S_AXI_BRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
output wire  S_AXI_BVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
input wire  S_AXI_BREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
input wire [S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
input wire  S_AXI_ARVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
output wire  S_AXI_ARREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
output wire [S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
output wire [1:0] S_AXI_RRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
output wire  S_AXI_RVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
input wire  S_AXI_RREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWID" *)
output wire [AXI_ID_WIDTH-1:0] M00_AXI_AWID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWID" *)
output wire [AXI_ID_WIDTH-1:0] M01_AXI_AWID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWADDR" *)
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M00_AXI, PROTOCOL AXI4, FREQ_HZ 100000000" *)
output wire [AXI_ADDR_WIDTH-1:0] M00_AXI_AWADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWADDR" *)
output wire [AXI_ADDR_WIDTH-1:0] M01_AXI_AWADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWLEN" *)
output wire [7:0] M00_AXI_AWLEN,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWLEN" *)
output wire [7:0] M01_AXI_AWLEN,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWSIZE" *)
output wire [2:0] M00_AXI_AWSIZE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWSIZE" *)
output wire [2:0] M01_AXI_AWSIZE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWBURST" *)
output wire [1:0] M00_AXI_AWBURST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWBURST" *)
output wire [1:0] M01_AXI_AWBURST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWLOCK" *)
output wire  M00_AXI_AWLOCK,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWLOCK" *)
output wire  M01_AXI_AWLOCK,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWCACHE" *)
output wire [3:0] M00_AXI_AWCACHE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWCACHE" *)
output wire [3:0] M01_AXI_AWCACHE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWPROT" *)
output wire [2:0] M00_AXI_AWPROT,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWPROT" *)
output wire [2:0] M01_AXI_AWPROT,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWQOS" *)
output wire [3:0] M00_AXI_AWQOS,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWQOS" *)
output wire [3:0] M01_AXI_AWQOS,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWVALID" *)
output wire  M00_AXI_AWVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWVALID" *)
output wire  M01_AXI_AWVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI AWREADY" *)
input wire  M00_AXI_AWREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI AWREADY" *)
input wire  M01_AXI_AWREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI WDATA" *)
output wire [AXI_DATA_WIDTH-1:0] M00_AXI_WDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI WDATA" *)
output wire [IDLE_AXI_DATA_WIDTH-1:0] M01_AXI_WDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI WSTRB" *)
output wire [AXI_DATA_WIDTH/8-1:0] M00_AXI_WSTRB,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI WSTRB" *)
output wire [IDLE_AXI_DATA_WIDTH/8-1:0] M01_AXI_WSTRB,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI WLAST" *)
output wire  M00_AXI_WLAST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI WLAST" *)
output wire  M01_AXI_WLAST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI WVALID" *)
output wire  M00_AXI_WVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI WVALID" *)
output wire  M01_AXI_WVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI WREADY" *)
input wire  M00_AXI_WREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI WREADY" *)
input wire  M01_AXI_WREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI BID" *)
input wire [AXI_ID_WIDTH-1:0] M00_AXI_BID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI BID" *)
input wire [AXI_ID_WIDTH-1:0] M01_AXI_BID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI BRESP" *)
input wire [1:0] M00_AXI_BRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI BRESP" *)
input wire [1:0] M01_AXI_BRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI BVALID" *)
input wire  M00_AXI_BVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI BVALID" *)
input wire  M01_AXI_BVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI BREADY" *)
output wire  M00_AXI_BREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI BREADY" *)
output wire  M01_AXI_BREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARID" *)
output wire [AXI_ID_WIDTH-1:0] M00_AXI_ARID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARID" *)
output wire [AXI_ID_WIDTH-1:0] M01_AXI_ARID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARADDR" *)
output wire [AXI_ADDR_WIDTH-1:0] M00_AXI_ARADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARADDR" *)
output wire [AXI_ADDR_WIDTH-1:0] M01_AXI_ARADDR,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARLEN" *)
output wire [7:0] M00_AXI_ARLEN,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARLEN" *)
output wire [7:0] M01_AXI_ARLEN,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARSIZE" *)
output wire [2:0] M00_AXI_ARSIZE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARSIZE" *)
output wire [2:0] M01_AXI_ARSIZE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARBURST" *)
output wire [1:0] M00_AXI_ARBURST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARBURST" *)
output wire [1:0] M01_AXI_ARBURST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARLOCK" *)
output wire  M00_AXI_ARLOCK,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARLOCK" *)
output wire  M01_AXI_ARLOCK,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARCACHE" *)
output wire [3:0] M00_AXI_ARCACHE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARCACHE" *)
output wire [3:0] M01_AXI_ARCACHE,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARPROT" *)
output wire [2:0] M00_AXI_ARPROT,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARPROT" *)
output wire [2:0] M01_AXI_ARPROT,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARQOS" *)
output wire [3:0] M00_AXI_ARQOS,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARQOS" *)
output wire [3:0] M01_AXI_ARQOS,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARVALID" *)
output wire  M00_AXI_ARVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARVALID" *)
output wire  M01_AXI_ARVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI ARREADY" *)
input wire  M00_AXI_ARREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI ARREADY" *)
input wire  M01_AXI_ARREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RID" *)
input wire [AXI_ID_WIDTH-1:0] M00_AXI_RID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RID" *)
input wire [AXI_ID_WIDTH-1:0] M01_AXI_RID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RDATA" *)
input wire [AXI_DATA_WIDTH-1:0] M00_AXI_RDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RDATA" *)
input wire [IDLE_AXI_DATA_WIDTH-1:0] M01_AXI_RDATA,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RRESP" *)
input wire [1:0] M00_AXI_RRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RRESP" *)
input wire [1:0] M01_AXI_RRESP,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RLAST" *)
input wire  M00_AXI_RLAST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RLAST" *)
input wire  M01_AXI_RLAST,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RVALID" *)
input wire  M00_AXI_RVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RVALID" *)
input wire  M01_AXI_RVALID,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M00_AXI RREADY" *)
output wire  M00_AXI_RREADY,
(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M01_AXI RREADY" *)
output wire  M01_AXI_RREADY
);
vit_axi_top #(.AXI_ID_WIDTH(AXI_ID_WIDTH), .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
    .AXI_DATA_WIDTH(AXI_DATA_WIDTH), .S_AXI_ADDR_WIDTH(S_AXI_ADDR_WIDTH),
    .S_AXI_DATA_WIDTH(S_AXI_DATA_WIDTH),
    .DDR_TARGET_SLAVE_BASE_ADDR(32'h80000000)) u_verified_vit (
.clk(clk),
.aresetn(aresetn),
.s_axi_awaddr(S_AXI_AWADDR),
.s_axi_awvalid(S_AXI_AWVALID),
.s_axi_awready(S_AXI_AWREADY),
.s_axi_wdata(S_AXI_WDATA),
.s_axi_wstrb(S_AXI_WSTRB),
.s_axi_wvalid(S_AXI_WVALID),
.s_axi_wready(S_AXI_WREADY),
.s_axi_bresp(S_AXI_BRESP),
.s_axi_bvalid(S_AXI_BVALID),
.s_axi_bready(S_AXI_BREADY),
.s_axi_araddr(S_AXI_ARADDR),
.s_axi_arvalid(S_AXI_ARVALID),
.s_axi_arready(S_AXI_ARREADY),
.s_axi_rdata(S_AXI_RDATA),
.s_axi_rresp(S_AXI_RRESP),
.s_axi_rvalid(S_AXI_RVALID),
.s_axi_rready(S_AXI_RREADY),
.m00_axi_awid(M00_AXI_AWID),
.m00_axi_awaddr(M00_AXI_AWADDR),
.m00_axi_awlen(M00_AXI_AWLEN),
.m00_axi_awsize(M00_AXI_AWSIZE),
.m00_axi_awburst(M00_AXI_AWBURST),
.m00_axi_awlock(M00_AXI_AWLOCK),
.m00_axi_awcache(M00_AXI_AWCACHE),
.m00_axi_awprot(M00_AXI_AWPROT),
.m00_axi_awqos(M00_AXI_AWQOS),
.m00_axi_awvalid(M00_AXI_AWVALID),
.m00_axi_awready(M00_AXI_AWREADY),
.m00_axi_wdata(M00_AXI_WDATA),
.m00_axi_wstrb(M00_AXI_WSTRB),
.m00_axi_wlast(M00_AXI_WLAST),
.m00_axi_wvalid(M00_AXI_WVALID),
.m00_axi_wready(M00_AXI_WREADY),
.m00_axi_bid(M00_AXI_BID),
.m00_axi_bresp(M00_AXI_BRESP),
.m00_axi_bvalid(M00_AXI_BVALID),
.m00_axi_bready(M00_AXI_BREADY),
.m00_axi_arid(M00_AXI_ARID),
.m00_axi_araddr(M00_AXI_ARADDR),
.m00_axi_arlen(M00_AXI_ARLEN),
.m00_axi_arsize(M00_AXI_ARSIZE),
.m00_axi_arburst(M00_AXI_ARBURST),
.m00_axi_arlock(M00_AXI_ARLOCK),
.m00_axi_arcache(M00_AXI_ARCACHE),
.m00_axi_arprot(M00_AXI_ARPROT),
.m00_axi_arqos(M00_AXI_ARQOS),
.m00_axi_arvalid(M00_AXI_ARVALID),
.m00_axi_arready(M00_AXI_ARREADY),
.m00_axi_rid(M00_AXI_RID),
.m00_axi_rdata(M00_AXI_RDATA),
.m00_axi_rresp(M00_AXI_RRESP),
.m00_axi_rlast(M00_AXI_RLAST),
.m00_axi_rvalid(M00_AXI_RVALID),
.m00_axi_rready(M00_AXI_RREADY));
assign M01_AXI_AWID = 0;
assign M01_AXI_AWADDR = 0;
assign M01_AXI_AWLEN = 0;
assign M01_AXI_AWSIZE = $clog2(IDLE_AXI_DATA_WIDTH/8);
assign M01_AXI_AWBURST = 2'b01;
assign M01_AXI_AWLOCK = 0;
assign M01_AXI_AWCACHE = 0;
assign M01_AXI_AWPROT = 0;
assign M01_AXI_AWQOS = 0;
assign M01_AXI_AWVALID = 0;
assign M01_AXI_WDATA = 0;
assign M01_AXI_WSTRB = 0;
assign M01_AXI_WLAST = 0;
assign M01_AXI_WVALID = 0;
assign M01_AXI_BREADY = 1'b1;
assign M01_AXI_ARID = 0;
assign M01_AXI_ARADDR = 0;
assign M01_AXI_ARLEN = 0;
assign M01_AXI_ARSIZE = $clog2(IDLE_AXI_DATA_WIDTH/8);
assign M01_AXI_ARBURST = 2'b01;
assign M01_AXI_ARLOCK = 0;
assign M01_AXI_ARCACHE = 0;
assign M01_AXI_ARPROT = 0;
assign M01_AXI_ARQOS = 0;
assign M01_AXI_ARVALID = 0;
assign M01_AXI_RREADY = 1'b1;
endmodule
`default_nettype wire
