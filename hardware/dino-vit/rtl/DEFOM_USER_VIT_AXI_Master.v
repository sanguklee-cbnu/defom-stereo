`timescale 1 ns / 1 ps
// 24.07.11 Designed by Sangbo Park.
// Ph.D Course in MSIS Lab

	module DEFOM_USER_VIT_AXI_Master #
	(
		// Base address of targeted slave
		parameter C_M_TARGET_SLAVE_BASE_ADDR	= 32'h00000000,
		// Thread ID Width
		parameter C_M_AXI_ID_WIDTH	= 1,
		// Width of Address Bus
		parameter C_M_AXI_ADDR_WIDTH	= 32,
		// Width of Data Bus
		parameter C_M_AXI_DATA_WIDTH	= 256
	)
	(
		// Users to add ports here
        input wire  INIT_AXI_TXN_W,     // Initiate AXI write transactions
        input wire  INIT_AXI_TXN_R,     // Initiate AXI read transactions
        input wire [C_M_AXI_ADDR_WIDTH-1 : 0] 	init_axi_awaddr,
        input wire [C_M_AXI_ADDR_WIDTH-1 : 0] 	init_axi_araddr,
        input wire [16:0] init_write_length,
        input wire [16:0] init_read_length,
        output reg write_done,
        output reg read_done,
        output wire read_last,
        
        // [ADD] Source-side WDATA stream (from write_controller/controller_top)
        input  wire [C_M_AXI_DATA_WIDTH-1:0] S_WDATA,
        input  wire                          S_WVALID,
        output wire                          S_WREADY,
		// User ports ends
		// Do not modify the ports beyond this line

		// Global Clock Signal.
		input wire  M_AXI_ACLK,
		// Global Reset Singal. This Signal is Active Low
		input wire  M_AXI_ARESETN,
		// Master Interface Write Address ID
		output wire [C_M_AXI_ID_WIDTH-1 : 0] M_AXI_AWID,
		// Master Interface Write Address
		output wire [C_M_AXI_ADDR_WIDTH-1 : 0] M_AXI_AWADDR,
		// Burst length. The burst length gives the exact number of transfers in a burst
		output wire [7 : 0] M_AXI_AWLEN,
		// Burst size. This signal indicates the size of each transfer in the burst
		output wire [2 : 0] M_AXI_AWSIZE,
		// Burst type. The burst type and the size information, 
        // determine how the address for each transfer within the burst is calculated.
		output wire [1 : 0] M_AXI_AWBURST,
		// Lock type. Provides additional information about the
        // atomic characteristics of the transfer.
		output wire  M_AXI_AWLOCK,
		// Memory type. This signal indicates how transactions
        // are required to progress through a system.
		output wire [3 : 0] M_AXI_AWCACHE,
		// Protection type. This signal indicates the privilege
        // and security level of the transaction, and whether
        // the transaction is a data access or an instruction access.
		output wire [2 : 0] M_AXI_AWPROT,
		// Quality of Service, QoS identifier sent for each write transaction.
		output wire [3 : 0] M_AXI_AWQOS,
		// Write address valid. This signal indicates that
        // the channel is signaling valid write address and control information.
		output wire  M_AXI_AWVALID,
		// Write address ready. This signal indicates that
        // the slave is ready to accept an address and associated control signals
		input wire  M_AXI_AWREADY,
		// Master Interface Write Data.
		output wire [C_M_AXI_DATA_WIDTH-1 : 0] M_AXI_WDATA,
		// Write strobes. This signal indicates which byte
        // lanes hold valid data. There is one write strobe
        // bit for each eight bits of the write data bus.
		output wire [C_M_AXI_DATA_WIDTH/8-1 : 0] M_AXI_WSTRB,
		// Write last. This signal indicates the last transfer in a write burst.
		output wire  M_AXI_WLAST,
		// Write valid. This signal indicates that valid write
        // data and strobes are available
		output wire  M_AXI_WVALID,
		// Write ready. This signal indicates that the slave
        // can accept the write data.
		input wire  M_AXI_WREADY,
		// Master Interface Write Response.
		input wire [C_M_AXI_ID_WIDTH-1 : 0] M_AXI_BID,
		// Write response. This signal indicates the status of the write transaction.
		input wire [1 : 0] M_AXI_BRESP,
		// Write response valid. This signal indicates that the
        // channel is signaling a valid write response.
		input wire  M_AXI_BVALID,
		// Response ready. This signal indicates that the master
        // can accept a write response.
		output wire  M_AXI_BREADY,
		// Master Interface Read Address.
		output wire [C_M_AXI_ID_WIDTH-1 : 0] M_AXI_ARID,
		// Read address. This signal indicates the initial
        // address of a read burst transaction.
		output wire [C_M_AXI_ADDR_WIDTH-1 : 0] M_AXI_ARADDR,
		// Burst length. The burst length gives the exact number of transfers in a burst
		output wire [7 : 0] M_AXI_ARLEN,
		// Burst size. This signal indicates the size of each transfer in the burst
		output wire [2 : 0] M_AXI_ARSIZE,
		// Burst type. The burst type and the size information, 
        // determine how the address for each transfer within the burst is calculated.
		output wire [1 : 0] M_AXI_ARBURST,
		// Lock type. Provides additional information about the
        // atomic characteristics of the transfer.
		output wire  M_AXI_ARLOCK,
		// Memory type. This signal indicates how transactions
        // are required to progress through a system.
		output wire [3 : 0] M_AXI_ARCACHE,
		// Protection type. This signal indicates the privilege
        // and security level of the transaction, and whether
        // the transaction is a data access or an instruction access.
		output wire [2 : 0] M_AXI_ARPROT,
		// Quality of Service, QoS identifier sent for each read transaction
		output wire [3 : 0] M_AXI_ARQOS,
		// Write address valid. This signal indicates that
        // the channel is signaling valid read address and control information
		output wire  M_AXI_ARVALID,
		// Read address ready. This signal indicates that
        // the slave is ready to accept an address and associated control signals
		input wire  M_AXI_ARREADY,
		// Read ID tag. This signal is the identification tag
        // for the read data group of signals generated by the slave.
		input wire [C_M_AXI_ID_WIDTH-1 : 0] M_AXI_RID,
		// Master Read Data
//		input wire [C_M_AXI_DATA_WIDTH-1 : 0] M_AXI_RDATA,
		// Read response. This signal indicates the status of the read transfer
		input wire [1 : 0] M_AXI_RRESP,
		// Read last. This signal indicates the last transfer in a read burst
		input wire  M_AXI_RLAST,
		// Read valid. This signal indicates that the channel
        // is signaling the required read data.
		input wire  M_AXI_RVALID,
		// Read ready. This signal indicates that the master can
        // accept the read data and response information.
		output wire  M_AXI_RREADY
	);
                                                  
    // Maximum Burst Length. Supports 1, 2, 4, 8, 16, 32, 64, 128, 256 burst lengths
    localparam integer C_M_AXI_BURST_LEN	= (C_M_AXI_DATA_WIDTH==512) ? 64 : (C_M_AXI_DATA_WIDTH==256) ? 128 : 256;
	// C_TRANSACTIONS_NUM is the width of the index counter for number of write or read transaction.
	localparam integer C_TRANSACTIONS_NUM  = $clog2(C_M_AXI_BURST_LEN-1);
    // Bytes size of data width.
    localparam integer C_BYTES_SIZE        = $clog2(C_M_AXI_DATA_WIDTH/8);
	// Burst length for transactions, in C_M_AXI_DATA_WIDTHs.
	// Non-2^n lengths will eventually cause bursts across 4K address boundaries.
//	 localparam integer C_MASTER_LENGTH	= 12;
	// total number of burst transfers is master length divided by burst length and burst size
//	 localparam integer C_NO_BURSTS_REQ = C_MASTER_LENGTH-clogb2((C_M_AXI_BURST_LEN*C_M_AXI_DATA_WIDTH/8)-1);


	// AXI4LITE signals
	//AXI4 internal temp signals
	reg [C_M_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
	reg  	axi_awvalid;
	reg [16:0] axi_awlen_remain;
	reg  [8:0] axi_awlen_active;
	wire [8:0] axi_awlen;
	wire [8:0] axi_awlen_4k;
	wire [8:0] axi_awlen_mux;
//	reg [C_M_AXI_DATA_WIDTH-1 : 0] 	axi_wdata;
	reg  	axi_wlast;
	reg  	axi_wvalid;
	reg  	axi_bready;
	reg [C_M_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
	reg  	axi_arvalid;
	reg [16:0] axi_arlen_remain;
	wire [8:0] axi_arlen;
	wire [8:0] axi_arlen_4k;
	reg  	axi_rready;
	//write beat count in a burst
	reg  [C_TRANSACTIONS_NUM : 0] 	write_index;
	//read beat count in a burst
//	reg  [C_TRANSACTIONS_NUM : 0] 	read_index;
	//size of C_M_AXI_BURST_LEN length burst in bytes
	wire [C_TRANSACTIONS_NUM+C_BYTES_SIZE : 0] 	write_burst_size_bytes;
	wire [C_TRANSACTIONS_NUM+C_BYTES_SIZE : 0] 	read_burst_size_bytes;
	
	reg  	start_single_burst_write;
	reg  	start_single_burst_read;
	reg  	burst_write_active;
	reg  	burst_read_active;
	wire    aw_handshake;
	//Interface response error flags
	wire  	write_resp_error;
	wire  	read_resp_error;
	wire  	wnext;
	wire  	rnext;
	reg  	init_txn_ff_w;
	wire  	init_txn_pulse_w;	
	reg  	init_txn_ff_r;
	wire  	init_txn_pulse_r;


	// I/O Connections assignments
    reg  [C_M_AXI_DATA_WIDTH-1:0] wbuf_data;
    reg                           wbuf_valid;   
	//I/O Connections. Write Address (AW)
	assign M_AXI_AWID	= 'b0;
	//The AXI address is a concatenation of the target base address + active offset range
	assign M_AXI_AWADDR	= C_M_TARGET_SLAVE_BASE_ADDR + axi_awaddr;
	//Burst LENgth is number of transaction beats, minus 1
	assign M_AXI_AWLEN	= axi_awlen - 1'b1;
	//Size should be C_M_AXI_DATA_WIDTH, in 2^SIZE bytes, otherwise narrow bursts are used
	assign M_AXI_AWSIZE	= $clog2(($unsigned(C_M_AXI_DATA_WIDTH)/8)-1);
	//INCR burst type is usually used, except for keyhole bursts
	assign M_AXI_AWBURST	= 2'b01;
	assign M_AXI_AWLOCK	= 1'b0;
	//Update value to 4'b0011 if coherent accesses to be used via the Zynq ACP port. Not Allocated, Modifiable, not Bufferable. Not Bufferable since this example is meant to test memory, not intermediate cache. 
	assign M_AXI_AWCACHE	= 4'b0011;
	assign M_AXI_AWPROT	= 3'h0;
	assign M_AXI_AWQOS	= 4'h0;
	assign M_AXI_AWVALID	= axi_awvalid;
	//Write Data(W)
//	assign M_AXI_WDATA	= axi_wdata;
    assign M_AXI_WDATA = wbuf_data; 
	//All bursts are complete and aligned in this example
	assign M_AXI_WSTRB	= {(C_M_AXI_DATA_WIDTH/8){1'b1}};
	assign M_AXI_WLAST	= axi_wlast;
//	assign M_AXI_WVALID	= axi_wvalid;
    assign M_AXI_WVALID = axi_wvalid & wbuf_valid;
//	assign axi_wvalid	= M_AXI_WVALID;
	//Write Response (B)
	assign M_AXI_BREADY	= axi_bready;
	//Read Address (AR)
	assign M_AXI_ARID	= 'b0;
	assign M_AXI_ARADDR	= C_M_TARGET_SLAVE_BASE_ADDR + axi_araddr;
	//Burst LENgth is number of transaction beats, minus 1
	assign M_AXI_ARLEN	= axi_arlen - 1'b1;
	//Size should be C_M_AXI_DATA_WIDTH, in 2^n bytes, otherwise narrow bursts are used
	assign M_AXI_ARSIZE	= $clog2(($unsigned(C_M_AXI_DATA_WIDTH)/8)-1);
	//INCR burst type is usually used, except for keyhole bursts
	assign M_AXI_ARBURST	= 2'b01;
	assign M_AXI_ARLOCK	= 1'b0;
	//Update value to 4'b0011 if coherent accesses to be used via the Zynq ACP port. Not Allocated, Modifiable, not Bufferable. Not Bufferable since this example is meant to test memory, not intermediate cache. 
	assign M_AXI_ARCACHE	= 4'b0011;
	assign M_AXI_ARPROT	= 3'h0;
	assign M_AXI_ARQOS	= 4'h0;
	assign M_AXI_ARVALID	= axi_arvalid;
	//Read and Read Response (R)
	assign M_AXI_RREADY	= axi_rready;
	//Burst size in bytes
	assign write_burst_size_bytes	= axi_awlen << C_BYTES_SIZE;
	assign read_burst_size_bytes	= axi_arlen << C_BYTES_SIZE;
	assign init_txn_pulse_w	= (!init_txn_ff_w) && INIT_AXI_TXN_W;
	assign init_txn_pulse_r	= (!init_txn_ff_r) && INIT_AXI_TXN_R; 
      
	//Generate a pulse to initiate AXI transaction.
	always @(posedge M_AXI_ACLK)										      
	  begin                                                                        
	    // Initiates AXI transaction delay    
	    if (M_AXI_ARESETN == 0 )                                                   
	      begin                                                                    
	        init_txn_ff_w <= 1'b0;                                                   
	        init_txn_ff_r <= 1'b0;                                                   
	      end                                                                               
	    else                                                                       
	      begin  
	        init_txn_ff_w <= INIT_AXI_TXN_W;
	        init_txn_ff_r <= INIT_AXI_TXN_R;
	      end                                                                      
	  end     
	  
    assign S_WREADY = (!wbuf_valid) || wnext;
    always @(posedge M_AXI_ACLK) begin
      if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1) begin
        wbuf_valid <= 1'b0;
        // wbuf_data: no reset (reduces long reset->R timing)
      end else begin
        // consume without refill
        if (wnext && !(S_WVALID && S_WREADY)) begin
          wbuf_valid <= 1'b0;
        end
    
        // push (includes refill on same cycle as consume)
        if (S_WVALID && S_WREADY) begin
          wbuf_data  <= S_WDATA;
          wbuf_valid <= 1'b1;
        end
      end
    end

	//--------------------
	//Write Address Channel
	//--------------------

	// The purpose of the write address channel is to request the address and 
	// command information for the entire transaction.  It is a single beat
	// of information.

	// The AXI4 Write address channel in this example will continue to initiate
	// write commands as fast as it is allowed by the slave/interconnect.
	// The address will be incremented on each accepted address transaction,
	// by burst_size_byte to point to the next address. 

	  always @(posedge M_AXI_ACLK)                                   
	  begin                                                                
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1 )                                           
	        axi_awvalid <= 1'b0;                                           
	    // If previously not valid , start next transaction                
	    else if (~axi_awvalid && start_single_burst_write)                 
	        axi_awvalid <= 1'b1;                                           
	    /* Once asserted, VALIDs cannot be deasserted, so axi_awvalid must wait until transaction is accepted */                         
	    else if (M_AXI_AWREADY && axi_awvalid)                             
	        axi_awvalid <= 1'b0;                                           
	    else                                                               
	      axi_awvalid <= axi_awvalid;                                      
	  end                                                                
	                                                                       
	                                                                       
	// Next address after AWREADY indicates previous address acceptance    
	  always @(posedge M_AXI_ACLK)                                         
	  begin                                                                
	    if (M_AXI_ARESETN == 0)                                            
	      axi_awaddr <= 'b0;                                             
	    else if (init_txn_pulse_w == 1'b1)
	      axi_awaddr <= init_axi_awaddr;
	    else if (M_AXI_AWREADY && axi_awvalid)                             
	      axi_awaddr <= axi_awaddr + write_burst_size_bytes;                   
	    else                                                               
	      axi_awaddr <= axi_awaddr;                                        
	    end                                                                


	// Remain write burst length. to satisfy 4K address boundaries
	  assign axi_awlen = (axi_awlen_remain < axi_awlen_4k) ? axi_awlen_remain[8:0] : axi_awlen_4k;
	  assign axi_awlen_4k = (((13'h1000-axi_awaddr[11:0])>>C_BYTES_SIZE) < C_M_AXI_BURST_LEN) ? (13'h1000-axi_awaddr[11:0])>>C_BYTES_SIZE : C_M_AXI_BURST_LEN;
	  assign axi_awlen_mux = (M_AXI_AWREADY && axi_awvalid) ? axi_awlen : axi_awlen_active;
	  always @(posedge M_AXI_ACLK)                                         
	  begin                                                                
	    if (M_AXI_ARESETN == 0) 
	      begin
	        axi_awlen_remain <= 'd0;
	        axi_awlen_active <= 'd0;
	      end                                           
	    else if (init_txn_pulse_w == 1'b1)
	      begin
	        axi_awlen_remain <= init_write_length;
	      end
		else if (start_single_burst_write == 1'b1)
	      begin
			axi_awlen_active <= 'd0;
		  end
	    else if (M_AXI_AWREADY && axi_awvalid)
	      begin
	        axi_awlen_remain <= axi_awlen_remain - axi_awlen;
	        axi_awlen_active <= axi_awlen;
	      end
	    else
	      begin
	        axi_awlen_remain <= axi_awlen_remain; 
	        axi_awlen_active <= axi_awlen_active; 
	      end                                                          
	  end        

	//--------------------
	//Write Data Channel
	//--------------------

	//The write data will continually try to push write data across the interface.

	//The amount of data accepted will depend on the AXI slave and the AXI
	//Interconnect settings, such as if there are FIFOs enabled in interconnect.

	//Note that there is no explicit timing relationship to the write address channel.
	//The write channel has its own throttling flag, separate from the AW channel.

	//Synchronization between the channels must be determined by the user.

	//The simpliest but lowest performance would be to only issue one address write
	//and write data burst at a time.

	//In this example they are kept in sync by using the same address increment
	//and burst sizes. Then the AW and W channels have their transactions measured
	//with threshold counters as part of the user logic, to make sure neither 
	//channel gets too far ahead of each other.

	//Forward movement occurs when the write channel is valid and ready

//	  assign wnext = M_AXI_WREADY & axi_wvalid;                                   
	  assign wnext = M_AXI_WREADY & M_AXI_WVALID;	                                                                                    
      assign aw_handshake = M_AXI_AWREADY && axi_awvalid;
//	// WVALID logic, similar to the axi_awvalid always block above                      
	  always @(posedge M_AXI_ACLK)                                                      
	  begin                                                                             
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1 )                                                        
	      begin                                                                         
	        axi_wvalid <= 1'b0;                                                         
	      end                                                                           
	    // Start the W channel only after the matching AW command is accepted.
	    // This keeps WLAST aligned with the 4KB-split burst length.
	    else if (~axi_wvalid && aw_handshake)                               
	      begin                                                                         
	        axi_wvalid <= 1'b1;                                                         
	      end                                                                           
	    /* If WREADY and too many writes, throttle WVALID                               
	    Once asserted, VALIDs cannot be deasserted, so WVALID                           
	    must wait until burst is complete with WLAST */                                 
	    else if (wnext && axi_wlast)                                                    
	      axi_wvalid <= 1'b0;                                                           
	    else                                                                            
	      axi_wvalid <= axi_wvalid;                                                     
	  end                                                                               
	                                                                                    
	// WLAST generation on the MSB of a counter underflow                                
	// WVALID logic, similar to the axi_awvalid always block above                      
	  always @(posedge M_AXI_ACLK)                                                      
	  begin                                                                             
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1 || start_single_burst_write == 1'b1)                                                        
	      begin                                                                         
	        axi_wlast <= 1'b0;                                                          
	      end
        else if (aw_handshake)
          begin
            axi_wlast <= (axi_awlen == 9'd1);
          end
	    // axi_wlast is asserted when the write index                                   
	    // count reaches the penultimate count to synchronize                           
	    // with the last write data when write_index is b1111                           
        else if (write_index == axi_awlen_mux)
            axi_wlast <= 1'b0; 
	    else if (((write_index == axi_awlen_mux-2 && axi_awlen_mux >= 2) && wnext) || (axi_awlen_mux == 1 ))
	        axi_wlast <= 1'b1;                                                          
	    // Deassrt axi_wlast when the last write data has been                          
	    // accepted by the slave with a valid response                                  
	    else if (wnext)                                                                 
	      axi_wlast <= 1'b0;                                                                                                               
	    else                                                                            
	      axi_wlast <= axi_wlast;                                                       
	  end                                                                               
	                                                                                    
	/* Burst length counter. Uses extra counter register bit to indicate terminal       
	 count to reduce decode logic */                                                    
	  always @(posedge M_AXI_ACLK)                                                      
	  begin                                                                             
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1 || start_single_burst_write == 1'b1 || aw_handshake)    
	      begin                                                                         
	        write_index <= 0;                                                           
	      end                                                                           
	    else if (wnext && (write_index < axi_awlen_mux))                         
	      begin                                                                         
	        write_index <= write_index + 1;                                             
	      end                                                                           
	    else                                                                            
	      write_index <= write_index;                                                   
	  end                                                                               
	                                                                                    
	                                                                                    
//	/* Write Data Generator                                                             
//	 Data pattern is only a simple incrementing count from 0 for each burst  */         
//	  always @(posedge M_AXI_ACLK)                                                      
//	  begin                                                                             
//	    if (M_AXI_ARESETN == 0)                                                         
//	      axi_wdata <= 'b1;                                                             
//	    //else if (wnext && axi_wlast)                                                  
//	    //  axi_wdata <= 'b0;                                                           
//	    else if (wnext)                                                                 
//	      axi_wdata <= axi_wdata + 1;                                                   
//	    else                                                                            
//	      axi_wdata <= axi_wdata;                                                       
//	    end                                                                             


	//----------------------------
	//Write Response (B) Channel
	//----------------------------

	//The write response channel provides feedback that the write has committed
	//to memory. BREADY will occur when all of the data and the write address
	//has arrived and been accepted by the slave.

	//The write issuance (number of outstanding write addresses) is started by 
	//the Address Write transfer, and is completed by a BREADY/BRESP.

	//While negating BREADY will eventually throttle the AWREADY signal, 
	//it is best not to throttle the whole data channel this way.

	//The BRESP bit [1] is used indicate any errors from the interconnect or
	//slave for the entire write burst. This example will capture the error 
	//into the ERROR output. 

	  always @(posedge M_AXI_ACLK)                                     
	  begin                                                                 
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1 )                                            
	      axi_bready <= 1'b0;                                             
	    // accept/acknowledge bresp with axi_bready by the master when M_AXI_BVALID is asserted by slave                           
	    else if (M_AXI_BVALID && ~axi_bready)                               
	      axi_bready <= 1'b1;                                             
	    // deassert after one clock cycle                                   
	    else if (axi_bready)                                                
	      axi_bready <= 1'b0;                                             
	    // retain the previous value                                        
	    else                                                                
	      axi_bready <= axi_bready;                                         
	  end                                                                   
	                                                                        
	//Flag any write response errors                                        
	  assign write_resp_error = axi_bready & M_AXI_BVALID & M_AXI_BRESP[1]; 

	//----------------------------
	//Read Address Channel
	//----------------------------

	//The Read Address Channel (AW) provides a similar function to the
	//Write Address channel- to provide the tranfer qualifiers for the burst.

	//In this example, the read address increments in the same
	//manner as the write address channel.

	  always @(posedge M_AXI_ACLK)                                 
	  begin                                                              
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_r == 1'b1 )                                         
	      axi_arvalid <= 1'b0;                                         
	    // If previously not valid , start next transaction              
	    else if (~axi_arvalid && start_single_burst_read)                
	      axi_arvalid <= 1'b1;                                         
	    else if (M_AXI_ARREADY && axi_arvalid)                           
	      axi_arvalid <= 1'b0;                                         
	    else                                                             
	      axi_arvalid <= axi_arvalid;                                    
	  end                                                                
	                                                                     
	// Next address after ARREADY indicates previous address acceptance  
	  always @(posedge M_AXI_ACLK)                                       
	  begin                                                              
	    if (M_AXI_ARESETN == 0)                                          
	      axi_araddr <= 'b0;
	    else if (init_txn_pulse_r == 1'b1)
	      axi_araddr <= init_axi_araddr;                                
	    else if (M_AXI_ARREADY && axi_arvalid)                           
	      axi_araddr <= axi_araddr + read_burst_size_bytes;                 
	    else                                                             
	      axi_araddr <= axi_araddr;                                      
	  end                                                                

	// Remain read burst length. to satisfy 4K address boundaries
	  assign axi_arlen = (axi_arlen_remain < axi_arlen_4k) ? axi_arlen_remain[8:0] : axi_arlen_4k;
	  assign axi_arlen_4k = (((13'h1000-axi_araddr[11:0])>>C_BYTES_SIZE) < C_M_AXI_BURST_LEN) ? (13'h1000-axi_araddr[11:0])>>C_BYTES_SIZE : C_M_AXI_BURST_LEN;
	  always @(posedge M_AXI_ACLK)                                         
	  begin                                                                
	    if (M_AXI_ARESETN == 0) 
	      begin
	        axi_arlen_remain <= 'd0;
	      end                                           
	    else if (init_txn_pulse_r == 1'b1)
	      begin
	        axi_arlen_remain <= init_read_length;
	      end
	    else if (M_AXI_ARREADY && axi_arvalid)
	      begin
	        axi_arlen_remain <= axi_arlen_remain - axi_arlen;
	      end
	    else
	      begin
	        axi_arlen_remain <= axi_arlen_remain; 
	      end                                                          
	  end    

	//--------------------------------
	//Read Data (and Response) Channel
	//--------------------------------

	 // Forward movement occurs when the channel is valid and ready   
	  assign rnext = M_AXI_RVALID && axi_rready;                            
	                                                                                                                                         
      // The Read Data channel returns the results of the read request                      
	  always @(posedge M_AXI_ACLK)
	  begin
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_r == 1'b1)
	      axi_rready <= 1'b0;
	    else if (~axi_arvalid && start_single_burst_read)
	      axi_rready <= 1'b1;
	    else if (M_AXI_RVALID && axi_rready && M_AXI_RLAST)
	      axi_rready <= 1'b0;
	    else
	      axi_rready <= axi_rready;
	  end                            
	                                                                        
	//Flag any read response errors                                         
	  assign read_resp_error = axi_rready & M_AXI_RVALID & M_AXI_RRESP[1];  

    //implement master command interface state machine   
      localparam [0:0] IDLE  = 1'b0;
      localparam [0:0] WRITE = 1'b1;
      localparam [0:0] READ  = 1'b1;

	  reg [0:0] write_mst_exec_state;
	  always @ (posedge M_AXI_ACLK)
	  begin
	    if (M_AXI_ARESETN == 1'b0)
	      begin
	        write_mst_exec_state <= IDLE;
	        start_single_burst_write <= 1'b0;
	        write_done <= 1'b0;
	      end
	    else
	      begin
	        case (write_mst_exec_state)
	          IDLE: 
	            if(init_txn_pulse_w == 1'b1)
	              write_mst_exec_state  <= WRITE;
	            else
	              write_done <= 1'b0;
	              
	          WRITE:
	            if (M_AXI_BVALID && axi_bready && (write_index == axi_awlen_mux) && (axi_awlen_remain == 'd0))                                                                        
	              begin
	                write_done <= 1'b1;                                                                                         
	                write_mst_exec_state <= IDLE;//                                                              
	              end                                                                                           
	            else                                                                                            
	              begin                                                                                         
	                write_mst_exec_state  <= WRITE;                                                              
	                                                                                                            
	                if (~axi_awvalid && ~start_single_burst_write && ~burst_write_active &&
	                    (axi_awlen_remain != 'd0))                       
	                  begin                                                                                     
	                    start_single_burst_write <= 1'b1;                                                       
	                  end                                                                                       
	                else                                                                                        
	                  begin                                                                                     
	                    start_single_burst_write <= 1'b0; //Negate to generate a pulse                          
	                  end                                                                                       
	              end          
	        endcase
	      end
	  end   
	  
	  reg [0:0] read_mst_exec_state; 
	  assign read_last = (M_AXI_RVALID && axi_rready && M_AXI_RLAST && (axi_arlen_remain == 'd0));
	  always @ (posedge M_AXI_ACLK)
	  begin
	    if (M_AXI_ARESETN == 1'b0)
	      begin
	        read_mst_exec_state <= IDLE;
	        start_single_burst_read <= 1'b0;
	        read_done <= 1'b0;
	      end
	    else
	      begin
	        case (read_mst_exec_state)
	          IDLE: 
	            if(init_txn_pulse_r == 1'b1)
	              read_mst_exec_state  <= READ;
	            else
	              read_done <= 1'b0;
	              
	          READ:
	            if (M_AXI_RVALID && axi_rready && M_AXI_RLAST && (axi_arlen_remain == 'd0))
	              begin
	                read_done <= 1'b1;                                                                                         
	                read_mst_exec_state <= IDLE;                                                             
	              end                                                                                           
	            else                                                                                            
	              begin                                                                                         
	                read_mst_exec_state  <= READ;                                                               
	                                                                                                            
	                if (~axi_arvalid && ~burst_read_active && ~start_single_burst_read &&
	                    (axi_arlen_remain != 'd0))                         
	                  begin                                                                                     
	                    start_single_burst_read <= 1'b1;                                                        
	                  end                                                                                       
	               else                                                                                         
	                 begin                                                                                      
	                   start_single_burst_read <= 1'b0; //Negate to generate a pulse                            
	                 end                                                                                        
	              end    
	        endcase
	      end
	  end 
	                                                                                                                            
	  // burst_write_active signal is asserted when there is a burst write transaction                          
	  // is initiated by the assertion of start_single_burst_write. burst_write_active                          
	  // signal remains asserted until the burst write is accepted by the slave                                 
	  always @(posedge M_AXI_ACLK)                                                                              
	  begin                                                                                                     
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_w == 1'b1)                                                                                 
	      burst_write_active <= 1'b0;                                                                           
	                                                                                                            
	    //The burst_write_active is asserted when a write burst transaction is initiated                        
	    else if (start_single_burst_write)                                                                      
	      burst_write_active <= 1'b1;                                                                           
	    else if (M_AXI_BVALID && axi_bready)                                                                    
	      burst_write_active <= 0;                                                                              
	  end                                                                                                       
	                                                                                                                                                                                                      
	                                                                                                            
	  // burst_read_active signal is asserted when there is a burst write transaction                           
	  // is initiated by the assertion of start_single_burst_write. start_single_burst_read                     
	  // signal remains asserted until the burst read is accepted by the master                                 
	  always @(posedge M_AXI_ACLK)                                                                              
	  begin                                                                                                     
	    if (M_AXI_ARESETN == 0 || init_txn_pulse_r == 1'b1)                                                                                 
	      burst_read_active <= 1'b0;                                                                            
	                                                                                                            
	    //The burst_write_active is asserted when a write burst transaction is initiated                        
	    else if (start_single_burst_read)                                                                       
	      burst_read_active <= 1'b1;                                                                            
	    else if (M_AXI_RVALID && axi_rready && M_AXI_RLAST)                                                     
	      burst_read_active <= 0;                                                                               
	    end                                                                                                     

	endmodule
