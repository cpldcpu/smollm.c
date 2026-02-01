/**
 * SMOL-32 Top-Level Module
 * Core + Memory (256MB addressable)
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_top #(
    parameter MEM_SIZE = 32'h10000000,  // 256MB for full model forward pass
    parameter MEM_INIT_FILE = ""        // Optional memory initialization file
) (
    input  wire        clk,
    input  wire        rst_n,

    // Status outputs
    output wire        halted,
    output wire [31:0] pc,
    output wire [3:0]  state,

    // Debug interface (optional)
    input  wire        dbg_mem_re,
    input  wire [31:0] dbg_mem_addr,
    output wire [31:0] dbg_mem_rdata
);

    //=========================================================================
    // Core-Memory Interface
    //=========================================================================

    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [31:0] mem_rdata;
    wire        mem_we;
    wire        mem_re;

    //=========================================================================
    // Memory (Byte-addressable, Word-aligned access)
    //=========================================================================

    // Memory array - synthesizable with block RAM inference
    // Using words for efficiency
    localparam MEM_WORDS = MEM_SIZE / 4;
    reg [31:0] memory [0:MEM_WORDS-1];

    // Word address
    wire [31:0] word_addr = mem_addr[31:2];
    wire [31:0] dbg_word_addr = dbg_mem_addr[31:2];

    // Initialize memory if file provided
    initial begin
        if (MEM_INIT_FILE != "") begin
            $readmemh(MEM_INIT_FILE, memory);
        end
    end

    // Memory read
    assign mem_rdata = memory[word_addr];
    assign dbg_mem_rdata = memory[dbg_word_addr];

    // Memory write
    always @(posedge clk) begin
        if (mem_we) begin
            memory[word_addr] <= mem_wdata;
        end
    end

    //=========================================================================
    // SMOL-32 Core
    //=========================================================================

    smol32_core core (
        .clk        (clk),
        .rst_n      (rst_n),
        .mem_addr   (mem_addr),
        .mem_wdata  (mem_wdata),
        .mem_rdata  (mem_rdata),
        .mem_we     (mem_we),
        .mem_re     (mem_re),
        .halted     (halted),
        .pc_out     (pc),
        .state_out  (state)
    );

endmodule
