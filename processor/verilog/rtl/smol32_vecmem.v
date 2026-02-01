/**
 * SMOL-32 Vector Memory Unit
 * Handles vector load/store operations (LVF, SVF)
 *
 * LVF vd, offset(rs1): Load 16 consecutive FP32 values into vector register
 * SVF vs, offset(rs1): Store vector register as 16 consecutive FP32 values
 *
 * Multi-cycle implementation: 16 memory accesses (one per lane)
 */

`include "smol32_defines.vh"

module smol32_vecmem (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire        is_vec_load,   // LVF instruction
    input  wire        is_vec_store,  // SVF instruction
    input  wire        execute,       // Execute strobe
    input  wire [4:0]  vl,            // Vector length (1-16)

    // Address from rs1 (base address only, NOT rs1+imm)
    input  wire [31:0] base_addr,
    // Stride from immediate field (bytes between elements)
    input  wire [15:0] stride,

    // Vector data (for store)
    input  wire [511:0] vs_data,      // Source vector register

    // Memory interface
    output reg  [31:0] vec_mem_addr,
    output reg  [31:0] vec_mem_wdata,
    input  wire [31:0] vec_mem_rdata,
    output reg         vec_mem_we,
    output reg         vec_mem_re,

    // Output (for load)
    output reg  [511:0] vd_data,      // Loaded vector data

    // Status
    output reg         vec_busy,
    output reg         vec_done
);

    // State machine
    reg [2:0] state;
    reg [4:0] lane_count;    // Current lane (0-15)
    reg [4:0] total_lanes;   // Total lanes to process
    reg [31:0] current_addr; // Current memory address
    reg [15:0] elem_stride;  // Stride between elements (from imm)
    reg is_load;             // 1 for load, 0 for store

    // Temporary storage for building loaded vector
    reg [511:0] load_buffer;

    localparam STATE_IDLE    = 3'd0;
    localparam STATE_ACCESS  = 3'd1;  // Issue memory access
    localparam STATE_WAIT    = 3'd2;  // Wait for memory
    localparam STATE_NEXT    = 3'd3;  // Process result, move to next lane
    localparam STATE_DONE    = 3'd4;

    // Extract lane from vector for store
    function [31:0] get_lane;
        input [511:0] vec;
        input [3:0] lane;
        begin
            case (lane)
                4'd0:  get_lane = vec[31:0];
                4'd1:  get_lane = vec[63:32];
                4'd2:  get_lane = vec[95:64];
                4'd3:  get_lane = vec[127:96];
                4'd4:  get_lane = vec[159:128];
                4'd5:  get_lane = vec[191:160];
                4'd6:  get_lane = vec[223:192];
                4'd7:  get_lane = vec[255:224];
                4'd8:  get_lane = vec[287:256];
                4'd9:  get_lane = vec[319:288];
                4'd10: get_lane = vec[351:320];
                4'd11: get_lane = vec[383:352];
                4'd12: get_lane = vec[415:384];
                4'd13: get_lane = vec[447:416];
                4'd14: get_lane = vec[479:448];
                4'd15: get_lane = vec[511:480];
            endcase
        end
    endfunction

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            lane_count <= 5'd0;
            total_lanes <= 5'd0;
            current_addr <= 32'd0;
            elem_stride <= 16'd4;
            is_load <= 1'b0;
            load_buffer <= 512'd0;
            vd_data <= 512'd0;
            vec_busy <= 1'b0;
            vec_done <= 1'b0;
            vec_mem_addr <= 32'd0;
            vec_mem_wdata <= 32'd0;
            vec_mem_we <= 1'b0;
            vec_mem_re <= 1'b0;
        end else begin
            vec_done <= 1'b0;  // Clear done each cycle

            case (state)
                STATE_IDLE: begin
                    if (execute && (is_vec_load || is_vec_store)) begin
                        state <= STATE_ACCESS;
                        lane_count <= 5'd0;
                        total_lanes <= vl;
                        current_addr <= base_addr;
                        elem_stride <= stride;  // Capture stride from imm
                        is_load <= is_vec_load;
                        load_buffer <= 512'd0;
                        vec_busy <= 1'b1;

                        // Issue first memory access (lane 0 at base_addr)
                        vec_mem_addr <= base_addr;
                        if (is_vec_load) begin
                            vec_mem_re <= 1'b1;
                            vec_mem_we <= 1'b0;
                        end else begin
                            vec_mem_we <= 1'b1;
                            vec_mem_re <= 1'b0;
                            vec_mem_wdata <= get_lane(vs_data, 4'd0);
                        end
                    end
                end

                STATE_ACCESS: begin
                    // Wait one cycle for memory
                    state <= STATE_WAIT;
                end

                STATE_WAIT: begin
                    // Capture data for load
                    if (is_load) begin
                        case (lane_count[3:0])
                            4'd0:  load_buffer[31:0]     <= vec_mem_rdata;
                            4'd1:  load_buffer[63:32]    <= vec_mem_rdata;
                            4'd2:  load_buffer[95:64]    <= vec_mem_rdata;
                            4'd3:  load_buffer[127:96]   <= vec_mem_rdata;
                            4'd4:  load_buffer[159:128]  <= vec_mem_rdata;
                            4'd5:  load_buffer[191:160]  <= vec_mem_rdata;
                            4'd6:  load_buffer[223:192]  <= vec_mem_rdata;
                            4'd7:  load_buffer[255:224]  <= vec_mem_rdata;
                            4'd8:  load_buffer[287:256]  <= vec_mem_rdata;
                            4'd9:  load_buffer[319:288]  <= vec_mem_rdata;
                            4'd10: load_buffer[351:320]  <= vec_mem_rdata;
                            4'd11: load_buffer[383:352]  <= vec_mem_rdata;
                            4'd12: load_buffer[415:384]  <= vec_mem_rdata;
                            4'd13: load_buffer[447:416]  <= vec_mem_rdata;
                            4'd14: load_buffer[479:448]  <= vec_mem_rdata;
                            4'd15: load_buffer[511:480]  <= vec_mem_rdata;
                        endcase
                    end
                    state <= STATE_NEXT;
                end

                STATE_NEXT: begin
                    lane_count <= lane_count + 1;

                    if (lane_count + 1 >= total_lanes) begin
                        // Done with all lanes
                        state <= STATE_DONE;
                        vec_mem_re <= 1'b0;
                        vec_mem_we <= 1'b0;
                        if (is_load) begin
                            vd_data <= load_buffer;
                        end
                    end else begin
                        // More lanes to process - use stride from imm
                        current_addr <= current_addr + {16'b0, elem_stride};
                        vec_mem_addr <= current_addr + {16'b0, elem_stride};

                        if (is_load) begin
                            vec_mem_re <= 1'b1;
                        end else begin
                            vec_mem_we <= 1'b1;
                            vec_mem_wdata <= get_lane(vs_data, lane_count[3:0] + 4'd1);
                        end
                        state <= STATE_ACCESS;
                    end
                end

                STATE_DONE: begin
                    vec_busy <= 1'b0;
                    vec_done <= 1'b1;
                    state <= STATE_IDLE;
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule
