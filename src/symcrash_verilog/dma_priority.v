/* 256 complete 64-bit descriptors = the documented 2 KiB queue payload.
 * Sixteen FIFO lists use the descriptor's wire priority: 15 highest, 0 lowest.
 * Equal priorities retain admission order. A started burst is not preempted.
 * The next descriptor is chosen only when the exclusive burst engine is idle.
 * No selected low-priority descriptor waits in front of new higher-priority
 * admissions while that engine is busy. Full queues backpressure admission.
 *
 * Descriptors use two 32-bit SRAM words, preserving every descriptor bit.
 * Linked-list/free-pool metadata is additional storage, counted separately.
 * Reset initializes 256 free links before accepting work and also resets
 * dma_burst; the external ordered RAM adapter must be flushed/reset with it.
 */
module dma_priority #(
    parameter MAX_CYCLES = 40000
) (
    input wire clk, rst_n,
    input wire submit_valid,
    output wire submit_ready,
    input wire [63:0] submit_descriptor,
    output wire initialized,
    output reg [8:0] queued_count,
    output wire completion_valid,
    input wire completion_ready,
    output wire [63:0] completion_descriptor,
    output wire [3:0] completion_status,
    output wire [31:0] completion_cycles,
    output wire completion_interrupt, deadline_missed,
    output wire ram_request_valid,
    input wire ram_request_ready,
    output wire ram_request_write,
    output wire [31:0] ram_request_address,
    output wire [127:0] ram_request_data,
    input wire ram_response_valid,
    output wire ram_response_ready,
    input wire ram_response_error,
    input wire [127:0] ram_response_data,
    output wire bram_enable, bram_write,
    output wire [15:0] bram_address,
    output wire [127:0] bram_write_data,
    input wire [127:0] bram_read_data
);
    localparam INIT = 4'd0, RUN = 4'd1, ENQUEUE_LOW = 4'd2,
               ENQUEUE_HIGH = 4'd3, PUBLISH = 4'd4, DEQUEUE_HIGH = 4'd5,
               CAPTURE_HIGH = 4'd6, ISSUE = 4'd7, SELECT_HEAD = 4'd8, READ_LOW = 4'd9;
    reg [3:0] state;
    reg [31:0] words [0:511];
    reg [7:0] links [0:255];
    reg [7:0] heads [0:15], tails [0:15];
    reg [15:0] nonempty;
    reg [7:0] free_head, init_index, slot, next_head;
    reg [8:0] free_count;
    reg [3:0] selected_priority;
    reg [63:0] incoming, outgoing;
    reg [31:0] word_read;
    reg [7:0] link_read;
    reg [3:0] highest;
    integer priority_index;
    always @* begin
        highest = 0;
        for (priority_index = 0; priority_index < 16; priority_index = priority_index + 1)
            if (nonempty[priority_index]) highest = priority_index;
    end

    wire engine_ready;
    wire dispatch = state == RUN && nonempty != 0 && engine_ready;
    wire admission = submit_valid && submit_ready;
    wire issue = rst_n && state == ISSUE;
    assign initialized = rst_n && state != INIT;
    assign submit_ready = rst_n && state == RUN && free_count != 0 && !dispatch;

    // One explicit read and write port per memory; multiplex addresses before
    // the RAM, not independent read data afterward. No array reset.
    reg [8:0] word_read_address, word_write_address;
    reg [7:0] link_read_address, link_write_address, link_write_data;
    reg [31:0] word_write_data;
    reg word_read_enable, word_write_enable, link_read_enable, link_write_enable;
    always @* begin
        word_read_address = 0; word_write_address = {slot,1'b0};
        word_write_data = incoming[31:0];
        word_read_enable = 0; word_write_enable = 0;
        link_read_address = free_head; link_write_address = 0; link_write_data = 0;
        link_read_enable = 0; link_write_enable = 0;
        if (rst_n) begin
            if (state == INIT) begin
                link_write_enable = 1; link_write_address = init_index;
                link_write_data = init_index + 1'b1;
            end else if (state == ENQUEUE_HIGH && nonempty[selected_priority]) begin
                link_write_enable = 1; link_write_address = tails[selected_priority];
                link_write_data = slot;
            end else if (issue && engine_ready) begin
                link_write_enable = 1; link_write_address = slot; link_write_data = free_head;
            end
            if (admission) link_read_enable = 1;
            else if (state == READ_LOW) begin
                link_read_enable = 1; link_read_address = slot;
            end
            if (state == ENQUEUE_LOW) word_write_enable = 1;
            if (state == ENQUEUE_HIGH) begin
                word_write_enable = 1; word_write_address = {slot,1'b1};
                word_write_data = incoming[63:32];
            end
            if (state == READ_LOW) begin
                word_read_enable = 1; word_read_address = {slot,1'b0};
            end else if (state == DEQUEUE_HIGH) begin
                word_read_enable = 1; word_read_address = {slot,1'b1};
            end
        end
    end
    always @(posedge clk) begin
        if (link_write_enable) links[link_write_address] <= link_write_data;
        if (link_read_enable) link_read <= links[link_read_address];
        if (word_write_enable) words[word_write_address] <= word_write_data;
        if (word_read_enable) word_read <= words[word_read_address];
    end

    // Constant-index enables avoid a general variable-bit/word write network
    // for this small register bank. Read selection and all stored bits remain
    // unchanged; only the selected FIFO's metadata can be updated.
    genvar p;
    generate for (p = 0; p < 16; p = p+1) begin : priority_metadata
        wire selected = selected_priority == p;
        always @(posedge clk) begin
            if (rst_n && selected) begin
                if (state == PUBLISH) begin
                    if (!nonempty[p]) heads[p] <= slot;
                    tails[p] <= slot;
                end else if (state == ISSUE && engine_ready && heads[p] != tails[p])
                    heads[p] <= next_head;
            end
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) nonempty[p] <= 0;
            else if (selected) begin
                if (state == PUBLISH) nonempty[p] <= 1;
                else if (state == ISSUE && engine_ready && heads[p] == tails[p]) nonempty[p] <= 0;
            end
        end
    end endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= INIT; init_index <= 0; free_head <= 0; free_count <= 0;
            queued_count <= 0;
        end else begin
            case (state)
                INIT: if (init_index == 255) begin
                    free_count <= 9'd256; state <= RUN;
                end else init_index <= init_index + 1'b1;
                RUN: if (dispatch) begin
                    selected_priority <= highest; state <= SELECT_HEAD;
                end else if (admission) begin
                    incoming <= submit_descriptor;
                    selected_priority <= submit_descriptor[7:4];
                    slot <= free_head; state <= ENQUEUE_LOW;
                end
                ENQUEUE_LOW: begin
                    free_head <= link_read; free_count <= free_count - 1'b1;
                    state <= ENQUEUE_HIGH;
                end
                ENQUEUE_HIGH: state <= PUBLISH;
                PUBLISH: begin
                    queued_count <= queued_count + 1'b1; state <= RUN;
                end
                // Separate priority encoding, head selection and BRAM address
                // setup. Admissions remain paused, so the chosen queue cannot
                // change while its descriptor is read.
                SELECT_HEAD: begin slot <= heads[selected_priority]; state <= READ_LOW; end
                READ_LOW: state <= DEQUEUE_HIGH;
                DEQUEUE_HIGH: begin
                    outgoing[31:0] <= word_read; next_head <= link_read;
                    state <= CAPTURE_HIGH;
                end
                CAPTURE_HIGH: begin
                    outgoing[63:32] <= word_read; state <= ISSUE;
                end
                ISSUE: if (engine_ready) begin
                    free_head <= slot; free_count <= free_count + 1'b1;
                    queued_count <= queued_count - 1'b1; state <= RUN;
                end
                default: state <= INIT;
            endcase
        end
    end

    dma_burst #(.MAX_CYCLES(MAX_CYCLES)) engine (
        .clk(clk), .rst_n(rst_n), .descriptor_valid(issue), .descriptor_ready(engine_ready),
        .descriptor(outgoing), .completion_valid(completion_valid), .completion_ready(completion_ready),
        .completion_descriptor(completion_descriptor), .completion_status(completion_status),
        .completion_cycles(completion_cycles), .completion_interrupt(completion_interrupt),
        .deadline_missed(deadline_missed), .ram_request_valid(ram_request_valid),
        .ram_request_ready(ram_request_ready), .ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address), .ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid), .ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error), .ram_response_data(ram_response_data),
        .bram_enable(bram_enable), .bram_write(bram_write), .bram_address(bram_address),
        .bram_write_data(bram_write_data), .bram_read_data(bram_read_data)
    );

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && state == RUN) begin
            if (free_count + queued_count != 9'd256 || free_count > 256 || queued_count > 256)
                $fatal(1, "Descriptor pool accounting");
            if ((queued_count == 0) != (nonempty == 0)) $fatal(1, "Priority mask accounting");
        end
        if (rst_n && issue && !engine_ready) $fatal(1, "Exclusive engine lost readiness before dispatch");
    end
`endif
endmodule
