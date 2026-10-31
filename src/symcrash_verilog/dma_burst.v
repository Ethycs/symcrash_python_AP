/* WSE-STD-BRAM-RAM section 5 burst engine, 16-byte beats.
 * Descriptor: RAM[63:32], BRAM[31:16], length[15:8] x16 bytes,
 * priority[7:4], type[3:2], interrupt[1], valid[0].
 * READ=0, WRITE=1. Scatter/gather require an external list expander.
 * Addresses in this implementation profile are aligned byte addresses;
 * the unextended 16-bit BRAM field covers the lower 64 KiB only.
 *
 * RAM adapter: ordered request/response handshakes, one response per accepted
 * read OR write, write response only after commitment. It may backpressure
 * requests and delay responses. A read response is written directly to BRAM.
 * BRAM port: one-clock synchronous read, immediate enabled write, 128 bits.
 * Two data registers plus one reserved read tag sustain writes at one beat
 * per clock without overwriting a stalled RAM request. Payload bits are opaque.
 *
 * Only successful completion makes a staging buffer publishable. Error or
 * overdue transfers drain accepted work; they do not pretend cancellation.
 * Reset must also flush/reset the RAM adapter's outstanding transactions.
 * This engine does not implement the PHY, priority queue or buffer ownership.
 */
module dma_burst #(
    parameter MAX_CYCLES = 40000  // 0.4 ms DMA window at 100 MHz
) (
    input wire clk, rst_n,
    input wire descriptor_valid,
    output wire descriptor_ready,
    input wire [63:0] descriptor,
    output wire completion_valid,
    input wire completion_ready,
    output wire [63:0] completion_descriptor,
    output wire [3:0] completion_status,
    output wire [31:0] completion_cycles,
    output wire completion_interrupt,
    output wire deadline_missed,

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
    localparam IDLE = 2'd0, ACTIVE = 2'd1, REPORT = 2'd2;
    localparam BAD_DESCRIPTOR = 4'b0001, UNSUPPORTED = 4'b0010,
               MEMORY_ERROR = 4'b0100, OVERDUE = 4'b1000;
    reg [1:0] state;
    reg [63:0] command;
    reg [3:0] status;
    reg [31:0] elapsed;
    // 0..255 beats, including the completed endpoint, fit eight bits.
    reg [7:0] issued, completed, fetched;
    wire [7:0] length = command[15:8];
    wire write_transfer = command[2];
    reg [127:0] fifo [0:1];
    reg head, tail, read_pending;
    reg [1:0] fifo_count;

    wire [32:0] ram_end = {1'b0, descriptor[63:32]} + {21'd0, descriptor[15:8], 4'd0};
    wire [16:0] bram_end = {1'b0, descriptor[31:16]} + {5'd0, descriptor[15:8], 4'd0};
    // Preserve full endpoint widths, including address overflow. For a 2^n
    // limit, an endpoint is above it when a higher bit is set, or bit n and
    // any lower bit are both set. Exact endpoints remain legal. This avoids
    // a second wide carry comparison after the address sum.
    wire bad_descriptor = !descriptor[0] || descriptor[15:8] == 0
                       || descriptor[35:32] != 0 || descriptor[19:16] != 0
                       || (|ram_end[32:28]) || (ram_end[27] && (|ram_end[26:0]))
                       || (bram_end[16] && (|bram_end[15:0]));
    wire unsupported = descriptor[3];  // type 2 or 3
    wire request_fire = ram_request_valid && ram_request_ready;
    wire response_fire = ram_response_valid && ram_response_ready;
    wire pop = write_transfer && request_fire;
    wire [2:0] reserved = {1'b0, fifo_count} + {2'b0, read_pending} - {2'b0, pop};
    wire read_issue = rst_n && state == ACTIVE && write_transfer
                   && fetched < length && reserved < 3'd2;
    wire [3:0] active_errors = (response_fire && ram_response_error ? MEMORY_ERROR : 4'b0)
                            | (elapsed >= MAX_CYCLES ? OVERDUE : 4'b0);

    assign descriptor_ready = rst_n && state == IDLE;
    assign completion_valid = rst_n && state == REPORT;
    assign completion_descriptor = command;
    assign completion_status = status;
    assign completion_cycles = elapsed;
    assign completion_interrupt = completion_valid && command[1];
    // Visible even while a RAM response is still outstanding. The buffer
    // owner can hold/degrade at its deadline without treating DMA as done.
    assign deadline_missed = rst_n && state != IDLE && status[3];
    assign ram_request_valid = rst_n && state == ACTIVE && issued < length
                             && (!write_transfer || fifo_count != 0);
    assign ram_request_write = write_transfer;
    assign ram_request_address = command[63:32] + {20'd0, issued, 4'd0};
    assign ram_request_data = fifo[head];
    assign ram_response_ready = rst_n && state == ACTIVE && completed < issued;
    assign bram_write = rst_n && state == ACTIVE && !write_transfer
                      && response_fire && !ram_response_error;
    assign bram_enable = read_issue || bram_write;
    assign bram_address = command[31:16]
                        + (write_transfer ? {4'd0, fetched, 4'd0} : {4'd0, completed, 4'd0});
    assign bram_write_data = ram_response_data;

    // Payload registers do not need reset: valid/count metadata gates reads.
    always @(posedge clk) begin
        if (rst_n && read_pending) fifo[tail] <= bram_read_data;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; command <= 0; status <= 0; elapsed <= 0;
            issued <= 0; completed <= 0; fetched <= 0;
            head <= 0; tail <= 0; fifo_count <= 0; read_pending <= 0;
        end else begin
            case (state)
                IDLE: if (descriptor_valid) begin
                    command <= descriptor;
                    status <= (bad_descriptor ? BAD_DESCRIPTOR : 4'b0)
                            | (unsupported ? UNSUPPORTED : 4'b0);
                    elapsed <= 0; issued <= 0; completed <= 0; fetched <= 0;
                    head <= 0; tail <= 0; fifo_count <= 0; read_pending <= 0;
                    state <= (bad_descriptor || unsupported) ? REPORT : ACTIVE;
                end
                ACTIVE: begin
                    if (elapsed != 32'hffffffff) elapsed <= elapsed + 1'b1;
                    status <= status | active_errors;
                    if (request_fire) issued <= issued + 1'b1;
                    if (response_fire) begin
                        completed <= completed + 1'b1;
                        if (completed == length - 1'b1) state <= REPORT;
                    end
                    read_pending <= read_issue;
                    if (read_issue) fetched <= fetched + 1'b1;
                    if (read_pending) tail <= !tail;
                    if (pop) head <= !head;
                    case ({read_pending, pop})
                        2'b10: fifo_count <= fifo_count + 1'b1;
                        2'b01: fifo_count <= fifo_count - 1'b1;
                        default: fifo_count <= fifo_count;
                    endcase
                end
                REPORT: if (completion_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && state == ACTIVE) begin
            if (completed > issued || issued > length || fetched > length)
                $fatal(1, "DMA count invariant");
            if ({1'b0, fifo_count} + {2'b0, read_pending} > 3'd2)
                $fatal(1, "DMA write prefetch overflow");
            if (read_pending && fifo_count == 2 && !pop)
                $fatal(1, "DMA overwrote a stalled write");
        end
    end
    initial if (MAX_CYCLES < 1) $fatal(1, "DMA cycle window must be positive");
`endif
endmodule
