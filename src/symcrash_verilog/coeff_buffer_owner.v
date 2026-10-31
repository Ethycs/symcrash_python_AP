/* Ownership of two complete coefficient frames in the existing 32 KiB
 * coefficient partition. N=64, signed 16-bit coefficients, K=2/4/8 unchanged.
 * DMA completion is an already-consumed pulse from the shared queue. Reset
 * MUST flush that queue and its RAM adapter before either can run again.
 * Other clients must not write these reserved addresses.
 */
module coeff_buffer_owner #(
    parameter K = 4,
    parameter [15:0] BASE = 16'h8000
) (
    input wire clk, rst_n,
    input wire prefetch_valid,
    output wire prefetch_ready,
    input wire [31:0] prefetch_tag, prefetch_ram_address,
    input wire [1:0] prefetch_class,
    output wire prefetch_bank,
    output wire dma_submit_valid,
    input wire dma_submit_ready,
    output wire [63:0] dma_submit_descriptor,
    input wire dma_done_valid,
    input wire [63:0] dma_done_descriptor,
    input wire [3:0] dma_done_status,
    input wire acquire_valid,
    output wire acquire_ready,
    input wire [31:0] acquire_tag,
    output reg acquire_result_valid, acquire_hit, acquire_bank,
    output reg [15:0] acquire_address,
    input wire release_valid, release_bank,
    input wire cancel_valid,
    input wire [31:0] cancel_tag,
    output wire [1:0] free_mask, ready_mask, reading_mask,
    output reg fault_valid,
    output reg [31:0] fault_tag,
    output reg [3:0] fault_status,
    output reg protocol_error
);
    localparam FRAME_BYTES = 64 * K * 2;
    localparam [7:0] FRAME_BEATS = FRAME_BYTES / 16;
    localparam [15:0] SECOND_BASE = BASE + FRAME_BYTES;
    localparam FREE=3'd0, WAITING=3'd1, FILLING=3'd2, READY=3'd3,
               READING=3'd4, DISCARD_WAIT=3'd5, DISCARD_FILL=3'd6;
    reg [2:0] state [0:1];
    reg [31:0] tag [0:1];
    reg [63:0] descriptor [0:1];
    reg pending_valid, pending_bank;
    wire claim, submit, acquire;
    wire [1:0] match_ready;
    wire hit = |match_ready;
    wire selected = !free_mask[0];
    wire duplicate = (state[0] != FREE && tag[0] == prefetch_tag) ||
                     (state[1] != FREE && tag[1] == prefetch_tag);
    wire [3:0] wire_priority = 4'd15 - {2'b0,prefetch_class};
    wire [15:0] destination = selected ? SECOND_BASE : BASE;
    wire [63:0] incoming = {prefetch_ram_address, destination,
                           FRAME_BEATS, wire_priority, 2'b00, 1'b1, 1'b1};
    assign prefetch_bank = selected;
    assign prefetch_ready = rst_n && |free_mask && !pending_valid && !duplicate &&
                            !(cancel_valid && cancel_tag == prefetch_tag) &&
                            !(acquire && acquire_tag == prefetch_tag);
    assign claim = prefetch_valid && prefetch_ready;
    assign dma_submit_valid = rst_n && pending_valid;
    assign dma_submit_descriptor = descriptor[pending_bank];
    assign submit = dma_submit_valid && dma_submit_ready;
    assign acquire_ready = rst_n && reading_mask == 0;
    assign acquire = acquire_valid && acquire_ready;

    genvar bank;
    generate for (bank=0; bank<2; bank=bank+1) begin : buffers
        wire expire = (cancel_valid && cancel_tag == tag[bank]) ||
                      (acquire && !hit && acquire_tag == tag[bank]);
        wire finished = dma_done_valid && dma_done_descriptor == descriptor[bank];
        assign free_mask[bank] = state[bank] == FREE;
        assign ready_mask[bank] = state[bank] == READY;
        assign reading_mask[bank] = state[bank] == READING;
        assign match_ready[bank] = ready_mask[bank] && tag[bank] == acquire_tag &&
                                  !(cancel_valid && cancel_tag == acquire_tag);
        always @(posedge clk) begin
            if (claim && selected == bank) begin
                tag[bank] <= prefetch_tag;
                descriptor[bank] <= incoming;
            end
        end
        wire bank_claim=claim && selected==bank;
        wire bank_submit=submit && pending_bank==bank;
        wire bank_acquire=acquire && match_ready[bank];
        wire bank_release=release_valid && release_bank==bank;
        // Decode transitions into the existing destination values. The ORs
        // below directly form the same three stored bits, without a priority
        // mux across state branches. No extra state, latency or encoding.
        wire next_wait=(state[bank]==FREE && bank_claim) ||
                       (state[bank]==WAITING && !bank_submit && !expire);
        wire next_fill=(state[bank]==WAITING && bank_submit && !expire) ||
                       (state[bank]==FILLING && !finished && !expire);
        wire next_ready=(state[bank]==FILLING && finished && !expire && dma_done_status==0) ||
                        (state[bank]==READY && !expire && !bank_acquire);
        wire next_read=(state[bank]==READY && !expire && bank_acquire) ||
                       (state[bank]==READING && !bank_release);
        wire next_discard_wait=(state[bank]==WAITING && !bank_submit && expire) ||
                               (state[bank]==DISCARD_WAIT && !bank_submit);
        wire next_discard_fill=(state[bank]==WAITING && bank_submit && expire) ||
                               (state[bank]==DISCARD_WAIT && bank_submit) ||
                               (state[bank]==FILLING && !finished && expire) ||
                               (state[bank]==DISCARD_FILL && !finished);
        wire [2:0] next_state={next_read || next_discard_wait || next_discard_fill,
                               next_fill || next_ready || next_discard_fill,
                               next_wait || next_ready || next_discard_wait};
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) state[bank] <= FREE;
            else state[bank] <= next_state;
        end
    end endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_valid <= 0; pending_bank <= 0;
            acquire_result_valid <= 0; acquire_hit <= 0; acquire_bank <= 0; acquire_address <= 0;
            fault_valid <= 0; fault_tag <= 0; fault_status <= 0; protocol_error <= 0;
        end else begin
            if (claim) begin pending_valid <= 1; pending_bank <= selected; end
            else if (submit) pending_valid <= 0;
            acquire_result_valid <= acquire;
            if (acquire) begin
                acquire_hit <= hit;
                acquire_bank <= match_ready[1];
                acquire_address <= hit ? (match_ready[1] ? SECOND_BASE : BASE) : 16'd0;
            end
            fault_valid <= 0;
            if (dma_done_valid && dma_done_status != 0) begin
                if ((state[0] == FILLING || state[0] == DISCARD_FILL) && dma_done_descriptor == descriptor[0]) begin
                    fault_valid <= 1; fault_tag <= tag[0]; fault_status <= dma_done_status;
                end else if ((state[1] == FILLING || state[1] == DISCARD_FILL) && dma_done_descriptor == descriptor[1]) begin
                    fault_valid <= 1; fault_tag <= tag[1]; fault_status <= dma_done_status;
                end
            end
            if (release_valid && !reading_mask[release_bank]) protocol_error <= 1;
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (K != 2 && K != 4 && K != 8) $fatal(1, "K must be 2, 4 or 8");
        if (BASE < 16'h8000 || BASE % 16 != 0 || BASE + 2*FRAME_BYTES > 65536)
            $fatal(1, "Coefficient buffers must be aligned and inside 0x8000..0xffff");
    end
    always @(posedge clk) if (rst_n) begin
        if (reading_mask == 2'b11) $fatal(1, "Only one DSP lease is allowed");
        if (state[0] != FREE && state[1] != FREE && tag[0] == tag[1]) $fatal(1, "Duplicate live frame tag");
        if (submit && state[pending_bank] != WAITING && state[pending_bank] != DISCARD_WAIT)
            $fatal(1, "DMA submission without ownership");
    end
`endif
endmodule
