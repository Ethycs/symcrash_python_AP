/* Ownership of two complete coefficient frames in the existing 32 KiB
 * coefficient partition. N=64, signed 16-bit coefficients, K=2/4/8 unchanged.
 * DMA completion is an already-consumed pulse from the shared queue. Reset
 * MUST flush that queue and its RAM adapter before either can run again.
 * Other clients must not write these reserved addresses.
 */
module coeff_buffer_owner #(
    parameter K = 4,
    parameter MATCHED_COMPLETION = 0,
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
    // Optional match flags captured with the existing DMA callback. The
    // combined window owns that register stage; standalone clients use the
    // original complete-descriptor comparison by default.
    input wire [1:0] dma_done_matches,
    output wire [127:0] dma_match_descriptors,
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
    wire [2:0] state [0:1];
    reg [31:0] tag [0:1];
    reg [63:0] descriptor [0:1];
    assign dma_match_descriptors={descriptor[1],descriptor[0]};
    wire [1:0] completed_matches=MATCHED_COMPLETION ? dma_done_matches :
        {dma_done_descriptor==descriptor[1],dma_done_descriptor==descriptor[0]};
    reg pending_valid, pending_bank;
    wire claim, submit, acquire;
    wire [1:0] match_ready;
    wire hit = |match_ready;
    wire selected = !free_mask[0];
    // Factor admission per destination bank. A free destination cannot
    // itself contain a live duplicate; only the other bank must be checked.
    // Bank 0 retains priority. This removes the shared duplicate/selection
    // chain from the metadata clock enables without adding storage/latency.
    wire [1:0] tag_matches={tag[1]==prefetch_tag,tag[0]==prefetch_tag};
    (* keep *) wire [1:0] claimable={
        !free_mask[0] && free_mask[1] && !tag_matches[0],
        free_mask[0] && (free_mask[1] || !tag_matches[1])};
    (* keep *) wire prefetch_eligible=rst_n && !pending_valid &&
        !(cancel_valid && cancel_tag==prefetch_tag) &&
        !(acquire && acquire_tag==prefetch_tag);
    wire [3:0] wire_priority = 4'd15 - {2'b0,prefetch_class};
    wire [15:0] destination = selected ? SECOND_BASE : BASE;
    wire [63:0] incoming = {prefetch_ram_address, destination,
                           FRAME_BEATS, wire_priority, 2'b00, 1'b1, 1'b1};
    assign prefetch_bank = selected;
    assign prefetch_ready = prefetch_eligible && |claimable;
    assign claim = prefetch_valid && prefetch_ready;
    assign dma_submit_valid = rst_n && pending_valid;
    assign dma_submit_descriptor = descriptor[pending_bank];
    assign submit = dma_submit_valid && dma_submit_ready;
    assign acquire_ready = rst_n && reading_mask == 0;
    assign acquire = acquire_valid && acquire_ready;

    genvar bank;
    generate for (bank=0; bank<2; bank=bank+1) begin : buffers
        // Keep the logical 0..6 policy while requesting one-hot physical
        // encoding of each independent controller. Expected cost: 7 FF per
        // bank instead of 3; actual mapping and timing must be measured.
        (* fsm_encoding = "one-hot" *) reg [2:0] current_state;
        assign state[bank]=current_state;
`ifndef SYNTHESIS
        // Only a simultaneous new FREE->WAITING claim can invalidate the
        // earlier comparison; WAITING ignores completion for state/faults.
        always @(posedge clk) if(rst_n && MATCHED_COMPLETION && dma_done_valid && state[bank]!=WAITING)
            if(dma_done_matches[bank] !== (dma_done_descriptor==descriptor[bank]))
                $fatal(1,"Prematched callback differs outside a new claim");
`endif
        wire expire = (cancel_valid && cancel_tag == tag[bank]) ||
                      (acquire && !hit && acquire_tag == tag[bank]);
        (* keep *) wire finished = dma_done_valid && completed_matches[bank];
        assign free_mask[bank] = state[bank] == FREE;
        assign ready_mask[bank] = state[bank] == READY;
        assign reading_mask[bank] = state[bank] == READING;
        assign match_ready[bank] = ready_mask[bank] && tag[bank] == acquire_tag &&
                                  !(cancel_valid && cancel_tag == acquire_tag);
        wire bank_claim=prefetch_valid && prefetch_eligible && claimable[bank];
        wire bank_submit=submit && pending_bank==bank;
        wire bank_acquire=acquire && match_ready[bank];
        wire bank_release=release_valid && release_bank==bank;
        reg [2:0] next_state;
        always @* begin
            next_state=current_state;
            case(current_state)
                FREE: if(bank_claim) next_state=WAITING;
                WAITING: if(bank_submit) next_state=expire ? DISCARD_FILL : FILLING;
                         else if(expire) next_state=DISCARD_WAIT;
                FILLING: if(finished) next_state=(!expire && dma_done_status==0) ? READY : FREE;
                         else if(expire) next_state=DISCARD_FILL;
                READY: if(expire) next_state=FREE;
                       else if(bank_acquire) next_state=READING;
                READING: if(bank_release) next_state=FREE;
                DISCARD_WAIT: if(bank_submit) next_state=DISCARD_FILL;
                DISCARD_FILL: if(finished) next_state=FREE;
                default: next_state=FREE;
            endcase
        end
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) current_state <= FREE;
            else current_state <= next_state;
        end
        always @(posedge clk) begin
            if (bank_claim) begin
                tag[bank] <= prefetch_tag;
                descriptor[bank] <= incoming;
            end
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
                if ((state[0] == FILLING || state[0] == DISCARD_FILL) && completed_matches[0]) begin
                    fault_valid <= 1; fault_tag <= tag[0]; fault_status <= dma_done_status;
                end else if ((state[1] == FILLING || state[1] == DISCARD_FILL) && completed_matches[1]) begin
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
