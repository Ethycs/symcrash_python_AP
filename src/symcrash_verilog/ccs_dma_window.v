/* Prepared-frame pipeline plus queued DMA and a single batch deadline.
 * Open a window before submitting prefetches. Sealing stops new prefetch
 * claims, lets the owner's pending descriptor reach the queue, then seals
 * the window. Deadline expiry can reject that pending descriptor instead.
 * Audio requests remain independent and degrade through ccs_stream.
 */
module ccs_dma_window #(
    parameter K=4, HEADROOM_BITS=2, WINDOW_CYCLES=40000,
    parameter [15:0] BASE=16'h8000
) (
    input wire clk, rst_n,
    input wire window_valid,
    output wire window_ready,
    input wire [31:0] window_tag,
    input wire seal,
    output wire window_busy, window_accepting,
    output wire result_valid,
    input wire result_ready,
    output wire [31:0] result_tag, elapsed_cycles, accepted_count, completed_count, rejected_count,
    output wire [3:0] result_status,
    output wire [8:0] outstanding_count,
    output wire deadline_missed, initialized,
    input wire prefetch_valid,
    output wire prefetch_ready,
    input wire [31:0] prefetch_tag, prefetch_ram_address,
    input wire [1:0] prefetch_class,
    output wire prefetch_bank,
    output reg prefetch_rejected,
    output reg [31:0] prefetch_rejected_tag, ingress_rejected_count,
    input wire block_valid,
    output wire block_ready,
    input wire [31:0] block_tag,
    output wire audio_valid,
    input wire audio_ready,
    output wire [63:0] audio_data,
    output wire audio_last,
    output wire [31:0] audio_tag,
    output wire audio_miss, block_done,
    input wire cancel_valid,
    input wire [31:0] cancel_tag,
    output wire [1:0] free_mask, ready_mask, reading_mask,
    output wire fault_valid,
    output wire [31:0] fault_tag,
    output wire [3:0] fault_status,
    output wire protocol_error, memory_fault,
    output wire ram_request_valid,
    input wire ram_request_ready,
    output wire ram_request_write,
    output wire [31:0] ram_request_address,
    output wire [127:0] ram_request_data,
    input wire ram_response_valid,
    output wire ram_response_ready,
    input wire ram_response_error,
    input wire [127:0] ram_response_data
);
    reg seal_pending;
    wire submit_valid, submit_ready, done_valid;
    wire [127:0] match_descriptors;
    wire [1:0] done_matches;
    wire [63:0] submit_descriptor, done_descriptor;
    wire [3:0] done_status;
    wire [31:0] done_service_cycles;
    wire dma_bram_enable, dma_bram_write;
    wire [15:0] dma_bram_address;
    wire [127:0] dma_bram_write_data;
    reg payload_enable, payload_write;
    reg [15:0] payload_address;
    reg [127:0] payload_write_data;
    wire owner_prefetch_ready;
    wire [3:0] dma_result_status;
    wire dma_deadline_missed;
    wire window_final_cycle;
    // Leave one clock for the owner's registered descriptor to become visible
    // before the deadline; otherwise an uncounted final-edge claim could
    // outlive a window that appeared empty at closing.
    // Within an active window, inner sealing/expiry implies seal_pending or
    // window_final_cycle. Those flags already exclude every closed state;
    // do not repeat the nested window_accepting gate in owner claim fanout.
    wire allow_prefetch=window_busy && !seal_pending && !seal && !window_final_cycle;
    wire canceled_prefetch=cancel_valid && cancel_tag==prefetch_tag;
    wire drop_prefetch=!allow_prefetch || canceled_prefetch;
    assign prefetch_ready=rst_n && initialized && (drop_prefetch || owner_prefetch_ready);
    wire ingress_drop=rst_n && initialized && prefetch_valid && drop_prefetch;
    assign result_status=dma_result_status | (ingress_rejected_count!=0 ? 4'b1000 : 4'b0000);
    assign deadline_missed=dma_deadline_missed || ingress_rejected_count!=0;
    // This prepared-frame client only submits RAM-to-BRAM transfers. Register
    // their write port to separate burst address arithmetic from payload-bank
    // selection/ownership checks. The registered owner callback follows this
    // write by one edge, so publication still requires the final stored word.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            payload_enable<=0;payload_write<=0;payload_address<=0;payload_write_data<=0;
        end else begin
            payload_enable<=dma_bram_enable;
            // Only payload_enable qualifies these fields at the consumer.
            // Sampling invalid fields too removes enable fanout to 145 FFs;
            // width/latency stay fixed, at the cost of invalid-cycle toggles.
            payload_write<=dma_bram_write;payload_address<=dma_bram_address;
            payload_write_data<=dma_bram_write_data;
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seal_pending<=0;prefetch_rejected<=0;prefetch_rejected_tag<=0;ingress_rejected_count<=0;
        end else begin
            prefetch_rejected<=ingress_drop;
            if (ingress_drop) prefetch_rejected_tag<=prefetch_tag;
            if (window_valid && window_ready) begin seal_pending<=seal;ingress_rejected_count<=0;end
            else begin
                if (window_busy && seal) seal_pending<=1;
                if (window_busy && ingress_drop && !canceled_prefetch && ingress_rejected_count!=32'hffffffff)
                    ingress_rejected_count<=ingress_rejected_count+1'b1;
            end
        end
    end

    dma_window #(.WINDOW_CYCLES(WINDOW_CYCLES)) dma (
        .clk(clk),.rst_n(rst_n),.window_valid(window_valid),.window_ready(window_ready),
        .window_tag(window_tag),.seal(seal_pending && !submit_valid),
        .window_busy(window_busy),.window_accepting(window_accepting),
        .window_final_cycle(window_final_cycle),
        .result_valid(result_valid),.result_ready(result_ready),.result_tag(result_tag),
        .elapsed_cycles(elapsed_cycles),.accepted_count(accepted_count),.completed_count(completed_count),
        .rejected_count(rejected_count),.result_status(dma_result_status),.outstanding_count(outstanding_count),
        .deadline_missed(dma_deadline_missed),.initialized(initialized),
        .submit_valid(submit_valid),.submit_ready(submit_ready),.submit_descriptor(submit_descriptor),
        .done_valid(done_valid),.done_descriptor(done_descriptor),.done_status(done_status),
        .match_descriptors(match_descriptors),.done_matches(done_matches),
        .done_service_cycles(done_service_cycles),.ram_request_valid(ram_request_valid),
        .ram_request_ready(ram_request_ready),.ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address),.ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid),.ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error),.ram_response_data(ram_response_data),
        .bram_enable(dma_bram_enable),.bram_write(dma_bram_write),.bram_address(dma_bram_address),
        .bram_write_data(dma_bram_write_data),.bram_read_data(128'd0)
    );
    ccs_stream #(.K(K),.HEADROOM_BITS(HEADROOM_BITS),.BASE(BASE),.MATCHED_COMPLETION(K==2)) stream (
        // The owner's ready already excludes matching cancellation. Passing
        // the direct admission branch avoids a second cancellation gate.
        .clk(clk),.rst_n(rst_n),.prefetch_valid(prefetch_valid && allow_prefetch),
        .prefetch_ready(owner_prefetch_ready),.prefetch_tag(prefetch_tag),.prefetch_ram_address(prefetch_ram_address),
        .prefetch_class(prefetch_class),.prefetch_bank(prefetch_bank),
        .dma_submit_valid(submit_valid),.dma_submit_ready(submit_ready),.dma_submit_descriptor(submit_descriptor),
        .dma_done_valid(done_valid),.dma_done_descriptor(done_descriptor),.dma_done_status(done_status),
        .dma_done_matches(done_matches),.dma_match_descriptors(match_descriptors),
        .dma_bram_enable(payload_enable),.dma_bram_write(payload_write),.dma_bram_address(payload_address),
        .dma_bram_write_data(payload_write_data),.block_valid(block_valid),.block_ready(block_ready),.block_tag(block_tag),
        .audio_valid(audio_valid),.audio_ready(audio_ready),.audio_data(audio_data),.audio_last(audio_last),
        .audio_tag(audio_tag),.audio_miss(audio_miss),.block_done(block_done),
        .cancel_valid(cancel_valid),.cancel_tag(cancel_tag),.free_mask(free_mask),.ready_mask(ready_mask),
        .reading_mask(reading_mask),.fault_valid(fault_valid),.fault_tag(fault_tag),.fault_status(fault_status),
        .protocol_error(protocol_error),.memory_fault(memory_fault)
    );
`ifndef SYNTHESIS
    always @(posedge clk) if (rst_n) begin
        if (allow_prefetch !== (window_accepting && !seal_pending && !seal && !window_final_cycle))
            $fatal(1,"Prefetch eligibility disagreed with the shared-window lifecycle");
        if (ingress_drop !== (prefetch_valid && prefetch_ready && drop_prefetch))
            $fatal(1,"Ingress rejection disagreed with its public handshake");
    end
`endif
endmodule
