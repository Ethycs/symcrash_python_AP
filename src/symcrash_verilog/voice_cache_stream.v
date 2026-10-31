/* Shared voice cache plus one runtime decoder. The external state port
 * and renderer arbitrate acquisition; the accepted owner retains the DSP
 * port until release. Renderer wins a simultaneous acquisition, while an
 * existing external lease is never preempted. DMA is independent.
 * Coefficient offset/mode are caller-bound page layout, not a serializer.
 */
module voice_cache_stream #(parameter HEADROOM_BITS=2) (
    input wire clk,
    input wire rst_n,
    input wire prefetch_valid,
    output wire prefetch_ready,
    input wire [3:0] prefetch_slot,
    input wire [15:0] prefetch_voice_id,
    input wire [31:0] prefetch_tag,
    input wire [31:0] prefetch_ram_address,
    input wire [1:0] prefetch_class,
    output wire prefetch_rejected,
    input wire state_acquire_valid,
    output wire state_acquire_ready,
    input wire [3:0] state_acquire_slot,
    input wire [15:0] state_acquire_voice_id,
    input wire [31:0] state_acquire_tag,
    output wire state_acquire_result_valid,
    output wire state_acquire_hit,
    output wire state_lease_valid,
    output wire [3:0] state_lease_slot,
    input wire state_dsp_valid,
    output wire state_dsp_ready,
    input wire state_dsp_write,
    input wire [6:0] state_dsp_word,
    input wire [127:0] state_dsp_write_data,
    output wire state_dsp_response_valid,
    input wire state_dsp_response_ready,
    output wire [127:0] state_dsp_response_data,
    input wire state_release_valid,
    output wire state_release_ready,
    input wire retry_valid,
    output wire retry_ready,
    input wire [3:0] retry_slot,
    output wire retry_rejected,
    output wire dma_submit_valid,
    input wire dma_submit_ready,
    output wire [63:0] dma_submit_descriptor,
    input wire dma_done_valid,
    input wire [63:0] dma_done_descriptor,
    input wire [3:0] dma_done_status,
    input wire dma_bram_enable,
    input wire dma_bram_write,
    input wire [15:0] dma_bram_address,
    input wire [127:0] dma_bram_write_data,
    output wire [127:0] dma_bram_read_data,
    output wire completion_valid,
    output wire completion_write,
    output wire [3:0] completion_slot,
    output wire [3:0] completion_status,
    output wire [15:0] completion_voice_id,
    output wire [31:0] completion_tag,
    output wire [15:0] valid_mask,
    output wire [15:0] dirty_mask,
    output wire dma_busy,
    output wire memory_fault,
    input wire block_valid,
    output wire block_ready,
    input wire [3:0] block_slot,
    input wire [15:0] block_voice_id,
    input wire [31:0] block_context_tag,
    input wire [31:0] block_tag,
    input wire [1:0] block_k,
    input wire [6:0] block_coeff_word,
    output wire audio_valid,
    input wire audio_ready,
    output wire [63:0] audio_data,
    output wire audio_last,
    output wire [15:0] audio_voice_id,
    output wire [31:0] audio_tag,
    output wire [1:0] audio_status,
    output wire block_done
);
    wire c_acquire_valid;
    wire c_acquire_ready;
    wire [3:0] c_acquire_slot;
    wire [15:0] c_acquire_voice_id;
    wire [31:0] c_acquire_tag;
    wire c_acquire_result_valid;
    wire c_acquire_hit;
    wire c_lease_valid;
    wire [3:0] c_lease_slot;
    wire c_dsp_valid;
    wire c_dsp_ready;
    wire c_dsp_write;
    wire [6:0] c_dsp_word;
    wire [127:0] c_dsp_write_data;
    wire c_dsp_response_valid;
    wire c_dsp_response_ready;
    wire [127:0] c_dsp_response_data;
    wire c_release_valid;
    wire c_release_ready;
    wire r_acquire_valid;
    wire r_acquire_ready;
    wire [3:0] r_acquire_slot;
    wire [15:0] r_acquire_voice_id;
    wire [31:0] r_acquire_tag;
    wire r_acquire_result_valid;
    wire r_acquire_hit;
    wire r_dsp_valid;
    wire r_dsp_ready;
    wire [6:0] r_dsp_word;
    wire r_dsp_response_valid;
    wire r_dsp_response_ready;
    wire [127:0] r_dsp_response_data;
    wire r_release_valid;
    wire r_release_ready;
    // One accepted request register separates arbitration from page lookup.
    // 52 identity/address bits + pending/owner bits, one added acquire clock.
    // Reserve the selected slot against same-edge prefetch while enqueuing.
    reg request_pending,request_renderer;
    wire renderer_owner;
    reg [3:0] request_slot;
    reg [15:0] request_voice_id;
    reg [31:0] request_tag;
    wire front_ready=rst_n && !request_pending && c_acquire_ready;
    wire front_accept=front_ready && (r_acquire_valid || state_acquire_valid);
    wire [3:0] selected_slot=r_acquire_valid ? r_acquire_slot : state_acquire_slot;
    wire reserve_prefetch=front_accept && selected_slot==prefetch_slot;
    wire c_prefetch_valid=prefetch_valid && !reserve_prefetch;
    wire c_prefetch_ready;
    assign prefetch_ready=c_prefetch_ready && !reserve_prefetch;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) begin request_pending<=0;request_renderer<=0;end
        else begin
            if(c_acquire_valid && c_acquire_ready) begin
                request_pending<=0;
            end
            if(front_accept) begin request_pending<=1;request_renderer<=r_acquire_valid;end
        end
    always @(posedge clk) if(front_accept) begin
        request_slot<=selected_slot;
        request_voice_id<=r_acquire_valid ? r_acquire_voice_id : state_acquire_voice_id;
        request_tag<=r_acquire_valid ? r_acquire_tag : state_acquire_tag;
    end
    assign c_acquire_valid=rst_n && request_pending;
    assign r_acquire_ready=front_ready;
    assign state_acquire_ready=front_ready && !r_acquire_valid;
    assign c_acquire_slot=request_slot;
    assign c_acquire_voice_id=request_voice_id;
    assign c_acquire_tag=request_tag;
    assign r_acquire_result_valid=c_acquire_result_valid && renderer_owner;
    assign state_acquire_result_valid=c_acquire_result_valid && !renderer_owner;
    assign r_acquire_hit=c_acquire_hit;
    assign state_acquire_hit=c_acquire_hit;
    assign state_lease_valid=c_lease_valid && !renderer_owner;
    assign state_lease_slot=c_lease_slot;
    assign c_dsp_valid=state_dsp_valid;
    assign c_dsp_write=state_dsp_write;
    assign c_dsp_word=state_dsp_word;
    assign c_dsp_write_data=state_dsp_write_data;
    assign state_dsp_ready=c_dsp_ready;
    assign state_dsp_response_valid=c_dsp_response_valid;
    assign r_dsp_response_data=c_dsp_response_data;
    assign state_dsp_response_data=c_dsp_response_data;
    assign c_dsp_response_ready=state_dsp_response_ready;
    assign c_release_valid=state_release_valid;
    assign state_release_ready=c_release_ready;
    voice_state_cache cache (
        .clk(clk),
        .rst_n(rst_n),
        .prefetch_valid(c_prefetch_valid),
        .prefetch_ready(c_prefetch_ready),
        .prefetch_slot(prefetch_slot),
        .prefetch_voice_id(prefetch_voice_id),
        .prefetch_tag(prefetch_tag),
        .prefetch_ram_address(prefetch_ram_address),
        .prefetch_class(prefetch_class),
        .prefetch_rejected(prefetch_rejected),
        .acquire_valid(c_acquire_valid),
        .acquire_readonly(request_renderer),
        .lease_readonly(renderer_owner),
        .read_valid(r_dsp_valid),
        .read_ready(r_dsp_ready),
        .read_word(r_dsp_word),
        .read_response_valid(r_dsp_response_valid),
        .read_response_ready(r_dsp_response_ready),
        .read_release_valid(r_release_valid),
        .read_release_ready(r_release_ready),
        .acquire_ready(c_acquire_ready),
        .acquire_slot(c_acquire_slot),
        .acquire_voice_id(c_acquire_voice_id),
        .acquire_tag(c_acquire_tag),
        .acquire_result_valid(c_acquire_result_valid),
        .acquire_hit(c_acquire_hit),
        .lease_valid(c_lease_valid),
        .lease_slot(c_lease_slot),
        .dsp_valid(c_dsp_valid),
        .dsp_ready(c_dsp_ready),
        .dsp_write(c_dsp_write),
        .dsp_word(c_dsp_word),
        .dsp_write_data(c_dsp_write_data),
        .dsp_response_valid(c_dsp_response_valid),
        .dsp_response_ready(c_dsp_response_ready),
        .dsp_response_data(c_dsp_response_data),
        .release_valid(c_release_valid),
        .release_ready(c_release_ready),
        .retry_valid(retry_valid),
        .retry_ready(retry_ready),
        .retry_slot(retry_slot),
        .retry_rejected(retry_rejected),
        .dma_submit_valid(dma_submit_valid),
        .dma_submit_ready(dma_submit_ready),
        .dma_submit_descriptor(dma_submit_descriptor),
        .dma_done_valid(dma_done_valid),
        .dma_done_descriptor(dma_done_descriptor),
        .dma_done_status(dma_done_status),
        .dma_bram_enable(dma_bram_enable),
        .dma_bram_write(dma_bram_write),
        .dma_bram_address(dma_bram_address),
        .dma_bram_write_data(dma_bram_write_data),
        .dma_bram_read_data(dma_bram_read_data),
        .completion_valid(completion_valid),
        .completion_write(completion_write),
        .completion_slot(completion_slot),
        .completion_status(completion_status),
        .completion_voice_id(completion_voice_id),
        .completion_tag(completion_tag),
        .valid_mask(valid_mask),
        .dirty_mask(dirty_mask),
        .dma_busy(dma_busy),
        .memory_fault(memory_fault)
    );
    voice_cache_renderer #(.HEADROOM_BITS(HEADROOM_BITS)) renderer (
        .clk(clk),
        .rst_n(rst_n),
        .block_valid(block_valid),
        .block_ready(block_ready),
        .block_slot(block_slot),
        .block_voice_id(block_voice_id),
        .block_context_tag(block_context_tag),
        .block_tag(block_tag),
        .block_k(block_k),
        .block_coeff_word(block_coeff_word),
        .cache_acquire_valid(r_acquire_valid),
        .cache_acquire_ready(r_acquire_ready),
        .cache_acquire_slot(r_acquire_slot),
        .cache_acquire_voice_id(r_acquire_voice_id),
        .cache_acquire_tag(r_acquire_tag),
        .cache_acquire_result_valid(r_acquire_result_valid),
        .cache_acquire_hit(r_acquire_hit),
        .cache_dsp_valid(r_dsp_valid),
        .cache_dsp_ready(r_dsp_ready),
        .cache_dsp_word(r_dsp_word),
        .cache_dsp_response_valid(r_dsp_response_valid),
        .cache_dsp_response_ready(r_dsp_response_ready),
        .cache_dsp_response_data(r_dsp_response_data),
        .cache_release_valid(r_release_valid),
        .cache_release_ready(r_release_ready),
        .audio_valid(audio_valid),
        .audio_ready(audio_ready),
        .audio_data(audio_data),
        .audio_last(audio_last),
        .audio_voice_id(audio_voice_id),
        .audio_tag(audio_tag),
        .audio_status(audio_status),
        .block_done(block_done)
    );
endmodule
