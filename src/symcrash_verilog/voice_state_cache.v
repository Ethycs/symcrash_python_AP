/* WSE-STD-VOICE-TIERS 5.3 / WSE-STD-BRAM-RAM 6: sixteen opaque 2 KiB
 * voice-state pages at 0x0000..0x7fff, eight active plus eight prefetch.
 * No serialization format or coefficient precision is changed here.
 * The scheduler chooses a physical slot and supplies full u16 voice identity
 * and a u32 context tag. Acquire is a bounded hit/miss, never a DMA wait.
 * One DSP lease and one independent cache DMA transaction can coexist.
 * Modified leases store the entire page before it becomes reusable. Failed
 * stores remain dirty and locked for retry; they cannot be evicted/reloaded.
 * RAM consistency is at successful page-store completion. Readers of external
 * voice state must obey this ownership contract. Reset must flush the shared
 * DMA queue/adapter and reestablish external state after interrupted stores.
 */
module voice_state_cache (
    input wire clk, rst_n,
    input wire prefetch_valid,
    output wire prefetch_ready,
    input wire [3:0] prefetch_slot,
    input wire [15:0] prefetch_voice_id,
    input wire [31:0] prefetch_tag, prefetch_ram_address,
    input wire [1:0] prefetch_class,
    output reg prefetch_rejected,
    input wire acquire_valid,
    input wire acquire_readonly,
    output wire acquire_ready,
    input wire [3:0] acquire_slot,
    input wire [15:0] acquire_voice_id,
    input wire [31:0] acquire_tag,
    output reg acquire_result_valid, acquire_hit,
    output reg lease_valid,
    output reg lease_readonly,
    output reg [3:0] lease_slot,
    input wire dsp_valid,
    output wire dsp_ready,
    input wire dsp_write,
    input wire [6:0] dsp_word,
    input wire [127:0] dsp_write_data,
    output wire dsp_response_valid,
    input wire dsp_response_ready,
    output wire [127:0] dsp_response_data,
    input wire release_valid,
    output wire release_ready,
    // Two logical clients share one physical DSP RAM port under one lease.
    // The read-only client has no write-data or write-enable input.
    input wire read_valid,
    output wire read_ready,
    input wire [6:0] read_word,
    output wire read_response_valid,
    input wire read_response_ready,
    input wire read_release_valid,
    output wire read_release_ready,
    input wire retry_valid,
    output wire retry_ready,
    input wire [3:0] retry_slot,
    output reg retry_rejected,
    output wire dma_submit_valid,
    input wire dma_submit_ready,
    output wire [63:0] dma_submit_descriptor,
    input wire dma_done_valid,
    input wire [63:0] dma_done_descriptor,
    input wire [3:0] dma_done_status,
    input wire dma_bram_enable, dma_bram_write,
    input wire [15:0] dma_bram_address,
    input wire [127:0] dma_bram_write_data,
    output reg [127:0] dma_bram_read_data,
    output reg completion_valid, completion_write,
    output reg [3:0] completion_slot, completion_status,
    output reg [15:0] completion_voice_id,
    output reg [31:0] completion_tag,
    output wire [15:0] valid_mask, dirty_mask,
    output wire dma_busy,
    output reg memory_fault
);
    localparam IDLE=2'd0, SUBMIT=2'd1, ACTIVE=2'd2, PUBLISH=2'd3;
    reg [1:0] dma_state;
    reg [3:0] dma_slot;
    reg dma_store;
    reg [63:0] descriptor;
    reg [7:0] transferred_words, next_word;
    reg access_error;
    // Redundant active-lease dirty flag removes a sixteen-way metadata read
    // from release arbitration. One extra FF, no data/capacity/latency change.
    reg lease_dirty;
    reg response_pending;
    // One added acquisition clock and 17 FF: sixteen local comparison bits
    // plus their valid flag. The selected slot is reserved until publication.
    reg query_pending;
    (* keep *) reg [15:0] query_matches;
    wire query_hit=|query_matches;
    reg [15:0] voice_id [0:15];
    reg [31:0] context_tag [0:15], ram_address [0:15];
    assign dma_busy=dma_state!=IDLE;
    assign dma_submit_valid=rst_n && dma_state==SUBMIT;
    assign dma_submit_descriptor=descriptor;
    // A pending DSP response always retains its lease (proved invariant).
    assign acquire_ready=rst_n && !lease_valid && !query_pending;
    wire acquire=acquire_valid && acquire_ready;
    // Compare full identities in each page before reducing the selected hit.
    // This removes serial wide metadata multiplexing from a registered query.
    // Full identity comparisons are captured before the selected-hit reduction.
    wire [15:0] acquire_matches;
    wire hit=|acquire_matches;
    assign dsp_ready=rst_n && lease_valid && !lease_readonly && (!response_pending || dsp_response_ready);
    assign read_ready=rst_n && lease_valid && lease_readonly && (!response_pending || read_response_ready);
    wire state_accept=dsp_valid && dsp_ready;
    wire read_accept=read_valid && read_ready;
    wire dsp_accept=state_accept || read_accept;
    wire memory_write=state_accept && dsp_write;
    wire [6:0] memory_word=lease_readonly ? read_word : dsp_word;
    assign dsp_response_valid=response_pending && !lease_readonly;
    assign read_response_valid=response_pending && lease_readonly;
    wire response_ready=lease_readonly ? read_response_ready : dsp_response_ready;
    assign release_ready=rst_n && lease_valid && !lease_readonly && !response_pending && !dsp_valid &&
        (!lease_dirty || !dma_busy);
    assign read_release_ready=rst_n && lease_valid && lease_readonly && !response_pending && !read_valid;
    wire state_release=release_valid && release_ready;
    wire read_release=read_release_valid && read_release_ready;
    wire release_lease=state_release || read_release;
    // Only the writable client can dirty or store a lease. Its exclusion
    // depends on its own request, never on read-only renderer flow control.
    wire release_dirty=state_release && lease_dirty;
    wire release_waiting=release_valid && lease_valid && !lease_readonly && lease_dirty;
    assign retry_ready=rst_n && !dma_busy && !release_waiting;
    wire retry=retry_valid && retry_ready;
    // Dirty implies valid; this invariant avoids a second metadata lookup.
    wire retry_eligible=dirty_mask[retry_slot] &&
        !(lease_valid && lease_slot==retry_slot);
    wire retry_store=retry && retry_eligible;
    wire bad_prefetch=(|prefetch_ram_address[3:0]) || prefetch_voice_id==16'hffff;
    wire available=!dirty_mask[prefetch_slot] && !(lease_valid && lease_slot==prefetch_slot) &&
        !(acquire && acquire_slot==prefetch_slot) && !(query_pending && lease_slot==prefetch_slot);
    assign prefetch_ready=rst_n && !dma_busy && !release_waiting && !retry_valid &&
        (bad_prefetch || available);
    wire prefetch=prefetch_valid && prefetch_ready && !bad_prefetch;
    (* keep *) wire prefetch_eligible=rst_n && !dma_busy && !release_waiting && !retry_valid && !bad_prefetch;
    wire [15:0] claim_mask;
    wire begin_store=release_dirty || retry_store;
    wire [3:0] store_slot=release_dirty ? lease_slot : retry_slot;
    wire begin_transfer=prefetch || begin_store;
    wire [3:0] begin_slot=begin_store ? store_slot : prefetch_slot;
    // Select each cached address independently of the late release/retry grant.
    // This adds a metadata read mux, preserving every stored bit and clock.
    wire [31:0] begin_ram=release_dirty ? ram_address[lease_slot] :
        retry_store ? ram_address[retry_slot] : prefetch_ram_address;
    wire [3:0] begin_priority=begin_store ? 4'd15 : 4'd15-{2'b0,prefetch_class};
    wire [63:0] incoming={begin_ram,1'b0,begin_slot,11'b0,8'd128,begin_priority,1'b0,begin_store,2'b11};

    wire cache_address=!dma_bram_address[15];
    // Failed RAM read responses do not write a word. Allow strictly increasing
    // addresses with holes, but publish only all 128 distinct words plus success.
    // The extra eight-bit next-word counter distinguishes holes from duplicates.
    wire dma_access=rst_n && dma_bram_enable && cache_address && dma_state==ACTIVE &&
        dma_bram_address[14:11]==dma_slot && dma_bram_address[3:0]==0 &&
        dma_bram_write!=dma_store && transferred_words<128 &&
        {1'b0,dma_bram_address[10:4]}>=next_word;
    wire bad_access=rst_n && dma_bram_enable && cache_address && !dma_access;
    wire done=rst_n && dma_state==ACTIVE && dma_done_valid && dma_done_descriptor==descriptor;
    wire complete_words=transferred_words==128 || (dma_access && transferred_words==127);
    wire [3:0] effective_status=dma_done_status | {3'b0,access_error || bad_access || (dma_done_status==0 && !complete_words)};
    // Register the checked DMA result before publishing metadata. This adds
    // one clock to cache completion, not to the shared DMA window or payload.
    // Keep ownership throughout that clock; no partially checked page escapes.
    wire publish=rst_n && dma_state==PUBLISH;
    wire success=publish && completion_status==0;

    // Register the metadata-write mask before driving sixteen wide banks.
    // 16 enable FF + full 16/32/32-bit payload (80 FF), no narrower fields.
    // Data captures on every idle edge, including accepted prefetch; idle
    // values are invalid. Hold during DMA so the next edge writes the banks.
    // Page invalidation and DMA admission still happen at the original edge.
    reg [15:0] metadata_pending;
    reg [15:0] metadata_voice;
    reg [31:0] metadata_tag,metadata_ram;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) metadata_pending<=0;
        else metadata_pending<=claim_mask;
    always @(posedge clk) if(dma_state==IDLE) begin
        metadata_voice<=prefetch_voice_id;metadata_tag<=prefetch_tag;
        metadata_ram<=prefetch_ram_address;
    end

    // The two true memory ports use identical widths. DMA and DSP may access
    // different pages concurrently; ownership forbids a shared-page collision.
    // Cross-port same-page access is forbidden by the control invariant.
    // ECP5 RAM need not implement old-data semantics for that unreachable case.
    (* no_rw_check *) reg [127:0] pages [0:2047];
    reg [127:0] dsp_read_data, dsp_write_ack_data;
    reg dsp_was_write;
    assign dsp_response_data=dsp_was_write ? dsp_write_ack_data : dsp_read_data;
    always @(posedge clk) begin
        if (dma_access && dma_bram_write) pages[dma_bram_address[14:4]]<=dma_bram_write_data;
        if (dma_access && !dma_bram_write) dma_bram_read_data<=pages[dma_bram_address[14:4]];
    end
    always @(posedge clk) begin
        if (memory_write) pages[{lease_slot,memory_word}]<=dsp_write_data;
        if (dsp_accept && !memory_write) dsp_read_data<=pages[{lease_slot,memory_word}];
        if (dsp_accept) begin dsp_was_write<=memory_write;dsp_write_ack_data<=dsp_write_data;end
    end
    genvar slot;
    generate for(slot=0;slot<16;slot=slot+1) begin: metadata
        reg valid, dirty;
        // Factor each page's metadata enable directly. The public ready mux
        // need not feed the sixteen wide metadata register banks.
        wire bank_prefetch=prefetch_valid && prefetch_eligible && prefetch_slot==slot &&
            !dirty && !(lease_valid && lease_slot==slot) && !(acquire && acquire_slot==slot) &&
            !(query_pending && lease_slot==slot);
        assign claim_mask[slot]=bank_prefetch;
        assign valid_mask[slot]=valid;
        assign dirty_mask[slot]=dirty;
        assign acquire_matches[slot]=valid && !dirty && acquire_slot==slot &&
            !(dma_busy && dma_slot==slot) && voice_id[slot]==acquire_voice_id && context_tag[slot]==acquire_tag;
        always @(posedge clk or negedge rst_n) begin
            if(!rst_n) begin valid<=0;dirty<=0;end
            else begin
                if(bank_prefetch) begin valid<=0;dirty<=0;end
                if(memory_write && lease_slot==slot) dirty<=1;
                if(publish && dma_slot==slot) begin
                    if(dma_store) begin if(success) dirty<=0;end
                    else valid<=success;
                end
            end
        end
        always @(posedge clk) if(metadata_pending[slot]) begin
            voice_id[slot]<=metadata_voice;
            context_tag[slot]<=metadata_tag;
            ram_address[slot]<=metadata_ram;
        end
    end endgenerate
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            dma_state<=IDLE;dma_slot<=0;dma_store<=0;descriptor<=0;
            transferred_words<=0;next_word<=0;access_error<=0;
            lease_valid<=0;lease_readonly<=0;lease_slot<=0;lease_dirty<=0;query_pending<=0;acquire_result_valid<=0;acquire_hit<=0;
            response_pending<=0;prefetch_rejected<=0;retry_rejected<=0;memory_fault<=0;
            completion_valid<=0;completion_write<=0;completion_slot<=0;
            completion_status<=0;completion_voice_id<=0;completion_tag<=0;
        end else begin
            query_pending<=acquire;
            acquire_result_valid<=query_pending;
            if(acquire) begin
                query_matches<=acquire_matches;
                // This slot has no DSP lease until the captured result grants
                // one. Pending reservation prevents replacement in between.
                lease_slot<=acquire_slot;lease_dirty<=0;lease_readonly<=acquire_readonly;
            end
            if(query_pending) begin
                acquire_hit<=query_hit;lease_valid<=query_hit;
            end
            if(release_lease) begin lease_valid<=0;lease_dirty<=0;end
            if(memory_write) lease_dirty<=1;
            if(response_ready) response_pending<=0;
            if(dsp_accept) response_pending<=1;
            prefetch_rejected<=prefetch_valid && prefetch_ready && bad_prefetch;
            retry_rejected<=retry && !retry_eligible;
            completion_valid<=publish;
            if(bad_access) begin memory_fault<=1;if(dma_busy) access_error<=1;end
            if(dma_access) begin
                transferred_words<=transferred_words+1'b1;
                next_word<={1'b0,dma_bram_address[10:4]}+1'b1;
            end
            // IDLE descriptor fields are not valid. Sample them independently
            // of the late admission grant to shorten the wide register enable.
            // Hold all fields once SUBMIT/ACTIVE/PUBLISH owns a transaction.
            // No extra registers/clocks; idle field toggling is the tradeoff.
            if(dma_state==IDLE) begin
                dma_slot<=begin_slot;dma_store<=begin_store;descriptor<=incoming;
            end
            // Transfer counters have no meaning in IDLE. Reset them every
            // idle edge, including the admission edge, without a late grant
            // in their enable path. No extra state or transaction clock.
            if(dma_state==IDLE) begin
                transferred_words<=0;next_word<=0;access_error<=0;
            end
            if(begin_transfer) dma_state<=SUBMIT;
            else if(dma_submit_valid && dma_submit_ready) dma_state<=ACTIVE;
            if(done) begin
                dma_state<=PUBLISH;completion_write<=dma_store;completion_slot<=dma_slot;
                completion_status<=effective_status;completion_voice_id<=voice_id[dma_slot];
                completion_tag<=context_tag[dma_slot];
                if(dma_done_status==0 && effective_status!=0) memory_fault<=1;
            end
            if(publish) dma_state<=IDLE;
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk) if(rst_n) begin
        if(lease_valid && (!valid_mask[lease_slot] || (dma_busy && dma_slot==lease_slot)))
            $fatal(1,"Voice cache lease overlaps unavailable page");
        if(lease_valid && lease_dirty!=dirty_mask[lease_slot]) $fatal(1,"Lease dirty metadata diverged");
        if(lease_readonly && (lease_dirty || memory_write)) $fatal(1,"Read-only lease wrote state");
        if(state_accept && read_accept) $fatal(1,"Two DSP clients accepted together");
        if(dma_access && dsp_accept && dma_slot==lease_slot)
            $fatal(1,"Voice cache ports collide");
        if(dma_state==ACTIVE && transferred_words>128) $fatal(1,"Voice page transfer overflow");
        if(begin_store && !dirty_mask[store_slot]) $fatal(1,"Clean page store");
    end
`endif
endmodule
