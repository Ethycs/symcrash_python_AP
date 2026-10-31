/* Two-buffer DMA-to-CCS stream. Coefficients retain the normative leaf-major
 * int16 layout. A source lease ends on the actual final decoder input beat.
 * K8 rereads each 128-bit word for slots 4..7 instead of caching another
 * half-frame. Misses replay the last complete output block, initially zero.
 * Shared DMA queue/adapter must reset with this module.
 */
module ccs_stream #(
    parameter K=4,
    parameter HEADROOM_BITS=2,
    parameter [15:0] BASE=16'h8000
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
    input wire dma_bram_enable, dma_bram_write,
    input wire [15:0] dma_bram_address,
    input wire [127:0] dma_bram_write_data,
    input wire block_valid,
    output wire block_ready,
    input wire [31:0] block_tag,
    output wire audio_valid,
    input wire audio_ready,
    output wire [63:0] audio_data,
    output wire audio_last,
    output reg [31:0] audio_tag,
    output reg audio_miss,
    output reg block_done,
    input wire cancel_valid,
    input wire [31:0] cancel_tag,
    output wire [1:0] free_mask, ready_mask, reading_mask,
    output wire fault_valid,
    output wire [31:0] fault_tag,
    output wire [3:0] fault_status,
    output wire protocol_error,
    output reg memory_fault
);
    localparam LANES=(K>4)?4:K, GROUPS=K/LANES;
    localparam WORDS=64*K/8, ADDRESS_BITS=$clog2(WORDS);
    localparam CHUNKS=(K==8)?1:8/K;
    localparam IDLE=3'd0, ACQUIRE=3'd1, LOAD=3'd2, DRAIN=3'd3, REPLAY=3'd4;
    reg [2:0] state;
    wire acquire_valid, acquire_ready, acquire_result_valid, acquire_hit, acquire_bank;
    wire [15:0] acquire_address;
    wire release_valid;
    reg lease_bank;
    assign block_ready=rst_n && state==IDLE && acquire_ready;
    assign acquire_valid=block_valid && block_ready;

    coeff_buffer_owner #(.K(K), .BASE(BASE)) owner (
        .clk(clk), .rst_n(rst_n), .prefetch_valid(prefetch_valid), .prefetch_ready(prefetch_ready),
        .prefetch_tag(prefetch_tag), .prefetch_ram_address(prefetch_ram_address),
        .prefetch_class(prefetch_class), .prefetch_bank(prefetch_bank),
        .dma_submit_valid(dma_submit_valid), .dma_submit_ready(dma_submit_ready),
        .dma_submit_descriptor(dma_submit_descriptor), .dma_done_valid(dma_done_valid),
        .dma_done_descriptor(dma_done_descriptor), .dma_done_status(dma_done_status),
        .acquire_valid(acquire_valid), .acquire_ready(acquire_ready), .acquire_tag(block_tag),
        .acquire_result_valid(acquire_result_valid), .acquire_hit(acquire_hit),
        .acquire_bank(acquire_bank), .acquire_address(acquire_address),
        .release_valid(release_valid), .release_bank(lease_bank),
        .cancel_valid(cancel_valid), .cancel_tag(cancel_tag),
        .free_mask(free_mask), .ready_mask(ready_mask), .reading_mask(reading_mask),
        .fault_valid(fault_valid), .fault_tag(fault_tag), .fault_status(fault_status),
        .protocol_error(protocol_error)
    );

    reg [127:0] payload [0:2*WORDS-1];
    reg [127:0] payload_read;
    wire [15:0] dma_offset=dma_bram_address-BASE;
    wire in_partition=dma_bram_address>=BASE && {1'b0,dma_bram_address}<BASE+2*WORDS*16;
    wire dma_bank=dma_offset[ADDRESS_BITS+4];
    wire writable=!(free_mask[dma_bank] || ready_mask[dma_bank] || reading_mask[dma_bank]);
    wire write_payload=rst_n && dma_bram_enable && dma_bram_write && in_partition &&
                       dma_bram_address[3:0]==0 && writable;
    reg [6:0] read_index;
    wire read_payload;
    always @(posedge clk) begin
        if (write_payload) payload[dma_offset[ADDRESS_BITS+4:4]] <= dma_bram_write_data;
        // read_pending alone qualifies the following FIFO write. Reading on
        // idle cycles removes the output-hold mux from the BRAM-to-FIFO path;
        // addresses, data width, read latency and reservation counts stay fixed.
        payload_read <= payload[{lease_bank,read_index[ADDRESS_BITS-1:0]}];
    end

    // Two queued memory words plus a reserved synchronous read. Reservation
    // plus occupancy never exceeds two; consumption can fund a same-edge read.
    reg [127:0] words [0:1];
    reg read_pointer, write_pointer, read_pending;
    reg [1:0] word_count, chunk;
    reg [5:0] leaf;
    reg group_index;
    wire [127:0] current_word=words[read_pointer];
    wire [16*LANES-1:0] coeff_data;
    generate
        if (K==2) begin
            assign coeff_data=current_word[32*chunk +: 32];
        end else if (K==4) begin
            assign coeff_data=chunk[0] ? current_word[127:64] : current_word[63:0];
        end else begin
            assign coeff_data=group_index ? current_word[127:64] : current_word[63:0];
        end
    endgenerate
    wire coeff_valid=rst_n && state==LOAD && word_count!=0;
    wire coeff_ready;
    wire accept_coeff=coeff_valid && coeff_ready;
    wire pop_word=accept_coeff && chunk==CHUNKS-1;
    assign read_payload=rst_n && state==LOAD && read_index<WORDS &&
                        (({1'b0,word_count}+{2'b0,read_pending}<3'd2) || pop_word);
    assign release_valid=accept_coeff && leaf==63 && group_index==GROUPS-1;
    always @(posedge clk) if (rst_n && read_pending) words[write_pointer] <= payload_read;

    wire decoded_valid, decoded_last;
    wire [63:0] decoded_data;
    ccs_idwpt #(.K(K), .HEADROOM_BITS(HEADROOM_BITS)) decoder (
        .clk(clk), .rst_n(rst_n), .coeff_valid(coeff_valid), .coeff_ready(coeff_ready),
        .coeff_data(coeff_data), .audio_valid(decoded_valid),
        .audio_ready(state==DRAIN && audio_ready), .audio_data(decoded_data), .audio_last(decoded_last)
    );

    reg [63:0] history [0:15];
    reg history_valid;
    reg [3:0] output_index;
    assign audio_valid=rst_n && ((state==DRAIN && decoded_valid) || state==REPLAY);
    assign audio_data=state==REPLAY ? (history_valid ? history[output_index] : 64'd0) : decoded_data;
    assign audio_last=state==REPLAY ? output_index==15 : decoded_last;
    wire accept_audio=audio_valid && audio_ready;
    always @(posedge clk)
        if (rst_n && state==DRAIN && accept_audio) history[output_index] <= decoded_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE; lease_bank<=0; audio_tag<=0; audio_miss<=0; block_done<=0;
            read_index<=0; read_pointer<=0; write_pointer<=0; read_pending<=0;
            word_count<=0; chunk<=0; leaf<=0; group_index<=0;
            history_valid<=0; output_index<=0; memory_fault<=0;
        end else begin
            block_done<=0;
            if (dma_bram_enable && in_partition && (!dma_bram_write || dma_bram_address[3:0]!=0 || !writable))
                memory_fault<=1;
            read_pending<=read_payload;
            if (read_payload) read_index<=read_index+1'b1;
            if (read_pending) write_pointer<=!write_pointer;
            if (pop_word) read_pointer<=!read_pointer;
            case ({read_pending,pop_word})
                2'b10: word_count<=word_count+1'b1;
                2'b01: word_count<=word_count-1'b1;
            endcase
            if (accept_coeff) begin
                chunk<=pop_word ? 2'd0 : chunk+1'b1;
                leaf<=leaf+1'b1;
                if (leaf==63) begin
                    if (group_index==GROUPS-1) state<=DRAIN;
                    else begin
                        group_index<=1; read_index<=0;
                        read_pointer<=0; write_pointer<=0; word_count<=0; read_pending<=0;
                    end
                end
            end
            case (state)
                IDLE: if (acquire_valid) begin
                    audio_tag<=block_tag; output_index<=0; state<=ACQUIRE;
                end
                ACQUIRE: if (acquire_result_valid) begin
                    audio_miss<=!acquire_hit;
                    if (acquire_hit) begin
                        lease_bank<=acquire_bank; state<=LOAD;
                        read_index<=0; read_pointer<=0; write_pointer<=0; read_pending<=0;
                        word_count<=0; chunk<=0; leaf<=0; group_index<=0;
                    end else state<=REPLAY;
                end
                DRAIN, REPLAY: if (accept_audio) begin
                    output_index<=output_index+1'b1;
                    if (audio_last) begin
                        if (state==DRAIN) history_valid<=1;
                        state<=IDLE; block_done<=1;
                    end
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) if (rst_n) begin
        if ({1'b0,word_count}+{2'b0,read_pending}>3'd2) $fatal(1,"Coefficient read reservation overflow");
        if (read_payload && !reading_mask[lease_bank]) $fatal(1,"Payload read without source lease");
        if (release_valid && (word_count!=1 || read_pending || read_index!=WORDS))
            $fatal(1,"Source released before final coefficient consumption");
        if (accept_audio && audio_last!=(output_index==15)) $fatal(1,"Malformed decoded audio block");
    end
`endif
endmodule
