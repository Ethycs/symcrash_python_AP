/* Shared N64 decoder for an acquired voice-state page. The caller supplies
 * the page's stored K and the aligned coefficient-word offset; this module
 * does not define a complete hardware state serialization. Every K-slot
 * int16 coefficient is read leaf-major. Held rendering never writes state.
 * Missing/invalid contexts produce a tagged silent block, never another
 * voice's history. Queue/adapter/cache/decoder reset must be coordinated.
 */
module voice_cache_renderer #(
    parameter HEADROOM_BITS=2
) (
    input wire clk, rst_n,
    input wire block_valid,
    output wire block_ready,
    input wire [3:0] block_slot,
    input wire [15:0] block_voice_id,
    input wire [31:0] block_context_tag, block_tag,
    input wire [1:0] block_k,
    input wire [6:0] block_coeff_word,
    output wire cache_acquire_valid,
    input wire cache_acquire_ready,
    output wire [3:0] cache_acquire_slot,
    output wire [15:0] cache_acquire_voice_id,
    output wire [31:0] cache_acquire_tag,
    input wire cache_acquire_result_valid, cache_acquire_hit,
    output wire cache_dsp_valid,
    input wire cache_dsp_ready,
    output wire [6:0] cache_dsp_word,
    input wire cache_dsp_response_valid,
    output wire cache_dsp_response_ready,
    input wire [127:0] cache_dsp_response_data,
    output wire cache_release_valid,
    input wire cache_release_ready,
    output wire audio_valid,
    input wire audio_ready,
    output wire [63:0] audio_data,
    output wire audio_last,
    output reg [15:0] audio_voice_id,
    output reg [31:0] audio_tag,
    // 0: rendered retained coefficients, 1: cache miss, 2: invalid command.
    output reg [1:0] audio_status,
    output reg block_done
);
    localparam IDLE=3'd0, ACQUIRE=3'd1, COMMAND=3'd2, LOAD=3'd3,
               DRAIN=3'd4, SILENCE=3'd5;
    reg [2:0] state;
    wire [7:0] requested_words=block_k==0 ? 8'd16 : block_k==1 ? 8'd32 : 8'd64;
    // Preserve the endpoint carry: offset+length==128 is legal, >128 is not.
    wire [7:0] requested_end={1'b0,block_coeff_word}+requested_words;
    wire invalid_command=block_k==3 || block_voice_id==16'hffff || requested_end>128;
    assign block_ready=rst_n && state==IDLE && cache_acquire_ready;
    wire start=block_valid && block_ready;
    // Present the acquisition request independently of ready. Acceptance
    // remains exactly start && !invalid_command, since block_ready includes
    // cache_acquire_ready. Avoid serial ready->valid->arbitration gating.
    assign cache_acquire_valid=rst_n && state==IDLE && block_valid && !invalid_command;
    assign cache_acquire_slot=block_slot;
    assign cache_acquire_voice_id=block_voice_id;
    assign cache_acquire_tag=block_context_tag;
    reg [1:0] active_k;
    reg [6:0] coeff_base,read_index;
    wire [6:0] frame_words=active_k==0 ? 7'd16 : active_k==1 ? 7'd32 : 7'd64;
    wire [1:0] final_group=active_k==0 ? 2'd0 : active_k==1 ? 2'd1 : 2'd3;
    wire [1:0] final_chunk=active_k==0 ? 2'd3 : active_k==1 ? 2'd1 : 2'd0;
    (* ram_style = "logic" *) reg [127:0] words [0:1];
    reg read_pointer,write_pointer,read_pending;
    reg [1:0] word_count,chunk,group_index;
    reg [5:0] leaf;
    reg release_pending;
    wire [1:0] word_lane=active_k==0 ? chunk : active_k==1 ? {chunk[0],group_index[0]} : group_index;
    wire [127:0] current_word=words[read_pointer];
    wire [31:0] coeff_data=current_word[32*word_lane +:32];
    wire coeff_valid=rst_n && state==LOAD && word_count!=0;
    wire coeff_ready;
    wire accept_coeff=coeff_valid && coeff_ready;
    wire pop_word=accept_coeff && chunk==final_chunk;
    wire reading=state==COMMAND || state==LOAD;
    // Do not fund a new RAM request from same-edge coefficient consumption.
    // This breaks the decoder-ready -> RAM-enable/release control path.
    // The FIFO remains two words; K8 read bubbles are measured explicitly.
    // The frame lengths are exactly powers of two (16/32/64 words).
    // Test the high index bits directly, preserving all seven index bits.
    // This is the same comparison without a variable carry-chain bound.
    wire more_words=active_k==0 ? !(|read_index[6:4]) :
        active_k==1 ? !(|read_index[6:5]) : !read_index[6];
    wire free_word_credit=!word_count[1] && !(word_count[0] && read_pending);
    assign cache_dsp_valid=rst_n && reading && more_words && free_word_credit;
    assign cache_dsp_word=coeff_base+read_index;
    // Every issued read reserves a FIFO entry. A returning response has
    // read_pending=1, hence word_count<=1 under the two-entry invariant.
    // Its ready signal need not traverse coefficient consumption logic.
    assign cache_dsp_response_ready=rst_n && reading;
    wire request=cache_dsp_valid && cache_dsp_ready;
    wire response=cache_dsp_response_valid && cache_dsp_response_ready;
    assign cache_release_valid=rst_n && release_pending;
    always @(posedge clk) if(response) words[write_pointer]<=cache_dsp_response_data;

    wire decoder_frame_ready,decoder_frame_error,decoded_valid,decoded_last;
    wire [63:0] decoded_data;
    ccs_idwpt_runtime #(.HEADROOM_BITS(HEADROOM_BITS)) decoder (
        .clk(clk),.rst_n(rst_n),.frame_valid(state==COMMAND),.frame_ready(decoder_frame_ready),
        .frame_k(active_k),.frame_error(decoder_frame_error),.coeff_valid(coeff_valid),
        .coeff_ready(coeff_ready),.coeff_data(coeff_data),.audio_valid(decoded_valid),
        .audio_ready(state==DRAIN && !release_pending && audio_ready),.audio_data(decoded_data),.audio_last(decoded_last)
    );
    reg [3:0] output_index;
    assign audio_valid=rst_n && ((state==DRAIN && !release_pending && decoded_valid) || state==SILENCE);
    assign audio_data=state==SILENCE ? 64'd0 : decoded_data;
    assign audio_last=state==SILENCE ? output_index==15 : decoded_last;
    wire accept_audio=audio_valid && audio_ready;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state<=IDLE;active_k<=0;coeff_base<=0;read_index<=0;
            read_pointer<=0;write_pointer<=0;read_pending<=0;word_count<=0;
            chunk<=0;group_index<=0;leaf<=0;release_pending<=0;
            output_index<=0;audio_voice_id<=0;audio_tag<=0;audio_status<=0;block_done<=0;
        end else begin
            block_done<=0;
            if(request) read_index<=read_index+1'b1;
            case({request,response})
                2'b10:read_pending<=1;
                2'b01:read_pending<=0;
            endcase
            if(response) write_pointer<=!write_pointer;
            if(pop_word) read_pointer<=!read_pointer;
            case({response,pop_word})
                2'b10:word_count<=word_count+1'b1;
                2'b01:word_count<=word_count-1'b1;
            endcase
            if(cache_release_valid && cache_release_ready) release_pending<=0;
            if(accept_coeff) begin
                chunk<=pop_word ? 2'd0 : chunk+1'b1;
                leaf<=leaf+1'b1;
                if(leaf==63) begin
                    if(group_index==final_group) begin state<=DRAIN;release_pending<=1;end
                    else begin
                        group_index<=group_index+1'b1;read_index<=0;
                        read_pointer<=0;write_pointer<=0;word_count<=0;read_pending<=0;
                    end
                end
            end
            case(state)
                IDLE:if(start) begin
                    audio_voice_id<=block_voice_id;audio_tag<=block_tag;output_index<=0;
                    active_k<=block_k;coeff_base<=block_coeff_word;
                    read_index<=0;read_pointer<=0;write_pointer<=0;read_pending<=0;
                    word_count<=0;chunk<=0;leaf<=0;group_index<=0;release_pending<=0;
                    audio_status<=invalid_command ? 2'd2 : 2'd0;
                    state<=invalid_command ? SILENCE : ACQUIRE;
                end
                ACQUIRE:if(cache_acquire_result_valid) begin
                    audio_status<=cache_acquire_hit ? 2'd0 : 2'd1;
                    state<=cache_acquire_hit ? COMMAND : SILENCE;
                end
                COMMAND:if(decoder_frame_ready) state<=LOAD;
                DRAIN,SILENCE:if(accept_audio) begin
                    output_index<=output_index+1'b1;
                    if(audio_last) begin state<=IDLE;block_done<=1;end
                end
            endcase
        end
    end
`ifndef SYNTHESIS
    always @(posedge clk) if(rst_n) begin
        if(decoder_frame_error) $fatal(1,"Invalid cache decoder mode");
        if({1'b0,word_count}+{2'b0,read_pending}>3'd2) $fatal(1,"Cache gather reservation overflow");
        if(response && !read_pending) $fatal(1,"Unrequested cache response");
        if(request && read_pending && !response) $fatal(1,"Multiple outstanding cache reads");
        if(accept_coeff && leaf==63 && (word_count!=1 || read_pending || read_index!=frame_words))
            $fatal(1,"Cache group ended before all source words were consumed");
        if(accept_audio && audio_last!=(output_index==15)) $fatal(1,"Malformed cache audio block");
    end
`endif
endmodule
