`timescale 1ns/1ps
module tb_ccs_stream;
    parameter K=4, HEADROOM_BITS=2, STALLS=1, CASES=36;
    localparam LANES=(K>4)?4:K, GROUPS=K/LANES, BYTES=64*K*2, BEATS=BYTES/16;
    localparam IDEAL_CYCLES=(K==2 ? 183 : 311*GROUPS+20);
    wire [1:0] dma_done_matches=2'b00;
    wire [127:0] dma_match_descriptors;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0, prefetch_valid=0, block_valid=0, cancel_valid=0;
    reg [31:0] prefetch_tag=0, prefetch_ram_address=0, block_tag=0, cancel_tag=0;
    reg [1:0] prefetch_class=2;
    wire prefetch_ready, prefetch_bank, block_ready, audio_valid, audio_last, audio_miss, block_done;
    reg audio_ready=0;
    wire [63:0] audio_data;
    wire [31:0] audio_tag;
    wire [1:0] free_mask, ready_mask, reading_mask;
    wire fault_valid, protocol_error, memory_fault;
    wire [31:0] fault_tag;
    wire [3:0] fault_status;
    wire dma_submit_valid, dma_submit_ready, dma_done_valid;
    wire [63:0] dma_submit_descriptor, dma_done_descriptor;
    wire [3:0] dma_done_status;
    wire dma_bram_enable, dma_bram_write;
    wire [15:0] dma_bram_address;
    wire [127:0] dma_bram_write_data;
    ccs_stream #(.K(K),.HEADROOM_BITS(HEADROOM_BITS)) stream (.*);

    wire initialized, completion_valid, completion_interrupt, deadline_missed;
    wire [8:0] queued_count;
    wire [63:0] completion_descriptor;
    wire [3:0] completion_status;
    wire [31:0] completion_cycles;
    wire ram_request_valid, ram_request_write, ram_response_ready;
    wire [31:0] ram_request_address;
    wire [127:0] ram_request_data;
    reg ram_request_ready=0, ram_response_valid=0, ram_response_error=0;
    reg [127:0] ram_response_data=0;
    reg hold_response=0, hold_audio=0, hold_completion=0;
    reg [31:0] fail_address=32'hffffffff;
    assign dma_done_valid=completion_valid && !hold_completion;
    assign dma_done_descriptor=completion_descriptor;
    assign dma_done_status=completion_status;
    dma_priority queue (
        .clk(clk),.rst_n(rst_n),.submit_valid(dma_submit_valid),.submit_ready(dma_submit_ready),
        .submit_descriptor(dma_submit_descriptor),.initialized(initialized),.queued_count(queued_count),
        .completion_valid(completion_valid),.completion_ready(!hold_completion),
        .completion_descriptor(completion_descriptor),.completion_status(completion_status),
        .completion_cycles(completion_cycles),.completion_interrupt(completion_interrupt),
        .deadline_missed(deadline_missed),.ram_request_valid(ram_request_valid),
        .ram_request_ready(ram_request_ready),.ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address),.ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid),.ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error),.ram_response_data(ram_response_data),
        .bram_enable(dma_bram_enable),.bram_write(dma_bram_write),.bram_address(dma_bram_address),
        .bram_write_data(dma_bram_write_data),.bram_read_data(128'd0)
    );

    reg [15:0] coefficients [0:CASES*64*K-1], expected [0:CASES*64-1];
    reg [4095:0] coeff_file, expected_file;
    reg [31:0] responses [0:4095];
    integer wr=0,rd=0,cycle=0,word_lane,coefficient_index;
    integer expected_input_case=0,input_count=0,decoded_inputs=0,payload_reads=0,releases=0;
    integer admitted=0,completed=0,faults=0,successes=0,misses=0,output_samples=0,resets=0,overlaps=0;
    integer min_cycles=2147483647,max_cycles=0,min_miss=2147483647,max_miss=0;
    integer final_input_cycle=-1,last_release_cycle=-1,frame,probe,wait_count,observed_outputs;
    reg submitted_bank;
    reg [31:0] tag_value;

    function [31:0] source_address;
        input integer row;
        begin source_address=32'h200000+row*BYTES; end
    endfunction
    always @(negedge clk) begin
        ram_request_ready=rst_n && (!STALLS || cycle%7<5);
        audio_ready=rst_n && !hold_audio && (!STALLS || cycle%11<7);
        if (!rst_n) ram_response_valid=0;
        else if (!ram_response_valid && !hold_response && rd<wr && (!STALLS || cycle%5!=1)) begin
            ram_response_valid=1;
            for (word_lane=0;word_lane<8;word_lane=word_lane+1) begin
                coefficient_index=(responses[rd%4096]-32'h200000)/2+word_lane;
                if (coefficient_index<0 || coefficient_index>=CASES*64*K) $fatal(1,"DMA source escaped coefficient image");
                ram_response_data[16*word_lane +: 16]=coefficients[coefficient_index];
            end
            ram_response_error=responses[rd%4096]==fail_address;
        end
    end
    integer lane,leaf,group_id;
    always @(posedge clk) begin
        cycle<=cycle+1;
        if (!rst_n) begin wr=0;rd=0;input_count=0; end
        else begin
            if (prefetch_valid && prefetch_ready) admitted=admitted+1;
            if (ram_request_valid && ram_request_ready) begin
                if (ram_request_write || wr-rd>=4096) $fatal(1,"Unexpected DMA direction/overflow");
                responses[wr%4096]=ram_request_address;wr=wr+1;
            end
            if (ram_response_valid && ram_response_ready) begin rd=rd+1;ram_response_valid<=0; end
            if (dma_done_valid) completed=completed+1;
            if (fault_valid) faults=faults+1;
            if (block_valid && block_ready) begin input_count=0;final_input_cycle=-1;last_release_cycle=-1; end
            if (stream.coeff_valid && stream.coeff_ready) begin
                if (input_count>=64*GROUPS) $fatal(1,"Extra decoder input");
                leaf=input_count%64;group_id=input_count/64;
                for (lane=0;lane<LANES;lane=lane+1)
                    if (stream.coeff_data[16*lane +:16] !== coefficients[expected_input_case*64*K+leaf*K+group_id*LANES+lane])
                        $fatal(1,"Leaf/slot order mismatch row=%0d beat=%0d lane=%0d",expected_input_case,input_count,lane);
                input_count=input_count+1;decoded_inputs=decoded_inputs+1;
                if (input_count==64*GROUPS) final_input_cycle=cycle;
            end
            if (stream.release_valid) begin
                if (input_count!=64*GROUPS || final_input_cycle!=cycle) $fatal(1,"Early or late source release");
                last_release_cycle=cycle;releases=releases+1;
            end
            if (stream.read_payload) payload_reads=payload_reads+1;
            if (dma_bram_enable && dma_bram_write && stream.state==stream.DRAIN) overlaps=overlaps+1;
            if (protocol_error || memory_fault) $fatal(1,"Stream ownership/memory violation");
        end
    end

    task tick; begin @(posedge clk);#1;end endtask
    task reset_all;
        begin
            @(negedge clk);rst_n=0;prefetch_valid=0;block_valid=0;cancel_valid=0;
            hold_response=0;hold_completion=0;hold_audio=0;fail_address=32'hffffffff;
            repeat (3) tick;
            if (audio_valid || dma_submit_valid || block_ready) $fatal(1,"Reset leaked valid work");
            @(negedge clk);rst_n=1;
            repeat (256) tick;
            if (!initialized || !block_ready || free_mask!=3 || stream.history_valid) $fatal(1,"Reset failed");
            resets=resets+1;
        end
    endtask
    task prefetch;
        input [31:0] wanted_tag;
        input integer row;
        begin
            @(negedge clk);prefetch_valid=1;prefetch_tag=wanted_tag;prefetch_ram_address=source_address(row);
            @(posedge clk);wait_count=0;
            while (!prefetch_ready) begin
                @(posedge clk);wait_count=wait_count+1;
                if (wait_count>1000) $fatal(1,"Prefetch stuck");
            end
            submitted_bank=prefetch_bank;
            @(negedge clk);prefetch_valid=0;
        end
    endtask
    task wait_ready;
        input [1:0] mask;
        begin
            wait_count=0;
            while ((ready_mask & mask)!=mask) begin
                tick;wait_count=wait_count+1;
                if (wait_count>2000) $fatal(1,"Frame publication stuck");
            end
        end
    endtask
    task wait_free;
        begin
            wait_count=0;
            while (free_mask!=3) begin
                tick;wait_count=wait_count+1;
                if (wait_count>2000) $fatal(1,"Discard failed to drain");
            end
        end
    endtask
    task render;
        input [31:0] wanted_tag;
        input integer expected_row;
        input expected_miss;
        input integer next_row;
        input [31:0] next_tag;
        integer start_cycle,beats,elapsed,n,inputs_before,reads_before;
        reg sent_next,was_stalled;
        reg [63:0] stalled_data;
        reg stalled_last;
        reg [15:0] wanted_sample;
        begin
            expected_input_case=(expected_row<0)?0:expected_row;
            inputs_before=decoded_inputs;reads_before=payload_reads;
            @(negedge clk);block_valid=1;block_tag=wanted_tag;
            if (!block_ready) $fatal(1,"Block launch unexpectedly blocked");
            @(posedge clk);start_cycle=cycle;
            @(negedge clk);block_valid=0;
            beats=0;sent_next=next_row<0;was_stalled=0;
            while (beats<16) begin
                prefetch_valid=!sent_next;
                if (!sent_next) begin prefetch_tag=next_tag;prefetch_ram_address=source_address(next_row);end
                @(posedge clk);
                if (prefetch_valid && prefetch_ready) sent_next=1;
                if (was_stalled && (!audio_valid || audio_data!==stalled_data || audio_last!==stalled_last))
                    $fatal(1,"Audio changed under backpressure");
                was_stalled=audio_valid && !audio_ready;stalled_data=audio_data;stalled_last=audio_last;
                if (audio_valid) begin
                    if (audio_miss!==expected_miss || audio_tag!==wanted_tag || audio_last!==(beats==15))
                        $fatal(1,"Wrong stream tag/miss/last");
                    for (n=0;n<4;n=n+1) begin
                        wanted_sample=(expected_row<0)?16'd0:expected[expected_row*64+beats*4+n];
                        if (audio_data[16*n +:16] !== wanted_sample)
                            $fatal(1,"Audio mismatch K=%0d row=%0d sample=%0d got=%0d expected=%0d",K,expected_row,beats*4+n,$signed(audio_data[16*n +:16]),$signed(wanted_sample));
                    end
                    if (audio_ready) begin beats=beats+1;output_samples=output_samples+4;end
                end
                elapsed=cycle-start_cycle+1;
                if (elapsed>1500) $fatal(1,"Stream waited for unavailable data or stalled");
                #1;
                if (block_done !== (beats==16)) $fatal(1,"Wrong block completion edge");
                @(negedge clk);
            end
            prefetch_valid=0;
            if (!sent_next) $fatal(1,"N+2 prefetch never admitted before current output completed");
            if (expected_miss) begin
                misses=misses+1;
                if (decoded_inputs!=inputs_before || payload_reads!=reads_before || reading_mask)
                    $fatal(1,"Miss touched decoder inputs or retained a lease");
                if (!STALLS && elapsed!=18) $fatal(1,"Wrong ideal fallback latency: %0d",elapsed);
                if (elapsed<min_miss) min_miss=elapsed;
                if (elapsed>max_miss) max_miss=elapsed;
            end else begin
                successes=successes+1;
                if (decoded_inputs-inputs_before!=64*GROUPS || payload_reads-reads_before!=BEATS*GROUPS || last_release_cycle!=final_input_cycle)
                    $fatal(1,"Wrong transfer/read/release accounting");
                if (!STALLS && elapsed!=IDEAL_CYCLES) $fatal(1,"Wrong ideal stream latency: %0d expected=%0d",elapsed,IDEAL_CYCLES);
                if (elapsed<min_cycles) min_cycles=elapsed;
                if (elapsed>max_cycles) max_cycles=elapsed;
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("coeffs=%s",coeff_file) || !$value$plusargs("expected=%s",expected_file)) $fatal(1,"Missing vectors");
        $readmemh(coeff_file,coefficients);$readmemh(expected_file,expected);
        reset_all;
        render(99,-1,1,-1,0); // Cold miss is silence.
        prefetch(100,3);wait_ready(1);
        render(100,3,0,-1,0);
        hold_response=1;prefetch(101,4);
        render(101,3,1,-1,0); // In-flight miss replays retained coefficients.
        if (free_mask==3) $fatal(1,"Reused expired in-flight buffer");
        hold_response=0;wait_free;
        fail_address=source_address(5)+16*(BEATS/2);
        prefetch(102,5);wait_free;repeat (2) tick;
        render(102,3,1,-1,0);fail_address=32'hffffffff;
        hold_response=1;prefetch(103,6);
        repeat (40020) tick;
        if (!deadline_missed) $fatal(1,"Missed deliberate overdue transfer");
        render(103,3,1,-1,0);
        hold_response=0;wait_free;
        prefetch(104,7);wait_ready(1);
        @(negedge clk);cancel_valid=1;cancel_tag=104;tick;
        @(negedge clk);cancel_valid=0;
        render(104,3,1,-1,0);
        if (faults!=2) $fatal(1,"Missing RAM/overdue fault reports");

        // Continuous N+2 sequence; no waiting for ready is allowed between blocks.
        tag_value=32'hffffff00;
        prefetch(tag_value,0);prefetch(tag_value+1,1);wait_ready(3);
        for (frame=0;frame<CASES;frame=frame+1) begin
            render(tag_value,frame,0,(frame+2<CASES)?frame+2:-1,tag_value+2);
            tag_value=tag_value+1;
        end
        render(tag_value+100,CASES-1,1,-1,0);
        if (overlaps==0) $fatal(1,"Never refilled a released bank during decoding");

        // Reset new integration state, then prove cold fallback and a full
        // successful reconstruction. Decoder primitive resets have their own suite.
        for (probe=0;probe<8;probe=probe+1) begin
            prefetch(200+probe,3);wait_ready(1);expected_input_case=3;
            @(negedge clk);block_valid=1;block_tag=(probe==6)?999:200+probe;
            @(posedge clk);
            @(negedge clk);block_valid=0;
            case (probe)
                0: begin end
                1: while (!stream.read_pending) tick;
                2: while (input_count<17) tick;
                3: if (K==8) begin while (stream.group_index!=1 || input_count<70) tick; end
                   else begin while (stream.state!=stream.DRAIN) tick; end
                4: begin hold_audio=1;while (!audio_valid) tick;end
                5: begin
                    observed_outputs=0;
                    while (observed_outputs<7) begin @(posedge clk);if (audio_valid && audio_ready) observed_outputs=observed_outputs+1;#1;end
                end
                6: while (stream.state!=stream.REPLAY) tick;
                7: begin while (input_count<64) tick;if (K==8 && !reading_mask) $fatal(1,"K8 source released after first group");end
            endcase
            reset_all;
            render(999,-1,1,-1,0);
            prefetch(300+probe,3);wait_ready(1);render(300+probe,3,0,-1,0);
        end
        $display("PASS stream K=%0d headroom=%0d stalls=%0d cases=%0d success=%0d misses=%0d samples=%0d inputs=%0d reads=%0d releases=%0d overlaps=%0d admitted=%0d completions=%0d faults=%0d resets=%0d cycles=%0d..%0d miss_cycles=%0d..%0d",K,HEADROOM_BITS,STALLS,CASES,successes,misses,output_samples,decoded_inputs,payload_reads,releases,overlaps,admitted,completed,faults,resets,min_cycles,max_cycles,min_miss,max_miss);
        $finish;
    end
    initial begin #100000000;$fatal(1,"Stream integration timed out");end
endmodule
