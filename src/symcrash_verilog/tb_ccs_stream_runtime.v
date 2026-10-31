`timescale 1ns/1ps
module tb_ccs_stream_runtime;
    parameter HEADROOM_BITS=2, STALLS=1, CASES=108;
    localparam LANES=2;
    reg [1:0] modes[0:CASES-1];
    function integer slots; input integer row; begin slots=2<<modes[row];end endfunction
    function integer groups; input integer row; begin groups=1<<modes[row];end endfunction
    function integer frame_beats; input integer row; begin frame_beats=8*slots(row);end endfunction
    wire [1:0] dma_done_matches=2'b00;
    wire [127:0] dma_match_descriptors;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0, prefetch_valid=0, block_valid=0, cancel_valid=0;
    reg [31:0] prefetch_tag=0, prefetch_ram_address=0, block_tag=0, cancel_tag=0;
    reg [1:0] prefetch_class=2,prefetch_k=0;
    wire prefetch_rejected;
    wire [31:0] prefetch_rejected_tag;
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
    ccs_stream_runtime #(.HEADROOM_BITS(HEADROOM_BITS)) stream (.*);

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
    reg hold_response=0, hold_audio=0, hold_completion=0,hold_submit=0;
    wire queue_submit_ready;
    assign dma_submit_ready=queue_submit_ready && !hold_submit;
    reg [31:0] fail_address=32'hffffffff;
    assign dma_done_valid=completion_valid && !hold_completion;
    assign dma_done_descriptor=completion_descriptor;
    assign dma_done_status=completion_status;
    dma_priority queue (
        .clk(clk),.rst_n(rst_n),.submit_valid(dma_submit_valid && !hold_submit),.submit_ready(queue_submit_ready),
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

    reg [15:0] coefficients [0:CASES*512-1], expected [0:CASES*64-1];
    reg [4095:0] coeff_file, expected_file,mode_file;
    reg [31:0] responses [0:4095];
    integer wr=0,rd=0,cycle=0,word_lane,coefficient_index;
    integer expected_input_case=0,input_count=0,decoded_inputs=0,payload_reads=0,releases=0;
    integer admitted=0,completed=0,faults=0,successes=0,misses=0,output_samples=0,resets=0,overlaps=0;
    integer min_cycles=2147483647,max_cycles=0,min_miss=2147483647,max_miss=0;
    integer final_input_cycle=-1,last_release_cycle=-1,frame,probe,probe_row,wait_count,observed_outputs;
    integer min_mode[0:2],max_mode[0:2],mode_frames[0:2];
    integer rejected=0,descriptor_checks=0,request_row,request_offset;
    reg [63:0] expected_descriptor[0:1],stalled_descriptor;
    reg submit_was_stalled=0;
    reg submitted_bank;
    reg [31:0] tag_value;

    function [31:0] source_address;
        input integer row;
        begin source_address=32'h200000+row*1024; end
    endfunction
    always @(negedge clk) begin
        ram_request_ready=rst_n && (!STALLS || cycle%7<5);
        audio_ready=rst_n && !hold_audio && (!STALLS || cycle%11<7);
        if (!rst_n) ram_response_valid=0;
        else if (!ram_response_valid && !hold_response && rd<wr && (!STALLS || cycle%5!=1)) begin
            ram_response_valid=1;
            for (word_lane=0;word_lane<8;word_lane=word_lane+1) begin
                coefficient_index=(responses[rd%4096]-32'h200000)/2+word_lane;
                if (coefficient_index<0 || coefficient_index>=CASES*512) $fatal(1,"DMA source escaped coefficient image");
                ram_response_data[16*word_lane +: 16]=coefficients[coefficient_index];
            end
            ram_response_error=responses[rd%4096]==fail_address;
        end
    end
    integer lane,leaf,group_id;
    always @(posedge clk) begin
        cycle<=cycle+1;
        if (!rst_n) begin wr=0;rd=0;input_count=0;submit_was_stalled=0; end
        else begin
            if (prefetch_valid && prefetch_ready && prefetch_k<3) begin
                admitted=admitted+1;
                expected_descriptor[prefetch_bank]={prefetch_ram_address,
                    (prefetch_bank ? 16'h8400 : 16'h8000),
                    (8'd16<<prefetch_k), (4'd15-{2'b0,prefetch_class}), 4'b0011};
            end
            if (submit_was_stalled && (!dma_submit_valid || dma_submit_descriptor!==stalled_descriptor))
                $fatal(1,"Runtime descriptor changed while stalled");
            submit_was_stalled=dma_submit_valid && !dma_submit_ready;stalled_descriptor=dma_submit_descriptor;
            if (dma_submit_valid && dma_submit_ready) begin
                if (dma_submit_descriptor!==expected_descriptor[dma_submit_descriptor[26]])
                    $fatal(1,"Wrong runtime length/source/destination/priority binding");
                descriptor_checks=descriptor_checks+1;
            end
            if (stream.acquire_result_valid && stream.acquire_hit &&
                stream.acquire_k!==modes[expected_input_case]) $fatal(1,"Acquisition returned a different K");
            if (ram_request_valid && ram_request_ready) begin
                if (ram_request_write || wr-rd>=4096) $fatal(1,"Unexpected DMA direction/overflow");
                request_row=(ram_request_address-32'h200000)/1024;
                request_offset=(ram_request_address-32'h200000)%1024;
                if(request_row<0 || request_row>=CASES || request_offset>=128*slots(request_row))
                    $fatal(1,"DMA fetched padded bytes outside the exact K payload");
                responses[wr%4096]=ram_request_address;wr=wr+1;
            end
            if (ram_response_valid && ram_response_ready) begin rd=rd+1;ram_response_valid<=0; end
            if (dma_done_valid) completed=completed+1;
            if (fault_valid) faults=faults+1;
            if (block_valid && block_ready) begin input_count=0;final_input_cycle=-1;last_release_cycle=-1; end
            if (stream.coeff_valid && stream.coeff_ready) begin
                if (input_count>=64*groups(expected_input_case)) $fatal(1,"Extra decoder input");
                leaf=input_count%64;group_id=input_count/64;
                for (lane=0;lane<LANES;lane=lane+1)
                    if (stream.coeff_data[16*lane +:16] !== coefficients[expected_input_case*512+leaf*slots(expected_input_case)+group_id*LANES+lane])
                        $fatal(1,"Leaf/slot order mismatch row=%0d beat=%0d lane=%0d",expected_input_case,input_count,lane);
                input_count=input_count+1;decoded_inputs=decoded_inputs+1;
                if (input_count==64*groups(expected_input_case)) final_input_cycle=cycle;
            end
            if (stream.release_valid) begin
                if (input_count!=64*groups(expected_input_case) || final_input_cycle!=cycle) $fatal(1,"Early or late source release");
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
            hold_response=0;hold_completion=0;hold_audio=0;hold_submit=0;fail_address=32'hffffffff;
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
            @(negedge clk);prefetch_valid=1;prefetch_tag=wanted_tag;prefetch_ram_address=source_address(row);prefetch_k=modes[row];
            @(posedge clk);wait_count=0;
            while (!prefetch_ready) begin
                @(posedge clk);wait_count=wait_count+1;
                if (wait_count>1000) $fatal(1,"Prefetch stuck");
            end
            submitted_bank=prefetch_bank;
            @(negedge clk);prefetch_valid=0;prefetch_k=3;
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
                if (!sent_next) begin prefetch_tag=next_tag;prefetch_ram_address=source_address(next_row);prefetch_k=modes[next_row];end
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
                            $fatal(1,"Audio mismatch K=%0d row=%0d sample=%0d got=%0d expected=%0d",slots(expected_input_case),expected_row,beats*4+n,$signed(audio_data[16*n +:16]),$signed(wanted_sample));
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
                if (decoded_inputs-inputs_before!=64*groups(expected_input_case) || payload_reads-reads_before!=frame_beats(expected_input_case)*groups(expected_input_case) || last_release_cycle!=final_input_cycle)
                    $fatal(1,"Wrong transfer/read/release accounting");
                if (!STALLS && elapsed!=(20+163*groups(expected_input_case))) $fatal(1,"Wrong ideal stream latency: %0d expected=%0d",elapsed,20+163*groups(expected_input_case));
                if (elapsed<min_mode[modes[expected_input_case]]) min_mode[modes[expected_input_case]]=elapsed;
                if (elapsed>max_mode[modes[expected_input_case]]) max_mode[modes[expected_input_case]]=elapsed;
                mode_frames[modes[expected_input_case]]=mode_frames[modes[expected_input_case]]+1;
                if (elapsed<min_cycles) min_cycles=elapsed;
                if (elapsed>max_cycles) max_cycles=elapsed;
            end
        end
    endtask

    task reject_reserved;
        input [31:0] wanted_tag;
        reg [1:0] old_free,old_ready,old_reading;
        integer old_admitted;
        begin
            @(negedge clk);prefetch_valid=1;prefetch_tag=wanted_tag;prefetch_k=3;
            old_free=free_mask;old_ready=ready_mask;old_reading=reading_mask;old_admitted=admitted;
            @(posedge clk);
            if(!prefetch_ready) $fatal(1,"Reserved K was not acknowledged");
            #1;
            if(!prefetch_rejected || prefetch_rejected_tag!==wanted_tag || admitted!=old_admitted ||
               free_mask!==old_free || ready_mask!==old_ready || reading_mask!==old_reading)
                $fatal(1,"Reserved K claimed/changed ownership");
            rejected=rejected+1;
            @(negedge clk);prefetch_valid=0;prefetch_k=0;
            tick;if(prefetch_rejected) $fatal(1,"Reserved rejection failed to pulse");
        end
    endtask

    initial begin
        if (!$value$plusargs("coeffs=%s",coeff_file) || !$value$plusargs("expected=%s",expected_file) || !$value$plusargs("modes=%s",mode_file)) $fatal(1,"Missing vectors");
        $readmemh(coeff_file,coefficients);$readmemh(expected_file,expected);$readmemh(mode_file,modes);
        for(frame=0;frame<3;frame=frame+1) begin min_mode[frame]=2147483647;max_mode[frame]=0;mode_frames[frame]=0;end
        reset_all;
        render(99,-1,1,-1,0); // Cold miss is silence.
        reject_reserved(32'hbad00000);
        hold_submit=1;prefetch(100,3);
        reject_reserved(32'hbad00001);
        hold_submit=0;wait_ready(1);
        render(100,3,0,-1,0);
        hold_response=1;prefetch(101,4);
        render(101,3,1,-1,0); // In-flight miss replays retained coefficients.
        if (free_mask==3) $fatal(1,"Reused expired in-flight buffer");
        hold_response=0;wait_free;
        fail_address=source_address(5)+16*(frame_beats(5)/2);
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
        prefetch(tag_value,0);prefetch(tag_value+1,1);wait_ready(3);reject_reserved(32'hbad00002);
        for (frame=0;frame<CASES;frame=frame+1) begin
            render(tag_value,frame,0,(frame+2<CASES)?frame+2:-1,tag_value+2);
            tag_value=tag_value+1;
        end
        render(tag_value+100,CASES-1,1,-1,0);
        if (overlaps==0) $fatal(1,"Never refilled a released bank during decoding");

        // Reset new integration state, then prove cold fallback and a full
        // successful reconstruction. Decoder primitive resets have their own suite.
        for (probe=0;probe<24;probe=probe+1) begin
            probe_row=(probe/8==0) ? 0 : (probe/8==1) ? 2 : 4;
            if(modes[probe_row]!==probe/8) $fatal(1,"Wrong reset probe mode vector");
            prefetch(200+probe,probe_row);wait_ready(1);expected_input_case=probe_row;
            @(negedge clk);block_valid=1;block_tag=(probe%8==6)?999:200+probe;
            @(posedge clk);
            @(negedge clk);block_valid=0;
            case (probe%8)
                0: begin end
                1: while (!stream.read_pending) tick;
                2: while (input_count<17) tick;
                3: if (groups(probe_row)>1) begin while (stream.group_index!=1 || input_count<70) tick; end
                   else begin while (stream.state!=stream.DRAIN) tick; end
                4: begin hold_audio=1;while (!audio_valid) tick;end
                5: begin
                    observed_outputs=0;
                    while (observed_outputs<7) begin @(posedge clk);if (audio_valid && audio_ready) observed_outputs=observed_outputs+1;#1;end
                end
                6: while (stream.state!=stream.REPLAY) tick;
                7: begin while (input_count<64) tick;if (groups(probe_row)>1 && !reading_mask) $fatal(1,"K8 source released after first group");end
            endcase
            reset_all;
            render(999,-1,1,-1,0);
            prefetch(300+probe,probe_row);wait_ready(1);render(300+probe,probe_row,0,-1,0);
        end
        $display("PASS runtime_stream headroom=%0d stalls=%0d cases=%0d success=%0d misses=%0d samples=%0d inputs=%0d reads=%0d releases=%0d overlaps=%0d admitted=%0d completions=%0d faults=%0d resets=%0d rejected=%0d descriptors=%0d K2=%0d:%0d..%0d K4=%0d:%0d..%0d K8=%0d:%0d..%0d miss_cycles=%0d..%0d",
            HEADROOM_BITS,STALLS,CASES,successes,misses,output_samples,decoded_inputs,payload_reads,releases,overlaps,admitted,completed,faults,resets,rejected,descriptor_checks,
            mode_frames[0],min_mode[0],max_mode[0],mode_frames[1],min_mode[1],max_mode[1],mode_frames[2],min_mode[2],max_mode[2],min_miss,max_miss);
        $finish;
    end
    initial begin #100000000;$fatal(1,"Stream integration timed out");end
endmodule
