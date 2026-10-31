`timescale 1ns/1ps
module tb_ccs_dma_window;
    parameter K=4, HEADROOM_BITS=2, STALLS=1, CASES=36, WINDOW_CYCLES=40000;
    localparam BYTES=64*K*2, BEATS=BYTES/16;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0,window_valid=0,seal=0,result_ready=0,prefetch_valid=0,block_valid=0,cancel_valid=0;
    reg [31:0] window_tag=0,prefetch_tag=0,prefetch_ram_address=0,block_tag=0,cancel_tag=0;
    reg [1:0] prefetch_class=2;
    wire window_ready,window_busy,window_accepting,result_valid,deadline_missed,initialized;
    wire [31:0] result_tag,elapsed_cycles,accepted_count,completed_count,rejected_count;
    wire [3:0] result_status;
    wire [8:0] outstanding_count;
    wire prefetch_ready,prefetch_bank,prefetch_rejected;
    wire [31:0] prefetch_rejected_tag,ingress_rejected_count;
    wire block_ready,audio_valid,audio_last,audio_miss,block_done;
    reg audio_ready=0;
    wire [63:0] audio_data;
    wire [31:0] audio_tag;
    wire [1:0] free_mask,ready_mask,reading_mask;
    wire fault_valid,protocol_error,memory_fault;
    wire [31:0] fault_tag;
    wire [3:0] fault_status;
    wire ram_request_valid,ram_request_write,ram_response_ready;
    wire [31:0] ram_request_address;
    wire [127:0] ram_request_data;
    reg ram_request_ready=0,ram_response_valid=0,ram_response_error=0;
    reg [127:0] ram_response_data=0;
    ccs_dma_window #(.K(K),.HEADROOM_BITS(HEADROOM_BITS),.WINDOW_CYCLES(WINDOW_CYCLES)) dut (.*);

    reg [15:0] coefficients [0:CASES*64*K-1],expected [0:CASES*64-1];
    reg [4095:0] coeff_file,expected_file;
    reg [31:0] requests [0:4095];
    integer release_last [0:CASES-1];
    integer wr=0,rd=0,cycle=0,start_cycle=0,row,word_lane,index;
    integer windows=0,successes=0,misses=0,samples=0,faults=0,ingress_drops=0,resets=0,overlaps=0;
    integer last_service=0,callback_time=0,waits,frame,beats,previous_faults,saved_elapsed,saved_accepted;
    reg hold_response=0;
    reg [31:0] next_tag;
    integer stored_beats[0:1],fill_row[0:1],last_write_cycle[0:1];
    integer fill_bank,write_beat,check_lane,payload_writes=0,publications=0;

    function [31:0] source_address;
        input integer case_id;
        begin source_address=32'h200000+case_id*BYTES;end
    endfunction
    always @(negedge clk) begin
        ram_request_ready=rst_n && (!STALLS || cycle%7<5);
        audio_ready=rst_n && (!STALLS || cycle%11<7);
        if (!rst_n) ram_response_valid=0;
        else if (!ram_response_valid && !hold_response && rd<wr && (!STALLS || cycle%5!=1)) begin
            row=(requests[rd%4096]-32'h200000)/BYTES;
            if ((requests[rd%4096]-32'h200000)%BYTES!=BYTES-16 || cycle>=release_last[row]) begin
                ram_response_valid=1;ram_response_error=0;
                for (word_lane=0;word_lane<8;word_lane=word_lane+1) begin
                    index=(requests[rd%4096]-32'h200000)/2+word_lane;
                    if (index<0 || index>=CASES*64*K) $fatal(1,"Windowed DMA escaped coefficient image");
                    ram_response_data[16*word_lane +:16]=coefficients[index];
                end
            end
        end
    end
    always @(posedge clk) begin
        cycle<=cycle+1;
        if (!rst_n) begin
            wr=0;rd=0;
            stored_beats[0]=0;stored_beats[1]=0;fill_row[0]=0;fill_row[1]=0;
            last_write_cycle[0]=-1;last_write_cycle[1]=-1;
        end
        else begin
            if (window_valid && window_ready) begin start_cycle=cycle;windows=windows+1;end
            if (ram_request_valid && ram_request_ready) begin
                if (ram_request_write || wr-rd>=4096) $fatal(1,"Unexpected prepared-frame write/overflow");
                requests[wr%4096]=ram_request_address;wr=wr+1;
            end
            if (ram_response_valid && ram_response_ready) begin rd=rd+1;ram_response_valid<=0;end
            if (fault_valid) faults=faults+1;
            if (prefetch_rejected) ingress_drops=ingress_drops+1;
            if (prefetch_valid && prefetch_ready && !dut.drop_prefetch) begin
                stored_beats[prefetch_bank]=0;
                fill_row[prefetch_bank]=(prefetch_ram_address-32'h200000)/BYTES;
            end
            if (dut.payload_enable && dut.payload_write) begin
                fill_bank=(dut.payload_address-16'h8000)/BYTES;
                write_beat=(dut.payload_address-16'h8000)%BYTES/16;
                if (fill_bank<0 || fill_bank>1 || dut.payload_address%16 || write_beat!=stored_beats[fill_bank])
                    $fatal(1,"Payload pipeline lost/reordered a write");
                for (check_lane=0;check_lane<8;check_lane=check_lane+1)
                    if (dut.payload_write_data[16*check_lane +:16]!==coefficients[fill_row[fill_bank]*64*K+8*write_beat+check_lane])
                        $fatal(1,"Payload pipeline changed coefficient data");
                stored_beats[fill_bank]=stored_beats[fill_bank]+1;
                last_write_cycle[fill_bank]=cycle;payload_writes=payload_writes+1;
                if (dut.stream.state==dut.stream.DRAIN) overlaps=overlaps+1;
            end
            if (dut.dma.done_valid) begin last_service=dut.dma.done_service_cycles;callback_time=cycle-start_cycle;end
            if (dut.dma.done_valid && dut.dma.done_status==0) begin
                fill_bank=(dut.dma.done_descriptor[31:16]-16'h8000)/BYTES;
                if (stored_beats[fill_bank]!=BEATS || last_write_cycle[fill_bank]>=cycle)
                    $fatal(1,"Published before the final payload write committed");
                publications=publications+1;
            end
            if (protocol_error || memory_fault) $fatal(1,"Window integration ownership violation");
        end
    end
    task tick;begin @(posedge clk);#1;end endtask
    task reset_all;
        begin
            @(negedge clk);rst_n=0;window_valid=0;seal=0;result_ready=0;prefetch_valid=0;block_valid=0;cancel_valid=0;hold_response=0;
            for (row=0;row<CASES;row=row+1) release_last[row]=0;
            repeat (3) tick;
            if (audio_valid || result_valid || window_busy || prefetch_ready || prefetch_rejected) $fatal(1,"Reset leaked window work");
            @(negedge clk);rst_n=1;
            repeat (256) tick;
            if (!initialized || !window_ready || !block_ready || free_mask!=3 || ingress_rejected_count || deadline_missed)
                $fatal(1,"Windowed stream reset incomplete");
            resets=resets+1;
        end
    endtask
    task begin_window;
        input [31:0] wanted_tag;
        begin
            @(negedge clk);window_valid=1;window_tag=wanted_tag;
            @(posedge clk);waits=0;
            while (!window_ready) begin @(posedge clk);waits=waits+1;if (waits>100) $fatal(1,"New window blocked after drain");end
            @(negedge clk);window_valid=0;
        end
    endtask
    task prefetch;
        input [31:0] wanted_tag;
        input integer case_id;
        input expected_rejection;
        begin
            @(negedge clk);prefetch_valid=1;prefetch_tag=wanted_tag;prefetch_ram_address=source_address(case_id);
            @(posedge clk);waits=0;
            while (!prefetch_ready) begin
                @(posedge clk);waits=waits+1;
                if (waits>WINDOW_CYCLES+5) $fatal(1,"Obsolete ingress request was not retired");
            end
            #1;
            if (prefetch_rejected!==expected_rejection || (expected_rejection && prefetch_rejected_tag!==wanted_tag))
                $fatal(1,"Wrong ingress admission/rejection");
            @(negedge clk);prefetch_valid=0;
        end
    endtask
    task seal_window;begin @(negedge clk);seal=1;tick;@(negedge clk);seal=0;end endtask
    task wait_result;
        begin
            waits=0;
            while (!result_valid) begin tick;waits=waits+1;if (waits>WINDOW_CYCLES+2000) $fatal(1,"Window result stuck");end
            if (window_busy || outstanding_count) $fatal(1,"Window reported before all DMA drained");
        end
    endtask
    task consume_result;begin @(negedge clk);result_ready=1;tick;@(negedge clk);result_ready=0;end endtask
    task check_good_result;
        input integer count;
        begin
            wait_result;
            if (result_status || accepted_count!=count || completed_count!=count || rejected_count || ingress_rejected_count)
                $fatal(1,"Healthy batch accounting count=%0d accepted=%0d completed=%0d status=%0d",count,accepted_count,completed_count,result_status);
        end
    endtask
    task render;
        input [31:0] wanted_tag;
        input integer expected_row;
        input expected_miss;
        input integer next_row;
        input [31:0] future_tag;
        integer output_beat,n,render_start,elapsed;
        reg sent_next,need_seal,was_stalled;
        reg [63:0] stalled_data;
        reg stalled_last;
        reg [15:0] wanted_sample;
        begin
            @(negedge clk);block_valid=1;block_tag=wanted_tag;
            if (!block_ready) $fatal(1,"Audio launch blocked by DMA");
            @(posedge clk);render_start=cycle;
            @(negedge clk);block_valid=0;
            output_beat=0;sent_next=next_row<0;need_seal=0;was_stalled=0;
            while (output_beat<16) begin
                seal=need_seal;need_seal=0;
                prefetch_valid=!sent_next;
                if (!sent_next) begin prefetch_tag=future_tag;prefetch_ram_address=source_address(next_row);end
                @(posedge clk);
                if (prefetch_valid && prefetch_ready) begin sent_next=1;need_seal=1;end
                if (was_stalled && (!audio_valid || audio_data!==stalled_data || audio_last!==stalled_last)) $fatal(1,"Audio changed under backpressure");
                was_stalled=audio_valid && !audio_ready;stalled_data=audio_data;stalled_last=audio_last;
                if (audio_valid) begin
                    if (audio_tag!==wanted_tag || audio_miss!==expected_miss || audio_last!==(output_beat==15)) $fatal(1,"Wrong audio tag/status/last");
                    for (n=0;n<4;n=n+1) begin
                        wanted_sample=(expected_row<0)?16'd0:expected[expected_row*64+output_beat*4+n];
                        if (audio_data[16*n +:16]!==wanted_sample) $fatal(1,"Windowed audio mismatch K=%0d row=%0d sample=%0d",K,expected_row,output_beat*4+n);
                    end
                    if (audio_ready) begin output_beat=output_beat+1;samples=samples+4;end
                end
                elapsed=cycle-render_start+1;
                if (elapsed>1000) $fatal(1,"Audio waited for DMA deadline/drain");
                #1;
                if (prefetch_rejected && next_row>=0) $fatal(1,"Healthy N+2 refill rejected");
                if (block_done!==(output_beat==16)) $fatal(1,"Wrong final audio handshake");
                @(negedge clk);
            end
            seal=0;prefetch_valid=0;
            if (!sent_next) $fatal(1,"No N+2 refill admission during decode");
            if (!STALLS && elapsed!=(expected_miss ? 18 : (K==8 ? 642 : 331))) $fatal(1,"Budget wrapper changed audio latency");
            if (expected_miss) misses=misses+1;else successes=successes+1;
        end
    endtask

    initial begin
        if (!$value$plusargs("coeffs=%s",coeff_file) || !$value$plusargs("expected=%s",expected_file)) $fatal(1,"Missing windowed audio vectors");
        $readmemh(coeff_file,coefficients);$readmemh(expected_file,expected);
        reset_all;
        prefetch(99,3,1); // Closed ingress consumes obsolete work without ownership.
        if (free_mask!=3 || dut.submit_valid) $fatal(1,"Closed ingress reserved a buffer");
        render(99,-1,1,-1,0);
        // Each batch includes two transfers and preserves its report while
        // both audio blocks run. Sealing must include the final pending owner command.
        for (frame=0;frame<CASES;frame=frame+2) begin
            begin_window(frame);
            prefetch(1000+frame,frame,0);prefetch(1001+frame,frame+1,0);seal_window;
            check_good_result(2);
            saved_elapsed=elapsed_cycles;saved_accepted=accepted_count;
            render(1000+frame,frame,0,-1,0);render(1001+frame,frame+1,0,-1,0);
            if (!result_valid || elapsed_cycles!=saved_elapsed || accepted_count!=saved_accepted || window_ready) $fatal(1,"Held window result mutated");
            consume_result;
        end
        // Two buffers already ready: release N through the actual decoder,
        // then refill N+2 during its compute under a newly opened shared window.
        begin_window(100);prefetch(10,3,0);prefetch(11,4,0);seal_window;check_good_result(2);consume_result;
        begin_window(101);
        render(10,3,0,5,12);
        check_good_result(1);
        if (elapsed_cycles<64 || overlaps==0) $fatal(1,"Buffer wait/refill did not enter the shared window");
        render(11,4,0,-1,0);render(12,5,0,-1,0);consume_result;

        // Both transfers individually fit, but queue wait plus service crosses
        // the shared deadline. Fallback completes while the last RAM beat is held.
        begin_window(200);previous_faults=faults;
        release_last[3]=start_cycle+WINDOW_CYCLES/2-1;
        release_last[4]=start_cycle+WINDOW_CYCLES+1000;
        prefetch(20,3,0);prefetch(21,4,0);seal_window;
        while (!ready_mask[0]) tick;
        render(20,3,0,-1,0);
        while (!deadline_missed) tick;
        saved_elapsed=elapsed_cycles;
        @(negedge clk);window_valid=1;window_tag=32'hbadbad;
        render(21,3,1,-1,0);
        if (!window_busy || window_ready || result_valid || free_mask==3 || elapsed_cycles<=saved_elapsed || result_tag!=200)
            $fatal(1,"Late DMA blocked audio, reset its budget or freed its bank early");
        @(negedge clk);window_valid=0;release_last[4]=0;
        wait_result;repeat (2) tick;
        if (result_status!=8 || accepted_count!=2 || completed_count!=2 || last_service>=WINDOW_CYCLES || callback_time<=WINDOW_CYCLES || faults!=previous_faults+1 || free_mask!=3)
            $fatal(1,"Cumulative deadline did not gate publication");
        consume_result;release_last[3]=0;

        // A held prefetch cannot wait through a full pool and silently inherit
        // the next window. Its rejection is acknowledged at the deadline.
        begin_window(300);prefetch(30,3,0);prefetch(31,4,0);seal_window;check_good_result(2);consume_result;
        begin_window(301);
        prefetch(32,5,1);wait_result;
        if (result_status!=8 || elapsed_cycles!=WINDOW_CYCLES || accepted_count || completed_count || ingress_rejected_count!=1 || ready_mask!=3)
            $fatal(1,"Unadmitted obsolete prefetch was not retired/accounted");
        saved_elapsed=elapsed_cycles;
        prefetch(33,6,1);
        if (elapsed_cycles!=saved_elapsed || ingress_rejected_count!=1) $fatal(1,"Closed rejection mutated held report");
        consume_result;
        begin_window(302);
        @(negedge clk);prefetch_valid=1;prefetch_tag=34;prefetch_ram_address=source_address(5);
        repeat (4) begin tick;if (prefetch_ready) $fatal(1,"Full buffers admitted another frame");end
        @(negedge clk);cancel_valid=1;cancel_tag=34;
        tick;
        if (!prefetch_rejected || prefetch_rejected_tag!=34 || ready_mask!=3 || ingress_rejected_count) $fatal(1,"Explicitly canceled ingress was not retired safely");
        @(negedge clk);prefetch_valid=0;cancel_valid=0;
        seal_window;check_good_result(0);consume_result;
        render(30,3,0,-1,0);render(31,4,0,-1,0);

        // One clock is reserved for the owner's registered submission. A
        // final-edge ingress request is rejected rather than escaping accounting.
        begin_window(400);
        while (elapsed_cycles<WINDOW_CYCLES-1) tick;
        prefetch(40,3,1);wait_result;
        if (accepted_count || completed_count || ingress_rejected_count!=1 || result_status!=8 || free_mask!=3)
            $fatal(1,"Final-edge owner claim escaped the window");
        consume_result;
        // The last allowed owner claim is visible on the deadline edge; its
        // later real DMA completion is overdue and never becomes ready.
        begin_window(401);
        while (elapsed_cycles<WINDOW_CYCLES-2) tick;
        prefetch(41,3,0);wait_result;
        if (accepted_count!=1 || completed_count!=1 || result_status!=8 || ready_mask || free_mask!=3) $fatal(1,"Late pending descriptor published");
        render(41,4,1,-1,0);consume_result;

        // Reset during an expired active window flushes queue/adapter and
        // clears replay history; no pre-reset late response may return.
        begin_window(500);hold_response=1;prefetch(50,3,0);seal_window;
        while (!deadline_missed) tick;
        reset_all;
        render(50,-1,1,-1,0);
        // Reset between RAM acceptance and the registered payload store.
        begin_window(502);prefetch(52,3,0);seal_window;
        while (!dut.payload_enable) tick;
        reset_all;
        render(52,-1,1,-1,0);
        begin_window(501);prefetch(51,3,0);seal_window;check_good_result(1);
        render(51,3,0,-1,0);consume_result;
        $display("PASS windowed_audio K=%0d headroom=%0d stalls=%0d cases=%0d windows=%0d success=%0d misses=%0d samples=%0d faults=%0d ingress_drops=%0d resets=%0d overlaps=%0d payload_writes=%0d publications=%0d",K,HEADROOM_BITS,STALLS,CASES,windows,successes,misses,samples,faults,ingress_drops,resets,overlaps,payload_writes,publications);
        $finish;
    end
    initial begin #100000000;$fatal(1,"Windowed audio test timed out");end
endmodule
