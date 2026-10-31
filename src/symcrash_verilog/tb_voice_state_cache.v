`timescale 1ns/1ps
module tb_voice_state_cache;
    parameter STALLS=0, LATENCY=0;
    reg clk=0;always #5 clk=!clk;
    reg rst_n=0;
    reg prefetch_valid=0;wire prefetch_ready;reg [3:0] prefetch_slot=0;
    reg [15:0] prefetch_voice_id=0;reg [31:0] prefetch_tag=0,prefetch_ram_address=0;
    reg [1:0] prefetch_class=0;wire prefetch_rejected;
    reg acquire_valid=0;wire acquire_ready;reg [3:0] acquire_slot=0;
    reg [15:0] acquire_voice_id=0;reg [31:0] acquire_tag=0;
    wire acquire_result_valid,acquire_hit,lease_valid;wire [3:0] lease_slot;
    reg dsp_valid=0,dsp_write=0;wire dsp_ready;reg [6:0] dsp_word=0;
    reg [127:0] dsp_write_data=0;wire dsp_response_valid;
    reg dsp_response_ready=0;wire [127:0] dsp_response_data;
    reg release_valid=0;wire release_ready;reg retry_valid=0;wire retry_ready;
    reg [3:0] retry_slot=0;wire retry_rejected;
    wire dma_submit_valid,dma_submit_ready;wire [63:0] dma_submit_descriptor;
    wire actual_done_valid;wire [63:0] actual_done_descriptor;wire [3:0] actual_done_status;
    reg false_done=0;reg [63:0] false_descriptor=0;
    wire dma_done_valid=actual_done_valid || false_done;
    wire [63:0] dma_done_descriptor=false_done ? false_descriptor : actual_done_descriptor;
    wire [3:0] dma_done_status=false_done ? 4'd0 : actual_done_status;
    wire dma_bram_enable,dma_bram_write;wire [15:0] dma_bram_address;
    wire [127:0] dma_bram_write_data,dma_bram_read_data;
    wire completion_valid,completion_write;wire [3:0] completion_slot,completion_status;
    wire [15:0] completion_voice_id;wire [31:0] completion_tag;
    wire [15:0] valid_mask,dirty_mask;wire dma_busy,memory_fault;

    reg acquire_readonly=0;
    wire lease_readonly;
    reg read_valid=0,read_response_ready=0,read_release_valid=0;
    reg [6:0] read_word=0;
    wire read_ready,read_response_valid,read_release_ready;
    integer readonly_reads=0,owner_conflicts=0;
    voice_state_cache dut(.*);
    reg window_valid=0,seal=0,result_ready=0;reg [31:0] window_tag=0;
    wire window_ready,window_busy,window_accepting,window_final_cycle,result_valid,deadline_missed,initialized;
    wire [31:0] result_tag,elapsed_cycles,accepted_count,completed_count,rejected_count,done_service_cycles;
    wire [3:0] result_status;wire [8:0] outstanding_count;
    wire ram_request_valid,ram_request_write,ram_response_ready;
    wire [31:0] ram_request_address;wire [127:0] ram_request_data;
    reg ram_request_ready=0,ram_response_valid=0,ram_response_error=0;
    reg [127:0] ram_response_data=0;
    dma_window #(.WINDOW_CYCLES(40000)) window_engine (
        .clk(clk),.rst_n(rst_n),.window_valid(window_valid),.window_ready(window_ready),.window_tag(window_tag),
        .seal(seal),.result_valid(result_valid),.result_ready(result_ready),.result_tag(result_tag),
        .window_busy(window_busy),.window_accepting(window_accepting),.window_final_cycle(window_final_cycle),
        .elapsed_cycles(elapsed_cycles),.accepted_count(accepted_count),.completed_count(completed_count),
        .rejected_count(rejected_count),.result_status(result_status),.outstanding_count(outstanding_count),
        .deadline_missed(deadline_missed),.initialized(initialized),
        .submit_valid(dma_submit_valid),.submit_ready(dma_submit_ready),.submit_descriptor(dma_submit_descriptor),
        .done_valid(actual_done_valid),.done_descriptor(actual_done_descriptor),.done_status(actual_done_status),
        .match_descriptors(128'd0),.done_matches(),.done_service_cycles(done_service_cycles),
        .ram_request_valid(ram_request_valid),.ram_request_ready(ram_request_ready),.ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address),.ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid),.ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error),.ram_response_data(ram_response_data),
        .bram_enable(dma_bram_enable),.bram_write(dma_bram_write),.bram_address(dma_bram_address),
        .bram_write_data(dma_bram_write_data),.bram_read_data(dma_bram_read_data));
    reg [127:0] ram[0:32767],expected[0:32767];
    reg [31:0] requests[0:65535];reg [127:0] request_data[0:65535];reg request_write[0:65535];
    integer request_cycle[0:65535];
    integer observed_words=0,resets_with_payload=0;
    integer wr=0,rd=0,cycle=0,i,j,lane,windows=0,completions=0,reads=0,writes=0,hits=0,misses=0;
    integer overlaps=0,stores=0,fetches=0,resets=0,max_window=0,ram_reads=0,ram_writes=0,late_windows=0,max_late=0;
    integer fault_word=-1,last_status=0,last_slot=0,last_voice=0,last_tag=0;
    integer start_count,saved_count,waits,index;
    reg hold_response=0,held_response=0,was_submit_stalled=0;
    reg [127:0] held_data;
    reg [63:0] held_descriptor;
    function [127:0] pattern;
        input integer address,version;
        integer k;reg [31:0] value;
        begin
            for(k=0;k<4;k=k+1) begin
                value=(address*32'h9e3779b9) ^ (version*32'h13579bdf) ^ (32'h80008001+k*32'h01020408);
                pattern[k*32 +:32]=value;
            end
        end
    endfunction
    always @(negedge clk) begin
        ram_request_ready=rst_n && wr-rd<8 && (!STALLS || cycle%7<5);
        if(!rst_n) ram_response_valid=0;
        else if(!ram_response_valid && rd<wr && cycle-request_cycle[rd]>=LATENCY && !hold_response && (!STALLS || cycle%5!=1)) begin
            ram_response_valid=1;ram_response_data=ram[requests[rd]>>4];
            ram_response_error=(fault_word>=0 && requests[rd][10:4]==fault_word);
            if(ram_response_error) fault_word=-1;
        end
    end
    always @(posedge clk) begin
        cycle<=cycle+1;
        if(!rst_n) begin wr=0;rd=0;observed_words=0;was_submit_stalled=0;held_response=0;end
        else begin
            if(ram_request_valid && ram_request_ready) begin
                if(ram_request_address>=524288 || ram_request_address%16 || wr>=65536) $fatal(1,"RAM request bounds");
                requests[wr]=ram_request_address;request_data[wr]=ram_request_data;request_write[wr]=ram_request_write;
                request_cycle[wr]=cycle;wr=wr+1;
                if(ram_request_write) ram_writes=ram_writes+1;else ram_reads=ram_reads+1;
            end
            if(ram_response_valid && ram_response_ready) begin
                if(request_write[rd] && !ram_response_error) ram[requests[rd]>>4]=request_data[rd];
                rd=rd+1;ram_response_valid<=0;
            end
            if(held_response && (!dsp_response_valid || dsp_response_data!==held_data)) $fatal(1,"DSP response changed under backpressure");
            held_response=dsp_response_valid && !dsp_response_ready;held_data=dsp_response_data;
            if(was_submit_stalled && (!dma_submit_valid || dma_submit_descriptor!==held_descriptor)) $fatal(1,"Descriptor changed under backpressure");
            was_submit_stalled=dma_submit_valid && !dma_submit_ready;held_descriptor=dma_submit_descriptor;
            if((prefetch_valid && prefetch_ready) || (release_valid && release_ready && dirty_mask[lease_slot]) ||
                (retry_valid && retry_ready)) observed_words=0;
            if(dma_submit_valid && dma_submit_ready) begin
                if(dma_submit_descriptor[15:8]!=128 || dma_submit_descriptor[26:16]!=0) $fatal(1,"Wrong page geometry");
                if(dma_submit_descriptor[2]) stores=stores+1;else fetches=fetches+1;
            end
            if(dma_bram_enable && !dma_bram_address[15]) observed_words=observed_words+1;
            if(completion_valid) begin
                completions=completions+1;last_status=completion_status;last_slot=completion_slot;
                last_voice=completion_voice_id;last_tag=completion_tag;
            end
            if(dma_busy && dsp_valid && dsp_ready) overlaps=overlaps+1;
            if(memory_fault) $fatal(1,"Unexpected cache memory protocol fault");
        end
    end
    task tick;begin @(posedge clk);#1;end endtask
    task reset_all;
        begin
            if(dma_busy) begin
                if(observed_words==0) $fatal(1,"Reset probe missed the intended transfer stage");
                resets_with_payload=resets_with_payload+1;
            end
            @(negedge clk);rst_n=0;window_valid=0;seal=0;result_ready=0;prefetch_valid=0;
            acquire_readonly=0;read_valid=0;read_response_ready=0;read_release_valid=0;acquire_valid=0;dsp_valid=0;dsp_response_ready=0;release_valid=0;retry_valid=0;hold_response=0;false_done=0;
            repeat(3) tick;
            if(valid_mask || dirty_mask || lease_valid || dsp_response_valid || dma_submit_valid || acquire_result_valid) $fatal(1,"Reset exposed state");
            @(negedge clk);rst_n=1;repeat(260) tick;
            if(!initialized || !window_ready) $fatal(1,"DMA reset incomplete");resets=resets+1;
        end
    endtask
    task open_window;
        begin
            @(negedge clk);window_valid=1;window_tag=windows+1;
            waits=0;while(!window_ready) begin tick;waits=waits+1;if(waits>300) $fatal(1,"Open timeout");end
            tick;@(negedge clk);window_valid=0;windows=windows+1;
        end
    endtask
    task close_window;
        input [3:0] wanted_status;
        begin
            @(negedge clk);seal=1;
            waits=0;while(!result_valid) begin tick;waits=waits+1;if(waits>41000) $fatal(1,"Window timeout");end
            if(result_status!==wanted_status || completed_count!=accepted_count || outstanding_count!=0 ||
                (!wanted_status[3] && elapsed_cycles>40000) || (wanted_status[3] && elapsed_cycles<40000))
                $fatal(1,"Window result status=%d wanted=%d accepted=%d completed=%d elapsed=%d",result_status,wanted_status,accepted_count,completed_count,elapsed_cycles);
            if(wanted_status[3]) begin late_windows=late_windows+1;if(elapsed_cycles>max_late) max_late=elapsed_cycles;end
            else if(elapsed_cycles>max_window) max_window=elapsed_cycles;
            repeat(3) tick;@(negedge clk);result_ready=1;tick;@(negedge clk);result_ready=0;seal=0;
        end
    endtask
    task start_fetch;
        input integer slot,voice,tag,page;
        begin
            @(negedge clk);prefetch_slot=slot;prefetch_voice_id=voice;prefetch_tag=tag;
            prefetch_ram_address=page*2048;prefetch_class=voice%4;prefetch_valid=1;
            waits=0;while(!prefetch_ready) begin tick;waits=waits+1;if(waits>5000) $fatal(1,"Fetch admission timeout");end
            tick;@(negedge clk);prefetch_valid=0;
        end
    endtask
    task wait_completion;
        input integer before_count,slot,voice,tag,status;
        begin
            waits=0;while(completions==before_count) begin tick;waits=waits+1;if(waits>6000) $fatal(1,"Completion timeout");end
            if(completions!=before_count+1 || last_status!=status || last_slot!=slot || last_voice!=voice || last_tag!=tag)
                $fatal(1,"Wrong completion status=%d slot=%d voice=%d tag=%d",last_status,last_slot,last_voice,last_tag);
        end
    endtask
    task acquire_page;
        input integer slot,voice,tag,wanted_hit;
        begin
            @(negedge clk);acquire_slot=slot;acquire_voice_id=voice;acquire_tag=tag;acquire_valid=1;
            if(!acquire_ready) $fatal(1,"Acquire blocked instead of bounded hit/miss");
            tick;
            if(acquire_result_valid || lease_valid || acquire_ready)
                $fatal(1,"Pending cache query escaped reservation");
            @(negedge clk);acquire_valid=0;acquire_slot=slot^15;acquire_voice_id=16'hffff;acquire_tag=0;
            tick;
            if(!acquire_result_valid || acquire_hit!=wanted_hit || lease_valid!=wanted_hit) $fatal(1,"Wrong acquired context slot=%d voice=%d tag=%d",slot,voice,tag);
            if(wanted_hit) hits=hits+1;else misses=misses+1;
            @(negedge clk);acquire_valid=0;
        end
    endtask
    task access_word;
        input integer word_index,is_write,page,version,hold;
        reg [127:0] wanted;
        begin
            wanted=is_write ? pattern(page*128+word_index,version) : expected[page*128+word_index];
            @(negedge clk);dsp_valid=1;dsp_write=is_write;dsp_word=word_index;dsp_write_data=wanted;dsp_response_ready=0;
            if(!dsp_ready) $fatal(1,"DSP unexpectedly blocked");tick;
            @(negedge clk);dsp_valid=0;
            if(!dsp_response_valid || dsp_response_data!==wanted) $fatal(1,"Wrong page data page=%d word=%d write=%d",page,word_index,is_write);
            repeat(hold) begin
                tick;
                if(dsp_ready || release_ready) $fatal(1,"Stalled response failed to retain lease");
            end
            if(is_write) begin expected[page*128+word_index]=wanted;writes=writes+1;end else reads=reads+1;
            @(negedge clk);dsp_response_ready=1;tick;@(negedge clk);dsp_response_ready=0;
        end
    endtask
    task release_page;
        begin
            @(negedge clk);release_valid=1;
            waits=0;while(!release_ready) begin tick;waits=waits+1;if(waits>6000) $fatal(1,"Release timeout");end
            tick;@(negedge clk);release_valid=0;
            if(lease_valid) $fatal(1,"Lease remained after release");
        end
    endtask
    task retry_page;
        input integer slot;
        begin
            @(negedge clk);retry_valid=1;retry_slot=slot;
            if(!retry_ready) $fatal(1,"Retry unexpectedly blocked");tick;@(negedge clk);retry_valid=0;
        end
    endtask
    initial begin
        for(i=0;i<32768;i=i+1) begin ram[i]=pattern(i,0);expected[i]=ram[i];end
        reset_all;
        for(i=0;i<16;i=i+1) acquire_page(i,16'hfffe,32'hfffffff0+i,0);
        open_window;
        // All sixteen physical pages, both documented eight-page batches.
        for(i=0;i<16;i=i+1) begin
            start_count=completions;start_fetch(i,16'h100+i,32'hfffffff0+i,i);
            acquire_page(i,16'h100+i,32'hfffffff0+i,0);
            wait_completion(start_count,i,16'h100+i,32'hfffffff0+i,0);
        end
        close_window(0);
        if(valid_mask!=16'hffff || dirty_mask) $fatal(1,"Cache geometry/publication");

        // Both clients drive requests: only the captured lease class may act.
        acquire_readonly=1;acquire_page(0,16'h100,32'hfffffff0,1);
        acquire_readonly=0;
        if(!lease_readonly) $fatal(1,"Read lease class followed live acquisition input");
        dsp_valid=1;dsp_write=1;dsp_write_data=128'hdeadbeef;dsp_response_ready=1;release_valid=1;
        for(j=0;j<128;j=j+1) begin
            @(negedge clk);read_valid=1;read_word=j;read_response_ready=0;dsp_word=127-j;
            #1;if(!read_ready || dsp_ready || release_ready || dsp_response_valid)
                $fatal(1,"Wrong client granted during readonly lease");
            tick;
            @(negedge clk);read_valid=0;
            repeat(3) begin
                tick;
                if(!lease_valid || !read_response_valid || dsp_response_data!==expected[j] ||
                    read_release_ready || dsp_ready || release_ready || dsp_response_valid || dirty_mask)
                    $fatal(1,"Wrong client consumed readonly response or changed page");
            end
            readonly_reads=readonly_reads+1;
            @(negedge clk);read_response_ready=1;tick;
        end
        @(negedge clk);read_release_valid=1;
        #1;if(!read_release_ready) $fatal(1,"Readonly release blocked by non-owner requests");
        tick;
        if(lease_valid || dma_busy || dirty_mask) $fatal(1,"Readonly release dirtied page");
        @(negedge clk);read_release_valid=0;read_response_ready=0;
        dsp_valid=0;dsp_write=0;dsp_response_ready=0;release_valid=0;
        // A non-owner read/release must not steal a held writable response.
        acquire_page(0,16'h100,32'hfffffff0,1);acquire_readonly=1;
        read_valid=1;read_response_ready=1;read_release_valid=1;
        for(j=0;j<128;j=j+1) begin
            access_word(j,0,0,0,3);
            if(read_ready || read_response_valid || read_release_ready || !lease_valid)
                $fatal(1,"Wrong client granted during writable lease");
        end
        release_page;
        read_valid=0;read_response_ready=0;read_release_valid=0;acquire_readonly=0;
        owner_conflicts=2;
        // Full identity, full tag and all 2048 bytes survive; held reads create no stores.
        saved_count=stores;
        for(i=0;i<16;i=i+1) begin
            acquire_page(i,i,32'hfffffff0+i,0);
            acquire_page(i,16'h100+i,32'hffffffe0+i,0);
            acquire_page(i,16'h100+i,32'hfffffff0+i,1);
            for(j=0;j<128;j=j+1) access_word(j,0,i,0,(j==63) ? 3 : 0);
            release_page;
        end
        if(stores!=saved_count || dirty_mask) $fatal(1,"Held context advanced");
        // One active page and an unrelated next-context prefetch use both ports.
        acquire_page(0,16'h100,32'hfffffff0,1);open_window;
        start_count=completions;start_fetch(8,16'hfffe,7,31);
        for(j=0;j<128;j=j+1) access_word(j,0,0,0,0);
        wait_completion(start_count,8,16'hfffe,7,0);
        acquire_slot=8; // no second acquire is admitted while the first lease exists.
        if(acquire_ready) $fatal(1,"Second DSP lease admitted");
        for(j=0;j<128;j=j+1) access_word(j,1,0,1,0);
        start_count=completions;release_page;wait_completion(start_count,0,16'h100,32'hfffffff0,0);
        if(dirty_mask) $fatal(1,"Successful store remained dirty");
        for(j=0;j<128;j=j+1) if(ram[j]!==expected[j]) $fatal(1,"Write-through changed bytes");
        close_window(0);
        // Reuse a clean slot for another voice, then restore the stored context.
        open_window;start_count=completions;start_fetch(0,1,8,32);wait_completion(start_count,0,1,8,0);
        acquire_page(0,16'h100,32'hfffffff0,0);
        start_count=completions;start_fetch(0,16'h100,32'hfffffff0,0);wait_completion(start_count,0,16'h100,32'hfffffff0,0);
        close_window(0);acquire_page(0,16'h100,32'hfffffff0,1);
        for(j=0;j<128;j=j+1) access_word(j,0,0,0,0);release_page;
        // A partial fetch never publishes, even though later words arrive.
        open_window;fault_word=17;start_count=completions;start_fetch(15,16'hfffe,9,33);
        wait_completion(start_count,15,16'hfffe,9,4);close_window(4);
        acquire_page(15,16'hfffe,9,0);
        // Failed store keeps authoritative modified state, preventing reload/eviction.
        acquire_page(1,16'h101,32'hfffffff1,1);
        for(j=0;j<128;j=j+1) access_word(j,1,1,2,0);
        open_window;fault_word=13;start_count=completions;release_page;
        wait_completion(start_count,1,16'h101,32'hfffffff1,4);close_window(4);
        if(!dirty_mask[1] || !valid_mask[1]) $fatal(1,"Failed store discarded modified state");
        acquire_page(1,16'h101,32'hfffffff1,0);
        @(negedge clk);prefetch_slot=1;prefetch_ram_address=34*2048;prefetch_valid=1;
        repeat(5) begin tick;if(prefetch_ready || dma_submit_valid) $fatal(1,"Dirty line evicted");end
        @(negedge clk);prefetch_valid=0;
        @(negedge clk);prefetch_valid=1;prefetch_voice_id=16'hffff;prefetch_ram_address=34*2048;
        tick;if(!prefetch_rejected || dma_busy) $fatal(1,"Global identity used as voice page");
        @(negedge clk);prefetch_valid=0;
        open_window;start_count=completions;retry_page(1);wait_completion(start_count,1,16'h101,32'hfffffff1,0);close_window(0);
        for(j=0;j<128;j=j+1) if(ram[128+j]!==expected[128+j]) $fatal(1,"Retry lost modified state");
        acquire_page(1,16'h101,32'hfffffff1,1);
        for(j=0;j<128;j=j+1) access_word(j,0,1,0,0);release_page;
        // Reserved invalid operations leave metadata and memory untouched.
        retry_page(1);if(!retry_rejected) $fatal(1,"Clean retry accepted");
        @(negedge clk);prefetch_valid=1;prefetch_slot=1;prefetch_ram_address=3;
        tick;if(!prefetch_rejected || dma_busy) $fatal(1,"Unaligned prefetch accepted");
        @(negedge clk);prefetch_valid=0;
        // Full callback identity: an unrelated completion cannot release an active page.
        open_window;hold_response=1;start_count=completions;start_fetch(2,16'hfffd,10,35);
        repeat(8) tick;
        @(negedge clk);false_descriptor=dma_submit_descriptor ^ 64'h0000000100000000;false_done=1;
        tick;@(negedge clk);false_done=0;
        if(!dma_busy || completions!=start_count || valid_mask[2]) $fatal(1,"Unmatched callback published");
        hold_response=0;wait_completion(start_count,2,16'hfffd,10,0);close_window(0);
        // The real deadline can expire while RAM owns outstanding transfers.
        // Fetches remain unavailable; stores keep dirty state for a new-window retry.
        open_window;hold_response=1;start_count=completions;start_fetch(15,16'hfffc,24,37);
        repeat(40005) tick;
        if(!deadline_missed || !dma_busy || valid_mask[15]) $fatal(1,"Late fetch escaped ownership");
        hold_response=0;wait_completion(start_count,15,16'hfffc,24,8);close_window(8);
        acquire_page(15,16'hfffc,24,0);
        acquire_page(4,16'h104,32'hfffffff4,1);
        access_word(0,1,4,4,0);access_word(127,1,4,4,0);
        open_window;hold_response=1;start_count=completions;release_page;
        repeat(40005) tick;
        if(!deadline_missed || !dma_busy || !dirty_mask[4]) $fatal(1,"Late store lost modified state");
        hold_response=0;wait_completion(start_count,4,16'h104,32'hfffffff4,8);close_window(8);
        acquire_page(4,16'h104,32'hfffffff4,0);
        open_window;start_count=completions;retry_page(4);wait_completion(start_count,4,16'h104,32'hfffffff4,0);close_window(0);
        for(j=0;j<128;j=j+1) if(ram[4*128+j]!==expected[4*128+j]) $fatal(1,"Late retry changed bytes");
        // Same-edge acquisition wins over a replacement request for its page.
        open_window;start_count=completions;
        @(negedge clk);acquire_valid=1;acquire_slot=4;acquire_voice_id=16'h104;acquire_tag=32'hfffffff4;
        prefetch_valid=1;prefetch_slot=4;prefetch_voice_id=16'h2222;prefetch_tag=31;prefetch_ram_address=38*2048;
        #1;if(prefetch_ready || !acquire_ready) $fatal(1,"Prefetch won acquire collision");
        tick;if(prefetch_ready || acquire_result_valid || lease_valid)
            $fatal(1,"Replacement escaped pending query reservation");
        @(negedge clk);acquire_valid=0;acquire_slot=15;acquire_voice_id=16'hffff;acquire_tag=0;
        tick;if(!acquire_result_valid || !acquire_hit || !lease_valid || prefetch_ready)
            $fatal(1,"Acquire collision lost captured context");hits=hits+1;
        @(negedge clk);acquire_valid=0;
        access_word(0,0,4,0,5);access_word(127,0,4,0,0);release_page;
        tick;@(negedge clk);prefetch_valid=0;
        wait_completion(start_count,4,16'h2222,31,0);close_window(0);
        acquire_page(4,16'h104,32'hfffffff4,0);acquire_page(4,16'h2222,31,1);
        for(j=0;j<128;j=j+1) access_word(j,0,38,0,0);release_page;
        // Coordinated resets abort outstanding fetches and dirty/storing leases.
        for(i=0;i<3;i=i+1) begin
            open_window;start_fetch(3,5,11+i,36);
            while(observed_words<(i==0 ? 1 : i==1 ? 64 : 127)) tick;
            reset_all;acquire_page(3,5,11+i,0);
        end
        open_window;start_count=completions;start_fetch(3,5,20,36);wait_completion(start_count,3,5,20,0);close_window(0);
        acquire_page(3,5,20,1);access_word(127,1,36,3,5);
        open_window;release_page;while(observed_words<64) tick;
        reset_all;acquire_page(3,5,20,0);
        if(overlaps==0 || max_window>=40000 || resets_with_payload!=4) $fatal(1,"Missing overlap, deadline or reset coverage");
        if(readonly_reads!=128 || owner_conflicts!=2) $fatal(1,"Missing native ownership coverage");
        $display("PASS voice_cache stalls=%0d latency=%0d credits=8 pages=16 page_bytes=2048 reads=%0d writes=%0d hits=%0d misses=%0d completions=%0d fetches=%0d stores=%0d windows=%0d max_window=%0d late_windows=%0d max_late=%0d overlaps=%0d resets=%0d payload_resets=%0d ram_read_bytes=%0d ram_write_bytes=%0d readonly_reads=%0d owner_conflicts=%0d",STALLS,LATENCY,reads,writes,hits,misses,completions,fetches,stores,windows,max_window,late_windows,max_late,overlaps,resets,resets_with_payload,ram_reads*16,ram_writes*16,readonly_reads,owner_conflicts);
        $finish;
    end
    initial begin #20000000;$fatal(1,"Global timeout");end
endmodule
