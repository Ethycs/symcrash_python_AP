`timescale 1ns/1ps
module tb_voice_cache_stream;
    parameter STALLS=0, LATENCY=20, HEADROOM_BITS=2, CASES=108;
    reg block_valid=0;wire block_ready;reg [3:0] block_slot=0;
    reg [15:0] block_voice_id=0;reg [31:0] block_context_tag=0,block_tag=0;
    reg [1:0] block_k=0;reg [6:0] block_coeff_word=0;
    wire audio_valid,audio_ready,audio_last,block_done;wire [63:0] audio_data;
    wire [15:0] audio_voice_id;wire [31:0] audio_tag;wire [1:0] audio_status;
    reg hold_audio=0,collide_host=0,collide_prefetch=0;
    integer prefetch_conflicts=0;
    assign audio_ready=rst_n && !hold_audio && (!STALLS || (cycle%5!=1 && cycle%7!=3));
    reg [127:0] source_pages[0:CASES*128-1],updated_pages[0:CASES*128-1];
    reg [15:0] golden[0:CASES*128-1];reg [1:0] modes[0:CASES-1];reg [6:0] offsets[0:CASES-1];
    string source_path,update_path,golden_path,modes_path,offset_path;
    integer frame_index,slot_case[0:15],slot_version[0:15],frames=0,samples=0;
    integer min_cycles[0:2],max_cycles[0:2],silent_frames=0,held_frames=0,updated_frames=0;
    integer check_word,base_word,words_per_frame,reset_stage,voice_conflicts=0;
    reg audio_stalled=0;reg [63:0] stalled_audio;reg [15:0] stalled_voice;
    reg [31:0] stalled_tag;reg [1:0] stalled_status;reg stalled_last;
    always @(posedge clk) begin
        if(!rst_n) audio_stalled=0;
        else begin
            if(audio_stalled && (!audio_valid || audio_data!==stalled_audio || audio_voice_id!==stalled_voice ||
                audio_tag!==stalled_tag || audio_status!==stalled_status || audio_last!==stalled_last))
                $fatal(1,"Voice audio changed while stalled");
            audio_stalled=audio_valid && !audio_ready;stalled_audio=audio_data;stalled_voice=audio_voice_id;
            stalled_tag=audio_tag;stalled_status=audio_status;stalled_last=audio_last;
        end
    end
    function integer voice_for;input integer c;begin voice_for=16'h100+c%16+((c/16)%2)*256;end endfunction
    function [31:0] context_for;input integer c;begin context_for=32'ha0000000+c;end endfunction
    task begin_render;
        input integer c,voice,context_value,mode,offset,tag;
        begin
            @(negedge clk);block_valid=1;block_slot=c%16;block_voice_id=voice;block_context_tag=context_value;
            block_k=mode;block_coeff_word=offset;block_tag=tag;
            if(collide_host) begin
                state_acquire_valid=1;state_acquire_slot=c%16;state_acquire_voice_id=voice;state_acquire_tag=context_value;
                #1;if(state_acquire_ready) $fatal(1,"State port won simultaneous renderer acquisition");
            end
            if(collide_prefetch) begin
                prefetch_valid=1;prefetch_slot=c%16;prefetch_voice_id=16'heeee;
                prefetch_tag=32'habcdef01;prefetch_ram_address=255*2048;
                #1;if(prefetch_ready) $fatal(1,"Prefetch replaced queued acquired context");
            end
            waits=0;while(!block_ready) begin tick;waits=waits+1;if(waits>2000) $fatal(1,"Render admission timeout");end
            tick;@(negedge clk);block_valid=0;
            // Accepted command metadata must survive unrelated next inputs.
            block_voice_id=16'hffff;block_context_tag=0;block_k=3;block_coeff_word=127;block_tag=0;
        end
    endtask
    task render_case;
        input integer c,version,status,voice,context_value,mode,offset;
        integer beat,n,start_cycle,tag,elapsed,host_granted,host_accept,prefetch_accept,prefetch_granted;reg [15:0] wanted;
        begin
            tag=32'hd0000000+frames;
            begin_render(c,voice,context_value,mode,offset,tag);start_cycle=cycle-1;beat=0;elapsed=0;host_granted=0;prefetch_granted=0;
            while(beat<16) begin
                @(posedge clk);
                host_accept=collide_host && state_acquire_valid && state_acquire_ready;
                prefetch_accept=collide_prefetch && prefetch_valid && prefetch_ready;
                if(audio_valid && audio_ready) begin
                    if(audio_voice_id!==voice[15:0] || audio_tag!==tag || audio_status!==status[1:0] || audio_last!=(beat==15))
                        $fatal(1,"Audio identity/status case=%d beat=%d status=%d expected=%d",c,beat,audio_status,status);
                    for(n=0;n<4;n=n+1) begin
                        wanted=status ? 16'd0 : golden[c*128+version*64+beat*4+n];
                        if(audio_data[16*n +:16]!==wanted)
                            $fatal(1,"Voice sample case=%d version=%d sample=%d actual=%h expected=%h",c,version,beat*4+n,audio_data[16*n +:16],wanted);
                    end
                    beat=beat+1;samples=samples+4;
                end
                #1;
                if(prefetch_accept) begin prefetch_valid=0;prefetch_granted=1;end
                if(host_accept) state_acquire_valid=0;
                if(collide_host && state_acquire_result_valid) begin
                    if(!state_acquire_hit || !state_lease_valid) $fatal(1,"Queued state acquisition lost identity");
                    host_granted=1;
                end
                elapsed=elapsed+1;if(elapsed>3000) $fatal(1,"Render completion timeout");
            end
            if(!block_done) $fatal(1,"Final audio did not complete block");
            elapsed=cycle-start_cycle;
            if(status==0) begin
                if(elapsed<min_cycles[mode]) min_cycles[mode]=elapsed;
                if(elapsed>max_cycles[mode]) max_cycles[mode]=elapsed;
                if(version) updated_frames=updated_frames+1;else held_frames=held_frames+1;
            end else silent_frames=silent_frames+1;
            frames=frames+1;@(negedge clk);
            if(collide_prefetch) begin
                if(!prefetch_granted) $fatal(1,"Replacement stayed blocked after final coefficient read");
                collide_prefetch=0;prefetch_conflicts=prefetch_conflicts+1;
            end
            if(collide_host) begin
                if(!host_granted || !state_lease_valid || block_ready) $fatal(1,"State lease failed to isolate decoder drain");
                release_page;voice_conflicts=voice_conflicts+1;collide_host=0;
            end
        end
    endtask
    task write_coefficients;
        input integer c;
        integer word_i,first,last;
        begin
            first=offsets[c];last=first+(16<<modes[c]);
            acquire_page(c%16,voice_for(c),context_for(c),1);
            for(word_i=first;word_i<last;word_i=word_i+1) begin
                @(negedge clk);state_dsp_valid=1;state_dsp_write=1;state_dsp_word=word_i;
                state_dsp_write_data=updated_pages[c*128+word_i];state_dsp_response_ready=0;
                if(!state_dsp_ready) $fatal(1,"State update lost lease");tick;
                @(negedge clk);state_dsp_valid=0;
                if(!state_dsp_response_valid || state_dsp_response_data!==updated_pages[c*128+word_i]) $fatal(1,"State write echo");
                state_dsp_response_ready=1;tick;@(negedge clk);state_dsp_response_ready=0;writes=writes+1;
                expected[(c%256)*128+word_i]=updated_pages[c*128+word_i];
            end
        end
    endtask
    reg clk=0;always #5 clk=!clk;
    reg rst_n=0;
    reg prefetch_valid=0;wire prefetch_ready;reg [3:0] prefetch_slot=0;
    reg [15:0] prefetch_voice_id=0;reg [31:0] prefetch_tag=0,prefetch_ram_address=0;
    reg [1:0] prefetch_class=0;wire prefetch_rejected;
    reg state_acquire_valid=0;wire state_acquire_ready;reg [3:0] state_acquire_slot=0;
    reg [15:0] state_acquire_voice_id=0;reg [31:0] state_acquire_tag=0;
    wire state_acquire_result_valid,state_acquire_hit,state_lease_valid;wire [3:0] state_lease_slot;
    reg state_dsp_valid=0,state_dsp_write=0;wire state_dsp_ready;reg [6:0] state_dsp_word=0;
    reg [127:0] state_dsp_write_data=0;wire state_dsp_response_valid;
    reg state_dsp_response_ready=0;wire [127:0] state_dsp_response_data;
    reg state_release_valid=0;wire state_release_ready;reg retry_valid=0;wire retry_ready;
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
    voice_cache_stream #(.HEADROOM_BITS(HEADROOM_BITS)) dut(.*);
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
    reg [31:0] requests[0:7];reg [127:0] request_data[0:7];reg request_write[0:7];
    integer request_cycle[0:7];
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
        else if(!ram_response_valid && rd<wr && cycle-request_cycle[rd%8]>=LATENCY && !hold_response && (!STALLS || cycle%5!=1)) begin
            ram_response_valid=1;ram_response_data=ram[requests[rd%8]>>4];
            ram_response_error=(fault_word>=0 && requests[rd%8][10:4]==fault_word);
            if(ram_response_error) fault_word=-1;
        end
    end
    always @(posedge clk) begin
        cycle<=cycle+1;
        if(!rst_n) begin wr=0;rd=0;observed_words=0;was_submit_stalled=0;held_response=0;end
        else begin
            if(ram_request_valid && ram_request_ready) begin
                if(ram_request_address>=524288 || ram_request_address%16 || wr-rd>=8) $fatal(1,"RAM request bounds");
                requests[wr%8]=ram_request_address;request_data[wr%8]=ram_request_data;request_write[wr%8]=ram_request_write;
                request_cycle[wr%8]=cycle;wr=wr+1;
                if(ram_request_write) ram_writes=ram_writes+1;else ram_reads=ram_reads+1;
            end
            if(ram_response_valid && ram_response_ready) begin
                if(request_write[rd%8] && !ram_response_error) ram[requests[rd%8]>>4]=request_data[rd%8];
                rd=rd+1;ram_response_valid<=0;
            end
            if(held_response && (!state_dsp_response_valid || state_dsp_response_data!==held_data)) $fatal(1,"DSP response changed under backpressure");
            held_response=state_dsp_response_valid && !state_dsp_response_ready;held_data=state_dsp_response_data;
            if(was_submit_stalled && (!dma_submit_valid || dma_submit_descriptor!==held_descriptor)) $fatal(1,"Descriptor changed under backpressure");
            was_submit_stalled=dma_submit_valid && !dma_submit_ready;held_descriptor=dma_submit_descriptor;
            if((prefetch_valid && prefetch_ready) || (state_release_valid && state_release_ready && dirty_mask[state_lease_slot]) ||
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
            if(dma_busy && state_dsp_valid && state_dsp_ready) overlaps=overlaps+1;
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
            @(negedge clk);rst_n=0;block_valid=0;hold_audio=0;window_valid=0;seal=0;result_ready=0;prefetch_valid=0;
            state_acquire_valid=0;state_dsp_valid=0;state_dsp_response_ready=0;state_release_valid=0;retry_valid=0;hold_response=0;false_done=0;
            repeat(3) tick;
            if(valid_mask || dirty_mask || state_lease_valid || state_dsp_response_valid || dma_submit_valid || state_acquire_result_valid) $fatal(1,"Reset exposed state");
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
            @(negedge clk);state_acquire_slot=slot;state_acquire_voice_id=voice;state_acquire_tag=tag;state_acquire_valid=1;
            if(!state_acquire_ready) $fatal(1,"Acquire blocked instead of bounded hit/miss");
            tick;@(negedge clk);state_acquire_valid=0;tick;
            if(state_acquire_result_valid || state_acquire_ready) $fatal(1,"Pending query escaped its reservation");
            tick;
            if(!state_acquire_result_valid || state_acquire_hit!=wanted_hit || state_lease_valid!=wanted_hit) $fatal(1,"Wrong acquired context_value slot=%d voice=%d tag=%d",slot,voice,tag);
            if(wanted_hit) hits=hits+1;else misses=misses+1;
            @(negedge clk);state_acquire_valid=0;
        end
    endtask
    task access_word;
        input integer word_index,is_write,page,version,hold;
        reg [127:0] wanted;
        begin
            wanted=is_write ? pattern(page*128+word_index,version) : expected[page*128+word_index];
            @(negedge clk);state_dsp_valid=1;state_dsp_write=is_write;state_dsp_word=word_index;state_dsp_write_data=wanted;state_dsp_response_ready=0;
            if(!state_dsp_ready) $fatal(1,"DSP unexpectedly blocked");tick;
            @(negedge clk);state_dsp_valid=0;
            if(!state_dsp_response_valid || state_dsp_response_data!==wanted) $fatal(1,"Wrong page data page=%d word=%d write=%d",page,word_index,is_write);
            repeat(hold) begin
                tick;
                if(state_dsp_ready || state_release_ready) $fatal(1,"Stalled response failed to retain lease");
            end
            if(is_write) begin expected[page*128+word_index]=wanted;writes=writes+1;end else reads=reads+1;
            @(negedge clk);state_dsp_response_ready=1;tick;@(negedge clk);state_dsp_response_ready=0;
        end
    endtask
    task release_page;
        begin
            @(negedge clk);state_release_valid=1;
            waits=0;while(!state_release_ready) begin tick;waits=waits+1;if(waits>6000) $fatal(1,"Release timeout");end
            tick;@(negedge clk);state_release_valid=0;
            if(state_lease_valid) $fatal(1,"Lease remained after release");
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
        if(!$value$plusargs("pages=%s",source_path) || !$value$plusargs("updates=%s",update_path) ||
           !$value$plusargs("expected=%s",golden_path) || !$value$plusargs("modes=%s",modes_path) ||
           !$value$plusargs("offsets=%s",offset_path)) $fatal(1,"Missing voice vectors");
        $readmemh(source_path,source_pages);$readmemh(update_path,updated_pages);$readmemh(golden_path,golden);
        $readmemh(modes_path,modes);$readmemh(offset_path,offsets);
        for(i=0;i<32768;i=i+1) begin ram[i]=pattern(i,0);expected[i]=ram[i];end
        for(i=0;i<16;i=i+1) begin slot_case[i]=-1;slot_version[i]=0;end
        for(i=0;i<3;i=i+1) begin min_cycles[i]=999999;max_cycles[i]=0;end
        reset_all;
        for(frame_index=0;frame_index<CASES;frame_index=frame_index+1) begin
            for(j=0;j<128;j=j+1) begin
                ram[(frame_index%256)*128+j]=source_pages[frame_index*128+j];
                expected[(frame_index%256)*128+j]=source_pages[frame_index*128+j];
            end
            open_window;start_count=completions;
            start_fetch(frame_index%16,voice_for(frame_index),context_for(frame_index),frame_index%256);
            if(frame_index>0) begin
                i=frame_index-1;
                render_case(i,slot_version[i%16],0,voice_for(i),context_for(i),modes[i],offsets[i]);
            end
            wait_completion(start_count,frame_index%16,voice_for(frame_index),context_for(frame_index),0);close_window(0);
            slot_case[frame_index%16]=frame_index;slot_version[frame_index%16]=0;
            if(frame_index>=16) begin
                i=frame_index-16;
                render_case(i,0,1,voice_for(i),context_for(i),modes[i],offsets[i]);
            end
            if(frame_index==0) collide_host=1;
            render_case(frame_index,0,0,voice_for(frame_index),context_for(frame_index),modes[frame_index],offsets[frame_index]);
            render_case(frame_index,0,0,voice_for(frame_index),context_for(frame_index),modes[frame_index],offsets[frame_index]);
            // Full identity and tag mismatches cannot reuse the previous audio.
            render_case(frame_index,0,1,voice_for(frame_index)^16'h1000,context_for(frame_index),modes[frame_index],offsets[frame_index]);
            render_case(frame_index,0,1,voice_for(frame_index),context_for(frame_index)^32'h10000000,modes[frame_index],offsets[frame_index]);
            if(frame_index%8==0) begin
                render_case(frame_index,0,2,voice_for(frame_index),context_for(frame_index),3,0);
                render_case(frame_index,0,2,16'hffff,context_for(frame_index),modes[frame_index],offsets[frame_index]);
                render_case(frame_index,0,2,voice_for(frame_index),context_for(frame_index),2,65);
            end
            // Every opaque word survives held rendering, including bytes outside coefficients.
            acquire_page(frame_index%16,voice_for(frame_index),context_for(frame_index),1);
            if(block_ready) $fatal(1,"Renderer admitted while external state lease held");
            for(j=0;j<128;j=j+1) access_word(j,0,frame_index%256,0,0);
            release_page;
            if(dirty_mask || dma_busy) $fatal(1,"Held render advanced state");
            if(frame_index%4==0) begin
                write_coefficients(frame_index);open_window;start_count=completions;
                if(frame_index==0) fault_word=17;
                release_page;
                wait_completion(start_count,frame_index%16,voice_for(frame_index),context_for(frame_index),frame_index==0 ? 4 : 0);
                close_window(frame_index==0 ? 4 : 0);
                if(frame_index==0) begin
                    render_case(frame_index,0,1,voice_for(frame_index),context_for(frame_index),modes[frame_index],offsets[frame_index]);
                    open_window;start_count=completions;retry_page(frame_index%16);
                    wait_completion(start_count,frame_index%16,voice_for(frame_index),context_for(frame_index),0);close_window(0);
                end
                for(j=0;j<128;j=j+1) if(ram[(frame_index%256)*128+j]!==updated_pages[frame_index*128+j]) $fatal(1,"State write-through lost bytes");
                slot_version[frame_index%16]=1;
                render_case(frame_index,1,0,voice_for(frame_index),context_for(frame_index),modes[frame_index],offsets[frame_index]);
            end
        end
        // Admission reserves the old page before the registered cache lookup.
        i=CASES-1;
        for(j=0;j<128;j=j+1) ram[255*128+j]=source_pages[i*128+j];
        open_window;start_count=completions;collide_prefetch=1;
        render_case(i,slot_version[i%16],0,voice_for(i),context_for(i),modes[i],offsets[i]);
        wait_completion(start_count,i%16,16'heeee,32'habcdef01,0);close_window(0);
        render_case(i,0,1,voice_for(i),context_for(i),modes[i],offsets[i]);
        render_case(i,0,0,16'heeee,32'habcdef01,modes[i],offsets[i]);
        open_window;start_count=completions;start_fetch(i%16,voice_for(i),context_for(i),i%256);
        wait_completion(start_count,i%16,voice_for(i),context_for(i),0);close_window(0);
        // Reset during the pending lookup, accepted decode, and stalled output.
        i=CASES-1;
        for(reset_stage=0;reset_stage<3;reset_stage=reset_stage+1) begin
            hold_audio=1;begin_render(i,voice_for(i),context_for(i),modes[i],offsets[i],77);
            if(reset_stage==0) begin
                // The second command clock captures the cache query; its
                // result is due at the third. Reset before publication.
                tick;
                if(state_acquire_ready || state_lease_valid || audio_valid)
                    $fatal(1,"Pending renderer query did not retain ownership");
            end
            else if(reset_stage==1) repeat(10) tick;
            else begin waits=0;while(!audio_valid) begin tick;waits=waits+1;if(waits>2000) $fatal(1,"Reset output timeout");end end
            reset_all;
            if(audio_valid || state_lease_valid) $fatal(1,"Reset leaked audio or lease");
            render_case(i,0,1,voice_for(i),context_for(i),modes[i],offsets[i]);
            open_window;start_count=completions;start_fetch(i%16,voice_for(i),context_for(i),i%256);
            wait_completion(start_count,i%16,voice_for(i),context_for(i),0);close_window(0);
            render_case(i,slot_version[i%16],0,voice_for(i),context_for(i),modes[i],offsets[i]);
        end
        if(voice_conflicts!=1 || prefetch_conflicts!=1) $fatal(1,"Missing lease arbitration coverage");
        $display("PASS voice_stream cases=%0d headroom=%0d stalls=%0d latency=%0d frames=%0d samples=%0d held=%0d updated=%0d silent=%0d K2=%0d..%0d K4=%0d..%0d K8=%0d..%0d windows=%0d max_window=%0d resets=%0d state_reads=%0d state_writes=%0d conflicts=%0d prefetch_conflicts=%0d",CASES,HEADROOM_BITS,STALLS,LATENCY,frames,samples,held_frames,updated_frames,silent_frames,min_cycles[0],max_cycles[0],min_cycles[1],max_cycles[1],min_cycles[2],max_cycles[2],windows,max_window,resets,reads,writes,voice_conflicts,prefetch_conflicts);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"Global timeout");end
endmodule
