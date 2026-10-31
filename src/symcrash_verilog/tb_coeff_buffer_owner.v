`timescale 1ns/1ps
module tb_coeff_buffer_owner;
    parameter K=4, STALLS=1;
    localparam BYTES=64*K*2, BEATS=BYTES/16;
    wire [1:0] dma_done_matches=2'b00;
    wire [127:0] dma_match_descriptors;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0, prefetch_valid=0, acquire_valid=0, release_valid=0, release_bank=0, cancel_valid=0;
    reg [31:0] prefetch_tag=0, prefetch_ram_address=0, acquire_tag=0, cancel_tag=0;
    reg [1:0] prefetch_class=0;
    wire prefetch_ready, prefetch_bank, acquire_ready, acquire_result_valid, acquire_hit, acquire_bank;
    wire [15:0] acquire_address;
    wire [1:0] free_mask, ready_mask, reading_mask;
    wire fault_valid, protocol_error;
    wire [31:0] fault_tag;
    wire [3:0] fault_status;
    wire dma_submit_valid, dma_submit_ready;
    wire [63:0] dma_submit_descriptor;
    wire dma_done_valid;
    wire [63:0] dma_done_descriptor;
    wire [3:0] dma_done_status;
    coeff_buffer_owner #(.K(K)) owner (.*);

    reg hold_submit=0, hold_response=0, hold_completion=0, extra_done=0;
    reg [31:0] fail_address=32'hffffffff;
    wire submit_ready, initialized, completion_valid, completion_interrupt, deadline_missed;
    wire [8:0] queued_count;
    wire [63:0] completion_descriptor;
    wire [3:0] completion_status;
    wire [31:0] completion_cycles;
    wire completion_ready = !hold_completion;
    assign dma_submit_ready = submit_ready && !hold_submit;
    assign dma_done_valid = (completion_valid && completion_ready) || extra_done;
    assign dma_done_descriptor = extra_done ? 64'hfeedbaad00000101 : completion_descriptor;
    assign dma_done_status = extra_done ? 4'd4 : completion_status;
    wire ram_request_valid, ram_request_write, ram_response_ready;
    wire [31:0] ram_request_address;
    wire [127:0] ram_request_data;
    reg ram_request_ready=0, ram_response_valid=0, ram_response_error=0;
    reg [127:0] ram_response_data=0;
    wire bram_enable, bram_write;
    wire [15:0] bram_address;
    wire [127:0] bram_write_data;
    wire [127:0] bram_read_data=0;
    dma_priority queue (
        .clk(clk), .rst_n(rst_n), .submit_valid(dma_submit_valid && !hold_submit),
        .submit_ready(submit_ready), .submit_descriptor(dma_submit_descriptor),
        .initialized(initialized), .queued_count(queued_count),
        .completion_valid(completion_valid), .completion_ready(completion_ready),
        .completion_descriptor(completion_descriptor), .completion_status(completion_status),
        .completion_cycles(completion_cycles), .completion_interrupt(completion_interrupt),
        .deadline_missed(deadline_missed), .ram_request_valid(ram_request_valid),
        .ram_request_ready(ram_request_ready), .ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address), .ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid), .ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error), .ram_response_data(ram_response_data),
        .bram_enable(bram_enable), .bram_write(bram_write), .bram_address(bram_address),
        .bram_write_data(bram_write_data), .bram_read_data(bram_read_data)
    );

    reg [127:0] memory [0:4095];
    reg [31:0] responses [0:4095];
    reg [31:0] expected_source [0:1], expected_tag [0:1];
    reg [63:0] expected_descriptor [0:1];
    reg [63:0] stalled_descriptor;
    reg stalled=0;
    reg dsp_enable=0;
    reg [15:0] dsp_address=0;
    reg [127:0] dsp_data;
    integer wr=0, rd=0, clock_count=0, admitted=0, completed=0, hits=0, misses=0;
    integer writes=0, read_beats=0, faults=0, resets=0, last_fault_status=0;
    integer max_service=0, wait_count, b, j, frame, probe, before_faults, before_writes;
    reg [31:0] last_fault_tag;
    reg [31:0] current_tag;
    reg leased;
    reg lease_bank;

    function [127:0] ram_word;
        input [31:0] address;
        begin ram_word={address^32'hd34db33f,~address,address+32'h12345678,address^32'h9e3779b9}; end
    endfunction
    function [31:0] frame_source;
        input [31:0] frame_tag;
        begin frame_source=32'h100000 + (frame_tag & 32'h1ff)*4096; end
    endfunction

    // Ordered adapter model. Requests and responses have independent stalls.
    always @(negedge clk) begin
        ram_request_ready=rst_n && (!STALLS || clock_count%7<5);
        if (!rst_n) ram_response_valid=0;
        else if (!ram_response_valid && !hold_response && rd<wr && (!STALLS || clock_count%5!=1)) begin
            ram_response_valid=1;
            ram_response_data=ram_word(responses[rd%4096]);
            ram_response_error=responses[rd%4096]==fail_address;
        end
    end
    always @(posedge clk) begin
        clock_count=clock_count+1;
        if (!rst_n) begin wr=0; rd=0; stalled=0; end
        else begin
            if (stalled && (!dma_submit_valid || dma_submit_descriptor !== stalled_descriptor))
                $fatal(1,"Canceled or changed a backpressured descriptor");
            stalled=dma_submit_valid && !dma_submit_ready;
            stalled_descriptor=dma_submit_descriptor;
            if (prefetch_valid && prefetch_ready) begin
                b=prefetch_bank;
                if (!free_mask[b]) $fatal(1,"Reserved an occupied bank");
                expected_source[b]=prefetch_ram_address; expected_tag[b]=prefetch_tag;
                expected_descriptor[b]={prefetch_ram_address,16'(16'h8000+b*BYTES),8'(BEATS),4'(15-prefetch_class),4'b0011};
                admitted=admitted+1;
            end
            if (dma_submit_valid && dma_submit_ready) begin
                b=(dma_submit_descriptor[31:16]-16'h8000)/BYTES;
                if (b<0 || b>1 || dma_submit_descriptor !== expected_descriptor[b])
                    $fatal(1,"DMA descriptor changed frame size, class, source or destination");
            end
            if (ram_request_valid && ram_request_ready) begin
                if (ram_request_write || wr-rd>=4096) $fatal(1,"Invalid adapter request");
                responses[wr%4096]=ram_request_address; wr=wr+1;
            end
            if (ram_response_valid && ram_response_ready) begin rd=rd+1; ram_response_valid<=0; end
            if (bram_enable) begin
                if (!bram_write || bram_address<16'h8000 || bram_address>=16'h8000+2*BYTES)
                    $fatal(1,"DMA escaped reserved coefficient buffers");
                b=(bram_address-16'h8000)/BYTES;
                if (reading_mask[b] || ready_mask[b] || free_mask[b]) $fatal(1,"DMA wrote an unowned or published bank");
                memory[bram_address>>4] <= bram_write_data;
                writes=writes+1;
            end
            if (completion_valid && completion_ready) begin
                completed=completed+1;
                if (completion_cycles>max_service) max_service=completion_cycles;
            end
            if (fault_valid) begin faults=faults+1; last_fault_status=fault_status; last_fault_tag=fault_tag; end
        end
        if (dsp_enable) begin
            b=(dsp_address-16'h8000)/BYTES;
            if (!reading_mask[b]) $fatal(1,"DSP read without a lease");
            dsp_data <= memory[dsp_address>>4];
            read_beats=read_beats+1;
        end
    end

    task tick; begin @(posedge clk); #1; end endtask
    task reset_all;
        begin
            @(negedge clk); rst_n=0; prefetch_valid=0; acquire_valid=0; release_valid=0; cancel_valid=0;
            dsp_enable=0; hold_submit=0; hold_response=0; hold_completion=0; extra_done=0; fail_address=32'hffffffff;
            repeat (3) tick;
            if (dma_submit_valid || acquire_ready || prefetch_ready || ready_mask || reading_mask)
                $fatal(1,"Valid ownership survived reset");
            @(negedge clk); rst_n=1;
            repeat (256) tick;
            if (!initialized || free_mask!=3 || protocol_error) $fatal(1,"Reset failed");
            resets=resets+1; leased=0;
        end
    endtask
    task prefetch;
        input [31:0] wanted_tag, source;
        input [1:0] service_class;
        begin
            @(negedge clk); prefetch_valid=1; prefetch_tag=wanted_tag;
            prefetch_ram_address=source; prefetch_class=service_class;
            @(posedge clk);
            wait_count=0;
            while (!prefetch_ready) begin
                wait_count=wait_count+1;
                if (wait_count>1000) $fatal(1,"Prefetch admission stuck tag=%h free=%b",wanted_tag,free_mask);
                @(posedge clk);
            end
            @(negedge clk); prefetch_valid=0;
        end
    endtask
    task wait_ready;
        input [1:0] mask;
        begin
            wait_count=0;
            while ((ready_mask & mask)!=mask) begin
                tick; wait_count=wait_count+1;
                if (wait_count>2000) $fatal(1,"Publication stuck wanted=%b actual=%b",mask,ready_mask);
            end
        end
    endtask
    task wait_free;
        input [1:0] mask;
        begin
            wait_count=0;
            while ((free_mask & mask)!=mask) begin
                tick; wait_count=wait_count+1;
                if (wait_count>2000) $fatal(1,"Drain stuck wanted=%b actual=%b",mask,free_mask);
            end
        end
    endtask
    task acquire_frame;
        input [31:0] wanted_tag;
        input expected_hit;
        begin
            @(negedge clk); acquire_valid=1; acquire_tag=wanted_tag;
            if (!acquire_ready) $fatal(1,"Unexpected existing DSP lease");
            tick;
            if (!acquire_result_valid || acquire_hit !== expected_hit) $fatal(1,"Wrong acquire result tag=%h",wanted_tag);
            if (expected_hit) begin
                hits=hits+1; leased=1; lease_bank=acquire_bank;
                if (expected_tag[lease_bank] !== wanted_tag || acquire_address !== 16'h8000+lease_bank*BYTES)
                    $fatal(1,"Wrong acquired identity/address");
            end else begin
                misses=misses+1;
                if (acquire_address!=0 || reading_mask) $fatal(1,"Miss exposed a buffer");
            end
            @(negedge clk); acquire_valid=0;
            if (expected_hit) begin
                for (j=0;j<BEATS;j=j+1) begin
                    dsp_enable=1; dsp_address=acquire_address+16*j;
                    tick;
                    if (dsp_data !== ram_word(expected_source[lease_bank]+16*j)) $fatal(1,"Partial, stale or corrupt DSP frame");
                    @(negedge clk);
                end
                dsp_enable=0;
            end
        end
    endtask
    task release_frame;
        begin
            if (!leased) $fatal(1,"Test released absent lease");
            @(negedge clk); release_valid=1; release_bank=lease_bank;
            tick;
            if (!free_mask[lease_bank]) $fatal(1,"Release failed");
            @(negedge clk); release_valid=0; leased=0;
        end
    endtask
    task cancel_frame;
        input [31:0] wanted_tag;
        begin
            @(negedge clk); cancel_valid=1; cancel_tag=wanted_tag;
            tick;
            @(negedge clk); cancel_valid=0;
        end
    endtask

    initial begin
        for (j=0;j<4096;j=j+1) memory[j]=128'hbad0bad1bad2bad3bad4bad5bad6bad7;
        reset_all;
        // Duplicate tags and same-edge expiry cannot consume the other free bank.
        hold_response=1;
        prefetch(80,frame_source(80),2);
        while (dma_submit_valid) tick;
        @(negedge clk); prefetch_valid=1; prefetch_tag=80;
        repeat (4) begin tick; if (prefetch_ready || free_mask!=2) $fatal(1,"Duplicate live tag admitted"); end
        @(negedge clk); prefetch_tag=81; cancel_valid=1; cancel_tag=81;
        tick;
        if (prefetch_ready || free_mask!=2) $fatal(1,"Admitted a canceled new tag");
        @(negedge clk); cancel_valid=0; acquire_valid=1; acquire_tag=81;
        tick;
        if (prefetch_ready || free_mask!=2 || !acquire_result_valid || acquire_hit)
            $fatal(1,"Admitted a new frame at its missed deadline");
        misses=misses+1;
        @(negedge clk); prefetch_valid=0; acquire_valid=0; hold_response=0;
        wait_ready(1); acquire_frame(80,1);
        @(negedge clk); release_valid=1; release_bank=!lease_bank;
        tick;
        if (!protocol_error || reading_mask!=1 || free_mask!=2) $fatal(1,"Bad release damaged the active lease");
        @(negedge clk); release_valid=0;
        release_frame;
        reset_all;
        // Incomplete frame misses immediately; late writes keep their bank.
        hold_response=1;
        prefetch(10,frame_source(10),2);
        while (wr==0) tick;
        acquire_frame(10,0);
        if (free_mask[0]) $fatal(1,"Expired in-flight bank reused early");
        prefetch(11,frame_source(11),0);
        @(negedge clk); prefetch_valid=1; prefetch_tag=12;
        repeat (12) begin tick; if (prefetch_ready) $fatal(1,"Third bank admitted"); end
        @(negedge clk); prefetch_valid=0; hold_response=0;
        wait_free(1); wait_ready(2);
        acquire_frame(10,0);
        acquire_frame(11,1);
        cancel_frame(11); // A lease remains valid until release, even after cancellation.
        if (reading_mask!=2 || acquire_ready) $fatal(1,"Cancellation freed a DSP lease");
        prefetch(12,frame_source(12),1);
        wait_ready(1);
        @(negedge clk); acquire_valid=1; acquire_tag=12;
        repeat (3) begin tick; if (acquire_ready || acquire_result_valid) $fatal(1,"Second DSP lease admitted"); end
        @(negedge clk); acquire_valid=0;
        release_frame;
        acquire_frame(12,1); release_frame;

        // Pending submit cannot be withdrawn on cancel; test submit/cancel coincidence too.
        for (probe=0;probe<2;probe=probe+1) begin
            hold_submit=1;
            prefetch(30+probe,frame_source(30+probe),3);
            repeat (12) tick;
            if (probe==0) cancel_frame(30);
            else begin @(negedge clk); cancel_valid=1; cancel_tag=31; hold_submit=0; tick; @(negedge clk); cancel_valid=0; end
            hold_submit=0;
            wait_free(3);
            acquire_frame(30+probe,0);
        end

        // Data in RAM is insufficient: completion must be consumed and successful.
        for (probe=0;probe<3;probe=probe+1) begin
            hold_completion=1;
            prefetch(50+probe,frame_source(50+probe),probe);
            while (!completion_valid) tick;
            if (ready_mask) $fatal(1,"Published before completion handoff");
            if (probe==0) acquire_frame(50,0);
            @(negedge clk); hold_completion=0;
            if (probe==1) begin acquire_valid=1; acquire_tag=51; end
            if (probe==2) begin cancel_valid=1; cancel_tag=52; end
            tick;
            if (probe==1 && (!acquire_result_valid || acquire_hit)) $fatal(1,"Same-edge deadline published late data");
            if (probe==1) misses=misses+1;
            @(negedge clk); acquire_valid=0; cancel_valid=0;
            wait_free(3);
            if (ready_mask) $fatal(1,"Expired completion published");
        end
        prefetch(60,frame_source(60),0); wait_ready(1);
        @(negedge clk); extra_done=1;
        tick;
        if (ready_mask!=1 || fault_valid) $fatal(1,"Unrelated completion changed ownership");
        @(negedge clk); extra_done=0; acquire_valid=1; acquire_tag=60; cancel_valid=1; cancel_tag=60;
        tick;
        if (!acquire_result_valid || acquire_hit || free_mask!=3) $fatal(1,"Cancel must win ready/acquire race");
        misses=misses+1;
        @(negedge clk); acquire_valid=0; cancel_valid=0;

        // Errors on first/middle/last beat, invalid address and per-transfer expiry.
        for (probe=0;probe<5;probe=probe+1) begin
            before_faults=faults;
            if (probe<3) fail_address=frame_source(70+probe)+16*((probe==0)?0:((probe==1)?BEATS/2:BEATS-1));
            prefetch(70+probe,frame_source(70+probe)+(probe==3 ? 1 : 0),1);
            if (probe==4) begin
                hold_response=1;
                repeat (40020) tick;
                if (!deadline_missed || free_mask[0]) $fatal(1,"Late DMA lost reservation/alarm");
                acquire_frame(74,0);
                hold_response=0;
            end
            wait_free(3); repeat (2) tick;
            if (faults!=before_faults+1 || last_fault_tag!=70+probe || last_fault_status!=(probe<3 ? 4 : (probe==3 ? 1 : 8)))
                $fatal(1,"Lost or incorrect fault tag/status probe=%0d status=%0d",probe,last_fault_status);
            acquire_frame(70+probe,0);
            fail_address=32'hffffffff;
        end

        // Reset while waiting, filling, partially filled, ready, leased,
        // discarding before/after submit, or with an unconsumed completion.
        for (probe=0;probe<8;probe=probe+1) begin
            if (probe==0 || probe==5) hold_submit=1;
            if (probe==1 || probe==6) hold_response=1;
            if (probe==7) hold_completion=1;
            before_writes=writes;
            prefetch(100+probe,frame_source(100+probe),0);
            if (probe==2) while (writes-before_writes<BEATS/2) tick;
            if (probe==3 || probe==4) wait_ready(1);
            if (probe==4) acquire_frame(104,1);
            if (probe==5 || probe==6) cancel_frame(100+probe);
            if (probe==7) while (!completion_valid) tick;
            reset_all;
            acquire_frame(100+probe,0);
        end

        // Read a full frame into the DSP before releasing its source bank.
        // Refill that bank for N+2 while N+1 is already ready; tags wrap at 2^32.
        current_tag=32'hffffff80;
        prefetch(current_tag,frame_source(current_tag),0);
        prefetch(current_tag+1,frame_source(current_tag+1),2);
        wait_ready(3);
        for (frame=0;frame<256;frame=frame+1) begin
            acquire_frame(current_tag,1);
            release_frame;
            if (frame<254) prefetch(current_tag+2,frame_source(current_tag+2),2);
            repeat (192) tick;
            current_tag=current_tag+1;
        end
        wait_free(3);
        if (protocol_error) $fatal(1,"Unexpected release protocol error");
        @(negedge clk); release_valid=1; release_bank=0;
        tick;
        if (!protocol_error || free_mask!=3) $fatal(1,"Bad release not diagnosed safely");
        @(negedge clk); release_valid=0;
        for (j=0;j<4096;j=j+1)
            if ((j<2048 || j>=(16'h8000+2*BYTES)/16) && memory[j]!==128'hbad0bad1bad2bad3bad4bad5bad6bad7)
                $fatal(1,"Unrelated memory corrupted");
        $display("PASS ownership K=%0d stalls=%0d admitted=%0d completed=%0d hits=%0d misses=%0d faults=%0d resets=%0d written_bytes=%0d read_bytes=%0d max_service=%0d",K,STALLS,admitted,completed,hits,misses,faults,resets,writes*16,read_beats*16,max_service);
        $finish;
    end
    initial begin #20000000; $fatal(1,"Ownership test timed out"); end
endmodule
