`timescale 1ns/1ps
module tb_dma_window;
    parameter WINDOW_CYCLES=40000;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0,window_valid=0,seal=0,result_ready=0,submit_valid=0;
    reg [31:0] window_tag=0;
    reg [63:0] submit_descriptor=0;
    wire window_ready,window_busy,window_accepting,window_final_cycle,result_valid,deadline_missed,submit_ready,initialized;
    wire [31:0] result_tag,elapsed_cycles,accepted_count,completed_count,rejected_count;
    wire [3:0] result_status;
    wire [8:0] outstanding_count;
    wire done_valid;
    wire [63:0] done_descriptor;
    wire [3:0] done_status;
    wire [31:0] done_service_cycles;
    wire ram_request_valid,ram_request_write,ram_response_ready;
    wire [31:0] ram_request_address;
    wire [127:0] ram_request_data;
    reg ram_request_ready=1,ram_response_valid=0,ram_response_error=0;
    reg [127:0] ram_response_data=0;
    wire bram_enable,bram_write;
    wire [15:0] bram_address;
    wire [127:0] bram_write_data;
    reg [127:0] bram_read_data;
    dma_window #(.WINDOW_CYCLES(WINDOW_CYCLES)) dut (.*);

    reg [63:0] descriptors [0:299];
    reg live [0:299],queued [0:299];
    integer release_at [0:299],done_at [0:299],service_cycles [0:299];
    reg [127:0] memory [0:4095];
    reg [31:0] requests [0:4095];
    integer cycle=0,start_cycle=0,wr=0,rd=0,job,n,probe,target,waits;
    integer callbacks=0,queue_callbacks=0,dropped=0,windows=0,resets=0,request_total=0;
    integer prior_requests,prior_callbacks,saved_elapsed,saved_accepted,saved_completed,saved_rejected;
    reg [3:0] saved_status;
    reg hold_response=0;
    reg pair_check=0;

    function [127:0] ram_word;
        input [31:0] address;
        begin ram_word={address^32'hd34db33f,~address,address+32'h12345678,address^32'h9e3779b9};end
    endfunction
    always @(negedge clk) begin
        if (!rst_n) ram_response_valid=0;
        else if (!ram_response_valid && !hold_response && rd<wr) begin
            job=(requests[rd%4096]-32'h100000)/4096;
            if (cycle>=release_at[job]) begin
                ram_response_valid=1;ram_response_data=ram_word(requests[rd%4096]);ram_response_error=0;
            end
        end
    end
    always @(posedge clk) begin
        cycle<=cycle+1;
        if (!rst_n) begin wr=0;rd=0;end
        else begin
            if (window_valid && window_ready) begin start_cycle=cycle;windows=windows+1;end
            if (submit_valid && submit_ready) begin
                job=(submit_descriptor[63:32]-32'h100000)/4096;
                if (live[job]) $fatal(1,"Reused an unfinished test command");
                live[job]=1;queued[job]=window_accepting;done_at[job]=-1;
            end
            if (dut.queue.submit_valid && !dut.queue.submit_ready) $fatal(1,"Presented withdrawable valid to the queue");
            if (ram_request_valid && ram_request_ready) begin
                requests[wr%4096]=ram_request_address;wr=wr+1;request_total=request_total+1;
                job=(ram_request_address-32'h100000)/4096;
                if (!live[job] || !queued[job]) $fatal(1,"Rejected/unowned command issued RAM");
                if (ram_request_write && ram_request_data!==memory[descriptors[job][31:16]>>4]) $fatal(1,"Write payload changed");
            end
            if (ram_response_valid && ram_response_ready) begin rd=rd+1;ram_response_valid<=0;end
            if (bram_enable) begin
                if (bram_write) memory[bram_address>>4]<=bram_write_data;
                else bram_read_data<=memory[bram_address>>4];
            end
            if (done_valid) begin
                job=(done_descriptor[63:32]-32'h100000)/4096;
                if (!live[job] || done_descriptor!==descriptors[job]) $fatal(1,"Lost/duplicate/altered completion");
                if (done_status!==(queued[job] && cycle-start_cycle<=WINDOW_CYCLES ? 4'd0 : 4'd8))
                    $fatal(1,"Wrong aggregate deadline status job=%0d time=%0d status=%0d",job,cycle-start_cycle,done_status);
                if (pair_check && done_service_cycles>=WINDOW_CYCLES) $fatal(1,"Pair did not isolate cumulative wait from individual service");
                if (!queued[job] && done_service_cycles!=0) $fatal(1,"Rejected command claimed service");
                done_at[job]=cycle-start_cycle;service_cycles[job]=done_service_cycles;
                live[job]=0;callbacks=callbacks+1;
                if (queued[job]) queue_callbacks=queue_callbacks+1;else dropped=dropped+1;
            end
        end
    end
    task tick;begin @(posedge clk);#1;end endtask
    task reset_all;
        begin
            @(negedge clk);rst_n=0;window_valid=0;seal=0;submit_valid=0;result_ready=0;hold_response=0;pair_check=0;
            for (n=0;n<300;n=n+1) begin live[n]=0;release_at[n]=0;end
            repeat (3) tick;
            if (done_valid || window_busy || window_ready || result_valid || submit_ready) $fatal(1,"Window reset leaked work");
            @(negedge clk);rst_n=1;
            repeat (256) tick;
            if (!initialized || !window_ready || elapsed_cycles!=0 || outstanding_count!=0) $fatal(1,"Window reset incomplete");
            resets=resets+1;
        end
    endtask
    task begin_window;
        input [31:0] tag_value;
        begin
            @(negedge clk);window_valid=1;window_tag=tag_value;
            @(posedge clk);waits=0;
            while (!window_ready) begin @(posedge clk);waits=waits+1;if (waits>100) $fatal(1,"Window could not start");end
            @(negedge clk);window_valid=0;
        end
    endtask
    task submit;
        input integer index;
        begin
            @(negedge clk);submit_valid=1;submit_descriptor=descriptors[index];
            @(posedge clk);waits=0;
            while (!submit_ready) begin
                @(posedge clk);waits=waits+1;
                if (waits>WINDOW_CYCLES+100) $fatal(1,"Input did not retire at expiry");
            end
            @(negedge clk);submit_valid=0;
        end
    endtask
    task seal_window;
        begin @(negedge clk);seal=1;tick;@(negedge clk);seal=0;end
    endtask
    task wait_result;
        begin
            waits=0;
            while (!result_valid) begin tick;waits=waits+1;if (waits>WINDOW_CYCLES+10000) $fatal(1,"Window did not drain");end
            if (outstanding_count!=0 || window_busy) $fatal(1,"Reported before drain");
        end
    endtask
    task consume_result;
        begin @(negedge clk);result_ready=1;tick;@(negedge clk);result_ready=0;end
    endtask

    initial begin
        for (n=0;n<300;n=n+1) begin
            descriptors[n]={32'(32'h100000+4096*n),16'(16'h8000+(n%4)*4096),8'd1,4'(n%16),2'(n%2),2'b11};
            live[n]=0;queued[n]=0;release_at[n]=0;done_at[n]=-1;service_cycles[n]=0;
        end
        for (n=0;n<4096;n=n+1) memory[n]=ram_word(16*n);
        reset_all;
        // A descriptor outside any window retires without consuming RAM.
        prior_requests=request_total;submit(0);repeat (3) tick;
        if (live[0] || request_total!=prior_requests || result_valid) $fatal(1,"Closed-window rejection failed");
        begin_window(1);seal_window;wait_result;
        if (result_status || accepted_count || completed_count) $fatal(1,"Empty seal failed");
        consume_result;

        // Sealing an empty batch while rejecting a descriptor must wait for
        // its registered callback before freezing the final status/report.
        begin_window(2);prior_requests=request_total;
        seal=1;submit_valid=1;submit_descriptor=descriptors[6];
        @(posedge clk);if (!submit_ready || window_accepting) $fatal(1,"Sealed rejection handshake failed");
        @(negedge clk);seal=0;submit_valid=0;
        if (result_valid) $fatal(1,"Result preceded rejection capture");
        tick;if (result_valid || !dut.callback_valid) $fatal(1,"Result preceded callback delivery");
        wait_result;
        if (elapsed_cycles!=3 || result_status!=8 || accepted_count || completed_count || rejected_count!=1 || live[6] || request_total!=prior_requests)
            $fatal(1,"Registered rejection escaped its batch");
        consume_result;

        if (WINDOW_CYCLES>=16) begin
            // Completion eligibility is measured at owner callback, including
            // queue dispatch and the report edge, with exact boundary checks.
            for (probe=0;probe<4;probe=probe+1) begin
                begin_window(10+probe);target=WINDOW_CYCLES-1+probe;
                // Account for the registered callback by returning RAM one
                // clock earlier. The asserted wall-clock deadline is unchanged.
                release_at[probe]=start_cycle+target-2;
                // Admit at T=1 so the 16-clock write probe can reach its
                // original exact deadline despite the callback register.
                submit_valid=1;submit_descriptor=descriptors[probe];
                @(posedge clk);if (!submit_ready) $fatal(1,"Boundary probe was not admitted");
                @(negedge clk);submit_valid=0;seal_window;wait_result;
                if (done_at[probe]!=target || elapsed_cycles!=target || accepted_count!=1 || completed_count!=1 || rejected_count!=0)
                    $fatal(1,"Wrong exact boundary accounting probe=%0d event=%0d elapsed=%0d",probe,done_at[probe],elapsed_cycles);
                if (result_status!==(target>WINDOW_CYCLES ? 4'd8 : 4'd0)) $fatal(1,"Wrong exact deadline result");
                consume_result;
            end
        end else begin
            // Even the smallest legal window cannot be restarted to hide work.
            begin_window(20);
            submit_valid=1;submit_descriptor=descriptors[2];
            @(posedge clk);if (!submit_ready || !window_accepting) $fatal(1,"Tiny-window first clock unavailable");
            @(negedge clk);submit_valid=0;
            wait_result;
            if (result_status!=8 || accepted_count!=1 || completed_count!=1) $fatal(1,"Tiny-window expiry failed");
            consume_result;
        end
        if (WINDOW_CYCLES>=32) begin
            begin_window(30);pair_check=1;
            release_at[0]=start_cycle+WINDOW_CYCLES/2-2;
            release_at[1]=start_cycle+WINDOW_CYCLES+1;
            submit(0);submit(1);seal_window;wait_result;
            if (done_at[0]!=WINDOW_CYCLES/2 || done_at[1]!=WINDOW_CYCLES+3 || result_status!=8 || accepted_count!=2 || completed_count!=2)
                $fatal(1,"Independent transfer budgets hid cumulative deadline failure");
            pair_check=0;consume_result;
        end
        if (WINDOW_CYCLES>=2048) begin
            begin_window(40);hold_response=1;
            for (probe=0;probe<257;probe=probe+1) submit(probe);
            while (outstanding_count!=257) tick;
            // This unaccepted descriptor must be rejected at expiry without
            // withdrawing any stalled queue-valid or resetting the timer.
            @(negedge clk);window_valid=1;window_tag=32'hbadbad;
            submit(257);repeat (3) tick;
            if (!deadline_missed || outstanding_count!=257 || window_ready || result_tag!=40 || live[257])
                $fatal(1,"Full queue did not retire pending input safely");
            if (request_total==0) $fatal(1,"No active transfer in full-queue test");
            hold_response=0;wait_result;
            if (accepted_count!=257 || completed_count!=257 || rejected_count!=1 || result_status!=8)
                $fatal(1,"Full queue batch accounting");
            @(negedge clk);window_valid=0;
            saved_elapsed=elapsed_cycles;saved_accepted=accepted_count;saved_completed=completed_count;saved_rejected=rejected_count;saved_status=result_status;
            prior_requests=request_total;
            submit(258);repeat (4) tick;
            if (!result_valid || window_ready || request_total!=prior_requests || elapsed_cycles!=saved_elapsed || accepted_count!=saved_accepted ||
                completed_count!=saved_completed || rejected_count!=saved_rejected || result_status!=saved_status)
                $fatal(1,"Result backpressure changed the completed window");
            consume_result;
        end
        // Reset at active/queued/rejected/result boundaries; adapter flush is mandatory.
        for (probe=0;probe<5;probe=probe+1) begin
            if (probe<2) begin begin_window(50+probe);hold_response=1;submit(probe);if (probe==1) submit(3);end
            if (probe==2) begin submit(4);rst_n=0;end
            if (probe==3) begin begin_window(53);seal_window;wait_result;end
            if (probe==4) begin
                submit(5);tick;
                if (!dut.callback_valid) $fatal(1,"Reset probe missed pending callback");
            end
            reset_all;
        end
        begin_window(60);seal_window;wait_result;
        if (result_status || accepted_count || completed_count || rejected_count) $fatal(1,"Old window work survived reset");
        $display("PASS window limit=%0d windows=%0d callbacks=%0d queued=%0d rejected=%0d requests=%0d resets=%0d",WINDOW_CYCLES,windows,callbacks,queue_callbacks,dropped,request_total,resets);
        $finish;
    end
    initial begin #20000000;$fatal(1,"DMA window test timed out");end
endmodule
