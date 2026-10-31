`timescale 1ns/1ps
module tb_dma_priority;
    parameter JOBS = 1024, STALLS = 1, FILL = 1, MAX_CYCLES = 40000;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, submit_valid = 0, completion_ready = 0;
    reg [63:0] submit_descriptor = 0;
    wire submit_ready, initialized, completion_valid, completion_interrupt, deadline_missed;
    wire [8:0] queued_count;
    wire [63:0] completion_descriptor;
    wire [3:0] completion_status;
    wire [31:0] completion_cycles;
    wire ram_request_valid, ram_request_write, ram_response_ready;
    wire [31:0] ram_request_address;
    wire [127:0] ram_request_data;
    reg ram_request_ready = 0, ram_response_valid = 0, ram_response_error = 0;
    reg [127:0] ram_response_data = 0;
    wire bram_enable, bram_write;
    wire [15:0] bram_address;
    wire [127:0] bram_write_data;
    reg [127:0] bram_read_data;
    dma_priority #(.MAX_CYCLES(MAX_CYCLES)) dut (.*);

    reg [63:0] jobs [0:JOBS-1];
    reg [3:0] statuses [0:JOBS-1];
    reg waiting [0:JOBS-1];
    reg [127:0] memory [0:4095], expected_write [0:254];
    reg [4095:0] jobs_file, statuses_file;
    integer i, n, target, probe, accepted, started, finished, active;
    integer cycles, chosen, request_count, response_count, read_count, direction, length;
    integer dispatch_cycle, done_cycle, full_cycle, fullness = 0, total_bytes = 0;
    integer expected_status;
    integer max_wait = 0, min_wait = 2147483647, admitted_at [0:JOBS-1];
    integer selection_cycle;
    reg pending_response, seen_done, released, full_checked, pending_submit;
    reg [63:0] active_descriptor;

    function [127:0] ram_word;
        input [31:0] address;
        begin
            ram_word = {address ^ 32'hd34db33f,~address,address+32'h12345678,address^32'h9e3779b9};
        end
    endfunction

    always @(posedge clk) begin
        if (bram_enable) begin
            if (bram_write) memory[bram_address >> 4] <= bram_write_data;
            else bram_read_data <= memory[bram_address >> 4];
        end
    end

    task reset_device;
        begin
            #1;
            rst_n = 0; submit_valid = 0; completion_ready = 0;
            ram_request_ready = 0; ram_response_valid = 0;
            repeat (2) begin
                @(posedge clk); #1;
                if (initialized || submit_ready || completion_valid || ram_request_valid || bram_enable)
                    $fatal(1, "Priority DMA traffic during reset");
            end
            // RAM adapter responses are flushed with the DUT.
            @(negedge clk); rst_n = 1;
            for (n = 1; n <= 256; n = n+1) begin
                @(posedge clk); #1;
                if (queued_count != 0 || completion_valid || ram_request_valid || bram_enable)
                    $fatal(1, "Old queue work survived reset");
                if (initialized !== (n == 256) || submit_ready !== (n == 256))
                    $fatal(1, "Wrong queue initialization interval");
            end
        end
    endtask

    task submit_probe;
        input [3:0] priority_value;
        begin
            @(negedge clk);
            submit_descriptor = {32'h1000,16'h8000,8'd1,priority_value,2'b00,2'b11};
            submit_valid = 1;
            @(posedge clk);
            while (!submit_ready) @(posedge clk);
            @(negedge clk); submit_valid = 0;
        end
    endtask

    task reset_probes;
        begin
            // Interrupted free-list initialization at three positions.
            for (probe = 0; probe < 3; probe = probe+1) begin
                @(negedge clk); rst_n = 0;
                @(negedge clk); rst_n = 1;
                repeat ((probe == 0) ? 1 : ((probe == 1) ? 127 : 255)) @(posedge clk);
                reset_device;
            end
            // Partial descriptor storage, publication, both reads and issue.
            for (target = 2; target <= 9; target = target+1) begin
                submit_probe(target);
                while (dut.state != target) begin @(posedge clk); #1; end
                reset_device;
            end
            submit_probe(3);
            while (!ram_request_valid) begin @(posedge clk); #1; end
            repeat (3) @(posedge clk);
            reset_device;

            submit_probe(4);
            while (!ram_request_valid) begin @(posedge clk); #1; end
            @(negedge clk); ram_request_ready = 1;
            @(posedge clk);
            @(negedge clk); ram_response_valid = 1; ram_response_data = ram_word(32'h1000);
            @(posedge clk); #1;
            if (!completion_valid) $fatal(1, "Missed queued completion reset probe");
            @(negedge clk); ram_response_valid = 0;
            repeat (3) @(posedge clk);
            reset_device;

            // One active request and all 256 queue slots occupied.
            for (probe = 0; probe < 257; probe = probe+1) submit_probe(probe[3:0]);
            while (queued_count != 256) begin @(posedge clk); #1; end
            if (submit_ready) $fatal(1, "Full queue accepted another descriptor");
            reset_device;
        end
    endtask

    initial begin
        if (!$value$plusargs("jobs=%s", jobs_file) || !$value$plusargs("statuses=%s", statuses_file))
            $fatal(1, "Missing priority DMA vectors");
        $readmemh(jobs_file, jobs);
        $readmemh(statuses_file, statuses);
        reset_device;
        reset_probes;
        for (i = 0; i < JOBS; i = i+1) waiting[i] = 0;
        for (i = 0; i < 4096; i = i+1) memory[i] = ram_word(16*i);
        accepted = 0; started = 0; finished = 0; active = -1; cycles = 0;
        request_count = 0; response_count = 0; pending_response = 0;
        seen_done = 0; released = !FILL; full_checked = !FILL; full_cycle = -1;
        pending_submit = 0;
        while (finished < JOBS) begin
            @(negedge clk);
            if (!pending_submit && accepted < JOBS && (FILL || !STALLS || cycles%7 == 0)) pending_submit = 1;
            submit_valid = pending_submit;
            submit_descriptor = accepted < JOBS ? jobs[accepted] : 0;
            if (FILL && queued_count == 256 && full_cycle == -1) full_cycle = cycles;
            if (full_cycle >= 0 && cycles-full_cycle >= 16) begin released = 1; full_checked = 1; end
            ram_request_ready = !STALLS || cycles%7 < 5;
            if (released && !pending_response && active >= 0 && response_count < request_count
                    && (!STALLS || cycles%5 != 1)) pending_response = 1;
            ram_response_valid = pending_response;
            ram_response_error = 0;
            ram_response_data = ram_word(active_descriptor[63:32] + 16*response_count);
            completion_ready = seen_done && (!STALLS || cycles-done_cycle >= 9);
            @(posedge clk);
            cycles = cycles+1;
            if (submit_valid && submit_ready) begin
                if (FILL && !released && accepted >= 2 && accepted < 257
                        && cycles-admitted_at[accepted-1] != 4)
                    $fatal(1, "Wrong enqueue throughput while engine blocked");
                waiting[accepted] = 1; admitted_at[accepted] = cycles;
                accepted = accepted+1; pending_submit = 0;
            end
            if (queued_count > fullness) fullness = queued_count;
            if (queued_count == 256 && submit_ready) $fatal(1, "Queue overflow admission");
            if (dut.dispatch) selection_cycle = cycles;
            if (dut.issue && dut.engine_ready) begin
                if (active >= 0) $fatal(1, "Started a second DMA before completion consumption");
                if (cycles-selection_cycle != 5) $fatal(1, "Wrong dispatch latency");
                chosen = -1;
                // Independent arrival-order scan: first maximum priority wins.
                for (i = 0; i < accepted; i = i+1)
                    if (waiting[i] && (chosen < 0 || jobs[i][7:4] > jobs[chosen][7:4])) chosen = i;
                if (chosen < 0 || dut.outgoing !== jobs[chosen])
                    $fatal(1, "Priority/FIFO or descriptor-data mismatch at dispatch %0d", started);
                waiting[chosen] = 0; active = chosen; active_descriptor = jobs[chosen];
                if (cycles-admitted_at[chosen] > max_wait) max_wait = cycles-admitted_at[chosen];
                if (cycles-admitted_at[chosen] < min_wait) min_wait = cycles-admitted_at[chosen];
                if (started == 0 && cycles-admitted_at[chosen] != 9)
                    $fatal(1, "Wrong empty-queue admission latency");
                started = started+1; request_count = 0; response_count = 0; read_count = 0;
                dispatch_cycle = cycles; seen_done = 0;
                length = active_descriptor[15:8]; direction = active_descriptor[3:2];
                expected_status = statuses[chosen];
            end
            if (bram_enable) begin
                if (active < 0 || expected_status != 0 || bram_write !== (direction == 0) ||
                        bram_address !== active_descriptor[31:16] + 16*read_count)
                    $fatal(1, "Queued BRAM address/direction mismatch");
                if (bram_write && bram_write_data !== ram_word(active_descriptor[63:32] + 16*read_count))
                    $fatal(1, "Queued read payload mismatch");
                if (!bram_write) expected_write[read_count] = memory[bram_address >> 4];
                read_count = read_count+1;
            end
            if (ram_request_valid && ram_request_ready) begin
                if (active < 0 || expected_status != 0 || request_count >= length || ram_request_write !== direction[0]
                        || ram_request_address !== active_descriptor[63:32] + 16*request_count)
                    $fatal(1, "Queued RAM request mismatch");
                if (ram_request_write && ram_request_data !== expected_write[request_count])
                    $fatal(1, "Queued write payload mismatch");
                request_count = request_count+1;
            end
            if (ram_response_valid && ram_response_ready) begin
                response_count = response_count+1; pending_response = 0;
            end
            if (completion_valid) begin
                if (active < 0 || !seen_done || completion_descriptor !== active_descriptor ||
                        completion_cycles !== done_cycle-dispatch_cycle || completion_interrupt !== active_descriptor[1]
                        || completion_status !== (expected_status | ((done_cycle-dispatch_cycle > MAX_CYCLES) ? 8 : 0)))
                    $fatal(1, "Queued completion mismatch");
                if (completion_ready) begin
                    if (read_count != (expected_status == 0 ? length : 0)) $fatal(1, "Wrong queued BRAM beats");
                    if (expected_status == 0) total_bytes = total_bytes + 16*length;
                    active = -1; finished = finished+1; seen_done = 0;
                end
            end
            #1;
            if (completion_valid && !seen_done) begin
                if (request_count != (expected_status == 0 ? length : 0)
                        || response_count != (expected_status == 0 ? length : 0)) $fatal(1, "Premature queued completion");
                done_cycle = cycles; seen_done = 1;
            end
            if (cycles > JOBS*2000+10000) $fatal(1, "Priority DMA deadlock");
        end
        if (accepted != JOBS || started != JOBS || queued_count != 0 || !full_checked)
            $fatal(1, "Lost/duplicate descriptor or missing full-queue coverage");
        for (i = 0; i < JOBS; i = i+1) if (waiting[i]) $fatal(1, "Undelivered queued descriptor");
        $display("PASS priority jobs=%0d bytes=%0d stalls=%0d fill=%0d max_queue=%0d min_wait=%0d max_wait=%0d total_cycles=%0d enqueue_spacing=4 dispatch_delay=5",
                 JOBS,total_bytes,STALLS,FILL,fullness,min_wait,max_wait,cycles);
        $finish;
    end
    initial begin
        #(JOBS*30000 + 10000000);
        $fatal(1, "Global priority DMA timeout");
    end
endmodule
