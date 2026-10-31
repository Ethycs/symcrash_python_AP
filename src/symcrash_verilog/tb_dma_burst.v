`timescale 1ns/1ps
module tb_dma_burst;
    parameter CASES = 1, MAX_CYCLES = 40000;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, descriptor_valid = 0, completion_ready = 0;
    reg [63:0] descriptor = 0;
    wire descriptor_ready, completion_valid, completion_interrupt, deadline_missed;
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
    dma_burst #(.MAX_CYCLES(MAX_CYCLES)) dut (.*);
    reg [127:0] configs [0:CASES-1], payload [0:CASES*255-1];
    reg [127:0] memory [0:4095], expected_memory [0:4095];
    reg [4095:0] config_file, payload_file;
    reg [63:0] original;
    integer case_index, length, direction, scenario, fault, expected_status;
    integer i, cycles, request_count, response_count, done_cycle, bram_count;
    integer total_bytes = 0, max_latency = 0, min_read = 2147483647, min_write = 2147483647;
    integer write_gaps, last_request_cycle, expected_bram_index;
    reg pending_response, seen_done, request_stalled;
    reg [160:0] stalled_request;
    reg [127:0] expected_word;

    always @(posedge clk) begin
        if (bram_enable) begin
            if (bram_write) memory[bram_address >> 4] <= bram_write_data;
            else bram_read_data <= memory[bram_address >> 4];
        end
    end

    task reset_device;
        begin
            @(negedge clk);
            rst_n = 0; descriptor_valid = 0; completion_ready = 0;
            ram_request_ready = 0; ram_response_valid = 0;
            repeat (2) begin
                @(posedge clk); #1;
                if (bram_enable || ram_request_valid || ram_response_ready || completion_valid || deadline_missed)
                    $fatal(1, "DMA traffic during reset");
            end
            // The RAM adapter is reset/flushed with the DMA engine.
            @(negedge clk); rst_n = 1;
            repeat (3) begin
                @(posedge clk); #1;
                if (!descriptor_ready || completion_valid || bram_enable || ram_request_valid)
                    $fatal(1, "DMA work leaked after reset");
            end
        end
    endtask

    task reset_probe;
        input integer kind;
        begin
            @(negedge clk);
            descriptor = {32'h1000,16'h8000,8'd16,4'h5,1'b0,kind[0],2'b11};
            descriptor_valid = 1;
            @(posedge clk); #1;
            @(negedge clk); descriptor_valid = 0;
            ram_request_ready = (kind < 10);
            // No responses: reset with reads in flight, pending BRAM reads,
            // a full write FIFO, or a stalled request. Fresh cases follow.
            repeat ((kind < 10) ? kind+1 : 20) @(posedge clk);
            #1;
            if (completion_valid || descriptor_ready) $fatal(1, "Reset probe missed active DMA");
            reset_device;
        end
    endtask

    task report_reset_probe;
        begin
            @(negedge clk);
            descriptor = {32'h1000,16'h8000,8'd1,4'h5,2'b00,2'b11};
            descriptor_valid = 1;
            @(posedge clk); #1;
            @(negedge clk); descriptor_valid = 0; ram_request_ready = 1;
            @(posedge clk);
            if (!ram_request_valid) $fatal(1, "Missing reset-probe request");
            @(negedge clk); ram_response_valid = 1; ram_response_data = 128'hdeadbeef;
            ram_response_error = 0;
            @(posedge clk); #1;
            if (!completion_valid) $fatal(1, "Reset probe missed completion");
            @(negedge clk); ram_response_valid = 0;
            repeat (3) @(posedge clk);
            reset_device;
        end
    endtask

    initial begin
        if (!$value$plusargs("configs=%s", config_file) || !$value$plusargs("payload=%s", payload_file))
            $fatal(1, "Missing DMA vector files");
        $readmemh(config_file, configs);
        $readmemh(payload_file, payload);
        reset_device;
        report_reset_probe;
        for (case_index = 0; case_index < CASES; case_index = case_index + 1) begin
            if (case_index < 12) reset_probe(case_index);
            original = configs[case_index][127:64];
            expected_status = configs[case_index][63:60];
            fault = configs[case_index][59:52];
            scenario = configs[case_index][51:48];
            length = original[15:8]; direction = original[3:2];
            for (i = 0; i < 4096; i = i+1) begin
                memory[i] = {32'hfeedbeef,32'h01234567,case_index[31:0],i[31:0]};
                expected_memory[i] = memory[i];
            end
            if (expected_status == 0) begin
                for (i = 0; i < length; i = i+1) begin
                    if (direction == 1) begin
                        memory[(original[31:16] >> 4)+i] = payload[case_index*255+i];
                        expected_memory[(original[31:16] >> 4)+i] = payload[case_index*255+i];
                    end else if (i != fault)
                        expected_memory[(original[31:16] >> 4)+i] = payload[case_index*255+i];
                end
            end
            @(negedge clk);
            descriptor = original; descriptor_valid = 1; completion_ready = 0;
            ram_request_ready = 0; ram_response_valid = 0;
            @(posedge clk);
            if (!descriptor_ready) $fatal(1, "DMA not idle at case start");
            #1;
            cycles = 0; request_count = 0; response_count = 0; bram_count = 0;
            pending_response = 0; request_stalled = 0;
            seen_done = completion_valid; done_cycle = 0;
            write_gaps = 0; last_request_cycle = -1;
            while (!completion_ready) begin
                @(negedge clk);
                // Poison while busy; another descriptor must not be accepted.
                descriptor = ~original; descriptor_valid = 1;
                ram_request_ready = (scenario == 0) || (cycles%7 < 5 && (scenario != 2 || cycles >= 20));
                if (!pending_response && response_count < request_count
                        && (scenario == 0 || (cycles%5 != 1 && (scenario != 3 || cycles >= 64)))
                        && (scenario != 4 || request_count == length))
                    pending_response = 1;
                ram_response_valid = pending_response;
                ram_response_data = payload[case_index*255+response_count];
                ram_response_error = response_count == fault;
                completion_ready = seen_done && cycles-done_cycle >= 5;
                @(posedge clk);
                cycles = cycles + 1;
                if (descriptor_ready) $fatal(1, "Accepted poison descriptor while busy");
                if (request_stalled && (!ram_request_valid ||
                        {ram_request_write,ram_request_address,(ram_request_write ? ram_request_data : 128'b0)} !== stalled_request))
                    $fatal(1, "RAM request changed under backpressure");
                request_stalled = ram_request_valid && !ram_request_ready;
                stalled_request = {ram_request_write,ram_request_address,(ram_request_write ? ram_request_data : 128'b0)};
                if (ram_request_valid && ram_request_ready) begin
                    if (expected_status != 0 || request_count >= length) $fatal(1, "Unexpected RAM request");
                    if (ram_request_address !== original[63:32] + 16*request_count || ram_request_write !== direction[0])
                        $fatal(1, "Wrong RAM address/direction case=%0d index=%0d", case_index, request_count);
                    if (direction == 1 && ram_request_data !== payload[case_index*255+request_count])
                        $fatal(1, "Corrupted write case=%0d index=%0d", case_index, request_count);
                    if (last_request_cycle >= 0 && cycles-last_request_cycle != 1) write_gaps = write_gaps+1;
                    last_request_cycle = cycles;
                    request_count = request_count + 1;
                end
                if (ram_response_valid && ram_response_ready) begin
                    response_count = response_count + 1; pending_response = 0;
                end
                if (bram_enable) begin
                    if (expected_status != 0 || bram_address[3:0] != 0 || bram_count >= length)
                        $fatal(1, "Unexpected BRAM access");
                    expected_bram_index = bram_count;
                    if (direction == 0 && fault < length && bram_count >= fault)
                        expected_bram_index = bram_count + 1;
                    if (bram_address !== original[31:16] + 16*expected_bram_index || bram_write !== (direction == 0))
                        $fatal(1, "Wrong BRAM address/direction");
                    if (bram_write && bram_write_data !== payload[case_index*255+expected_bram_index])
                        $fatal(1, "Corrupted read payload");
                    bram_count = bram_count + 1;
                end
                // Observe before this edge acknowledges completion, and also
                // after NBA when the final response first publishes completion.
                if (completion_valid) begin
                    if (completion_descriptor !== original || completion_cycles !== done_cycle ||
                            completion_interrupt !== original[1]) $fatal(1, "Unstable completion");
                    if (completion_status !== (expected_status | ((expected_status == 0 && fault < length) ? 4 : 0)
                            | ((done_cycle > MAX_CYCLES) ? 8 : 0))) $fatal(1, "Wrong completion status");
                end
                #1;
                if (completion_valid && !seen_done) begin
                    if (request_count != length || response_count != length) $fatal(1, "Premature completion");
                    seen_done = 1; done_cycle = cycles;
                end
                if (!completion_ready && deadline_missed !== ((seen_done ? done_cycle : cycles) > MAX_CYCLES))
                    $fatal(1, "Deadline alarm missing or early, including unfinished transfers");
                if (cycles > 3000) $fatal(1, "DMA test timeout");
            end
            if (!descriptor_ready || completion_valid) $fatal(1, "Completion did not drain");
            if (expected_status != 0 && (request_count || response_count || bram_count || done_cycle))
                $fatal(1, "Rejected descriptor mutated memory");
            if (expected_status == 0) begin
                if (bram_count != length - ((direction == 0 && fault < length) ? 1 : 0))
                    $fatal(1, "Wrong BRAM beat count");
                if (scenario == 0 && write_gaps != 0) $fatal(1, "Burst is not one beat per clock");
                if (scenario == 0 && done_cycle != length + ((direction == 0) ? 1 : 3))
                    $fatal(1, "Unexpected ideal DMA latency: %0d", done_cycle);
                if (done_cycle > max_latency) max_latency = done_cycle;
                if (direction == 0 && done_cycle < min_read) min_read = done_cycle;
                if (direction == 1 && done_cycle < min_write) min_write = done_cycle;
                total_bytes = total_bytes + 16*length;
            end
            for (i = 0; i < 4096; i = i+1)
                if (memory[i] !== expected_memory[i]) $fatal(1, "BRAM mismatch case=%0d word=%0d", case_index, i);
        end
        $display("PASS DMA cases=%0d bytes=%0d window=%0d max_latency=%0d min_read=%0d min_write=%0d",
                 CASES, total_bytes, MAX_CYCLES, max_latency, min_read, min_write);
        $finish;
    end
    initial begin
        #(CASES*40000 + 1000000);
        $fatal(1, "Global DMA simulation timeout");
    end
endmodule
