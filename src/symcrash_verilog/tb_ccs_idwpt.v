`timescale 1ns/1ps
module tb_ccs_idwpt;
    parameter K = 4, HEADROOM_BITS = 2, CASES = 1, STALLS = 0;
    localparam LANES = (K > 4) ? 4 : K;
    localparam GROUPS = K / LANES;
    localparam COMPUTE_CYCLES = (K==2) ? 99 : 247;
    localparam IDEAL_CYCLES = (K==2) ? 179 : 311*GROUPS + 16;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, coeff_valid = 0, audio_ready = 0;
    wire coeff_ready, audio_valid, audio_last;
    reg [16*LANES-1:0] coeff_data = 0;
    wire [63:0] audio_data;
    ccs_idwpt #(.K(K), .HEADROOM_BITS(HEADROOM_BITS)) dut (
        .clk(clk), .rst_n(rst_n), .coeff_valid(coeff_valid),
        .coeff_ready(coeff_ready), .coeff_data(coeff_data),
        .audio_valid(audio_valid), .audio_ready(audio_ready),
        .audio_data(audio_data), .audio_last(audio_last)
    );
    wire [2:0] monitored_state,monitored_level;
    wire monitored_group;
    generate if(K==2) begin
        assign monitored_state=dut.k2.state;
        assign monitored_level=dut.k2.level;
        assign monitored_group=1'b0;
    end else begin
        assign monitored_state=dut.serial.state;
        assign monitored_level=dut.serial.level;
        assign monitored_group=dut.serial.group_index;
    end endgenerate
    reg [15:0] coefficients [0:CASES*64*K-1];
    reg [15:0] expected [0:CASES*64-1];
    reg [4095:0] coeff_file, expected_file;
    integer case_index, input_beat, output_beat, cycles, first_cycle;
    integer lane, leaf, group_id, index, group_last_cycle, elapsed;
    integer min_cycles = 2147483647, max_cycles = 0;
    integer wait_count, discarded, probe;
    reg pending, awaiting_group, was_stalled;
    reg [63:0] stalled_data;
    reg stalled_last;

    task reset_device;
        begin
            @(negedge clk);
            rst_n = 0; coeff_valid = 0; audio_ready = 0;
            repeat (2) @(posedge clk);
            @(negedge clk); rst_n = 1;
            #1;
            if (!coeff_ready || audio_valid) $fatal(1, "Reset handshake failed");
            repeat (3) begin
                @(posedge clk); #1;
                if (audio_valid || !coeff_ready) $fatal(1, "Aborted block leaked after reset");
            end
        end
    endtask

    // Each reset probe is followed by a complete, checked block. Memories
    // contain poison from a partial load/transform/output before that block.
    task reset_probe;
        input integer kind;
        begin
            discarded = 0;
            while (discarded < ((kind == 15 || kind == 16) ? 64*GROUPS : ((kind == 0) ? 17 : 64))) begin
                @(negedge clk); coeff_valid = 1; coeff_data = {LANES{16'h8001}};
                @(posedge clk);
                if (coeff_ready) discarded = discarded + 1;
            end
            @(negedge clk); coeff_valid = 0;
            // Every arithmetic pipeline age, the final-pair flush, later
            // levels, root-only update, stalled output and K8's second group.
            case (kind)
                0: wait_count = 0;
                11: wait_count = (K==2) ? 9 : 36;
                12: wait_count = (K==2) ? 11 : 38;
                13: wait_count = (K==2) ? 13 : 41;
                14: wait_count = (K==2) ? 31 : 125;
                15: wait_count = (K==2) ? 99 : 247;
                16: wait_count = 7;
                17: wait_count = (K==2) ? 88 : 212;
                default: wait_count = kind;
            endcase
            repeat (wait_count) @(posedge clk);
            #1;
            if (kind == 15 && !audio_valid) $fatal(1, "Output reset probe missed output state");
            if (kind != 0 && kind != 15 && (audio_valid || coeff_ready))
                $fatal(1, "Pipeline reset probe missed computation");
            if (kind >= 11 && kind <= 14 && monitored_state != 3'd2)
                $fatal(1, "Missed stream drain");
            if (kind == 17 && monitored_level != 5) $fatal(1, "Missed root-only update");
            if (kind == 16 && K == 8 && monitored_group != 1) $fatal(1, "Missed second group");
            reset_device;
        end
    endtask

    initial begin
        if (!$value$plusargs("coeffs=%s", coeff_file) || !$value$plusargs("expected=%s", expected_file))
            $fatal(1, "Missing vector filenames");
        $readmemh(coeff_file, coefficients);
        $readmemh(expected_file, expected);
        reset_device;
        cycles = 0;
        for (case_index = 0; case_index < CASES; case_index = case_index + 1) begin
            if (case_index < 18) reset_probe(case_index);
            input_beat = 0; output_beat = 0; pending = 0;
            awaiting_group = 0; was_stalled = 0; first_cycle = -1;
            while (output_beat < 16) begin
                @(negedge clk);
                if (!pending && input_beat < 64*GROUPS && (!STALLS || cycles%7 != 0)) pending = 1;
                coeff_valid = pending;
                // Assert valid with poison while busy after the last input.
                // The decoder must not accept a new block before output drains.
                if (input_beat == 64*GROUPS) begin
                    coeff_valid = 1; coeff_data = {LANES{16'h7fff}};
                end else begin
                    leaf = input_beat%64; group_id = input_beat/64;
                    for (lane = 0; lane < LANES; lane = lane + 1)
                        coeff_data[16*lane +: 16] = coefficients[case_index*64*K + leaf*K + group_id*LANES + lane];
                end
                audio_ready = !STALLS || cycles%11 < 7;
                @(posedge clk);
                if (coeff_valid && coeff_ready) begin
                    if (input_beat == 64*GROUPS) $fatal(1, "Accepted input while current output pending");
                    if (input_beat == 0) first_cycle = cycles;
                    input_beat = input_beat + 1; pending = 0;
                    if (input_beat%64 == 0) begin
                        group_last_cycle = cycles; awaiting_group = 1;
                    end
                end
                if (was_stalled && (!audio_valid || audio_data !== stalled_data || audio_last !== stalled_last))
                    $fatal(1, "Output changed under backpressure");
                was_stalled = audio_valid && !audio_ready;
                stalled_data = audio_data; stalled_last = audio_last;
                if (audio_valid) begin
                    if (coeff_ready) $fatal(1, "Input and output incorrectly overlap");
                    if (audio_last !== (output_beat == 15)) $fatal(1, "Wrong audio_last");
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        index = case_index*64 + output_beat*4 + lane;
                        if (audio_data[16*lane +: 16] !== expected[index])
                            $fatal(1, "Mismatch case=%0d sample=%0d got=%0d expected=%0d",
                                case_index, output_beat*4+lane,
                                $signed(audio_data[16*lane +: 16]), $signed(expected[index]));
                    end
                    if (audio_ready) output_beat = output_beat + 1;
                end
                #1;
                if (awaiting_group && (coeff_ready || audio_valid)) begin
                    if (cycles-group_last_cycle != COMPUTE_CYCLES)
                        $fatal(1, "Wrong compute latency: %0d", cycles-group_last_cycle);
                    awaiting_group = 0;
                end
                cycles = cycles + 1;
                if (first_cycle >= 0 && cycles-first_cycle > IDEAL_CYCLES*3)
                    $fatal(1, "Decoder timeout");
            end
            elapsed = cycles-first_cycle;
            if (!STALLS && elapsed != IDEAL_CYCLES) $fatal(1, "Wrong no-stall transaction cost: %0d", elapsed);
            if (elapsed < min_cycles) min_cycles = elapsed;
            if (elapsed > max_cycles) max_cycles = elapsed;
            // First output of the next frame must not inherit valid/last.
            if (audio_valid || !coeff_ready) $fatal(1, "Output did not drain");
        end
        $display("PASS cases=%0d K=%0d headroom=%0d stalls=%0d compute_group=%0d min=%0d max=%0d",
                 CASES, K, HEADROOM_BITS, STALLS, COMPUTE_CYCLES, min_cycles, max_cycles);
        $finish;
    end
    initial begin
        #(CASES*100000 + 1000000);
        $fatal(1, "Global simulation timeout");
    end
endmodule
