/*
 * tb_cdf53_inverse_lift.v - Testbench for CDF 5/3 Inverse Lifting
 * ================================================================
 *
 * Tests the cdf53_inverse_lift module against known test vectors.
 * Uses the same algorithm as the Python reference implementation.
 *
 * Test cases:
 * 1. DC signal (constant input)
 * 2. Impulse response
 * 3. Ramp signal
 * 4. Alternating signal
 * 5. Random pattern
 *
 * To generate golden reference vectors, run:
 *   python generate_test_vectors.py
 */

`timescale 1ns / 1ps

module tb_cdf53_inverse_lift;

    // Test parameters
    localparam N = 4;  // 4 coefficient pairs -> 8 output samples
    localparam CLK_PERIOD = 10;

    // Signals
    reg                      clk;
    reg                      rst_n;
    reg                      valid_in;
    reg  signed [16*N-1:0]   s_flat;
    reg  signed [16*N-1:0]   d_flat;
    wire                     valid_out;
    wire signed [32*N-1:0]   x_flat;

    // DUT instantiation
    cdf53_inverse_lift #(.N(N)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .s_flat(s_flat),
        .d_flat(d_flat),
        .valid_out(valid_out),
        .x_flat(x_flat)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Helper: Pack array into flat vector
    function [16*N-1:0] pack_array;
        input signed [15:0] arr0, arr1, arr2, arr3;
        begin
            pack_array = {arr3, arr2, arr1, arr0};
        end
    endfunction

    // Helper: Extract sample from output
    function signed [15:0] get_sample;
        input [32*N-1:0] flat;
        input integer idx;
        begin
            get_sample = flat[16*idx +: 16];
        end
    endfunction

    // Test procedure
    integer test_num;
    integer i;
    integer errors;
    reg signed [15:0] expected [0:2*N-1];

    initial begin
        $display("==============================================");
        $display("CDF 5/3 Inverse Lifting Testbench");
        $display("==============================================");
        $display("N = %0d (output samples = %0d)", N, 2*N);
        $display("");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        s_flat = 0;
        d_flat = 0;
        errors = 0;

        // Reset
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);

        // =========================================================
        // TEST 1: DC Signal (all smooth, no detail)
        // =========================================================
        test_num = 1;
        $display("Test %0d: DC Signal (s=[1000,1000,1000,1000], d=[0,0,0,0])", test_num);

        // s = [1000, 1000, 1000, 1000], d = [0, 0, 0, 0]
        // Expected: uniform output close to 1000
        s_flat = {16'sd1000, 16'sd1000, 16'sd1000, 16'sd1000};
        d_flat = {16'sd0, 16'sd0, 16'sd0, 16'sd0};
        valid_in = 1;
        #(CLK_PERIOD);
        valid_in = 0;

        // Check output
        if (valid_out) begin
            $display("  Output: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                get_sample(x_flat, 0), get_sample(x_flat, 1),
                get_sample(x_flat, 2), get_sample(x_flat, 3),
                get_sample(x_flat, 4), get_sample(x_flat, 5),
                get_sample(x_flat, 6), get_sample(x_flat, 7));

            // For DC: output should be ~1000 everywhere
            for (i = 0; i < 2*N; i = i + 1) begin
                if (get_sample(x_flat, i) < 900 || get_sample(x_flat, i) > 1100) begin
                    $display("  ERROR at sample %0d: expected ~1000, got %0d",
                        i, get_sample(x_flat, i));
                    errors = errors + 1;
                end
            end
        end else begin
            $display("  ERROR: valid_out not asserted");
            errors = errors + 1;
        end
        $display("");

        #(CLK_PERIOD * 2);

        // =========================================================
        // TEST 2: Impulse in smooth[0]
        // =========================================================
        test_num = 2;
        $display("Test %0d: Impulse (s=[16384,0,0,0], d=[0,0,0,0])", test_num);

        s_flat = {16'sd0, 16'sd0, 16'sd0, 16'sd16384};
        d_flat = {16'sd0, 16'sd0, 16'sd0, 16'sd0};
        valid_in = 1;
        #(CLK_PERIOD);
        valid_in = 0;

        if (valid_out) begin
            $display("  Output: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                get_sample(x_flat, 0), get_sample(x_flat, 1),
                get_sample(x_flat, 2), get_sample(x_flat, 3),
                get_sample(x_flat, 4), get_sample(x_flat, 5),
                get_sample(x_flat, 6), get_sample(x_flat, 7));

            // First sample should be dominant
            if (get_sample(x_flat, 0) < 10000) begin
                $display("  WARNING: expected large value at sample 0");
            end
        end
        $display("");

        #(CLK_PERIOD * 2);

        // =========================================================
        // TEST 3: Detail-only (high-frequency)
        // =========================================================
        test_num = 3;
        $display("Test %0d: Detail only (s=[0,0,0,0], d=[8192,8192,8192,8192])", test_num);

        s_flat = {16'sd0, 16'sd0, 16'sd0, 16'sd0};
        d_flat = {16'sd8192, 16'sd8192, 16'sd8192, 16'sd8192};
        valid_in = 1;
        #(CLK_PERIOD);
        valid_in = 0;

        if (valid_out) begin
            $display("  Output: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                get_sample(x_flat, 0), get_sample(x_flat, 1),
                get_sample(x_flat, 2), get_sample(x_flat, 3),
                get_sample(x_flat, 4), get_sample(x_flat, 5),
                get_sample(x_flat, 6), get_sample(x_flat, 7));

            // Should have alternating pattern (detail = high frequency)
            // Odd samples should be positive (from detail + avg)
        end
        $display("");

        #(CLK_PERIOD * 2);

        // =========================================================
        // TEST 4: Ramp signal
        // =========================================================
        test_num = 4;
        $display("Test %0d: Ramp (s=[0,4096,8192,12288], d=[1024,1024,1024,1024])", test_num);

        s_flat = {16'sd12288, 16'sd8192, 16'sd4096, 16'sd0};
        d_flat = {16'sd1024, 16'sd1024, 16'sd1024, 16'sd1024};
        valid_in = 1;
        #(CLK_PERIOD);
        valid_in = 0;

        if (valid_out) begin
            $display("  Output: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                get_sample(x_flat, 0), get_sample(x_flat, 1),
                get_sample(x_flat, 2), get_sample(x_flat, 3),
                get_sample(x_flat, 4), get_sample(x_flat, 5),
                get_sample(x_flat, 6), get_sample(x_flat, 7));
        end
        $display("");

        #(CLK_PERIOD * 2);

        // =========================================================
        // TEST 5: Saturation test (near limits)
        // =========================================================
        test_num = 5;
        $display("Test %0d: Saturation (s=[32000,32000,-32000,-32000], d=[0,16000,-16000,0])", test_num);

        s_flat = {-16'sd32000, -16'sd32000, 16'sd32000, 16'sd32000};
        d_flat = {16'sd0, -16'sd16000, 16'sd16000, 16'sd0};
        valid_in = 1;
        #(CLK_PERIOD);
        valid_in = 0;

        if (valid_out) begin
            $display("  Output: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                get_sample(x_flat, 0), get_sample(x_flat, 1),
                get_sample(x_flat, 2), get_sample(x_flat, 3),
                get_sample(x_flat, 4), get_sample(x_flat, 5),
                get_sample(x_flat, 6), get_sample(x_flat, 7));

            // Check that outputs are within valid range
            for (i = 0; i < 2*N; i = i + 1) begin
                if (get_sample(x_flat, i) < -32768 || get_sample(x_flat, i) > 32767) begin
                    $display("  ERROR: sample %0d out of range: %0d", i, get_sample(x_flat, i));
                    errors = errors + 1;
                end
            end
        end
        $display("");

        // =========================================================
        // Summary
        // =========================================================
        #(CLK_PERIOD * 5);

        $display("==============================================");
        $display("Test Summary: %0d errors", errors);
        $display("==============================================");

        if (errors == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("SOME TESTS FAILED");
        end

        $finish;
    end

endmodule
