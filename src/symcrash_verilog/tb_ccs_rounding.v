`timescale 1ns/1ps
module tb_ccs_rounding;
    ccs_idwpt dut (.clk(1'b0), .rst_n(1'b0), .coeff_valid(1'b0),
        .coeff_data(64'd0), .audio_ready(1'b0));
    reg [4095:0] expected_file;
    reg [16:0] expected [0:327678];
    reg signed [15:0] quarter_value;
    reg signed [16:0] half_value;
    integer value, index, bounded;
    initial begin
        if (!$value$plusargs("expected=%s", expected_file)) $fatal(1, "Missing rounding oracle");
        $readmemh(expected_file, expected);
        index = 0;
        for (value = -65536; value <= 65535; value = value + 1) begin
            quarter_value = dut.quarter_rne(value);
            if ({quarter_value[15], quarter_value} !== expected[index])
                $fatal(1, "Quarter RNE mismatch at %0d", value);
            index = index + 1;
        end
        // The complete proven sum range of two wide inverse even samples.
        for (value = -98304; value <= 98302; value = value + 1) begin
            half_value = dut.half_rne(value);
            if (half_value !== expected[index]) $fatal(1, "Half RNE mismatch at %0d", value);
            index = index + 1;
        end
        for (value = -131072; value <= 131071; value = value + 1) begin
            bounded = (value > 32767) ? 32767 : ((value < -32768) ? -32768 : value);
            if (dut.clamp18(value) !== bounded[15:0])
                $fatal(1, "SAT16 mismatch at %0d", value);
        end
        $display("PASS rounding_sums=%0d signed_ties=RNE saturation_inputs=262144", index);
        $finish;
    end
endmodule
