`timescale 1ns/1ps
module tb_ccs_headroom;
    wire [15:0] done;
    genvar h;
    generate for (h = 0; h <= 15; h = h + 1) begin : cases
        reg complete = 0;
        integer value, bounded, expected;
        ccs_idwpt #(.K(2), .HEADROOM_BITS(h)) dut (
            .clk(1'b0), .rst_n(1'b0), .coeff_valid(1'b0),
            .coeff_data(32'b0), .audio_ready(1'b0)
        );
        assign done[h] = complete;
        initial begin
            // Independent literal specification: SAT16, widen, multiply,
            // SAT16. All products fit a signed 32-bit integer for h<=15.
            for (value = -65536; value <= 65535; value = value + 1) begin
                bounded = (value > 32767) ? 32767 : ((value < -32768) ? -32768 : value);
                expected = bounded * (1 << h);
                expected = (expected > 32767) ? 32767 : ((expected < -32768) ? -32768 : expected);
                if (dut.restore_headroom(value) !== expected[15:0])
                    $fatal(1, "Headroom mismatch h=%0d value=%0d", h, value);
            end
            complete = 1;
        end
    end endgenerate
    initial begin
        wait (&done);
        $display("PASS headroom_values=2097152 widths=17 headroom=0..15 SAT16_before_and_after");
        $finish;
    end
endmodule
