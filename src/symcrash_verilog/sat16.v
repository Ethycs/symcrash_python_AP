/*
 * sat16.v - Q1.15 Saturation Module
 * ==================================
 *
 * Saturates a wider signed value to 16-bit signed range [-32768, 32767].
 * Used throughout the CDF 5/3 wavelet pipeline to prevent overflow.
 *
 * Input:  17-bit or wider signed value (from add operations)
 * Output: 16-bit saturated signed value
 */

module sat16 #(
    parameter IN_WIDTH = 17
) (
    input  wire signed [IN_WIDTH-1:0] in,
    output wire signed [15:0]         out
);

    // Check for overflow/underflow
    // If high bits don't match sign extension, we need to saturate
    wire overflow  = (in > 17'sd32767);
    wire underflow = (in < -17'sd32768);

    assign out = overflow  ? 16'sd32767 :
                 underflow ? -16'sd32768 :
                 in[15:0];

endmodule


/*
 * sat16_add - Saturating 16-bit Adder
 * ===================================
 *
 * Adds two 16-bit signed values with saturation.
 */

module sat16_add (
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output wire signed [15:0] sum
);

    wire signed [16:0] wide_sum = a + b;

    sat16 #(.IN_WIDTH(17)) sat_inst (
        .in(wide_sum),
        .out(sum)
    );

endmodule


/*
 * sat16_sub - Saturating 16-bit Subtractor
 * ========================================
 *
 * Subtracts two 16-bit signed values with saturation.
 */

module sat16_sub (
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output wire signed [15:0] diff
);

    wire signed [16:0] wide_diff = a - b;

    sat16 #(.IN_WIDTH(17)) sat_inst (
        .in(wide_diff),
        .out(diff)
    );

endmodule
