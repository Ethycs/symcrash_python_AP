/*
 * cdf53_inverse_lift.v - CDF 5/3 Inverse Lifting Module (Single Level)
 * =====================================================================
 *
 * Performs one level of CDF 5/3 inverse wavelet transform.
 * Takes N smooth coefficients (s) and N detail coefficients (d),
 * produces 2N output samples (x).
 *
 * CDF 5/3 Inverse Lifting Steps:
 *   1. Undo update: x[2i]   = s[i] - floor((d[i-1] + d[i] + 2) / 4)
 *   2. Undo predict: x[2i+1] = d[i] + floor((x[2i] + x[2i+2] + 1) / 2)
 *
 * This is bit-exact with the Python implementation using shift_round().
 * All arithmetic is Q1.15 signed fixed-point.
 *
 * Parameters:
 *   N - Number of input coefficient pairs (output is 2*N samples)
 *
 * Timing:
 *   Combinational (single cycle) for small N
 *   Can be pipelined for larger N if needed
 */

module cdf53_inverse_lift #(
    parameter N = 4  // Number of coefficient pairs (power of 2)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire signed [16*N-1:0]   s_flat,    // Smooth coefficients (packed)
    input  wire signed [16*N-1:0]   d_flat,    // Detail coefficients (packed)
    output wire                     valid_out,
    output wire signed [32*N-1:0]   x_flat     // Output samples (packed)
);

    // Unpack inputs
    wire signed [15:0] s [0:N-1];
    wire signed [15:0] d [0:N-1];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : unpack
            assign s[gi] = s_flat[16*gi +: 16];
            assign d[gi] = d_flat[16*gi +: 16];
        end
    endgenerate

    // Intermediate even samples (after undo update)
    wire signed [15:0] x_even [0:N-1];

    // Output samples
    wire signed [15:0] x [0:2*N-1];

    // =========================================================================
    // STEP 1: Undo Update - Compute even samples
    // x[2i] = s[i] - ((d[i-1] + d[i] + 2) >> 2)
    // =========================================================================

    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : undo_update
            // Boundary handling: d[-1] = d[0] (symmetric extension)
            wire signed [15:0] d_left  = (gi == 0) ? d[0] : d[gi-1];
            wire signed [15:0] d_right = d[gi];

            // Sum with rounding bias, then shift
            // (d_left + d_right + 2) >> 2
            // Use 18 bits for intermediate to prevent overflow
            wire signed [17:0] sum = d_left + d_right + 18'sd2;
            wire signed [15:0] update = sum[17:2];  // Arithmetic right shift by 2

            // Saturating subtraction
            wire signed [16:0] diff = s[gi] - update;

            sat16 #(.IN_WIDTH(17)) sat_even (
                .in(diff),
                .out(x_even[gi])
            );

            assign x[2*gi] = x_even[gi];
        end
    endgenerate

    // =========================================================================
    // STEP 2: Undo Predict - Compute odd samples
    // x[2i+1] = d[i] + ((x[2i] + x[2i+2] + 1) >> 1)
    // =========================================================================

    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : undo_predict
            // Boundary handling: x[2N] = x[2N-2] (symmetric extension)
            wire signed [15:0] x_left  = x_even[gi];
            wire signed [15:0] x_right = (gi == N-1) ? x_even[N-1] : x_even[gi+1];

            // Average with rounding
            // (x_left + x_right + 1) >> 1
            wire signed [16:0] sum = x_left + x_right + 17'sd1;
            wire signed [15:0] avg = sum[16:1];  // Arithmetic right shift by 1

            // Saturating addition
            wire signed [16:0] result = d[gi] + avg;

            sat16 #(.IN_WIDTH(17)) sat_odd (
                .in(result),
                .out(x[2*gi + 1])
            );
        end
    endgenerate

    // =========================================================================
    // Pack outputs
    // =========================================================================

    generate
        for (gi = 0; gi < 2*N; gi = gi + 1) begin : pack_out
            assign x_flat[16*gi +: 16] = x[gi];
        end
    endgenerate

    // Valid signal passes through (combinational)
    assign valid_out = valid_in;

endmodule
