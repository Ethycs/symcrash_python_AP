/*
 * tree_rom.v - Wavelet Packet Tree ROM Tables
 * ============================================
 *
 * Contains ROM tables for all 5 tree types.
 * Each tree has 32 leaves with associated (depth, k) pairs.
 *
 * Tree Types:
 *   0: Balanced (all leaves at depth 5)
 *   1: Harmonic (wider lows, detailed highs)
 *   2: Texture (high-band resolution)
 *   3: Transient (mid-band emphasis)
 *   4: Drone (low-band detail)
 *
 * Generated from Python wavelet_trees package.
 */

module tree_rom (
    input  wire [2:0]  tree_id,
    input  wire [4:0]  leaf_idx,
    output reg  [2:0]  leaf_depth,
    output reg  [5:0]  leaf_k,
    output reg         is_complex
);

    // =========================================================================
    // TREE 0: BALANCED (Uniform resolution, all leaves at depth 5)
    // =========================================================================

    // All 32 leaves at depth 5, k = 0..31
    // Complex mask: leaves 16-31 (upper half)

    // =========================================================================
    // TREE 1: HARMONIC (Wider lows, detailed highs)
    // =========================================================================

    // Lower frequencies have shallower depth (wider bands)
    // Higher frequencies have deeper depth (narrower bands)

    // =========================================================================
    // ROM LOOKUP
    // =========================================================================

    always @(*) begin
        case (tree_id)
            3'd0: begin  // Balanced
                leaf_depth = 3'd5;
                leaf_k = {1'b0, leaf_idx};
                is_complex = (leaf_idx >= 16);
            end

            3'd1: begin  // Harmonic
                // Shallower depth for lower leaves (bass)
                case (leaf_idx)
                    5'd0:  begin leaf_depth = 3'd3; leaf_k = 6'd0; end  // Very low
                    5'd1:  begin leaf_depth = 3'd3; leaf_k = 6'd1; end
                    5'd2:  begin leaf_depth = 3'd4; leaf_k = 6'd4; end
                    5'd3:  begin leaf_depth = 3'd4; leaf_k = 6'd5; end
                    5'd4:  begin leaf_depth = 3'd4; leaf_k = 6'd6; end
                    5'd5:  begin leaf_depth = 3'd4; leaf_k = 6'd7; end
                    5'd6:  begin leaf_depth = 3'd5; leaf_k = 6'd16; end
                    5'd7:  begin leaf_depth = 3'd5; leaf_k = 6'd17; end
                    5'd8:  begin leaf_depth = 3'd5; leaf_k = 6'd18; end
                    5'd9:  begin leaf_depth = 3'd5; leaf_k = 6'd19; end
                    5'd10: begin leaf_depth = 3'd5; leaf_k = 6'd20; end
                    5'd11: begin leaf_depth = 3'd5; leaf_k = 6'd21; end
                    5'd12: begin leaf_depth = 3'd5; leaf_k = 6'd22; end
                    5'd13: begin leaf_depth = 3'd5; leaf_k = 6'd23; end
                    5'd14: begin leaf_depth = 3'd5; leaf_k = 6'd24; end
                    5'd15: begin leaf_depth = 3'd5; leaf_k = 6'd25; end
                    5'd16: begin leaf_depth = 3'd5; leaf_k = 6'd26; end
                    5'd17: begin leaf_depth = 3'd5; leaf_k = 6'd27; end
                    5'd18: begin leaf_depth = 3'd5; leaf_k = 6'd28; end
                    5'd19: begin leaf_depth = 3'd5; leaf_k = 6'd29; end
                    5'd20: begin leaf_depth = 3'd5; leaf_k = 6'd30; end
                    5'd21: begin leaf_depth = 3'd5; leaf_k = 6'd31; end
                    default: begin leaf_depth = 3'd5; leaf_k = {1'b0, leaf_idx}; end
                endcase
                is_complex = (leaf_idx >= 6);
            end

            3'd2: begin  // Texture (high-band focus)
                // Deeper splits in high frequency region
                case (leaf_idx)
                    5'd0:  begin leaf_depth = 3'd4; leaf_k = 6'd0; end
                    5'd1:  begin leaf_depth = 3'd4; leaf_k = 6'd1; end
                    5'd2:  begin leaf_depth = 3'd4; leaf_k = 6'd2; end
                    5'd3:  begin leaf_depth = 3'd4; leaf_k = 6'd3; end
                    5'd4:  begin leaf_depth = 3'd4; leaf_k = 6'd4; end
                    5'd5:  begin leaf_depth = 3'd4; leaf_k = 6'd5; end
                    5'd6:  begin leaf_depth = 3'd4; leaf_k = 6'd6; end
                    5'd7:  begin leaf_depth = 3'd4; leaf_k = 6'd7; end
                    5'd8:  begin leaf_depth = 3'd5; leaf_k = 6'd16; end
                    5'd9:  begin leaf_depth = 3'd5; leaf_k = 6'd17; end
                    5'd10: begin leaf_depth = 3'd5; leaf_k = 6'd18; end
                    5'd11: begin leaf_depth = 3'd5; leaf_k = 6'd19; end
                    5'd12: begin leaf_depth = 3'd5; leaf_k = 6'd20; end
                    5'd13: begin leaf_depth = 3'd5; leaf_k = 6'd21; end
                    5'd14: begin leaf_depth = 3'd5; leaf_k = 6'd22; end
                    5'd15: begin leaf_depth = 3'd5; leaf_k = 6'd23; end
                    5'd16: begin leaf_depth = 3'd6; leaf_k = 6'd48; end
                    5'd17: begin leaf_depth = 3'd6; leaf_k = 6'd49; end
                    5'd18: begin leaf_depth = 3'd6; leaf_k = 6'd50; end
                    5'd19: begin leaf_depth = 3'd6; leaf_k = 6'd51; end
                    5'd20: begin leaf_depth = 3'd6; leaf_k = 6'd52; end
                    5'd21: begin leaf_depth = 3'd6; leaf_k = 6'd53; end
                    5'd22: begin leaf_depth = 3'd6; leaf_k = 6'd54; end
                    5'd23: begin leaf_depth = 3'd6; leaf_k = 6'd55; end
                    5'd24: begin leaf_depth = 3'd6; leaf_k = 6'd56; end
                    5'd25: begin leaf_depth = 3'd6; leaf_k = 6'd57; end
                    5'd26: begin leaf_depth = 3'd6; leaf_k = 6'd58; end
                    5'd27: begin leaf_depth = 3'd6; leaf_k = 6'd59; end
                    5'd28: begin leaf_depth = 3'd6; leaf_k = 6'd60; end
                    5'd29: begin leaf_depth = 3'd6; leaf_k = 6'd61; end
                    5'd30: begin leaf_depth = 3'd6; leaf_k = 6'd62; end
                    5'd31: begin leaf_depth = 3'd6; leaf_k = 6'd63; end
                    default: begin leaf_depth = 3'd5; leaf_k = {1'b0, leaf_idx}; end
                endcase
                is_complex = (leaf_idx >= 8);
            end

            3'd3: begin  // Transient (mid-band emphasis)
                // More detail in mid frequencies (100-2000 Hz typical)
                case (leaf_idx)
                    5'd0:  begin leaf_depth = 3'd3; leaf_k = 6'd0; end
                    5'd1:  begin leaf_depth = 3'd3; leaf_k = 6'd1; end
                    5'd2:  begin leaf_depth = 3'd5; leaf_k = 6'd8; end
                    5'd3:  begin leaf_depth = 3'd5; leaf_k = 6'd9; end
                    5'd4:  begin leaf_depth = 3'd5; leaf_k = 6'd10; end
                    5'd5:  begin leaf_depth = 3'd5; leaf_k = 6'd11; end
                    5'd6:  begin leaf_depth = 3'd5; leaf_k = 6'd12; end
                    5'd7:  begin leaf_depth = 3'd5; leaf_k = 6'd13; end
                    5'd8:  begin leaf_depth = 3'd5; leaf_k = 6'd14; end
                    5'd9:  begin leaf_depth = 3'd5; leaf_k = 6'd15; end
                    5'd10: begin leaf_depth = 3'd5; leaf_k = 6'd16; end
                    5'd11: begin leaf_depth = 3'd5; leaf_k = 6'd17; end
                    5'd12: begin leaf_depth = 3'd5; leaf_k = 6'd18; end
                    5'd13: begin leaf_depth = 3'd5; leaf_k = 6'd19; end
                    5'd14: begin leaf_depth = 3'd5; leaf_k = 6'd20; end
                    5'd15: begin leaf_depth = 3'd5; leaf_k = 6'd21; end
                    5'd16: begin leaf_depth = 3'd4; leaf_k = 6'd11; end
                    5'd17: begin leaf_depth = 3'd4; leaf_k = 6'd12; end
                    5'd18: begin leaf_depth = 3'd4; leaf_k = 6'd13; end
                    5'd19: begin leaf_depth = 3'd4; leaf_k = 6'd14; end
                    5'd20: begin leaf_depth = 3'd4; leaf_k = 6'd15; end
                    default: begin leaf_depth = 3'd5; leaf_k = {1'b0, leaf_idx}; end
                endcase
                is_complex = (leaf_idx >= 2);
            end

            3'd4: begin  // Drone (low-band detail)
                // Very detailed bass, simpler highs
                case (leaf_idx)
                    5'd0:  begin leaf_depth = 3'd5; leaf_k = 6'd0; end
                    5'd1:  begin leaf_depth = 3'd5; leaf_k = 6'd1; end
                    5'd2:  begin leaf_depth = 3'd5; leaf_k = 6'd2; end
                    5'd3:  begin leaf_depth = 3'd5; leaf_k = 6'd3; end
                    5'd4:  begin leaf_depth = 3'd5; leaf_k = 6'd4; end
                    5'd5:  begin leaf_depth = 3'd5; leaf_k = 6'd5; end
                    5'd6:  begin leaf_depth = 3'd5; leaf_k = 6'd6; end
                    5'd7:  begin leaf_depth = 3'd5; leaf_k = 6'd7; end
                    5'd8:  begin leaf_depth = 3'd4; leaf_k = 6'd4; end
                    5'd9:  begin leaf_depth = 3'd4; leaf_k = 6'd5; end
                    5'd10: begin leaf_depth = 3'd4; leaf_k = 6'd6; end
                    5'd11: begin leaf_depth = 3'd4; leaf_k = 6'd7; end
                    5'd12: begin leaf_depth = 3'd3; leaf_k = 6'd4; end
                    5'd13: begin leaf_depth = 3'd3; leaf_k = 6'd5; end
                    5'd14: begin leaf_depth = 3'd3; leaf_k = 6'd6; end
                    5'd15: begin leaf_depth = 3'd3; leaf_k = 6'd7; end
                    default: begin leaf_depth = 3'd5; leaf_k = {1'b0, leaf_idx}; end
                endcase
                is_complex = (leaf_idx < 8);
            end

            default: begin  // Default to balanced
                leaf_depth = 3'd5;
                leaf_k = {1'b0, leaf_idx};
                is_complex = (leaf_idx >= 16);
            end
        endcase
    end

endmodule


/*
 * tree_rom_balanced - Optimized ROM for Tree 0 Only
 * ==================================================
 *
 * Simpler version when only Tree 0 (balanced) is needed.
 */

module tree_rom_balanced (
    input  wire [4:0]  leaf_idx,
    output wire [2:0]  leaf_depth,
    output wire [5:0]  leaf_k,
    output wire        is_complex
);

    // Tree 0 is trivial: all leaves at depth 5, k = leaf_idx
    assign leaf_depth = 3'd5;
    assign leaf_k = {1'b0, leaf_idx};
    assign is_complex = (leaf_idx >= 16);

endmodule
