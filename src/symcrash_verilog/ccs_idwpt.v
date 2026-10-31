/* Current CCS decoder: N=64, depth six, K=2/4/8, signed RNE.
 * Input: 64 leaves of min(K,4) int16 slots, low lane first; K8 repeats
 * for slots 4..7 after ready returns. Output: 16 chronological four-sample
 * beats, stable until accepted. Active-low reset aborts partial work.
 *
 * One pair enters the arithmetic pipeline each clock. Memory read, operand
 * alignment, sum, RNE and second lift operation occupy separate stages.
 * Wide evens and their original details stream directly into undo-predict.
 * One pending pair supplies the left even when the next even arrives;
 * node ends repeat that pending even. No scratch pass or detail reread.
 * Five levels issue 32 pairs and drain ten clocks, then the root update
 * takes 37 clocks: 247 compute clocks/group. No-stall: 327 K2/K4, 638 K8.
 * These are decoder costs, not integrated device/tier qualification.
 *
 * Stored data remains int16. The inverse range proof requires 17-bit wide
 * evens and 18-bit even sums/odds. Node SAT16 and final headroom restoration
 * are unchanged; registers retime operations without reducing precision.
 */
module ccs_idwpt #(
    parameter K = 4,
    parameter HEADROOM_BITS = 2
) (
    input wire clk,
    input wire rst_n,
    input wire coeff_valid,
    output wire coeff_ready,
    input wire [16*((K > 4) ? 4 : K)-1:0] coeff_data,
    output wire audio_valid,
    input wire audio_ready,
    output wire [63:0] audio_data,
    output wire audio_last
);
    localparam LANES = (K > 4) ? 4 : K;
    localparam WORD_BITS = 16 * LANES;
    localparam EVEN_BITS = 17 * LANES;
    localparam GROUPS = K / LANES;
    localparam LOAD = 3'd0, UPDATE = 3'd1, UPDATE_DRAIN = 3'd2, OUTPUT = 3'd5;

    reg [2:0] state, level;
    reg [4:0] pair_index;
    reg [5:0] load_leaf;
    reg group_index, source_bank;
    reg [3:0] output_index;
    reg [WORD_BITS-1:0] bank_a [0:63], bank_b [0:63];
    reg [63:0] output_words [0:15];
    reg [WORD_BITS-1:0] previous_detail;
    reg [EVEN_BITS-1:0] pending_even;
    reg [WORD_BITS-1:0] pending_detail;
    reg [4:0] pending_index;
    reg pending_valid, pending_right_boundary, right_boundary1;

    // Valid/tag pipelines are reset; unconsumed data registers need no reset.
    reg [4:0] update_valid, update_right_boundary;
    reg [3:0] predict_valid;
    reg [4:0] update_index [0:4], predict_index [0:3];
    reg update_left_boundary, left_boundary1;
    wire [EVEN_BITS-1:0] wide_even;
    wire [WORD_BITS-1:0] wide_even_detail, clipped_even, clipped_odd, final_samples;
    wire predict_issue = (level != 5) && pending_valid
                       && (pending_right_boundary || update_valid[4]);

    // Fixed bit permutations implement node addresses without a carry chain.
    reg [5:0] smooth_address, detail_address;
    reg left_boundary, right_boundary;
    always @* begin
        case (level)
            0: begin
                smooth_address = {pair_index,1'b0};
                detail_address = {pair_index,1'b1};
                left_boundary = 1; right_boundary = 1;
            end
            1: begin
                smooth_address = {pair_index[4:1],1'b0,pair_index[0]};
                detail_address = {pair_index[4:1],1'b1,pair_index[0]};
                left_boundary = !pair_index[0]; right_boundary = pair_index[0];
            end
            2: begin
                smooth_address = {pair_index[4:2],1'b0,pair_index[1:0]};
                detail_address = {pair_index[4:2],1'b1,pair_index[1:0]};
                left_boundary = (pair_index[1:0] == 0); right_boundary = &pair_index[1:0];
            end
            3: begin
                smooth_address = {pair_index[4:3],1'b0,pair_index[2:0]};
                detail_address = {pair_index[4:3],1'b1,pair_index[2:0]};
                left_boundary = (pair_index[2:0] == 0); right_boundary = &pair_index[2:0];
            end
            4: begin
                smooth_address = {pair_index[4],1'b0,pair_index[3:0]};
                detail_address = {pair_index[4],1'b1,pair_index[3:0]};
                left_boundary = (pair_index[3:0] == 0); right_boundary = &pair_index[3:0];
            end
            default: begin
                smooth_address = {1'b0,pair_index};
                detail_address = {1'b1,pair_index};
                left_boundary = (pair_index == 0); right_boundary = &pair_index;
            end
        endcase
    end

    reg [5:0] a_addr0, a_addr1, b_addr0, b_addr1;
    reg a_write0, a_write1, b_write0, b_write1;
    reg [WORD_BITS-1:0] a_data0, a_data1, b_data0, b_data1;
    reg [WORD_BITS-1:0] a_read0, a_read1, b_read0, b_read1;
    wire reading0 = rst_n && (state == UPDATE);
    wire reading1 = reading0;
    always @(posedge clk) begin
        if (a_write0) bank_a[a_addr0] <= a_data0;
        if (reading0 && !source_bank) a_read0 <= bank_a[a_addr0];
    end
    always @(posedge clk) begin
        if (a_write1) bank_a[a_addr1] <= a_data1;
        if (reading1 && !source_bank) a_read1 <= bank_a[a_addr1];
    end
    always @(posedge clk) begin
        if (b_write0) bank_b[b_addr0] <= b_data0;
        if (reading0 && source_bank) b_read0 <= bank_b[b_addr0];
    end
    always @(posedge clk) begin
        if (b_write1) bank_b[b_addr1] <= b_data1;
        if (reading1 && source_bank) b_read1 <= bank_b[b_addr1];
    end
    wire [WORD_BITS-1:0] source0 = source_bank ? b_read0 : a_read0;
    wire [WORD_BITS-1:0] source1 = source_bank ? b_read1 : a_read1;

    function signed [15:0] quarter_rne;
        input signed [16:0] value;
        reg signed [16:0] quotient;
        begin
            quotient = value >>> 2;
            if ((value[1:0] == 2'b11) || ((value[1:0] == 2'b10) && value[2]))
                quotient = quotient + 17'sd1;
            quarter_rne = quotient;
        end
    endfunction

    function signed [16:0] half_rne;
        input signed [17:0] value;
        reg signed [17:0] quotient;
        begin
            quotient = value >>> 1;
            if (value[0] && value[1]) quotient = quotient + 18'sd1;
            half_rne = quotient;
        end
    endfunction

    function signed [15:0] clamp18;
        input signed [17:0] value;
        begin
            // An 18-bit value fits int16 exactly when bits 17:15 are a
            // sign extension. Overflow polarity is the original sign bit.
            if (value[17:15] == 3'b000 || value[17:15] == 3'b111)
                clamp18 = value[15:0];
            else clamp18 = value[17] ? 16'sh8000 : 16'sh7fff;
        end
    endfunction

    function signed [15:0] restore_headroom;
        input signed [16:0] value;
        begin
            // For h>=0, SAT16(SAT16(x)*2^h) == SAT16(x*2^h).
            // Compare before scaling; both thresholds are exact for h<=15.
            // The in-range shift is wiring, with no intermediate truncation.
            if (value > (17'sd32767 >>> HEADROOM_BITS)) restore_headroom = 16'sd32767;
            else if (value < (-17'sd32768 >>> HEADROOM_BITS)) restore_headroom = -16'sd32768;
            else restore_headroom = value[15:0] << HEADROOM_BITS;
        end
    endfunction


    genvar lane;
    generate for (lane = 0; lane < LANES; lane = lane + 1) begin : arithmetic
        reg signed [15:0] smooth1, detail1, left_detail1, smooth2, smooth3, update3;
        reg signed [16:0] detail_sum2, even4;
        reg signed [16:0] left_even1, right_even1, average3;
        reg signed [17:0] even_sum2, odd4;
        reg signed [15:0] odd_detail1, odd_detail2, odd_detail3;
        reg signed [15:0] even2, even3, even_for_write4;
        reg signed [15:0] detail2, detail3, detail4;
        wire signed [15:0] selected_left1 = left_boundary1 ? detail1 : left_detail1;
        wire signed [16:0] selected_right1 = right_boundary1 ? left_even1 : right_even1;

        always @(posedge clk) begin
            if (update_valid[0]) begin
                smooth1 <= source0[16*lane +: 16];
                detail1 <= source1[16*lane +: 16];
                left_detail1 <= previous_detail[16*lane +: 16];
            end
            if (update_valid[1]) begin
                detail_sum2 <= $signed({selected_left1[15],selected_left1})
                             + $signed({detail1[15],detail1});
                smooth2 <= smooth1;
                detail2 <= detail1;
            end
            if (update_valid[2]) begin
                update3 <= quarter_rne(detail_sum2);
                smooth3 <= smooth2;
                detail3 <= detail2;
            end
            if (update_valid[3]) begin
                even4 <= $signed({smooth3[15],smooth3}) - $signed({update3[15],update3});
                detail4 <= detail3;
            end

            if (predict_issue) begin
                left_even1 <= pending_even[17*lane +: 17];
                right_even1 <= wide_even[17*lane +: 17];
                odd_detail1 <= pending_detail[16*lane +: 16];
            end
            if (predict_valid[0]) begin
                even_sum2 <= $signed({left_even1[16],left_even1})
                           + $signed({selected_right1[16],selected_right1});
                even2 <= clamp18({left_even1[16],left_even1});
                odd_detail2 <= odd_detail1;
            end
            if (predict_valid[1]) begin
                average3 <= half_rne(even_sum2);
                odd_detail3 <= odd_detail2;
                even3 <= even2;
            end
            if (predict_valid[2]) begin
                odd4 <= $signed({{2{odd_detail3[15]}},odd_detail3})
                      + $signed({average3[16],average3});
                even_for_write4 <= even3;
            end
        end
        assign wide_even[17*lane +: 17] = even4;
        assign wide_even_detail[16*lane +: 16] = detail4;
        assign final_samples[16*lane +: 16] = restore_headroom(even4);
        assign clipped_even[16*lane +: 16] = even_for_write4;
        assign clipped_odd[16*lane +: 16] = clamp18(odd4);
    end endgenerate

    always @* begin
        a_addr0 = 0; a_addr1 = 0; b_addr0 = 0; b_addr1 = 0;
        a_write0 = 0; a_write1 = 0; b_write0 = 0; b_write1 = 0;
        a_data0 = 0; a_data1 = 0; b_data0 = 0; b_data1 = 0;
        if (rst_n) begin
            if (state == LOAD) begin
                a_addr0 = load_leaf; a_data0 = coeff_data; a_write0 = coeff_valid;
            end
            if (state == UPDATE) begin
                if (source_bank) begin
                    b_addr0 = smooth_address; b_addr1 = detail_address;
                end else begin
                    a_addr0 = smooth_address; a_addr1 = detail_address;
                end
            end
            if (predict_valid[3]) begin
                if (source_bank) begin
                    a_addr0 = {predict_index[3],1'b0}; a_addr1 = {predict_index[3],1'b1};
                    a_data0 = clipped_even; a_data1 = clipped_odd;
                    a_write0 = 1; a_write1 = 1;
                end else begin
                    b_addr0 = {predict_index[3],1'b0}; b_addr1 = {predict_index[3],1'b1};
                    b_data0 = clipped_even; b_data1 = clipped_odd;
                    b_write0 = 1; b_write1 = 1;
                end
            end
        end
    end

    assign coeff_ready = rst_n && (state == LOAD);
    assign audio_valid = rst_n && (state == OUTPUT);
    assign audio_data = output_words[output_index];
    assign audio_last = audio_valid && (output_index == 15);

    // Reset invalidates pipeline tags, not memories or wide data registers.
    always @(posedge clk) begin
        if (rst_n && update_valid[4] && level != 5) begin
            pending_even <= wide_even;
            pending_detail <= wide_even_detail;
            pending_index <= update_index[4];
            pending_right_boundary <= update_right_boundary[4];
        end
        if (rst_n && predict_issue) right_boundary1 <= pending_right_boundary;
        if (rst_n && update_valid[0]) begin
            previous_detail <= source1;
            left_boundary1 <= update_left_boundary;
        end
        if (rst_n && update_valid[4]) begin
            if (level == 5) begin
                if (K == 2)
                    output_words[update_index[4] >> 1][32*update_index[4][0] +: 32] <= final_samples;
                else if (K == 4 && !update_index[4][0])
                    output_words[update_index[4] >> 1] <= final_samples;
                else if (K == 8 && update_index[4][1:0] == 0)
                    output_words[{update_index[4][4:2],group_index}] <= final_samples;
            end
        end
    end

    integer stage;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= LOAD; level <= 0; pair_index <= 0; load_leaf <= 0;
            group_index <= 0; source_bank <= 0; output_index <= 0;
            update_valid <= 0; predict_valid <= 0; pending_valid <= 0;
        end else begin
            update_valid <= {update_valid[3:0],state == UPDATE};
            predict_valid <= {predict_valid[2:0],predict_issue};
            if (update_valid[4] && level != 5) pending_valid <= 1;
            else if (predict_issue) pending_valid <= 0;
            update_index[0] <= pair_index;
            predict_index[0] <= pending_index;
            update_left_boundary <= left_boundary;
            update_right_boundary <= {update_right_boundary[3:0],right_boundary};
            for (stage = 1; stage < 5; stage = stage + 1)
                update_index[stage] <= update_index[stage-1];
            for (stage = 1; stage < 4; stage = stage + 1)
                predict_index[stage] <= predict_index[stage-1];
            case (state)
                LOAD: if (coeff_valid) begin
                    if (load_leaf == 63) begin
                        load_leaf <= 0; pair_index <= 0; level <= 0;
                        source_bank <= 0; state <= UPDATE;
                    end else load_leaf <= load_leaf + 1'b1;
                end
                UPDATE: begin
                    if (pair_index == 31) state <= UPDATE_DRAIN;
                    else pair_index <= pair_index + 1'b1;
                end
                UPDATE_DRAIN: begin
                    if (level == 5 && update_valid[4] && update_index[4] == 31) begin
                        pair_index <= 0;
                        if (group_index == GROUPS-1) begin
                            output_index <= 0; state <= OUTPUT;
                        end else begin
                            group_index <= group_index + 1'b1; state <= LOAD;
                        end
                    end else if (level != 5 && predict_valid[3] && predict_index[3] == 31) begin
                        pair_index <= 0; level <= level + 1'b1;
                        source_bank <= !source_bank; state <= UPDATE;
                    end
                end
                OUTPUT: if (audio_ready) begin
                    if (output_index == 15) begin
                        output_index <= 0; group_index <= 0; state <= LOAD;
                    end else output_index <= output_index + 1'b1;
                end
                default: state <= LOAD;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (a_write0 && a_write1 && a_addr0 == a_addr1)
            $fatal(1,"Working bank A write collision");
        if (b_write0 && b_write1 && b_addr0 == b_addr1)
            $fatal(1,"Working bank B write collision");
        if (reading0 && ((!source_bank && (a_write0 || a_write1))
                       || (source_bank && (b_write0 || b_write1))))
            $fatal(1,"Working bank read/write overlap");
    end
    initial begin
        if ((K != 2 && K != 4 && K != 8) || HEADROOM_BITS < 0 || HEADROOM_BITS > 15)
            $fatal(1,"Unsupported CCS K or headroom");
    end
`endif
endmodule
