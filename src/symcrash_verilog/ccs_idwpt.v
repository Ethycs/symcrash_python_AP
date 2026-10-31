/* CCS decoder: N=64, K=2/4/8, int16 storage, signed RNE and node SAT16.
 * K2: four pair positions per clock, eight-way banked working storage,
 * 99 compute / 179 no-stall transaction clocks. K4/K8 retain the serial
 * pair schedule: 247 compute clocks per group, 327/638 transaction clocks.
 * Physical parallelism changes; transform geometry and numeric reductions
 * do not. Headroom uses the proved equivalent sign-extension range test.
 * Runtime mixed-tier sharing and complete device budgets remain separate.
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
            // Scaling by 2^h fits int16 exactly when the input fits a
            // signed (16-h)-bit value: upper bits are all sign extension.
            // This implements the same two thresholds, with no carry chain.
            if ((&value[16:15-HEADROOM_BITS]) || !(|value[16:15-HEADROOM_BITS]))
                restore_headroom = value[15:0] << HEADROOM_BITS;
            else restore_headroom = value[16] ? 16'sh8000 : 16'sh7fff;
        end
    endfunction

    generate if (K == 2) begin : k2
    localparam LOAD=3'd0, UPDATE=3'd1, UPDATE_DRAIN=3'd2, OUTPUT=3'd5;
    reg [2:0] state,level,pair_group;
    reg [5:0] load_leaf;
    reg source_bank;
    reg [3:0] output_index;
    reg [5:0] update_valid;
    reg [3:0] predict_valid;
    reg [2:0] update_group [0:5],predict_group [0:3];
    reg [3:0] right_tag [0:4];
    reg [3:0] left_tag0,left_boundary1,right_boundary1;
    reg [3:0] left_boundary,right_boundary;
    reg [31:0] previous_detail;
    reg [33:0] pending_even [0:3];
    reg [31:0] pending_detail [0:3];
    reg [3:0] pending_right;
    reg [2:0] pending_group;
    reg pending_valid;
    wire predict_issue=(level!=0 && level!=5) && pending_valid && (pending_right[3] || update_valid[4]);
    wire [33:0] wide_even [0:3];
    wire [31:0] wide_detail [0:3],clipped_even [0:3],clipped_odd [0:3],final_samples [0:3];
    wire reading=rst_n && state==UPDATE;
    // Register the write enable beside its existing final result registers.
    // No arithmetic/data stage or service clock is added.
    reg node_write_valid;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) node_write_valid<=0;
        else node_write_valid<=(level==0 && update_valid[4]) ||
                              (level!=0 && level!=5 && predict_valid[2]);
    end
    wire writing=rst_n && node_write_valid;
    // Buffer accepted input once. The final leaf writes row7 while the first
    // compute group reads row0; subsequent use of row7 is seven groups later.
    // Data/address need no reset: reset clears the write authorization.
    reg load_pending;
    reg [5:0] load_address;
    reg [31:0] load_data;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) load_pending<=0;
        else load_pending<=state==LOAD && coeff_valid;
    end
    always @(posedge clk) begin
        load_address<=load_leaf;
        load_data<=coeff_data;
    end
    wire [2:0] write_group=(level==0) ? update_group[5] : predict_group[3];
    // Retimed parity tags travel with the already registered result. No extra
    // data stage: each is computed alongside its final arithmetic register.
    reg write_flip;
    always @(posedge clk) begin
        if(level==0) begin
            if(update_valid[4]) write_flip<=^update_group[4];
        end else if(predict_valid[2]) write_flip<=^predict_group[2];
    end
    wire [31:0] a_read [0:7],b_read [0:7];
    wire [31:0] source0 [0:3],source1 [0:3];

    function [5:0] address;
        input [2:0] current_level;
        input [4:0] pair_number;
        input detail;
        begin
            case(current_level)
                0: address={pair_number,detail};
                1: address={pair_number[4:1],detail,pair_number[0]};
                2: address={pair_number[4:2],detail,pair_number[1:0]};
                3: address={pair_number[4:3],detail,pair_number[2:0]};
                4: address={pair_number[4],detail,pair_number[3:0]};
                default: address={detail,pair_number};
            endcase
        end
    endfunction
    function [2:0] bank_id;
        input [5:0] location;
        begin bank_id={^location[5:2],location[1:0]}; end
    endfunction
    reg [2:0] smooth_row;
    always @* begin
        case(level)
            3: smooth_row={pair_group[2:1],1'b0};
            4: smooth_row={pair_group[2],1'b0,pair_group[1]};
            5: smooth_row={1'b0,pair_group[2:1]};
            default: smooth_row=pair_group;
        endcase
    end
    genvar bank;
     for(bank=0;bank<8;bank=bank+1) begin: storage
        reg [31:0] a [0:7],b [0:7];
        reg [31:0] ar,br;
        wire detail_row=bank[2] ^ pair_group[0] ^ (^smooth_row);
        reg [2:0] row;
        always @* begin
            case(level)
                3: row={smooth_row[2:1],detail_row};
                4: row={smooth_row[2],detail_row,smooth_row[0]};
                5: row={detail_row,smooth_row[1:0]};
                default: row=pair_group;
            endcase
        end
        wire load=rst_n && load_pending && bank_id(load_address)==bank;
        wire [1:0] pair_select={bank[2] ^ write_flip,bank[1]};
        wire [31:0] even_data=clipped_even[pair_select];
        wire [31:0] odd_data=clipped_odd[pair_select];
        wire [31:0] data=bank[0] ? odd_data : even_data;
        always @(posedge clk) begin
            if(load || (writing && source_bank)) a[load ? load_address[5:3] : write_group] <= load ? load_data : data;
            if(writing && !source_bank) b[write_group] <= data;
            if(reading && !source_bank) ar <= a[row];
            if(reading && source_bank) br <= b[row];
        end
`ifndef SYNTHESIS
        always @(posedge clk) if(rst_n) begin
            if(load && writing && source_bank) $fatal(1,"Two A writes");
            if(load && reading && !source_bank && load_address[5:3]==row)
                $fatal(1,"Input pipeline read/write collision");
        end
`endif
        assign a_read[bank]=ar;
        assign b_read[bank]=br;
    end

    // Factor the read permutation into a shared bank-role/parity selection
    // followed by fixed pair wiring for the three low-level permutations.
    wire [31:0] ordered_read [0:7];
    genvar port_index;
     for(port_index=0;port_index<8;port_index=port_index+1) begin: read_order
        wire [2:0] selected_bank={port_index[2] ^ (^update_group[0]),port_index[1:0]};
        assign ordered_read[port_index]=source_bank ? b_read[selected_bank] : a_read[selected_bank];
    end
    genvar pair;
     for(pair=0;pair<4;pair=pair+1) begin: operands
        wire [5:0] smooth=address(level,{update_group[0],pair[1:0]},1'b0);
        wire [5:0] detail=address(level,{update_group[0],pair[1:0]},1'b1);
        assign source0[pair]=(level==0) ? ordered_read[2*pair] :
                             (level==1) ? ordered_read[4*(pair/2)+(pair%2)] : ordered_read[pair];
        assign source1[pair]=(level==0) ? ordered_read[2*pair+1] :
                             (level==1) ? ordered_read[4*(pair/2)+(pair%2)+2] : ordered_read[pair+4];
        always @* begin
            case(level)
                0: begin left_boundary[pair]=1;right_boundary[pair]=1;end
                1: begin left_boundary[pair]=!pair[0];right_boundary[pair]=pair[0];end
                2: begin left_boundary[pair]=(pair==0);right_boundary[pair]=(pair==3);end
                3: begin left_boundary[pair]=!pair_group[0] && pair==0;right_boundary[pair]=pair_group[0] && pair==3;end
                4: begin left_boundary[pair]=pair_group[1:0]==0 && pair==0;right_boundary[pair]=(&pair_group[1:0]) && pair==3;end
                default: begin left_boundary[pair]=pair_group==0 && pair==0;right_boundary[pair]=(&pair_group) && pair==3;end
            endcase
        end
    end






    genvar q,lane;
     for(q=0;q<4;q=q+1) begin: pairs
        for(lane=0;lane<2;lane=lane+1) begin: arithmetic
            reg signed [15:0] smooth1,detail1,left_detail1,smooth2,smooth3,update3;
            reg signed [16:0] detail_sum2,even4,left_even1,right_even1,average3;
            reg signed [17:0] even_sum2,odd4;
            reg signed [15:0] odd_detail1,odd_detail2,odd_detail3,even2,even3,even_for_write4;
            reg signed [15:0] detail2,detail3,detail4;
            wire signed [15:0] selected_left1=left_boundary1[q] ? detail1 : left_detail1;
            wire signed [16:0] selected_right1=right_boundary1[q] ? left_even1 : right_even1;
            always @(posedge clk) begin
                if(update_valid[0]) begin
                    smooth1<=source0[q][16*lane +:16];
                    detail1<=source1[q][16*lane +:16];
                    if(q==0) left_detail1<=previous_detail[16*lane +:16];
                    else left_detail1<=source1[q-1][16*lane +:16];
                end
                if(update_valid[1]) begin
                    detail_sum2<=$signed({selected_left1[15],selected_left1})+$signed({detail1[15],detail1});
                    smooth2<=smooth1;detail2<=detail1;
                end
                if(update_valid[2]) begin
                    update3<=quarter_rne(detail_sum2);smooth3<=smooth2;detail3<=detail2;
                end
                if(update_valid[3]) begin
                    even4<=$signed({smooth3[15],smooth3})-$signed({update3[15],update3});detail4<=detail3;
                end
                if(update_valid[4] && level==0) begin
                    even_for_write4<=clamp18({even4[16],even4});
                    odd4<=$signed({{2{detail4[15]}},detail4})+$signed({even4[16],even4});
                end
                if(predict_issue) begin
                    left_even1<=pending_even[q][17*lane +:17];
                    if(q==3) right_even1<=wide_even[0][17*lane +:17];
                    else right_even1<=pending_even[q+1][17*lane +:17];
                    odd_detail1<=pending_detail[q][16*lane +:16];
                end
                if(predict_valid[0]) begin
                    even_sum2<=$signed({left_even1[16],left_even1})+$signed({selected_right1[16],selected_right1});
                    even2<=clamp18({left_even1[16],left_even1});odd_detail2<=odd_detail1;
                end
                if(predict_valid[1]) begin
                    average3<=half_rne(even_sum2);odd_detail3<=odd_detail2;even3<=even2;
                end
                if(predict_valid[2] && level!=0) begin
                    odd4<=$signed({{2{odd_detail3[15]}},odd_detail3})+$signed({average3[16],average3});
                    even_for_write4<=even3;
                end
            end
            assign wide_even[q][17*lane +:17]=even4;
            assign wide_detail[q][16*lane +:16]=detail4;
            assign final_samples[q][16*lane +:16]=restore_headroom(even4);
            assign clipped_even[q][16*lane +:16]=even_for_write4;
            assign clipped_odd[q][16*lane +:16]=clamp18(odd4);
        end
        always @(posedge clk) if(rst_n && update_valid[4] && level!=0 && level!=5) begin
            pending_even[q]<=wide_even[q];pending_detail[q]<=wide_detail[q];
        end
    end

    reg [63:0] output_lo [0:7],output_hi [0:7];
    always @(posedge clk) begin
        if(rst_n && update_valid[0]) begin previous_detail<=source1[3];left_boundary1<=left_tag0;end
        if(rst_n && update_valid[4] && level!=0 && level!=5) begin
            pending_right<=right_tag[4];pending_group<=update_group[4];
        end
        if(rst_n && predict_issue) right_boundary1<=pending_right;
        if(rst_n && update_valid[4] && level==5) begin
            output_lo[update_group[4]]<={final_samples[1],final_samples[0]};
            output_hi[update_group[4]]<={final_samples[3],final_samples[2]};
        end
    end
    assign coeff_ready=rst_n && state==LOAD;
    assign audio_valid=rst_n && state==OUTPUT;
    assign audio_last=audio_valid && output_index==15;
    assign audio_data=output_index[0] ? output_hi[output_index[3:1]] : output_lo[output_index[3:1]];
    integer stage;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state<=LOAD;level<=0;pair_group<=0;load_leaf<=0;source_bank<=0;output_index<=0;
            update_valid<=0;predict_valid<=0;pending_valid<=0;
        end else begin
            update_valid<={update_valid[4:0],state==UPDATE};
            predict_valid<={predict_valid[2:0],predict_issue};
            update_group[0]<=pair_group;predict_group[0]<=pending_group;
            left_tag0<=left_boundary;right_tag[0]<=right_boundary;
            for(stage=1;stage<6;stage=stage+1) update_group[stage]<=update_group[stage-1];
            for(stage=1;stage<5;stage=stage+1) right_tag[stage]<=right_tag[stage-1];
            for(stage=1;stage<4;stage=stage+1) predict_group[stage]<=predict_group[stage-1];
            if(update_valid[4] && level!=0 && level!=5) pending_valid<=1;
            else if(predict_issue) pending_valid<=0;
            case(state)
                LOAD: if(coeff_valid) begin
                    if(load_leaf==63) begin load_leaf<=0;pair_group<=0;level<=0;source_bank<=0;state<=UPDATE;end
                    else load_leaf<=load_leaf+1'b1;
                end
                UPDATE: if(pair_group==7) state<=UPDATE_DRAIN;
                        else pair_group<=pair_group+1'b1;
                UPDATE_DRAIN: begin
                    if(level==5 && update_valid[4] && update_group[4]==7) begin output_index<=0;state<=OUTPUT;end
                    else if((level==0 && update_valid[5] && update_group[5]==7) ||
                            (level!=0 && level!=5 && predict_valid[3] && predict_group[3]==7)) begin
                        pair_group<=0;level<=level+1'b1;source_bank<=!source_bank;state<=UPDATE;
                    end
                end
                OUTPUT: if(audio_ready) begin
                    if(output_index==15) begin output_index<=0;state<=LOAD;end
                    else output_index<=output_index+1'b1;
                end
                default: state<=LOAD;
            endcase
        end
    end
`ifndef SYNTHESIS
    integer test_pair,test_kind,test_bank,seen;
    reg [5:0] test_address;
    always @(posedge clk) if(rst_n) begin
        if(writing != (rst_n && ((level==0 && update_valid[5]) ||
            (level!=0 && level!=5 && predict_valid[3])))) $fatal(1,"Write valid retiming");
        if(writing && write_flip!=(^write_group)) $fatal(1,"Write parity tag does not match result");
        if(writing && (state==LOAD || state==OUTPUT)) $fatal(1,"Write outside compute state");
        if(reading) begin
            seen=0;
            for(test_pair=0;test_pair<4;test_pair=test_pair+1)
                for(test_kind=0;test_kind<2;test_kind=test_kind+1) begin
                    test_address=address(level,{pair_group,test_pair[1:0]},test_kind[0]);
                    test_bank=bank_id(test_address);
                    if(seen & (1<<test_bank)) $fatal(1,"Bank read collision");
                    seen=seen | (1<<test_bank);
                end
            if(seen!=255) $fatal(1,"Incomplete bank group");
        end
    end
    initial if(K!=2 || HEADROOM_BITS<0 || HEADROOM_BITS>15) $fatal(1,"Unsupported candidate profile");
`endif
    end else begin : serial
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






    genvar lane;
     for (lane = 0; lane < LANES; lane = lane + 1) begin : arithmetic
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
    end

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
    end endgenerate
endmodule
