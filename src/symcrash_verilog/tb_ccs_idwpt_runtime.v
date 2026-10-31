`timescale 1ns/1ps
module tb_ccs_idwpt_runtime;
    parameter HEADROOM_BITS=2, CASES=1, STALLS=0;
    reg clk=0;
    always #5 clk=!clk;
    reg rst_n=0,frame_valid=0,coeff_valid=0,audio_ready=0;
    reg [1:0] frame_k=0;
    reg [31:0] coeff_data=0;
    wire frame_ready,frame_error,coeff_ready,audio_valid,audio_last;
    wire [63:0] audio_data;
    ccs_idwpt_runtime #(.HEADROOM_BITS(HEADROOM_BITS)) dut(.*);
    reg [1:0] modes[0:CASES-1];
    reg [15:0] coefficients[0:CASES*512-1],expected[0:CASES*64-1];
    reg [4095:0] mode_file,coeff_file,expected_file;
    integer case_index,slots,groups,cycles,input_beat,output_beat,lane,index;
    integer group_last,leaf,slot_group,reset_count=0,rejections=0,completed_groups=0;
    integer min_cost[0:2],max_cost[0:2],mode_count[0:2];
    reg pending,awaiting_group,was_stalled;
    reg [63:0] held_data;
    reg held_last;

    task reset_device;
        begin
            @(negedge clk);rst_n=0;frame_valid=0;coeff_valid=0;audio_ready=0;
            #1;
            if(frame_ready || coeff_ready || audio_valid) $fatal(1,"Reset did not suppress handshakes");
            repeat(2) @(posedge clk);
            @(negedge clk);rst_n=1;
            repeat(3) begin
                @(posedge clk);#1;
                if(!frame_ready || coeff_ready || audio_valid || frame_error) $fatal(1,"Reset did not return idle");
            end
            reset_count=reset_count+1;
        end
    endtask

    task command;
        input [1:0] mode;
        begin
            @(negedge clk);frame_valid=1;frame_k=mode;
            // No coefficient can be accepted before a valid frame command.
            coeff_valid=1;coeff_data=32'h7fff8000;audio_ready=0;
            @(posedge clk);
            if(!frame_ready || coeff_ready || audio_valid) $fatal(1,"Frame command not isolated from data");
            #1;
            if(mode==3) begin
                if(!frame_error || !frame_ready || coeff_ready || audio_valid) $fatal(1,"Reserved mode was not rejected");
                rejections=rejections+1;
            end else if(frame_error || frame_ready || !coeff_ready) $fatal(1,"Frame command was not latched");
        end
    endtask

    task reject_reserved;
        begin
            command(3);
            @(negedge clk);frame_valid=0;coeff_valid=0;
            @(posedge clk);#1;
            if(frame_error || !frame_ready || audio_valid || coeff_ready) $fatal(1,"Rejected command left work active");
        end
    endtask

    task reset_probe;
        input integer kind;
        input [1:0] mode;
        integer accepted,target,delay_cycles,probe_groups,timeout;
        begin
            probe_groups=1<<mode;
            command(mode);
            target=(kind==0) ? 17 : (kind==15) ? 64*probe_groups :
                   (kind==16) ? 64*(probe_groups-1)+17 : 64;
            accepted=0;timeout=0;
            while(accepted<target) begin
                @(negedge clk);frame_valid=0;frame_k=3;coeff_valid=1;coeff_data=32'h80017fff;
                @(posedge clk);
                if(coeff_ready) accepted=accepted+1;
                timeout=timeout+1;
                if(timeout>1000) $fatal(1,"Reset probe input stalled");
            end
            @(negedge clk);coeff_valid=0;
            case(kind)
                0,16: delay_cycles=0;
                11: delay_cycles=9;
                12: delay_cycles=11;
                13: delay_cycles=13;
                14: delay_cycles=31;
                15: delay_cycles=99;
                17: delay_cycles=88;
                default: delay_cycles=kind;
            endcase
            repeat(delay_cycles) @(posedge clk);
            #1;
            if(kind==15 && !audio_valid) $fatal(1,"Reset probe missed stalled output");
            if(kind==16 && dut.slot_group!=probe_groups-1) $fatal(1,"Reset probe missed later slot group");
            if(kind==17 && dut.level!=5) $fatal(1,"Reset probe missed root level");
            reset_device;
        end
    endtask

    initial begin
        if(!$value$plusargs("modes=%s",mode_file) || !$value$plusargs("coeffs=%s",coeff_file) ||
           !$value$plusargs("expected=%s",expected_file)) $fatal(1,"Missing vector files");
        $readmemh(mode_file,modes);$readmemh(coeff_file,coefficients);$readmemh(expected_file,expected);
        for(index=0;index<3;index=index+1) begin min_cost[index]=2147483647;max_cost[index]=0;mode_count[index]=0;end
        reset_device;
        for(case_index=0;case_index<CASES;case_index=case_index+1) begin
            // All 18 reset positions at all three modes, each followed by a
            // fresh checked frame (which may use a different mode).
            if(case_index<54) reset_probe(case_index/3,case_index%3);
            if(case_index%7==0) reject_reserved;
            slots=2<<modes[case_index];groups=slots/2;
            command(modes[case_index]);
            cycles=1;input_beat=0;output_beat=0;pending=0;awaiting_group=0;was_stalled=0;
            while(output_beat<16) begin
                @(negedge clk);frame_valid=0;frame_k=cycles%4;
                if(!pending && input_beat<64*groups && (!STALLS || cycles%7!=0)) pending=1;
                coeff_valid=pending;
                if(input_beat==64*groups) begin coeff_valid=1;coeff_data=32'h7fff7fff;end
                else begin
                    leaf=input_beat%64;slot_group=input_beat/64;
                    for(lane=0;lane<2;lane=lane+1)
                        coeff_data[16*lane +:16]=coefficients[case_index*512+leaf*8+slot_group*2+lane];
                end
                audio_ready=!STALLS || cycles%11<7;
                @(posedge clk);
                if(frame_ready || frame_error) $fatal(1,"Busy decoder accepted/reinterpreted a new mode");
                if(coeff_valid && coeff_ready) begin
                    if(input_beat==64*groups) $fatal(1,"Accepted poison after final group");
                    input_beat=input_beat+1;pending=0;
                    if(input_beat%64==0) begin group_last=cycles;awaiting_group=1;end
                end
                if(was_stalled && (!audio_valid || audio_data!==held_data || audio_last!==held_last))
                    $fatal(1,"Output changed under backpressure");
                was_stalled=audio_valid && !audio_ready;held_data=audio_data;held_last=audio_last;
                if(audio_valid) begin
                    if(coeff_ready || input_beat!=64*groups) $fatal(1,"Output escaped an incomplete frame");
                    if(audio_last!==(output_beat==15)) $fatal(1,"Wrong audio_last");
                    for(lane=0;lane<4;lane=lane+1) begin
                        index=case_index*64+4*output_beat+lane;
                        if(audio_data[16*lane +:16]!==expected[index])
                            $fatal(1,"Runtime mismatch case=%0d K=%0d sample=%0d got=%0d expected=%0d",
                                   case_index,slots,4*output_beat+lane,$signed(audio_data[16*lane +:16]),$signed(expected[index]));
                    end
                    if(audio_ready) output_beat=output_beat+1;
                end
                #1;
                if(awaiting_group && (coeff_ready || audio_valid)) begin
                    if(cycles-group_last!=99) $fatal(1,"Wrong group compute latency: %0d",cycles-group_last);
                    awaiting_group=0;completed_groups=completed_groups+1;
                end
                cycles=cycles+1;
                if(cycles>3*(1+163*groups+16)) $fatal(1,"Runtime decoder timeout");
            end
            if(!STALLS && cycles!=1+163*groups+16) $fatal(1,"Wrong runtime transaction cost: %0d",cycles);
            if(audio_valid || coeff_ready || !frame_ready) $fatal(1,"Frame did not drain to command idle");
            if(cycles<min_cost[modes[case_index]]) min_cost[modes[case_index]]=cycles;
            if(cycles>max_cost[modes[case_index]]) max_cost[modes[case_index]]=cycles;
            mode_count[modes[case_index]]=mode_count[modes[case_index]]+1;
        end
        $display("PASS runtime cases=%0d headroom=%0d stalls=%0d samples=%0d groups=%0d resets=%0d rejected=%0d K2=%0d:%0d..%0d K4=%0d:%0d..%0d K8=%0d:%0d..%0d",
                 CASES,HEADROOM_BITS,STALLS,CASES*64,completed_groups,reset_count,rejections,
                 mode_count[0],min_cost[0],max_cost[0],mode_count[1],min_cost[1],max_cost[1],mode_count[2],min_cost[2],max_cost[2]);
        $finish;
    end
    initial begin #(CASES*100000+1000000);$fatal(1,"Runtime simulation timeout");end
endmodule
