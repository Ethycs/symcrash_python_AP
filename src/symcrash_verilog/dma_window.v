/* One shared DMA budget, including queue wait and completion consumption.
 * This wrapper consumes queue completions and emits non-backpressured pulses
 * to owners. There is no new budget per descriptor. Late queued work drains;
 * not-yet-admitted descriptors receive OVERDUE without issuing RAM traffic.
 * A new window requires the old window, queue and rejected command to drain.
 */
module dma_window #(
    parameter WINDOW_CYCLES=40000
) (
    input wire clk, rst_n,
    input wire window_valid,
    output wire window_ready,
    input wire [31:0] window_tag,
    input wire seal,
    output reg window_busy,
    output wire window_accepting,
    output reg window_final_cycle,
    output reg result_valid,
    input wire result_ready,
    output reg [31:0] result_tag, elapsed_cycles,
    output reg [31:0] accepted_count, completed_count, rejected_count,
    output reg [3:0] result_status,
    output reg [8:0] outstanding_count,
    output wire deadline_missed,
    input wire submit_valid,
    output wire submit_ready,
    input wire [63:0] submit_descriptor,
    output wire initialized,
    output wire done_valid,
    output wire [63:0] done_descriptor,
    output wire [3:0] done_status,
    output wire [31:0] done_service_cycles,
    output wire ram_request_valid,
    input wire ram_request_ready,
    output wire ram_request_write,
    output wire [31:0] ram_request_address,
    output wire [127:0] ram_request_data,
    input wire ram_response_valid,
    output wire ram_response_ready,
    input wire ram_response_error,
    input wire [127:0] ram_response_data,
    output wire bram_enable, bram_write,
    output wire [15:0] bram_address,
    output wire [127:0] bram_write_data,
    input wire [127:0] bram_read_data
);
    reg sealed, reject_valid, expired;
    reg [63:0] rejected_descriptor;
    reg callback_valid, callback_queued;
    reg [63:0] callback_descriptor;
    reg [3:0] callback_status;
    reg [31:0] callback_cycles;
    wire queue_ready, queue_done;
    wire [63:0] queue_descriptor;
    wire [3:0] queue_status;
    wire [31:0] queue_cycles;
    wire [8:0] queued_count;
    wire queue_interrupt, service_overdue;
    wire start=window_valid && window_ready;
    // Idle implies zero outstanding work: reset clears both, admission requires
    // busy, and closure requires the next count to be zero. Avoid repeating
    // that nine-bit comparison in the start/reset fanout to diagnostic counters.
    assign window_ready=rst_n && initialized && !window_busy && !result_valid &&
                        !reject_valid && !callback_valid && !submit_valid;
    assign window_accepting=rst_n && window_busy && !sealed && !seal && !expired;
    assign submit_ready=rst_n && initialized && (window_accepting ? queue_ready : !reject_valid);
    // Factor the two handshake branches directly: routing an admission through
    // the public ready mux adds rejection logic to the queue/counter path.
    // queue_ready already requires reset released and initialized RUN state.
    wire admit=submit_valid && queue_ready && window_accepting;
    wire reject=submit_valid && rst_n && initialized && !reject_valid && !window_accepting;
    wire reject_done=rst_n && reject_valid && !queue_done;
    wire next_callback_valid=queue_done || reject_valid;
    wire complete=done_valid && callback_queued;
    assign done_valid=rst_n && callback_valid;
    assign done_descriptor=callback_descriptor;
    // At an event exactly WINDOW_CYCLES after start, the pre-edge elapsed
    // value is WINDOW_CYCLES-1: completion is still on time at that edge.
    wire late=!window_busy || expired;
    assign done_status=callback_status | (callback_queued && late ? 4'b1000 : 4'b0000);
    assign done_service_cycles=callback_cycles;
    assign deadline_missed=result_status[3];
    // Select independently computed +/-1 values instead of cascading two
    // adders through admission. Derive emptiness directly from current state,
    // so deadline closure does not wait for a count carry chain and reduction.
    wire [9:0] next_outstanding=(admit==complete) ? {1'b0,outstanding_count} :
                               admit ? {1'b0,outstanding_count}+10'd1 :
                                       {1'b0,outstanding_count}-10'd1;
    wire drained=!admit && (outstanding_count==0 || (outstanding_count==1 && complete));
    wire next_reject=(reject_valid && !reject_done) || reject;
    wire closing=sealed || seal || window_final_cycle;
    wire expired_pending=window_final_cycle && !drained;

    dma_priority #(.MAX_CYCLES(WINDOW_CYCLES)) queue (
        // Queue readiness is independent of submit_valid. Present only an
        // immediately accepted command, so expiry never withdraws a stalled
        // valid from that queue. The upstream transaction is instead rejected.
        .clk(clk),.rst_n(rst_n),.submit_valid(admit),
        .submit_ready(queue_ready),.submit_descriptor(submit_descriptor),
        .initialized(initialized),.queued_count(queued_count),
        .completion_valid(queue_done),.completion_ready(1'b1),
        .completion_descriptor(queue_descriptor),.completion_status(queue_status),
        .completion_cycles(queue_cycles),.completion_interrupt(queue_interrupt),
        .deadline_missed(service_overdue),.ram_request_valid(ram_request_valid),
        .ram_request_ready(ram_request_ready),.ram_request_write(ram_request_write),
        .ram_request_address(ram_request_address),.ram_request_data(ram_request_data),
        .ram_response_valid(ram_response_valid),.ram_response_ready(ram_response_ready),
        .ram_response_error(ram_response_error),.ram_response_data(ram_response_data),
        .bram_enable(bram_enable),.bram_write(bram_write),.bram_address(bram_address),
        .bram_write_data(bram_write_data),.bram_read_data(bram_read_data)
    );

    // Admission/delivery already imply an active window. Keep diagnostic
    // updates separate from the lifecycle mux, avoiding redundant busy gating.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin accepted_count<=0;completed_count<=0;rejected_count<=0;end
        else if (start) begin accepted_count<=0;completed_count<=0;rejected_count<=0;end
        else begin
            if (admit && accepted_count!=32'hffffffff) accepted_count<=accepted_count+1'b1;
            if (complete && completed_count!=32'hffffffff) completed_count<=completed_count+1'b1;
            if (window_busy && reject && rejected_count!=32'hffffffff) rejected_count<=rejected_count+1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_busy<=0;sealed<=0;result_valid<=0;result_tag<=0;elapsed_cycles<=0;
            result_status<=0;
            outstanding_count<=0;reject_valid<=0;rejected_descriptor<=0;
            window_final_cycle<=WINDOW_CYCLES==1;expired<=0;
            callback_valid<=0;callback_queued<=0;callback_descriptor<=0;callback_status<=0;callback_cycles<=0;
        end else begin
            // Isolate completion selection from owner descriptor matching.
            // The callback register is part of the shared deadline: outstanding
            // work retires only on delivery, and lateness is evaluated then.
            callback_valid<=next_callback_valid;
            if (next_callback_valid) begin
                callback_queued<=queue_done;
                callback_descriptor<=queue_done ? queue_descriptor : rejected_descriptor;
                callback_status<=queue_done ? queue_status : 4'b1000;
                callback_cycles<=queue_done ? queue_cycles : 32'd0;
            end
            if (result_valid && result_ready) result_valid<=0;
            if (reject) begin reject_valid<=1;rejected_descriptor<=submit_descriptor;end
            else if (reject_done) reject_valid<=0;
            outstanding_count<=next_outstanding[8:0];
            if (start) begin
                window_busy<=1;sealed<=0;result_tag<=window_tag;elapsed_cycles<=0;
                result_status<=0;
                window_final_cycle<=WINDOW_CYCLES==1;expired<=0;
            end else if (window_busy) begin
                // Monotonic threshold flags are prepared one edge ahead.
                // They have exactly the old >= comparisons' truth values,
                // without putting a 32-bit compare in the admission path.
                if (WINDOW_CYCLES>1 && elapsed_cycles==WINDOW_CYCLES-2) window_final_cycle<=1;
                if (window_final_cycle) expired<=1;
                if (elapsed_cycles!=32'hffffffff) elapsed_cycles<=elapsed_cycles+1'b1;
                if (done_valid || expired_pending)
                    result_status<=result_status | (done_valid ? done_status : 4'd0) |
                                   (expired_pending ? 4'b1000 : 4'b0000);
                if (closing) sealed<=1;
                if (closing && drained && !next_reject && !next_callback_valid) begin
                    window_busy<=0;result_valid<=1;
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial if (WINDOW_CYCLES<1 || WINDOW_CYCLES>2147483647) $fatal(1,"Invalid shared DMA window");
    always @(posedge clk) if (rst_n) begin
        if (!window_busy && outstanding_count!=0) $fatal(1,"Idle window retained outstanding work");
        if (next_outstanding>257) $fatal(1,"Window outstanding count overflow/underflow");
        if ((queue_done || complete) && (!window_busy || outstanding_count==0)) $fatal(1,"Completion escaped its window");
    end
`endif
endmodule
