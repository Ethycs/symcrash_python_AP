"""Prove native read-only/writable cache ownership, query and metadata control.

Payload RAM is cut after guarding its physical port bindings. Both clients
remain arbitrary inputs; no client nonoverlap assumption is introduced.
Use --source to qualify a candidate before adopting its interface.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest
from tools.check_hdl_voice_cache_control import control_model as base_control_model

def query_control_model(text):
    text = base_control_model(text)
    at = text.index('    assign formal_safe=')
    ghost = '''
    // Independent original full-width lookup, sampled at acceptance.
    wire expected_input_hit=valid_mask[acquire_slot] && !dirty_mask[acquire_slot] &&
        !(dma_busy && dma_slot==acquire_slot) &&
        voice_id[acquire_slot]==acquire_voice_id && context_tag[acquire_slot]==acquire_tag;
    reg expected_pending,expected_result_valid,expected_hit,expected_result_hit;
    reg [3:0] expected_slot;
    reg [15:0] expected_voice;
    reg [31:0] expected_tag;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) begin expected_pending<=0;expected_result_valid<=0;expected_result_hit<=0;end
        else begin
            expected_pending<=acquire;
            expected_result_valid<=expected_pending;
            if(expected_pending) expected_result_hit<=expected_hit;
        end
    always @(posedge clk) if(acquire) begin
        expected_hit<=expected_input_hit;expected_slot<=acquire_slot;
        expected_voice<=acquire_voice_id;expected_tag<=acquire_tag;
    end
    wire query_safe=!rst_n || (
        query_pending==expected_pending && acquire_result_valid==expected_result_valid &&
        (!acquire_result_valid || (acquire_hit==expected_result_hit && lease_valid==expected_result_hit)) &&
        (!query_pending || (!lease_valid && !acquire_ready && lease_slot==expected_slot &&
            query_matches==((16'b1<<expected_slot) & {16{expected_hit}}) &&
            !(prefetch && prefetch_slot==expected_slot))) &&
        (!(query_pending && query_hit) || (valid_mask[lease_slot] && !dirty_mask[lease_slot] &&
            !(dma_busy && dma_slot==lease_slot) && voice_id[lease_slot]==expected_voice &&
            context_tag[lease_slot]==expected_tag)));
'''
    text = text[:at] + ghost + text[at:]
    return text.replace('assign formal_safe=!rst_n || (', 'assign formal_safe=query_safe && (!rst_n || (', 1).replace(
        '(!(dma_access && dsp_accept) || dma_slot!=lease_slot));',
        '(!(dma_access && dsp_accept) || dma_slot!=lease_slot)));', 1)


def metadata_control_model(text):
    text = query_control_model(text)
    at = text.index('    assign formal_safe=')
    ghost = '''
    reg [15:0] expected_metadata_pending;
    reg [15:0] accepted_metadata_voice;
    reg [31:0] accepted_metadata_tag,accepted_metadata_ram;
    reg [15:0] accepted_voice[0:15];
    reg [31:0] accepted_tag[0:15],accepted_ram[0:15];
    wire [15:0] metadata_bank_safe;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) expected_metadata_pending<=0;
        else expected_metadata_pending<=prefetch ? (16'b1<<prefetch_slot) : 16'd0;
    always @(posedge clk) if(prefetch) begin
        accepted_metadata_voice<=prefetch_voice_id;
        accepted_metadata_tag<=prefetch_tag;accepted_metadata_ram<=prefetch_ram_address;
    end
    for(genvar page=0;page<16;page=page+1) begin: expected_metadata
        always @(posedge clk) if(prefetch && prefetch_slot==page) begin
            accepted_voice[page]<=prefetch_voice_id;accepted_tag[page]<=prefetch_tag;
            accepted_ram[page]<=prefetch_ram_address;
        end
        assign metadata_bank_safe[page]=metadata_pending[page] ||
            !(valid_mask[page] || (dma_busy && dma_slot==page)) ||
            (voice_id[page]==accepted_voice[page] && context_tag[page]==accepted_tag[page] &&
                ram_address[page]==accepted_ram[page]);
    end
    wire metadata_safe=!rst_n || (
        metadata_pending==expected_metadata_pending && (&metadata_bank_safe) &&
        (metadata_pending==0 || (dma_state==SUBMIT && !dma_store &&
            metadata_pending==(16'b1<<dma_slot) && metadata_voice==accepted_metadata_voice &&
            metadata_tag==accepted_metadata_tag && metadata_ram==accepted_metadata_ram)));
'''
    text = text[:at] + ghost + text[at:]
    return text.replace('assign formal_safe=query_safe &&', 'assign formal_safe=metadata_safe && query_safe &&', 1)


def control_model(text):
    # Bind the control write predicate to the actual physical RAM port before
    # cutting data storage. The read-only input has no separate write path.
    assert text.count('if (memory_write) pages[{lease_slot,memory_word}]<=dsp_write_data;') == 1
    assert text.count('if (dsp_accept && !memory_write) dsp_read_data<=pages[{lease_slot,memory_word}];') == 1
    text=metadata_control_model(text)
    at=text.index('    assign formal_safe=')
    ghost='''
    reg expected_readonly,expected_response;
    wire accepted_state=dsp_valid && dsp_ready;
    wire accepted_read=read_valid && read_ready;
    wire consumed_response=(dsp_response_valid && dsp_response_ready) ||
        (read_response_valid && read_response_ready);
    always @(posedge clk or negedge rst_n)
        if(!rst_n) begin expected_readonly<=0;expected_response<=0;end
        else begin
            if(acquire) expected_readonly<=acquire_readonly;
            if(accepted_state || accepted_read) expected_response<=1;
            else if(consumed_response) expected_response<=0;
        end
    wire ports_safe=!rst_n || (
        lease_readonly==expected_readonly && response_pending==expected_response &&
        (!response_pending || lease_valid) &&
        (!lease_readonly || (!lease_dirty && !memory_write)) &&
        !(state_accept && read_accept) &&
        (lease_valid || (!dsp_ready && !read_ready && !release_ready && !read_release_ready)) &&
        (!lease_readonly || (!dsp_ready && !release_ready && !dsp_response_valid)) &&
        (lease_readonly || (!read_ready && !read_release_ready && !read_response_valid)) &&
        dsp_response_valid==(response_pending && !lease_readonly) &&
        read_response_valid==(response_pending && lease_readonly) &&
        (!(release_dirty || release_waiting) || !lease_readonly) &&
        (!read_release || !release_dirty) && memory_write==(accepted_state && dsp_write));
'''
    text=text[:at]+ghost+text[at:]
    return text.replace('assign formal_safe=metadata_safe &&','assign formal_safe=ports_safe && metadata_safe &&',1)


def prove(text, directory, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    module, script, log = (directory / ('control' + suffix) for suffix in ('.v', '.ys', '.log'))
    module.write_text(control_model(text), encoding='utf-8', newline='\n')
    script.write_text(f'read_verilog -sv "{tool.path(module)}"\nprep -top voice_state_cache -flatten\n'
        'memory_map\nasync2sync\ndffunmap\nopt_clean\n'
        'sat -verify -tempinduct -seq 2 -maxsteps 6 -set rst_n 1 -set-at 1 rst_n 0 -prove formal_safe 1 -show-ports\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    if result.returncode or 'Induction step proven: SUCCESS' not in result.stdout:
        raise RuntimeError(f'Native cache ownership proof failed: {log}')
    return {'result': 'PASS', 'module_sha256': digest(module), 'script_sha256': digest(script), 'log_sha256': digest(log)}


def check(source, directory):
    key = digest(source)
    row = prove(source.read_text(encoding='utf-8'), directory)
    assert digest(source) == key
    report = {'source_sha256': key, 'runner_sha256': digest(Path(__file__)),
        'base_model_sha256': digest(ROOT / 'tools/check_hdl_voice_cache_control.py'), 'control': row,
        'scope': 'Reset base plus induction of actual control. Proves full captured query/metadata identity, native client ownership, owner-only response consumption/release, readonly cleanliness and DMA/page invariants. Payload cut; no arithmetic, mapped equivalence or board claim.'}
    (directory / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=ROOT / 'build/hdl-tests/voice_cache_ports')
    args = parser.parse_args()
    print(json.dumps(check(args.source, args.directory)), flush=True)
