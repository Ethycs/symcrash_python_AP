"""Prove voice-cache ownership and bounded control after reset.

Only payload storage is cut. Its data cannot affect the control being proved;
metadata, arbitration, address checks, status and all counters remain RTL.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest

SOURCE = ROOT / 'src/symcrash_verilog/voice_state_cache.v'
DIRECTORY = ROOT / 'build/hdl-tests/voice_cache_control'


def control_model(text):
    start = text.index('    // The two true memory ports')
    end = text.index('    genvar slot;', start)
    text = text[:start] + "    assign dsp_response_data=128'd0;\n    assign dma_bram_read_data=128'd0;\n" + text[end:]
    text = text.replace('output reg [127:0] dma_bram_read_data', 'output wire [127:0] dma_bram_read_data')
    text = text[:text.index('`ifndef SYNTHESIS')]
    text = text.replace('output reg memory_fault\n', 'output reg memory_fault,\n    output wire formal_safe\n', 1)
    return text + '''
    assign formal_safe=!rst_n || (
        dma_state<=PUBLISH && transferred_words<=128 && next_word<=128 && transferred_words<=next_word &&
        (dirty_mask & ~valid_mask)==0 &&
        (!lease_valid || (valid_mask[lease_slot] && !(dma_busy && dma_slot==lease_slot))) &&
        (!lease_valid || lease_dirty==dirty_mask[lease_slot]) &&
        (!(dma_busy && dma_store) || (valid_mask[dma_slot] && dirty_mask[dma_slot])) &&
        (!(dma_busy && !dma_store) || (!valid_mask[dma_slot] && !dirty_mask[dma_slot])) &&
        (!dsp_response_valid || lease_valid) &&
        claim_mask==((16'b1<<prefetch_slot) & {16{prefetch}}) &&
        (!(dma_access && dsp_accept) || dma_slot!=lease_slot));
endmodule
'''


def prove(text, directory=None, tool=None):
    directory, tool = directory or DIRECTORY, tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    module, script, log = (directory / ('control' + suffix) for suffix in ('.v', '.ys', '.log'))
    model = control_model
    if 'acquire_readonly' in text:
        # The pipelined native interface needs the captured query/metadata
        # invariants to close induction. Keep control_model as the payload-cut
        # base used by that stronger model; dispatch only at this entry point.
        from tools.check_hdl_voice_cache_ports import control_model as model
    module.write_text(model(text), encoding='utf-8', newline='\n')
    script.write_text(f'read_verilog -sv "{tool.path(module)}"\nprep -top voice_state_cache -flatten\n'
        'memory_map\nasync2sync\ndffunmap\nopt_clean\n'
        'sat -verify -tempinduct -seq 2 -maxsteps 4 -set rst_n 1 -set-at 1 rst_n 0 -prove formal_safe 1 -show-ports\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    if result.returncode or 'Induction step proven: SUCCESS' not in result.stdout:
        raise RuntimeError(f'Voice cache control proof failed: {log}')
    return {'result': 'PASS', 'module_sha256': digest(module), 'script_sha256': digest(script), 'log_sha256': digest(log)}


def check():
    key = digest(SOURCE)
    row = prove(SOURCE.read_text(encoding='utf-8'))
    assert digest(SOURCE) == key
    report = {'source_sha256': key, 'runner_sha256': digest(Path(__file__)), 'control': row,
        'scope': 'Reset base plus temporal induction for actual control, with payload cut. No payload or mapped equivalence claim.'}
    if 'acquire_readonly' in SOURCE.read_text(encoding='utf-8'):
        report['ports_model_sha256'] = digest(ROOT / 'tools/check_hdl_voice_cache_ports.py')
    (DIRECTORY / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(check()), flush=True)
