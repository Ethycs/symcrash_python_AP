"""All-input proofs for owner transitions and valid-qualified payload fields.

These local cones do not prove complete transport, mapped logic or timing.
RTL integration and placement have separate gates. Missing Yosys fails CLI.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import DIRECTORY, Tool, digest

HDL = ROOT / 'src/symcrash_verilog'
SOURCES = {'owner': HDL / 'coeff_buffer_owner.v',
           'payload_write': HDL / 'ccs_dma_window.v',
           'payload_read': HDL / 'ccs_stream.v'}


def owner_module(text):
    factored = 'wire [2:0] state [0:1];' in text
    if factored:
        for fragment in ('(* fsm_encoding = "one-hot" *) reg [2:0] current_state;',
                         'assign state[bank]=current_state;',
                         'if (!rst_n) current_state <= FREE;',
                         'else current_state <= next_state;'):
            assert fragment in text, 'Owner state storage is outside this proof model'
    else:
        assert 'reg [2:0] state [0:1];' in text, 'Owner state width is outside this proof model'
    start = text.index('        wire bank_claim=')
    end = text.index('        always @(posedge clk or negedge rst_n)', start)
    cone = text[start:end].replace('state[bank]', 'current_state').replace('match_ready[bank]', 'match_ready')
    if factored:
        # This proof covers the local transition policy. The independent
        # admission proof in check_hdl_owner_pipeline establishes that the
        # per-bank predicate equals the original claim/selection expression.
        predicate = 'wire bank_claim=prefetch_valid && prefetch_eligible && claimable[bank];'
        assert predicate in cone, 'Owner admission is outside this proof model'
        cone = cone.replace(predicate, 'wire bank_claim=claim && selected==bank;')
    constants = re.search(r'localparam FREE=.*?;', text, re.S).group(0)
    return '''module control_proof(
input [2:0] current_state,
input claim,selected,bank,submit,pending_bank,expire,finished,
input [3:0] dma_done_status,
input acquire,match_ready,release_valid,release_bank,
output same);
reg [2:0] reference;
'''+constants+'\n'+cone+'''
// Independent transition policy: FREE, WAITING, FILLING, READY, READING,
// DISCARD_WAIT, DISCARD_FILL = 0..6. Unused encoding 7 returns to FREE.
always @* begin
    reference=current_state;
    case (current_state)
        0: if (claim && selected==bank) reference=1;
        1: if (submit && pending_bank==bank) reference=expire ? 6 : 2;
           else if (expire) reference=5;
        2: if (finished) reference=(!expire && dma_done_status==0) ? 3 : 0;
           else if (expire) reference=6;
        3: if (expire) reference=0;
           else if (acquire && match_ready) reference=4;
        4: if (release_valid && release_bank==bank) reference=0;
        5: if (submit && pending_bank==bank) reference=6;
        6: if (finished) reference=0;
        default: reference=0;
    endcase
end
assign same=reference==next_state;
endmodule
'''


def payload_write_module(text):
    assert 'reg [15:0] payload_address;' in text and 'reg [127:0] payload_write_data;' in text, 'Payload width is outside this proof model'
    start = text.index('    always @(posedge clk or negedge rst_n) begin')
    end = text.index('    always @(posedge clk or negedge rst_n) begin', start+1)
    cone = text[start:end].replace('always @(posedge clk or negedge rst_n) begin',
        'always @* begin\n{payload_write,payload_address,payload_write_data}=held;', 1).replace('<=', '=')
    return '''module control_proof(
input rst_n,dma_bram_enable,dma_bram_write,
input [15:0] dma_bram_address,
input [127:0] dma_bram_write_data,
input [144:0] held,
output same);
reg payload_enable,payload_write;
reg [15:0] payload_address;
reg [127:0] payload_write_data;
'''+cone+'''
assign same=payload_enable==(rst_n && dma_bram_enable) &&
    (!payload_enable || {payload_write,payload_address,payload_write_data}==
                       {dma_bram_write,dma_bram_address,dma_bram_write_data});
endmodule
'''


def payload_read_module(text):
    assert 'reg [127:0] payload_read;' in text, 'Read width is outside this proof model'
    # Fail if the consumer's qualification changes; this local proof depends
    # on that exact one-clock handoff. Integration also checks real RAM data.
    assert 'read_pending<=read_payload;' in text
    assert 'if (rst_n && read_pending) words[write_pointer] <= payload_read;' in text
    cone = next(line for line in text.splitlines() if 'payload_read <=' in line)
    address = 'payload[{lease_bank,read_index[ADDRESS_BITS-1:0]}]'
    assert cone.count(address) == 1
    cone = cone.replace(address, 'memory_value').replace('<=', '=')
    return '''module control_proof(
input read_payload,
input [127:0] memory_value,held,
output same);
reg [127:0] payload_read;
always @* begin
payload_read=held;
'''+cone+'''
end
assign same=!read_payload || payload_read==memory_value;
endmodule
'''


BUILDERS = {'owner': owner_module, 'payload_write': payload_write_module,
            'payload_read': payload_read_module}


def prove(kind, source=None, directory=DIRECTORY, tool=None):
    source = source or SOURCES[kind]
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    stem = f'control_{kind}_{source_hash[:12]}'
    evidence = directory / (stem+'.json')
    if evidence.exists():
        evidence.unlink()
    module = directory / (stem+'.v')
    module.write_text(BUILDERS[kind](source.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    script = directory / (stem+'.ys')
    script.write_text(f'read_verilog "{tool.path(module)}"\nprep -top control_proof\n'
                      'sat -verify -prove same 1 -show-inputs\n', encoding='utf-8', newline='\n')
    result = tool.run(['-s', tool.path(script)])
    log = directory / (stem+'.log')
    log.write_text(result.stdout+result.stderr, encoding='utf-8')
    if result.returncode or 'SUCCESS' not in result.stdout:
        raise RuntimeError(f'{kind} control proof failed; see {log}')
    report = {'kind': kind, 'result': 'PASS', 'source_sha256': source_hash,
              'runner_sha256': digest(Path(__file__)), 'proof_module_sha256': digest(module),
              'proof_log_sha256': digest(log), 'tool': tool.version('-V'),
              'scope': 'All binary inputs to extracted local next-state cone; full integration and mapped timing are separate gates'}
    evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps([prove(kind) for kind in SOURCES], indent=2))
