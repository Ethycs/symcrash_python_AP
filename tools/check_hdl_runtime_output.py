"""Prove runtime output placement from the actual bank-write/read expressions.

This is an all-input combinational step proof, conditional on a legal latched
mode/group. Reset reachability and pipeline timing are separate simulation gates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest, DIRECTORY

SOURCE = ROOT/'src/symcrash_verilog/ccs_idwpt_runtime.v'
STATIC = SOURCE.with_name('ccs_idwpt.v')


def preserved_datapath(text, static):
    """Bind reused arithmetic and working-bank implementation to qualified K2."""
    def section(value, first, last):
        return value[value.index(first):value.index(last, value.index(first))].strip()
    regions = (
        ('    function signed', '    localparam IDLE=', '    generate if (K == 2)'),
        ('    reg [5:0] load_leaf;', '    // Four fixed 32-bit banks', '    reg [63:0] output_lo'),
    )
    for start, runtime_end, static_end in regions:
        assert section(text, start, runtime_end) == section(static, start, static_end), 'Reused datapath changed'


def output_module(text):
    start = text.index('    wire [31:0] output_sample [0:3];')
    end = text.index('    always @(posedge clk) begin', start)
    block = text[start:end]
    assert block.count('reg [31:0] words [0:7];') == 1
    block = block.replace('reg [31:0] words [0:7];', '''wire [31:0] words [0:7];
        for(genvar row=0;row<8;row=row+1) begin
            assign words[row]=before_bits[32*(4*row+output_part) +:32];
        end''')
    write = re.search(r'always @\(posedge clk\)\s*if\((.*?)\) words\[(.*?)\]<=(.*?);', block, re.S)
    assert write is not None and block.count('always') == 1, 'Unmodelled output write port'
    enable, address, data = write.groups()
    block = block[:write.start()]+f'''for(genvar row_next=0;row_next<8;row_next=row_next+1) begin
            assign after_bits[32*(4*row_next+output_part) +:32]=
                ({enable}) && ({address})==row_next ? ({data}) : words[row_next];
        end'''+block[write.end():]
    read = re.search(r'    assign audio_data=.*?;', text).group(0)
    references = []
    coverage = {}
    for mode, k in enumerate((2, 4, 8)):
        # Normative root coordinates: j=4*g+q, keep 2*j divisible by K,
        # then emit slots 2*s,2*s+1 at samples 2*j+2*s and 2*j+2*s+1.
        # 8*g is divisible by all legal K, so q alone selects retained pairs.
        statements = [f'reference_bits[32*(4*root_group+{q}+slot_group) +:32]=final_samples[{q}];'
                      for q in range(4) if (2*q) % k == 0]
        references.append(f'{mode}: begin '+''.join(statements)+' end')
        counts = Counter(2*(4*g+q)+2*s+lane for g in range(8) for q in range(4)
                         for s in range(k//2) for lane in range(2) if (2*(4*g+q)) % k == 0)
        assert counts == Counter(range(64)), 'Missing or multiply written output sample'
        coverage[k] = {'samples': len(counts), 'writes_per_sample': 1}
    module = '''module runtime_output_proof(input rst_n,input [1:0] active_k,slot_group,
        input root_valid,input [2:0] level,root_group,input [3:0] output_index,
        input [127:0] samples,input [1023:0] before_bits,output same);
        wire [31:0] final_samples[0:3];
        for(genvar q=0;q<4;q=q+1) assign final_samples[q]=samples[32*q +:32];
        wire [5:0] update_valid={1'b0,root_valid,4'b0};
        wire [2:0] update_group[0:5]; assign update_group[4]=root_group;
        wire [1023:0] after_bits;
        wire [63:0] audio_data;
    '''+block+read+'''
        reg [1023:0] reference_bits;
        always @* begin
            reference_bits=before_bits;
            if(rst_n && root_valid && level==5) begin
                case(active_k)
    '''+'\n'.join(references)+'''
                endcase
            end
        end
        wire legal=active_k<3 && slot_group<(1<<active_k);
        assign same=!legal || (after_bits==reference_bits && audio_data==before_bits[64*output_index +:64]);
    endmodule
    '''
    return module, coverage


def prove(source=SOURCE, directory=DIRECTORY, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    key = digest(source)
    preserved_datapath(source.read_text(), STATIC.read_text())
    module, coverage = output_module(source.read_text())
    stem = f'runtime_output_{key[:12]}'
    paths = {suffix: directory/(stem+suffix) for suffix in ('.v', '.ys', '.log', '_inventory.json')}
    if paths['_inventory.json'].exists():
        paths['_inventory.json'].unlink()
    paths['.v'].write_text(module, encoding='utf-8', newline='\n')
    paths['.ys'].write_text(f'read_verilog -sv "{tool.path(paths[".v"])}"\n'
        'prep -top runtime_output_proof -flatten\n'
        'sat -verify -prove same 1 -show-inputs\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(paths['.ys'])], paths['.log'])
    if result.returncode or 'no model found: SUCCESS' not in result.stdout:
        raise RuntimeError(f'Runtime output proof failed; see {paths[".log"]}')
    assert digest(source) == key
    report = {'result': 'PASS', 'source_sha256': key, 'static_source_sha256': digest(STATIC),
        'runner_sha256': digest(Path(__file__)), 'coverage': coverage, 'tool': tool.version('-V'),
        'artifacts': {s: digest(paths[s]) for s in ('.v', '.ys', '.log')},
        'scope': 'Actual output bank next-state/read expressions vs canonical CCS coordinates, all inputs with legal latched mode/group. Arithmetic and working banks byte-matched to fixed-K2 source. Reset reachability/pipeline timing separately tested.'}
    paths['_inventory.json'].write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(prove(args.source, args.directory), indent=2))
