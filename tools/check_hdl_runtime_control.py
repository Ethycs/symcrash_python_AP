"""Prove runtime command/group policy and inductive mode/group legality.

Pipeline result-valid/group signals are arbitrary inputs, so this proof does
not establish the 99-clock computation schedule or arithmetic correctness.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest, DIRECTORY
from tools.check_hdl_runtime_output import SOURCE

REGISTERS = {'state': 3, 'level': 3, 'pair_group': 3, 'load_leaf': 6,
             'source_bank': 1, 'output_index': 4, 'active_k': 2, 'slot_group': 2, 'frame_error': 1}


def control_module(text):
    start = text.index('    integer stage;')
    block = text[start:text.index('`ifndef SYNTHESIS', start)]
    for name in REGISTERS:
        pattern = r'\b'+name+r'\s*<='
        assert len(re.findall(pattern, text)) == len(re.findall(pattern, block)), 'Control written outside extracted FSM'
    reset_start = block.index('        if(!rst_n) begin')+len('        if(!rst_n) begin')
    split = block.index('        end else begin', reset_start)
    reset = block[reset_start:split]
    assert not re.sub(r'\w+\s*<=\s*[^;]+;', '', reset).strip(), 'Conditional reset outside proof model'
    case_start = block.index('            case(state)', split)
    case_end = block.index('            endcase', case_start)+len('            endcase')
    prefix = block[split+len('        end else begin'):case_start]
    assert prefix.lstrip().startswith('frame_error<=0;'), 'Error pulse clear must be unconditional'
    assert re.sub(r'\s+', '', block[case_end:]) == 'endend', 'Unmodelled controller suffix'
    assigned = set(re.findall(r'(\w+)(?:\[[^\]]+\])?\s*<=', prefix))
    assert assigned <= {'frame_error', 'update_valid', 'predict_valid', 'update_group', 'predict_group',
                        'left_tag0', 'right_tag', 'pending_valid'}, 'Unmodelled control update outside FSM'
    assert re.findall(r'frame_error\s*<=\s*([^;]+);', prefix) == ['0']
    case = block[case_start:case_end]
    assert set(re.findall(r'(\w+)\s*<=', case)) <= REGISTERS.keys()
    actual_reset = []
    for name, value in re.findall(r'(\w+)\s*<=\s*([^;]+);', reset):
        if name in REGISTERS:
            actual_reset.append(f'next_{name}={value};')
    assert len(actual_reset) == len(REGISTERS)
    case = re.sub(r'(\w+)\s*<=', r'next_\1=', case)
    parameters = re.search(r'    localparam IDLE=.*?;', text).group(0)
    final_group = re.search(r'    wire \[1:0\] final_group=.*?;', text).group(0)
    handshakes = []
    for name in ('frame_ready', 'coeff_ready', 'audio_valid'):
        assignments = re.findall(r'    assign '+name+r'=.*?;', text)
        assert len(assignments) == 1, 'Unmodelled handshake drivers'
        handshakes.append(assignments[0])
    handshakes = '\n'.join(handshakes)
    inputs = ',\n'.join(f'input [{width-1}:0] {name}' for name, width in REGISTERS.items())
    declarations = '\n'.join(f'reg [{width-1}:0] next_{name};' for name, width in REGISTERS.items())
    defaults = '\n'.join(f'next_{name}={name};' for name in REGISTERS)
    return '''module runtime_control_proof(input rst_n,frame_valid,coeff_valid,audio_ready,
        input [1:0] frame_k,input [5:0] update_valid,input [3:0] predict_valid,
        input [2:0] ugroup4,ugroup5,pgroup3,
    '''+inputs+''',output same);
        wire [2:0] update_group[0:5],predict_group[0:3];
        assign update_group[4]=ugroup4;assign update_group[5]=ugroup5;assign predict_group[3]=pgroup3;
        wire frame_ready,coeff_ready,audio_valid;
    '''+parameters+final_group+handshakes+declarations+'''
        always @* begin
    '''+defaults+'''
            if(!rst_n) begin
    '''+''.join(actual_reset)+'''
            end else begin
                next_frame_error=0;
    '''+case+'''
            end
        end
        wire context_ok=(state==0 || state==1 || state==2 || state==5 || state==6) &&
                        active_k<3 && slot_group<(1<<active_k);
        wire next_legal=(next_state==0 || next_state==1 || next_state==2 || next_state==5 || next_state==6) &&
                        next_active_k<3 && next_slot_group<(1<<next_active_k);
        wire command=state==6 && frame_valid;
        wire accept=command && frame_k<3;
        wire finish_group=state==2 && level==5 && update_valid[4] && ugroup4==7;
        wire last_group=slot_group==((1<<active_k)-1);
        wire [1:0] expected_mode=accept ? frame_k : active_k;
        wire [1:0] expected_group=accept ? 2'd0 : finish_group && !last_group ? slot_group+2'd1 : slot_group;
        wire policy=next_active_k==expected_mode && next_slot_group==expected_group &&
            next_frame_error==(command && frame_k==3) &&
            (!accept || (next_state==0 && next_load_leaf==0)) &&
            (!(command && frame_k==3) || next_state==6) &&
            (!finish_group || next_state==(last_group ? 5 : 0)) &&
            (frame_ready==(rst_n && state==6)) && (coeff_ready==(rst_n && state==0)) &&
            (audio_valid==(rst_n && state==5));
        wire reset_ok=next_state==6 && next_active_k==0 && next_slot_group==0 &&
            next_legal && !next_frame_error && !frame_ready && !coeff_ready && !audio_valid;
        assign same=!rst_n ? reset_ok : (!context_ok || (next_legal && policy));
    endmodule
    '''


def prove(source=SOURCE, directory=DIRECTORY, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    key = digest(source);stem = f'runtime_control_{key[:12]}'
    paths = {s: directory/(stem+s) for s in ('.v', '.ys', '.log', '_inventory.json')}
    if paths['_inventory.json'].exists():
        paths['_inventory.json'].unlink()
    paths['.v'].write_text(control_module(source.read_text()), encoding='utf-8', newline='\n')
    paths['.ys'].write_text(f'read_verilog -sv "{tool.path(paths[".v"])}"\n'
        'prep -top runtime_control_proof -flatten\nsat -verify -prove same 1 -show-inputs\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(paths['.ys'])], paths['.log'])
    if result.returncode or 'no model found: SUCCESS' not in result.stdout:
        raise RuntimeError(f'Runtime control proof failed; see {paths[".log"]}')
    assert digest(source) == key
    report = {'result': 'PASS', 'source_sha256': key, 'runner_sha256': digest(Path(__file__)),
        'tool': tool.version('-V'), 'artifacts': {s: digest(paths[s]) for s in ('.v', '.ys', '.log')},
        'scope': 'Source-extracted command/group FSM: reset base, legal-mode/group invariant and command/group policy. Arbitrary pipeline-valid/group inputs; excludes 99-clock schedule and arithmetic.'}
    paths['_inventory.json'].write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(prove(args.source, args.directory), indent=2))
