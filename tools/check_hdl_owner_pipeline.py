"""Prove owner admission, captured descriptor matching and physical FSM encoding.

Admission and matching are all-input local/compositional proofs. Encoding
uses Yosys matched-point equivalence before/after FSM recoding; reset-driven
RTL and mapped simulations are separate checks. No board timing claim.
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
from tools.check_hdl_control import owner_module

HDL = ROOT / 'src/symcrash_verilog'
OWNER = HDL / 'coeff_buffer_owner.v'
WINDOW = HDL / 'dma_window.v'


def compact(text):
    return re.sub(r'\s+', '', text)


def admission_module(text):
    for fragment in ('wire [2:0] state [0:1];', 'reg [31:0] tag [0:1];',
                     'wire selected = !free_mask[0];',
                     'assign free_mask[bank] = state[bank] == FREE;',
                     'assign claim = prefetch_valid && prefetch_ready;',
                     'wire bank_claim=prefetch_valid && prefetch_eligible && claimable[bank];',
                     'if (bank_claim) begin\n                tag[bank] <= prefetch_tag;\n                descriptor[bank] <= incoming;'):
        assert compact(fragment) in compact(text), 'Admission wiring is outside this proof model'
    cone=re.search(r'    wire \[1:0\] tag_matches=.*?prefetch_tag\);',text,re.S).group(0)
    ready=next(line for line in text.splitlines() if 'assign prefetch_ready =' in line)
    model='''module admission_proof(input rst_n,pending_valid,prefetch_valid,
    input cancel_valid,acquire, input [31:0] cancel_tag,acquire_tag,prefetch_tag,
    input [1:0] free_mask,input [31:0] tag0,tag1,output same);
    wire [31:0] tag[0:1]; assign tag[0]=tag0;assign tag[1]=tag1;
    wire prefetch_ready;
    '''+cone+'\n'+ready+'''
    wire duplicate=(!free_mask[0] && tag0==prefetch_tag) || (!free_mask[1] && tag1==prefetch_tag);
    wire reference=rst_n && |free_mask && !pending_valid && !duplicate &&
        !(cancel_valid && cancel_tag==prefetch_tag) && !(acquire && acquire_tag==prefetch_tag);
    wire selected=!free_mask[0];
    wire [1:0] old_claim={prefetch_valid && reference && selected,prefetch_valid && reference && !selected};
    wire [1:0] new_claim=claimable & {2{prefetch_valid && prefetch_eligible}};
    assign same=(prefetch_ready==reference) && (new_claim==old_claim);
    endmodule
    '''
    return model


def prematch_module(o, w):
    admission_module(o)  # Guard the composed per-bank admission identity.
    for fragment in ('wire selected = !free_mask[0];','assign claim = prefetch_valid && prefetch_ready;',
                     'assign prefetch_ready = prefetch_eligible && |claimable;',
                     'assign free_mask[bank] = state[bank] == FREE;',
                     'if (bank_claim) begin\n                tag[bank] <= prefetch_tag;\n                descriptor[bank] <= incoming;',
                     'if (!rst_n) current_state <= FREE;\n            else current_state <= next_state;'):
        assert compact(fragment) in compact(o),fragment
    for fragment in ('wire [63:0] next_callback_descriptor=queue_done ? queue_descriptor : rejected_descriptor;',
                     'reg [7:0] callback_matches;',
                     'assign done_matches={&callback_matches[7:4],&callback_matches[3:0]};',
                     'assign done_valid=rst_n && callback_valid;',
                     'callback_valid<=next_callback_valid;\n            if (next_callback_valid) begin',
                     'callback_descriptor<=next_callback_descriptor;\n                callback_matches<=next_callback_matches;',
                     'callback_valid<=0;callback_queued<=0;callback_descriptor<=0;callback_matches<=0;'):
        assert compact(fragment) in compact(w),fragment
    for fragment in ('reg [63:0] descriptor [0:1];',
                     'assign dma_match_descriptors={descriptor[1],descriptor[0]};',
                     'wire [1:0] completed_matches=MATCHED_COMPLETION ? dma_done_matches :\n        {dma_done_descriptor==descriptor[1],dma_done_descriptor==descriptor[0]};',
                     'if (dma_done_valid && dma_done_status != 0)',
                     '(state[0] == FILLING || state[0] == DISCARD_FILL) && completed_matches[0]',
                     '(state[1] == FILLING || state[1] == DISCARD_FILL) && completed_matches[1]'):
        assert compact(fragment) in compact(o), 'Completion wiring is outside this proof model'
    assert 'dma_done_valid && completed_matches[bank]' in o
    assert '&& completed_matches[0])' in o and '&& completed_matches[1])' in o
    cone=owner_module(o)
    cone=cone.replace('module control_proof(', 'module owner_step(').replace('output same);','output same, output wire [2:0] actual_state);')
    cone=cone.replace('assign same=reference==next_state;','assign same=reference==next_state;\nassign actual_state=next_state;')
    matches=re.search(r'    wire \[7:0\] next_callback_matches=.*?;',w,re.S).group(0)
    body=cone+'''
    module prematch_proof(input rst_n,prefetch_valid,other_eligibility,bank,
    input [2:0] state0,state1,
    input [127:0] match_descriptors,
    input [63:0] callback_descriptor,next_callback_descriptor,incoming_descriptor,
    input callback_valid,next_callback_valid,input [1:0] callback_matches,
    input submit,pending_bank,expire,acquire,match_ready,release_valid,release_bank,
    input [3:0] dma_done_status,output same);
    wire [1:0] free_mask={state1==0,state0==0};
    wire selected=!free_mask[0];
    // Remaining public admission predicates are arbitrary here: this is a
    // superset of their realizable values, while preserving the free-bank gate.
    wire claim=rst_n && prefetch_valid && (|free_mask) && other_eligibility;
    wire bank_claim=claim && selected==bank;
    wire [2:0] current_state=bank ? state1 : state0;
    wire [63:0] descriptor=bank ? match_descriptors[127:64] : match_descriptors[63:0];
    wire current_match=callback_descriptor==descriptor;
    wire cached_match=callback_matches[bank];
    wire context_ok=!callback_valid || cached_match==current_match || current_state==1;
    wire [2:0] original_state,cached_state;
    wire original_policy,cached_policy;
    owner_step original(.current_state(current_state),.claim(claim),.selected(selected),.bank(bank),
        .submit(submit),.pending_bank(pending_bank),.expire(expire),.finished(callback_valid && current_match),
        .dma_done_status(dma_done_status),.acquire(acquire),.match_ready(match_ready),
        .release_valid(release_valid),.release_bank(release_bank),.same(original_policy),.actual_state(original_state));
    owner_step cached(.current_state(current_state),.claim(claim),.selected(selected),.bank(bank),
        .submit(submit),.pending_bank(pending_bank),.expire(expire),.finished(callback_valid && cached_match),
        .dma_done_status(dma_done_status),.acquire(acquire),.match_ready(match_ready),
        .release_valid(release_valid),.release_bank(release_bank),.same(cached_policy),.actual_state(cached_state));
    '''+matches+'''
    wire [1:0] reduced_next_matches={&next_callback_matches[7:4],&next_callback_matches[3:0]};
    wire [63:0] next_descriptor=bank_claim ? incoming_descriptor : descriptor;
    wire [63:0] next_returned_descriptor=!rst_n ? 64'd0 : next_callback_valid ? next_callback_descriptor : callback_descriptor;
    wire next_cached_match=rst_n && (next_callback_valid ? reduced_next_matches[bank] : cached_match);
    wire next_valid=rst_n && next_callback_valid;
    wire [2:0] next_state=rst_n ? cached_state : 3'd0;
    wire next_context=!next_valid || next_cached_match==(next_returned_descriptor==next_descriptor) || next_state==1;
    wire fault_eligible=callback_valid && dma_done_status!=0 && (current_state==2 || current_state==6);
    wire same_fault=(fault_eligible && current_match)==(fault_eligible && cached_match);
    assign same=!rst_n ? next_context : (!context_ok ||
        (original_policy && cached_policy && original_state==cached_state && same_fault && next_context &&
         (!bank_claim || current_state==0)));
    endmodule
    '''
    return body


def prove(kind, owner=OWNER, window=WINDOW, directory=DIRECTORY, tool=None):
    if kind not in ('admission', 'prematch'):
        raise ValueError('Unknown owner proof')
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {owner.name: digest(owner)}
    if kind == 'prematch':
        hashes[window.name] = digest(window)
    stem = 'owner_'+kind+'_'+digest(owner)[:12]
    if kind == 'prematch':
        stem += '_'+digest(window)[:12]
    result_path = directory / (stem+'_inventory.json')
    if result_path.exists():
        result_path.unlink()
    dependency = None
    if kind == 'prematch':
        dependency = prove('admission', owner, window, directory, tool)
    body = admission_module(owner.read_text(encoding='utf-8')) if kind == 'admission' else prematch_module(
        owner.read_text(encoding='utf-8'), window.read_text(encoding='utf-8'))
    module, script, log = (directory / (stem+suffix) for suffix in ('.v', '.ys', '.log'))
    module.write_text(body, encoding='utf-8', newline='\n')
    script.write_text(f'read_verilog "{tool.path(module)}"\nprep -top {kind}_proof -flatten\n'
                      'sat -verify -prove same 1 -show-inputs\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    if result.returncode or 'no model found: SUCCESS' not in result.stdout:
        raise RuntimeError(f'Owner {kind} proof failed; see {log}')
    report = {'kind': kind, 'result': 'PASS', 'sources': hashes,
              'runner_sha256': digest(Path(__file__)), 'control_runner_sha256': digest(ROOT/'tools/check_hdl_control.py'),
              'module_sha256': digest(module), 'script_sha256': digest(script), 'log_sha256': digest(log),
              'tool': tool.version('-V'),
              'scope': 'All-input admission equivalence' if kind == 'admission' else
                       'Qualified captured-match induction composed with proved admission; full transport is a separate gate'}
    if dependency:
        report['admission'] = dependency
    result_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def encoding_equivalence(directory, tool, gate=None):
    """Check actual before/after FSM JSON, optionally with a corrupted gate."""
    gate = gate or directory/'after.json'
    name = 'equivalence' if gate.name == 'after.json' else gate.stem+'_equivalence'
    script, log = (directory/(name+suffix) for suffix in ('.ys', '.log'))
    script.write_text(f'read_json "{tool.path(directory/"before.json")}"\nrename coeff_buffer_owner gold\n'
        f'read_json "{tool.path(gate)}"\nrename coeff_buffer_owner gate\n'
        f'equiv_make -encfile {tool.path(directory/"encoding.map")} gold gate equiv\nhierarchy -top equiv\n'
        'async2sync\ndffunmap\nopt_clean\nequiv_simple -undef -seq 2\nequiv_induct -undef -seq 4\nequiv_status -assert\n',
        encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    if result.returncode:
        raise RuntimeError(f'Owner encoding proof failed; see {log}')
    return {'script_sha256': digest(script), 'log_sha256': digest(log)}


def recode(owner=OWNER, directory=DIRECTORY, k=2, matched=1, tool=None):
    if k not in (2, 4, 8) or matched not in (0, 1):
        raise ValueError('Unsupported owner profile')
    tool = tool or Tool('yosys')
    directory = directory/f'owner_encoding_{digest(owner)[:12]}_k{k}_m{matched}'
    directory.mkdir(parents=True, exist_ok=True)
    inventory = directory/'inventory.json'
    if inventory.exists():
        inventory.unlink()
    before, recoded, after = (directory/name for name in ('before.json', 'recoded.json', 'after.json'))
    enc, script, log = (directory/name for name in ('encoding.map', 'recode.ys', 'recode.log'))
    script.write_text(f'read_verilog -sv "{tool.path(owner)}"\n'
        f'chparam -set K {k} -set MATCHED_COMPLETION {matched} coeff_buffer_owner\n'
        'hierarchy -top coeff_buffer_owner\nproc\nflatten\nopt_expr\nopt_clean\ncheck\nopt -nodffe -nosdff\n'
        f'write_json "{tool.path(before)}"\nfsm_detect\nfsm_extract\nfsm_opt\nfsm_recode -encfile {tool.path(enc)}\n'
        f'write_json "{tool.path(recoded)}"\nfsm_info\nfsm_map\nopt\nwrite_json "{tool.path(after)}"\n',
        encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    cells = json.loads(recoded.read_text())['modules']['coeff_buffer_owner']['cells']
    machines = [cell for cell in cells.values() if cell['type'] == '$fsm']
    assert len(machines) == 2
    assert all(int(c['parameters']['STATE_NUM'], 2) == 7 and int(c['parameters']['STATE_BITS'], 2) == 7 for c in machines)
    proof = encoding_equivalence(directory, tool)
    report = {'result': 'PASS', 'source_sha256': digest(owner), 'K': k, 'matched_completion': matched,
              'runner_sha256': digest(Path(__file__)), 'directory': str(directory),
              'artifacts': {p.name: digest(p) for p in (before, recoded, after, enc, script, log)},
              'proof': proof, 'tool': tool.version('-V'),
              'scope': 'Encoding-aware matched-point equivalence before/after two seven-state FSMs. RTL/reset/mapped tests are separate.'}
    inventory.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('admission', 'prematch', 'encoding', 'all'), default='all', nargs='?')
    args = parser.parse_args()
    results = []
    if args.kind in ('admission', 'prematch'):
        results.append(prove(args.kind))
    if args.kind == 'all':
        results.append(prove('prematch'))
    if args.kind in ('encoding', 'all'):
        results.extend(recode(k=k, matched=mode) for k, mode in ((2, 1), (8, 0)))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
