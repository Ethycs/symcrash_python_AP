"""Check actual runtime-owner encoding, mapped FIFO and mapped-owner integration.

The full decoder/window stay RTL in the owner integration simulations. Board
timing and complete mapped transport equivalence remain separate gates.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.build_hdl_dma import Tool,digest
from tools.hdl_mapping import PRELUDE,evidence
from tools.check_hdl_idwpt import Simulator
from tools import check_hdl_fifo as fifo
from tools import check_hdl_runtime_transport as check

def check_encoding():
    root=ROOT;base=root/'build/hdl-tests/runtime_transport_mapping/encoding';base.mkdir(parents=True,exist_ok=True)
    source=root/'src/symcrash_verilog/coeff_buffer_owner_runtime.v';tool=Tool('yosys');rows=[]
    top='coeff_buffer_owner_runtime'
    def equivalence(d,gate):
        stem=gate.stem+'_equivalence';script,log=(d/(stem+s) for s in ('.ys','.log'))
        script.write_text(f'read_json "{tool.path(d/"before.json")}"\nrename {top} gold\n'
            f'read_json "{tool.path(gate)}"\nrename {top} gate\n'
            f'equiv_make -encfile {tool.path(d/"encoding.map")} gold gate equiv\nhierarchy -top equiv\n'
            'async2sync\ndffunmap\nopt_clean\nequiv_simple -undef -seq 2\nequiv_induct -undef -seq 4\nequiv_status -assert\n',encoding='utf-8')
        result=tool.run_to_log(['-s',tool.path(script)],log)
        return result.returncode,{'script_sha256':digest(script),'log_sha256':digest(log)}
    for matched in (0,1):
        d=base/f'owner_encoding_m{matched}';d.mkdir(exist_ok=True)
        before,recoded,after=(d/n for n in ('before.json','recoded.json','after.json'));script=d/'recode.ys';log=d/'recode.log';enc=d/'encoding.map'
        script.write_text(f'read_verilog -sv "{tool.path(source)}"\nchparam -set MATCHED_COMPLETION {matched} {top}\n'
            f'hierarchy -top {top}\nproc\nflatten\nopt_expr\nopt_clean\ncheck\nopt -nodffe -nosdff\n'
            f'write_json "{tool.path(before)}"\nfsm_detect\nfsm_extract\nfsm_opt\nfsm_recode -encfile {tool.path(enc)}\n'
            f'write_json "{tool.path(recoded)}"\nfsm_info\nfsm_map\nopt\nwrite_json "{tool.path(after)}"\n',encoding='utf-8')
        result=tool.run_to_log(['-s',tool.path(script)],log);result.check_returncode()
        machines=[c for c in json.loads(recoded.read_text())['modules'][top]['cells'].values() if c['type']=='$fsm']
        assert len(machines)==2 and all(int(c['parameters']['STATE_BITS'],2)==7 and int(c['parameters']['STATE_NUM'],2)==7 for c in machines)
        code,positive=equivalence(d,after);assert code==0
        mutant=json.loads(after.read_text());targets=[c for c in mutant['modules'][top]['cells'].values()
            if c['type'] in ('$adff','$adffe') and int(c['parameters']['WIDTH'],2)==7]
        assert len(targets)==2
        cell=targets[0];value=int(cell['parameters']['ARST_VALUE'],2);assert value and not value&(value-1)
        cell['parameters']['ARST_VALUE']=f'{((value<<1)&127)|(value>>6):07b}'
        bad=d/'bad_reset.json';bad.write_text(json.dumps(mutant)+'\n',encoding='utf-8')
        code,negative=equivalence(d,bad);assert code!=0
        row={'matched':matched,'positive':positive,'negative':negative,'artifacts':{p.name:digest(p) for p in (before,recoded,after,enc,script,log,bad)}}
        rows.append(row);print('Runtime owner physical encoding:',matched,'PASS; corrupted reset rejected.',flush=True)
    d=base/'fifo_mapping';mapped,metadata,stem=fifo.synthesize(source=root/'src/symcrash_verilog/ccs_stream_runtime.v',directory=d,tool=tool)
    fifo_report=fifo.prove(mapped,metadata,stem,directory=d,tool=tool)
    (base/'encoding_inventory.json').write_text(json.dumps({'source_sha256':digest(source),'runner_sha256':digest(Path(__file__)),
        'owner':rows,'fifo':fifo_report,'scope':'Encoding-aware matched-point owner equivalence; corrupted physical resets reject. Actual mapped FIFO all-state one-step proof. Full transport/timing separately checked.'},indent=2)+'\n',encoding='utf-8')
    print('Runtime owner encodings and mapped FIFO proofs complete.',flush=True)
    return json.loads((base/'encoding_inventory.json').read_text(encoding='utf-8'))


def check_owner():
    root=ROOT;origin=check.HDL;directory=root/'build/hdl-tests/runtime_transport_mapping/owner';directory.mkdir(parents=True,exist_ok=True)
    sources={n:digest(origin/n) for n in check.source_names('window')};source=origin/'coeff_buffer_owner_runtime.v';tool=Tool('yosys')
    library=root/'build/hdl-tools/yosys/usr/share/yosys/ecp5/cells_sim.v'
    for n in (*check.source_names('window'),'tb_ccs_dma_window_runtime.v'):
        if n!=source.name:(directory/n).write_bytes((origin/n).read_bytes())
    script,log,netlist,mapped=(directory/n for n in ('mapping.ys','mapping.log','owner.json','owner_mapped.v'))
    script.write_text(PRELUDE+f'read_verilog -sv "{tool.path(source)}"\n'
        'chparam -set MATCHED_COMPLETION 1 coeff_buffer_owner_runtime\n'
        f'synth_ecp5 -abc9 -top coeff_buffer_owner_runtime -json "{tool.path(netlist)}"\ncheck -assert\n'
        f'rename coeff_buffer_owner_runtime mapped_owner\nwrite_verilog -noattr "{tool.path(mapped)}"\n',encoding='utf-8')
    result=tool.run_to_log(['-s',tool.path(script)],log);result.check_returncode()
    text=source.read_text(encoding='utf-8');header=text[text.index('module coeff_buffer_owner_runtime'):text.index(');')+2].replace('output reg','output wire')
    (directory/source.name).write_text(header+'''\n    mapped_owner core(.*);
        initial if(MATCHED_COMPLETION!=1 || BASE!=16'h8000) $fatal(1,"Mapped owner profile mismatch");
    endmodule\n''',encoding='utf-8',newline='\n')
    class MappedSimulator(Simulator):
        def run(self,command):
            if '-g2012' in command:command += ['-I',self.path(library.parent),self.path(mapped),self.path(library)]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=600)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            return result.stdout+result.stderr
    check.HDL=directory
    report={'sources':sources,'runner_sha256':digest(Path(__file__)),'mapped_sha256':digest(mapped),'netlist_sha256':digest(netlist),
        'library_sha256':digest(library),'includes':{n:digest(library.parent/n) for n in ('cells_ff.vh','cells_io.vh')},
        **evidence(log),'runs':[], 'scope':'Actual mapped runtime owner with real RTL runtime decoder/gatherer/window/queue, all nine modes transitions, headroom2 both stalls. Other components remain RTL.'}
    try:
        for stalls in (False,True):
            report['runs'].append(check.check('window',2,stalls,False,simulator=MappedSimulator()));print(report['runs'][-1]['summary'],flush=True);assert all(digest(origin/n)==h for n,h in sources.items())
            (directory/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Mapped runtime owner integration complete.',flush=True)
    finally:
        check.HDL=origin
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('encoding','owner','all'),nargs='?',default='all')
    args=parser.parse_args()
    if args.kind in ('encoding','all'):check_encoding()
    if args.kind in ('owner','all'):check_owner()
