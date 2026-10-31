"""Synthesize/route runtime K transport at the documented 100 MHz target."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.build_hdl_dma import Tool,digest
from tools.hdl_mapping import PRELUDE,MAPPING,evidence,matches
HDL=ROOT/'src/symcrash_verilog';d=ROOT/'build/hdl-tests';HEADROOM=2;BASE=0x8000

def source_names(kind):
    common=('ccs_stream_runtime.v','coeff_buffer_owner_runtime.v','ccs_idwpt_runtime.v','dma_priority.v','dma_burst.v')
    return common if kind=='stream' else ('ccs_dma_window_runtime.v','dma_window.v',*common)

def hashes(kind):return {n:digest(HDL/n) for n in source_names(kind)}
def fingerprint(kind):return hashlib.sha256(json.dumps(hashes(kind),sort_keys=True).encode()).hexdigest()
def top(kind):return 'ccs_stream_runtime' if kind=='stream' else 'ccs_dma_window_runtime'
def synth(kind):
    d.mkdir(parents=True,exist_ok=True)
    tool=Tool('yosys');sources=hashes(kind);key=fingerprint(kind);stem=f'{top(kind)}_{key[:12]}_abc9'
    script,log,netlist=(d/(stem+s) for s in ('_ecp5.ys','_ecp5.log','_ecp5.json'))
    script.write_text(PRELUDE+'\n'.join(f'read_verilog -sv "{tool.path(HDL/n)}"' for n in source_names(kind))+'\n'+
        f'chparam -set HEADROOM_BITS {HEADROOM} -set BASE {BASE} {top(kind)}\n'+
        f'synth_ecp5 -abc9 -top {top(kind)} -json "{tool.path(netlist)}"\ncheck -assert\nstat\n',encoding='utf-8')
    result=tool.run_to_log(['-s',tool.path(script)],log);result.check_returncode()
    cells=Counter(c['type'] for c in json.loads(netlist.read_text())['modules'][top(kind)]['cells'].values())
    assert hashes(kind)==sources
    report={'sources':sources,'fingerprint':key,'mapping':MAPPING,'headroom':HEADROOM,'base':BASE,
        'window_cycles':40000 if kind=='window' else None,'tool':tool.version('-V'),'runner_sha256':digest(Path(__file__)),
        'netlist':netlist.name,'netlist_sha256':digest(netlist),**evidence(log),'cells':dict(sorted(cells.items())),
        'payload_bytes':2048,'fifo_bytes':32,'decoder_logical_bytes':640,'replay_bytes':128,
        'physical_bram_bytes':cells['DP16KD']*2304,'physical_dpr_bytes':cells['TRELLIS_DPR16X4']*8,
        'scope':'Runtime K2/K4/K8 transport for one logical stream; shared multi-voice context/state and board/PHY remain open.'}
    (d/f'runtime_transport_{kind}_synthesis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
    return report
def inventory(kind):
    report=json.loads((d/f'runtime_transport_{kind}_synthesis.json').read_text());key=fingerprint(kind);stem=f'{top(kind)}_{key[:12]}_abc9'
    netlist=d/(stem+'_ecp5.json')
    valid=(report.get('sources')==hashes(kind) and report.get('fingerprint')==key and
        report.get('headroom')==HEADROOM and report.get('base')==BASE and report.get('mapping')==MAPPING and
        report.get('window_cycles')==(40000 if kind=='window' else None) and report.get('netlist')==netlist.name and
        netlist.is_file() and digest(netlist)==report.get('netlist_sha256') and matches(report,d,stem))
    if not valid:raise RuntimeError('Missing, stale or incompatible runtime transport synthesis')
    return report
def route(kind):
    s=inventory(kind);tool=Tool('nextpnr-ecp5');rows=[]
    report={'sources':s['sources'],'fingerprint':s['fingerprint'],'netlist_sha256':s['netlist_sha256'],
        'mapping':MAPPING,'headroom':HEADROOM,'base':BASE,'window_cycles':s['window_cycles'],
        'device':'ECP5-25K','package':'CSFBGA285','speed_grade':6,'clock_mhz':100,'tool':tool.version('--version'),
        'scope':'Out-of-context runtime transport, no board I/O/global clocks, no PHY or complete voice workload.','runs':rows}
    for seed in (1,2,3):
        stem=f'{top(kind)}_{s["fingerprint"][:12]}_abc9_seed{seed}';log,timing=(d/(stem+x) for x in ('_route.log','_timing.json'))
        if timing.exists():timing.unlink()
        result=tool.run_to_log(['--25k','--package','CSFBGA285','--speed','6','--out-of-context','--freq','100',
            '--seed',str(seed),'--json',tool.path(d/s['netlist']),'--report',tool.path(timing)],log)
        raw=json.loads(timing.read_text()) if timing.exists() else {};clock=raw.get('fmax',{}).get('clk',{})
        row={'seed':seed,'exit_code':result.returncode,'passed':result.returncode==0 and clock.get('constraint')==100 and clock.get('achieved',0)>=100,
            'fmax':raw.get('fmax',{}),'utilization':raw.get('utilization',{}),'route_log_sha256':digest(log)}
        if timing.exists():row['timing_sha256']=digest(timing)
        assert inventory(kind)['netlist_sha256']==s['netlist_sha256'];rows.append(row)
        (d/f'runtime_transport_{kind}_timing.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(row),flush=True)
    return all(r['passed'] for r in rows)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('stream','window'))
    parser.add_argument('action',choices=('synth','route'))
    args=parser.parse_args();kind,action=args.kind,args.action
    if action=='synth':synth(kind)
    else:sys.exit(0 if route(kind) else 1)
