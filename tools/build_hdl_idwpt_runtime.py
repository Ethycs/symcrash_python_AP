"""Synthesize and route the shared runtime-K decoder at the fixed 100 MHz target."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.build_hdl_dma import Tool,digest,DIRECTORY
from tools.hdl_mapping import MAPPING,NETLIST_TAG,PRELUDE,evidence,matches

SOURCE=ROOT/'src/symcrash_verilog/ccs_idwpt_runtime.v'
HEADROOM=2


def synthesize():
    tool=Tool('yosys');DIRECTORY.mkdir(parents=True,exist_ok=True)
    key=digest(SOURCE);stem=f'idwpt_runtime_{key[:12]}_{NETLIST_TAG}'
    script,log,netlist=(DIRECTORY/(stem+s) for s in ('_ecp5.ys','_ecp5.log','_ecp5.json'))
    script.write_text(PRELUDE+f'read_verilog -sv "{tool.path(SOURCE)}"\n'
        f'chparam -set HEADROOM_BITS {HEADROOM} ccs_idwpt_runtime\n'
        f'synth_ecp5 -abc9 -top ccs_idwpt_runtime -json "{tool.path(netlist)}"\ncheck -assert\nstat\n',encoding='utf-8')
    result=tool.run_to_log(['-s',tool.path(script)],log);result.check_returncode()
    cells=Counter(c['type'] for c in json.loads(netlist.read_text())['modules']['ccs_idwpt_runtime']['cells'].values())
    assert digest(SOURCE)==key
    report={'source_sha256':key,'headroom':HEADROOM,'mapping':MAPPING,'tool':tool.version('-V'),
        'supported_slots':[2,4,8],'input_lanes':2,'logical_array_bytes':640,
        'physical_block_ram_bytes':cells['DP16KD']*18432//8,
        'physical_distributed_ram_bytes':cells['TRELLIS_DPR16X4']*8,
        'netlist':netlist.name,'netlist_sha256':digest(netlist),**evidence(log),'cells':dict(sorted(cells.items())),
        'scope':'One runtime K2/K4/K8 decoder. Excludes coefficient generation, feeder, ownership/DMA, mix and board/PHY.'}
    (DIRECTORY/'idwpt_runtime_synthesis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


def inventory():
    report=json.loads((DIRECTORY/'idwpt_runtime_synthesis.json').read_text())
    key=digest(SOURCE);stem=f'idwpt_runtime_{key[:12]}_{NETLIST_TAG}'
    netlist=DIRECTORY/(stem+'_ecp5.json')
    if not (report.get('source_sha256')==key and report.get('headroom')==HEADROOM and report.get('mapping')==MAPPING
            and report.get('supported_slots')==[2,4,8] and report.get('input_lanes')==2
            and report.get('netlist')==netlist.name and netlist.is_file()
            and report.get('netlist_sha256')==digest(netlist) and matches(report,DIRECTORY,stem)):
        raise RuntimeError('Missing, stale or incompatible runtime-decoder synthesis; rerun synthesis')
    return report


def route(seeds=(1,2,3)):
    synthesis=inventory();tool=Tool('nextpnr-ecp5');key=synthesis['source_sha256']
    report={'source_sha256':key,'netlist_sha256':synthesis['netlist_sha256'],'headroom':HEADROOM,
        'mapping':MAPPING,'tool':tool.version('--version'),'device':'ECP5-25K','package':'CSFBGA285',
        'speed_grade':6,'clock_mhz':100,'runs':[],
        'scope':'Runtime-mode decoder only; OOC excludes board I/O/global clocks and the complete voice workload.'}
    out=DIRECTORY/f'idwpt_runtime_{key[:12]}_{NETLIST_TAG}_timing_inventory.json'
    for seed in seeds:
        stem=f'idwpt_runtime_{key[:12]}_{NETLIST_TAG}_seed{seed}'
        log,timing=(DIRECTORY/(stem+s) for s in ('_route.log','_timing.json'))
        if timing.exists():timing.unlink()
        result=tool.run_to_log(['--25k','--package','CSFBGA285','--speed','6','--out-of-context','--freq','100',
            '--seed',str(seed),'--json',tool.path(DIRECTORY/synthesis['netlist']),'--report',tool.path(timing)],log)
        raw=json.loads(timing.read_text()) if timing.exists() else {};clock=raw.get('fmax',{}).get('clk',{})
        row={'seed':seed,'exit_code':result.returncode,
            'passed':result.returncode==0 and clock.get('constraint')==100 and clock.get('achieved',0)>=100,
            'fmax':raw.get('fmax',{}),'utilization':raw.get('utilization',{}),'route_log_sha256':digest(log)}
        if timing.exists():row['timing_sha256']=digest(timing)
        assert inventory()['netlist_sha256']==synthesis['netlist_sha256']
        report['runs'].append(row);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)
    return all(r['passed'] for r in report['runs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('synth','route'))
    parser.add_argument('--seeds',type=int,nargs='+',default=(1,2,3))
    args=parser.parse_args()
    if args.action=='synth':synthesize();return 0
    return 0 if route(args.seeds) else 1


if __name__=='__main__':
    sys.exit(main())
