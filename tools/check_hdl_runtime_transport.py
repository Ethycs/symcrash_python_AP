"""Check mixed-K prepared-frame transport and shared deadlines against exact audio."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.check_hdl_idwpt import Simulator,reference_audio
from tools.check_hdl_idwpt_runtime import mixed_cases
from tools.build_hdl_runtime_transport import source_names,top
from tools.build_hdl_dma import digest

HDL=ROOT/'src/symcrash_verilog'
DIRECTORY=ROOT/'build/hdl-tests'
CONFIGURATIONS=[(kind,h,stalls,kind=='stream' and h==2)
                for kind in ('stream','window') for h in (2,0,15) for stalls in (False,True)]


def check(kind='stream',headroom=2,stalls=False,exhaustive=False,simulator=None):
    from symcrash_python_ap.vm.hotpath_jit import idwpt_polyphase
    if kind not in ('stream','window') or headroom not in range(16):raise ValueError('Unsupported runtime transport profile')
    simulator=simulator or Simulator();DIRECTORY.mkdir(parents=True,exist_ok=True)
    bench='tb_'+top(kind);names=source_names(kind)
    sources={n:digest(HDL/n) for n in (*names,bench+'.v')}
    key=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    evidence=DIRECTORY/f'runtime_transport_{kind}_{key[:12]}_h{headroom}_s{int(stalls)}_e{int(exhaustive)}.json'
    if evidence.exists():evidence.unlink()
    cases=mixed_cases(exhaustive);expected=[reference_audio(c,headroom) for c in cases]
    padded=np.full((len(cases),512),0x5a5a,dtype=np.int16);modes=[]
    for row,(coefficients,oracle) in enumerate(zip(cases,expected)):
        padded[row,:coefficients.size]=coefficients.ravel();modes.append({2:0,4:1,8:2}[coefficients.shape[1]])
        actual=np.empty(64,dtype=np.int16);idwpt_polyphase(coefficients.ravel(),actual,headroom)
        np.testing.assert_array_equal(actual,oracle)
    with TemporaryDirectory(prefix='runtime-transport-',dir=DIRECTORY) as temporary:
        temporary=Path(temporary)
        coeff,want,mode,binary=(temporary/n for n in ('coeffs.hex','expected.hex','modes.hex','simulation.vvp'))
        for path,values in ((coeff,padded),(want,expected)):
            path.write_text(''.join(f'{int(v)&65535:04x}\n' for v in np.asarray(values).flat),encoding='ascii')
        mode.write_text(''.join(f'{v:x}\n' for v in modes),encoding='ascii')
        simulator.run(simulator.compiler+['-g2012','-s',bench,f'-P{bench}.HEADROOM_BITS={headroom}',
            f'-P{bench}.STALLS={int(stalls)}',f'-P{bench}.CASES={len(cases)}','-o',simulator.path(binary),
            *(simulator.path(HDL/n) for n in names),simulator.path(HDL/(bench+'.v'))])
        output=simulator.run(simulator.runtime+[simulator.path(binary),'+coeffs='+simulator.path(coeff),
            '+expected='+simulator.path(want),'+modes='+simulator.path(mode)])
        summary=next(line for line in output.splitlines() if line.startswith(f'PASS runtime_{kind} '))
        assert f'headroom={headroom} stalls={int(stalls)} cases={len(cases)} ' in summary
        assert all(digest(HDL/n)==value for n,value in sources.items())
        report={'kind':kind,'sources':sources,'fingerprint':key,'headroom':headroom,'stalls':bool(stalls),
            'exhaustive':bool(exhaustive),'cases':len(cases),'exact_samples':int(re.search(r'samples=(\d+)',summary)[1]),
            'window_cycles':40000 if kind=='window' else None,'runner_sha256':digest(Path(__file__)),
            'vector_runner_sha256':digest(ROOT/'tools/check_hdl_idwpt_runtime.py'),
            'reference_sha256':digest(ROOT/'tools/audit_lifting_arithmetic.py'),
            'coefficients_sha256':digest(coeff),'expected_sha256':digest(want),'modes_sha256':digest(mode),'summary':summary}
        evidence.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('stream','window','all'),default='all')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();rows=[]
    configurations=[(k,2,False,False) for k in ('stream','window')] if args.quick else CONFIGURATIONS
    for kind,h,stalls,full in configurations:
        if args.kind!='all' and kind!=args.kind:continue
        rows.append(check(kind,h,stalls,full));print(rows[-1]['summary'],flush=True)
    (DIRECTORY/'runtime_transport_matrix.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
