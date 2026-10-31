"""Exercise runtime K transitions against independent rational CCS output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.check_hdl_idwpt import Simulator,coefficient_cases,reference_audio
from tools.build_hdl_dma import digest

HDL=ROOT/'src/symcrash_verilog'


def mixed_cases(exhaustive=True):
    cases={k:coefficient_cases(k,exhaustive) for k in (2,4,8)}
    positions={k:0 for k in cases}
    # Includes every ordered K transition, including repeated modes.
    order=(2,2,4,2,8,4,4,8,8)
    result=[]
    while any(positions[k]<len(cases[k]) for k in cases):
        for k in order:
            if positions[k]<len(cases[k]):
                result.append(cases[k][positions[k]])
                positions[k]+=1
    assert {(a.shape[1],b.shape[1]) for a,b in zip(result,result[1:])}=={(a,b) for a in cases for b in cases}
    return result


def check(headroom=2,stalls=False,exhaustive=True,simulator=None):
    from symcrash_python_ap.vm.hotpath_jit import idwpt_polyphase
    if headroom not in range(16):
        raise ValueError('Unsupported headroom')
    simulator=simulator or Simulator()
    sources={n:digest(HDL/n) for n in ('ccs_idwpt_runtime.v','tb_ccs_idwpt_runtime.v')}
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    directory=ROOT/'build/hdl-tests';directory.mkdir(parents=True,exist_ok=True)
    evidence=directory/f'idwpt_runtime_{fingerprint[:12]}_h{headroom}_s{int(stalls)}_e{int(exhaustive)}.json'
    if evidence.exists():evidence.unlink()
    cases=mixed_cases(exhaustive);expected=[reference_audio(c,headroom) for c in cases]
    padded=np.full((len(cases),64,8),0x5a5a,dtype=np.int16)
    modes=[]
    for i,(c,oracle) in enumerate(zip(cases,expected)):
        k=c.shape[1];padded[i,:,:k]=c;modes.append({2:0,4:1,8:2}[k])
        actual=np.empty(64,dtype=np.int16)
        idwpt_polyphase(c.ravel(),actual,headroom)
        np.testing.assert_array_equal(actual,oracle)
    with TemporaryDirectory(prefix='runtime-idwpt-',dir=directory) as temp:
        temp=Path(temp);coeffs=temp/'coeffs.hex';outputs=temp/'expected.hex';mode_file=temp/'modes.hex';binary=temp/'simulation.vvp'
        for path,values in ((coeffs,padded),(outputs,expected)):
            path.write_text(''.join(f'{int(v)&0xffff:04x}\n' for v in np.asarray(values).flat),encoding='ascii')
        mode_file.write_text(''.join(f'{m:x}\n' for m in modes),encoding='ascii')
        simulator.run(simulator.compiler+['-g2012','-s','tb_ccs_idwpt_runtime',
            f'-Ptb_ccs_idwpt_runtime.HEADROOM_BITS={headroom}',f'-Ptb_ccs_idwpt_runtime.STALLS={int(stalls)}',
            f'-Ptb_ccs_idwpt_runtime.CASES={len(cases)}','-o',simulator.path(binary),
            simulator.path(HDL/'ccs_idwpt_runtime.v'),simulator.path(HDL/'tb_ccs_idwpt_runtime.v')])
        output=simulator.run(simulator.runtime+[simulator.path(binary),'+coeffs='+simulator.path(coeffs),
            '+expected='+simulator.path(outputs),'+modes='+simulator.path(mode_file)])
        summary=next(s for s in output.splitlines() if s.startswith('PASS runtime '))
        assert f'cases={len(cases)} headroom={headroom} stalls={int(stalls)}' in summary
        assert all(digest(HDL/n)==h for n,h in sources.items())
        report={'sources':sources,'fingerprint':fingerprint,'headroom':headroom,'stalls':bool(stalls),'exhaustive':bool(exhaustive),
            'frames':len(cases),'exact_samples':len(cases)*64,'runner_sha256':digest(Path(__file__)),
            'vector_runner_sha256':digest(ROOT/'tools/check_hdl_idwpt.py'),
            'reference_sha256':digest(ROOT/'tools/audit_lifting_arithmetic.py'),
            'coefficients_sha256':digest(coeffs),'expected_sha256':digest(outputs),'modes_sha256':digest(mode_file),'summary':summary}
        evidence.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headroom',type=int,choices=range(16),default=2)
    parser.add_argument('--stalls',action='store_true')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    print(check(args.headroom,args.stalls,not args.quick)['summary'],flush=True)


if __name__=='__main__':
    main()
