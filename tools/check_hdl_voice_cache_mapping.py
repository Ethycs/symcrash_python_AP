"""Mapped cache integration, with an explicitly restricted DP16KD model.

Yosys' ECP5 DP16KD simulation declaration is a black box. This model supports
only the measured 2048x9, same-clock, NOREG/NORMAL profile. It is not a vendor
timing model. Semantics: Lattice FPGA-TN-02204-1.6, section4.3 and appendixA;
primitive pin/address assignment: pinned Yosys ecp5/brams_map.v and brams.txt.
"""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.build_hdl_dma import Tool,digest
from tools.check_hdl_idwpt import Simulator
from tools.hdl_mapping import evidence
from tools.build_hdl_voice_cache import inventory

from tools.hdl_voice_ram_model import validate_ram_mapping,ram_library

def check():
    root=ROOT;d=root/'build/hdl-tests';mapped_dir=d/'voice_cache_mapped';mapped_dir.mkdir(parents=True,exist_ok=True)
    source=root/'src/symcrash_verilog/voice_state_cache.v';bench=root/'src/symcrash_verilog/tb_voice_state_cache.v'
    s=inventory()
    inputs=[source,bench,*(root/'src/symcrash_verilog'/n for n in ('dma_window.v','dma_priority.v','dma_burst.v'))]
    input_hashes={p.name:digest(p) for p in inputs}
    net=json.loads((d/s['netlist']).read_text());top=net['modules']['voice_state_cache']
    validate_ram_mapping(top)
    tool=Tool('yosys');script,log,verilog=(mapped_dir/n for n in ('export.ys','export.log','mapped_cache.v'))
    script.write_text(f'read_json "{tool.path(d/s["netlist"])}"\nrename voice_state_cache mapped_cache\nwrite_verilog -noattr "{tool.path(verilog)}"\n',encoding='utf-8')
    result=tool.run_to_log(['-s',tool.path(script)],log);result.check_returncode()
    header=source.read_text(encoding='utf-8');header=header[header.index('module voice_state_cache'):header.index(');')+2].replace('output reg','output wire')
    wrapper=mapped_dir/'voice_state_cache.v'
    wrapper.write_text(header+'\n    mapped_cache core(.*);\nendmodule\n',encoding='utf-8')
    library=root/'build/hdl-tools/yosys/usr/share/yosys/ecp5/cells_sim.v'
    model=mapped_dir/'cells_with_dp16kd_model.v';model.write_text(ram_library(library),encoding='utf-8')
    sim=Simulator();rows=[]
    sources=[wrapper,verilog,model,*(root/'src/symcrash_verilog'/n for n in ('dma_window.v','dma_priority.v','dma_burst.v')),bench]
    report={'source_sha256':digest(source),'bench_sha256':digest(bench),'runner_sha256':digest(Path(__file__)),
        'netlist_sha256':s['netlist_sha256'],'mapped_sha256':digest(verilog),'wrapper_sha256':digest(wrapper),'model_sha256':digest(model),
        'ram_model_sha256':digest(root/'tools/hdl_voice_ram_model.py'),'library_sha256':digest(library),'library_includes':{n:digest(library.parent/n) for n in ('cells_ff.vh','cells_io.vh')},
        'mapping_sources':{n:digest(library.parent/n) for n in ('brams_map.v','brams.txt')},'runs':rows,
        'scope':'Complete mapped cache with restricted, checked synchronous2048x9 DP16KD functional model. Real DMA/window remain RTL. Not vendor timing or board verification.'}
    for stalls in (False,True):
        binary=mapped_dir/f'simulation_s{int(stalls)}.vvp'
        command=sim.compiler+['-g2012','-s','tb_voice_state_cache',f'-Ptb_voice_state_cache.STALLS={int(stalls)}',
            '-Ptb_voice_state_cache.LATENCY=20','-I',sim.path(library.parent),'-o',sim.path(binary),*(sim.path(p) for p in sources)]
        result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        result=subprocess.run(sim.runtime+[sim.path(binary)],cwd=root,capture_output=True,text=True,timeout=600)
        run_log=mapped_dir/f'simulation_s{int(stalls)}.log';run_log.write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(run_log.read_text())
        summary=next(line for line in result.stdout.splitlines() if line.startswith('PASS voice_cache '))
        rows.append({'stalls':stalls,'latency':20,'summary':summary,'sources':{p.name:digest(p) for p in sources},'log':run_log.name,'log_sha256':digest(run_log)})
        assert all(digest(p)==input_hashes[p.name] for p in inputs)
        assert inventory()['netlist_sha256']==s['netlist_sha256']
        (mapped_dir/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(summary,flush=True)
    print('Mapped cache simulation with restricted DP16KD model complete.',flush=True)

    return report


if __name__=='__main__':
    check()
