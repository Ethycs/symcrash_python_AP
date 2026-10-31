"""Reject lost prefetch reservation, identity truncation, wrong ownership and offset loss."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pathlib import Path
import json
from tools.build_hdl_dma import digest
from tools.check_hdl_idwpt import Simulator

def check():
    root=ROOT;d=root/'src/symcrash_verilog';hdl=root/'src/symcrash_verilog'
    vectors=root/'build/hdl-tests/voice_stream/profile_h2_l20_s0_e0_dense20';sim=Simulator();rows={};folder=root/'build/hdl-tests/voice_stream_mutations';folder.mkdir(parents=True,exist_ok=True)
    top=d/'voice_cache_stream.v';renderer=d/'voice_cache_renderer.v';bench=d/'tb_voice_cache_stream.v'
    for name,target,old,new in [
        ('prefetch_reservation',top,'wire reserve_prefetch=front_accept && selected_slot==prefetch_slot;', "wire reserve_prefetch=1'b0;"),
        ('voice_id_truncation',top,'request_voice_id<=r_acquire_valid ? r_acquire_voice_id : state_acquire_voice_id;',
            "request_voice_id<=r_acquire_valid ? {8'd0,r_acquire_voice_id[7:0]} : state_acquire_voice_id;"),
        ('wrong_lease_owner',top,'.acquire_readonly(request_renderer)',".acquire_readonly(1'b0)"),
        ('coefficient_offset',renderer,'assign cache_dsp_word=coeff_base+read_index;', 'assign cache_dsp_word=read_index;')]:
        case=folder/name;case.mkdir(exist_ok=True);text=target.read_text(encoding='utf-8');assert text.count(old)==1
        altered=case/target.name;altered.write_text(text.replace(old,new),encoding='utf-8',newline='\n')
        sources=[altered if p==target else p for p in [top,renderer,d/'voice_state_cache.v',hdl/'ccs_idwpt_runtime.v',
            hdl/'dma_window.v',hdl/'dma_priority.v',hdl/'dma_burst.v',bench]]
        binary=case/'simulation.vvp';log=case/'simulation.log'
        sim.run(sim.compiler+['-g2012','-s','tb_voice_cache_stream','-Ptb_voice_cache_stream.CASES=20','-Ptb_voice_cache_stream.LATENCY=20',
            '-o',sim.path(binary),*(sim.path(p) for p in sources)])
        try:
            output=sim.run(sim.runtime+[sim.path(binary),*(f'+{n}='+sim.path(vectors/(f+'.hex')) for n,f in [
                ('pages','pages'),('updates','updates'),('expected','expected'),('modes','modes'),('offsets','offsets')])])
        except AssertionError as exc:
            output=str(exc);assert 'FATAL:' in output and 'Unable to open' not in output
        else:raise AssertionError(f'Corruption escaped detection: {name}')
        log.write_text(output,encoding='utf-8')
        rows[name]={'rejected':True,'sources':{p.name:digest(p) for p in sources},'log_sha256':digest(log)}
        print(name+': rejected',flush=True)
    report={'sources':{p.name:digest(p) for p in (top,renderer,bench)},'runner_sha256':digest(Path(__file__)),'checks':rows}
    (folder/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

    return report

if __name__=='__main__':
    check()
