"""Production voice-stream ownership proof and corruption controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pathlib import Path
import json,re
from tools.build_hdl_dma import Tool,digest
from tools.check_hdl_voice_cache_ports import control_model

def check():
    root=ROOT;d=root/'src/symcrash_verilog';folder=root/'build/hdl-tests/voice_stream_proofs/ownership';folder.mkdir(parents=True,exist_ok=True)
    source=d/'voice_cache_stream.v';cache=d/'voice_state_cache.v';tool=Tool('yosys')
    def model(text):
        start=text.index('module voice_cache_stream');end=text.index(');',start)+2
        header=text[start:end]
        header='\n'.join(line for line in header.splitlines() if not re.search(r'\b(?:block_\w+|audio_\w+)\s*[,)]?$',line.strip()))
        abstract=[('', 'r_acquire_valid'),('[3:0]','r_acquire_slot'),('[15:0]','r_acquire_voice_id'),
            ('[31:0]','r_acquire_tag'),('','r_dsp_valid'),('[6:0]','r_dsp_word'),('','r_dsp_response_ready'),('','r_release_valid')]
        header=header.replace('\n);', '\n'+''.join('    input wire '+width+' '+name+',\n' for width,name in abstract)+'    output wire formal_safe\n);')
        body=text[end:text.index('    voice_cache_renderer #',end)]
        for width,name in abstract:
            pattern=r'    wire '+(re.escape(width)+' ' if width else '')+name+r';\n'
            body,n=re.subn(pattern,'',body);assert n==1,name
        body=body.replace('    voice_state_cache cache (','    wire cache_safe;\n    voice_state_cache cache (')
        body=body.replace('.memory_fault(memory_fault)', '.memory_fault(memory_fault),.formal_safe(cache_safe)')
        ghost='''
        reg [3:0] expected_slot;
        reg [15:0] expected_voice;
        reg [31:0] expected_tag;
        reg expected_renderer;
        always @(posedge clk) if(front_accept) begin
            expected_slot<=r_acquire_valid ? r_acquire_slot : state_acquire_slot;
            expected_voice<=r_acquire_valid ? r_acquire_voice_id : state_acquire_voice_id;
            expected_tag<=r_acquire_valid ? r_acquire_tag : state_acquire_tag;
            expected_renderer<=r_acquire_valid;
        end
        assign formal_safe=cache_safe && (!rst_n || (
            (!request_pending || (!c_lease_valid && request_slot==expected_slot &&
                request_voice_id==expected_voice && request_tag==expected_tag && request_renderer==expected_renderer)) &&
            (!(front_accept && selected_slot==prefetch_slot) || (!prefetch_ready && !c_prefetch_valid)) &&
            (!renderer_owner || (!state_dsp_ready && !state_release_ready && !state_dsp_response_valid)) &&
            (renderer_owner || (!r_dsp_ready && !r_release_ready && !r_dsp_response_valid)) &&
            (!(c_acquire_result_valid || c_lease_valid) || renderer_owner==expected_renderer) &&
            (!r_dsp_response_valid || (renderer_owner && c_lease_valid)) &&
            (!state_dsp_response_valid || (!renderer_owner && c_lease_valid)) &&
            !(r_acquire_result_valid && state_acquire_result_valid) &&
            (!r_acquire_valid || !state_acquire_ready)));
    endmodule
    '''
        return header+body+ghost+control_model(cache.read_text(encoding='utf-8'))
    def prove(name,text,expected=True):
        mod,script,log=(folder/(name+s) for s in ('.v','.ys','.log'))
        mod.write_text(model(text),encoding='utf-8',newline='\n')
        script.write_text(f'read_verilog -sv "{tool.path(mod)}"\nprep -top voice_cache_stream -flatten\n'
            'memory_map\nasync2sync\ndffunmap\nopt_clean\n'
            'sat -verify -tempinduct -seq 2 -maxsteps 6 -set rst_n 1 -set-at 1 rst_n 0 -prove formal_safe 1 -show-ports\n',encoding='utf-8')
        result=tool.run_to_log(['-s',tool.path(script)],log)
        passed=result.returncode==0 and 'Induction step proven: SUCCESS' in result.stdout
        if expected:assert passed,log
        else:assert result.returncode!=0 and 'proof did fail' in result.stdout,log
        return {'proved':passed,'module_sha256':digest(mod),'script_sha256':digest(script),'log_sha256':digest(log)}
    text=source.read_text(encoding='utf-8');rows={'positive':prove('control',text)}
    for name,old,new in [
        ('prefetch_reservation','wire c_prefetch_valid=prefetch_valid && !reserve_prefetch;','wire c_prefetch_valid=prefetch_valid;'),
        ('truncated_identity','request_voice_id<=r_acquire_valid ? r_acquire_voice_id : state_acquire_voice_id;',
            "request_voice_id<=r_acquire_valid ? {8'd0,r_acquire_voice_id[7:0]} : state_acquire_voice_id;"),
        ('wrong_owner','.acquire_readonly(request_renderer)',".acquire_readonly(1'b0)")]:
        assert text.count(old)==1;rows[name]=prove(name,text.replace(old,new),False)
    report={'sources':{p.name:digest(p) for p in (source,cache)},'runner_sha256':digest(Path(__file__)),
        'cache_model_sha256':digest(root/'tools/check_hdl_voice_cache_control.py'),
        'ports_model_sha256':digest(root/'tools/check_hdl_voice_cache_ports.py'),'checks':rows,
        'scope':'Reset base plus temporal induction of actual cache/control and request/port arbitration. Cache payload is cut and renderer requests are arbitrary inputs. Does not prove decoder arithmetic or renderer temporal protocol.'}
    (folder/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Composed cache/request ownership induction and three corruption controls pass.',flush=True)

    return report

if __name__=='__main__':
    check()
