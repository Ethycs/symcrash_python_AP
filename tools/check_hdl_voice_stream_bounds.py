"""Production voice-stream bounds proof and corruption controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pathlib import Path
import json,re
from tools.build_hdl_dma import Tool,digest
from tools.check_hdl_runtime_transport_proofs import gather_module

def check():
    root=ROOT;d=root/'src/symcrash_verilog';tool=Tool('yosys');folder=root/'build/hdl-tests/voice_stream_proofs/bounds';folder.mkdir(parents=True,exist_ok=True)
    source=d/'voice_cache_renderer.v';text=source.read_text(encoding='utf-8')
    def wire(text,name):return re.search(r'    wire (?:\[[^\]]+\] )?'+name+r'=.*?;',text,re.S).group(0)
    def bounds_module(text):
        for expected in ('active_k<=block_k;coeff_base<=block_coeff_word;',
            'state<=invalid_command ? SILENCE : ACQUIRE;',
            'assign block_ready=rst_n && state==IDLE && cache_acquire_ready;'):
            assert expected in text
        parts=[wire(text,n) for n in ('requested_words','requested_end','invalid_command','frame_words')]
        address=re.search(r'    assign cache_dsp_word=.*?;',text).group(0)
        admission='\n'.join(re.search(r'    assign '+name+r'=.*?;',text).group(0)
            for name in ('block_ready','cache_acquire_valid'))
        return '''module bounds_proof(input [1:0] block_k,active_k,input [15:0] block_voice_id,
            input [6:0] block_coeff_word,coeff_base,read_index,
            input rst_n,block_valid,cache_acquire_ready,input [2:0] state,output same);
            localparam IDLE=3'd0;
            wire [6:0] cache_dsp_word;
            wire block_ready,cache_acquire_valid;
        '''+'\n'.join(parts)+'\n'+address+'\n'+admission+'''
            wire [31:0] command_words=16<<block_k;
            wire reference_invalid=block_k>2 || block_voice_id==65535 || block_coeff_word+command_words>128;
            wire [31:0] active_words=16<<active_k;
            wire legal_read=active_k<3 && coeff_base+active_words<=128 && read_index<active_words;
            wire [31:0] expected_address={25'd0,coeff_base}+{25'd0,read_index};
            assign same=invalid_command==reference_invalid &&
                (cache_acquire_valid && cache_acquire_ready)==(block_ready && block_valid && !reference_invalid) &&
                (!legal_read || ({25'd0,cache_dsp_word}==expected_address && frame_words==active_words));
        endmodule
    '''
    def prove(name,module,expect=True):
        mod,script,log=(folder/(name+s) for s in ('.v','.ys','.log'))
        mod.write_text(module,encoding='utf-8',newline='\n')
        top=re.search(r'module (\w+)',module)[1]
        script.write_text(f'read_verilog -sv "{tool.path(mod)}"\nprep -top {top} -flatten\n'
            'sat -verify -prove same 1 -show-inputs\n',encoding='utf-8')
        result=tool.run_to_log(['-s',tool.path(script)],log)
        passed=result.returncode==0 and 'no model found: SUCCESS' in result.stdout
        if expect:assert passed,log
        else:assert result.returncode!=0 and 'proof did fail' in result.stdout,log
        return {'passed':passed,'module_sha256':digest(mod),'script_sha256':digest(script),'log_sha256':digest(log)}
    rows={'bounds':prove('bounds',bounds_module(text)),'gather':prove('gather',gather_module(text))}
    for name,old,new,builder in [
        ('truncated_endpoint','wire [7:0] requested_end=','wire [6:0] requested_end=',bounds_module),
        ('reserved_mode','block_k==3 ||',"1'b0 ||",bounds_module),
        ('group_swap','{chunk[0],group_index[0]}','{group_index[0],chunk[0]}',gather_module),
        ('group_zero',': group_index;',": 2'd0;",gather_module)]:
        assert text.count(old)==1
        rows[name]=prove(name,builder(text.replace(old,new)),False)
    report={'source_sha256':digest(source),'runner_sha256':digest(Path(__file__)),
        'gather_builder_sha256':digest(root/'tools/check_hdl_runtime_transport_proofs.py'),'checks':rows,
        'scope':'All-input command range/identity validity and conditional read-address/gather coordinates, with captured-field guards. Not complete temporal cache/decoder equivalence.'}
    (folder/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Voice renderer bounds/gather proofs and four corruption controls pass.',flush=True)

    return report

if __name__=='__main__':
    check()
