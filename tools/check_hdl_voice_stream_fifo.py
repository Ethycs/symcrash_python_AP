"""Production voice-stream fifo proof and corruption controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pathlib import Path
import json
from tools.build_hdl_dma import Tool,digest

def check():
    root=ROOT;d=root/'src/symcrash_verilog';folder=root/'build/hdl-tests/voice_stream_proofs/fifo';folder.mkdir(parents=True,exist_ok=True)
    source=d/'voice_cache_renderer.v';tool=Tool('yosys')
    def model(text):
        text=text[:text.index('`ifndef SYNTHESIS')]
        for line in ('    input wire cache_dsp_ready,\n','    input wire cache_dsp_response_valid,\n',
            '    (* ram_style = "logic" *) reg [127:0] words [0:1];\n',
            '    wire coeff_ready;\n','    always @(posedge clk) if(response) words[write_pointer]<=cache_dsp_response_data;\n'):
            assert text.count(line)==1; text=text.replace(line,'')
        text=text.replace('wire [127:0] current_word=words[read_pointer];', "wire [127:0] current_word=128'd0;")
        text=text.replace('wire decoder_frame_ready,decoder_frame_error,decoded_valid,decoded_last;', "wire decoder_frame_error=1'b0;")
        text=text.replace('output reg block_done\n', '''output reg block_done,
        input wire decoder_frame_ready,coeff_ready,decoded_valid,decoded_last,cache_permit,
        output wire formal_safe
    ''')
        start=text.index('    ccs_idwpt_runtime #');end=text.index('    reg [3:0] output_index;',start)
        text=text[:start]+'''    assign decoded_data=64'd0;
        reg cache_dsp_response_valid;
        wire cache_dsp_ready=cache_permit && (!cache_dsp_response_valid || cache_dsp_response_ready);
        always @(posedge clk or negedge rst_n)
            if(!rst_n) cache_dsp_response_valid<=0;
            else if(request) cache_dsp_response_valid<=1;
            else if(response) cache_dsp_response_valid<=0;
    '''+text[end:]
        return text+'''
        wire [6:0] consumed_words=active_k==0 ? {3'd0,leaf[5:2]} : active_k==1 ? {2'd0,leaf[5:1]} : {1'b0,leaf};
        wire [8:0] accounted={2'd0,consumed_words}+{7'd0,word_count}+{8'd0,read_pending};
        wire [1:0] expected_chunk=active_k==0 ? leaf[1:0] : active_k==1 ? {1'b0,leaf[0]} : 2'd0;
        assign formal_safe=!rst_n || (state<=SILENCE && word_count<=2 && read_index<=frame_words &&
            {1'b0,word_count}+{2'b0,read_pending}<=3'd2 &&
            read_pending==cache_dsp_response_valid &&
            (!reading || (active_k<3 && chunk==expected_chunk && {2'd0,read_index}==accounted)) &&
            (reading || (word_count==0 && !read_pending)) &&
            (!(reading && cache_dsp_response_valid) || word_count<2));
    endmodule
    '''
    def prove(name,text,expect=True):
        mod,script,log=(folder/(name+s) for s in ('.v','.ys','.log'))
        mod.write_text(model(text),encoding='utf-8',newline='\n')
        script.write_text(f'read_verilog -sv "{tool.path(mod)}"\nprep -top voice_cache_renderer -flatten\n'
            'async2sync\ndffunmap\nopt_clean\n'
            'sat -verify -tempinduct -seq 2 -maxsteps 6 -set rst_n 1 -set-at 1 rst_n 0 -prove formal_safe 1 -show-ports\n',encoding='utf-8')
        result=tool.run_to_log(['-s',tool.path(script)],log)
        passed=result.returncode==0 and 'Induction step proven: SUCCESS' in result.stdout
        if expect:assert passed,log
        else:assert result.returncode!=0 and 'proof did fail' in result.stdout,log
        return {'proved':passed,'module_sha256':digest(mod),'script_sha256':digest(script),'log_sha256':digest(log)}
    text=source.read_text(encoding='utf-8');rows={'positive':prove('control',text)}
    for name,old,new in [
        ('unreserved_read',"!word_count[1] && !(word_count[0] && read_pending)", "1'b1"),
        ('lost_pending_clear',"2'b01:read_pending<=0;", "2'b01:read_pending<=1;")]:
        assert text.count(old)==1;rows[name]=prove(name,text.replace(old,new),False)
    report={'source_sha256':digest(source),'runner_sha256':digest(Path(__file__)),'checks':rows,
        'scope':'Actual renderer control with payload/decoder cut, arbitrary decoder flow control, and the cache one-response ready/valid contract. Proves reservation, group-word accounting and response capacity from reset; no audio arithmetic equivalence claim.'}
    (folder/'inventory.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Renderer FIFO/control induction and two corruption controls pass.',flush=True)

    return report

if __name__=='__main__':
    check()
