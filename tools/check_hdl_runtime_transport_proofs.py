"""Source-bound mode admission, gathering, ownership and captured-match proofs."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.build_hdl_dma import Tool,digest
from tools import check_hdl_control as control,check_hdl_owner_pipeline as owner_checks
HDL=ROOT/'src/symcrash_verilog';d=ROOT/'build/hdl-tests';owner=HDL/'coeff_buffer_owner_runtime.v';stream=HDL/'ccs_stream_runtime.v'

def normalize_owner(text):
    replacements={
        'assign prefetch_ready = rst_n && (!valid_mode || (prefetch_eligible && |claimable));':'assign prefetch_ready = prefetch_eligible && |claimable;',
        'assign claim = prefetch_valid && prefetch_ready && valid_mode;':'assign claim = prefetch_valid && prefetch_ready;',
        'wire bank_claim=prefetch_valid && valid_mode && prefetch_eligible && claimable[bank];':'wire bank_claim=prefetch_valid && prefetch_eligible && claimable[bank];'}
    for old,new in replacements.items():assert text.count(old)==1;text=text.replace(old,new)
    return text

def admission_module(text):
    assert 'localparam FRAME_BYTES = 1024;' in text
    assert 'localparam [15:0] SECOND_BASE = BASE + FRAME_BYTES;' in text
    # Guard mode storage and acquisition at the existing metadata edges.
    for fragment in ('if (bank_claim) begin\n                tag[bank] <= prefetch_tag;\n                descriptor[bank] <= incoming;\n                mode[bank] <= prefetch_k;',
                     "acquire_k <= hit ? mode[match_ready[1]] : 2'd0;",
                     'prefetch_rejected <= prefetch_valid && prefetch_ready && !valid_mode;'):
        assert fragment in text
    normalized=normalize_owner(text);owner_checks.admission_module(normalized)
    cone=re.search(r'    wire \[1:0\] tag_matches=.*?prefetch_tag\);',text,re.S).group(0)
    assignments=[]
    for name in ('valid_mode','frame_beats','wire_priority','destination','incoming'):
        assignments.append(re.search(r'    wire (?:\[[^\]]+\] )?'+name+r'\s*=.*?;',text,re.S).group(0))
    for name in ('prefetch_ready','claim'):
        assignments.append(re.search(r'    assign '+name+r'\s*=.*?;',text).group(0))
    bank_claim=re.search(r'wire bank_claim=.*?;',text).group(0).replace('wire bank_claim=', 'assign actual_claims[bank]=')
    return '''module admission_proof(input rst_n,pending_valid,prefetch_valid,cancel_valid,acquire,
        input [31:0] cancel_tag,acquire_tag,prefetch_tag,prefetch_ram_address,tag0,tag1,
        input [1:0] free_mask,prefetch_k,prefetch_class,output same);
        localparam BASE=16'h8000,SECOND_BASE=16'h8400;
        wire [31:0] tag[0:1];assign tag[0]=tag0;assign tag[1]=tag1;
        wire selected=!free_mask[0];wire prefetch_ready,claim;wire [1:0] actual_claims;
    '''+cone+'\n'+'\n'.join(assignments)+'''
        for(genvar bank=0;bank<2;bank=bank+1) begin
    '''+bank_claim+'''
        end
        wire duplicate=(!free_mask[0] && tag0==prefetch_tag) || (!free_mask[1] && tag1==prefetch_tag);
        wire ordinary=rst_n && |free_mask && !pending_valid && !duplicate &&
            !(cancel_valid && cancel_tag==prefetch_tag) && !(acquire && acquire_tag==prefetch_tag);
        wire legal=prefetch_k<3;
        wire reference_ready=rst_n && (!legal || ordinary);
        wire reference_claim=prefetch_valid && legal && ordinary;
        wire [1:0] reference_claims={reference_claim && selected,reference_claim && !selected};
        wire [7:0] expected_beats=8'd16<<prefetch_k;
        wire [15:0] expected_destination=selected ? 16'h8400 : 16'h8000;
        wire [3:0] expected_priority=15-prefetch_class;
        wire [63:0] expected_descriptor={prefetch_ram_address,expected_destination,expected_beats,expected_priority,4'b0011};
        assign same=prefetch_ready==reference_ready && claim==reference_claim && actual_claims==reference_claims &&
            (!claim || incoming==expected_descriptor);
    endmodule
    '''

def gather_module(text):
    fragments=[]
    for name in ('frame_words','final_group','final_chunk','word_lane','coeff_data','pop_word'):
        fragments.append(re.search(r'    wire (?:\[[^\]]+\] )?'+name+r'=.*?;',text,re.S).group(0))
    assert 'chunk<=pop_word ? 2\'d0 : chunk+1\'b1;' in text
    step=re.search(r'chunk<=(pop_word[^;]+);',text).group(1)
    return '''module gather_proof(input [1:0] active_k,group_index,input [5:0] leaf,
        input accept_coeff,input [127:0] current_word,output same);
        wire [1:0] chunk=(active_k==0) ? leaf[1:0] : (active_k==1) ? {1'b0,leaf[0]} : 2'd0;
    '''+'\n'.join(fragments)+'''
        wire [31:0] k=2<<active_k;
        wire legal=active_k<3 && group_index<k/2;
        wire [31:0] slot=(leaf*k+2*group_index)%8;
        wire [31:0] reference=current_word[16*slot +:32];
        wire [5:0] next_leaf=leaf+1'b1;
        wire [1:0] expected_chunk=(next_leaf*k/2)%4;
        wire [1:0] next_chunk=accept_coeff ? ('''+step+''') : chunk;
        wire should_pop=accept_coeff && ((leaf+1)*k)%8==0;
        assign same=!legal || (coeff_data==reference && pop_word==should_pop &&
            next_chunk==(accept_coeff ? expected_chunk/(k/2) : chunk) &&
            frame_words==8*k && final_group==k/2-1 && final_chunk==8/k-1);
    endmodule
    '''

def prove(name,module,tool):
    folder=d/'runtime_transport_proofs';folder.mkdir(parents=True,exist_ok=True)
    file,script,log=(folder/(name+s) for s in ('.v','.ys','.log'))
    file.write_text(module,encoding='utf-8',newline='\n')
    script.write_text(f'read_verilog -sv "{tool.path(file)}"\nprep -top {name}_proof -flatten\n'
        'sat -verify -prove same 1 -show-inputs\n',encoding='utf-8')
    result=tool.run_to_log(['-s',tool.path(script)],log)
    if result.returncode or 'no model found: SUCCESS' not in result.stdout:raise RuntimeError(f'{name} proof failed: {log}')
    return {'result':'PASS','module_sha256':digest(file),'script_sha256':digest(script),'log_sha256':digest(log)}

def check():
    tool=Tool('yosys');ot=owner.read_text(encoding='utf-8');st=stream.read_text(encoding='utf-8');normalized=normalize_owner(ot)
    result={'sources':{p.name:digest(p) for p in (owner,stream,HDL/'dma_window.v')},'runner_sha256':digest(Path(__file__)),
        'admission':prove('admission',admission_module(ot),tool),
        'gather':prove('gather',gather_module(st),tool),
        'policy':prove('policy',control.owner_module(normalized).replace('module control_proof','module policy_proof'),tool),
        'prematch':prove('prematch',owner_checks.prematch_module(normalized,(HDL/'dma_window.v').read_text(encoding='utf-8')),tool),
        'scope':'All-input mode admission/descriptor binding and conditional gather coordinates. Mode metadata source guards and legacy owner policy/prematch composed through legal claims. Full timing/reset/DMA tested separately.'}
    (d/'runtime_transport_proofs/inventory.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
if __name__=='__main__':print(json.dumps(check()),flush=True)
