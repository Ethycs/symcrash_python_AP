"""Prove the mapped 2x128-bit feeder FIFO's complete one-clock transition.

All stored bits and input fields are unconstrained. This verifies the isolated
FIFO after ECP5 mapping, not the payload BRAM, complete stream or routed device.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest, DIRECTORY
from tools.hdl_mapping import PRELUDE, evidence as mapping_evidence

SOURCE = ROOT / 'src/symcrash_verilog/ccs_stream.v'


def extract(text):
    declaration = '(* ram_style = "logic" *) reg [127:0] words [0:1];'
    read = 'wire [127:0] current_word=words[read_pointer];'
    write = 'always @(posedge clk) if (rst_n && read_pending) words[write_pointer] <= payload_read;'
    for fragment in (declaration, read, write):
        assert text.count(fragment) == 1, 'FIFO storage is outside this proof model'
    assert len(re.findall(r'\bwords\b', re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S))) == 3, 'Additional FIFO access is outside this proof model'
    assert 'reg read_pointer, write_pointer, read_pending;' in text, 'FIFO pointer width is outside this proof model'
    assert 'reg [127:0] payload_read;' in text, 'FIFO input width is outside this proof model'
    return '''module fifo_storage(input clk,rst_n,read_pending,write_pointer,read_pointer,
input [127:0] payload_read,output [127:0] current_word);
'''+declaration+'\n'+read.replace('wire [127:0] ', 'assign ')+'\n'+write+'\nendmodule\n'


def transition_proof(mapped):
    """Cut all 256 mapped FF outputs at arbitrary shared prior state.

For this accepted FF configuration the ECP5 cells_sim.v model's next Q is
CE ? DI : Q: positive CLK, no set/reset, no clock inversion. Fail on any
other configuration instead of silently simplifying it. LUT/mux behavior
comes from the installed vendor-cell simulation models.
"""
    cells = mapped['cells']
    state = mapped['netnames']['words[0]']['bits'] + mapped['netnames']['words[1]']['bits']
    assert len(state) == len(set(state)) == 256 and all(isinstance(b, int) for b in state)
    ff = [c for c in cells.values() if c['type'] == 'TRELLIS_FF']
    assert len(ff) == 256 and {c['connections']['Q'][0] for c in ff} == set(state)
    assert all(c['type'] in ('TRELLIS_FF', 'LUT4', 'PFUMX', 'L6MUX21') for c in cells.values())
    ports = mapped['ports']
    assert {n: len(p['bits']) for n, p in ports.items()} == {
        'clk': 1, 'rst_n': 1, 'read_pending': 1, 'write_pointer': 1,
        'read_pointer': 1, 'payload_read': 128, 'current_word': 128}
    bits = {b for c in cells.values() for v in c['connections'].values() for b in v if isinstance(b, int)}
    bits.update(b for p in ports.values() for b in p['bits'] if isinstance(b, int))

    def wire(bit):
        if isinstance(bit, int):
            return f'b{bit}'
        assert bit in ('0', '1'), 'Unknown constants are outside this proof'
        return "1'b"+bit

    def bus(values):
        return '{'+','.join(wire(b) for b in reversed(values))+'}'

    lines = ['module fifo_proof(input clk,rst_n,read_pending,write_pointer,read_pointer,',
             'input [127:0] payload_read,input [255:0] held,output same);',
             'wire [255:0] actual_next; reg [255:0] expected;',
             'wire '+','.join(f'b{b}' for b in sorted(bits))+';']
    for name, port in ports.items():
        if port['direction'] == 'input':
            lines.append(f'assign {bus(port["bits"])}={name};')
    for i, cell in enumerate(cells.values()):
        conn = cell['connections']
        params = cell['parameters']
        if cell['type'] == 'TRELLIS_FF':
            assert params == {'CEMUX': 'CE', 'CLKMUX': 'CLK', 'GSR': 'DISABLED',
                              'LSRMUX': 'LSR', 'REGSET': 'RESET'}, params
            assert conn['CLK'] == ports['clk']['bits'] and conn['LSR'] == ['0']
            assert set(conn) == {'CLK', 'LSR', 'CE', 'DI', 'Q'} and all(len(v) == 1 for v in conn.values())
            index = state.index(conn['Q'][0])
            lines += [f'assign {wire(conn["Q"][0])}=held[{index}];',
                      f'assign actual_next[{index}]={wire(conn["CE"][0])} ? {wire(conn["DI"][0])} : held[{index}];']
        else:
            assert cell['type'] == 'LUT4' or not params
            parameter = ''
            if params:
                assert set(params) == {'INIT'} and re.fullmatch('[01]{16}', params['INIT'])
                parameter = "#(.INIT(16'b"+params['INIT']+'))'
            lines.append(f'{cell["type"]} {parameter} c{i} ('+
                         ','.join(f'.{port}({bus(value)})' for port, value in conn.items())+');')
    lines += ['always @* begin', 'expected=held;',
              'if (rst_n && read_pending) begin',
              'if (write_pointer) expected[255:128]=payload_read;',
              'else expected[127:0]=payload_read;', 'end', 'end',
              'assign same=(actual_next==expected) && ('+bus(ports['current_word']['bits'])+
              '==(read_pointer ? held[255:128] : held[127:0]));', 'endmodule']
    return '\n'.join(lines)+'\n'


def synthesize(source=SOURCE, directory=DIRECTORY, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    stem = 'fifo_mapping_'+digest(source)[:12]
    module = directory / (stem+'.v')
    module.write_text(extract(source.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    netlist, script, log = (directory / (stem+suffix) for suffix in ('_ecp5.json', '_ecp5.ys', '_ecp5.log'))
    script.write_text(PRELUDE+f'read_verilog "{tool.path(module)}"\n'
                      f'synth_ecp5 -abc9 -top fifo_storage -json "{tool.path(netlist)}"\n'
                      'check -assert\n', encoding='utf-8', newline='\n')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    metadata = mapping_evidence(log)
    return json.loads(netlist.read_text())['modules']['fifo_storage'], {
        'source_sha256': digest(source), 'extracted_module_sha256': digest(module),
        'netlist': netlist.name, 'netlist_sha256': digest(netlist), **metadata}, stem


def prove(mapped, metadata, stem, directory=DIRECTORY, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / (stem+'_proof.json')
    if report_path.exists():
        report_path.unlink()
    module, script, log = (directory / (stem+suffix) for suffix in ('_proof.v', '_proof.ys', '_proof.log'))
    flattened = directory / (stem+'_proof_flat.json')
    module.write_text(transition_proof(mapped), encoding='utf-8', newline='\n')
    script.write_text('read_verilog -sv +/ecp5/cells_sim.v\n'
                      f'read_verilog "{tool.path(module)}"\nhierarchy -check -top fifo_proof\n'
                      'flatten -wb\nproc\nopt_clean\n'
                      f'write_json "{tool.path(flattened)}"\n'
                      'sat -verify -prove same 1 -show-inputs\n',
                      encoding='utf-8', newline='\n')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    if result.returncode or 'SUCCESS' not in result.stdout:
        raise RuntimeError(f'Mapped FIFO transition proof failed; see {log}')
    report = {**metadata, 'result': 'PASS', 'runner_sha256': digest(Path(__file__)),
              'proof_module_sha256': digest(module), 'proof_log_sha256': digest(log),
              'proof_netlist_sha256': digest(flattened),
              'tool': tool.version('-V'),
              'scope': 'Isolated mapped 2x128-bit FIFO: all binary prior states and read/write inputs, one-clock transition and asynchronous read; no initialization or complete-stream claim'}
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    mapped, metadata, stem = synthesize()
    print(json.dumps(prove(mapped, metadata, stem), indent=2))
