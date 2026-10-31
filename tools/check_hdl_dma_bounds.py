"""Prove the actual DMA validation cone against full-precision byte bounds.

All 64 descriptor input bits are unconstrained. This is a combinational
address-validation proof; transport state, payloads and timing have separate
simulation and routing gates. Missing Yosys fails the standalone command.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import SOURCE, DIRECTORY, Tool, digest


def prove(source=SOURCE, directory=DIRECTORY, tool=None):
    tool = tool or Tool('yosys')
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    stem = 'dma_bounds_formal_' + source_hash[:12]
    evidence = directory / (stem + '.json')
    if evidence.exists():
        evidence.unlink()
    text = source.read_text(encoding='utf-8')
    # Use the actual RTL expressions. A changed/moved cone must fail clearly,
    # rather than silently proving a stale copy of the implementation.
    start = text.index('    wire [32:0] ram_end')
    end = text.index('    wire unsupported', start)
    cone = text[start:end]
    module = directory / (stem + '.v')
    module.write_text('''module dma_bounds_formal(input [63:0] descriptor, output equivalent);
'''+cone+'''
    // Independent reference uses 64-bit unsigned byte arithmetic. Its largest
    // endpoint is 2^32-1 + 255*16, which is representable without overflow.
    wire [63:0] reference_ram_end = {32'd0, descriptor[63:32]} + {52'd0, descriptor[15:8], 4'd0};
    wire [63:0] reference_bram_end = {48'd0, descriptor[31:16]} + {52'd0, descriptor[15:8], 4'd0};
    wire reference_bad = !descriptor[0] || descriptor[15:8] == 0
                      || descriptor[63:32] % 16 != 0 || descriptor[31:16] % 16 != 0
                      || reference_ram_end > 64'd134217728 || reference_bram_end > 64'd65536;
    assign equivalent = bad_descriptor == reference_bad;
endmodule
''', encoding='utf-8', newline='\n')
    script = directory / (stem + '.ys')
    script.write_text(f'read_verilog "{tool.path(module)}"\n'
                      'prep -top dma_bounds_formal\n'
                      'sat -verify -prove equivalent 1 -show-inputs\n', encoding='utf-8', newline='\n')
    result = tool.run(['-s', tool.path(script)])
    log = directory / (stem + '.log')
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode or 'SUCCESS' not in result.stdout:
        raise RuntimeError(f'DMA bounds equivalence failed; see {log}')
    report = {'result': 'PASS', 'source_sha256': source_hash, 'runner_sha256': digest(Path(__file__)),
              'proof_module_sha256': digest(module), 'proof_log_sha256': digest(log),
              'tool': tool.version('-V'), 'unconstrained_descriptor_bits': 64,
              'reference_arithmetic_bits': 64, 'ram_bytes': 134217728, 'bram_bytes': 65536,
              'scope': 'Combinational descriptor validity only; transport and mapped timing are separate gates'}
    evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(prove(), indent=2))
