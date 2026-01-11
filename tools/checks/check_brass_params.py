import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from symcrash_python_ap.instrument_ir import load_instrument

ir = load_instrument('presets/instruments/brass_stab_v0.yaml')
for node in ir.layers.L0_graph.nodes:
    if node.type == 'FM6':
        print(f'Algorithm: {node.params.algorithm}')
        print(f'Feedback: {node.params.feedback}')
        print(f'Op Ratios: {node.params.op_ratios}')
        print(f'Op Levels: {node.params.op_levels}')
