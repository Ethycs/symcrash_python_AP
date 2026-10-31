"""Run the mapped ECP5 owner inside the actual windowed audio pipeline.

The queue/window/stream/decoder remain RTL. This checks functional primitive
mapping with stalls and resets; it does not establish board or routed timing.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import Tool, digest, DIRECTORY
from tools.check_hdl_idwpt import Simulator
from tools import check_hdl_dma_window as window
from tools.hdl_mapping import PRELUDE, evidence

LIBRARY = ROOT/'build/hdl-tools/yosys/usr/share/yosys/ecp5/cells_sim.v'


def check(k, library=LIBRARY, directory=DIRECTORY):
    if k not in (2, 8):
        raise ValueError('Mapped checks cover K2 captured and K8 direct matching')
    for path in (library, library.parent/'cells_ff.vh', library.parent/'cells_io.vh'):
        if not path.is_file():
            raise RuntimeError('Supply the installed Yosys ECP5 cells_sim.v with --library')
    mode = int(k == 2)
    source_dir = window.HDL
    sources = window.source_hashes()
    fingerprint = window.fingerprint()
    owner = source_dir/'coeff_buffer_owner.v'
    directory = directory/f'owner_mapping_{fingerprint[:12]}_k{k}'
    directory.mkdir(parents=True, exist_ok=True)
    inventory = directory/'inventory.json'
    if inventory.exists():
        inventory.unlink()
    for name in (*window.SOURCES, 'tb_ccs_dma_window.v'):
        if name != owner.name:
            (directory/name).write_bytes((source_dir/name).read_bytes())
    tool = Tool('yosys')
    script, log, netlist, mapped = (directory/name for name in ('mapped_owner.ys', 'mapped_owner.log', 'mapped_owner.json', 'mapped_owner.v'))
    script.write_text(PRELUDE+f'read_verilog -sv "{tool.path(owner)}"\n'
        f'chparam -set K {k} -set MATCHED_COMPLETION {mode} coeff_buffer_owner\n'
        f'synth_ecp5 -abc9 -top coeff_buffer_owner -json "{tool.path(netlist)}"\ncheck -assert\nstat\n'
        f'write_verilog -noattr "{tool.path(mapped)}"\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    mapping = evidence(log)
    cells = Counter(c['type'] for c in json.loads(netlist.read_text())['modules']['coeff_buffer_owner']['cells'].values())
    assert not cells.get('DP16KD', 0)
    text = mapped.read_text()
    assert text.count('module coeff_buffer_owner(') == 1
    mapped.write_text(text.replace('module coeff_buffer_owner(', 'module mapped_owner('), encoding='utf-8', newline='\n')
    text = owner.read_text()
    header = text[text.index('module coeff_buffer_owner'):text.index(');')+2].replace('output reg', 'output wire')
    wrapper = header+f'''
    mapped_owner core(.*);
    initial if(K!={k} || BASE!=16'h8000 || MATCHED_COMPLETION!={mode})
        $fatal(1,"Mapped owner profile mismatch");
endmodule
'''
    (directory/owner.name).write_text(wrapper, encoding='utf-8', newline='\n')

    class MappedSimulator(Simulator):
        def run(self, command):
            if '-g2012' in command:
                command = command+['-I', self.path(library.parent), self.path(mapped), self.path(library)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
            return result.stdout+result.stderr

    report = {'sources': sources, 'fingerprint': fingerprint, 'K': k, 'matched_completion': mode,
              'runner_sha256': digest(Path(__file__)), 'testbench_sha256': digest(directory/'tb_ccs_dma_window.v'),
              'netlist_sha256': digest(netlist), 'mapped_sha256': digest(mapped),
              'library_sha256': digest(library),
              'library_includes': {n: digest(library.parent/n) for n in ('cells_ff.vh', 'cells_io.vh')},
              'cells': dict(cells), **mapping, 'summaries': [],
              'scope': 'Mapped owner in actual RTL audio pipeline; headroom2, both stalls, faults/deadlines/publication/leases/resets. No device timing claim.'}
    try:
        window.HDL = directory
        simulator = MappedSimulator()
        for stalls in (False, True):
            result = window.check_stream(k, 2, stalls, simulator=simulator)
            print(result, flush=True)
            report['summaries'].append(result)
            assert all(digest(source_dir/n) == h for n, h in sources.items())
            inventory.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    finally:
        window.HDL = source_dir
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slots', type=int, choices=(2, 8), nargs='+', default=(2, 8))
    parser.add_argument('--library', type=Path, default=LIBRARY)
    args = parser.parse_args()
    for k in args.slots:
        check(k, args.library)


if __name__ == '__main__':
    main()
