"""Simulate the actual mapped runtime decoder across all ordered mode transitions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import build_hdl_idwpt_runtime as build
from tools import check_hdl_idwpt_runtime as checks
from tools.build_hdl_dma import Tool, digest, DIRECTORY
from tools.check_hdl_idwpt import Simulator

LIBRARY = ROOT/'build/hdl-tools/yosys/usr/share/yosys/ecp5/cells_sim.v'


def check(source=build.SOURCE, synthesis_directory=DIRECTORY, library=LIBRARY):
    saved_source, saved_directory = build.SOURCE, build.DIRECTORY
    try:
        build.SOURCE, build.DIRECTORY = source, synthesis_directory
        synthesis = build.inventory()
    finally:
        build.SOURCE, build.DIRECTORY = saved_source, saved_directory
    for path in (library, library.parent/'cells_ff.vh', library.parent/'cells_io.vh'):
        if not path.is_file():
            raise RuntimeError('Supply the installed Yosys ECP5 cells_sim.v with --library')
    key = digest(source)
    directory = synthesis_directory/f'runtime_mapping_{key[:12]}'
    directory.mkdir(parents=True, exist_ok=True)
    inventory = directory/'inventory.json'
    if inventory.exists():
        inventory.unlink()
    tool = Tool('yosys')
    script, log, mapped = (directory/name for name in ('export.ys', 'export.log', 'mapped_runtime.v'))
    netlist = synthesis_directory/synthesis['netlist']
    script.write_text(f'read_json "{tool.path(netlist)}"\n'
        'rename ccs_idwpt_runtime mapped_runtime\n'
        f'write_verilog -noattr "{tool.path(mapped)}"\n', encoding='utf-8')
    result = tool.run_to_log(['-s', tool.path(script)], log)
    result.check_returncode()
    original = source.read_text()
    header = original[original.index('module ccs_idwpt_runtime'):original.index(');')+2].replace('output reg', 'output wire')
    wrapper = header+f'''
    mapped_runtime core(.*);
    initial if(HEADROOM_BITS!={build.HEADROOM}) $fatal(1,"Mapped headroom mismatch");
endmodule
'''
    (directory/source.name).write_text(wrapper, encoding='utf-8', newline='\n')
    bench = checks.HDL/'tb_ccs_idwpt_runtime.v'
    # ABC9 removes these internal names. Preserve the same reset schedule and
    # all public checks; use public load/compute boundaries for these two probes.
    # The unmodified RTL bench separately asserts the precise group/root state.
    bench_text = bench.read_text()
    for old, new in (
        ('kind==16 && dut.slot_group!=probe_groups-1', 'kind==16 && !coeff_ready'),
        ('kind==17 && dut.level!=5', 'kind==17 && (coeff_ready || audio_valid)'),
    ):
        assert bench_text.count(old) == 1
        bench_text = bench_text.replace(old, new)
    (directory/bench.name).write_text(bench_text, encoding='utf-8', newline='\n')

    class MappedSimulator(Simulator):
        def run(self, command):
            if '-g2012' in command:
                command += ['-I', self.path(library.parent), self.path(mapped), self.path(library)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stdout+result.stderr)
            return result.stdout+result.stderr

    report = {'source_sha256': key, 'headroom': build.HEADROOM, 'netlist_sha256': digest(netlist),
        'runner_sha256': digest(Path(__file__)), 'rtl_testbench_sha256': digest(bench),
        'mapped_testbench_sha256': digest(directory/bench.name),
        'mapped_sha256': digest(mapped), 'wrapper_sha256': digest(directory/source.name),
        'export_script_sha256': digest(script), 'export_log_sha256': digest(log),
        'library_sha256': digest(library),
        'library_includes': {n: digest(library.parent/n) for n in ('cells_ff.vh', 'cells_io.vh')},
        'runs': [], 'scope': 'Complete mapped runtime decoder, headroom2, both stalls, all nine mode transitions, resets and reserved commands. Two internal-position observations use public load/compute boundaries; precise positions checked in RTL. No board timing claim.'}
    saved = checks.HDL
    try:
        checks.HDL = directory
        simulator = MappedSimulator()
        for stalls in (False, True):
            row = checks.check(build.HEADROOM, stalls, False, simulator)
            assert digest(source) == key and digest(netlist) == synthesis['netlist_sha256']
            report['runs'].append(row)
            print(row['summary'], flush=True)
            inventory.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    finally:
        checks.HDL = saved
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=build.SOURCE)
    parser.add_argument('--synthesis-directory', type=Path, default=DIRECTORY)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    args = parser.parse_args()
    check(args.source, args.synthesis_directory, args.library)
