"""Exercise coefficient ownership with the actual priority queue and burst engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.check_hdl_idwpt import Simulator, SimulatorUnavailable

HDL = ROOT / 'src/symcrash_verilog'
SOURCES = ('coeff_buffer_owner.v', 'dma_priority.v', 'dma_burst.v')
CONFIGURATIONS = [(k, stalls) for k in (2, 4, 8) for stalls in (False, True)]


def source_hashes():
    return {name: hashlib.sha256((HDL / name).read_bytes()).hexdigest() for name in SOURCES}


def fingerprint():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def check(k=4, stalls=True, simulator=None):
    if k not in (2, 4, 8): raise ValueError('K must be 2, 4 or 8')
    simulator = simulator or Simulator()
    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    evidence = directory / f'coeff_buffers_simulation_{fingerprint()[:12]}_k{k}_{int(stalls)}.json'
    if evidence.exists(): evidence.unlink()
    with TemporaryDirectory(prefix='coeff-buffers-', dir=directory) as temporary:
        executable = Path(temporary) / 'simulation.vvp'
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_coeff_buffer_owner',
            f'-Ptb_coeff_buffer_owner.K={k}', f'-Ptb_coeff_buffer_owner.STALLS={int(stalls)}',
            '-o', simulator.path(executable), *(simulator.path(HDL / name) for name in SOURCES),
            simulator.path(HDL / 'tb_coeff_buffer_owner.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable)])
        assert f'PASS ownership K={k} ' in output, output
        summary = next(line for line in output.splitlines() if line.startswith('PASS '))
        report = {'sources': source_hashes(), 'fingerprint': fingerprint(), 'K': k, 'stalls': bool(stalls),
                  'testbench_sha256': hashlib.sha256((HDL / 'tb_coeff_buffer_owner.v').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'summary': summary}
        evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    simulator = Simulator()
    for configuration in ([(4, True)] if args.quick else CONFIGURATIONS):
        print(check(*configuration, simulator=simulator), flush=True)


if __name__ == '__main__':
    main()
