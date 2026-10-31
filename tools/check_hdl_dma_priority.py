"""Simulate priority/FIFO ordering and the real burst engine under backpressure."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.check_hdl_idwpt import Simulator, SimulatorUnavailable

HDL = ROOT / 'src/symcrash_verilog'


def vectors(pattern, count):
    rng = random.Random(0xD4A4900 + count)
    result = []
    for i in range(count):
        priority = 7 if pattern == 'fifo' else (15-i%16 if pattern == 'descending' else rng.randrange(16))
        length = 1 if i == 0 else (255 if i%73 == 0 else rng.randrange(1, 9))
        kind, active = i%2, 1
        if i and i%113 == 0:
            defect = (i//113)%4
            if defect == 0: active = 0
            elif defect == 1: length = 0
            else: kind = defect
        ram, bram = 0x200000 + i*4096, 0x8000 + (i%4)*0x1000
        descriptor = (ram << 32) | (bram << 16) | (length << 8) | (priority << 4) | (kind << 2) | ((i%2) << 1) | active
        status = (0 if active and length else 1) | (2 if kind >= 2 else 0)
        result.append((descriptor, status))
    return result


def source_hashes():
    return {name: hashlib.sha256((HDL / name).read_bytes()).hexdigest()
            for name in ('dma_burst.v', 'dma_priority.v')}


def fingerprint():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def check(pattern='mixed', count=1024, stalls=True, fill=True, max_cycles=40000, simulator=None):
    if count < 258 and fill:
        raise ValueError('Full-queue coverage requires an active descriptor and 256 queued plus pending input')
    simulator = simulator or Simulator()
    rows = vectors(pattern, count)
    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    name = f'dma_priority_simulation_{fingerprint()[:12]}_{pattern}_{count}_{int(stalls)}_{int(fill)}_{max_cycles}.json'
    evidence = directory / name
    if evidence.exists(): evidence.unlink()
    with TemporaryDirectory(prefix='dma-priority-', dir=directory) as temporary:
        temporary = Path(temporary)
        jobs, statuses, executable = (temporary / name for name in ('jobs.hex', 'statuses.hex', 'simulation.vvp'))
        jobs.write_text(''.join(f'{desc:016x}\n' for desc, status in rows), encoding='ascii')
        statuses.write_text(''.join(f'{status:x}\n' for desc, status in rows), encoding='ascii')
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_dma_priority',
            f'-Ptb_dma_priority.JOBS={count}', f'-Ptb_dma_priority.STALLS={int(stalls)}',
            f'-Ptb_dma_priority.FILL={int(fill)}', f'-Ptb_dma_priority.MAX_CYCLES={max_cycles}',
            '-o', simulator.path(executable), simulator.path(HDL / 'dma_burst.v'),
            simulator.path(HDL / 'dma_priority.v'), simulator.path(HDL / 'tb_dma_priority.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable),
            '+jobs=' + simulator.path(jobs), '+statuses=' + simulator.path(statuses)])
        assert f'PASS priority jobs={count} ' in output, output
        summary = next(line for line in output.splitlines() if line.startswith('PASS '))
        report = {'sources': source_hashes(), 'fingerprint': fingerprint(),
                  'testbench_sha256': hashlib.sha256((HDL / 'tb_dma_priority.v').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'jobs_sha256': hashlib.sha256(jobs.read_bytes()).hexdigest(),
                  'statuses_sha256': hashlib.sha256(statuses.read_bytes()).hexdigest(),
                  'pattern': pattern, 'jobs': count, 'stalls': bool(stalls), 'fill': bool(fill),
                  'max_cycles': max_cycles, 'summary': summary}
        evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return summary


CONFIGURATIONS = [('mixed', 1024, True, True, 40000), ('fifo', 512, False, True, 40000),
                  ('mixed', 512, True, False, 40000), ('descending', 768, True, True, 40000),
                  ('mixed', 512, True, True, 64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    simulator = Simulator()
    for configuration in (CONFIGURATIONS[:1] if args.quick else CONFIGURATIONS):
        print(check(*configuration, simulator=simulator), flush=True)
    print('PASS: queued priority/FIFO, complete descriptors, pool reuse, full backpressure, payloads and reset.', flush=True)


if __name__ == '__main__':
    main()
