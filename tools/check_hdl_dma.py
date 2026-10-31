"""Execute the DMA RTL with byte-exact payloads and an ordered stalled RAM adapter.

This is the burst engine gate, not a PHY, prefetch or whole-device qualification.
The full run covers every descriptor length 1..255 in both directions.
"""
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


def cases(exhaustive=True):
    rows = []

    def add(count=16, kind=0, ram=0x1200, bram=0x8000, priority=5,
            interrupt=1, valid=1, scenario=0, fault=255):
        descriptor = (ram << 32) | (bram << 16) | (count << 8) | (priority << 4) | (kind << 2) | (interrupt << 1) | valid
        bad = (not valid or count == 0 or ram % 16 or bram % 16
               or ram + 16*count > 128*1024*1024 or bram + 16*count > 65536)
        status = int(bool(bad)) | (2 if kind in (2, 3) else 0)
        rows.append((descriptor, status, fault, scenario, count))

    for count in (range(1, 256) if exhaustive else (1, 2, 16, 255)):
        for kind in (0, 1):
            for scenario in (0, 1):
                add(count, kind, priority=count % 16, interrupt=count % 2, scenario=scenario)
    for kind in (0, 1):
        for count in (1, 16, 255):
            add(count, kind, ram=128*1024*1024-16*count, bram=65536-16*count)
            for fault in sorted({0, count//2, count-1}):
                add(count, kind, fault=fault, scenario=1)
        add(kind=kind, ram=0, bram=0)
        for scenario in (2, 3, 4):
            add(count=255 if scenario == 4 else 128, kind=kind, scenario=scenario)
        add(kind=kind, valid=0)
        add(kind=kind, count=0)
        for ram in (1, 15, 128*1024*1024, 0xfffffff0):
            add(kind=kind, ram=ram)
        for bram in (1, 15, 0xfff0):
            add(kind=kind, bram=bram)
    add(kind=2)
    add(kind=3)
    add(kind=3, valid=0, ram=1)
    return rows


def check(exhaustive=True, max_cycles=40000, simulator=None):
    simulator = simulator or Simulator()
    rows = cases(exhaustive)
    rng = random.Random(0xD0A5EED)
    directory = ROOT / 'build/hdl-tests'
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256((HDL / 'dma_burst.v').read_bytes()).hexdigest()
    evidence = directory / f'dma_burst_simulation_{source_hash[:12]}_{"full" if exhaustive else "quick"}_{max_cycles}.json'
    if evidence.exists():
        evidence.unlink()  # A failing rerun must not retain a previous pass.
    with TemporaryDirectory(prefix='dma-', dir=directory) as temporary:
        temporary = Path(temporary)
        configs, payload, executable = (temporary / name for name in ('configs.hex', 'payload.hex', 'simulation.vvp'))
        configs.write_text(''.join(f'{(desc << 64) | (status << 60) | (fault << 52) | (scenario << 48):032x}\n'
                                   for desc, status, fault, scenario, count in rows), encoding='ascii')
        payload.write_text(''.join(f'{rng.getrandbits(128) if i < row[4] else 0:032x}\n'
                                   for row in rows for i in range(255)), encoding='ascii')
        simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_dma_burst',
            f'-Ptb_dma_burst.CASES={len(rows)}', f'-Ptb_dma_burst.MAX_CYCLES={max_cycles}',
            '-o', simulator.path(executable), simulator.path(HDL / 'dma_burst.v'),
            simulator.path(HDL / 'tb_dma_burst.v')])
        output = simulator.run(simulator.runtime + [simulator.path(executable),
            '+configs=' + simulator.path(configs), '+payload=' + simulator.path(payload)])
        assert f'PASS DMA cases={len(rows)} ' in output, output
        summary = next(line for line in output.splitlines() if line.startswith('PASS '))
        report = {'source_sha256': source_hash,
                  'testbench_sha256': hashlib.sha256((HDL / 'tb_dma_burst.v').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'config_sha256': hashlib.sha256(configs.read_bytes()).hexdigest(),
                  'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
                  'all_lengths': bool(exhaustive), 'max_cycles': max_cycles,
                  'cases': len(rows), 'valid_transfer_bytes': sum(16*row[4] for row in rows if row[1] == 0),
                  'summary': summary}
        evidence.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Diagnostic subset of lengths')
    args = parser.parse_args()
    simulator = Simulator()
    print(check(not args.quick, simulator=simulator), flush=True)
    for window in (1, 2, 3, 4, 8):
        print(check(False, window, simulator), flush=True)
    print('PASS: both directions, payloads, bounds, stalls, completion, errors, deadline edges and reset.', flush=True)
    print('PHY, queue priority, staging ownership and prefetch deadlines remain separate gates.', flush=True)


if __name__ == '__main__':
    main()
