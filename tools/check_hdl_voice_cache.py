"""Check complete state pages, isolation, dirty retries and real DMA deadlines."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import digest
from tools.check_hdl_idwpt import Simulator

HDL = ROOT / 'src/symcrash_verilog'
SOURCE = HDL / 'voice_state_cache.v'
BENCH = HDL / 'tb_voice_state_cache.v'
DIRECTORY = ROOT / 'build/hdl-tests'


def check(latency=100, stalls=False, source=None, bench=None, directory=None, simulator=None):
    if latency not in (0, 20, 100):
        raise ValueError('Qualified cache adapter latencies are 0, 20 and 100 clocks')
    source = source or SOURCE
    bench = bench or BENCH
    directory = directory or DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    paths = [source, *(HDL / name for name in ('dma_window.v', 'dma_priority.v', 'dma_burst.v')), bench]
    sources = {p.name: digest(p) for p in paths}
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    name = f'voice_cache_{key[:12]}_l{latency}_s{int(stalls)}'
    log, evidence_path = directory / (name + '.log'), directory / (name + '.json')
    if evidence_path.exists():
        evidence_path.unlink()
    simulator = simulator or Simulator()
    with TemporaryDirectory(prefix='voice-cache-', dir=directory) as temporary:
        binary = Path(temporary) / 'simulation.vvp'
        try:
            simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_voice_state_cache',
                f'-Ptb_voice_state_cache.STALLS={int(stalls)}', f'-Ptb_voice_state_cache.LATENCY={latency}',
                '-o', simulator.path(binary), *(simulator.path(p) for p in paths)])
            output = simulator.run(simulator.runtime + [simulator.path(binary)])
        except (AssertionError, RuntimeError) as exc:
            log.write_text(str(exc), encoding='utf-8')
            raise
    log.write_text(output, encoding='utf-8')
    summary = next(line for line in output.splitlines() if line.startswith('PASS voice_cache '))
    assert all(digest(p) == sources[p.name] for p in paths)
    fields = {name: int(value) for name, value in re.findall(r'(\w+)=(\d+)', summary)}
    assert fields['stalls'] == int(stalls) and fields['latency'] == latency
    assert fields['pages'] == 16 and fields['page_bytes'] == 2048 and fields['credits'] == 8
    assert fields['max_window'] <= 40000 and fields['late_windows'] == 2 and fields['max_late'] >= 40000
    assert fields['resets'] == 5 and fields['payload_resets'] == 4
    assert fields['readonly_reads'] == 128 and fields['owner_conflicts'] == 2
    report = {'sources': sources, 'fingerprint': key, 'runner_sha256': digest(Path(__file__)),
        'latency': latency, 'stalls': bool(stalls), 'credits': 8, 'window_cycles': 40000,
        'summary': summary, 'counts': fields, 'log': log.name, 'log_sha256': digest(log)}
    evidence_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    profiles = [(100, True)] if args.quick else [(l, s) for l in (0, 20, 100) for s in (False, True)]
    rows = []
    for latency, stalls in profiles:
        rows.append(check(latency, stalls))
        print(rows[-1]['summary'], flush=True)
    (DIRECTORY / 'voice_cache_matrix.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
