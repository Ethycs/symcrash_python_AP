"""Check retained voice-state rendering through the real cache and DMA window."""
import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.build_hdl_dma import digest
from tools.check_hdl_idwpt import Simulator, reference_audio
from tools.check_hdl_idwpt_runtime import mixed_cases
from symcrash_python_ap.vm.hotpath_jit import idwpt_polyphase

HDL = ROOT / 'src/symcrash_verilog'
DIRECTORY = ROOT / 'build/hdl-tests/voice_stream'
SOURCE_NAMES = ('voice_cache_stream.v', 'voice_cache_renderer.v', 'voice_state_cache.v',
    'ccs_idwpt_runtime.v', 'dma_window.v', 'dma_priority.v', 'dma_burst.v', 'tb_voice_cache_stream.v')


def write_words(path, values):
    data = np.asarray(values, dtype='<u2').tobytes()
    path.write_text(''.join(f'{int.from_bytes(data[i:i+16], "little"):032x}\n'
        for i in range(0, len(data), 16)), encoding='ascii')


def prepare_vectors(folder, headroom=2, exhaustive=False, limit=0, dense=False):
    if headroom not in range(16) or limit < 0 or (limit and dense):
        raise ValueError('Use headroom0..15 and either a prefix limit or dense selection')
    folder.mkdir(parents=True, exist_ok=True)
    cases = mixed_cases(exhaustive)
    if dense:
        cases = cases[:10] + cases[-10:]
    elif limit:
        cases = cases[:limit]
    pages, updates, expected, modes, offsets = [], [], [], [], []
    for i, coefficients in enumerate(cases):
        mode = {2: 0, 4: 1, 8: 2}[coefficients.shape[1]]
        words = 16 << mode
        offset = (0, 128 - words, 17)[i % 3]
        page = ((np.arange(1024, dtype=np.uint64) * 0x91d5 + i * 0x1777 + 0x8001) % 65536).astype('<u2')
        page[offset * 8:offset * 8 + coefficients.size] = coefficients.ravel().view(np.uint16)
        changed = np.bitwise_xor(coefficients.view(np.uint16), np.uint16(0x5a5a)).view(np.int16)
        update = page.copy()
        update[offset * 8:offset * 8 + coefficients.size] = changed.ravel().view(np.uint16)
        pages.append(page)
        updates.append(update)
        modes.append(mode)
        offsets.append(offset)
        for values in (coefficients, changed):
            oracle = reference_audio(values, headroom)
            actual = np.empty(64, dtype=np.int16)
            idwpt_polyphase(values.ravel(), actual, headroom)
            np.testing.assert_array_equal(actual, oracle)
            expected.append(oracle)
    write_words(folder / 'pages.hex', pages)
    write_words(folder / 'updates.hex', updates)
    (folder / 'expected.hex').write_text(''.join(f'{int(v) & 65535:04x}\n'
        for v in np.asarray(expected).flat), encoding='ascii')
    (folder / 'modes.hex').write_text(''.join(f'{v:x}\n' for v in modes), encoding='ascii')
    (folder / 'offsets.hex').write_text(''.join(f'{v:02x}\n' for v in offsets), encoding='ascii')
    return {'cases': len(cases), 'state_writes': sum(16 << modes[i] for i in range(0, len(cases), 4))}


def validate_summary(summary, cases, headroom, latency, stalls, state_writes):
    if not summary.startswith('PASS voice_stream '):
        raise RuntimeError('Voice stream did not report successful completion')
    fields = {name: int(value) for name, value in re.findall(r'(\w+)=(\d+)', summary)}
    frames = 5 * cases + max(cases - 16, 0) + 3 * ((cases + 7) // 8) + (cases + 3) // 4 + 9
    required = {'cases': cases, 'headroom': headroom, 'latency': latency, 'stalls': int(stalls),
        'frames': frames, 'samples': frames * 64, 'resets': 4, 'state_reads': cases * 128,
        'state_writes': state_writes, 'conflicts': 1, 'prefetch_conflicts': 1}
    if any(fields.get(name) != value for name, value in required.items()):
        raise RuntimeError(f'Incomplete or incompatible voice-stream coverage: {summary}')
    if not 0 < fields.get('max_window', 0) <= 40000:
        raise RuntimeError(f'Voice-stream DMA window exceeded its budget: {summary}')
    service = {mode: (int(low), int(high)) for mode, low, high in re.findall(r'\b(K[248])=(\d+)\.\.(\d+)', summary)}
    if set(service) != {'K2', 'K4', 'K8'}:
        raise RuntimeError('Missing per-mode service measurements')
    return fields, service


def check(headroom=2, latency=20, stalls=False, exhaustive=False, limit=0, dense=False,
          source_dir=None, directory=None, simulator=None):
    if latency not in (0, 20, 100):
        raise ValueError('Qualified adapter latencies are 0, 20 and 100 clocks')
    source_dir = Path(source_dir) if source_dir is not None else HDL
    directory = Path(directory) if directory is not None else DIRECTORY
    paths = [source_dir / name if (source_dir / name).is_file() else HDL / name for name in SOURCE_NAMES]
    sources = {p.name: digest(p) for p in paths}
    suffix = 'dense20' if dense else f'n{limit}'
    folder = directory / f'profile_h{headroom}_l{latency}_s{int(stalls)}_e{int(exhaustive)}_{suffix}'
    vectors = prepare_vectors(folder, headroom, exhaustive, limit, dense)
    evidence_path = folder / 'inventory.json'
    if evidence_path.exists():
        evidence_path.unlink()
    simulator = simulator or Simulator()
    log = folder / 'simulation.log'
    with TemporaryDirectory(prefix='simulation-', dir=folder) as temporary:
        binary = Path(temporary) / 'simulation.vvp'
        try:
            simulator.run(simulator.compiler + ['-g2012', '-s', 'tb_voice_cache_stream',
                f'-Ptb_voice_cache_stream.CASES={vectors["cases"]}', f'-Ptb_voice_cache_stream.HEADROOM_BITS={headroom}',
                f'-Ptb_voice_cache_stream.LATENCY={latency}', f'-Ptb_voice_cache_stream.STALLS={int(stalls)}',
                '-o', simulator.path(binary), *(simulator.path(p) for p in paths)])
            output = simulator.run(simulator.runtime + [simulator.path(binary), *(f'+{name}=' + simulator.path(folder / f'{file}.hex')
                for name, file in (('pages', 'pages'), ('updates', 'updates'), ('expected', 'expected'), ('modes', 'modes'), ('offsets', 'offsets')))])
        except (AssertionError, RuntimeError) as exc:
            log.write_text(str(exc), encoding='utf-8')
            raise
    log.write_text(output, encoding='utf-8')
    summary = next(line for line in output.splitlines() if line.startswith('PASS voice_stream '))
    fields, service = validate_summary(summary, vectors['cases'], headroom, latency, stalls, vectors['state_writes'])
    assert all(digest(path) == sources[path.name] for path in paths)
    report = {'sources': sources, 'runner_sha256': digest(Path(__file__)),
        'reference_sha256': digest(ROOT / 'tools/audit_lifting_arithmetic.py'),
        'vector_runner_sha256': digest(ROOT / 'tools/check_hdl_idwpt.py'),
        'case_runner_sha256': digest(ROOT / 'tools/check_hdl_idwpt_runtime.py'),
        'cases': vectors['cases'], 'headroom': headroom, 'latency': latency, 'stalls': bool(stalls),
        'exhaustive': bool(exhaustive), 'dense': bool(dense), 'summary': summary, 'counts': fields,
        'service_cycles': service, 'artifacts': {p.name: digest(p) for p in folder.iterdir() if p.suffix in {'.hex', '.log'}}}
    evidence_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HDL)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args()
    profiles = [(2, 100, True)] if args.quick else [(2, l, s) for l in (0, 20, 100) for s in (False, True)] + [(h, 100, s) for h in (0, 15) for s in (False, True)]
    rows = []
    for headroom, latency, stalls in profiles:
        rows.append(check(headroom, latency, stalls, dense=args.dense, source_dir=args.source_dir, directory=args.directory))
        print(rows[-1]['summary'], flush=True)
        (args.directory / 'matrix.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
