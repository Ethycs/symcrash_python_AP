"""Measure the real control-to-stereo live path without opening an audio device.

Includes queued controls, VoiceEngine, VM, pan/mix, Material and final Safety.
Preset compilation and construction are startup work. First-block time is
reported separately; sustained measurements follow warmup. Reference replay
checks audio, not whole-host checkpoint equivalence. An observed deadline miss
makes the command fail, even when the mean passes. This is a desktop observation,
not a worst-case device or FPGA timing guarantee.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import time

import numpy as np

from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.preset_compiler import compiled_preset_from_ir
from symcrash_python_ap.primitives import BLOCK_SIZE, SAMPLE_RATE_HZ, Q15_ONE
from symcrash_python_ap.vm import hotpath_jit as hot

ROOT = Path(__file__).resolve().parents[1]
BUDGET_US = BLOCK_SIZE / SAMPLE_RATE_HZ * 1_000_000


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def make_renderer(preset, voices):
    if not 1 <= voices <= 16:
        raise ValueError('This workload supports 1..16 distinct notes')
    renderer = LiveSynthRenderer(max_voices=voices)
    renderer.select_preset(preset)
    for voice in range(voices):
        renderer.note_on(48 + voice * 3, 100)
    return renderer


def verify_audio(preset, voices, blocks):
    """Compare every stereo sample through attack, release and a new note."""
    original = hot.jit_enabled()
    reference = None
    try:
        for enabled in (False, True):
            hot.set_jit_enabled(enabled)
            renderer = make_renderer(preset, voices)
            audio = np.empty((blocks, 2, BLOCK_SIZE), dtype=np.int16)
            for block in range(blocks):
                if block == blocks // 3:
                    renderer.note_off(48)
                if block == 2 * blocks // 3:
                    renderer.note_on(48, 77)
                audio[block] = renderer.render_block()
            if not enabled:
                reference = audio
            elif not np.array_equal(reference, audio):
                mismatch = np.argwhere(reference != audio)[0].tolist()
                raise AssertionError(f'Live reference/JIT audio differs at block/channel/sample {mismatch}')
        return hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest()
    finally:
        hot.set_jit_enabled(original)


def timing_summary(times):
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Timing data must be a nonempty vector of finite, nonnegative microseconds')
    misses = int(np.count_nonzero(values > BUDGET_US))
    return dict(blocks=len(values), mean_us=float(values.mean()),
                p95_us=float(np.percentile(values, 95)), p99_us=float(np.percentile(values, 99)),
                max_us=float(values.max()), deadline_misses=misses,
                mean_within_budget=bool(values.mean() <= BUDGET_US),
                no_observed_deadline_misses=misses == 0)


def measure(preset, voices, blocks, warmup, profile=None):
    original = hot.jit_enabled()
    try:
        hot.set_jit_enabled(True)
        renderer = make_renderer(preset, voices)
        start = time.perf_counter_ns()
        renderer.render_block()
        first_us = (time.perf_counter_ns() - start) / 1000
        for _ in range(warmup):
            renderer.render_block()
        times = np.empty(blocks, dtype=np.float64)
        audio = np.empty((blocks, 2, BLOCK_SIZE), dtype=np.int16)
        minimum_voices = voices
        for block in range(blocks):
            start = time.perf_counter_ns()
            output = renderer.render_block()
            times[block] = (time.perf_counter_ns() - start) / 1000
            audio[block] = output
            minimum_voices = min(minimum_voices, len(renderer.last_engine_output.render_batch))
        if minimum_voices != voices or not np.any(audio):
            raise AssertionError('Benchmark lost sounding voices or rendered only silence')
        result = timing_summary(times)
        result.update(voices=voices, warmup_blocks=warmup, first_block_us=first_us,
                      minimum_rendered_voices=minimum_voices,
                      material_program=renderer.stage._material_id,
                      filter_enabled=any(item.frame.filter_cutoff_q15 < Q15_ONE
                          for item in renderer.last_engine_output.render_batch),
                      audio_sha256=hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest())
        if profile is not None:
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(100):
                renderer.render_block()
            profiler.disable()
            with Path(profile).open('w', encoding='utf-8') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(50)
        return result, times, audio
    finally:
        hot.set_jit_enabled(original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', type=Path, default=ROOT / 'presets/instruments/acid_bass_v0.yaml')
    parser.add_argument('--voices', type=positive, nargs='+', default=[1, 4])
    parser.add_argument('--blocks', type=positive, default=750)
    parser.add_argument('--warmup', type=positive, default=100)
    parser.add_argument('--verify-blocks', type=positive, default=48)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; never overwritten')
    args = parser.parse_args(argv)
    if not hot.NUMBA_AVAILABLE:
        parser.error('Numba is required for the accelerated live benchmark')
    if any(voices > 16 for voices in args.voices) or len(set(args.voices)) != len(args.voices):
        parser.error('--voices must contain distinct values in 1..16')
    if args.verify_blocks < 3:
        parser.error('--verify-blocks must be at least 3 to cover note transitions')
    args.output.mkdir(parents=True, exist_ok=False)
    preset = compiled_preset_from_ir(load_instrument(str(args.preset)))
    report = dict(scope='desktop live control-to-stereo; no device I/O',
                  sample_rate=SAMPLE_RATE_HZ, block_size=BLOCK_SIZE, budget_us=BUDGET_US,
                  python=platform.python_version(), platform=platform.platform(),
                  first_block_context='after construction and reference replay; resources/JIT may already be warm',
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preset=str(args.preset.resolve()),
                  preset_sha256=hashlib.sha256(args.preset.read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((ROOT / 'src/symcrash_python_ap').rglob('*.py'))}, results=[])
    for voices in args.voices:
        replay_hash = verify_audio(preset, voices, args.verify_blocks)
        result, times, audio = measure(preset, voices, args.blocks, args.warmup,
                                       args.output / f'voices-{voices}-profile.txt')
        result.update(reference_jit_blocks=args.verify_blocks, reference_jit_audio_sha256=replay_hash)
        np.savez(args.output / f'voices-{voices}.npz', times_us=times, audio=audio)
        report['results'].append(result)
        print(json.dumps(result), flush=True)
    report['passed'] = all(row['no_observed_deadline_misses'] for row in report['results'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: no observed sustained misses' if report['passed'] else 'FAIL: observed sustained deadline misses')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
