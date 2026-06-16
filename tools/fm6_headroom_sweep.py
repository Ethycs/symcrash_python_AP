"""Packet E2 sweep harness: FM6 gateway carrier-level distortion sweep.

Renders a pure sine carrier (algorithm 1, modulators silent) through the
full executor FM6 path (FM6 -> FWPT -> coeff ops -> IDWPT) at carrier
levels 0.3 .. 1.0 and reports FFT harmonic ratios (H2/H1, H3/H1, THD-ish).

Usage:
    pixi run python _fm6_headroom_sweep.py [--reference]
"""

import sys

import numpy as np

from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm import vm_step_block, VMState
from symcrash_python_ap.vm import hotpath_jit as hot

SAMPLE_RATE = 48000
BLOCK = 64
NOTE = 69          # A4 = 440 Hz
F0 = 440.0
NUM_BLOCKS = 96    # 6144 samples
SKIP_BLOCKS = 24   # skip IDWPT warmup / attack
FFT_LEN = 4096


def render_level(level_q15: int) -> np.ndarray:
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=NOTE, cents=0)
    out = []
    for _ in range(NUM_BLOCKS):
        frame = VoiceFrame(
            voice_id=0,
            note=NOTE,
            f0_hz=F0,
            coeff_osc_mode=3,  # OSC_MODE_FM6
            gain_q15=32767,
            fm_algorithm_id=1,             # single carrier (Op1)
            fm_op_ratios=[65536] * 6,
            fm_op_levels=[level_q15, 0, 0, 0, 0, 0],  # pure sine carrier
            fm_feedback_depth=0,
            is_active=True,
        )
        state, block = vm_step_block(state, frame)
        out.append(np.asarray(block, dtype=np.int64).copy())
    return np.concatenate(out)


def harmonic_mags(audio: np.ndarray):
    seg = audio[SKIP_BLOCKS * BLOCK: SKIP_BLOCKS * BLOCK + FFT_LEN].astype(np.float64)
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    bin_hz = SAMPLE_RATE / len(seg)

    def peak(freq):
        center = int(round(freq / bin_hz))
        lo, hi = max(center - 3, 0), min(center + 4, len(spec))
        return spec[lo:hi].max()

    return [peak(F0 * h) for h in (1, 2, 3, 4, 5)]


def main():
    if "--reference" in sys.argv:
        hot.set_jit_enabled(False)
        path = "reference"
    else:
        path = "jit" if hot.jit_enabled() else "reference"

    print(f"FM6 gateway sweep (path={path})")
    print(f"{'level':>6} {'sum':>6} {'peak':>6} {'H2/H1 %':>9} {'H3/H1 %':>9} "
          f"{'H4/H1 %':>9} {'H5/H1 %':>9}")
    for lvl in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        q15 = int(round(lvl * 32767))
        audio = render_level(q15)
        h = harmonic_mags(audio)
        peak = np.abs(audio[SKIP_BLOCKS * BLOCK:]).max() / 32767.0
        print(f"{lvl:>6.2f} {q15:>6d} {peak:>6.3f} "
              f"{100*h[1]/h[0]:>9.3f} {100*h[2]/h[0]:>9.3f} "
              f"{100*h[3]/h[0]:>9.3f} {100*h[4]/h[0]:>9.3f}")


if __name__ == "__main__":
    main()
