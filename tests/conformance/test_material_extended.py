"""Behavior and fixed-point contracts for material programs 6-10."""

import hashlib
import math
import struct

import numpy as np
import pytest

from symcrash_python_ap.frames import ParamFrame
from symcrash_python_ap.material.material_engine import MaterialEngine, SVFState
from symcrash_python_ap.material.svf import coefficients, process_sample, div_rne


def cutoff_control(hz):
    return round(math.log(hz / 20, 1000) * 32767)


def render_filter(signal, cutoff=1000, resonance=0, mode=0):
    engine = MaterialEngine()
    output = []
    for offset in range(0, len(signal), 64):
        block = signal[offset:offset + 64]
        left, right = engine.step_block(
            block, [0] * 64, ParamFrame(), 9,
            filter_cutoff_q15=cutoff_control(cutoff),
            filter_reso_q15=resonance, filter_mode=mode,
        )
        assert right == [0] * 64  # No crosstalk through state.
        output.extend(left)
    return np.array(output, dtype=np.float64)


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("cutoff", [20, 1000, 20000])
@pytest.mark.parametrize("resonance", [0, 16000, 32767])
def test_filter_matches_independent_trapezoidal_reference(mode, cutoff, resonance):
    # Low amplitude avoids the intentionally bounded output/state rails.
    signal = np.random.default_rng(43).integers(-100, 101, 4096).tolist()
    output = render_filter(signal, cutoff, resonance, mode)
    fc = 20 * 1000 ** (cutoff_control(cutoff) / 32767)
    g = math.tan(math.pi * fc / 48000)
    k = 1 / (0.5 + 24.5 * resonance / 32767)
    s1 = s2 = 0.0
    reference = []
    for sample in signal:
        # Solve the two trapezoidal integrator equations directly, independently
        # of the production a1/a2/a3 recurrence and ROM interpolation.
        bp = (s1 + g * (sample - s2)) / (1 + g * (g + k))
        lp = s2 + g * bp
        hp = sample - k * bp - lp
        s1, s2 = 2 * bp - s1, 2 * lp - s2
        reference.append((lp, hp, bp, lp + hp)[mode])
    assert np.max(np.abs(output - reference)) < 2.0


@pytest.mark.parametrize("mode,frequency,expected", [
    (0, 100, (0.95, 1.02)), (0, 10000, (0, 0.02)),
    (1, 100, (0, 0.02)), (1, 10000, (0.95, 1.02)),
    (2, 1000, (0.45, 0.55)), (3, 1000, (0, 0.02)),
])
def test_filter_frequency_response(mode, frequency, expected):
    time = np.arange(8192) / 48000
    signal = np.rint(4000 * np.sin(2 * np.pi * frequency * time)).astype(int)
    output = render_filter(signal.tolist(), mode=mode)
    gain = np.linalg.norm(output[4096:]) / np.linalg.norm(signal[4096:])
    assert expected[0] <= gain <= expected[1]


def test_more_resonance_increases_cutoff_peak():
    signal = np.rint(200 * np.sin(2 * np.pi * 1000 * np.arange(8192) / 48000)).astype(int).tolist()
    energies = [np.linalg.norm(render_filter(signal, resonance=r)[4096:])
                for r in (0, 8000, 16000, 32767)]
    assert all(b > a * 1.5 for a, b in zip(energies, energies[1:]))


@pytest.mark.parametrize("mode", range(4))
def test_filter_extremes_and_parameter_jumps_are_bounded(mode):
    state = SVFState()
    for cutoff in (0, 32767, -100, 50000, 16000):
        for resonance in (0, 32767, -100, 50000):
            coeff = coefficients(cutoff, resonance)
            for x in ([32767, -32768] * 256 + [0] * 256):
                y = process_sample(x, state, coeff, mode)
                assert isinstance(y, int) and -32768 <= y <= 32767
                assert -(1 << 31) <= state.ic1eq < (1 << 31)
                assert -(1 << 31) <= state.ic2eq < (1 << 31)


@pytest.mark.parametrize("numerator,denominator,expected", [
    (1, 2, 0), (3, 2, 2), (5, 2, 2), (-1, 2, 0), (-3, 2, -2), (-5, 2, -2),
])
def test_division_rounds_ties_to_even(numerator, denominator, expected):
    assert div_rne(numerator, denominator) == expected


def test_material_render_needs_no_floating_transcendentals(monkeypatch):
    from symcrash_python_ap.wavelet import sine_core
    monkeypatch.setattr(sine_core, "_SINE_LUT", None)
    def forbidden(*args):
        raise AssertionError("Floating-point transcendental used during playback")

    for name in ("sin", "cos", "tan", "log", "log2", "exp"):
        monkeypatch.setattr(math, name, forbidden)
    engine = MaterialEngine()  # Include a cold start, not just a warmed cache.
    for program in range(11):
        engine.step_block([100] * 64, [-100] * 64, ParamFrame(), program)


def test_exciter_replays_independently_of_global_rng():
    import random
    a, b, c = MaterialEngine(seed=42), MaterialEngine(seed=42), MaterialEngine(seed=43)
    for _ in range(2):
        a.excite_karplus(16000)
        for _ in range(99):
            random.random()
        b.excite_karplus(16000)
        c.excite_karplus(16000)
        assert a.ks_L.buffer == b.ks_L.buffer
        assert a.ks_R.buffer == b.ks_R.buffer
        assert a.ks_L.buffer != c.ks_L.buffer
        assert a.ks_L.buffer != a.ks_R.buffer


def material_trace(program):
    engine = MaterialEngine(seed=173)
    engine.set_karplus_pitch(220)
    engine.excite_karplus(16000)
    trace = bytearray()
    for block in range(192):
        # Long enough for delay, chorus and reverb buffers to become audible;
        # asymmetric channels, changing parameters, followed by a silent tail.
        left = [((block * 64 + i) * 7919 % 24001) - 12000 for i in range(64)]
        right = [x // -3 for x in left]
        if block > 95:
            left = right = [0] * 64
        params = ParamFrame(drive_q15=2000, tone_q15=12000 + (block % 4) * 4000,
                            pressure_q15=24000, bias_q15=-12000 if block < 64 else 12000)
        out_l, out_r = engine.step_block(
            left, right, params, program, filter_cutoff_q15=10000 + (block % 4) * 6000,
            filter_reso_q15=16000, filter_mode=(block // 48) % 4,
        )
        trace.extend(struct.pack("<128h", *out_l, *out_r))
    return hashlib.sha256(trace).hexdigest()


# Frozen after the SVF reference check and audited proportional-slew correction.
# See tests/artifacts/regen_reports/material_dynamics_20261030.txt.
GOLDEN = {
    6: "81a6b6452862b4e4a19168d364444a71b8351b4fe1f52bfa1242d0f267d067af",
    7: "fcc58b5cf6e569f06c9ef3b4beace3133125f2539029f7b8271cb8c237294a88",
    8: "847c0efa8c62cf9097361533e74afdb681af8d6e1b15597228f4fec35b10112c",
    9: "30e3ae153279ba526fb73606d45d591458edf40b531c465abc93311fbb603844",
    10: "307ef646be8df63883607f8b0d24e71a7a81b72106740e25b5be478182a65b34",
}


@pytest.mark.parametrize("program", range(6, 11))
def test_extended_material_golden_trace(program):
    assert material_trace(program) == GOLDEN[program]
