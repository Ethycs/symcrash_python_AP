"""
Noise exciter tests (E4 packet, handoff 2026-06-11).

Covers the full chain: preset YAML `noise:` schema -> InstrumentIR ->
extract_noise_params -> VoiceFrame fields -> decode -> executor mix, plus
the module-level guarantees: counter-based determinism, color spectrum
shapes (white flat, pink ~-3 dB/oct, brown ~-6 dB/oct), envelope bounds,
and the bit-exact no-op for presets that author no noise.
"""

import numpy as np
import pytest

from symcrash_python_ap.instrument_ir import (
    load_instrument,
    load_instrument_from_dict,
    extract_noise_params,
)
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.primitives import Q15_ONE
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.noise_exciter import (
    NOISE_WHITE, NOISE_PINK, NOISE_BROWN,
    NoiseExciterState, noise_seed_base, white_block_q15,
    noise_env_q15, render_noise_block,
)


SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def _preset_dict(graph: dict) -> dict:
    return {
        'instrument_ir_v0': {
            'meta': {'id': 'noise-test', 'name': 'T', 'version': 1,
                     'seed': 0xC0FFEE},
            'audio': {'sample_rate': SAMPLE_RATE, 'block_size': BLOCK_SIZE},
            'graph': graph,
        }
    }


# =============================================================================
# Schema -> IR -> extractor
# =============================================================================

class TestNoiseSchema:
    """`noise:` in preset YAML reaches NodeParams and the extractor."""

    def test_noise_block_on_osc_node(self):
        ir = load_instrument_from_dict(_preset_dict({
            'osc1': {'type': 'Osc', 'waveform': 'sine', 'output': 'amp1',
                     'noise': {'color': 'pink', 'attack_ms': 2.0,
                               'decay_ms': 30.0, 'level': 0.4}},
            'amp1': {'type': 'Amp', 'attack': 1.0, 'decay': 100.0,
                     'sustain': 0.5, 'release': 50.0},
        }))
        params = extract_noise_params(ir)
        assert params == {
            'color': NOISE_PINK,
            'attack_ms': 2.0,
            'decay_ms': 30.0,
            'level_q15': round(0.4 * 32767),
            'seed_u32': 0xC0FFEE,
        }

    def test_noise_node_flat_form(self):
        ir = load_instrument_from_dict(_preset_dict({
            'osc1': {'type': 'Osc', 'waveform': 'sine', 'output': 'amp1'},
            'breath': {'type': 'Noise', 'color': 'brown', 'attack_ms': 5.0,
                       'decay_ms': 80.0, 'level': 0.3, 'output': 'amp1'},
            'amp1': {'type': 'Amp', 'attack': 1.0, 'decay': 100.0,
                     'sustain': 0.5, 'release': 50.0},
        }))
        params = extract_noise_params(ir)
        assert params['color'] == NOISE_BROWN
        assert params['attack_ms'] == 5.0
        assert params['decay_ms'] == 80.0
        assert params['level_q15'] == round(0.3 * 32767)

    def test_bare_noise_node_gets_default_level(self):
        ir = load_instrument_from_dict(_preset_dict({
            'hiss': {'type': 'Noise', 'output': 'amp1'},
            'amp1': {'type': 'Amp', 'attack': 1.0, 'decay': 100.0,
                     'sustain': 0.5, 'release': 50.0},
        }))
        params = extract_noise_params(ir)
        assert params['color'] == NOISE_WHITE
        assert params['level_q15'] == round(0.25 * 32767)

    def test_absent_noise_extracts_none(self):
        ir = load_instrument("presets/instruments/dx_marimba_v0.yaml")
        assert extract_noise_params(ir) is None

    def test_level_zero_extracts_none(self):
        ir = load_instrument_from_dict(_preset_dict({
            'osc1': {'type': 'Osc', 'waveform': 'sine', 'output': 'amp1',
                     'noise': {'color': 'white', 'level': 0.0}},
            'amp1': {'type': 'Amp', 'attack': 1.0, 'decay': 100.0,
                     'sustain': 0.5, 'release': 50.0},
        }))
        assert extract_noise_params(ir) is None


# =============================================================================
# Module: PRNG determinism + counter independence
# =============================================================================

class TestNoisePRNG:
    """Counter-based PRNG: deterministic, order-independent, seed-keyed."""

    def test_white_block_repeatable(self):
        seed = noise_seed_base(0x41434D42, 0)
        a = white_block_q15(seed, 7)
        b = white_block_q15(seed, 7)
        assert np.array_equal(a, b)

    def test_white_block_order_independent(self):
        seed = noise_seed_base(0x41434D42, 0)
        forward = [white_block_q15(seed, b).copy() for b in range(8)]
        backward = [white_block_q15(seed, b) for b in reversed(range(8))]
        for b in range(8):
            assert np.array_equal(forward[b], backward[7 - b])

    def test_seed_and_voice_change_stream(self):
        a = white_block_q15(noise_seed_base(1, 0), 0)
        b = white_block_q15(noise_seed_base(2, 0), 0)
        c = white_block_q15(noise_seed_base(1, 1), 0)
        assert not np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_white_range_is_q15(self):
        seed = noise_seed_base(0xDEAD, 3)
        x = np.concatenate([white_block_q15(seed, b) for b in range(64)])
        # WSE-STD-040 Section 9 conversion: [-16384, 16383]
        assert x.min() >= -16384 and x.max() <= 16383


# =============================================================================
# Module: color spectrum sanity
# =============================================================================

def _color_run(color: int, blocks: int = 512) -> np.ndarray:
    seed = noise_seed_base(0x41434D42, 0)
    ns = NoiseExciterState()
    out = [render_noise_block(ns, seed, b, color, Q15_ONE, 1, 10 ** 6)
           for b in range(blocks)]
    x = np.concatenate(out).astype(np.float64)
    return x[BLOCK_SIZE:]  # skip the attack-ramp block


def _octave_slope_db(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    bands = [(125, 250), (250, 500), (500, 1000), (1000, 2000),
             (2000, 4000), (4000, 8000), (8000, 16000)]
    power_db = [10 * np.log10(spec[(freqs >= lo) & (freqs < hi)].mean())
                for lo, hi in bands]
    return float(np.diff(power_db).mean())


class TestNoiseColors:
    """White flat, pink ~-3 dB/oct, brown ~-6 dB/oct; all bounded Q15."""

    def test_white_is_flat(self):
        assert abs(_octave_slope_db(_color_run(NOISE_WHITE))) < 1.0

    def test_pink_trend(self):
        slope = _octave_slope_db(_color_run(NOISE_PINK))
        assert -4.5 < slope < -1.5, f"pink slope {slope:.2f} dB/oct"

    def test_brown_steeper_than_pink(self):
        pink = _octave_slope_db(_color_run(NOISE_PINK))
        brown = _octave_slope_db(_color_run(NOISE_BROWN))
        assert brown < pink - 1.0, f"brown {brown:.2f} vs pink {pink:.2f}"

    @pytest.mark.parametrize("color", [NOISE_WHITE, NOISE_PINK, NOISE_BROWN])
    def test_output_bounded(self, color):
        x = _color_run(color, blocks=256)
        assert np.abs(x).max() <= Q15_ONE

    def test_colored_run_deterministic(self):
        a = _color_run(NOISE_PINK, blocks=64)
        b = _color_run(NOISE_PINK, blocks=64)
        assert np.array_equal(a, b)


# =============================================================================
# Module: envelope
# =============================================================================

class TestNoiseEnvelope:
    """One-shot linear attack/decay envelope, bounded and silent after."""

    def test_bounds_and_shape(self):
        attack, decay = 4, 16
        env = [noise_env_q15(b, attack, decay) for b in range(attack + decay + 8)]
        assert all(0 <= v <= Q15_ONE for v in env)
        assert env[0] == 0
        # Rising through attack, peak at the attack boundary
        assert env[1] < env[2] < env[3] < env[attack]
        assert env[attack] == Q15_ONE
        # Falling through decay, zero at and after the end
        assert env[attack + 1] < env[attack]
        assert all(v == 0 for v in env[attack + decay:])

    def test_render_skips_after_envelope(self):
        seed = noise_seed_base(1, 0)
        ns = NoiseExciterState()
        assert render_noise_block(ns, seed, 0, NOISE_WHITE, Q15_ONE, 2, 10) is not None
        assert render_noise_block(ns, seed, 11, NOISE_WHITE, Q15_ONE, 2, 10) is not None
        # attack(2) + decay(10) = 12 blocks -> envelope exhausted
        assert render_noise_block(ns, seed, 12, NOISE_WHITE, Q15_ONE, 2, 10) is None
        assert render_noise_block(ns, seed, 1000, NOISE_WHITE, Q15_ONE, 2, 10) is None

    def test_level_zero_renders_nothing(self):
        ns = NoiseExciterState()
        assert render_noise_block(ns, noise_seed_base(1, 0), 0,
                                  NOISE_WHITE, 0, 1, 100) is None


# =============================================================================
# Executor integration: mix, determinism, bit-exact no-op
# =============================================================================

def _render_voice(noise_kwargs: dict, blocks: int = 64) -> np.ndarray:
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=60)
    samples = []
    for _ in range(blocks):
        frame = VoiceFrame(
            note=60,
            coeff_osc_mode=1,  # Mode B
            H_request=16,
            gain_q15=16384,
            beta_target_q15=0,
            **noise_kwargs,
        )
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())  # copy: vm reuses the output buffer
    return np.array(samples, dtype=np.int32)


NOISE_FRAME = dict(
    noise_color=NOISE_PINK,
    noise_level_q15=12000,
    noise_attack_blocks=2,
    noise_decay_blocks=20,
    noise_seed_u32=0x41434D42,
)


class TestExecutorMix:
    """Noise frames reach vm_step_block and mix deterministically."""

    def test_render_is_deterministic(self):
        a = _render_voice(NOISE_FRAME)
        b = _render_voice(NOISE_FRAME)
        assert np.array_equal(a, b)

    def test_noise_changes_attack_not_tail(self):
        clean = _render_voice({})
        noisy = _render_voice(NOISE_FRAME)
        attack_end = (2 + 20) * BLOCK_SIZE
        # Audible broadband transient during the exciter envelope...
        assert not np.array_equal(noisy[:attack_end], clean[:attack_end])
        # ...and bit-exact with the clean voice after it decays to zero
        assert np.array_equal(noisy[attack_end:], clean[attack_end:])

    def test_level_zero_is_bitexact_noop(self):
        clean = _render_voice({})
        zeroed = _render_voice(dict(
            noise_color=NOISE_BROWN, noise_level_q15=0,
            noise_attack_blocks=5, noise_decay_blocks=50,
            noise_seed_u32=123,
        ))
        assert np.array_equal(clean, zeroed)

    def test_spectral_flatness_attack_vs_sustain(self):
        """The exciter raises broadband (high-frequency) energy during the
        attack window relative to the tonal sustain."""
        noisy = _render_voice(dict(
            noise_color=NOISE_WHITE, noise_level_q15=16000,
            noise_attack_blocks=2, noise_decay_blocks=20,
            noise_seed_u32=0x41434D42,
        ), blocks=128)

        def flatness(x):
            # Spectral flatness: geometric / arithmetic mean of the PSD.
            # ~1 for broadband noise, ->0 for a pure tone.
            psd = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
            psd = psd[1:]  # drop DC
            psd /= psd.sum()
            return float(np.exp(np.mean(np.log(psd + 1e-30))) / np.mean(psd))

        # Envelope-active window (peak at block 2, zero by block 22)
        attack = noisy[2 * BLOCK_SIZE:22 * BLOCK_SIZE]
        sustain = noisy[108 * BLOCK_SIZE:128 * BLOCK_SIZE]
        assert flatness(attack) > 10 * flatness(sustain)


# =============================================================================
# Snapshot/restore replay
# =============================================================================

class TestSnapshotReplay:
    """Pink/brown filter state survives VMState snapshot/restore.

    (Full vm_step_block Mode B replay is out of scope: ModeBState itself
    is not snapshotted - a pre-existing engine limitation independent of
    the noise exciter.)
    """

    def test_restore_continues_bitexact(self):
        seed = noise_seed_base(0x41434D42, 0)

        state = VMState.create(voice_id=0)
        for b in range(8):
            render_noise_block(state.noise_state, seed, b,
                               NOISE_PINK, Q15_ONE, 1, 10 ** 6)
        snap = state.snapshot()
        assert snap['noise_state'] is not None

        restored = VMState.restore(snap)
        for b in range(8, 16):
            a = render_noise_block(state.noise_state, seed, b,
                                   NOISE_PINK, Q15_ONE, 1, 10 ** 6)
            r = render_noise_block(restored.noise_state, seed, b,
                                   NOISE_PINK, Q15_ONE, 1, 10 ** 6)
            assert np.array_equal(a, r)

    def test_unused_noise_state_snapshots_none(self):
        state = VMState.create(voice_id=0)
        assert state.snapshot()['noise_state'] is None
