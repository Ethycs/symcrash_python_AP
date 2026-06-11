"""
OEIS partial-selection wiring tests (E7 packet, handoff 2026-06-11).

Covers the full chain: preset YAML `oeis:` schema -> InstrumentIR ->
extract_oeis_params -> VoiceFrame fields -> decode -> executor harmonic
weights. The engine side (oeis_tables, set_oeis, executor guard) predates
this wiring; these tests pin the authoring path that makes it reachable.
"""

import numpy as np
import pytest

from symcrash_python_ap.instrument_ir import (
    load_instrument,
    load_instrument_from_dict,
    extract_oeis_params,
)
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.oeis_tables import OEISSequenceID


SAMPLE_RATE = 48000
BLOCK_SIZE = 64


# =============================================================================
# Schema -> IR -> extractor
# =============================================================================

class TestOEISSchema:
    """`oeis:` in preset YAML reaches NodeParams and the extractor."""

    @pytest.mark.parametrize("path,expected_id", [
        ("presets/instruments/oeis_bell_v0.yaml", int(OEISSequenceID.PRIMES)),
        ("presets/instruments/oeis_pad_v0.yaml", int(OEISSequenceID.BEATTY_PHI)),
        ("presets/instruments/oeis_bass_v0.yaml", int(OEISSequenceID.FIBONACCI)),
    ])
    def test_oeis_presets_carry_sequence(self, path, expected_id):
        ir = load_instrument(path)
        params = extract_oeis_params(ir)
        assert params is not None, f"{path} should declare an OEIS sequence"
        assert params['sequence_id'] == expected_id
        assert params['rolloff_db'] == pytest.approx(-3.0)

    def test_sequence_name_and_int_forms(self):
        base = {
            'instrument_ir_v0': {
                'meta': {'id': 'oeis-test', 'name': 'T', 'version': 1, 'seed': 1},
                'audio': {'sample_rate': SAMPLE_RATE, 'block_size': BLOCK_SIZE},
                'graph': {
                    'osc1': {'type': 'Osc',
                             'oeis': {'sequence': 'mersenne', 'rolloff_db': -6.0},
                             'output': 'amp1'},
                    'amp1': {'type': 'Amp', 'attack': 1.0, 'decay': 100.0,
                             'sustain': 0.5, 'release': 50.0},
                },
            }
        }
        ir = load_instrument_from_dict(base)
        params = extract_oeis_params(ir)
        assert params == {'sequence_id': int(OEISSequenceID.MERSENNE),
                          'rolloff_db': -6.0}

        base['instrument_ir_v0']['graph']['osc1']['oeis'] = {'sequence': 3}
        ir = load_instrument_from_dict(base)
        assert extract_oeis_params(ir)['sequence_id'] == int(OEISSequenceID.TRIANGULAR)

    def test_absent_oeis_extracts_none(self):
        ir = load_instrument("presets/instruments/dx_marimba_v0.yaml")
        assert extract_oeis_params(ir) is None


# =============================================================================
# Frame -> decode -> executor -> spectrum
# =============================================================================

def _render_oeis(sequence_id: int, blocks: int = 256) -> np.ndarray:
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
            oeis_sequence_id=sequence_id,
            oeis_rolloff_db=-3.0,
        )
        state, output = vm_step_block(state, frame)
        samples.extend(output.tolist())  # copy: vm reuses the output buffer
    return np.array(samples, dtype=np.float64)


def _harmonic_mags(samples: np.ndarray, f0: float = 261.6256):
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / SAMPLE_RATE)

    def mag(harmonic: int) -> float:
        target = f0 * harmonic
        idx = int(np.argmin(np.abs(freqs - target)))
        return float(spectrum[max(0, idx - 2):idx + 3].max())

    return mag


class TestOEISRender:
    """OEIS frames drive the Mode B spectrum, deterministically."""

    def test_primes_spectrum(self):
        # PRIMES partials: 2, 3, 5, 7... -> H1 and H4 must be absent
        samples = _render_oeis(int(OEISSequenceID.PRIMES))
        # Skip the first 32 blocks of Mode B onset ramp
        mag = _harmonic_mags(samples[32 * BLOCK_SIZE:])
        h2, h3, h5 = mag(2), mag(3), mag(5)
        h1, h4 = mag(1), mag(4)
        assert h2 > 0 and h3 > 0 and h5 > 0
        for present in (h2, h3, h5):
            assert present > 20 * h1, f"H1 should be silenced (h1={h1}, present={present})"
            assert present > 20 * h4, f"H4 should be silenced (h4={h4}, present={present})"

    def test_render_is_deterministic(self):
        a = _render_oeis(int(OEISSequenceID.PRIMES), blocks=64)
        b = _render_oeis(int(OEISSequenceID.PRIMES), blocks=64)
        assert np.array_equal(a, b)

    def test_disabled_oeis_keeps_sine(self):
        # oeis_sequence_id=-1 -> waveform_type sine drives the spectrum
        samples = _render_oeis(-1, blocks=128)
        mag = _harmonic_mags(samples[32 * BLOCK_SIZE:])
        assert mag(1) > 20 * mag(2)
