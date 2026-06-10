"""
Regression tests for the engine bugs found during analysis-driven preset
authoring (SESSION-HANDOFF-2026-06-09 §3):

1. waveform_type silently overrode custom partials: the matrix-LUT path was
   selected on waveform_type alone and never read mb_state.weights, so any
   PL waveform_type (SQUARE/SAW/TRIANGLE) rendered instead of the partials.
2. Preset `partials:` never reached the IR: _parse_node_params dropped the
   key, so Osc.params.harmonics stayed None after loading.
3. Per-block cents stepping (LFO vibrato / pitch bend) hard-reset Mode B
   phase every block, producing a ~750 Hz discontinuity comb that swamped
   the spectrum with inharmonic energy.
"""

import math

import numpy as np
import pytest

from symcrash_python_ap.fixed import midi_to_phase_inc_fixed
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.instrument_ir import (
    extract_partials_q15,
    load_instrument,
    load_instrument_from_dict,
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import PHASE_MASK, clear_basis_atlas
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.vm.waveform_matrix import WaveformType

FS = 48000
BLOCK_SIZE = 64

# Real-analysis violin profile (H2 << H4 — a saw/square render inverts this)
VIOLIN_PARTIALS = [0.94, 0.09, 0.21, 0.73, 0.11, 0.19, 0.04, 0.04]
VIOLIN_PARTIALS_Q15 = [round(v * 32767) for v in VIOLIN_PARTIALS]


def _make_frame(note=69, cents=0, waveform_type=0, partials=None, gain_q15=32767):
    return VoiceFrame(
        note=note,
        cents=cents,
        coeff_osc_mode=1,  # Mode B
        H_request=len(partials) if partials else 16,
        gain_q15=gain_q15,
        beta_target_q15=0,
        partial_amps_q15=list(partials) if partials else [],
        waveform_type=waveform_type,
    )


def _render(frames, note):
    clear_basis_atlas()
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=note)
    blocks = []
    for frame in frames:
        state, output = vm_step_block(state, frame)
        # vm_step_block reuses its output buffer — copy each block
        blocks.append(np.array(output, dtype=np.int16, copy=True))
    return state, np.concatenate(blocks)


def _band_energy(spectrum, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(spectrum[mask] ** 2))


# =========================================================================
# Bug 1: custom partials must win over waveform_type
# =========================================================================

class TestCustomHarmonicsWinOverWaveformType:

    def test_partials_identical_regardless_of_waveform_type(self):
        """With custom partials, waveform_type must not affect the output."""
        n_blocks = 40
        frames_saw = [
            _make_frame(waveform_type=WaveformType.SAW, partials=VIOLIN_PARTIALS_Q15)
            for _ in range(n_blocks)
        ]
        frames_sine = [
            _make_frame(waveform_type=WaveformType.SINE, partials=VIOLIN_PARTIALS_Q15)
            for _ in range(n_blocks)
        ]
        _, out_saw = _render(frames_saw, note=69)
        _, out_sine = _render(frames_sine, note=69)

        assert np.array_equal(out_saw, out_sine), (
            "waveform_type changed the render despite custom partials — "
            "the matrix-LUT path is bypassing mb_state.weights again"
        )

    def test_partials_render_matches_profile_not_saw(self):
        """Spectral check: the violin profile has H2 << H4; saw has H2 > H4."""
        n_blocks = 400  # ~0.53 s
        frames = [
            _make_frame(waveform_type=WaveformType.SAW, partials=VIOLIN_PARTIALS_Q15)
            for _ in range(n_blocks)
        ]
        _, out = _render(frames, note=69)  # A4 = 440 Hz

        # Skip the IDWPT latency / settle region
        samples = out[len(out) // 2:].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / FS)

        h2 = _band_energy(spectrum, freqs, 880 - 40, 880 + 40)
        h4 = _band_energy(spectrum, freqs, 1760 - 40, 1760 + 40)

        assert h4 > h2 * 4, (
            f"H4/H2 energy ratio {h4 / max(h2, 1e-12):.2f} too low — output "
            "resembles a saw/square, not the custom harmonic profile"
        )

    def test_plain_waveform_still_uses_matrix_path(self):
        """Without partials, a saw preset must still render (matrix path)."""
        n_blocks = 40
        frames = [_make_frame(waveform_type=WaveformType.SAW) for _ in range(n_blocks)]
        _, out = _render(frames, note=69)
        assert np.any(out != 0), "plain saw render is silent"


# =========================================================================
# Bug 2: partials must load from YAML into the IR
# =========================================================================

def _simple_form_dict(partials):
    return {
        'instrument_ir_v0': {
            'meta': {'id': 'test-partials', 'name': 'Test Partials'},
            'graph': {
                'osc1': {
                    'type': 'Osc',
                    'output': 'amp1',
                    'waveform': 'partials',
                    'partials': list(partials),
                },
                'amp1': {
                    'type': 'Amp',
                    'attack': 10.0,
                    'decay': 100.0,
                    'sustain': 0.7,
                    'release': 200.0,
                },
            },
        }
    }


class TestPartialsLoading:

    def test_simple_form_partials_reach_node_params(self):
        ir = load_instrument_from_dict(_simple_form_dict(VIOLIN_PARTIALS))
        osc_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Osc"]
        assert osc_nodes, "no Osc node parsed"
        harmonics = osc_nodes[0].params.harmonics
        assert harmonics is not None, "Osc.params.harmonics is None — loader gap"
        assert len(harmonics.amplitudes) == len(VIOLIN_PARTIALS)
        for got, expected in zip(harmonics.amplitudes, VIOLIN_PARTIALS):
            assert got == pytest.approx(expected, abs=1.0 / 32767)

    def test_canonical_layers_format_partials(self):
        data = {
            'instrument_ir_v0': {
                'meta': {'id': 'test-partials-canonical', 'name': 'Canonical'},
                'layers': {
                    'L0_graph': {
                        'nodes': [{
                            'id': 'osc1',
                            'type': 'Osc',
                            'params': {
                                'waveform': 'partials',
                                'partials': list(VIOLIN_PARTIALS),
                            },
                        }],
                        'edges': [],
                    },
                },
            }
        }
        ir = load_instrument_from_dict(data)
        amps = extract_partials_q15(ir)
        assert amps == VIOLIN_PARTIALS_Q15

    def test_extract_partials_q15_roundtrip(self):
        ir = load_instrument_from_dict(_simple_form_dict(VIOLIN_PARTIALS))
        amps = extract_partials_q15(ir)
        assert amps == VIOLIN_PARTIALS_Q15

    def test_extract_partials_q15_none_without_partials(self):
        data = _simple_form_dict(VIOLIN_PARTIALS)
        graph_osc = data['instrument_ir_v0']['graph']['osc1']
        graph_osc.pop('partials')
        graph_osc['waveform'] = 'saw'
        ir = load_instrument_from_dict(data)
        assert extract_partials_q15(ir) is None

    def test_violin_preset_file_loads_partials(self):
        from pathlib import Path
        preset = Path(__file__).parent.parent / "presets" / "instruments" / "violin_ir_v1.yaml"
        if not preset.exists():
            pytest.skip("violin_ir_v1.yaml not present")
        ir = load_instrument(str(preset))
        amps = extract_partials_q15(ir)
        assert amps is not None, "violin preset partials did not load"
        assert len(amps) == 12
        # Re-fitted profile (2026-06-09): dominant fundamental, weak H2
        assert amps[0] == 32767
        assert amps[1] < 32767 // 4


# =========================================================================
# Bug 3: per-block cents changes must be phase-continuous
# =========================================================================

class TestVibratoPhaseContinuity:

    def test_cents_change_does_not_reset_phase(self):
        """Mode B phase must accumulate across cents-only pitch changes."""
        note = 69
        cents_seq = [0, 15, 30]
        clear_basis_atlas()
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=note)

        phases = []
        for cents in cents_seq:
            frame = _make_frame(note=note, cents=cents, partials=VIOLIN_PARTIALS_Q15)
            state, _ = vm_step_block(state, frame)
            phases.append(state.mode_b_state.phase)

        inc0 = midi_to_phase_inc_fixed(note, cents_seq[0])
        inc1 = midi_to_phase_inc_fixed(note, cents_seq[1])
        inc2 = midi_to_phase_inc_fixed(note, cents_seq[2])
        assert len({inc0, inc1, inc2}) == 3, "cents steps must change phase_inc"

        # Block 1 starts from phase 0, so the per-block advance count falls
        # out of the observed phase without assuming block semantics.
        assert phases[0] % inc0 == 0 and phases[0] > 0
        advance_units = phases[0] // inc0

        expected_1 = (phases[0] + inc1 * advance_units) & PHASE_MASK
        expected_2 = (expected_1 + inc2 * advance_units) & PHASE_MASK
        assert phases[1] == expected_1, (
            "phase was reset on a cents-only change (vibrato retriggered the "
            "oscillator instead of taking the legato path)"
        )
        assert phases[2] == expected_2

    def test_note_change_still_retriggers(self):
        """A real note change (non-legato) must still hard-reset phase."""
        clear_basis_atlas()
        state = VMState.create(voice_id=0)
        state.reset_for_note_on(note=69)

        frame_a = _make_frame(note=69, partials=VIOLIN_PARTIALS_Q15)
        state, _ = vm_step_block(state, frame_a)

        frame_b = _make_frame(note=72, partials=VIOLIN_PARTIALS_Q15)
        state, _ = vm_step_block(state, frame_b)

        inc_b = midi_to_phase_inc_fixed(72, 0)
        # After a reset, phase restarts from 0 and advances one block's worth
        assert state.mode_b_state.phase % inc_b == 0, (
            "note change did not retrigger — non-legato reset semantics changed"
        )

    def test_per_block_partials_motion_is_phase_continuous(self):
        """Changing partials mid-note must move the spectrum (per-block
        re-latch) without resetting phase (no block-rate comb)."""
        note = 69
        bright = [32767, 26214, 19660, 14746, 11059, 8192, 6553, 4915]
        dark = [32767, 8192, 2048, 512, 0, 0, 0, 0]
        n_blocks = 300
        frames = []
        for i in range(n_blocks):
            t = i / (n_blocks - 1)
            amps = [round(b + (d - b) * t) for b, d in zip(bright, dark)]
            frames.append(_make_frame(note=note, partials=amps))
        _, out = _render(frames, note=note)

        def _centroid(seg):
            seg = seg.astype(np.float64)
            seg = (seg - seg.mean()) * np.hanning(len(seg))
            sp = np.abs(np.fft.rfft(seg)) ** 2
            fr = np.fft.rfftfreq(len(seg), 1.0 / FS)
            return float((fr * sp).sum() / max(sp.sum(), 1e-12))

        early = _centroid(out[len(out) // 8: len(out) // 4])
        late = _centroid(out[3 * len(out) // 4: 7 * len(out) // 8])
        assert late < early * 0.7, (
            f"spectrum did not darken ({early:.0f} -> {late:.0f} Hz): "
            "per-block partials are not re-latching"
        )

        # No 750 Hz block-rate comb from the re-latch
        seg = out[len(out) // 2: 3 * len(out) // 4].astype(np.float64)
        seg = (seg - seg.mean()) * np.hanning(len(seg))
        sp = np.abs(np.fft.rfft(seg))
        fr = np.fft.rfftfreq(len(seg), 1.0 / FS)
        fundamental = _band_energy(sp, fr, 440 - 60, 440 + 60)
        inter = _band_energy(sp, fr, 660, 780)
        assert inter < fundamental / 30, "re-latch injected block-rate hash"

    def test_phase_spread_preserves_magnitude_spectrum(self):
        """phase_spread changes the waveform, not the harmonic magnitudes."""
        n_blocks = 200
        aligned = [_make_frame(partials=VIOLIN_PARTIALS_Q15) for _ in range(n_blocks)]
        spread_frames = []
        for _ in range(n_blocks):
            f = _make_frame(partials=VIOLIN_PARTIALS_Q15)
            f.partial_phase_spread_q15 = 26214  # 0.8
            spread_frames.append(f)
        _, out_a = _render(aligned, note=69)
        _, out_s = _render(spread_frames, note=69)

        assert not np.array_equal(out_a, out_s), "phase spread had no effect"

        def _harms(out):
            seg = out[len(out) // 2:].astype(np.float64)
            seg = (seg - seg.mean()) * np.hanning(len(seg))
            sp = np.abs(np.fft.rfft(seg))
            fr = np.fft.rfftfreq(len(seg), 1.0 / FS)
            vals = []
            for h in range(1, 9):
                m = (fr > 440 * h - 100) & (fr < 440 * h + 100)
                vals.append(float(sp[m].max()))
            peak = max(vals) or 1.0
            return [v / peak for v in vals]

        ha, hs = _harms(out_a), _harms(out_s)
        for h, (a, s) in enumerate(zip(ha, hs), start=1):
            assert abs(a - s) < 0.15, (
                f"H{h} magnitude changed too much with phase spread: {a:.3f} vs {s:.3f}"
            )

    def test_vibrato_does_not_smear_spectrum(self):
        """6 Hz / 22-cent vibrato must not produce the block-rate (750 Hz)
        discontinuity comb between harmonics."""
        note = 69  # 440 Hz
        n_blocks = 400  # ~0.53 s
        frames = []
        for i in range(n_blocks):
            t = i * BLOCK_SIZE / FS
            cents = round(22 * math.sin(2 * math.pi * 6.0 * t))
            frames.append(_make_frame(note=note, cents=cents, partials=VIOLIN_PARTIALS_Q15))
        _, out = _render(frames, note=note)

        samples = out[len(out) // 2:].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / FS)

        fundamental = _band_energy(spectrum, freqs, 440 - 60, 440 + 60)
        # 660-780 Hz sits between H1 (440) and H2 (880) but contains the
        # ~750 Hz block-rate comb the phase-reset bug produced.
        inter_harmonic = _band_energy(spectrum, freqs, 660, 780)

        assert inter_harmonic < fundamental / 30, (
            f"inter-harmonic energy ratio {inter_harmonic / max(fundamental, 1e-12):.4f} "
            "— vibrato is injecting block-rate discontinuities again"
        )
