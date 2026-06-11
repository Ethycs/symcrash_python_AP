"""
Live Envelope Bindings Tests (packet E6)
========================================

Closes the IR -> engine gap for matrix envelope bindings:

1. Capsule pipeline carries `envelopes:` bindings (and amp dynamics) through
   serialization -> load, so a COMPILED CAPSULE retains them
   (CapsuleV0.envelopes / .amp_dynamics, ChunkType.ENVELOPES, bundle manifest).
2. VoiceEngine Simple mode (allocate_voice_from_preset -> note_on_simple ->
   step_voice_simple) computes the same per-block envelope target values as
   the offline render harness (tools/render/render_preset.py):
   - velocity_sens: velocity -> FM modulation index (brightening)
   - key_scale_rate: higher notes decay faster (DX7 EG rate scaling)
3. Audio-level acceptance: the dx_marimba preset played through the live
   engine shows velocity -> spectral-centroid brightening matching the
   harness within tolerance, for both the YAML-compiled preset and the
   compiled-capsule-loaded preset.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
MARIMBA_YAML = REPO_ROOT / "presets" / "instruments" / "dx_marimba_v0.yaml"

# tools/render is not a package; make render_preset importable
_tools_render = str(REPO_ROOT / "tools" / "render")
if _tools_render not in sys.path:
    sys.path.insert(0, _tools_render)

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.capsule_codec import (
    capsule_from_bytes,
    capsule_to_bytes,
    load_capsule_bundle,
    save_capsule_bundle,
)
from symcrash_python_ap.library.preset_compiler import (
    compiled_preset_from_capsule,
    compiled_preset_from_ir,
)
from symcrash_python_ap.vm.executor import vm_step_block
from symcrash_python_ap.vm.harmonic_bank import clear_basis_atlas
from symcrash_python_ap.vm.state import VMState
from symcrash_python_ap.voice.voice_engine import VoiceEngine

FS = 48000
BLOCK = 64

VEL_LO = int(0.35 * 32767)
VEL_HI = 32767


# =============================================================================
# Helpers
# =============================================================================

@pytest.fixture(scope="module")
def marimba_ir():
    return load_instrument(str(MARIMBA_YAML))


@pytest.fixture(scope="module")
def marimba_preset(marimba_ir):
    return compiled_preset_from_ir(marimba_ir)


@pytest.fixture(scope="module")
def marimba_capsule_preset(marimba_ir):
    """CompiledPreset reconstructed from a serialized+reloaded capsule."""
    capsule = marimba_ir.to_capsule()
    with tempfile.TemporaryDirectory() as td:
        save_capsule_bundle(capsule, td)
        loaded = load_capsule_bundle(td)
    return compiled_preset_from_capsule(loaded)


def live_frames(preset, note, velocity_q15, blocks=400):
    """Step the live engine, returning the VoiceFrame stream."""
    engine = VoiceEngine(max_voices=2, block_size=BLOCK)
    vid = engine.allocate_voice_from_preset(preset)
    engine.note_on_simple(vid, note=note, velocity_q15=velocity_q15)
    frames = []
    for _ in range(blocks):
        frame = engine.step_voice_simple(vid)
        if frame is None:
            break
        frames.append(frame)
    return frames


def live_audio(preset, note, velocity_q15, blocks=400):
    """Render live engine frames through the VM (same renderer as harness)."""
    clear_basis_atlas()
    engine = VoiceEngine(max_voices=2, block_size=BLOCK)
    vid = engine.allocate_voice_from_preset(preset)
    engine.note_on_simple(vid, note=note, velocity_q15=velocity_q15)
    vm = VMState.create(voice_id=0)
    vm.reset_for_note_on(note=note)
    samples = []
    for _ in range(blocks):
        frame = engine.step_voice_simple(vid)
        if frame is None:
            break
        vm, out = vm_step_block(vm, frame)
        samples.extend(out.tolist())  # copy: vm reuses the output buffer
    return np.array(samples, dtype=np.int16)


def harness_audio(ir, note, velocity_q15, duration=0.6):
    """Render via the offline harness (tools/render/render_preset.py)."""
    from render_preset import render_preset_note

    clear_basis_atlas()
    with contextlib.redirect_stdout(io.StringIO()):
        return render_preset_note(
            ir, note=note, duration_sec=duration, velocity_q15=velocity_q15)


def harness_env_series(ir, note, velocity_q15, blocks=400):
    """Per-block mod-envelope values the harness would apply (FM_OP2)."""
    from render_preset import _EnvelopeState

    binding = ir.envelopes[0]
    st = _EnvelopeState(binding, note)
    out = []
    for b in range(blocks):
        v = st.value_q15(b, blocks)  # sustain beyond range -> no release
        if st.velocity_sens_q15:
            scale = (32767 - st.velocity_sens_q15
                     + ((st.velocity_sens_q15 * velocity_q15) >> 15))
            v = (v * scale) >> 15
        out.append(v)
    return np.array(out)


def env_series_from_frames(frames, target_id=1):
    """Extract env_targets_q15[target_id] (FM_OP2 = 1) per block."""
    return np.array([(f.env_targets_q15 or {}).get(target_id, 0)
                     for f in frames])


def t50_blocks(series):
    """Blocks from peak until the series falls to half its peak."""
    peak = series.max()
    pi = int(np.argmax(series))
    for i in range(pi, len(series)):
        if series[i] <= peak / 2:
            return i - pi
    return len(series)


def spectral_centroid(audio, t0=0.0, t1=0.4):
    seg = audio[int(t0 * FS):int(t1 * FS)].astype(np.float64)
    assert len(seg) >= 256 and np.max(np.abs(seg)) > 1, "segment is silent"
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / FS)
    return float(np.sum(f * X) / np.sum(X))


# =============================================================================
# 1. Capsule pipeline carries envelope bindings
# =============================================================================

class TestCapsuleCarriesEnvelopes:
    def test_compile_from_ir_extracts_envelopes(self, marimba_ir):
        capsule = marimba_ir.to_capsule()
        assert capsule.envelopes, "compiled capsule lost envelope bindings"
        b = capsule.envelopes[0]
        assert b['id'] == 'mod_env'
        assert b['targets'] == ['FM_OP2', 'FM_OP4']
        assert b['velocity_sens'] == pytest.approx(0.5)
        assert b['key_scale_rate'] == pytest.approx(0.6)
        assert b['adsr']['d_ms'] == pytest.approx(150.0)
        # Amp dynamics from the Amp node
        assert capsule.amp_dynamics == {
            'velocity_sens': 0.5, 'key_scale_rate': 0.0}

    def test_binary_roundtrip_retains_envelopes(self, marimba_ir):
        capsule = marimba_ir.to_capsule()
        restored = capsule_from_bytes(capsule_to_bytes(capsule))
        assert restored.envelopes == capsule.envelopes
        assert restored.amp_dynamics == capsule.amp_dynamics

    def test_bundle_roundtrip_retains_envelopes(self, marimba_ir):
        capsule = marimba_ir.to_capsule()
        with tempfile.TemporaryDirectory() as td:
            save_capsule_bundle(capsule, td)
            restored = load_capsule_bundle(td)
        assert restored.envelopes == capsule.envelopes
        assert restored.amp_dynamics == capsule.amp_dynamics

    def test_compiled_preset_from_capsule(self, marimba_capsule_preset):
        p = marimba_capsule_preset
        assert p.envelope_bindings, "capsule preset lost envelope bindings"
        b = p.envelope_bindings[0]
        assert b.targets == ['FM_OP2', 'FM_OP4']
        assert b.velocity_sens == pytest.approx(0.5)
        assert b.key_scale_rate == pytest.approx(0.6)
        assert p.amp_velocity_sens == pytest.approx(0.5)
        # FM6 config reconstructed from voice params
        assert p.osc_mode == 3
        assert p.fm6 is not None
        assert p.fm6.algorithm_id == 5
        # Amp envelope reconstructed from control defaults (block-quantized)
        assert p.amp_env is not None
        assert p.amp_env.decay_ms == pytest.approx(500.0, abs=2.0)
        assert p.amp_env.sustain == pytest.approx(0.0, abs=1e-3)

    def test_envelope_free_capsule_has_no_chunk(self):
        """Presets without envelopes serialize with envelopes=None."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("E6 Plain Synth")
        capsule = ir.to_capsule()
        assert not capsule.envelopes
        restored = capsule_from_bytes(capsule_to_bytes(capsule))
        assert restored.envelopes is None


# =============================================================================
# 2. Live engine envelope values match the render harness
# =============================================================================

class TestLiveEnvelopesMatchHarness:
    @pytest.mark.parametrize("preset_fixture",
                             ["marimba_preset", "marimba_capsule_preset"])
    def test_velocity_scales_mod_env_peak(self, request, marimba_ir,
                                          preset_fixture):
        preset = request.getfixturevalue(preset_fixture)
        for vel in (VEL_LO, VEL_HI):
            live = env_series_from_frames(live_frames(preset, 60, vel))
            ref = harness_env_series(marimba_ir, 60, vel)
            # Peak modulation index identical (velocity_sens applied)
            assert abs(int(live.max()) - int(ref.max())) <= 64
        # And velocity actually scales the index:
        # peak ratio = 1 - sens + sens*vel = 0.675 for sens=0.5, vel=0.35
        lo_peak = env_series_from_frames(
            live_frames(preset, 60, VEL_LO)).max()
        hi_peak = env_series_from_frames(
            live_frames(preset, 60, VEL_HI)).max()
        assert lo_peak / hi_peak == pytest.approx(0.675, abs=0.02)

    @pytest.mark.parametrize("note", [48, 60, 72])
    def test_mod_env_decay_matches_harness(self, marimba_ir, marimba_preset,
                                           note):
        live = env_series_from_frames(live_frames(marimba_preset, note,
                                                  VEL_HI))
        ref = harness_env_series(marimba_ir, note, VEL_HI)
        # Decay-to-half time within 2 blocks (~2.7 ms) of the harness
        assert abs(t50_blocks(live) - t50_blocks(ref)) <= 2

    def test_key_scaled_decay_higher_notes_shorter(self, marimba_preset):
        """key_scale_rate=0.6: t50 scales by 2^(-0.6 * dnote / 24)."""
        t48 = t50_blocks(env_series_from_frames(
            live_frames(marimba_preset, 48, VEL_HI)))
        t72 = t50_blocks(env_series_from_frames(
            live_frames(marimba_preset, 72, VEL_HI)))
        assert t72 < t48, "higher note must decay faster"
        # Expected ratio 2^(0.6 * 24/24) = ~1.516; allow block quantization
        assert t48 / t72 == pytest.approx(1.516, rel=0.10)

    def test_key_scaling_capsule_path(self, marimba_capsule_preset):
        t48 = t50_blocks(env_series_from_frames(
            live_frames(marimba_capsule_preset, 48, VEL_HI)))
        t72 = t50_blocks(env_series_from_frames(
            live_frames(marimba_capsule_preset, 72, VEL_HI)))
        assert t72 < t48
        assert t48 / t72 == pytest.approx(1.516, rel=0.10)


# =============================================================================
# 3. Audio-level acceptance: velocity -> brightening, live vs harness
# =============================================================================

class TestAudioAcceptance:
    @pytest.fixture(scope="class")
    def harness_centroids(self, marimba_ir):
        lo = harness_audio(marimba_ir, 60, VEL_LO)
        hi = harness_audio(marimba_ir, 60, VEL_HI)
        return spectral_centroid(lo), spectral_centroid(hi)

    def test_live_yaml_brightening_matches_harness(self, marimba_preset,
                                                   harness_centroids):
        """YAML-compiled preset: live path matches harness within 3%."""
        c_lo = spectral_centroid(live_audio(marimba_preset, 60, VEL_LO))
        c_hi = spectral_centroid(live_audio(marimba_preset, 60, VEL_HI))
        h_lo, h_hi = harness_centroids

        # (a) higher velocity -> measurably brighter spectrum
        assert c_hi / c_lo > 1.08, (
            f"velocity brightening too weak: {c_hi:.0f}/{c_lo:.0f}")

        # Absolute centroids match the harness (same FM6 + post filter +
        # mod envelopes); measured agreement is ~0.2%
        assert c_lo == pytest.approx(h_lo, rel=0.03)
        assert c_hi == pytest.approx(h_hi, rel=0.03)
        # Brightening ratio matches within 5%
        assert (c_hi / c_lo) == pytest.approx(h_hi / h_lo, rel=0.05)

    def test_live_capsule_brightening_matches_harness(
            self, marimba_capsule_preset, harness_centroids):
        """Compiled capsule played live: brightening matches within 5%.

        (Capsules don't carry the FM6 post filter, so absolute centroids
        sit slightly above the harness; the velocity->brightening ratio
        is the acceptance metric.)
        """
        c_lo = spectral_centroid(live_audio(marimba_capsule_preset, 60,
                                            VEL_LO))
        c_hi = spectral_centroid(live_audio(marimba_capsule_preset, 60,
                                            VEL_HI))
        h_lo, h_hi = harness_centroids

        assert c_hi / c_lo > 1.08
        assert c_lo == pytest.approx(h_lo, rel=0.05)
        assert c_hi == pytest.approx(h_hi, rel=0.05)
        assert (c_hi / c_lo) == pytest.approx(h_hi / h_lo, rel=0.05)
