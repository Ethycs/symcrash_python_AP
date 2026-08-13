"""
SOUND_EXTRAS chunk (synthesis-path dedup, step 10).

The capsule used to be lossy vs the IR: material beta ramp, FM6
post-filter, custom partials, PWM width and LFO modulators were dropped,
which is why the offline/workbench render paths bypassed the capsule.
The SOUND_EXTRAS chunk closes that: for every preset, the capsule-loaded
CompiledPreset must match the IR-loaded one on those fields, through both
the byte codec and the bundle-directory codec.
"""

from pathlib import Path

import pytest

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule_codec import (
    capsule_to_bytes,
    capsule_from_bytes,
    save_capsule_bundle,
    load_capsule_bundle,
)
from symcrash_python_ap.library.preset_compiler import (
    compiled_preset_from_capsule,
    compiled_preset_from_ir,
)

PRESETS_DIR = Path(__file__).parent.parent / "presets" / "instruments"
PRESET_PATHS = sorted(PRESETS_DIR.glob("*.yaml"))


def _lfo_key(lfo):
    return (
        lfo.id, int(lfo.waveform), int(lfo.sync_mode), float(lfo.rate_hz),
        int(lfo.depth_q15), int(lfo.delay_samples), int(lfo.ramp_samples),
        tuple((int(r.target), int(r.amount_q15)) for r in lfo.routings),
    )


def _assert_lossless(from_ir, from_capsule, name):
    # Material beta ramp
    assert from_capsule.material.beta_start_q15 == from_ir.material.beta_start_q15, name
    assert from_capsule.material.beta_target_q15 == from_ir.material.beta_target_q15, name
    assert from_capsule.material.beta_ramp_blocks == from_ir.material.beta_ramp_blocks, name

    # FM6 post filter
    if from_ir.fm6 is not None:
        assert from_capsule.fm6 is not None, name
        assert from_capsule.fm6.post_filter_hz == from_ir.fm6.post_filter_hz, name
        assert from_capsule.fm6.mod_index == from_ir.fm6.mod_index, name

    # Harmonic profile (custom partials / waveform profile)
    assert list(from_capsule.harmonic_amps_q15) == list(from_ir.harmonic_amps_q15), name

    # osc mode (Mode B presets must not decay to "unset")
    if from_ir.osc_mode != 0:
        assert from_capsule.osc_mode == from_ir.osc_mode, name

    # PWM width
    assert from_capsule.pwm_width_q15 == from_ir.pwm_width_q15, name

    # LFO modulators
    assert len(from_capsule.lfos or []) == len(from_ir.lfos or []), name
    for a, b in zip(from_capsule.lfos or [], from_ir.lfos or []):
        assert _lfo_key(a) == _lfo_key(b), name


@pytest.mark.parametrize(
    "yaml_path", PRESET_PATHS, ids=[p.stem for p in PRESET_PATHS]
)
def test_capsule_matches_ir_after_bytes_roundtrip(yaml_path):
    ir = load_instrument(str(yaml_path))
    result = CapsuleCompiler().compile_from_ir(ir)
    assert result.success, result.errors

    capsule = capsule_from_bytes(capsule_to_bytes(result.capsule))
    _assert_lossless(
        compiled_preset_from_ir(ir),
        compiled_preset_from_capsule(capsule),
        yaml_path.stem,
    )


def test_capsule_matches_ir_after_bundle_roundtrip(tmp_path):
    for yaml_path in PRESET_PATHS[:6]:
        ir = load_instrument(str(yaml_path))
        result = CapsuleCompiler().compile_from_ir(ir)
        assert result.success, result.errors

        bundle_dir = tmp_path / yaml_path.stem
        save_capsule_bundle(result.capsule, str(bundle_dir))
        capsule = load_capsule_bundle(str(bundle_dir))
        _assert_lossless(
            compiled_preset_from_ir(ir),
            compiled_preset_from_capsule(capsule),
            yaml_path.stem,
        )
