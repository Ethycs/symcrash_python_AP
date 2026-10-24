"""G1: exact render request is quiet, deterministic and matches the legacy renderer."""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import yaml

from symcrash_inversion.render_service import RenderRequest, render, velocity_q15
from symcrash_python_ap.instrument_ir import load_instrument_from_dict

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "presets" / "instruments" / f"{name}.yaml").read_text(encoding="utf-8"))


def legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_render_preset", ROOT / "tools" / "render" / "render_preset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def test_quiet_deterministic_and_exact_length():
    d = load("dx7_marimba_v0")
    req = RenderRequest(d, note=60, velocity=0.7, gate_s=0.3, tail_s=0.2, seed=0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        r1 = render(req)
        r2 = render(req)
    assert buf.getvalue() == ""
    assert r1.audio.dtype == np.int16 and r1.audio.shape[-1] == round(0.5 * 48000)
    assert np.array_equal(r1.audio, r2.audio)
    assert r1.sample_rate_hz == 48000
    for key in ("git_revision", "schema_versions", "sample_rate_hz", "block_size", "seed",
                "renderer_id", "compiler_version", "vm_version"):
        assert key in r1.render_fingerprint
    assert r1.render_fingerprint["seed"] == 0 and r1.render_fingerprint["block_size"] == 64


def test_matches_legacy_renderer_bit_for_bit():
    d = load("dx7_marimba_v0")
    ir = load_instrument_from_dict(d)
    legacy = legacy_module()
    with redirect_stdout(io.StringIO()):
        ref = np.asarray(legacy._render_voice(ir, note=57, duration_sec=0.3, velocity_q15=velocity_q15(0.35),
                                              noise_seed_xor=0, verbose=False))
    out = render(RenderRequest(d, note=57, velocity=0.35, gate_s=0.3, tail_s=2.0, seed=0)).audio
    n = min(len(ref), out.shape[-1])
    assert np.array_equal(out[:n], ref[:n])
    assert velocity_q15(0.35) == round(0.35 * 32767) and velocity_q15(1.0) == 32767


def test_truncation_and_padding_are_reported():
    d = load("dx7_marimba_v0")
    short = render(RenderRequest(d, note=60, velocity=1.0, gate_s=0.2, tail_s=0.0, seed=0))
    long_ = render(RenderRequest(d, note=60, velocity=1.0, gate_s=0.2, tail_s=5.0, seed=0))
    assert short.diagnostics["truncated"] is True and short.audio.shape[-1] == round(0.2 * 48000)
    assert long_.diagnostics["padded"] is True and long_.audio.shape[-1] == round(5.2 * 48000)
    assert np.array_equal(long_.audio[: short.audio.shape[-1]], short.audio)
