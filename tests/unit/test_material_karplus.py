"""Karplus-Strong material program coverage.

Salvaged from the stale tools/test_ks.py dev script (2026-06-16): it was
the only Karplus-Strong coverage in the repo and the engine path is live,
so it became a real assertion-based test rather than a print harness.

Covers: all 11 material_ids render a full block via MaterialEngine, and
the KS-specific pitch/excite path produces a sensibly-tuned delay line.
"""

import pytest

from symcrash_python_ap.material.material_engine import (
    MaterialEngine,
    KarplusStrongState,
)
from symcrash_python_ap.frames import ParamFrame

BLOCK = 64
FS = 48000


@pytest.fixture
def engine():
    return MaterialEngine()


@pytest.mark.parametrize("material_id", range(11))
def test_material_id_renders_full_block(engine, material_id):
    """Every material program returns a full stereo block without error."""
    left = [1000] * BLOCK
    right = [1000] * BLOCK
    out_l, out_r = engine.step_block(left, right, ParamFrame(),
                                     material_id=material_id)
    assert len(out_l) == BLOCK
    assert len(out_r) == BLOCK


def test_karplus_pitch_sets_delay_line(engine):
    """set_karplus_pitch tunes the delay length to ~FS/freq samples."""
    engine.set_karplus_pitch(440.0)
    expected = round(FS / 440.0)  # ~109 at 48 kHz
    # allow +/-1 for fractional-delay rounding conventions
    assert abs(engine.ks_L.delay_samples - expected) <= 1


def test_karplus_excite_fills_string(engine):
    """excite_karplus fills the delay line with the pluck noise burst.

    Asserts the direct effect of exciting (buffer energy across the tuned
    delay length). Whether that rings through step_block audibly depends
    on the ParamFrame exciter-mix and is left to the integration tests.
    """
    engine.set_karplus_pitch(440.0)
    engine.excite_karplus(16000)
    n = engine.ks_L.delay_samples
    assert any(engine.ks_L.buffer[i] != 0 for i in range(n))
    assert any(engine.ks_R.buffer[i] != 0 for i in range(n))
