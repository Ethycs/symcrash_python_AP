"""
Capsule codec round-trip pin (synthesis-path dedup, step 4).

The compile pipeline used to serialize every capsule to .wsecap bytes and
immediately re-parse them before import — a wasteful round-trip that doubled
as an accidental serialization self-test. import_capsule() now accepts the
CapsuleV0 object directly; this test replaces that safety net explicitly:
for every instrument preset YAML, bytes -> capsule -> bytes must be stable.
"""

from pathlib import Path

import pytest

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.library.capsule import CapsuleV0
from symcrash_python_ap.library.capsule_codec import (
    capsule_to_bytes,
    capsule_from_bytes,
)

PRESETS_DIR = Path(__file__).parent.parent / "presets" / "instruments"
PRESET_PATHS = sorted(PRESETS_DIR.glob("*.yaml"))


@pytest.mark.parametrize(
    "yaml_path", PRESET_PATHS, ids=[p.stem for p in PRESET_PATHS]
)
def test_capsule_bytes_roundtrip_stable(yaml_path):
    ir = load_instrument(str(yaml_path))
    result = CapsuleCompiler().compile_from_ir(ir)
    assert result.success, f"compile failed: {result.errors}"

    first = capsule_to_bytes(result.capsule)
    reparsed = capsule_from_bytes(first)
    second = capsule_to_bytes(reparsed)

    assert first == second, "capsule bytes not stable across decode/encode"


def test_import_capsule_accepts_object(tmp_path):
    from symcrash_python_ap.library.preset_library import PresetLibraryV0

    ir = load_instrument(str(PRESET_PATHS[0]))
    result = CapsuleCompiler().compile_from_ir(ir)
    assert result.success

    library = PresetLibraryV0(str(tmp_path / "lib"))
    import_result = library.import_capsule(result.capsule)
    assert import_result.success, import_result.errors

    # Loading back yields the same serialized form as the original object
    record = library.resolve_capsule(import_result.capsule_id)
    assert record is not None
    assert isinstance(record.capsule, CapsuleV0)
    assert capsule_to_bytes(record.capsule) == capsule_to_bytes(result.capsule)
