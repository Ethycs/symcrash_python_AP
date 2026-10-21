from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.instrument_ir import (
    load_instrument,
    ms_to_blocks,
    pct_to_q15,
)
from symcrash_python_ap.library import PresetLibraryV0, ResolutionSelector
from symcrash_python_ap.library.compiler import CapsuleCompiler
from symcrash_python_ap.paths import PRESET_INSTRUMENTS_DIR

# Capsule IDs are now content-hashed (slug + digest); this is the violin
# capsule compiled from presets/instruments/violin_ir_v1.yaml.
VIOLIN_CAPSULE_ID = "violin-ir-256819fd62e4"


def test_violin_capsule_bundle_in_library(tmp_path) -> None:
    """Compile the source into an isolated generated library.

    A clean checkout intentionally contains no compiled preset payloads.
    """
    ir = load_instrument(str(PRESET_INSTRUMENTS_DIR / "violin_ir_v1.yaml"))
    compiled = CapsuleCompiler().compile_from_ir(ir)
    assert compiled.success
    assert compiled.capsule_id == VIOLIN_CAPSULE_ID

    library_root = tmp_path / "preset-build"
    library = PresetLibraryV0(str(library_root))
    imported = library.import_capsule(compiled.capsule)
    assert imported.success

    record = library.resolve_capsule(VIOLIN_CAPSULE_ID, selector=ResolutionSelector.LATEST)
    assert record is not None

    defaults = record.capsule.defaults.control_defaults

    # ADSR defaults compiled from violin_ir_v1.yaml (30ms / 250ms / 0.8 / 550ms).
    # TIMBRE/TEXTURE/MORPH/CHAOS defaults were removed when the control plane
    # collapsed to the continuous 7-axis emphasis model (Phase A).
    assert defaults[ControlID.ATK_BLOCKS] == ms_to_blocks(30.0)
    assert defaults[ControlID.DCY_BLOCKS] == ms_to_blocks(250.0)
    # +/-1 LSB: the preset compiler truncates int(0.8*32767) while
    # pct_to_q15 rounds, so allow one count of slack.
    assert abs(defaults[ControlID.SUS_Q15] - pct_to_q15(0.8)) <= 1
    assert defaults[ControlID.REL_BLOCKS] == ms_to_blocks(550.0)

    # Revision payload now lives under capsules/instruments/<id>/<revision>
    # and stores manifest.json + control_defaults.bin (no instrument_ir.json).
    capsule_path = library_root / record.storage_path
    assert (capsule_path / "manifest.json").exists()
    assert (capsule_path / "control_defaults.bin").exists()
