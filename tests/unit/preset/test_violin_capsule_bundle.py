from pathlib import Path

from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.instrument_ir import ms_to_blocks, pct_to_q15
from symcrash_python_ap.library import PresetLibraryV0, ResolutionSelector


def test_violin_capsule_bundle_in_library() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    library_root = repo_root / "preset_library"
    library = PresetLibraryV0(str(library_root))

    record = library.resolve_capsule("violin-ir-001", selector=ResolutionSelector.LATEST)
    assert record is not None

    defaults = record.capsule.defaults.control_defaults

    assert defaults[ControlID.TIMBRE] == pct_to_q15(0.42)
    assert defaults[ControlID.TEXTURE] == pct_to_q15(0.35)
    assert defaults[ControlID.MORPH] == pct_to_q15(0.15)
    assert defaults[ControlID.CHAOS] == pct_to_q15(0.10)
    assert defaults[ControlID.ATK_BLOCKS] == ms_to_blocks(30.0)
    assert defaults[ControlID.DCY_BLOCKS] == ms_to_blocks(220.0)
    assert defaults[ControlID.SUS_Q15] == pct_to_q15(0.75)
    assert defaults[ControlID.REL_BLOCKS] == ms_to_blocks(550.0)

    capsule_path = (
        library_root
        / "capsules"
        / record.capsule.identity.capsule_id
        / str(record.revision_id)
    )
    assert (capsule_path / "instrument_ir.json").exists()
