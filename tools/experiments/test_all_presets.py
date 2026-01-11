"""
Test All Presets Loading
========================

Verifies that all preset YAML files in the library load correctly.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import load_instrument


def test_all_presets():
    # Collect presets from both instruments and macros directories
    presets_root = Path(__file__).parent.parent / "presets"
    presets = sorted(
        list((presets_root / "instruments").glob("*.yaml")) +
        list((presets_root / "macros").glob("*.yaml"))
    )

    print(f"Found {len(presets)} presets:")
    print("=" * 70)

    failed = []
    for p in presets:
        try:
            ir = load_instrument(str(p))
            name = ir.meta.name if ir.meta.name else "(unnamed)"
            pid = ir.meta.instrument_id if ir.meta.instrument_id else "(no id)"
            print(f"  OK: {p.name:35} -> {name}")
        except Exception as e:
            print(f"FAIL: {p.name:35} -> {e}")
            failed.append((p.name, str(e)))

    print("=" * 70)
    if failed:
        print(f"FAILED: {len(failed)} presets")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return False
    else:
        print(f"All {len(presets)} presets loaded successfully!")
        return True


if __name__ == "__main__":
    success = test_all_presets()
    sys.exit(0 if success else 1)
