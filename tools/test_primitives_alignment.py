#!/usr/bin/env python3
"""Test that voice_engine.py uses primitives.py correctly."""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def main():
    print("Testing primitives alignment...")

    # Test imports
    from symcrash_python_ap.primitives import Regime, Intent, Family
    from symcrash_python_ap.voice import VoiceEngine, VoiceState
    print("  Imports successful")

    # Verify Regime is the same type
    from symcrash_python_ap.voice.voice_engine import Regime as VERegime
    assert Regime is VERegime, "Regime should be imported from primitives"
    print("  Regime is same type in both modules")

    # Verify canonical names exist
    print(f"  Regime.R0_IDLE = {Regime.R0_IDLE}")
    print(f"  Regime.R1_ATTACK = {Regime.R1_ATTACK}")
    print(f"  Regime.R2_STABLE = {Regime.R2_STABLE}")
    print(f"  Regime.R3_AIR = {Regime.R3_AIR}")
    print(f"  Regime.R4_GLITCH = {Regime.R4_GLITCH}")
    print(f"  Regime.R5_CATASTROPHE = {Regime.R5_CATASTROPHE}")
    print(f"  Regime.R6_RECOVERY = {Regime.R6_RECOVERY}")

    # Verify REGIME_SPECS uses canonical names
    from symcrash_python_ap.voice.voice_engine import REGIME_SPECS
    assert Regime.R1_ATTACK in REGIME_SPECS, "R1_ATTACK should be in REGIME_SPECS"
    assert Regime.R2_STABLE in REGIME_SPECS, "R2_STABLE should be in REGIME_SPECS"
    print("  REGIME_SPECS uses canonical names")

    # Verify REGIME_TRANSITIONS uses canonical names
    from symcrash_python_ap.voice.voice_engine import REGIME_TRANSITIONS
    assert Regime.R1_ATTACK in REGIME_TRANSITIONS, "R1_ATTACK should be in REGIME_TRANSITIONS"
    print("  REGIME_TRANSITIONS uses canonical names")

    # Test VoiceEngine creation
    engine = VoiceEngine()
    print("  VoiceEngine created successfully")

    print("\n=== All primitives alignment tests passed! ===")

if __name__ == "__main__":
    main()
