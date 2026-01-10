"""
Test Macro -> CMMM Integration
==============================

Proves that SystemMacros (BRIGHTNESS, DENSITY, RISK, etc.) influence
the CMMM pipeline: Gesture -> Intent -> Family Weights -> Operators.

This demonstrates the complete data flow:
1. MacroRuntime holds slewed macro values
2. apply_macros_to_gesture modifies GestureBasis
3. gesture_to_intent_with_macros produces macro-influenced IntentVector
4. compute_family_weights produces different operator weights

"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.macro_runtime import MacroRuntime, SystemMacro
from symcrash_python_ap.library.capsule import ControlProgram
from symcrash_python_ap.voice.voice_engine import (
    GestureBasis, IntentVector,
    default_gesture_to_intent,
    apply_macros_to_gesture,
    apply_macros_to_intent,
    gesture_to_intent_with_macros,
    compute_family_weights,
    Regime,
    Q15_ONE, Q15_HALF,
    I_TONE, I_BOW, I_AIR, I_GLITCH, I_WARP, I_SNAP, I_GLUE, I_CALM,
    F_COLOR, F_TRANSPORT, F_LOCAL, F_TOPOLOGY, F_GLUE, F_EVENTS, F_CATASTROPHE,
)


def print_intent(intent: IntentVector, label: str = "Intent"):
    """Pretty print intent vector."""
    print(f"  {label}:")
    print(f"    TONE={intent.tone:5d}  BOW={intent.bow:5d}  AIR={intent.air:5d}  GLITCH={intent.glitch:5d}")
    print(f"    WARP={intent.warp:5d}  SNAP={intent.snap:5d}  GLUE={intent.glue:5d}  CALM={intent.calm:5d}")


def print_family_weights(weights: list, label: str = "Family Weights"):
    """Pretty print family weights."""
    names = ["RESERVED", "COLOR", "TRANSPORT", "LOCAL", "TOPOLOGY", "GLUE", "EVENTS", "CATASTROPHE"]
    print(f"  {label}:")
    for i, (name, w) in enumerate(zip(names, weights)):
        bar = "#" * (w // 2000)
        print(f"    F_{name:12s} = {w:5d} {bar}")


def test_brightness_effect():
    """Test that BRIGHTNESS macro affects tilt -> COLOR family."""
    print("\n" + "=" * 60)
    print("TEST: BRIGHTNESS -> COLOR Family Weight")
    print("=" * 60)

    # Create base gesture (neutral)
    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF,
        speed_q15=Q15_HALF // 2,
    )

    # Create macro runtime with LOW brightness
    runtime_low = MacroRuntime()
    runtime_low.macro_curr_q15[SystemMacro.BRIGHTNESS] = 0  # Dark
    runtime_low.macro_target_q15[SystemMacro.BRIGHTNESS] = 0

    # Create macro runtime with HIGH brightness
    runtime_high = MacroRuntime()
    runtime_high.macro_curr_q15[SystemMacro.BRIGHTNESS] = Q15_ONE  # Bright
    runtime_high.macro_target_q15[SystemMacro.BRIGHTNESS] = Q15_ONE

    # Compute intent without macros
    intent_neutral = default_gesture_to_intent(gesture)

    # Compute intent with LOW brightness
    intent_dark = gesture_to_intent_with_macros(gesture, runtime_low)

    # Compute intent with HIGH brightness
    intent_bright = gesture_to_intent_with_macros(gesture, runtime_high)

    print("\n1. Base gesture (neutral):")
    print_intent(intent_neutral, "No macros")

    print("\n2. With BRIGHTNESS=0 (dark):")
    print_intent(intent_dark, "Dark")

    print("\n3. With BRIGHTNESS=32767 (bright):")
    print_intent(intent_bright, "Bright")

    # Compute family weights
    weights_neutral = compute_family_weights(intent_neutral, Regime.R2_STABLE_TONE)
    weights_dark = compute_family_weights(intent_dark, Regime.R2_STABLE_TONE)
    weights_bright = compute_family_weights(intent_bright, Regime.R2_STABLE_TONE)

    print("\n4. Family weights comparison:")
    print("\n  NEUTRAL (no macros):")
    print_family_weights(weights_neutral)

    print("\n  DARK (BRIGHTNESS=0):")
    print_family_weights(weights_dark)

    print("\n  BRIGHT (BRIGHTNESS=32767):")
    print_family_weights(weights_bright)

    # Verify COLOR family changes
    color_neutral = weights_neutral[F_COLOR]
    color_dark = weights_dark[F_COLOR]
    color_bright = weights_bright[F_COLOR]

    print(f"\n  COLOR family: dark={color_dark} < neutral={color_neutral} < bright={color_bright}")

    if color_dark < color_neutral < color_bright:
        print("  OK: BRIGHTNESS correctly influences COLOR family weight")
    else:
        print("  FAIL: BRIGHTNESS effect not as expected")


def test_risk_effect():
    """Test that RISK macro affects chaos -> GLITCH intent -> TOPOLOGY family."""
    print("\n" + "=" * 60)
    print("TEST: RISK -> GLITCH Intent -> TOPOLOGY Family")
    print("=" * 60)

    # Create calm gesture
    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF,
        chaos_q15=0,  # No chaos from controller
    )

    # Create macro runtime with NO risk
    runtime_safe = MacroRuntime()
    runtime_safe.macro_curr_q15[SystemMacro.RISK] = 0
    runtime_safe.macro_target_q15[SystemMacro.RISK] = 0

    # Create macro runtime with HIGH risk
    runtime_risky = MacroRuntime()
    runtime_risky.macro_curr_q15[SystemMacro.RISK] = Q15_ONE
    runtime_risky.macro_target_q15[SystemMacro.RISK] = Q15_ONE

    # Compute intents
    intent_safe = gesture_to_intent_with_macros(gesture, runtime_safe)
    intent_risky = gesture_to_intent_with_macros(gesture, runtime_risky)

    print("\n1. With RISK=0 (safe):")
    print_intent(intent_safe, "Safe")

    print("\n2. With RISK=32767 (risky):")
    print_intent(intent_risky, "Risky")

    # Compute family weights
    weights_safe = compute_family_weights(intent_safe, Regime.R2_STABLE_TONE)
    weights_risky = compute_family_weights(intent_risky, Regime.R2_STABLE_TONE)

    print("\n3. Family weights:")
    print("\n  SAFE (RISK=0):")
    print_family_weights(weights_safe)

    print("\n  RISKY (RISK=32767):")
    print_family_weights(weights_risky)

    # Verify TOPOLOGY family changes
    topo_safe = weights_safe[F_TOPOLOGY]
    topo_risky = weights_risky[F_TOPOLOGY]
    glitch_safe = intent_safe.glitch
    glitch_risky = intent_risky.glitch

    print(f"\n  GLITCH intent: safe={glitch_safe} -> risky={glitch_risky}")
    print(f"  TOPOLOGY family: safe={topo_safe} -> risky={topo_risky}")

    if glitch_risky > glitch_safe and topo_risky > topo_safe:
        print("  OK: RISK correctly influences GLITCH intent and TOPOLOGY family")
    else:
        print("  FAIL: RISK effect not as expected")


def test_density_effect():
    """Test that DENSITY macro affects pressure -> BOW intent."""
    print("\n" + "=" * 60)
    print("TEST: DENSITY -> Pressure -> BOW Intent")
    print("=" * 60)

    # Create gesture with moderate pressure
    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF // 2,  # 25% pressure
    )

    # Low density
    runtime_thin = MacroRuntime()
    runtime_thin.macro_curr_q15[SystemMacro.DENSITY] = 0
    runtime_thin.macro_target_q15[SystemMacro.DENSITY] = 0

    # High density
    runtime_thick = MacroRuntime()
    runtime_thick.macro_curr_q15[SystemMacro.DENSITY] = Q15_ONE
    runtime_thick.macro_target_q15[SystemMacro.DENSITY] = Q15_ONE

    # Get modified gestures
    gesture_thin = apply_macros_to_gesture(gesture, runtime_thin)
    gesture_thick = apply_macros_to_gesture(gesture, runtime_thick)

    print(f"\n1. Original pressure: {gesture.pressure_q15}")
    print(f"2. With DENSITY=0: {gesture_thin.pressure_q15}")
    print(f"3. With DENSITY=32767: {gesture_thick.pressure_q15}")

    # Compute intents
    intent_thin = gesture_to_intent_with_macros(gesture, runtime_thin)
    intent_thick = gesture_to_intent_with_macros(gesture, runtime_thick)

    print(f"\n4. BOW intent: thin={intent_thin.bow} -> thick={intent_thick.bow}")

    if gesture_thick.pressure_q15 > gesture.pressure_q15:
        print("  OK: DENSITY correctly boosts pressure")
    else:
        print("  FAIL: DENSITY effect not as expected")


def test_control_program_defaults():
    """Test applying ControlProgram defaults to MacroRuntime."""
    print("\n" + "=" * 60)
    print("TEST: ControlProgram -> MacroRuntime Defaults")
    print("=" * 60)

    # Create a ControlProgram with specific values
    control = ControlProgram(
        brightness_q15=24576,      # 75% brightness
        density_q15=20480,         # 62.5% density
        risk_q15=8192,             # 25% risk
        motion_q15=16384,          # 50% motion
    )

    # Create runtime and apply defaults
    runtime = MacroRuntime()
    runtime.apply_control_defaults(control)

    print("\n1. ControlProgram values:")
    print(f"   BRIGHTNESS: {control.brightness_q15}")
    print(f"   DENSITY: {control.density_q15}")
    print(f"   RISK: {control.risk_q15}")
    print(f"   MOTION: {control.motion_q15}")

    print("\n2. MacroRuntime after apply_control_defaults:")
    print(f"   BRIGHTNESS: {runtime.get_system_macro(SystemMacro.BRIGHTNESS)}")
    print(f"   DENSITY: {runtime.get_system_macro(SystemMacro.DENSITY)}")
    print(f"   RISK: {runtime.get_system_macro(SystemMacro.RISK)}")
    print(f"   MOTION: {runtime.get_system_macro(SystemMacro.MOTION)}")

    # Verify values match
    matches = (
        runtime.get_system_macro(SystemMacro.BRIGHTNESS) == control.brightness_q15 and
        runtime.get_system_macro(SystemMacro.DENSITY) == control.density_q15 and
        runtime.get_system_macro(SystemMacro.RISK) == control.risk_q15 and
        runtime.get_system_macro(SystemMacro.MOTION) == control.motion_q15
    )

    if matches:
        print("\n  OK: ControlProgram correctly applied to MacroRuntime")
    else:
        print("\n  FAIL: Values don't match")


def test_age_integrator():
    """Test that AGE integrates based on energy_state over time."""
    print("\n" + "=" * 60)
    print("TEST: AGE Integrator")
    print("=" * 60)

    runtime = MacroRuntime()

    # Initial state: AGE should be 0
    print(f"\n1. Initial AGE: {runtime.get_system_macro(SystemMacro.AGE)}")

    # Simulate note-on with high energy_state
    runtime.set_energy_state(Q15_ONE)  # Full energy
    print(f"2. Set energy_state to {Q15_ONE} (full)")

    # Tick several times to accumulate AGE
    ages = []
    for i in range(100):
        runtime.tick()
        if i % 20 == 0:
            ages.append(runtime.get_system_macro(SystemMacro.AGE))

    print(f"3. AGE after 100 ticks (high energy): {ages}")

    age_during_note = runtime.get_system_macro(SystemMacro.AGE)

    # Simulate note-off (release phase)
    runtime.set_energy_state(0)  # Zero energy
    print(f"4. Set energy_state to 0 (note-off/release)")

    # Tick to let AGE decay
    ages_release = []
    for i in range(100):
        runtime.tick()
        if i % 20 == 0:
            ages_release.append(runtime.get_system_macro(SystemMacro.AGE))

    print(f"5. AGE during release (100 ticks): {ages_release}")

    age_after_release = runtime.get_system_macro(SystemMacro.AGE)

    # Verify behavior
    if age_during_note > 0:
        print("  OK: AGE accumulated during note-on")
    else:
        print("  FAIL: AGE did not accumulate")

    if age_after_release < age_during_note:
        print("  OK: AGE decayed during release")
    else:
        print("  FAIL: AGE did not decay during release")


def test_age_influences_intent():
    """Test that AGE affects CALM and SNAP intent lanes."""
    print("\n" + "=" * 60)
    print("TEST: AGE -> CALM/SNAP Intent")
    print("=" * 60)

    gesture = GestureBasis(
        gate=True,
        pressure_q15=Q15_HALF,
    )

    # Young voice (AGE=0)
    runtime_young = MacroRuntime()
    runtime_young.macro_curr_q15[SystemMacro.AGE] = 0

    # Old voice (AGE=32767)
    runtime_old = MacroRuntime()
    runtime_old.macro_curr_q15[SystemMacro.AGE] = Q15_ONE

    intent_young = gesture_to_intent_with_macros(gesture, runtime_young)
    intent_old = gesture_to_intent_with_macros(gesture, runtime_old)

    print(f"\n1. Young voice (AGE=0):")
    print(f"   CALM={intent_young.calm}, SNAP={intent_young.snap}")

    print(f"\n2. Old voice (AGE={Q15_ONE}):")
    print(f"   CALM={intent_old.calm}, SNAP={intent_old.snap}")

    # Verify: old voices should be calmer
    if intent_old.calm > intent_young.calm:
        print("  OK: Old voices have more CALM")
    else:
        print("  FAIL: AGE did not increase CALM")

    if intent_old.snap <= intent_young.snap:
        print("  OK: Old voices have less SNAP")
    else:
        print("  FAIL: AGE did not reduce SNAP")


def run_all_tests():
    """Run all macro->CMMM integration tests."""
    print("\n" + "=" * 60)
    print("MACRO -> CMMM INTEGRATION TESTS")
    print("=" * 60)

    test_brightness_effect()
    test_risk_effect()
    test_density_effect()
    test_control_program_defaults()
    test_age_integrator()
    test_age_influences_intent()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
