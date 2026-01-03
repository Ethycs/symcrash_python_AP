"""
Pico2 Unified Controller
========================

Unified routing from Pico2 hardware controller to both VoiceEngine (ADSR/Filter)
and MacroRuntime (Preset Macros: BRIGHT, BODY, IMPACT).

This controller receives SetMacro messages from the Pico2 hardware and routes
them to both control paths simultaneously:

Path A (VoiceEngine):
    - Pots 0-3: ADSR envelope (Attack, Decay, Sustain, Release blocks)
    - Pots 4-5: Filter (Cutoff, Resonance via GestureBasis)

Path B (MacroRuntime):
    - Pot 0: BRIGHT → brightness (system macro 4)
    - Pot 1: BODY → density (system macro 5)
    - Pot 2: IMPACT → energy_intent (system macro 1)

Usage:
    from symcrash_python_ap.voice import VoiceEngine
    from symcrash_python_ap.macro_runtime import MacroRuntime
    from symcrash_python_ap.comm.pico2_controller import Pico2Controller

    engine = VoiceEngine(max_voices=8)
    macro_runtime = MacroRuntime()
    controller = Pico2Controller(engine, macro_runtime, port="COM3")
    controller.start()

    # Controller automatically routes pot changes to both targets
"""

from typing import Optional, Dict, Callable, List
from dataclasses import dataclass

from .pico2_receiver import Pico2Receiver, find_pico2_port
from .pico2_synth_bridge import (
    Pico2SynthBridge,
    POT_TIMBRE,
    POT_TEXTURE,
    POT_DYNAMICS,
    POT_MORPH,
    POT_CHAOS,
    POT_DETUNE,
    POT_MASTER_GAIN,
    POT_ATTACK,
    POT_DECAY,
    POT_SUSTAIN,
)
from ..cockpit_engine import CockpitEngine
from ..macro_runtime import MacroRuntime


# =============================================================================
# CONSTANTS
# =============================================================================

# Pot-to-preset-macro mapping for MacroRuntime path
POT_TO_PRESET_MACRO = {
    0: "BRIGHT",   # Pot 0 -> BRIGHT -> brightness (system macro 4)
    1: "BODY",     # Pot 1 -> BODY -> density (system macro 5)
    2: "IMPACT",   # Pot 2 -> IMPACT -> energy_intent (system macro 1)
}


# =============================================================================
# PICO2 UNIFIED CONTROLLER
# =============================================================================

class Pico2Controller:
    """
    Unified Pico2 → Python control routing.

    Routes hardware pot values to both:
    - CockpitEngine (control parameters via Pico2SynthBridge)
    - MacroRuntime (Preset macros: BRIGHT, BODY, IMPACT)
    """

    def __init__(
        self,
        cockpit: CockpitEngine,
        macro_runtime: MacroRuntime,
        port: Optional[str] = None,
    ):
        """
        Initialize the unified controller.

        Args:
            cockpit: CockpitEngine instance for control parameters
            macro_runtime: MacroRuntime instance for preset macro control
            port: Serial port (auto-detect if None)
        """
        self.cockpit = cockpit
        self.macro_runtime = macro_runtime
        self.port = port or find_pico2_port()

        # Create receiver
        self.receiver = Pico2Receiver(port=self.port)

        # Create synth bridge (handles CockpitEngine path)
        # Note: We share the same receiver but intercept at controller level
        self.synth_bridge = Pico2SynthBridge(
            cockpit,
            port=self.port,
        )
        # Remove the bridge's callback - we'll handle routing ourselves
        self.synth_bridge.receiver.remove_macro_callback(
            self.synth_bridge._on_macro_change
        )

        # Add our unified callback
        self.receiver.add_macro_callback(self._on_macro_change)

        # Additional callbacks for user extensions
        self._user_callbacks: List[Callable[[int, int], None]] = []

    # =========================================================================
    # MACRO ROUTING
    # =========================================================================

    def _on_macro_change(self, macro_id: int, value_q15: int):
        """
        Handle macro value change from Pico2.

        Routes to both VoiceEngine (ADSR/Filter) and MacroRuntime (preset macros).
        """
        # === Path A: VoiceEngine (ADSR/Filter) ===
        self._route_to_voice_engine(macro_id, value_q15)

        # === Path B: MacroRuntime (Preset Macros) ===
        self._route_to_macro_runtime(macro_id, value_q15)

        # === User callbacks ===
        for callback in self._user_callbacks:
            try:
                callback(macro_id, value_q15)
            except Exception as e:
                print(f"[Pico2Controller] User callback error: {e}")

    def _route_to_voice_engine(self, macro_id: int, value_q15: int):
        """Route macro to VoiceEngine ADSR/Filter/Oscillator parameters."""
        # Delegate to synth bridge's internal handlers
        if macro_id == MACRO_ATTACK:
            self.synth_bridge._update_attack(value_q15)
        elif macro_id == MACRO_DECAY:
            self.synth_bridge._update_decay(value_q15)
        elif macro_id == MACRO_SUSTAIN:
            self.synth_bridge._update_sustain(value_q15)
        elif macro_id == MACRO_RELEASE:
            self.synth_bridge._update_release(value_q15)
        elif macro_id == MACRO_CUTOFF:
            self.synth_bridge._update_cutoff(value_q15)
        elif macro_id == MACRO_RESO:
            self.synth_bridge._update_reso(value_q15)
        # Oscillator controls (ADC1: pots 6-9)
        elif macro_id == MACRO_HARMONICS:
            self.synth_bridge._update_harmonics(value_q15)
        elif macro_id == MACRO_ETA:
            self.synth_bridge._update_eta(value_q15)
        elif macro_id == MACRO_DETUNE:
            self.synth_bridge._update_detune(value_q15)
        elif macro_id == MACRO_GAIN:
            self.synth_bridge._update_gain(value_q15)

    def _route_to_macro_runtime(self, macro_id: int, value_q15: int):
        """Route macro to MacroRuntime preset macros."""
        preset_macro = POT_TO_PRESET_MACRO.get(macro_id)
        if preset_macro:
            self.macro_runtime.set_macro(preset_macro, value_q15)

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def add_callback(self, callback: Callable[[int, int], None]):
        """
        Add callback for macro value changes.

        Callback signature: callback(macro_id: int, value_q15: int)
        Called after routing to VoiceEngine and MacroRuntime.
        """
        self._user_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[int, int], None]):
        """Remove a user callback."""
        if callback in self._user_callbacks:
            self._user_callbacks.remove(callback)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> bool:
        """
        Start the controller.

        Returns:
            True if started successfully
        """
        if self.port is None:
            print("[Pico2Controller] No port specified and auto-detect failed")
            return False

        print(f"[Pico2Controller] Starting on {self.port}")
        return self.receiver.start()

    def stop(self):
        """Stop the controller."""
        self.receiver.stop()

    def is_connected(self) -> bool:
        """Check if connected to Pico2."""
        return self.receiver.is_connected()

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_control_display(self) -> Dict[str, str]:
        """Get control values formatted for display."""
        return self.synth_bridge.get_control_display()

    def get_preset_macro(self, preset_macro: str) -> int:
        """
        Get current value of a preset macro (from MacroRuntime).

        Args:
            preset_macro: Preset macro name ("BRIGHT", "BODY", "IMPACT")

        Returns:
            Current Q15 value (after slew)
        """
        return self.macro_runtime.get_preset_macro(preset_macro)

    def print_status(self):
        """Print current parameter status."""
        print("[Pico2Controller] Status:")

        # Controls from CockpitEngine
        self.synth_bridge.print_status()

        # Preset macros from macro runtime
        bright = self.macro_runtime.get_preset_macro("BRIGHT")
        body = self.macro_runtime.get_preset_macro("BODY")
        impact = self.macro_runtime.get_preset_macro("IMPACT")
        print(f"  Macros: BRIGHT={bright} BODY={body} IMPACT={impact}")


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_controller(
    cockpit: CockpitEngine,
    macro_runtime: MacroRuntime,
    port: Optional[str] = None
) -> Pico2Controller:
    """
    Create a Pico2Controller with auto-detection.

    Args:
        cockpit: CockpitEngine to control
        macro_runtime: MacroRuntime to control
        port: Serial port (auto-detect if None)

    Returns:
        Configured Pico2Controller
    """
    return Pico2Controller(cockpit, macro_runtime, port=port)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    # Create test instances
    cockpit = CockpitEngine()
    macro_runtime = MacroRuntime()

    # Get port from command line or auto-detect
    port = sys.argv[1] if len(sys.argv) > 1 else None

    # Create controller
    controller = Pico2Controller(cockpit, macro_runtime, port=port)

    if not controller.start():
        print("Failed to start controller")
        sys.exit(1)

    print("Listening for pot changes (Ctrl+C to exit)...")
    print("-" * 50)

    try:
        while True:
            # Tick cockpit and macro runtime slew filters
            cockpit.tick()
            macro_runtime.tick()

            controller.print_status()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.stop()
        print("Done")
