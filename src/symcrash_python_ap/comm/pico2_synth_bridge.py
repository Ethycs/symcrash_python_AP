"""
Pico2 Synth Bridge
==================

Bridges the Pico2 hardware controller to the CockpitEngine for control input.

Maps the 10 hardware potentiometers to controls via CockpitEngine:
- P0: TIMBRE
- P1: TEXTURE
- P2: DYNAMICS
- P3: MORPH
- P4: CHAOS
- P5: DETUNE
- P6: MASTER_GAIN
- P7: ATK_BLOCKS (Attack)
- P8: DCY_BLOCKS (Decay)
- P9: SUS_Q15 (Sustain)

Usage:
    from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge
    from symcrash_python_ap.cockpit_engine import CockpitEngine

    cockpit = CockpitEngine()
    bridge = Pico2SynthBridge(cockpit, port="COM3")
    bridge.start()

    # Bridge automatically routes pot changes to cockpit
"""

from typing import Optional, Dict

from .pico2_receiver import Pico2Receiver, find_pico2_port
from ..cockpit_engine import CockpitEngine
from ..control_registry import ControlID


# =============================================================================
# CONSTANTS
# =============================================================================

# Pot ID -> Control mapping (via CockpitEngine.set_pot())
# ADC0: Timbral controls (pots 0-4)
POT_TIMBRE = 0
POT_TEXTURE = 1
POT_DYNAMICS = 2
POT_MORPH = 3
POT_CHAOS = 4

# ADC0: Pitch and master (pots 5-6)
POT_DETUNE = 5
POT_MASTER_GAIN = 6

# ADC1: ADSR controls (pots 7-9)
POT_ATTACK = 7
POT_DECAY = 8
POT_SUSTAIN = 9

# Q15 maximum value
Q15_MAX = 32767


# =============================================================================
# SYNTH BRIDGE
# =============================================================================


class Pico2SynthBridge:
    """
    Bridges Pico2 hardware controller to CockpitEngine.

    Automatically routes pot changes to CockpitEngine controls.
    """

    def __init__(
        self,
        cockpit: CockpitEngine,
        port: Optional[str] = None,
        virtual_mode: bool = False,
    ):
        """
        Initialize the synth bridge.

        Args:
            cockpit: CockpitEngine instance to control
            port: Serial port (auto-detect if None, ignored in virtual_mode)
            virtual_mode: If True, run without hardware (ImGui only)
        """
        self.cockpit = cockpit
        self.virtual_mode = virtual_mode
        self.port = None if virtual_mode else (port or find_pico2_port())

        # Create receiver with callback (works even without port for inject_macro)
        self.receiver = Pico2Receiver(port=self.port)
        self.receiver.add_macro_callback(self._on_macro_change)

    # =========================================================================
    # POT HANDLING
    # =========================================================================

    def _on_macro_change(self, macro_id: int, value_q15: int):
        """
        Handle pot value change from Pico2.

        Called by Pico2Receiver when a SetMacro message is received.

        All pots are routed directly to CockpitEngine.set_pot().
        """
        # Route directly to CockpitEngine (handles all mapping and slew)
        self.cockpit.set_pot(macro_id, value_q15)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> bool:
        """
        Start the bridge.

        Returns:
            True if started successfully
        """
        if self.virtual_mode:
            print("[Pico2SynthBridge] Starting in virtual mode (no hardware)")
            return True

        if self.port is None:
            print("[Pico2SynthBridge] No port specified and auto-detect failed")
            return False

        print(f"[Pico2SynthBridge] Starting on {self.port}")
        return self.receiver.start()

    def stop(self):
        """Stop the bridge."""
        self.receiver.stop()

    def is_connected(self) -> bool:
        """Check if connected to Pico2."""
        return self.receiver.is_connected()

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_control_display(self) -> Dict[str, str]:
        """
        Get control values formatted for display.

        Returns:
            Dict with control names as keys and formatted values as strings
        """
        adsr = self.cockpit.get_adsr()

        # Block to ms conversion (64 samples @ 48kHz = 1.33ms per block)
        block_to_ms = lambda b: b * 1.33

        return {
            "TIMBRE": f"{100 * self.cockpit.get_control_current(ControlID.TIMBRE) / Q15_MAX:.0f}%",
            "TEXTURE": f"{100 * self.cockpit.get_control_current(ControlID.TEXTURE) / Q15_MAX:.0f}%",
            "DYNAMICS": f"{100 * self.cockpit.get_control_current(ControlID.DYNAMICS) / Q15_MAX:.0f}%",
            "MORPH": f"{100 * self.cockpit.get_control_current(ControlID.MORPH) / Q15_MAX:.0f}%",
            "CHAOS": f"{100 * self.cockpit.get_control_current(ControlID.CHAOS) / Q15_MAX:.0f}%",
            "DETUNE": f"{self.cockpit.get_control_current(ControlID.DETUNE)} cents",
            "MASTER": f"{100 * self.cockpit.get_control_current(ControlID.MASTER_GAIN) / Q15_MAX:.0f}%",
            "ATK": f"{block_to_ms(adsr['attack_blocks']):.0f}ms",
            "DCY": f"{block_to_ms(adsr['decay_blocks']):.0f}ms",
            "SUS": f"{100 * adsr['sustain_level_q15'] / Q15_MAX:.0f}%",
            "REL": f"{block_to_ms(adsr['release_blocks']):.0f}ms",
        }

    def print_status(self):
        """Print current parameter status."""
        print("[Pico2SynthBridge] Status:")
        values = self.get_control_display()
        print(f"  Timbral: TIMBRE={values['TIMBRE']} TEXTURE={values['TEXTURE']} "
              f"DYNAMICS={values['DYNAMICS']}")
        print(f"  Morph: MORPH={values['MORPH']} CHAOS={values['CHAOS']}")
        print(f"  Pitch: DETUNE={values['DETUNE']} MASTER={values['MASTER']}")
        print(f"  ADSR: ATK={values['ATK']} DCY={values['DCY']} "
              f"SUS={values['SUS']} REL={values['REL']}")


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_bridge(
    cockpit: CockpitEngine,
    port: Optional[str] = None,
    virtual_mode: bool = False,
) -> Pico2SynthBridge:
    """
    Create a Pico2SynthBridge with auto-detection.

    Args:
        cockpit: CockpitEngine to control
        port: Serial port (auto-detect if None, ignored in virtual_mode)
        virtual_mode: If True, run without hardware (ImGui only)

    Returns:
        Configured Pico2SynthBridge
    """
    return Pico2SynthBridge(cockpit, port=port, virtual_mode=virtual_mode)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    # Create a test CockpitEngine
    cockpit = CockpitEngine()

    # Get port from command line or auto-detect
    port = sys.argv[1] if len(sys.argv) > 1 else None

    # Create bridge
    bridge = Pico2SynthBridge(cockpit, port=port)

    if not bridge.start():
        print("Failed to start bridge")
        sys.exit(1)

    print("Listening for pot changes (Ctrl+C to exit)...")
    print("-" * 50)

    try:
        while True:
            bridge.print_status()

            # Tick the cockpit to see ControlFrame output
            frame = cockpit.tick()
            if frame.events or frame.voice_patches:
                print(f"  [Block {frame.block_index}] Events={len(frame.events)} Patches={len(frame.voice_patches)}")

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()
        print("Done")
