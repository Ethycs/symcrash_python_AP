"""
Pico2 Synth Bridge
==================

Bridges the Pico2 hardware controller to the CockpitEngine for control input.

Maps the 10 hardware potentiometers to controls via CockpitEngine:
- P0: ATK_BLOCKS (Attack, 0-500 blocks = 0-665ms)
- P1: DCY_BLOCKS (Decay, 0-1000 blocks = 0-1333ms)
- P2: SUS_Q15 (Sustain level, 0-100%)
- P3: REL_BLOCKS (Release, 0-1500 blocks = 0-2000ms)
- P4: CHAOS (Cutoff equivalent, 0-100%)
- P5: TEXTURE (Resonance equivalent, 0-100%)
- P6: RESERVED_7 (Harmonics - future expansion)
- P7: MORPH (Eta/dry-wet blend, 0-100%)
- P8: DETUNE (Pitch offset, -2400 to +2400 cents = ±2 octaves)
- P9: MASTER_GAIN (Master volume, 0-100%)

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
# Matches original Pico2 hardware layout
POT_ATTACK = 0      # ATK_BLOCKS
POT_DECAY = 1       # DCY_BLOCKS
POT_SUSTAIN = 2     # SUS_Q15
POT_RELEASE = 3     # REL_BLOCKS
POT_CUTOFF = 4      # CHAOS (filter cutoff)
POT_RESO = 5        # TEXTURE (filter resonance)
POT_HARMONICS = 6   # RESERVED_7 (future)
POT_ETA = 7         # MORPH (dry/wet blend)
POT_DETUNE = 8      # DETUNE (pitch offset)
POT_GAIN = 9        # MASTER_GAIN

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
            "ATK": f"{block_to_ms(adsr['attack_blocks']):.0f}ms",
            "DCY": f"{block_to_ms(adsr['decay_blocks']):.0f}ms",
            "SUS": f"{100 * adsr['sustain_level_q15'] / Q15_MAX:.0f}%",
            "REL": f"{block_to_ms(adsr['release_blocks']):.0f}ms",
            "CUTOFF": f"{100 * self.cockpit.get_control_current(ControlID.CHAOS) / Q15_MAX:.0f}%",
            "RESO": f"{100 * self.cockpit.get_control_current(ControlID.TEXTURE) / Q15_MAX:.0f}%",
            "HARM": f"{100 * self.cockpit.get_control_current(ControlID.RESERVED_7) / Q15_MAX:.0f}%",
            "ETA": f"{100 * self.cockpit.get_control_current(ControlID.MORPH) / Q15_MAX:.0f}%",
            "DETUNE": f"{self.cockpit.get_control_current(ControlID.DETUNE)} cents",
            "GAIN": f"{100 * self.cockpit.get_control_current(ControlID.MASTER_GAIN) / Q15_MAX:.0f}%",
        }

    def print_status(self):
        """Print current parameter status."""
        print("[Pico2SynthBridge] Status:")
        values = self.get_control_display()
        print(f"  ADSR: ATK={values['ATK']} DCY={values['DCY']} "
              f"SUS={values['SUS']} REL={values['REL']}")
        print(f"  Filter: CUTOFF={values['CUTOFF']} RESO={values['RESO']}")
        print(f"  Osc: HARM={values['HARM']} ETA={values['ETA']} "
              f"DETUNE={values['DETUNE']} GAIN={values['GAIN']}")


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
