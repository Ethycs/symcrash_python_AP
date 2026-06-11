"""
Synth Bridge - Injects control messages into the Pico2Receiver.

Instead of trying to open a separate UART connection (which would conflict
with pico2_live_synth.py), this directly injects macro changes into the
Pico2Receiver's callback chain - following the same control path as hardware.
"""

from typing import Optional, Callable
from symcrash_python_ap.comm.pico2_synth_bridge import (
    Pico2SynthBridge,
    POT_ATTACK as MACRO_ATTACK,
    POT_DECAY as MACRO_DECAY,
    POT_SUSTAIN as MACRO_SUSTAIN,
    POT_RELEASE as MACRO_RELEASE,
    POT_CUTOFF as MACRO_CUTOFF,
    POT_RESO as MACRO_RESO,
    POT_HARMONICS as MACRO_HARMONICS,
    POT_ETA as MACRO_ETA,
    POT_DETUNE as MACRO_DETUNE,
    POT_GAIN as MACRO_GAIN,
)


class ImGuiSynthBridge:
    """
    Bridge ImGui controls to VoiceEngine via Pico2SynthBridge.

    This uses the Pico2Receiver.inject_macro() method to send macro changes
    through the same callback chain as hardware messages. This means:
    - Same control path as hardware pots
    - No UART port conflicts
    - Works alongside hardware controls
    """

    def __init__(self, pico2_bridge: Optional[Pico2SynthBridge] = None):
        """
        Initialize the bridge.

        Args:
            pico2_bridge: Existing Pico2SynthBridge to share
                         (creates internal state if None)
        """
        self.bridge = pico2_bridge
        self._connected = pico2_bridge is not None

        # Callback for state changes (for UI updates)
        self._on_change_callback: Optional[Callable[[int, int], None]] = None

    def set_bridge(self, bridge: Pico2SynthBridge) -> None:
        """Connect to an existing Pico2SynthBridge."""
        self.bridge = bridge
        self._connected = True

    def is_connected(self) -> bool:
        """Check if connected to a bridge."""
        return self._connected and self.bridge is not None

    def set_on_change_callback(self, callback: Callable[[int, int], None]) -> None:
        """Set callback for when macros change."""
        self._on_change_callback = callback

    def send_macro(self, macro_id: int, value_q15: int) -> bool:
        """
        Send macro value change via Pico2Receiver.inject_macro().

        This follows the same control path as hardware:
        inject_macro() -> callbacks -> _on_macro_change() -> VoiceEngine

        Args:
            macro_id: Macro index (0-9)
            value_q15: Q15 value (0-32767)

        Returns:
            True if sent successfully
        """
        if not self.is_connected():
            return False

        try:
            # Inject through the receiver's callback chain (same as hardware)
            self.bridge.receiver.inject_macro(macro_id, value_q15)

            # Notify UI
            if self._on_change_callback:
                self._on_change_callback(macro_id, value_q15)

            return True
        except Exception as e:
            print(f"[ImGuiSynthBridge] send_macro error: {e}")
            return False

    def send_note(self, note: int, velocity: int, on: bool) -> bool:
        """
        Send note event.

        Note: The Pico2SynthBridge doesn't handle notes directly -
        they go to the VoiceEngine. This is a placeholder for
        future implementation.
        """
        if not self.is_connected():
            return False

        # The bridge's engine can handle notes
        if self.bridge and self.bridge.engine:
            try:
                from symcrash_python_ap.voice import GestureBasis

                if on:
                    # Allocate voice and trigger note
                    vid = self.bridge.engine.allocate_voice(program_id=1001)
                    if vid is not None:
                        pressure = (velocity * 32767) // 127
                        gesture = GestureBasis(gate=True, pressure_q15=pressure)
                        self.bridge.engine.note_on(vid, gesture=gesture)
                        return True
                else:
                    # Note off for all active voices
                    for vid in self.bridge.engine.active_voices():
                        self.bridge.engine.note_off(vid)
                    return True
            except Exception as e:
                print(f"[ImGuiSynthBridge] send_note error: {e}")
                return False

        return False

    def get_macro_value(self, macro_id: int) -> int:
        """Get current macro value from bridge state."""
        if not self.is_connected():
            return 0

        # Read from bridge's internal state
        if macro_id == MACRO_ATTACK:
            return int((self.bridge.adsr.attack_blocks / 250) * 32767)
        elif macro_id == MACRO_DECAY:
            return int((self.bridge.adsr.decay_blocks / 500) * 32767)
        elif macro_id == MACRO_SUSTAIN:
            return self.bridge.adsr.sustain_level_q15
        elif macro_id == MACRO_RELEASE:
            return int((self.bridge.adsr.release_blocks / 750) * 32767)
        elif macro_id == MACRO_CUTOFF:
            return self.bridge.filter.cutoff_q15
        elif macro_id == MACRO_RESO:
            return self.bridge.filter.reso_q15
        elif macro_id == MACRO_HARMONICS:
            return self.bridge.osc.harmonics_q15
        elif macro_id == MACRO_ETA:
            return self.bridge.osc.eta_q15
        elif macro_id == MACRO_DETUNE:
            return self.bridge.osc.detune_q15
        elif macro_id == MACRO_GAIN:
            return self.bridge.osc.gain_q15

        return 0

    def sync_state_from_bridge(self) -> list:
        """
        Get all macro values from bridge.

        Returns:
            List of 10 Q15 values
        """
        return [self.get_macro_value(i) for i in range(10)]
