"""
Pico2 Synth Bridge
==================

Bridges the Pico2 hardware controller to the VoiceEngine for ADSR + filter control.

Maps the 6 hardware potentiometers to synth parameters:
- P0: Attack (ADSR)
- P1: Decay (ADSR)
- P2: Sustain (ADSR)
- P3: Release (ADSR)
- P4: Cutoff (Filter)
- P5: Resonance (Filter)

Usage:
    from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge
    from symcrash_python_ap.voice import VoiceEngine

    engine = VoiceEngine(max_voices=8)
    bridge = Pico2SynthBridge(engine, port="COM3")
    bridge.start()

    # Bridge automatically routes pot changes to engine
"""

from typing import Optional, Dict
from dataclasses import dataclass

from .pico2_receiver import Pico2Receiver, find_pico2_port
from ..voice.voice_engine import VoiceEngine, GestureBasis


# =============================================================================
# CONSTANTS
# =============================================================================

# Macro ID -> Parameter mapping
# ADC0: ADSR + Filter (pots 0-5)
MACRO_ATTACK = 0
MACRO_DECAY = 1
MACRO_SUSTAIN = 2
MACRO_RELEASE = 3
MACRO_CUTOFF = 4
MACRO_RESO = 5

# ADC1: Oscillator controls (pots 6-9)
MACRO_HARMONICS = 6   # Harmonic mix / brightness
MACRO_ETA = 7         # Dry/wet blend (carrier vs wavelet texture)
MACRO_DETUNE = 8      # Detune / spread
MACRO_GAIN = 9        # Master gain

# Block timing constants (matching voice_engine.py)
BLOCK_SIZE = 64
SAMPLE_RATE = 48000
BLOCK_TIME_MS = (BLOCK_SIZE / SAMPLE_RATE) * 1000  # ~1.33ms per block

# ADSR range limits (in blocks)
MAX_ATTACK_BLOCKS = 250   # ~330ms max attack
MAX_DECAY_BLOCKS = 500    # ~665ms max decay
MAX_RELEASE_BLOCKS = 750  # ~1000ms max release

# Q15 maximum value
Q15_MAX = 32767


# =============================================================================
# SYNTH BRIDGE
# =============================================================================

@dataclass
class ADSRState:
    """Current ADSR parameter state."""
    attack_blocks: int = 8
    decay_blocks: int = 32
    sustain_level_q15: int = 24576  # 0.75
    release_blocks: int = 64


@dataclass
class FilterState:
    """Current filter parameter state."""
    cutoff_q15: int = 16384   # 0.5
    reso_q15: int = 0         # 0.0


@dataclass
class OscillatorState:
    """Current oscillator parameter state."""
    harmonics_q15: int = 0        # Harmonic mix (0 = fundamental only)
    eta_q15: int = 0              # Dry/wet blend (0 = pure carrier/dry)
    detune_q15: int = 16384       # Detune amount (center = no detune)
    gain_q15: int = 24576         # Master gain (~0.75)


class Pico2SynthBridge:
    """
    Bridges Pico2 hardware controller to VoiceEngine.

    Automatically routes pot changes to InstrumentProgram ADSR params
    and GestureBasis for filter/resonance control.
    """

    def __init__(
        self,
        engine: VoiceEngine,
        port: Optional[str] = None,
        program_id: int = 1001,
        virtual_mode: bool = False,
    ):
        """
        Initialize the synth bridge.

        Args:
            engine: VoiceEngine instance to control
            port: Serial port (auto-detect if None, ignored in virtual_mode)
            program_id: Default instrument program ID
            virtual_mode: If True, run without hardware (ImGui only)
        """
        self.engine = engine
        self.virtual_mode = virtual_mode
        self.port = None if virtual_mode else (port or find_pico2_port())
        self.program_id = program_id

        # State tracking
        self.adsr = ADSRState()
        self.filter = FilterState()
        self.osc = OscillatorState()

        # Create receiver with callback (works even without port for inject_macro)
        self.receiver = Pico2Receiver(port=self.port)
        self.receiver.add_macro_callback(self._on_macro_change)

    # =========================================================================
    # CONVERSION UTILITIES
    # =========================================================================

    @staticmethod
    def q15_to_blocks(value_q15: int, max_blocks: int) -> int:
        """
        Convert Q15 pot value to block count.

        Uses exponential scaling for more musical feel at low values.

        Args:
            value_q15: Q15 value (0-32767)
            max_blocks: Maximum block count

        Returns:
            Block count (1 to max_blocks)
        """
        # Linear scaling (could make exponential for better feel)
        return max(1, (value_q15 * max_blocks) // Q15_MAX)

    @staticmethod
    def q15_to_ms(value_q15: int, max_ms: float) -> float:
        """Convert Q15 to milliseconds for display."""
        return (value_q15 / Q15_MAX) * max_ms

    @staticmethod
    def blocks_to_ms(blocks: int) -> float:
        """Convert block count to milliseconds."""
        return blocks * BLOCK_TIME_MS

    # =========================================================================
    # MACRO HANDLING
    # =========================================================================

    def _on_macro_change(self, macro_id: int, value_q15: int):
        """
        Handle macro value change from Pico2.

        Called by Pico2Receiver when a SetMacro message is received.
        """
        if macro_id == MACRO_ATTACK:
            self._update_attack(value_q15)
        elif macro_id == MACRO_DECAY:
            self._update_decay(value_q15)
        elif macro_id == MACRO_SUSTAIN:
            self._update_sustain(value_q15)
        elif macro_id == MACRO_RELEASE:
            self._update_release(value_q15)
        elif macro_id == MACRO_CUTOFF:
            self._update_cutoff(value_q15)
        elif macro_id == MACRO_RESO:
            self._update_reso(value_q15)
        # Oscillator controls (ADC1)
        elif macro_id == MACRO_HARMONICS:
            self._update_harmonics(value_q15)
        elif macro_id == MACRO_ETA:
            self._update_eta(value_q15)
        elif macro_id == MACRO_DETUNE:
            self._update_detune(value_q15)
        elif macro_id == MACRO_GAIN:
            self._update_gain(value_q15)

    def _update_attack(self, value_q15: int):
        """Update attack time."""
        blocks = self.q15_to_blocks(value_q15, MAX_ATTACK_BLOCKS)
        self.adsr.attack_blocks = blocks

        # Update engine's default program
        self.engine.default_program.attack_blocks = blocks

        # Also update all existing voices' programs (hot-swap)
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].program.attack_blocks = blocks

    def _update_decay(self, value_q15: int):
        """Update decay time."""
        blocks = self.q15_to_blocks(value_q15, MAX_DECAY_BLOCKS)
        self.adsr.decay_blocks = blocks

        self.engine.default_program.decay_blocks = blocks

        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].program.decay_blocks = blocks

    def _update_sustain(self, value_q15: int):
        """Update sustain level."""
        self.adsr.sustain_level_q15 = value_q15

        self.engine.default_program.sustain_level_q15 = value_q15

        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].program.sustain_level_q15 = value_q15

    def _update_release(self, value_q15: int):
        """Update release time."""
        blocks = self.q15_to_blocks(value_q15, MAX_RELEASE_BLOCKS)
        self.adsr.release_blocks = blocks

        self.engine.default_program.release_blocks = blocks

        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].program.release_blocks = blocks

    def _update_cutoff(self, value_q15: int):
        """Update filter cutoff via gesture."""
        self.filter.cutoff_q15 = value_q15

        # Map cutoff to chaos_q15 in gesture (chaos drives filtering behavior)
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                state = self.engine.voices[vid]
                # Create new gesture with updated cutoff
                gesture = GestureBasis(
                    gate=state.gesture.gate,
                    pressure_q15=state.gesture.pressure_q15,
                    speed_q15=state.gesture.speed_q15,
                    chaos_q15=value_q15,  # Cutoff -> chaos
                    tilt_q15=state.gesture.tilt_q15,
                    position_q15=state.gesture.position_q15,
                    pan_q15=state.gesture.pan_q15,
                )
                self.engine.update_gesture(vid, gesture)

    def _update_reso(self, value_q15: int):
        """Update filter resonance via gesture."""
        self.filter.reso_q15 = value_q15

        # Map resonance to tilt_q15 in gesture
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                state = self.engine.voices[vid]
                gesture = GestureBasis(
                    gate=state.gesture.gate,
                    pressure_q15=state.gesture.pressure_q15,
                    speed_q15=state.gesture.speed_q15,
                    chaos_q15=state.gesture.chaos_q15,
                    tilt_q15=value_q15,  # Reso -> tilt
                    position_q15=state.gesture.position_q15,
                    pan_q15=state.gesture.pan_q15,
                )
                self.engine.update_gesture(vid, gesture)

    # =========================================================================
    # OSCILLATOR CONTROLS (ADC1)
    # =========================================================================

    def _update_harmonics(self, value_q15: int):
        """Update harmonic mix / brightness.

        This controls how much higher harmonics are mixed into the sound.
        0 = fundamental only, 32767 = full harmonic content.
        """
        self.osc.harmonics_q15 = value_q15

        # Apply to all active voices
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].harmonics_q15 = value_q15

    def _update_eta(self, value_q15: int):
        """Update dry/wet blend (carrier vs wavelet texture).

        0 = pure carrier (dry), 32767 = pure wavelet texture (wet).
        """
        self.osc.eta_q15 = value_q15

        # Apply to all active voices
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].eta_q15 = value_q15

    def _update_detune(self, value_q15: int):
        """Update detune / spread amount.

        Maps Q15 (0-32767) to cents (-2400 to +2400 = ±2 octaves).
        Center position (16384) = no detune.
        """
        self.osc.detune_q15 = value_q15

        # Convert Q15 to signed cents: 0 -> -2400, 16384 -> 0, 32767 -> +2400
        # Formula: cents = (value_q15 - 16384) * 2400 / 16384
        centered = value_q15 - 16384  # -16384 to +16383
        detune_cents = (centered * 2400) // 16384

        # Apply to all active voices
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].detune_cents = detune_cents

    def _update_gain(self, value_q15: int):
        """Update master gain.

        Controls the overall output level (multiplied with ADSR envelope).
        """
        self.osc.gain_q15 = value_q15

        # Apply to all active voices
        for vid in self.engine.active_voices():
            if vid in self.engine.voices:
                self.engine.voices[vid].master_gain_q15 = value_q15

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

    def get_adsr_display(self) -> Dict[str, str]:
        """
        Get ADSR values formatted for display.

        Returns:
            Dict with ATK, DCY, SUS, REL as strings
        """
        return {
            "ATK": f"{self.blocks_to_ms(self.adsr.attack_blocks):.0f}ms",
            "DCY": f"{self.blocks_to_ms(self.adsr.decay_blocks):.0f}ms",
            "SUS": f"{100 * self.adsr.sustain_level_q15 / Q15_MAX:.0f}%",
            "REL": f"{self.blocks_to_ms(self.adsr.release_blocks):.0f}ms",
        }

    def get_filter_display(self) -> Dict[str, str]:
        """
        Get filter values formatted for display.

        Returns:
            Dict with CUT, RES as strings
        """
        return {
            "CUT": f"{100 * self.filter.cutoff_q15 / Q15_MAX:.0f}%",
            "RES": f"{100 * self.filter.reso_q15 / Q15_MAX:.0f}%",
        }

    def get_osc_display(self) -> Dict[str, str]:
        """
        Get oscillator values formatted for display.

        Returns:
            Dict with HARM, ETA, DET, GAIN as strings
        """
        return {
            "HARM": f"{100 * self.osc.harmonics_q15 / Q15_MAX:.0f}%",
            "ETA": f"{100 * self.osc.eta_q15 / Q15_MAX:.0f}%",
            "DET": f"{100 * self.osc.detune_q15 / Q15_MAX:.0f}%",
            "GAIN": f"{100 * self.osc.gain_q15 / Q15_MAX:.0f}%",
        }

    def print_status(self):
        """Print current parameter status."""
        print("[Pico2SynthBridge] Status:")
        adsr = self.get_adsr_display()
        filter_ = self.get_filter_display()
        osc = self.get_osc_display()
        print(f"  ADSR: ATK={adsr['ATK']} DCY={adsr['DCY']} "
              f"SUS={adsr['SUS']} REL={adsr['REL']}")
        print(f"  Filter: CUT={filter_['CUT']} RES={filter_['RES']}")
        print(f"  Osc: HARM={osc['HARM']} ETA={osc['ETA']} "
              f"DET={osc['DET']} GAIN={osc['GAIN']}")


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_bridge(
    engine: VoiceEngine,
    port: Optional[str] = None,
    virtual_mode: bool = False,
) -> Pico2SynthBridge:
    """
    Create a Pico2SynthBridge with auto-detection.

    Args:
        engine: VoiceEngine to control
        port: Serial port (auto-detect if None, ignored in virtual_mode)
        virtual_mode: If True, run without hardware (ImGui only)

    Returns:
        Configured Pico2SynthBridge
    """
    return Pico2SynthBridge(engine, port=port, virtual_mode=virtual_mode)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    # Create a test engine
    engine = VoiceEngine(max_voices=4)

    # Get port from command line or auto-detect
    port = sys.argv[1] if len(sys.argv) > 1 else None

    # Create bridge
    bridge = Pico2SynthBridge(engine, port=port)

    if not bridge.start():
        print("Failed to start bridge")
        sys.exit(1)

    # Allocate a test voice
    vid = engine.allocate_voice(program_id=1001)
    gesture = GestureBasis(gate=True, pressure_q15=16384)
    engine.note_on(vid, gesture=gesture)
    print(f"Allocated test voice {vid}")

    print("Listening for pot changes (Ctrl+C to exit)...")
    print("-" * 50)

    try:
        while True:
            bridge.print_status()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()
        print("Done")
