"""
Pico2 Hardware Controller Demo
==============================

Demonstrates real-time synthesis controlled by the Pico2 hardware controller.
Potentiometers on the Pico2 send SetMacro messages which are received via USB CDC
and routed to the synthesis pipeline parameters.

Hardware setup:
- Pico2 W connected via USB
- 6 potentiometers on MCP3008 ADC (channels 0-5)

Macro Mapping:
- Macro 0: Gain/Volume
- Macro 1: Tone/Brightness
- Macro 2: Tree Blend
- Macro 3: Drive
- Macro 4: Attack Time
- Macro 5: Release Time

Usage:
    python pico2_hardware_demo.py [COM_PORT]
"""

import sys
import os
import time
import threading
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.symcrash_python_ap.comm.pico2_receiver import (
    Pico2Receiver,
    find_pico2_port,
)
from src.symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from src.symcrash_python_ap.frames import VoiceFrame, ParamFrame
from src.symcrash_python_ap.fixed import Q15_ONE, Q15_HALF


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SECOND = SAMPLE_RATE // BLOCK_SIZE

# Default note to play
DEFAULT_NOTE = 60  # Middle C


# =============================================================================
# HARDWARE MACRO ROUTER
# =============================================================================

class HardwareMacroRouter:
    """
    Routes Pico2 hardware macro values to synthesis parameters.

    Provides a clean interface between the hardware controller and the
    synthesis pipeline, handling scaling and parameter mapping.
    """

    def __init__(self, receiver: Pico2Receiver):
        self.receiver = receiver

        # Internal state for smooth parameter changes
        self._gain_q15 = Q15_HALF
        self._tone_q15 = 0
        self._blend_q15 = 0
        self._drive_q15 = Q15_HALF
        self._attack_ms = 10.0
        self._release_ms = 200.0

        # Register callback for real-time updates
        receiver.add_macro_callback(self._on_macro_change)

    def _on_macro_change(self, macro_id: int, value_q15: int):
        """Handle macro value changes from hardware."""
        if macro_id == 0:
            # Macro 0: Gain (direct mapping)
            self._gain_q15 = value_q15
        elif macro_id == 1:
            # Macro 1: Tone (-1 to +1 range)
            self._tone_q15 = (value_q15 * 2) - Q15_ONE
        elif macro_id == 2:
            # Macro 2: Tree blend (0 to 1)
            self._blend_q15 = value_q15
        elif macro_id == 3:
            # Macro 3: Drive (0 to 1)
            self._drive_q15 = value_q15
        elif macro_id == 4:
            # Macro 4: Attack time (1-500ms range)
            normalized = value_q15 / Q15_ONE
            self._attack_ms = 1.0 + (normalized * 499.0)
        elif macro_id == 5:
            # Macro 5: Release time (10-2000ms range)
            normalized = value_q15 / Q15_ONE
            self._release_ms = 10.0 + (normalized * 1990.0)

    @property
    def gain_q15(self) -> int:
        return self._gain_q15

    @property
    def tone_q15(self) -> int:
        return self._tone_q15

    @property
    def blend_q15(self) -> int:
        return self._blend_q15

    @property
    def drive_q15(self) -> int:
        return self._drive_q15

    @property
    def attack_ms(self) -> float:
        return self._attack_ms

    @property
    def release_ms(self) -> float:
        return self._release_ms

    def update_voice_frame(self, frame: VoiceFrame) -> VoiceFrame:
        """
        Apply current hardware macro values to a VoiceFrame.

        Args:
            frame: VoiceFrame to update

        Returns:
            Updated VoiceFrame
        """
        frame.gain_q15 = self._gain_q15
        frame.tree_blend_q15 = self._blend_q15
        return frame


# =============================================================================
# DEMO APPLICATION
# =============================================================================

class Pico2HardwareDemo:
    """
    Demo application showing real-time synthesis with Pico2 hardware control.
    """

    def __init__(self, port: str = None):
        # Initialize receiver
        self.receiver = Pico2Receiver(port=port)

        # Initialize macro router
        self.router = HardwareMacroRouter(self.receiver)

        # Initialize synthesis
        self.stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

        # State
        self.running = False
        self.audio_buffer = []
        self.current_note = DEFAULT_NOTE
        self.voice_active = False

    def start(self) -> bool:
        """Start the demo."""
        if not self.receiver.start():
            print("Failed to start receiver - check port")
            return False

        self.running = True
        print("Demo started!")
        print("-" * 50)
        print("Controls:")
        print("  Pot 0: Gain/Volume")
        print("  Pot 1: Tone/Brightness")
        print("  Pot 2: Tree Blend")
        print("  Pot 3: Drive")
        print("  Pot 4: Attack Time")
        print("  Pot 5: Release Time")
        print("-" * 50)

        return True

    def stop(self):
        """Stop the demo."""
        self.running = False
        self.receiver.stop()

    def note_on(self, note: int = DEFAULT_NOTE, velocity: int = 100):
        """Start playing a note."""
        self.current_note = note
        velocity_q15 = (velocity * Q15_ONE) // 127

        self.stage.voice_on(
            voice_id=0,
            note=note,
            cents=0,
            velocity_q15=velocity_q15,
            external_adsr=True
        )
        self.voice_active = True
        print(f"Note ON: {note}")

    def note_off(self):
        """Stop the current note."""
        if self.voice_active:
            self.stage.voice_off(0)
            self.voice_active = False
            print("Note OFF")

    def render_block(self) -> tuple:
        """
        Render one block of audio.

        Returns:
            Tuple of (left_samples, right_samples) as numpy arrays
        """
        if not self.voice_active:
            return np.zeros(BLOCK_SIZE), np.zeros(BLOCK_SIZE)

        # Create frame with current parameters
        from src.symcrash_python_ap.tuning import midi_to_hz

        f0_hz = midi_to_hz(self.current_note)

        frame = VoiceFrame(
            voice_id=0,
            note=self.current_note,
            cents=0,
            f0_hz=f0_hz,
            gain_q15=self.router.gain_q15,
            pan_q15=0,
            tree_a_id=1,
            tree_b_id=2,
            tree_blend_q15=self.router.blend_q15,
            eta_q15=Q15_HALF,
        )

        # Render
        left, right = self.stage.render_block(frame)

        return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32)

    def run_interactive(self):
        """Run interactive demo with keyboard input."""
        print("\nInteractive Mode:")
        print("  'n' = Note On (middle C)")
        print("  'o' = Note Off")
        print("  'm' = Show macro values")
        print("  's' = Show stats")
        print("  'q' = Quit")
        print()

        try:
            while self.running:
                try:
                    # Non-blocking input on Windows
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()

                        if key == 'n':
                            self.note_on()
                        elif key == 'o':
                            self.note_off()
                        elif key == 'm':
                            self._print_macros()
                        elif key == 's':
                            self.receiver.print_stats()
                        elif key == 'q':
                            break

                except ImportError:
                    # Unix: use select
                    import select
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()

                        if key == 'n':
                            self.note_on()
                        elif key == 'o':
                            self.note_off()
                        elif key == 'm':
                            self._print_macros()
                        elif key == 's':
                            self.receiver.print_stats()
                        elif key == 'q':
                            break

                # Brief sleep
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nInterrupted")

    def _print_macros(self):
        """Print current macro values."""
        print("\nCurrent Macro Values:")
        print(f"  Gain:     {self.router.gain_q15 / Q15_ONE:.3f}")
        print(f"  Tone:     {self.router.tone_q15 / Q15_ONE:.3f}")
        print(f"  Blend:    {self.router.blend_q15 / Q15_ONE:.3f}")
        print(f"  Drive:    {self.router.drive_q15 / Q15_ONE:.3f}")
        print(f"  Attack:   {self.router.attack_ms:.1f} ms")
        print(f"  Release:  {self.router.release_ms:.1f} ms")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Get port from command line or auto-detect
    port = sys.argv[1] if len(sys.argv) > 1 else None

    if port is None:
        port = find_pico2_port()
        if port:
            print(f"Auto-detected Pico2 on {port}")
        else:
            print("No Pico2 detected. Specify port manually:")
            print(f"  python {sys.argv[0]} COM3")
            return 1

    # Create and run demo
    demo = Pico2HardwareDemo(port=port)

    if not demo.start():
        return 1

    try:
        demo.run_interactive()
    finally:
        demo.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
