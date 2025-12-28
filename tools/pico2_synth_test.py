#!/usr/bin/env python3
"""
Pico2 Synth Integration Test
============================

Tests the Pico2 hardware controller -> VoiceEngine -> Audio pipeline.

Controls:
- Pots 0-3: ADSR envelope (Attack, Decay, Sustain, Release)
- Pots 4-5: Filter (Cutoff, Resonance)
- Buttons: Trigger notes

Usage:
    python pico2_synth_test.py [PORT]

    PORT: Serial port (e.g., COM3, /dev/ttyACM0)
          If not specified, auto-detects Pico2.

Example:
    python pico2_synth_test.py COM3
"""

import sys
import time
import threading
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, audio rendering disabled")

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available, audio output disabled")

from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge, create_bridge
from symcrash_python_ap.comm.pico2_receiver import find_pico2_port


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
NUM_VOICES = 4

# Test note parameters
TEST_NOTE = 60  # C4
TEST_VELOCITY = 100


# =============================================================================
# AUDIO RENDERING
# =============================================================================

class AudioRenderer:
    """
    Simple audio renderer for testing.

    Takes VoiceFrame output and renders basic audio.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.phase_inc = 0.0

    def set_frequency(self, freq_hz: float):
        """Set oscillator frequency."""
        self.phase_inc = freq_hz / self.sample_rate

    def render_block(self, gain_q15: int, block_size: int = BLOCK_SIZE) -> np.ndarray:
        """
        Render a block of audio samples.

        Args:
            gain_q15: Gain from VoiceFrame (0-32767)
            block_size: Number of samples

        Returns:
            Float32 array of audio samples
        """
        if not NUMPY_AVAILABLE:
            return None

        # Generate sine wave
        samples = np.zeros(block_size, dtype=np.float32)
        gain = gain_q15 / 32767.0

        for i in range(block_size):
            samples[i] = np.sin(2 * np.pi * self.phase) * gain
            self.phase += self.phase_inc
            if self.phase >= 1.0:
                self.phase -= 1.0

        return samples


# =============================================================================
# MAIN TEST
# =============================================================================

class Pico2SynthTest:
    """Integration test for Pico2 -> VoiceEngine -> Audio."""

    def __init__(self, port: Optional[str] = None):
        """
        Initialize the test.

        Args:
            port: Serial port (auto-detect if None)
        """
        # Create voice engine
        self.engine = VoiceEngine(max_voices=NUM_VOICES)

        # Create bridge
        self.bridge = create_bridge(self.engine, port=port)

        # Audio
        self.renderer = AudioRenderer() if NUMPY_AVAILABLE else None
        self.stream: Optional[sd.OutputStream] = None

        # State
        self.running = False
        self.current_voice_id = None
        self.audio_buffer = []
        self.lock = threading.Lock()

        # Stats
        self.frames_rendered = 0
        self.last_gain = 0

    def start(self) -> bool:
        """Start the test."""
        # Start bridge
        if not self.bridge.start():
            print("Failed to start Pico2 bridge")
            return False

        # Start audio output
        if AUDIO_AVAILABLE and self.renderer:
            self._start_audio()

        self.running = True
        return True

    def stop(self):
        """Stop the test."""
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.bridge.stop()

    def _start_audio(self):
        """Start audio output stream."""
        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")

            with self.lock:
                if self.audio_buffer:
                    # Get buffered audio
                    block = self.audio_buffer.pop(0)
                    if len(block) == frames:
                        outdata[:, 0] = block
                        return

            # Silence if no buffer
            outdata.fill(0)

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            callback=audio_callback,
        )
        self.stream.start()

    def trigger_note(self, on: bool = True):
        """Trigger or release a test note."""
        if on:
            # Allocate voice and trigger note
            self.current_voice_id = self.engine.allocate_voice(program_id=1001)
            if self.current_voice_id is not None:
                gesture = GestureBasis(
                    gate=True,
                    pressure_q15=16384,  # 0.5
                )
                self.engine.note_on(self.current_voice_id, gesture=gesture)

                # Set frequency for renderer
                freq = 440.0 * (2 ** ((TEST_NOTE - 69) / 12))  # MIDI to Hz
                if self.renderer:
                    self.renderer.set_frequency(freq)

                print(f"Note ON (voice {self.current_voice_id}, {freq:.1f}Hz)")
        else:
            # Release note
            if self.current_voice_id is not None:
                self.engine.note_off(self.current_voice_id)
                print(f"Note OFF (voice {self.current_voice_id})")

    def step(self):
        """Process one audio block."""
        if not self.running:
            return

        # Step voice engine
        frames = self.engine.step_all()

        # Get gain from active voice
        gain_q15 = 0
        if frames:
            for frame in frames:
                if frame.is_active:
                    gain_q15 = frame.gain_q15
                    break

        self.last_gain = gain_q15
        self.frames_rendered += 1

        # Render audio
        if self.renderer and AUDIO_AVAILABLE:
            samples = self.renderer.render_block(gain_q15)
            if samples is not None:
                with self.lock:
                    self.audio_buffer.append(samples)
                    # Keep buffer reasonable
                    while len(self.audio_buffer) > 10:
                        self.audio_buffer.pop(0)

    def print_status(self):
        """Print current status."""
        print("-" * 60)
        self.bridge.print_status()
        print(f"  Audio: frames={self.frames_rendered} gain={self.last_gain}")
        print(f"  Voice: {self.current_voice_id} "
              f"active={self.engine.active_voices()}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pico2 Synth Integration Test")
    parser.add_argument("port", nargs="?", help="Serial port (auto-detect if omitted)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio output")
    args = parser.parse_args()

    # Find port
    port = args.port or find_pico2_port()
    if not port:
        print("No Pico2 detected. Please specify port manually.")
        print(f"Usage: {sys.argv[0]} <PORT>")
        sys.exit(1)

    print(f"Using port: {port}")

    # Create test
    test = Pico2SynthTest(port=port)

    if not test.start():
        sys.exit(1)

    print("\nPico2 Synth Test")
    print("=" * 60)
    print("Controls:")
    print("  Pots 0-3: ADSR (Attack, Decay, Sustain, Release)")
    print("  Pots 4-5: Filter (Cutoff, Resonance)")
    print("\nKeyboard commands:")
    print("  SPACE: Toggle note on/off")
    print("  s: Print status")
    print("  q: Quit")
    print("=" * 60)

    # Start with note on
    test.trigger_note(on=True)

    # Main loop
    note_on = True
    try:
        while True:
            # Step engine
            test.step()

            # Print status periodically
            if test.frames_rendered % 750 == 0:  # ~1 second
                test.print_status()

            # Small sleep to prevent busy loop
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        test.stop()
        print("Done.")


if __name__ == "__main__":
    main()
