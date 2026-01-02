#!/usr/bin/env python3
"""
Virtual Synth with Multiprocessing
==================================

Runs synthesis in a separate process for zero GIL contention.
Audio callback just reads from shared memory - ultra fast.

Usage:
    python virtual_synth_mp.py
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available")

from symcrash_python_ap.audio import MultiprocessSynth

# Constants
BLOCK_SIZE = 64
SAMPLE_RATE = 48000


class VirtualSynthApp:
    """Virtual synth with multiprocess rendering and ImGui controls."""

    def __init__(self):
        self.synth = MultiprocessSynth(block_size=BLOCK_SIZE)
        self.stream = None
        self.current_note = 60

    def start(self) -> bool:
        """Start the synth."""
        if not self.synth.start():
            return False

        if AUDIO_AVAILABLE:
            self._start_audio()

        return True

    def stop(self):
        """Stop the synth."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.synth.stop()

    def _start_audio(self):
        """Start audio output - callback just reads from shared memory."""
        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio: {status}")

            num_blocks = frames // BLOCK_SIZE
            output_idx = 0

            for _ in range(num_blocks):
                block = self.synth.read_block()
                if block is not None:
                    outdata[output_idx:output_idx + BLOCK_SIZE, 0] = block
                else:
                    outdata[output_idx:output_idx + BLOCK_SIZE, 0] = 0.0
                output_idx += BLOCK_SIZE

            # Remainder
            remainder = frames % BLOCK_SIZE
            if remainder > 0:
                outdata[output_idx:, 0] = 0.0

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE * 16,
            channels=1,
            dtype='float32',
            callback=audio_callback,
            latency=0.05,  # 50ms (matches pico2_live_synth)
        )
        self.stream.start()

    def note_on(self, note: int = None):
        """Trigger note on."""
        if note is not None:
            self.current_note = note
        self.synth.note_on(self.current_note)
        print(f"Note ON: {self.current_note}")

    def note_off(self):
        """Trigger note off."""
        self.synth.note_off()
        print("Note OFF")

    def send_macro(self, macro_id: int, value_q15: int):
        """Send macro change."""
        self.synth.send_macro(macro_id, value_q15)


def main():
    """Run with ImGui controls."""
    app = VirtualSynthApp()

    if not app.start():
        print("Failed to start synth")
        sys.exit(1)

    # Start with note on
    app.note_on(60)

    try:
        from imgui_bundle import imgui, hello_imgui
        from symcrash_imgui_controls.app import SymcrashImGuiApp
        from symcrash_imgui_controls.synth_bridge import ImGuiSynthBridge

        print("\nLaunching ImGui control window...")
        print("Multiprocess mode: Zero GIL contention!")

        # Create a bridge adapter for the multiprocess synth
        class MultiprocessBridge:
            """Adapter to make MultiprocessSynth work with ImGuiSynthBridge interface."""

            def __init__(self, mp_synth: MultiprocessSynth):
                self.mp_synth = mp_synth
                # Dummy receiver with inject_macro that forwards to mp_synth
                self.receiver = type('Receiver', (), {
                    'inject_macro': lambda self, m, v: mp_synth.send_macro(m, v)
                })()
                # Dummy state objects for sync_state_from_bridge
                self.adsr = type('ADSR', (), {
                    'attack_blocks': 8, 'decay_blocks': 32,
                    'sustain_level_q15': 24576, 'release_blocks': 64
                })()
                self.filter = type('Filter', (), {
                    'cutoff_q15': 16384, 'reso_q15': 0
                })()
                self.osc = type('Osc', (), {
                    'harmonics_q15': 0, 'eta_q15': 16384,
                    'detune_q15': 16384, 'gain_q15': 24576
                })()
                self.engine = None  # No direct engine access

        # Create ImGui app with our bridge adapter
        bridge_adapter = MultiprocessBridge(app.synth)
        imgui_bridge = ImGuiSynthBridge(bridge_adapter)

        imgui_app = SymcrashImGuiApp()
        imgui_app.set_bridge(imgui_bridge)

        # Configure runner
        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = "Symcrash Virtual Synth (Multiprocess)"
        runner_params.app_window_params.window_geometry.size = (800, 500)
        runner_params.callbacks.show_gui = imgui_app.render

        # Run ImGui
        hello_imgui.run(runner_params)

    except ImportError as e:
        print(f"ImGui not available: {e}")
        print("Running in console mode...")

        # Console mode fallback
        print("\nControls: SPACE=toggle, q=quit")
        note_on = True

        if sys.platform == 'win32':
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b' ':
                        if note_on:
                            app.note_off()
                        else:
                            app.note_on()
                        note_on = not note_on
                    elif ch == b'q':
                        break
        else:
            import time
            time.sleep(10)

    finally:
        app.stop()
        print("Done.")


if __name__ == "__main__":
    main()
