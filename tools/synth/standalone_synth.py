#!/usr/bin/env python3
"""
Standalone Synthesis Test
=========================

Test synthesis pipeline without Pico2 hardware.
Uses keyboard input to trigger notes.
Uses sd.OutputStream callback for real-time audio.

Controls:
- SPACE: Toggle note on/off
- UP/DOWN: Change note (+/- 1 semitone)
- LEFT/RIGHT: Change note (+/- 12 semitones / octave)
- s: Print status
- q: Quit
"""

import sys
import threading
from typing import Optional

# Platform-specific keyboard input
if sys.platform == 'win32':
    import msvcrt
    def get_key() -> Optional[str]:
        """Get a keypress (non-blocking) on Windows."""
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b'\xe0':
                ch2 = msvcrt.getch()
                if ch2 == b'H': return 'UP'
                elif ch2 == b'P': return 'DOWN'
                elif ch2 == b'K': return 'LEFT'
                elif ch2 == b'M': return 'RIGHT'
            return ch.decode('utf-8', errors='ignore')
        return None
else:
    import select
    import tty
    import termios
    _old_settings = None

    def _setup_terminal():
        global _old_settings
        fd = sys.stdin.fileno()
        _old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def _restore_terminal():
        global _old_settings
        if _old_settings:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, _old_settings)

    def get_key() -> Optional[str]:
        """Get a keypress (non-blocking) on Unix."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A': return 'UP'
                        elif ch3 == 'B': return 'DOWN'
                        elif ch3 == 'C': return 'RIGHT'
                        elif ch3 == 'D': return 'LEFT'
            return ch
        return None

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

import sounddevice as sd
from symcrash_python_ap.audio.live_renderer import LiveSynthRenderer
from symcrash_python_ap.primitives import SAMPLE_RATE_HZ as SAMPLE_RATE, BLOCK_SIZE


class StandaloneSynth:
    def __init__(self):
        self.renderer = LiveSynthRenderer(block_size=BLOCK_SIZE)
        self.engine = self.renderer.engine
        self.stage = self.renderer.stage

        self.running = False
        self.stream = None

        self.current_note = 60
        self.current_voice_id = None
        self.note_active = False

        self.blocks_rendered = 0
        self.lock = threading.Lock()

    def _audio_callback(self, outdata, frames, time_info, status):
        """Fill arbitrary callback lengths from complete, safe mono blocks."""
        if status:
            print(f"Audio status: {status}")
        self.renderer.fill_mono(outdata[:, 0])
        with self.lock:
            self.blocks_rendered = self.engine.block_count

    def start(self):
        """Start the synthesizer with real-time audio callback."""
        print(f"Using default audio output: {sd.query_devices(kind='output')['name']}")

        self.running = True

        # Start audio stream with callback
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=BLOCK_SIZE * 16,  # Buffer 16 blocks
            latency='high',
        )
        self.stream.start()
        return True

    def stop(self):
        """Stop the synthesizer."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def note_on(self, note: int = 60, velocity: int = 100):
        """Queue the note; the callback owns voice and VM mutations."""
        self.current_note = note
        self.renderer.note_on(note, velocity)
        self.note_active = velocity > 0

    def note_off(self, note=None):
        """Release the addressed note, retaining all other sounding voices."""
        self.renderer.note_off(self.current_note if note is None else note)
        self.note_active = False

    def toggle_note(self):
        """Toggle note on/off."""
        if self.note_active:
            self.note_off()
        else:
            self.note_on(self.current_note)

    def print_status(self):
        """Print current status."""
        with self.lock:
            print(f"Status: blocks={self.blocks_rendered} "
                  f"note={self.current_note} "
                  f"voice={self.current_voice_id} "
                  f"active={self.note_active}")


def main():
    print("Standalone Synthesis Test (Real-time)")
    print("=" * 60)

    synth = StandaloneSynth()

    if not synth.start():
        sys.exit(1)

    print("\nControls:")
    print("  SPACE: Toggle note on/off")
    print("  UP/DOWN: Change note (+/- 1 semitone)")
    print("  LEFT/RIGHT: Change octave (+/- 12 semitones)")
    print("  s: Print status")
    print("  q: Quit")
    print("=" * 60)

    # Start with note on
    synth.note_on(60)

    # Setup terminal for Unix
    if sys.platform != 'win32':
        _setup_terminal()

    try:
        while synth.running:
            key = get_key()
            if key:
                if key == ' ':
                    synth.toggle_note()
                elif key == 'UP':
                    with synth.lock:
                        synth.current_note = min(127, synth.current_note + 1)
                    if synth.note_active:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    else:
                        print(f"Note: {synth.current_note}")
                elif key == 'DOWN':
                    with synth.lock:
                        synth.current_note = max(0, synth.current_note - 1)
                    if synth.note_active:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    else:
                        print(f"Note: {synth.current_note}")
                elif key == 'RIGHT':
                    with synth.lock:
                        synth.current_note = min(127, synth.current_note + 12)
                    if synth.note_active:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    else:
                        print(f"Note: {synth.current_note}")
                elif key == 'LEFT':
                    with synth.lock:
                        synth.current_note = max(0, synth.current_note - 12)
                    if synth.note_active:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    else:
                        print(f"Note: {synth.current_note}")
                elif key == 's':
                    synth.print_status()
                elif key == 'q':
                    print("\nQuitting...")
                    break

            sd.sleep(10)

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        if sys.platform != 'win32':
            _restore_terminal()
        synth.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
