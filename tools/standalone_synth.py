#!/usr/bin/env python3
"""
Standalone Synthesis Test
=========================

Test synthesis pipeline without Pico2 hardware.
Uses keyboard input to trigger notes.
Uses pre-rendering in a background thread + sd.play() (reliable on Windows).

Controls:
- SPACE: Toggle note on/off
- UP/DOWN: Change note (+/- 1 semitone)
- LEFT/RIGHT: Change note (+/- 12 semitones / octave)
- s: Print status
- q: Quit
"""

import sys
import threading
import queue
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

import numpy as np
import sounddevice as sd
from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame

SAMPLE_RATE = 44100
BLOCK_SIZE = 64
Q15_ONE = 32767
CHUNK_BLOCKS = 50  # Render 50 blocks at a time (~73ms at 44100Hz)


class StandaloneSynth:
    def __init__(self):
        self.engine = VoiceEngine(max_voices=4)
        self.stage = UnifiedOutputStage(block_size=BLOCK_SIZE)

        self.running = False
        self.render_thread = None
        self.play_thread = None

        self.current_note = 60
        self.current_voice_id = None
        self.note_active = False

        self.blocks_rendered = 0
        self.lock = threading.Lock()

        # Queue for rendered audio chunks
        self.audio_queue = queue.Queue(maxsize=4)

    def _render_loop(self):
        """Background thread that continuously renders audio chunks."""
        while self.running:
            chunk_samples = []

            for _ in range(CHUNK_BLOCKS):
                voice_frames = self.engine.step_all()

                # Find first active voice
                active_frame = None
                for vf in voice_frames:
                    if vf.is_active and vf.gain_q15 > 0:
                        active_frame = vf
                        break

                if active_frame is not None:
                    with self.lock:
                        note = self.current_note
                    f0_hz = 440.0 * (2.0 ** ((note - 69) / 12.0))
                    frame = VoiceFrame(
                        voice_id=active_frame.voice_id,
                        note=note,
                        f0_hz=f0_hz,
                        gain_q15=active_frame.gain_q15,
                        is_active=True,
                    )

                    left, right = self.stage.render_block(frame)
                    # Convert to float
                    left_float = np.array(left, dtype=np.float32) / Q15_ONE
                    chunk_samples.extend(left_float)
                else:
                    # Silence
                    chunk_samples.extend([0.0] * BLOCK_SIZE)

                with self.lock:
                    self.blocks_rendered += 1

            # Put chunk in queue (blocks if queue is full)
            if chunk_samples:
                audio = np.array(chunk_samples, dtype=np.float32)
                try:
                    self.audio_queue.put(audio, timeout=0.5)
                except queue.Full:
                    pass  # Drop chunk if queue full

    def _play_loop(self):
        """Background thread that plays audio chunks from queue."""
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=0.1)
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
            except queue.Empty:
                pass

    def start(self):
        """Start the synthesizer."""
        # Use device 4 (Realtek Digital Output - proven working)
        sd.default.device = 4
        print(f"Using audio device 4: {sd.query_devices(4)['name']}")

        self.running = True

        # Start render thread
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        # Start play thread
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

        return True

    def stop(self):
        """Stop the synthesizer."""
        self.running = False
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        sd.stop()

    def note_on(self, note: int = 60, velocity: int = 100):
        """Trigger a note."""
        with self.lock:
            self.current_note = note

        self.current_voice_id = self.engine.allocate_voice(program_id=1001)

        if self.current_voice_id is not None:
            pressure = (velocity * Q15_ONE) // 127
            gesture = GestureBasis(gate=True, pressure_q15=pressure)
            self.engine.note_on(self.current_voice_id, gesture=gesture)
            self.stage.voice_on(self.current_voice_id, note)
            self.note_active = True

            freq = 440.0 * (2 ** ((note - 69) / 12))
            print(f"Note ON: {note} (voice {self.current_voice_id}, {freq:.1f}Hz)")

    def note_off(self):
        """Release the current note."""
        if self.current_voice_id is not None:
            self.engine.note_off(self.current_voice_id)
            self.stage.voice_off(self.current_voice_id)
            print(f"Note OFF: voice {self.current_voice_id}")
            self.current_voice_id = None
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
                  f"active={self.note_active} "
                  f"queue={self.audio_queue.qsize()}")


def main():
    print("Standalone Synthesis Test")
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
