#!/usr/bin/env python3
"""
Pico2 Live Synthesis Controller
===============================

Complete live control: Pico2 Hardware -> VoiceEngine -> UnifiedOutputStage -> Audio

This script connects all the pieces for real-time hardware control of synthesis:
- Pico2 pots control ADSR + Filter parameters
- VoiceEngine manages voice allocation and envelopes
- UnifiedOutputStage renders actual audio (Carrier + Wavelet Lift)
- sounddevice streams audio in real-time

Controls:
- Pots 0-3: ADSR envelope (Attack, Decay, Sustain, Release)
- Pots 4-5: Filter (Cutoff, Resonance)
- Keyboard: SPACE=toggle note, s=status, q=quit

Usage:
    python pico2_live_synth.py [PORT]
    python pico2_live_synth.py COM3
"""

import sys
import time
import threading
import argparse
import queue
from typing import Optional, List, Tuple

# Platform-specific keyboard input
if sys.platform == 'win32':
    import msvcrt
    def get_key() -> Optional[str]:
        """Get a keypress (non-blocking) on Windows."""
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # Handle arrow keys (escape sequences)
            if ch == b'\xe0':
                ch2 = msvcrt.getch()
                if ch2 == b'H':
                    return 'UP'
                elif ch2 == b'P':
                    return 'DOWN'
                elif ch2 == b'K':
                    return 'LEFT'
                elif ch2 == b'M':
                    return 'RIGHT'
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
            # Handle arrow keys (escape sequences)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':
                            return 'UP'
                        elif ch3 == 'B':
                            return 'DOWN'
                        elif ch3 == 'C':
                            return 'RIGHT'
                        elif ch3 == 'D':
                            return 'LEFT'
            return ch
        return None

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Error: numpy is required for audio rendering")
    sys.exit(1)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available, audio output disabled")

from symcrash_python_ap.voice import VoiceEngine, GestureBasis
from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
from symcrash_python_ap.frames import VoiceFrame
from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge, create_bridge
from symcrash_python_ap.comm.pico2_receiver import find_pico2_ports


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
NUM_VOICES = 4
Q15_ONE = 32767

# Buffer configuration
# At 48kHz/64 samples = 750 blocks/sec, each block = 1.33ms
# Benchmarks show full render path takes ~1500us (112% of 1333us budget)
# Need extra buffering to handle GIL jitter and occasional spikes
# 32 blocks = ~43ms buffer latency
AUDIO_BUFFER_BLOCKS = 32  # Number of blocks to buffer


# =============================================================================
# LIVE SYNTH CONTROLLER
# =============================================================================

class Pico2LiveSynth:
    """
    Complete live synthesis controller.

    Connects Pico2 hardware -> VoiceEngine -> UnifiedOutputStage -> Audio
    """

    def __init__(self, port: Optional[str] = None):
        """
        Initialize the live synth.

        Args:
            port: Serial port (auto-detect if None)
        """
        # Voice engine for ADSR and voice management
        self.engine = VoiceEngine(max_voices=NUM_VOICES)

        # Synthesis pipeline for audio rendering
        self.stage = UnifiedOutputStage(
            block_size=BLOCK_SIZE,
        )

        # Hardware bridge
        self.bridge = create_bridge(self.engine, port=port)

        # Audio streaming
        self.stream: Optional[sd.OutputStream] = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=AUDIO_BUFFER_BLOCKS * 2)

        # State
        self.running = False
        self.render_thread: Optional[threading.Thread] = None
        self.current_voice_id: Optional[int] = None
        self.current_note = 60  # C4

        # Stats
        self.blocks_rendered = 0
        self.buffer_underruns = 0
        self.lock = threading.Lock()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> bool:
        """Start the live synth."""
        # Start hardware bridge
        if not self.bridge.start():
            print("Failed to start Pico2 bridge")
            return False

        self.running = True

        # Start render thread
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        # Start audio output
        if AUDIO_AVAILABLE:
            self._start_audio()

        return True

    def stop(self):
        """Stop the live synth."""
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.render_thread:
            self.render_thread.join(timeout=1.0)
            self.render_thread = None

        self.bridge.stop()

    def _find_output_device(self) -> Optional[int]:
        """Find a suitable audio output device - use system default which user has configured."""
        # Use system default output device (user has device 4 = Realtek Digital Output working)
        default_dev = sd.default.device[1]
        if default_dev is not None:
            name = sd.query_devices(default_dev)['name']
            print(f"[Audio] Using default device {default_dev}: {name}")
            return default_dev

        # Fallback: find any output device
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] >= 2:
                print(f"[Audio] Using device {i}: {dev['name']}")
                return i

        print("[Audio] WARNING: No output device found!")
        return 0

    def _start_audio(self):
        """Start audio output stream."""
        output_device = self._find_output_device()

        # Pre-allocate conversion buffers (reused each callback to avoid allocation)
        left_f32 = np.zeros(BLOCK_SIZE, dtype=np.float32)
        right_f32 = np.zeros(BLOCK_SIZE, dtype=np.float32)
        q15_scale = np.float32(1.0 / Q15_ONE)

        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")

            try:
                # Get pre-rendered audio block
                left, right = self.audio_queue.get_nowait()

                # Convert Q15 int16 arrays to float32 (vectorized, in-place)
                np.multiply(left, q15_scale, out=left_f32, casting='unsafe')
                np.multiply(right, q15_scale, out=right_f32, casting='unsafe')

                # Write to output
                outdata[:, 0] = left_f32
                outdata[:, 1] = right_f32

            except queue.Empty:
                # Buffer underrun - output silence
                outdata.fill(0)
                self.buffer_underruns += 1

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=2,
            dtype='float32',
            callback=audio_callback,
            device=output_device,
        )
        self.stream.start()

    # =========================================================================
    # RENDER LOOP
    # =========================================================================

    def _render_loop(self):
        """Background thread that renders audio blocks."""
        while self.running:
            try:
                # Check if we need to render more blocks
                if self.audio_queue.qsize() < AUDIO_BUFFER_BLOCKS:
                    self._render_block()
                else:
                    # Buffer is full, wait a bit
                    time.sleep(0.0001)
            except Exception as e:
                print(f"Render error: {e}")
                time.sleep(0.001)

    def _render_block(self):
        """Render one audio block."""
        # Step the voice engine to get voice frames
        voice_frames = self.engine.step_all()

        # Create VoiceFrame for the synthesis stage
        # Use first active voice, or silent if none
        frame = self._create_voice_frame(voice_frames)

        # Debug: log occasionally (every ~4 seconds to reduce overhead)
        if self.blocks_rendered % 3000 == 0:
            active_count = len([vf for vf in voice_frames if vf.is_active])
            print(f"[DEBUG] Block {self.blocks_rendered}: active={active_count} "
                  f"note={frame.note} gain={frame.gain_q15} "
                  f"eta={frame.beta_target_q15} harm={frame.partial_amps_q15[1] if frame.partial_amps_q15 else 0}")

        # Render through the synthesis pipeline
        left, right = self.stage.render_block(frame)

        # Queue for audio output (no lock needed - blocks_rendered is only for stats)
        try:
            self.audio_queue.put_nowait((left, right))
            self.blocks_rendered += 1
        except queue.Full:
            pass  # Drop block if buffer is full

    def _create_voice_frame(self, voice_frames) -> VoiceFrame:
        """
        Convert VoiceEngine output to synthesis VoiceFrame.

        Args:
            voice_frames: List of VoiceFrameABI from engine.step_all()

        Returns:
            VoiceFrame for UnifiedOutputStage
        """
        # Find first active voice
        active_frame = None
        for vf in voice_frames:
            if vf.is_active and vf.gain_q15 > 0:
                active_frame = vf
                break

        if active_frame is None:
            # No active voice - return silent frame
            return VoiceFrame(
                voice_id=0,
                note=60,
                cents=0,
                gain_q15=0,
                pan_q15=0,
                tree_a_id=0,
                tree_b_id=0,
                tree_blend_q15=0,
                is_active=False,
            )

        # Get VoiceState to read control parameters (eta, harmonics, gain)
        voice_state = self.engine.voices.get(active_frame.voice_id)

        # Use the current_note since VoiceEngine doesn't track note numbers
        # (It's gesture-based, not MIDI-centric)
        note = self.current_note

        # Read control values from VoiceState (set by Pico2SynthBridge)
        eta_q15 = voice_state.eta_q15 if voice_state else 24576
        master_gain_q15 = voice_state.master_gain_q15 if voice_state else 24576
        harmonics_q15 = voice_state.harmonics_q15 if voice_state else 0  # 0 = pure sine
        detune_cents = voice_state.detune_cents if voice_state else 0

        # Calculate f0_hz from note + detune
        cents = detune_cents
        f0_hz = 440.0 * (2.0 ** ((note - 69 + cents / 100.0) / 12.0))

        # Scale gain by master_gain control
        scaled_gain = (active_frame.gain_q15 * master_gain_q15) >> 15

        # Compute partial amplitudes from harmonics control
        # harmonics_q15 controls higher harmonic mix (0=fundamental only, 32767=full)
        base_amps = [32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096]
        partial_amps = [base_amps[0]]  # Fundamental always at full
        for amp in base_amps[1:]:
            partial_amps.append((amp * harmonics_q15) >> 15)

        return VoiceFrame(
            voice_id=active_frame.voice_id,
            note=note,
            cents=cents,
            f0_hz=f0_hz,
            gain_q15=scaled_gain,
            pan_q15=active_frame.pan_q15,
            tree_a_id=active_frame.tree_a_id,
            tree_b_id=active_frame.tree_b_id,
            tree_blend_q15=active_frame.tree_blend_q15,
            tilt_env_q15=active_frame.tilt_env_q15,
            density_env_q15=active_frame.density_env_q15,
            is_starting=active_frame.is_starting,
            is_stopping=active_frame.is_stopping,
            is_active=active_frame.is_active,
            beta_target_q15=eta_q15,
            partial_amps_q15=partial_amps,
        )

    # =========================================================================
    # NOTE CONTROL
    # =========================================================================

    def note_on(self, note: int = 60, velocity: int = 100):
        """Trigger a note."""
        self.current_note = note

        # Allocate voice
        self.current_voice_id = self.engine.allocate_voice(program_id=1001)

        if self.current_voice_id is not None:
            # Apply current pot values to newly allocated voice
            # (newly allocated voices get defaults, not current pot values)
            vid = self.current_voice_id
            if vid in self.engine.voices:
                vs = self.engine.voices[vid]
                vs.eta_q15 = self.bridge.osc.eta_q15
                vs.harmonics_q15 = self.bridge.osc.harmonics_q15
                vs.master_gain_q15 = self.bridge.osc.gain_q15
                # Convert detune_q15 to cents
                centered = self.bridge.osc.detune_q15 - 16384
                vs.detune_cents = (centered * 2400) // 16384

            # Create gesture
            pressure = (velocity * Q15_ONE) // 127
            gesture = GestureBasis(
                gate=True,
                pressure_q15=pressure,
            )
            self.engine.note_on(self.current_voice_id, gesture=gesture)

            # Tell the synthesis stage about the voice
            freq = 440.0 * (2 ** ((note - 69) / 12))
            self.stage.voice_on(self.current_voice_id, note)

            print(f"Note ON: {note} (voice {self.current_voice_id}, {freq:.1f}Hz)")

    def note_off(self):
        """Release the current note."""
        if self.current_voice_id is not None:
            self.engine.note_off(self.current_voice_id)
            self.stage.voice_off(self.current_voice_id)
            print(f"Note OFF: voice {self.current_voice_id}")
            self.current_voice_id = None

    def toggle_note(self):
        """Toggle note on/off."""
        if self.current_voice_id is not None:
            self.note_off()
        else:
            self.note_on(self.current_note)

    # =========================================================================
    # STATUS
    # =========================================================================

    def print_status(self):
        """Print current status."""
        print("-" * 60)

        # Hardware bridge status
        self.bridge.print_status()

        # Audio stats (no lock needed - these are just stats for display)
        print(f"  Audio: blocks={self.blocks_rendered} "
              f"underruns={self.buffer_underruns} "
              f"queue={self.audio_queue.qsize()}")

        # Voice status
        active = self.engine.active_voices()
        print(f"  Voices: active={len(active)} "
              f"current={self.current_voice_id}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pico2 Live Synthesis Controller")
    parser.add_argument("port", nargs="?", help="Serial port (auto-detect if omitted)")
    parser.add_argument("--note", type=int, default=60, help="Starting MIDI note (default: 60)")
    args = parser.parse_args()

    # Find port (Pico2 has dual CDC - control and data channels)
    if args.port:
        port = args.port
        print(f"Using specified port: {port}")
    else:
        control_port, data_port = find_pico2_ports()
        if not control_port:
            print("No Pico2 detected. Please specify port manually.")
            print(f"Usage: {sys.argv[0]} <PORT>")
            sys.exit(1)
        port = control_port
        print(f"Auto-detected control port: {control_port}")
        if data_port:
            print(f"Auto-detected data port: {data_port}")

    # Create live synth
    synth = Pico2LiveSynth(port=port)

    if not synth.start():
        sys.exit(1)

    print("\nPico2 Live Synthesis Controller")
    print("=" * 60)
    print("Hardware Controls:")
    print("  Pots 0-3: ADSR (Attack, Decay, Sustain, Release)")
    print("  Pots 4-5: Filter (Cutoff, Resonance)")
    print("\nKeyboard Commands:")
    print("  SPACE: Toggle note on/off")
    print("  UP/DOWN: Change note (+/- 1 semitone)")
    print("  s: Print status")
    print("  q: Quit")
    print("=" * 60)

    # Start with note on
    synth.note_on(args.note)

    # Setup terminal for Unix
    if sys.platform != 'win32':
        _setup_terminal()

    last_status_time = time.time()

    # Main loop
    try:
        while True:
            # Handle keyboard input
            key = get_key()
            if key:
                if key == ' ':
                    synth.toggle_note()
                elif key == 'UP':
                    synth.current_note = min(127, synth.current_note + 1)
                    if synth.current_voice_id is not None:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    print(f"Note: {synth.current_note}")
                elif key == 'DOWN':
                    synth.current_note = max(0, synth.current_note - 1)
                    if synth.current_voice_id is not None:
                        synth.note_off()
                        synth.note_on(synth.current_note)
                    print(f"Note: {synth.current_note}")
                elif key == 's':
                    synth.print_status()
                elif key == 'q':
                    print("\nQuitting...")
                    break

            # Print status periodically (every ~2 seconds)
            now = time.time()
            if now - last_status_time >= 2.0:
                last_status_time = now
                synth.print_status()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        # Restore terminal for Unix
        if sys.platform != 'win32':
            _restore_terminal()
        synth.stop()
        print("Done.")


if __name__ == "__main__":
    main()
