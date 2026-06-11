#!/usr/bin/env python3
"""
Pico2 Live Synthesis Controller
===============================

Complete live control: Pico2 Hardware -> CockpitEngine -> VoiceFrameEmitter -> UnifiedOutputStage -> Audio

This script connects all the pieces for real-time hardware control of synthesis:
- Pico2 pots control all synthesis parameters via CockpitEngine
- CockpitEngine emits ControlFrames (block-rate control snapshots)
- VoiceFrameEmitter consumes ControlFrames and emits VoiceFrames
- UnifiedOutputStage renders actual audio (Carrier + Wavelet Lift)
- sounddevice streams audio in real-time

Controls:
- Pots 0-3: ADSR envelope (Attack, Decay, Sustain, Release)
- Pots 4-5: Filter (Cutoff=CHAOS, Resonance=TEXTURE)
- Pot 6: Harmonics (RESERVED_7 - future expansion)
- Pot 7: Eta (dry/wet blend = MORPH)
- Pot 8: Detune (pitch offset)
- Pot 9: Master Gain
- Keyboard: SPACE=toggle note, UP/DOWN=change note, s=status, q=quit

Usage:
    python pico2_live_synth.py [PORT]
    python pico2_live_synth.py COM3
    python pico2_live_synth.py --no-hardware  # Virtual mode (no Pico2)
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

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.library.preset_compiler import PresetCompiler
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
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

    Connects Pico2 Hardware -> CockpitEngine -> VoiceFrameEmitter -> UnifiedOutputStage -> Audio
    Or in virtual mode: ImGui Controls -> CockpitEngine -> VoiceFrameEmitter -> UnifiedOutputStage -> Audio
    """

    def __init__(self, port: Optional[str] = None, virtual_mode: bool = False, preset_path: Optional[str] = None):
        """
        Initialize the live synth.

        Args:
            port: Serial port (auto-detect if None, ignored in virtual_mode)
            virtual_mode: If True, run without hardware (controls via ImGui)
            preset_path: Path to preset file (use default if None)
        """
        self.virtual_mode = virtual_mode

        # Cockpit engine for block-rate control
        self.cockpit = CockpitEngine()

        # Load preset for voice frame emission
        if preset_path:
            compiler = PresetCompiler()
            preset = compiler.compile_file(preset_path)
        else:
            # Use a simple default preset (create in-memory)
            from symcrash_python_ap.library.preset_compiler import MaterialConfig, LatchDefaults
            from symcrash_python_ap.envelope_scheduler import ADSRSpec
            from dataclasses import dataclass, field as dc_field

            @dataclass
            class _MinimalPreset:
                """Minimal preset for live synth."""
                preset_id: str = "default"
                preset_version: str = "1.0"
                author: str = "System"
                description: str = "Default live preset"
                voiceframe_std: str = "WSE-STD-020-v1"
                vm_std: str = "WSE-STD-VM-HYBRID-v1"
                operator_lib_id: str = "WSE-STD-030-v2"
                latch: LatchDefaults = dc_field(default_factory=lambda: LatchDefaults(
                    tree_a_id=0, tree_b_id=0, tree_blend_q15=0
                ))
                amp_env: ADSRSpec = dc_field(default_factory=lambda: ADSRSpec(
                    attack_ms=10.0, decay_ms=42.0, sustain=0.75, release_ms=85.0
                ))
                material: MaterialConfig = dc_field(default_factory=lambda: MaterialConfig(
                    beta_start_q15=0, beta_target_q15=24576, beta_ramp_blocks=16
                ))
                backbone: list = dc_field(default_factory=list)
                harmonic_amps_q15: list = dc_field(default_factory=lambda: [32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096])

                def backbone_ops(self):
                    return []

                def backbone_families(self):
                    # CompiledPreset contract: dict of family/op name ->
                    # strength (0.0-1.0). VoiceEngine.step_voice_simple()
                    # calls .values() on this. Default family weights for a
                    # basic sound (scaled by TIMBRE/TEXTURE/CHAOS modifiers).
                    return {
                        'color': 1.0,  # TIMBRE
                        'local': 1.0,  # TEXTURE
                    }

            preset = _MinimalPreset()

        # Voice frame emitter
        self.emitter = VoiceFrameEmitter(preset, voice_id=0)

        # Synthesis pipeline for audio rendering
        self.stage = UnifiedOutputStage(
            block_size=BLOCK_SIZE,
        )

        # Hardware bridge (or virtual mode bridge)
        self.bridge = create_bridge(self.cockpit, port=port, virtual_mode=virtual_mode)

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
        print("[Render] Thread started")
        while self.running:
            try:
                # Check if we need to render more blocks
                if self.audio_queue.qsize() < AUDIO_BUFFER_BLOCKS:
                    self._render_block()
                else:
                    # Buffer is full, wait a bit
                    time.sleep(0.0001)
            except Exception as e:
                print(f"[Render] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.001)
        print("[Render] Thread exiting")

    def _render_block(self):
        """Render one audio block."""
        # Tick cockpit to get ControlFrame
        control_frame = self.cockpit.tick()

        # Consume control frame in emitter
        self.emitter.consume_control_frame(control_frame)

        # Always emit VoiceFrame (emitter handles inactive state with gain=0)
        # This ensures all required fields (ops_enable, families, partial_amps_q15) are populated
        frame = self.emitter.emit_frame()

        # Debug: log first few blocks and then occasionally
        if self.blocks_rendered < 5 or self.blocks_rendered % 3000 == 0:
            print(f"[DEBUG] Block {self.blocks_rendered}: "
                  f"active={self.emitter.is_active} "
                  f"note={frame.note} gain={frame.gain_q15} "
                  f"beta={frame.beta_target_q15}")

        # Render through the synthesis pipeline
        left, right = self.stage.render_block(frame)

        # Queue for audio output (no lock needed - blocks_rendered is only for stats)
        try:
            self.audio_queue.put_nowait((left, right))
            self.blocks_rendered += 1
        except queue.Full:
            pass  # Drop block if buffer is full


    # =========================================================================
    # NOTE CONTROL
    # =========================================================================

    def note_on(self, note: int = 60, velocity: int = 100):
        """Trigger a note."""
        self.current_note = note

        # Send note-on via cockpit (will be in next ControlFrame)
        self.cockpit.midi_note_on(note, velocity)

        # Tell the emitter to start the note (this activates the voice)
        velocity_q15 = int((velocity / 127) * Q15_ONE)
        self.emitter.note_on(note, velocity_q15)

        # Tell the synthesis stage about the voice
        freq = 440.0 * (2 ** ((note - 69) / 12))
        self.stage.voice_on(0, note)  # Single voice (voice_id=0)

        print(f"Note ON: {note} ({freq:.1f}Hz, velocity={velocity})")

    def note_off(self):
        """Release the current note."""
        # Send note-off via cockpit
        self.cockpit.midi_note_off(self.current_note)

        # Tell the emitter to release the note
        self.emitter.note_off()

        # Tell the synthesis stage
        self.stage.voice_off(0)  # Single voice (voice_id=0)

        print(f"Note OFF: {self.current_note}")

    def toggle_note(self):
        """Toggle note on/off."""
        if self.emitter.is_active:
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
        print(f"  Voice: active={self.emitter.is_active} "
              f"note={self.current_note}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pico2 Live Synthesis Controller")
    parser.add_argument("port", nargs="?", help="Serial port (auto-detect if omitted)")
    parser.add_argument("--note", type=int, default=60, help="Starting MIDI note (default: 60)")
    parser.add_argument("--no-hardware", action="store_true",
                        help="Run in virtual mode without hardware (use --imgui for controls)")
    parser.add_argument("--imgui", action="store_true",
                        help="Show ImGui control window for knobs/sliders")
    args = parser.parse_args()

    # Determine virtual mode
    virtual_mode = args.no_hardware

    # Find port (Pico2 has dual CDC - control and data channels)
    port = None
    if not virtual_mode:
        if args.port:
            port = args.port
            print(f"Using specified port: {port}")
        else:
            control_port, data_port = find_pico2_ports()
            if not control_port:
                print("No Pico2 detected. Please specify port manually or use --no-hardware.")
                print(f"Usage: {sys.argv[0]} <PORT>")
                print(f"       {sys.argv[0]} --no-hardware --imgui")
                sys.exit(1)
            port = control_port
            print(f"Auto-detected control port: {control_port}")
            if data_port:
                print(f"Auto-detected data port: {data_port}")
    else:
        print("Running in virtual mode (no hardware)")

    # Create live synth
    synth = Pico2LiveSynth(port=port, virtual_mode=virtual_mode)

    if not synth.start():
        sys.exit(1)

    # Setup ImGui controls window if requested
    imgui_thread = None
    if args.imgui:
        try:
            from imgui_bundle import hello_imgui
            from symcrash_imgui_controls.synth_bridge import ImGuiSynthBridge
            from symcrash_imgui_controls.app import SymcrashImGuiApp

            # Create ImGui bridge connected to our synth's bridge
            imgui_bridge = ImGuiSynthBridge(synth.bridge)

            # Start ImGui app in background thread
            def run_imgui():
                app = SymcrashImGuiApp()
                app.bridge = imgui_bridge

                # Configure runner
                runner_params = hello_imgui.RunnerParams()
                runner_params.app_window_params.window_title = "Symcrash Sound Design Workbench (Live)"
                runner_params.app_window_params.window_geometry.size = (1050, 650)
                runner_params.callbacks.show_gui = app.render

                # Run
                hello_imgui.run(runner_params)

            imgui_thread = threading.Thread(target=run_imgui, daemon=True)
            imgui_thread.start()
            print("[ImGui] Control window started")
        except ImportError as e:
            print(f"[ImGui] Could not start ImGui controls: {e}")
            print("[ImGui] Install imgui-bundle: pip install imgui-bundle")

    mode_str = "Virtual Mode" if virtual_mode else "Hardware Mode"
    print(f"\nPico2 Live Synthesis Controller ({mode_str})")
    print("=" * 60)
    if not virtual_mode:
        print("Hardware Controls:")
        print("  Pots 0-3: ADSR (Attack, Decay, Sustain, Release)")
        print("  Pots 4-5: Filter (Cutoff, Resonance)")
    if args.imgui:
        print("ImGui Controls: Window with knobs for all parameters")
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
