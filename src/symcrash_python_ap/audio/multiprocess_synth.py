"""
Multiprocess Synth Engine
=========================

Runs synthesis in a separate process to eliminate GIL contention.
Uses shared memory ring buffer for zero-copy audio transfer.

Architecture:
┌─────────────────────┐     shared memory      ┌─────────────────────┐
│   Render Process    │ ──────────────────────>│    Main Process     │
│  (VoiceEngine +     │    ring buffer         │  (ImGui + Audio     │
│   UnifiedOutput)    │                        │   callback)         │
└─────────────────────┘                        └─────────────────────┘
          ▲                                              │
          │              control queue                   │
          └──────────────────────────────────────────────┘
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory, Process, Queue
from typing import Optional
import time

# Constants
BLOCK_SIZE = 64
SAMPLE_RATE = 48000
RING_BUFFER_BLOCKS = 128  # ~170ms buffer
PRE_FILL_BLOCKS = 32      # Pre-fill with silence (matches pico2_live_synth)
Q15_ONE = 32767


class SharedRingBuffer:
    """
    Lock-free single-producer single-consumer ring buffer in shared memory.

    Producer (render process) writes blocks.
    Consumer (audio callback) reads blocks.
    """

    def __init__(self, name: str, block_size: int, num_blocks: int, create: bool = False):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.name = name

        # Layout: [write_idx (4 bytes), read_idx (4 bytes), data (num_blocks * block_size * 4 bytes)]
        self.header_size = 8  # 2 x int32
        self.data_size = num_blocks * block_size * 4  # float32
        self.total_size = self.header_size + self.data_size

        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
            # Initialize indices to 0
            np.frombuffer(self.shm.buf[:8], dtype=np.int32)[:] = [0, 0]
        else:
            self.shm = shared_memory.SharedMemory(name=name)

        # Views into shared memory
        self._indices = np.ndarray((2,), dtype=np.int32, buffer=self.shm.buf[:8])
        self._data = np.ndarray((num_blocks, block_size), dtype=np.float32,
                                buffer=self.shm.buf[8:])

    @property
    def write_idx(self) -> int:
        return self._indices[0]

    @write_idx.setter
    def write_idx(self, val: int):
        self._indices[0] = val

    @property
    def read_idx(self) -> int:
        return self._indices[1]

    @read_idx.setter
    def read_idx(self, val: int):
        self._indices[1] = val

    def available_write(self) -> int:
        """Number of blocks available for writing."""
        return self.num_blocks - 1 - ((self.write_idx - self.read_idx) % self.num_blocks)

    def available_read(self) -> int:
        """Number of blocks available for reading."""
        return (self.write_idx - self.read_idx) % self.num_blocks

    def write(self, block: np.ndarray) -> bool:
        """Write a block. Returns False if buffer full."""
        if self.available_write() == 0:
            return False

        idx = self.write_idx % self.num_blocks
        self._data[idx, :] = block
        self.write_idx = (self.write_idx + 1) % (self.num_blocks * 2)
        return True

    def read(self) -> Optional[np.ndarray]:
        """Read a block. Returns None if buffer empty."""
        if self.available_read() == 0:
            return None

        idx = self.read_idx % self.num_blocks
        block = self._data[idx, :].copy()
        self.read_idx = (self.read_idx + 1) % (self.num_blocks * 2)
        return block

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


# Control message types
MSG_MACRO = 1
MSG_NOTE_ON = 2
MSG_NOTE_OFF = 3
MSG_STOP = 4


def render_process_main(
    ring_buffer_name: str,
    control_queue: Queue,
    block_size: int,
    num_blocks: int,
):
    """
    Main function for the render process.

    Runs the synthesis pipeline and writes audio to shared memory.
    """
    # Import here to avoid importing in main process
    from symcrash_python_ap.voice import VoiceEngine, GestureBasis
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage

    # Connect to shared ring buffer
    ring = SharedRingBuffer(ring_buffer_name, block_size, num_blocks, create=False)

    # Create synth components
    engine = VoiceEngine(max_voices=4)
    stage = UnifiedOutputStage(block_size=block_size)

    # State
    current_note = 60
    current_voice_id = None
    running = True
    silence = np.zeros(block_size, dtype=np.float32)

    # Macro state (mirrors Pico2SynthBridge behavior)
    macro_values = [0] * 10

    print("[RenderProcess] Started")

    while running:
        # Process control messages (non-blocking)
        while not control_queue.empty():
            try:
                msg = control_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == MSG_STOP:
                    running = False
                    break
                elif msg_type == MSG_MACRO:
                    macro_id, value_q15 = msg[1], msg[2]
                    macro_values[macro_id] = value_q15
                    # Update engine based on macro
                    _apply_macro(engine, macro_id, value_q15)
                elif msg_type == MSG_NOTE_ON:
                    note = msg[1]
                    current_note = note
                    current_voice_id = engine.allocate_voice(program_id=1001)
                    if current_voice_id is not None:
                        gesture = GestureBasis(gate=True, pressure_q15=24576)
                        engine.note_on(current_voice_id, gesture=gesture)
                elif msg_type == MSG_NOTE_OFF:
                    if current_voice_id is not None:
                        engine.note_off(current_voice_id)
                        current_voice_id = None
            except:
                break

        if not running:
            break

        # Render if buffer has space
        if ring.available_write() > 0:
            voice_frames = engine.step_all()

            # Find active voice
            active_frame = None
            for vf in voice_frames:
                if vf.is_active and vf.gain_q15 > 0:
                    active_frame = vf
                    break

            if active_frame is not None:
                voice_state = engine.voices.get(active_frame.voice_id)

                detune_cents = voice_state.detune_cents if voice_state else 0
                f0_hz = 440.0 * (2.0 ** ((current_note - 69 + detune_cents / 100.0) / 12.0))
                eta_q15 = voice_state.eta_q15 if voice_state else 32767
                master_gain = voice_state.master_gain_q15 if voice_state else 32767
                # Q15 multiply with rounding: (a * b + 16384) >> 15
                scaled_gain = (active_frame.gain_q15 * master_gain + 16384) >> 15

                harmonics_q15 = voice_state.harmonics_q15 if voice_state else 0  # 0 = pure sine
                base_amps = [32767, 16384, 10923, 8192, 6554, 5461, 4681, 4096]
                partial_amps = [base_amps[0]]
                for amp in base_amps[1:]:
                    # Q15 multiply with rounding
                    partial_amps.append((amp * harmonics_q15 + 16384) >> 15)

                frame = VoiceFrame(
                    voice_id=active_frame.voice_id,
                    note=current_note,
                    cents=detune_cents,
                    f0_hz=f0_hz,
                    gain_q15=scaled_gain,
                    beta_target_q15=eta_q15,
                    partial_amps_q15=partial_amps,
                    is_active=True,
                )

                samples = stage.render_voice(frame)
                block = np.array(samples, dtype=np.float32) / Q15_ONE
            else:
                block = silence

            ring.write(block)
        else:
            # Buffer full, wait a bit
            time.sleep(0.0001)

    ring.close()
    print("[RenderProcess] Stopped")


def _apply_macro(engine, macro_id: int, value_q15: int):
    """Apply macro value to engine (mirrors Pico2SynthBridge)."""
    # ADSR macros
    if macro_id == 0:  # Attack
        blocks = max(1, (value_q15 * 250) // 32767)
        engine.default_program.attack_blocks = blocks
    elif macro_id == 1:  # Decay
        blocks = max(1, (value_q15 * 500) // 32767)
        engine.default_program.decay_blocks = blocks
    elif macro_id == 2:  # Sustain
        engine.default_program.sustain_level_q15 = value_q15
    elif macro_id == 3:  # Release
        blocks = max(1, (value_q15 * 750) // 32767)
        engine.default_program.release_blocks = blocks

    # Apply to active voices
    for vid in engine.active_voices():
        if vid in engine.voices:
            vs = engine.voices[vid]
            if macro_id == 0:
                vs.program.attack_blocks = engine.default_program.attack_blocks
            elif macro_id == 1:
                vs.program.decay_blocks = engine.default_program.decay_blocks
            elif macro_id == 2:
                vs.program.sustain_level_q15 = value_q15
            elif macro_id == 3:
                vs.program.release_blocks = engine.default_program.release_blocks
            elif macro_id == 6:  # Harmonics
                vs.harmonics_q15 = value_q15
            elif macro_id == 7:  # Eta
                vs.eta_q15 = value_q15
            elif macro_id == 8:  # Detune
                centered = value_q15 - 16384
                vs.detune_cents = (centered * 2400) // 16384
            elif macro_id == 9:  # Gain
                vs.master_gain_q15 = value_q15


class MultiprocessSynth:
    """
    Synth that runs rendering in a separate process.

    Use this when GUI + audio causes GIL contention issues.
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.ring_buffer_name = f"symcrash_audio_{id(self)}"

        self.ring: Optional[SharedRingBuffer] = None
        self.control_queue: Optional[Queue] = None
        self.render_proc: Optional[Process] = None
        self.running = False

    def start(self) -> bool:
        """Start the render process."""
        try:
            # Create shared ring buffer
            self.ring = SharedRingBuffer(
                self.ring_buffer_name,
                self.block_size,
                RING_BUFFER_BLOCKS,
                create=True
            )

            # Pre-fill buffer with silence (matches pico2_live_synth behavior)
            silence = np.zeros(self.block_size, dtype=np.float32)
            for _ in range(PRE_FILL_BLOCKS):
                self.ring.write(silence)

            # Create control queue
            self.control_queue = Queue()

            # Start render process
            self.render_proc = Process(
                target=render_process_main,
                args=(
                    self.ring_buffer_name,
                    self.control_queue,
                    self.block_size,
                    RING_BUFFER_BLOCKS,
                ),
                daemon=True
            )
            self.render_proc.start()

            self.running = True
            print(f"[MultiprocessSynth] Started (ring buffer: {self.ring_buffer_name})")
            return True

        except Exception as e:
            print(f"[MultiprocessSynth] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the render process."""
        if self.control_queue:
            self.control_queue.put((MSG_STOP,))

        if self.render_proc:
            self.render_proc.join(timeout=1.0)
            if self.render_proc.is_alive():
                self.render_proc.terminate()

        if self.ring:
            self.ring.close()
            try:
                self.ring.unlink()
            except:
                pass

        self.running = False
        print("[MultiprocessSynth] Stopped")

    def send_macro(self, macro_id: int, value_q15: int):
        """Send macro change to render process."""
        if self.control_queue:
            self.control_queue.put((MSG_MACRO, macro_id, value_q15))

    def note_on(self, note: int):
        """Send note on to render process."""
        if self.control_queue:
            self.control_queue.put((MSG_NOTE_ON, note))

    def note_off(self):
        """Send note off to render process."""
        if self.control_queue:
            self.control_queue.put((MSG_NOTE_OFF,))

    def read_block(self) -> Optional[np.ndarray]:
        """Read a block from the ring buffer (for audio callback)."""
        if self.ring:
            return self.ring.read()
        return None

    def available_blocks(self) -> int:
        """Number of blocks available to read."""
        if self.ring:
            return self.ring.available_read()
        return 0
