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
from multiprocessing import shared_memory, Process, Queue
from typing import Optional
import time

# Import canonical constants from primitives (single source of truth)
from ..primitives import BLOCK_SIZE, SAMPLE_RATE_HZ

# Constants
SAMPLE_RATE = SAMPLE_RATE_HZ  # Alias for backward compatibility
RING_BUFFER_BLOCKS = 128  # ~170ms buffer
PRE_FILL_BLOCKS = 32      # Pre-fill with silence (matches pico2_live_synth)


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


def render_process_main(ring_buffer_name, control_queue, block_size, num_blocks):
    """Render complete polyphonic output blocks into the shared device queue."""
    from queue import Empty
    from .live_renderer import LiveSynthRenderer

    ring = SharedRingBuffer(ring_buffer_name, block_size, num_blocks, create=False)
    renderer = LiveSynthRenderer(block_size=block_size)
    running = True
    print("[RenderProcess] Started")
    try:
        while running:
            while True:
                try:
                    message = control_queue.get_nowait()
                except Empty:
                    break
                kind = message[0]
                if kind == MSG_STOP:
                    running = False
                    break
                if kind == MSG_MACRO:
                    renderer.set_pot(message[1], message[2])
                elif kind == MSG_NOTE_ON:
                    renderer.note_on(message[1], message[2] if len(message) > 2 else 100)
                elif kind == MSG_NOTE_OFF:
                    renderer.note_off(message[1] if len(message) > 1 else None)
                else:
                    raise ValueError(f'Unknown render control message: {kind}')
            if not running:
                break
            if ring.available_write() > 0:
                # Render only with room in the queue. Do not advance DSP and
                # drop its output while the consumer is behind.
                ring.write(renderer.render_mono_float())
            else:
                time.sleep(0.0001)
    finally:
        ring.close()
        print("[RenderProcess] Stopped")





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

    def note_on(self, note: int, velocity: int = 100):
        """Send note on to render process."""
        if self.control_queue:
            self.control_queue.put((MSG_NOTE_ON, note, velocity))

    def note_off(self, note=None):
        """Send note off to render process."""
        if self.control_queue:
            self.control_queue.put((MSG_NOTE_OFF,) if note is None else (MSG_NOTE_OFF, note))

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
