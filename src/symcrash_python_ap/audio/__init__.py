"""Audio subsystem for symcrash."""

from .multiprocess_synth import MultiprocessSynth, SharedRingBuffer

__all__ = ["MultiprocessSynth", "SharedRingBuffer"]
