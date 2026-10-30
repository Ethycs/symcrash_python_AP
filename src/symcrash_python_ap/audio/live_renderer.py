"""Device-independent live block rendering through the canonical control path."""
from queue import SimpleQueue, Empty

import numpy as np

from ..cockpit_engine import CockpitEngine
from ..primitives import BLOCK_SIZE, Q15_ONE, Q15_SCALE
from ..safety.unified_renderer import UnifiedOutputStage
from ..voice import VoiceEngine


class LiveSynthRenderer:
    """Queue controls; render every voice through one final output stage.

    Rendering is single-owner. UI/control threads enqueue commands; only the
    render thread changes voice, envelope, VM and output DSP state.
    """
    def __init__(self, block_size=BLOCK_SIZE, max_voices=4):
        self.block_size = block_size
        self.engine = VoiceEngine(max_voices=max_voices)
        self.cockpit = CockpitEngine()
        self.stage = UnifiedOutputStage(block_size)
        self.commands = SimpleQueue()
        self.last_note = 60
        self.last_engine_output = None
        self._pending_mono = np.empty(0, dtype=np.float32)
        self._pending_offset = 0

    def note_on(self, note, velocity=100):
        if (isinstance(note, bool) or not isinstance(note, int) or not 0 <= note <= 127
                or isinstance(velocity, bool) or not isinstance(velocity, int) or not 0 <= velocity <= 127):
            raise ValueError('MIDI note and velocity must be integers in 0..127')
        self.commands.put(('on', note, velocity))

    def note_off(self, note=None):
        if note is not None and (isinstance(note, bool) or not isinstance(note, int) or not 0 <= note <= 127):
            raise ValueError('MIDI note must be an integer in 0..127')
        self.commands.put(('off', note))

    def set_pot(self, index, value_q15):
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10
                or isinstance(value_q15, bool) or not isinstance(value_q15, int)
                or not 0 <= value_q15 <= Q15_ONE):
            raise ValueError('Pot must be 0..9 with an unsigned Q15 integer value')
        self.commands.put(('pot', index, value_q15))

    def select_preset(self, preset):
        self.commands.put(('preset', preset))

    def _render(self, mono):
        while True:
            try:
                command = self.commands.get_nowait()
            except Empty:
                break
            if command[0] == 'on':
                self.last_note = command[1]
                self.cockpit.midi_note_on(command[1], command[2])
            elif command[0] == 'off':
                self.cockpit.midi_note_off(self.last_note if command[1] is None else command[1])
            elif command[0] == 'pot':
                self.cockpit.set_pot(command[1], command[2])
            elif command[0] == 'preset':
                self.stage.configure_preset(command[1])
                self.engine.select_compiled_preset(command[1])
        control = self.cockpit.tick(voice_ids=range(self.engine.max_voices))
        output = self.engine.step_block(control)
        self.last_engine_output = output
        if self.engine.selected_preset is not None:
            self.stage.configure_preset(self.engine.selected_preset)
        return self.stage.render_frames([item.frame for item in output.render_batch],
                                        output.param_frame, mono=mono)

    def render_block(self):
        """One complete signed-Q15 stereo output block, including effect tails."""
        return self._render(False)

    def render_mono_block(self):
        """One mono signed-Q15 block, with downmix before final Safety."""
        return self._render(True)[0]

    def render_mono_float(self):
        """Device normalization is a power-of-two Q15 conversion."""
        return np.asarray(self.render_mono_block(), dtype=np.float32) / Q15_SCALE

    def fill_mono(self, destination):
        """Fill any callback length without dropping partial blocks or inserting gaps."""
        position = 0
        while position < len(destination):
            if self._pending_offset == len(self._pending_mono):
                self._pending_mono = self.render_mono_float()
                self._pending_offset = 0
            count = min(len(destination) - position, len(self._pending_mono) - self._pending_offset)
            destination[position:position+count] = self._pending_mono[self._pending_offset:self._pending_offset+count]
            self._pending_offset += count
            position += count
