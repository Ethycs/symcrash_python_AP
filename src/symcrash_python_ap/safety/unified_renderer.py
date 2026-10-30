"""VM voice rendering followed by the shared stereo or mono output stage.

Each output block renders all supplied voices, then runs pan/mix, Material,
auxiliary returns, master gain and Safety once, including during silence.
The default output profile matches SynthMaster's CCS settings. Legacy carrier,
wavelet and AntiClick objects remain available for compatibility; the VM owns
actual voice synthesis and StereoOutputMixin owns final output processing.
"""

from typing import List, Dict, Tuple

from ..primitives import Q15_ONE, Q15_HALF, BLOCK_SIZE


# =============================================================================
# IMPORTS FROM CANONICAL LOCATIONS
# =============================================================================

# Backends live in wavelet/backends.py
from ..wavelet.backends import (
    SynthesisBackend,
    HarmonicScaffoldBackend,
    WaveletBackend,
)
from ..primitives import Q15_3_4

# VM Executor (WSE-STD-VM-HYBRID v0)
from ..vm import vm_step_block, VMState, Tables
from ..vm.scheduling import ExecutionReport
from .output_stage import StereoOutputMixin


# =============================================================================
# UNIFIED OUTPUT STAGE
# =============================================================================
# (The deprecated RenderFrame dataclass and UnifiedRenderer stub were
# removed 2026-07-02; frames.VoiceFrame and UnifiedOutputStage are the
# single owners.)

class UnifiedOutputStage(StereoOutputMixin):
    """
    Render complete output blocks using the same output DSP as SynthMaster.

    Use render_frames for polyphony and render_mono_frames for a mono device.
    Native lifecycle flags and accepted wire packets own VM note resets.
    Explicit voice_on/voice_off methods retain the legacy allocation API;
    native emitters provide their own envelope gain and release frames.

    render_voice is a raw mono VM diagnostic/composition API. It does not
    process the final output or advance the shared output/tuning clock.

    Usage:
        stage = UnifiedOutputStage()
        stage.voice_on(0, note=60)

        for block in range(num_blocks):
            frame = VoiceFrame(voice_id=0, note=60, f0_hz=261.6, ...)
            L, R = stage.render_block(frame)

        stage.voice_off(0)
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        from ..frames import ParamFrame, VoiceFrame
        from ..wavelet.carrier import CarrierOscillator
        from ..wavelet.carrier_lift import (
            CarrierLiftVoiceState, NUM_LEAVES
        )
        from ..wavelet.leaf_generator import LeafGenerator
        from ..tuning import TuningMorpher, TUNING_HARMONIC
        from .anti_click import AntiClickOperator, AntiClickConfig, AntiClickState

        # Store imports for use
        self.VoiceFrame = VoiceFrame
        self.NUM_LEAVES = NUM_LEAVES
        self.AntiClickState = AntiClickState

        # Core components
        self.carrier = CarrierOscillator()
        self.leaf_gen = LeafGenerator()
        from ..vm.resources import prepare_oscillator_resources
        prepare_oscillator_resources()
        self.tuning = TuningMorpher()
        self._init_output_stage(block_size)
        self.params = ParamFrame()
        self.block_size = block_size
        self.last_execution_reports = {}

        # Wavelet backend for IDWPT lift
        self.wavelet_backend = WaveletBackend(block_size=block_size)

        # Retained for legacy callers; the VM and shared Safety process audio.
        self.anticlick = AntiClickOperator(AntiClickConfig(
            block_size=block_size,
            sample_rate=48000,
            wet_latency_blocks=5,
            bloom_time_s=0.05,
            ola_K=16,
            equal_power=True
        ))

        # Per-voice state
        self.voice_states: Dict[int, CarrierLiftVoiceState] = {}
        self.ac_states: Dict[int, AntiClickState] = {}

        # VM executor state (WSE-STD-VM-HYBRID v0)
        self.vm_states: Dict[int, VMState] = {}
        self.vm_tables = Tables()

        # Initialize with harmonic tuning
        self.tuning.set_tuning_immediate(TUNING_HARMONIC)

    def voice_on(self, voice_id: int, note: int = 60, cents: int = 0,
                 velocity_q15: int = Q15_ONE, external_adsr: bool = True) -> None:
        """
        Allocate and start a voice.

        Args:
            voice_id: Voice identifier
            note: MIDI note number
            cents: Fine tuning in cents
            velocity_q15: Note velocity (Q15)
            external_adsr: If True, caller handles ADSR via gain_q15 in VoiceFrame.
        """
        from ..wavelet.carrier_lift import CarrierLiftVoiceState

        # Create voice state
        state = CarrierLiftVoiceState(voice_id=voice_id)
        state.note_on(note, cents, velocity_q15)
        self.voice_states[voice_id] = state

        # Initialize AntiClick state
        self.ac_states[voice_id] = self.anticlick.init_state()
        self.anticlick.note_on(self.ac_states[voice_id])

        # Configure carrier
        self.carrier.configure_voice(voice_id, note, cents)

        # If external ADSR, set carrier envelope to instant
        if external_adsr:
            self.carrier.set_envelope_instant(voice_id)

        self.carrier.gate_on(voice_id)

        # Allocate wavelet voice AND reset wet state to prevent drones
        self.wavelet_backend.engine.voice_allocate(voice_id)
        self.wavelet_backend.engine.reset_wet_state(voice_id)
        self.wavelet_backend._allocated_voices.add(voice_id)

        # Safety
        self.safety.voice_on(voice_id)

        # Initialize VM state (WSE-STD-VM-HYBRID v0)
        vm_state = VMState.create(voice_id)
        vm_state.reset_for_note_on(note=note, cents=cents)
        self.vm_states[voice_id] = vm_state

    def voice_off(self, voice_id: int) -> None:
        """Signal voice release."""
        if voice_id in self.voice_states:
            self.voice_states[voice_id].note_off()
            self.carrier.gate_off(voice_id)

    def voice_deallocate(self, voice_id: int) -> None:
        """Fully deallocate a voice slot."""
        self.voice_states.pop(voice_id, None)
        self.ac_states.pop(voice_id, None)
        self.carrier.remove_state(voice_id)
        self.safety.voice_off(voice_id)
        # Clean up VM state
        self.vm_states.pop(voice_id, None)

    def set_tuning(self, tuning_name: str, morph_time_sec: float = 2.0) -> None:
        """
        Change global instrument tuning.

        Args:
            tuning_name: Tuning preset name (e.g., "harmonic", "bright")
            morph_time_sec: Transition time (1-5 seconds typical)
        """
        self.tuning.set_target_by_name(tuning_name, morph_time_sec)

    def render_voice(self, frame: 'VoiceFrame') -> List[int]:
        """
        Render raw mono VM audio for analysis/composition, without output DSP.

        Device playback must use render_frames/render_block/render_mono_frames.
        This method does not advance the shared output/tuning clock.

        Args:
            frame: VoiceFrame containing all voice parameters

        Returns:
            List of Q15 audio samples
        """
        voice_id = frame.voice_id
        report = ExecutionReport()
        self.last_execution_reports[voice_id] = report
        if frame.wire_data is not None and frame.wire_data.reject_code:
            return [0] * self.block_size

        # Ensure voice is allocated
        if voice_id not in self.voice_states:
            self.voice_on(voice_id, frame.note, frame.cents)

        # =====================================================================
        # VM EXECUTOR PATH (WSE-STD-VM-HYBRID v0)
        # =====================================================================
        # Get or create VM state
        if voice_id not in self.vm_states:
            vm_state = VMState.create(voice_id)
            vm_state.reset_for_note_on(note=frame.note, cents=frame.cents)
            self.vm_states[voice_id] = vm_state

        # Execute one block through VM
        vm_state = self.vm_states[voice_id]
        if frame.is_starting and frame.wire_data is None:
            vm_state.reset_for_note_on(note=frame.note, cents=frame.cents)
        vm_state, output_block = vm_step_block(vm_state, frame, self.vm_tables, report=report)
        self.vm_states[voice_id] = vm_state

        # VM output is int16 numpy array, convert to list
        output = output_block.tolist()

        # Mark first block done
        state = self.voice_states[voice_id]
        if state.is_starting:
            state.after_first_block()

        return output

    def render_frames(self, frames, params=None, *, mono=False):
        """Render all supplied voices and process their mix once per block.

        An empty collection still advances output effects and Safety. Rejected
        or stale packets cannot enter the mix or replay global consumer events.
        """
        frames = list(frames)
        if any(not isinstance(frame, self.VoiceFrame) for frame in frames):
            raise TypeError('Rendering requires VoiceFrame instances')
        ids = [frame.voice_id for frame in frames]
        if (any(isinstance(vid, bool) or not isinstance(vid, int) or vid < 0 for vid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Frames must address unique nonnegative voice IDs')
        params = self.params if params is None else params
        from ..frames import ParamFrame
        if not isinstance(params, ParamFrame):
            raise TypeError('Output parameters must be a ParamFrame')
        self.last_execution_reports = {}
        rendered, events = [], []
        for frame in sorted(frames, key=lambda f: f.voice_id):
            samples = self.render_voice(frame)
            report = self.last_execution_reports[frame.voice_id]
            if not report.serviced:
                continue
            rendered.append((frame, samples))
            events.extend((frame.voice_id, event) for event in report.consumer_events)
        output = self._mix_output(rendered, params, events, mono=mono)
        self.tuning.advance_block()
        return output

    def render_block(self, frame: 'VoiceFrame', params=None) -> Tuple[List[int], List[int]]:
        """Single-voice compatibility wrapper around the complete stereo output."""
        return self.render_frames([frame], params)

    def render_mono_frames(self, frames, params=None):
        """Render a mono device block; downmix before the final Safety stage."""
        return self.render_frames(frames, params, mono=True)[0]

    def render_block_simple(
        self,
        voice_id: int,
        note: int,
        gain_q15: int = Q15_HALF,
        beta_target_q15: int = Q15_3_4,
    ) -> Tuple[List[int], List[int]]:
        """
        Simplified render for common use case.
        """
        frame = self.VoiceFrame(
            voice_id=voice_id,
            note=note,
            gain_q15=gain_q15,
            beta_target_q15=beta_target_q15,
            f0_hz=440.0 * (2.0 ** ((note - 69) / 12.0)),
        )
        return self.render_block(frame)


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# CarrierLiftPipeline is now UnifiedOutputStage
CarrierLiftPipeline = UnifiedOutputStage


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class (v2 architecture)
    'UnifiedOutputStage',
    # Backwards compatibility alias
    'CarrierLiftPipeline',
    # Backend protocol and adapters (re-exported from wavelet.backends)
    'SynthesisBackend',
    'HarmonicScaffoldBackend',
    'WaveletBackend',
]
