"""
SynthMaster - Unified Synthesis Pipeline
=========================================

High-level interface for the complete synthesis pipeline.
All voice management goes through VoiceEngine (CMMM architecture).

Features:
- Single unified voice path through VoiceEngine
- Toggle pipeline stages on/off for debugging
- Preset loading from PresetLibrary
- Voice lifecycle management via VoiceEngine
- Health telemetry via CockpitEngine
- Standardized audio output via AudioWriter

Usage:
    from symcrash_python_ap import SynthMaster

    synth = SynthMaster()
    synth.load_preset("brass-stab-v0")
    synth.note_on(60, velocity=100)
    audio = synth.render(duration_sec=2.0)
    synth.save_audio("brass_test")

Toggle stages for debugging:
    synth.disable("material")  # No effects
    synth.disable("safety")    # No limiting
    synth.set_osc_mode("mode_b")  # Fallback for voices without a compiled preset
"""

from dataclasses import dataclass, field
from copy import copy, deepcopy
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .audio_io import AudioWriter, AudioMetadata
from .cockpit_engine import CockpitEngine
from .control_registry import ControlID
from .library.preset_library import PresetLibraryV0, Query
from .library.capsule import CapsuleKind
from .symbol.executor_scheduling import HealthTelemetry
from .primitives import BLOCK_SIZE
from .voice.voice_engine import VoiceEngine, VoiceEngineOutput, RenderBatchItem
from .safety.output_stage import StereoOutputMixin
from .frames import ParamFrame, VoiceFrame
from .vm.state import VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_FM6
from .vm.executor import vm_step_block
from .vm.scheduling import ExecutionPolicy, ExecutionReport
from .primitives import Q15_ONE, Q15_HALF, Q15_NEG_ONE, SAMPLE_RATE_HZ
from .primitives import Q16_ONE
from .paths import PRESET_BUILD_DIR


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SynthConfig:
    """Configuration for SynthMaster."""
    sample_rate: int = SAMPLE_RATE_HZ
    block_size: int = BLOCK_SIZE
    n_voices: int = 8
    output_dir: str = "audio_out"
    library_dir: str = str(PRESET_BUILD_DIR)


# =============================================================================
# OSC MODE CONSTANTS
# =============================================================================

OSC_MODES = {
    "mode_a": OSC_MODE_A,
    "mode_b": OSC_MODE_B,
    "fm6": OSC_MODE_FM6,
}


# =============================================================================
# VOICE SLOT (VM state holder)
# =============================================================================

@dataclass
class VoiceSlot:
    """Voice slot holding VM state for audio generation."""
    voice_id: int
    note: int = 60
    velocity_q15: int = Q15_ONE
    active: bool = False
    vm_state: Optional[VMState] = None
    gain_q15: int = 0


# =============================================================================
# SYNTH MASTER
# =============================================================================

class SynthMaster(StereoOutputMixin):
    """
    Unified synthesis pipeline with VoiceEngine at the core.

    All voice management goes through VoiceEngine (CMMM architecture).
    VM states handle audio generation from VoiceFrames.
    """

    def __init__(self, config: SynthConfig = None):
        """
        Initialize SynthMaster.

        Args:
            config: Configuration options (uses defaults if None)
        """
        self.config = config or SynthConfig()

        # Stage toggles
        self._enable_cockpit = True
        self._enable_material = True
        self._enable_safety = True

        # OSC mode
        self._osc_mode = OSC_MODE_B  # Default to Mode B

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        from .vm.resources import prepare_oscillator_resources
        prepare_oscillator_resources()
        # Control surface
        self.cockpit = CockpitEngine()

        # PresetLibrary
        library_path = Path(self.config.library_dir)
        if not library_path.is_absolute():
            # Make relative to project root (parent of src)
            project_root = Path(__file__).parent.parent.parent
            library_path = project_root / self.config.library_dir

        self.library = PresetLibraryV0(str(library_path))

        self._init_output_stage(self.config.block_size)

        # Audio output
        output_path = Path(self.config.output_dir)
        if not output_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / self.config.output_dir

        self.audio_writer = AudioWriter(output_dir=str(output_path))

        # ParamFrame for material/safety
        self._param_frame = ParamFrame()

        # Accumulator for render()
        self._audio_L: List[int] = []
        self._audio_R: List[int] = []

        # Current preset IR (if loaded)
        self._current_preset = None
        self._preset_name = None

        # ADSR parameters (blocks)
        self._attack_blocks = 7
        self._decay_blocks = 75
        self._sustain = 0.7
        self._release_blocks = 150

        # Harmonic profile (default saw-like)
        self._harmonic_amps = [Q15_ONE, Q15_HALF, 10922, 8191, 6553, 5461, 4681, 4096]

        # FM6 parameters (if applicable)
        self._fm6_params = None

        # VoiceEngine is the single source of truth for all voice management
        self.voice_engine = VoiceEngine(
            max_voices=self.config.n_voices,
            block_size=self.config.block_size,
        )
        self.last_capture_marks = []
        self.last_execution_reports = {}
        self._last_render_frames = {}

        # Reserved per-voice slots (kept for API compat; the CompiledPreset
        # Simple-mode chain replaced the old VoiceEngineAdapter indirection)
        self._voice_adapters: List[Optional[object]] = [None] * self.config.n_voices

        # VM states for audio generation (VoiceEngine provides VoiceFrames, VM renders audio)
        self._voice_slots: List[VoiceSlot] = [
            VoiceSlot(voice_id=i) for i in range(self.config.n_voices)
        ]

    # =========================================================================
    # PRESET LOADING
    # =========================================================================

    def load_preset(self, preset_id: str) -> bool:
        """
        Load a preset from the library.

        Args:
            preset_id: Capsule ID (e.g., "brass-stab-v0")

        Returns:
            True if preset was loaded successfully
        """
        record = self.library.resolve_capsule(preset_id)
        if record is None:
            return False

        # Compile outside the block loop so cockpit notes use the same sound
        # as the direct note API from the first rendered block onward.
        from .library.preset_compiler import compiled_preset_from_capsule
        preset = compiled_preset_from_capsule(record.capsule)
        self.load_compiled_preset(preset)
        self._current_preset = record.capsule
        self._preset_name = record.capsule.identity.name

        return True

    def load_preset_from_yaml(self, yaml_path: str) -> bool:
        """
        Load a preset directly from a YAML file.

        Args:
            yaml_path: Path to the YAML preset file

        Returns:
            True if preset was loaded successfully
        """
        try:
            from .instrument_ir import load_instrument
            from .library.preset_compiler import compiled_preset_from_ir
            ir = load_instrument(yaml_path)
            # Documented chain: IR -> CompiledPreset -> Simple-mode
            # VoiceEngine (step_voice_simple owns frame building)
            self.load_compiled_preset(compiled_preset_from_ir(ir))
            self._preset_name = ir.meta.name or Path(yaml_path).stem
            return True
        except Exception as e:
            print(f"Failed to load preset: {e}")
            return False

    def _extract_preset_params(self, capsule):
        """Compatibility entry point; use the same defaults as compiled loading."""
        from .library.preset_compiler import compiled_preset_from_capsule
        self._apply_preset_output(compiled_preset_from_capsule(capsule))

    def _extract_params_from_ir(self, ir):
        """Extract parameters from InstrumentIR."""
        from .primitives import ms_to_blocks
        from .library.preset_compiler import compiled_preset_from_ir
        self._apply_preset_output(compiled_preset_from_ir(ir))

        # Extract envelope and synthesis params from L0 graph
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Amp" and node.params.adsr:
                adsr = node.params.adsr
                self._attack_blocks = ms_to_blocks(adsr.a_ms)
                self._decay_blocks = ms_to_blocks(adsr.d_ms)
                self._sustain = adsr.s
                self._release_blocks = ms_to_blocks(adsr.r_ms)

            if node.type == "Osc" and node.params.waveform:
                wf = node.params.waveform.lower()
                # Map waveform to harmonic profile
                if wf == "sine":
                    self._harmonic_amps = [Q15_ONE, 0, 0, 0, 0, 0, 0, 0]
                elif wf == "saw":
                    self._harmonic_amps = [Q15_ONE, Q15_HALF, 10922, 8191, 6553, 5461, 4681, 4096]
                elif wf in ("square", "pulse"):
                    self._harmonic_amps = [Q15_ONE, 0, 10922, 0, 6553, 0, 4681, 0]
                elif wf in ("tri", "triangle"):
                    self._harmonic_amps = [Q15_ONE, 0, 3640, 0, 1310, 0, 668, 0]

            if node.type == "FM6":
                self._osc_mode = OSC_MODE_FM6
                self._fm6_params = {
                    'algorithm': getattr(node.params, 'algorithm', 1),
                    'feedback': getattr(node.params, 'feedback', 0),
                    'op_ratios': getattr(node.params, 'op_ratios', [Q16_ONE] * 6),
                    'op_levels': getattr(node.params, 'op_levels', [Q15_ONE, 0, 0, 0, 0, 0]),
                }

        # Custom partials (waveform: partials) take priority over the
        # waveform-derived harmonic profiles above
        from .instrument_ir import extract_partials_q15
        partials = extract_partials_q15(ir)
        if partials:
            self._harmonic_amps = partials

        # Extract Material config from IR
        if hasattr(ir, 'compile') and hasattr(ir.compile, 'material'):
            mat_cfg = ir.compile.material
            self._material_id = mat_cfg.material_id
            self._beta_target_q15 = mat_cfg.beta_target_q15
            self._beta_start_q15 = mat_cfg.beta_start_q15
            self._beta_ramp_blocks = mat_cfg.beta_ramp_blocks



    # =========================================================================
    # VOICE LIFECYCLE
    # =========================================================================

    def note_on(self, note: int, velocity: int = 100) -> Optional[int]:
        """
        Trigger a note through VoiceEngine.

        Args:
            note: MIDI note number (0-127)
            velocity: MIDI velocity (0-127)

        Returns:
            Voice ID if successful, None if no voices available
        """
        from .control_frame import ControlFrame
        note = max(0, min(127, int(note)))
        velocity = max(0, min(127, int(velocity)))
        if velocity == 0:
            self.note_off(note=note)
            return None
        frame = ControlFrame(mode_L=self.voice_engine.control_mode_L)
        frame.add_note_on(note, (velocity * Q15_ONE) // 127)
        self.voice_engine.consume_control_frame(frame)
        return self.voice_engine._find_voice_by_note(note)

    def _get_compiled_preset(self) -> Optional['CompiledPreset']:
        """Return the selected sound, including changes from ControlFrame."""
        return self.voice_engine.selected_preset

    def load_compiled_preset(self, preset: 'CompiledPreset') -> None:
        """
        Load a pre-compiled preset for VoiceEngine mode.

        Args:
            preset: CompiledPreset instance
        """
        self._apply_preset_output(preset)
        self._compiled_preset = preset
        self.voice_engine.select_compiled_preset(preset)
        self._preset_name = preset.preset_id

        # Clear existing adapters so they get recreated with new preset
        self._voice_adapters = [None] * self.config.n_voices

    def register_control_preset(self, preset_id: int, preset: 'CompiledPreset',
                                preset_hash32: int = 0) -> None:
        """Register a compiled sound for cockpit PRESET_LOAD commands."""
        self.voice_engine.register_control_preset(preset_id, preset, preset_hash32)

    def note_off(self, note: int = None, voice_id: int = None) -> None:
        """
        Release a note through VoiceEngine.

        Args:
            note: MIDI note number (finds first matching voice)
            voice_id: Specific voice ID to release
        """
        from .control_frame import ControlFrame
        frame = ControlFrame(mode_L=self.voice_engine.control_mode_L)
        if voice_id is not None:
            state = self.voice_engine.voices.get(voice_id)
            if state is None:
                return
            frame.add_note_off(state.note_u8, state.channel)
            frame.events[-1].voice_id = voice_id
        elif note is not None:
            frame.add_note_off(max(0, min(127, int(note))))
        self.voice_engine.consume_control_frame(frame)

    # =========================================================================
    # RENDER
    # =========================================================================

    def render_block(self, *, schedule=None, voice_budget=None) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """
        Render one audio block through VoiceEngine.

        schedule: optional host BlockScheduler result, computed before this block.
        voice_budget: optional RegimeBudget applied to each serviced VM voice.
        These use distinct units; execution reports never feed an uncalibrated
        coefficient tick count back into global compute/memory health bars.

        Returns:
            (L_samples, R_samples, health_telemetry)
        """
        # 1. Tick cockpit (if enabled) - produces ControlFrame
        control_frame = None
        if self._enable_cockpit:
            control_frame = self.cockpit.tick(voice_ids=range(self.voice_engine.max_voices))

        return self._render_block_voice_engine(control_frame, schedule=schedule,
                                               voice_budget=voice_budget)

    def _render_block_voice_engine(
        self,
        control_frame: Optional['ControlFrame'] = None,
        *, schedule=None, voice_budget=None,
    ) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """
        Render using VoiceEngine (CMMM) for voice management.

        Per WSE-ARCHITECTURE-v1:
        - VoiceEngine consumes ControlFrame, returns render_batch + ParamFrame
        - SynthMaster handles rendering (Stages 3-10: VM → IDWPT → mix → Material → Safety)

        Args:
            control_frame: Optional ControlFrame from CockpitEngine
        """
        # Step VoiceEngine with ControlFrame - this is the canonical API
        # VoiceEngine handles: events, lifecycle, scheduling, CMMM stepping
        engine_output = self.voice_engine.step_block(control_frame, schedule=schedule)
        return self._render_output(engine_output, scheduled=schedule is not None,
                                   voice_budget=voice_budget, apply_fallbacks=True)


    def render_frames(self, frames, *, params=None, schedule=None, voice_budget=None):
        """Render externally emitted VoiceFrames through the shared output path.

        Frozen packets can be supplied after to_voice_frame(). This advances
        the VM and output effects, not the native VoiceEngine's musical clock.
        The caller supplies current frames for every sounding or held slot.
        """
        frames = list(frames)
        if any(not isinstance(frame, VoiceFrame) for frame in frames):
            raise TypeError('Rendering requires VoiceFrame instances')
        if params is not None and not isinstance(params, ParamFrame):
            raise TypeError('Output parameters must be a ParamFrame')
        ids = [frame.voice_id for frame in frames]
        if (any(not isinstance(vid, int) or isinstance(vid, bool)
                or not 0 <= vid < self.config.n_voices for vid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Frames must address unique, valid voice slots')
        selected = None
        if schedule is not None:
            selected = {info.voice_id: info for info in schedule.serviced}
            if len(selected) != len(schedule.serviced) or any(vid not in ids for vid in selected):
                raise ValueError('Scheduled voices must be unique and have a supplied frame')
        batch = []
        for frame in sorted(frames, key=lambda f: f.voice_id):
            info = selected.get(frame.voice_id) if selected is not None else None
            serviced = selected is None or info is not None
            batch.append(RenderBatchItem(frame.voice_id, frame,
                mid_due=info.mid_due if info is not None else serviced,
                top_due=info.top_due if info is not None else serviced,
                serviced=serviced,
                degrade_level=schedule.degrade_state if schedule is not None else 0))
        output = VoiceEngineOutput(render_batch=batch, param_frame=params or self._param_frame,
                                   telemetry=schedule.telemetry if schedule is not None else None)
        return self._render_output(output, scheduled=schedule is not None,
                                   voice_budget=voice_budget, apply_fallbacks=False)

    def _render_output(self, engine_output, *, scheduled=False, voice_budget=None,
                       apply_fallbacks=False):
        selected = self.voice_engine.selected_preset
        if selected is not None and selected is not self._output_preset:
            self._apply_preset_output(selected)
        rendered_audio = []
        consumer_events = []
        rendered_frames = []
        self.last_capture_marks = engine_output.capture_marks
        self.last_execution_reports = {}

        # Use ParamFrame from VoiceEngine (derived from ControlFrame.globals)
        param_frame = engine_output.param_frame

        # Render each voice in the batch (Stages 3-10)
        for item in engine_output.render_batch:
            voice_id = item.voice_id
            frame = item.frame

            # The VM owns frozen-packet acceptance and retrigger semantics.
            # A held frame also holds its pan, sends, and post-filter controls.
            wire = frame.wire_data
            if wire is not None and wire.reject_code:
                self.last_execution_reports[voice_id] = ExecutionReport()
                continue
            if not item.serviced:
                frame = self._last_render_frames.get(voice_id)
                if frame is None:
                    continue

            # Ensure slot tracking is in sync
            if voice_id < len(self._voice_slots):
                slot = self._voice_slots[voice_id]
                # Ensure VM state exists
                if slot.vm_state is None or (item.serviced and frame.is_starting and frame.wire_data is None):
                    if slot.vm_state is None:
                        slot.vm_state = VMState.create(voice_id=voice_id)
                    slot.vm_state.reset_for_note_on(note=frame.note)

                # Compiled voices own every emitted value, including Mode A,
                # an empty named-waveform spectrum and a zero blend/ramp start.
                # Old notes retain their own preset after a program change.
                # Only a serviced, unowned native frame receives legacy defaults;
                # leave the VoiceEngine's cached frame and held output intact.
                if apply_fallbacks and item.serviced and frame.wire_data is None:
                    owner = self.voice_engine.voices.get(voice_id)
                    if owner is None or owner.compiled_preset is None:
                        frame = copy(frame)
                        if not frame.partial_amps_q15:
                            frame.partial_amps_q15 = list(self._harmonic_amps)
                        if frame.beta_target_q15 == 0:
                            frame.beta_target_q15 = self._beta_target_q15
                        if frame.coeff_osc_mode == 0:
                            frame.coeff_osc_mode = self._osc_mode

                # Run VM (Stage 3-6: Symbol Engine)
                policy = (ExecutionPolicy(item.serviced, item.mid_due, item.top_due, voice_budget)
                          if scheduled or voice_budget is not None else None)
                report = ExecutionReport() if policy is not None or frame.wire_data is not None else None
                slot.vm_state, output = vm_step_block(slot.vm_state, frame, policy=policy, report=report)
                if report is not None:
                    self.last_execution_reports[voice_id] = report
                    if item.serviced and not report.serviced:
                        slot.active = False
                        continue  # Rejected/stale packet: no audio or consumer events.
                    consumer_events.extend((voice_id, event) for event in report.consumer_events)
                slot.active = True
                slot.gain_q15 = frame.gain_q15
                slot.note = frame.note
                if item.serviced:
                    self._last_render_frames[voice_id] = deepcopy(frame)
                rendered_frames.append(frame)

                # VM scratch output may be reused for the next voice. Retain
                # this voice's samples before deferring the shared stereo mix.
                rendered_audio.append((frame, output.copy()))

        # Mark inactive voices (not in render batch)
        active_ids = {frame.voice_id for frame in rendered_frames}
        for i, slot in enumerate(self._voice_slots):
            if i not in active_ids:
                slot.active = False

        L, R = self._mix_output(rendered_audio, param_frame, consumer_events)

        # Use telemetry from VoiceEngine
        telemetry = engine_output.telemetry
        if self._enable_cockpit and telemetry is not None:
            self.cockpit.receive_health_telemetry(telemetry)

        # Accumulate for save_audio()
        self._audio_L.extend(L)
        self._audio_R.extend(R)

        return L, R, telemetry


    def render(self, duration_sec: float) -> Tuple[List[int], List[int]]:
        """
        Render audio for a specified duration.

        Args:
            duration_sec: Duration in seconds

        Returns:
            (L_samples, R_samples)
        """
        # Clear accumulators
        self._audio_L = []
        self._audio_R = []

        blocks_per_sec = self.config.sample_rate // self.config.block_size
        total_blocks = int(duration_sec * blocks_per_sec)

        for _ in range(total_blocks):
            self.render_block()

        return self._audio_L, self._audio_R

    # =========================================================================
    # AUDIO OUTPUT
    # =========================================================================

    def save_audio(
        self,
        filename: str,
        subdirectory: str = "renders",
        timestamp: bool = True,
    ) -> AudioMetadata:
        """
        Save accumulated audio to file.

        Args:
            filename: Base filename (without extension)
            subdirectory: Subdirectory within output_dir
            timestamp: Whether to add timestamp to filename

        Returns:
            AudioMetadata with file information
        """
        return self.audio_writer.write_audio(
            filename=filename,
            L=self._audio_L,
            R=self._audio_R,
            subdirectory=subdirectory,
            timestamp=timestamp,
        )

    def clear_audio(self) -> None:
        """Clear accumulated audio buffers."""
        self._audio_L = []
        self._audio_R = []

    def get_audio(self) -> Tuple[List[int], List[int]]:
        """Get accumulated audio without clearing."""
        return self._audio_L, self._audio_R

    # =========================================================================
    # CONTROL PLANE
    # =========================================================================

    def _apply_control_frame(self, control_frame: 'ControlFrame') -> None:
        """Compatibility entry point; the VoiceEngine owns control dispatch."""
        self.voice_engine.consume_control_frame(control_frame)

    # =========================================================================
    # TOGGLE API
    # =========================================================================

    def disable(self, stage: str) -> None:
        """
        Disable a pipeline stage.

        Args:
            stage: One of "cockpit", "material", "safety", "macro"
        """
        if stage == "cockpit":
            self._enable_cockpit = False
        elif stage in ("material", "safety"):
            self.disable_output_stage(stage)
        elif stage == "macro":
            self._enable_macro = False
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def enable(self, stage: str) -> None:
        """
        Enable a pipeline stage.

        Args:
            stage: One of "cockpit", "material", "safety", "macro"
        """
        if stage == "cockpit":
            self._enable_cockpit = True
        elif stage in ("material", "safety"):
            self.enable_output_stage(stage)
        elif stage == "macro":
            self._enable_macro = True
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def set_osc_mode(self, mode: str) -> None:
        """
        Set the fallback oscillator mode for voices without a compiled preset.

        Compiled voices use their emitted mode. A scheduled hold retains its
        last rendered frame until the voice is serviced again.

        Args:
            mode: One of "mode_a", "mode_b", "fm6"
        """
        if mode not in OSC_MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(OSC_MODES.keys())}")
        self._osc_mode = OSC_MODES[mode]


    # =========================================================================
    # CONTROL API
    # =========================================================================

    def set_control(self, control_id: int, value: int) -> None:
        """
        Set a control value.

        Args:
            control_id: ControlID enum value
            value: Control value
        """
        self.cockpit.set_control(control_id, value)

    def set_adsr(
        self,
        a_ms: float = None,
        d_ms: float = None,
        s: float = None,
        r_ms: float = None,
    ) -> None:
        """
        Set ADSR envelope parameters.

        Args:
            a_ms: Attack time in milliseconds
            d_ms: Decay time in milliseconds
            s: Sustain level (0.0 to 1.0)
            r_ms: Release time in milliseconds
        """
        from .primitives import ms_to_blocks

        if a_ms is not None:
            self._attack_blocks = ms_to_blocks(a_ms)
            self.cockpit.set_control(ControlID.ATK_BLOCKS, self._attack_blocks)
        if d_ms is not None:
            self._decay_blocks = ms_to_blocks(d_ms)
            self.cockpit.set_control(ControlID.DCY_BLOCKS, self._decay_blocks)
        if s is not None:
            self._sustain = max(0.0, min(1.0, s))
            self.cockpit.set_control(ControlID.SUS_Q15, int(self._sustain * Q15_ONE))
        if r_ms is not None:
            self._release_blocks = ms_to_blocks(r_ms)
            self.cockpit.set_control(ControlID.REL_BLOCKS, self._release_blocks)

    def pitch_bend(self, value: int) -> None:
        """
        Set pitch bend.

        Args:
            value: MIDI pitch bend value (0-16383, center=8192)
        """
        self.cockpit.midi_pitch_bend(value)

    # =========================================================================
    # TELEMETRY
    # =========================================================================

    def get_health_telemetry(self) -> Optional[HealthTelemetry]:
        """Get current health telemetry."""
        return self.cockpit.get_health_telemetry()

    def get_health_summary(self) -> Dict[str, int]:
        """Get health telemetry summary."""
        return self.cockpit.get_health_summary()

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_active_voice_count(self) -> int:
        """Get count of currently active voices."""
        return self.voice_engine.get_active_voice_count()

    def get_preset_name(self) -> Optional[str]:
        """Get name of currently loaded preset."""
        if self.voice_engine.selected_preset_id is not None:
            return self.voice_engine.selected_preset.preset_id
        return self._preset_name

    def get_stage_status(self) -> Dict[str, bool]:
        """Get status of all pipeline stages."""
        return {
            "cockpit": self._enable_cockpit,
            "material": self._enable_material,
            "safety": self._enable_safety,
            "macro": self._enable_macro,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SynthConfig',
    'SynthMaster',
]
