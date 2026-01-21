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
    synth.set_osc_mode("mode_b")  # Use harmonic bank instead of FM6
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .audio_io import AudioWriter, AudioMetadata
from .cockpit_engine import CockpitEngine
from .control_registry import ControlID
from .library.preset_library import PresetLibraryV0, Query
from .library.capsule import CapsuleKind
from .material.material_engine import MaterialEngine
from .safety.safety_engine import SafetyEngine
from .voice.scheduler import HealthTelemetry, BLOCK_SIZE
from .voice.voice_engine import VoiceEngine
from .voice.emitter_bridge import VoiceEngineAdapter
from .frames import ParamFrame, VoiceFrame
from .vm.state import VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_FM6
from .vm.executor import vm_step_block
from .primitives import Q15_ONE, Q15_HALF, Q15_NEG_ONE, SAMPLE_RATE_HZ


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
    library_dir: str = "preset_library"


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

class SynthMaster:
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
        # Control surface
        self.cockpit = CockpitEngine()

        # PresetLibrary
        library_path = Path(self.config.library_dir)
        if not library_path.is_absolute():
            # Make relative to project root (parent of src)
            project_root = Path(__file__).parent.parent.parent
            library_path = project_root / self.config.library_dir

        self.library = PresetLibraryV0(str(library_path))

        # Post-processing (toggleable)
        self.material = MaterialEngine(self.config.block_size)
        self._material_id = 0  # Clean default (can be overridden by IR)
        self._beta_target_q15 = Q15_HALF  # Default 0.5 blend (can be overridden by IR)
        self._beta_start_q15 = 0       # Start dry
        self._beta_ramp_blocks = 8     # Ramp over 8 blocks
        self.safety = SafetyEngine(self.config.block_size)
        self._safety_dc_removal = False    # Disabled for CCS path
        self._safety_limiter_threshold = 32000
        self._safety_slew_limit = 2000

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

        # VoiceEngineAdapters for each voice (created when preset is loaded)
        self._voice_adapters: List[Optional[VoiceEngineAdapter]] = [None] * self.config.n_voices

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

        self._current_preset = record.capsule
        self._preset_name = record.capsule.identity.name

        # Extract parameters from capsule
        self._extract_preset_params(record.capsule)

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
            ir = load_instrument(yaml_path)
            self._extract_params_from_ir(ir)
            self._preset_name = ir.meta.name or Path(yaml_path).stem
            return True
        except Exception as e:
            print(f"Failed to load preset: {e}")
            return False

    def _extract_preset_params(self, capsule):
        """Extract parameters from capsule."""
        # For now, use defaults. Full implementation would parse capsule.program
        pass

    def _extract_params_from_ir(self, ir):
        """Extract parameters from InstrumentIR."""
        # Helper function
        def ms_to_blocks(ms):
            return max(1, int(ms / 1000.0 * self.config.sample_rate / self.config.block_size))

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
                    'op_ratios': getattr(node.params, 'op_ratios', [65536] * 6),
                    'op_levels': getattr(node.params, 'op_levels', [32767, 0, 0, 0, 0, 0]),
                }

        # Extract Material config from IR
        if hasattr(ir, 'compile') and hasattr(ir.compile, 'material'):
            mat_cfg = ir.compile.material
            self._material_id = mat_cfg.material_id
            self._beta_target_q15 = mat_cfg.beta_target_q15
            self._beta_start_q15 = mat_cfg.beta_start_q15
            self._beta_ramp_blocks = mat_cfg.beta_ramp_blocks

        # Extract Safety config from IR
        if hasattr(ir, 'compile') and hasattr(ir.compile, 'safety'):
            safety_cfg = ir.compile.safety
            self._enable_safety = safety_cfg.enabled
            self._safety_dc_removal = safety_cfg.dc_removal
            self._safety_limiter_threshold = safety_cfg.limiter_threshold_q15
            self._safety_slew_limit = safety_cfg.slew_limit_q15

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
        velocity_q15 = (velocity * Q15_ONE) // 127

        # Find free voice slot
        voice_id = None
        for i, slot in enumerate(self._voice_slots):
            if not slot.active:
                voice_id = i
                break

        if voice_id is None:
            return None

        slot = self._voice_slots[voice_id]

        # Create adapter if needed (lazy creation per voice)
        if self._voice_adapters[voice_id] is None:
            preset = self._get_compiled_preset()
            if preset is not None:
                self._voice_adapters[voice_id] = VoiceEngineAdapter(
                    preset=preset,
                    engine=self.voice_engine,
                    voice_id=voice_id,
                )

        # Trigger note through adapter or directly on engine
        adapter = self._voice_adapters[voice_id]
        if adapter is not None:
            adapter.note_on(note=note, velocity_q15=velocity_q15)
        else:
            # Fallback: use engine directly (no preset loaded)
            self.voice_engine.note_on(voice_id, velocity_q15=velocity_q15)

        # Update slot state
        slot.note = note
        slot.velocity_q15 = velocity_q15
        slot.active = True

        # Create VM state for audio generation
        slot.vm_state = VMState.create(voice_id=voice_id)
        slot.vm_state.reset_for_note_on(note=note)

        return voice_id

    def _get_compiled_preset(self) -> Optional['CompiledPreset']:
        """Get compiled preset for VoiceEngine adapters."""
        # If we have a compiled preset cached, return it
        if hasattr(self, '_compiled_preset') and self._compiled_preset is not None:
            return self._compiled_preset

        # Try to compile from current preset
        if self._current_preset is not None:
            try:
                from .library.preset_compiler import PresetCompiler
                compiler = PresetCompiler()
                # For now return None - full implementation would compile from capsule
                return None
            except Exception:
                pass

        return None

    def load_compiled_preset(self, preset: 'CompiledPreset') -> None:
        """
        Load a pre-compiled preset for VoiceEngine mode.

        Args:
            preset: CompiledPreset instance
        """
        self._compiled_preset = preset

        # Clear existing adapters so they get recreated with new preset
        self._voice_adapters = [None] * self.config.n_voices

    def note_off(self, note: int = None, voice_id: int = None) -> None:
        """
        Release a note through VoiceEngine.

        Args:
            note: MIDI note number (finds first matching voice)
            voice_id: Specific voice ID to release
        """
        if voice_id is not None:
            # Release specific voice
            adapter = self._voice_adapters[voice_id] if voice_id < len(self._voice_adapters) else None
            if adapter is not None:
                adapter.note_off()
            else:
                self.voice_engine.note_off(voice_id)
        elif note is not None:
            # Find voice playing this note
            for i, slot in enumerate(self._voice_slots):
                if slot.active and slot.note == note:
                    adapter = self._voice_adapters[i] if i < len(self._voice_adapters) else None
                    if adapter is not None:
                        adapter.note_off()
                    else:
                        self.voice_engine.note_off(i)
                    break

    # =========================================================================
    # RENDER
    # =========================================================================

    def render_block(self) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """
        Render one audio block through VoiceEngine.

        Returns:
            (L_samples, R_samples, health_telemetry)
        """
        # 1. Tick cockpit (if enabled) - produces ControlFrame
        control_frame = None
        if self._enable_cockpit:
            control_frame = self.cockpit.tick()

        return self._render_block_voice_engine(control_frame)

    def _render_block_voice_engine(
        self,
        control_frame: Optional['ControlFrame'] = None
    ) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """
        Render using VoiceEngine (CMMM) for voice management.

        Per WSE-ARCHITECTURE-v1:
        - VoiceEngine consumes ControlFrame, returns render_batch + ParamFrame
        - SynthMaster handles rendering (Stages 3-10: VM → IDWPT → mix → Material → Safety)

        Args:
            control_frame: Optional ControlFrame from CockpitEngine
        """
        block_size = self.config.block_size
        L = [0] * block_size
        R = [0] * block_size

        # Step VoiceEngine with ControlFrame - this is the canonical API
        # VoiceEngine handles: events, lifecycle, scheduling, CMMM stepping
        engine_output = self.voice_engine.step_block(control_frame)

        # Use ParamFrame from VoiceEngine (derived from ControlFrame.globals)
        param_frame = engine_output.param_frame

        # Render each voice in the batch (Stages 3-10)
        for item in engine_output.render_batch:
            voice_id = item.voice_id
            frame = item.frame

            # Ensure slot tracking is in sync
            if voice_id < len(self._voice_slots):
                slot = self._voice_slots[voice_id]
                slot.active = True
                slot.gain_q15 = frame.gain_q15
                slot.note = getattr(frame, 'note', 60)

                # Ensure VM state exists
                if slot.vm_state is None:
                    slot.vm_state = VMState.create(voice_id=voice_id)
                    slot.vm_state.reset_for_note_on(note=slot.note)

                # Enrich frame with preset parameters if needed
                if not frame.partial_amps_q15:
                    frame.partial_amps_q15 = self._harmonic_amps
                if frame.beta_target_q15 == 0:
                    frame.beta_target_q15 = self._beta_target_q15
                if frame.coeff_osc_mode == 0:
                    frame.coeff_osc_mode = self._osc_mode

                # Run VM (Stage 3-6: Symbol Engine)
                slot.vm_state, output = vm_step_block(slot.vm_state, frame)

                # Mix into output (Stage 8: Voice Mix)
                for j in range(block_size):
                    L[j] += output[j]
                    R[j] += output[j]

        # Mark inactive voices (not in render batch)
        active_ids = {item.voice_id for item in engine_output.render_batch}
        for i, slot in enumerate(self._voice_slots):
            if i not in active_ids:
                slot.active = False

        # Clip to int16 range
        L = [max(Q15_NEG_ONE, min(Q15_ONE, s)) for s in L]
        R = [max(Q15_NEG_ONE, min(Q15_ONE, s)) for s in R]

        # Aggregate filter parameters from active voices
        # Use minimum cutoff (darkest voice wins) for musical coherence
        filter_cutoff_q15 = Q15_ONE
        filter_reso_q15 = 0
        filter_mode = 0
        filter_enabled = False

        for item in engine_output.render_batch:
            frame = item.voice_frame
            if frame is not None:
                fc = getattr(frame, 'filter_cutoff_q15', Q15_ONE)
                fr = getattr(frame, 'filter_reso_q15', 0)
                fm = getattr(frame, 'filter_mode', 0)
                # Use minimum cutoff (darkest voice)
                if fc < filter_cutoff_q15:
                    filter_cutoff_q15 = fc
                    filter_mode = fm
                # Use maximum resonance
                if fr > filter_reso_q15:
                    filter_reso_q15 = fr
                # Enable filter if any voice has cutoff below max
                if fc < Q15_ONE:
                    filter_enabled = True

        # Apply post-processing if enabled (Stages 9-10)
        if self._enable_material:
            # Use material_id=9 (SVF filter) if filter is enabled, otherwise use configured material
            material_id = 9 if filter_enabled else self._material_id
            L, R = self.material.step_block(
                L, R, param_frame, material_id, 0,
                filter_cutoff_q15=filter_cutoff_q15,
                filter_reso_q15=filter_reso_q15,
                filter_mode=filter_mode
            )
        if self._enable_safety:
            L, R = self.safety.step_block(L, R, param_frame)

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
        """
        Apply ControlFrame to VoiceEngine.

        This bridges the Cockpit → VoiceEngine control plane per
        WSE-STD-070 InstrumentIR Tower Organization Plan.

        The ControlFrame contains:
        - Global controls (master_gain, pan, tempo)
        - Voice patches (per-voice control updates)
        - Events (note on/off, preset load, mode set)

        Args:
            control_frame: ControlFrame from CockpitEngine
        """
        # Apply global controls
        if control_frame.globals.master_gain_q15 is not None:
            # Could route to master gain processing if needed
            pass

        # Process events
        for event in control_frame.events:
            from .control_frame import EventType

            if event.type == EventType.NOTE_ON:
                note = event.a & 0x7F
                velocity_q15 = (event.a >> 8) & 0xFFFF
                # Note: note_on should be called via public API
                # This is for events queued in cockpit
                pass

            elif event.type == EventType.NOTE_OFF:
                note = event.a & 0x7F
                # Note: note_off should be called via public API
                pass

            elif event.type == EventType.MODE_SET:
                # Mode change (Tower level)
                mode_L = event.a
                # Could update VoiceEngine tower level
                pass

        # Apply voice patches (sparse control updates)
        for patch in control_frame.voice_patches:
            voice_id = patch.voice_id
            if voice_id < len(self._voice_slots):
                slot = self._voice_slots[voice_id]
                if slot.active:
                    for item in patch.items:
                        # Apply control updates to VoiceEngine state
                        # The item.control_id maps to ControlID from control_registry
                        pass

        # Clear events after processing (consumed once)
        control_frame.clear_events()

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
        elif stage == "material":
            self._enable_material = False
        elif stage == "safety":
            self._enable_safety = False
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
        elif stage == "material":
            self._enable_material = True
        elif stage == "safety":
            self._enable_safety = True
        elif stage == "macro":
            self._enable_macro = True
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def set_osc_mode(self, mode: str) -> None:
        """
        Set oscillator mode.

        Args:
            mode: One of "mode_a", "mode_b", "fm6"
        """
        if mode not in OSC_MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(OSC_MODES.keys())}")
        self._osc_mode = OSC_MODES[mode]

    def set_material(self, material_id: int) -> None:
        """
        Set material effect.

        Args:
            material_id: Material ID (0=clean, 1=tape, 2=rank, 3=tropical, 4=catastrophe)
        """
        self._material_id = material_id

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
        blocks_per_sec = self.config.sample_rate // self.config.block_size

        if a_ms is not None:
            self._attack_blocks = max(1, int(a_ms / 1000.0 * blocks_per_sec))
            self.cockpit.set_control(ControlID.ATK_BLOCKS, self._attack_blocks)
        if d_ms is not None:
            self._decay_blocks = max(1, int(d_ms / 1000.0 * blocks_per_sec))
            self.cockpit.set_control(ControlID.DCY_BLOCKS, self._decay_blocks)
        if s is not None:
            self._sustain = max(0.0, min(1.0, s))
            self.cockpit.set_control(ControlID.SUS_Q15, int(self._sustain * Q15_ONE))
        if r_ms is not None:
            self._release_blocks = max(1, int(r_ms / 1000.0 * blocks_per_sec))
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
        if self._use_orchestrator:
            return self.orchestrator.get_active_voice_count()
        else:
            return sum(1 for slot in self._voice_slots if slot.active)

    def get_preset_name(self) -> Optional[str]:
        """Get name of currently loaded preset."""
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
