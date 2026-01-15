"""
SynthMaster - Unified Synthesis Pipeline
=========================================

High-level interface for the complete synthesis pipeline.
Uses BlockScheduler + BlockOrchestrator as the central coordinator.

Features:
- Toggle pipeline stages on/off for debugging
- Preset loading from PresetLibrary
- Voice lifecycle management via orchestrator
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
from .voice.scheduler import BlockScheduler, HealthTelemetry, BLOCK_SIZE
from .voice.orchestrator import BlockOrchestrator, VoiceState
from .frames import ParamFrame, VoiceFrame
from .vm.state import VMState, OSC_MODE_A, OSC_MODE_B, OSC_MODE_FM6
from .vm.executor import vm_step_block


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SynthConfig:
    """Configuration for SynthMaster."""
    sample_rate: int = 48000
    block_size: int = 64
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
# SIMPLE VOICE SLOT (for direct VM rendering)
# =============================================================================

@dataclass
class SimpleVoiceSlot:
    """Simple voice slot for direct VM rendering."""
    voice_id: int
    note: int = 60
    velocity_q15: int = 32767
    active: bool = False
    vm_state: Optional[VMState] = None

    # Envelope state
    envelope_phase: str = "idle"  # idle, attack, decay, sustain, release
    envelope_block: int = 0
    gain_q15: int = 0


# =============================================================================
# SYNTH MASTER
# =============================================================================

class SynthMaster:
    """
    Unified synthesis pipeline with toggleable stages.

    Built around BlockScheduler (decisions) + BlockOrchestrator (execution).
    Also supports direct VM rendering for simpler use cases.
    """

    def __init__(self, config: SynthConfig = None, use_orchestrator: bool = False):
        """
        Initialize SynthMaster.

        Args:
            config: Configuration options (uses defaults if None)
            use_orchestrator: If True, use full BlockOrchestrator pipeline.
                            If False (default), use simpler direct VM rendering.
        """
        self.config = config or SynthConfig()
        self._use_orchestrator = use_orchestrator

        # Stage toggles
        self._enable_cockpit = True
        self._enable_material = True
        self._enable_safety = True
        self._enable_macro = True

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
        self._material_id = 0  # Clean default
        self.safety = SafetyEngine(self.config.block_size)

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
        self._harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]

        # FM6 parameters (if applicable)
        self._fm6_params = None

        if self._use_orchestrator:
            # CORE: Scheduler + Orchestrator
            self.scheduler = BlockScheduler(
                b_compute=1000,
                b_mem=1000,
            )
            self.orchestrator = BlockOrchestrator(
                n_voices=self.config.n_voices,
                block_size=self.config.block_size,
                scheduler=self.scheduler,
            )
        else:
            # Simple voice slots for direct VM rendering
            self.scheduler = None
            self.orchestrator = None
            self._voice_slots: List[SimpleVoiceSlot] = [
                SimpleVoiceSlot(voice_id=i) for i in range(self.config.n_voices)
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
                    self._harmonic_amps = [32767, 0, 0, 0, 0, 0, 0, 0]
                elif wf == "saw":
                    self._harmonic_amps = [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096]
                elif wf in ("square", "pulse"):
                    self._harmonic_amps = [32767, 0, 10922, 0, 6553, 0, 4681, 0]
                elif wf in ("tri", "triangle"):
                    self._harmonic_amps = [32767, 0, 3640, 0, 1310, 0, 668, 0]

            if node.type == "FM6":
                self._osc_mode = OSC_MODE_FM6
                self._fm6_params = {
                    'algorithm': getattr(node.params, 'algorithm', 1),
                    'feedback': getattr(node.params, 'feedback', 0),
                    'op_ratios': getattr(node.params, 'op_ratios', [65536] * 6),
                    'op_levels': getattr(node.params, 'op_levels', [32767, 0, 0, 0, 0, 0]),
                }

    # =========================================================================
    # VOICE LIFECYCLE
    # =========================================================================

    def note_on(self, note: int, velocity: int = 100) -> Optional[int]:
        """
        Trigger a note.

        Args:
            note: MIDI note number (0-127)
            velocity: MIDI velocity (0-127)

        Returns:
            Voice ID if successful, None if no voices available
        """
        velocity_q15 = (velocity * 32767) // 127

        if self._use_orchestrator:
            # Get free voice from orchestrator
            voice_id = self.orchestrator.get_free_voice()
            if voice_id is None:
                return None

            # Allocate and arm
            self.orchestrator.voice_allocate(voice_id, owner_id=1, note_u8=note)
            self.orchestrator.voice_on(voice_id)
            return voice_id
        else:
            # Simple voice allocation
            for slot in self._voice_slots:
                if not slot.active:
                    slot.note = note
                    slot.velocity_q15 = velocity_q15
                    slot.active = True
                    slot.envelope_phase = "attack"
                    slot.envelope_block = 0
                    slot.gain_q15 = 0

                    # Create VM state
                    slot.vm_state = VMState.create(voice_id=slot.voice_id)
                    slot.vm_state.reset_for_note_on(note=note)

                    return slot.voice_id
            return None

    def note_off(self, note: int = None, voice_id: int = None) -> None:
        """
        Release a note.

        Args:
            note: MIDI note number (finds first matching voice)
            voice_id: Specific voice ID to release
        """
        if self._use_orchestrator:
            if voice_id is not None:
                self.orchestrator.voice_free(voice_id)
            elif note is not None:
                # Find voice playing this note
                for vid, slot in enumerate(self.orchestrator.voice_slots):
                    if slot.state == VoiceState.ACTIVE and slot.note_u8 == note:
                        self.orchestrator.voice_free(vid)
                        break
        else:
            # Simple voice release
            if voice_id is not None:
                if 0 <= voice_id < len(self._voice_slots):
                    slot = self._voice_slots[voice_id]
                    if slot.active:
                        slot.envelope_phase = "release"
                        slot.envelope_block = 0
            elif note is not None:
                for slot in self._voice_slots:
                    if slot.active and slot.note == note:
                        slot.envelope_phase = "release"
                        slot.envelope_block = 0
                        break

    # =========================================================================
    # RENDER
    # =========================================================================

    def render_block(self) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """
        Render one audio block.

        Returns:
            (L_samples, R_samples, health_telemetry)
        """
        # 1. Tick cockpit (if enabled) - produces ControlFrame
        if self._enable_cockpit:
            control_frame = self.cockpit.tick()

        if self._use_orchestrator:
            return self._render_block_orchestrated()
        else:
            return self._render_block_simple()

    def _render_block_simple(self) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """Render using simple direct VM approach."""
        block_size = self.config.block_size
        L = [0] * block_size
        R = [0] * block_size

        for slot in self._voice_slots:
            if not slot.active:
                continue

            # Calculate envelope gain
            gain_q15 = self._compute_envelope_gain(slot)
            slot.gain_q15 = gain_q15

            # Check if voice has finished
            if slot.envelope_phase == "idle":
                slot.active = False
                continue

            # Create VoiceFrame
            if self._osc_mode == OSC_MODE_FM6 and self._fm6_params:
                frame = VoiceFrame(
                    note=slot.note,
                    coeff_osc_mode=OSC_MODE_FM6,
                    gain_q15=gain_q15,
                    beta_target_q15=0,
                    fm_algorithm_id=self._fm6_params['algorithm'],
                    fm_op_ratios=self._fm6_params['op_ratios'],
                    fm_op_levels=self._fm6_params['op_levels'],
                    fm_feedback_depth=self._fm6_params['feedback'],
                )
            else:
                frame = VoiceFrame(
                    note=slot.note,
                    coeff_osc_mode=self._osc_mode,
                    H_request=len(self._harmonic_amps),
                    gain_q15=gain_q15,
                    beta_target_q15=0,
                    partial_amps_q15=self._harmonic_amps,
                )

            # Render through VM
            slot.vm_state, output = vm_step_block(slot.vm_state, frame)

            # Mix into output
            for i in range(block_size):
                L[i] += output[i]
                R[i] += output[i]

        # Clip to int16 range
        L = [max(-32768, min(32767, s)) for s in L]
        R = [max(-32768, min(32767, s)) for s in R]

        # 3. Material processing (if enabled)
        if self._enable_material:
            L, R = self.material.step_block(
                L, R, self._param_frame, self._material_id, 0
            )

        # 4. Safety limiting (if enabled)
        if self._enable_safety:
            L, R = self.safety.step_block(L, R, self._param_frame)

        # 5. Send health telemetry to cockpit
        telemetry = None
        if self._enable_cockpit:
            telemetry = HealthTelemetry(
                cpu_bar=0,
                mem_bar=0,
                stab_bar=32767,
                gear=5,
            )
            self.cockpit.receive_health_telemetry(telemetry)

        # 6. Accumulate for save_audio()
        self._audio_L.extend(L)
        self._audio_R.extend(R)

        return L, R, telemetry

    def _render_block_orchestrated(self) -> Tuple[List[int], List[int], Optional[HealthTelemetry]]:
        """Render using full orchestrator pipeline."""
        # Use orchestrator step_block
        L, R, schedule_result = self.orchestrator.step_block(
            self._param_frame,
            tail_stats=None,
        )

        # Material processing (if enabled)
        if self._enable_material:
            L, R = self.material.step_block(
                L, R, self._param_frame, self._material_id, 0
            )

        # Safety limiting (if enabled)
        if self._enable_safety:
            L, R = self.safety.step_block(L, R, self._param_frame)

        # Send health telemetry to cockpit
        if self._enable_cockpit:
            self.cockpit.receive_health_telemetry(schedule_result.telemetry)

        # Accumulate for save_audio()
        self._audio_L.extend(L)
        self._audio_R.extend(R)

        return L, R, schedule_result.telemetry

    def _compute_envelope_gain(self, slot: SimpleVoiceSlot) -> int:
        """Compute ADSR envelope gain for a voice slot."""
        attack_level_q15 = 32767
        sustain_level_q15 = int(self._sustain * 32767)

        if slot.envelope_phase == "attack":
            if slot.envelope_block < self._attack_blocks:
                t = slot.envelope_block / self._attack_blocks if self._attack_blocks > 0 else 1.0
                gain_q15 = int(attack_level_q15 * t)
                slot.envelope_block += 1
            else:
                slot.envelope_phase = "decay"
                slot.envelope_block = 0
                gain_q15 = attack_level_q15

        elif slot.envelope_phase == "decay":
            if slot.envelope_block < self._decay_blocks:
                t = slot.envelope_block / self._decay_blocks if self._decay_blocks > 0 else 1.0
                gain_q15 = int(attack_level_q15 - (attack_level_q15 - sustain_level_q15) * t)
                slot.envelope_block += 1
            else:
                slot.envelope_phase = "sustain"
                slot.envelope_block = 0
                gain_q15 = sustain_level_q15

        elif slot.envelope_phase == "sustain":
            gain_q15 = sustain_level_q15

        elif slot.envelope_phase == "release":
            if slot.envelope_block < self._release_blocks:
                t = slot.envelope_block / self._release_blocks if self._release_blocks > 0 else 1.0
                # Start from current gain, not sustain level
                start_gain = slot.gain_q15 if slot.gain_q15 > 0 else sustain_level_q15
                gain_q15 = int(start_gain * (1.0 - t))
                slot.envelope_block += 1
            else:
                slot.envelope_phase = "idle"
                gain_q15 = 0

        else:  # idle
            gain_q15 = 0

        return max(0, min(32767, gain_q15))

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
            self.cockpit.set_control(ControlID.SUS_Q15, int(self._sustain * 32767))
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
