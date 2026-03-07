"""
Preset Compiler
===============

Compiles preset YAML/JSON into executable form for the VoiceFrameEmitter.

Per WSE-STD-PRESET v0:
- YAML is the authoring format (human-editable)
- JSON is the compiled canonical artifact (frozen, versioned)
- Compiler validates compatibility with VM and VoiceFrame standards

Usage:
    compiler = PresetCompiler()
    compiled = compiler.compile(preset_dict)
    # or
    compiled = compiler.compile_file("brass_stab_v0.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from ..envelope_scheduler import ADSRSpec
from ..frames import CarrierModel, CarrierRole, TransitionKind
from ..vm.harmonic_bank import LFOState, LFOWaveform, LFOSyncMode, LFOTarget, LFORouting

# Import canonical primitives (single source of truth)
from ..primitives import K_LADDER_MASKS, TailMode as _TailMode, Q15_ONE, Q15_HALF, Q16_ONE

# Import human-readable conversion helpers
from ..instrument_ir import float_or_q15, _auto_detect_ratios, _auto_detect_levels

# Import central registry for validation
from .central_registry import get_central_registry


# =============================================================================
# CONSTANTS
# =============================================================================

# Supported standards
SUPPORTED_VOICEFRAME_STD = ["WSE-STD-020 v1"]
SUPPORTED_VM_STD = ["WSE-STD-VM-HYBRID v0"]
SUPPORTED_OPLIB_ID = ["WSE-OPLIB-24 v0"]


# =============================================================================
# BACKBONE OP
# =============================================================================

@dataclass
class BackboneOp:
    """A backbone operator with strength."""
    op: str                     # Operator name (e.g., "TILT", "DIFFUSE_DOWN")
    strength_q15: int = Q15_ONE # Strength (0-32767)
    param0_q15: int = 0         # Optional parameter 0
    param1_q15: int = 0         # Optional parameter 1


# =============================================================================
# FM6 CONFIG
# =============================================================================

@dataclass
class FM6Config:
    """
    FM6 synthesis configuration.

    Contains all parameters needed to configure the 6-operator FM engine
    from a YAML preset.
    """
    algorithm_id: int = 1                   # DX7 algorithm (1-32)
    op_ratios: list = field(default_factory=lambda: [Q16_ONE]*6)   # 6 × Q16 ratios
    op_levels: list = field(default_factory=lambda: [Q15_ONE]*6)   # 6 × Q15 levels
    feedback_depth: int = 0                 # DX7 feedback level (0-7)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FM6Config':
        """Parse FM6 config from dict.

        Supports both raw Q15/Q16 integer format and human-readable floats.
        Float ratios (e.g., [1.0, 2.0, 3.5]) are auto-detected and converted to Q16.
        Float levels (e.g., [1.0, 0.75, 0.5]) are auto-detected and converted to Q15.
        """
        # Support both 'algorithm_id' and 'algorithm' keys
        algorithm_id = d.get('algorithm_id', d.get('algorithm', 1))

        # Parse operator ratios with auto-detection (Q16: Q16_ONE = 1.0)
        op_ratios = _auto_detect_ratios(d.get('op_ratios', [Q16_ONE]*6))
        if len(op_ratios) < 6:
            op_ratios = op_ratios + [Q16_ONE] * (6 - len(op_ratios))

        # Parse operator levels with auto-detection (Q15: Q15_ONE = 1.0)
        op_levels = _auto_detect_levels(d.get('op_levels', [Q15_ONE]*6))
        if len(op_levels) < 6:
            op_levels = op_levels + [Q15_ONE] * (6 - len(op_levels))

        # Support both 'feedback_depth' and 'feedback' keys
        feedback_depth = d.get('feedback_depth', d.get('feedback', 0))

        return cls(
            algorithm_id=algorithm_id,
            op_ratios=op_ratios[:6],
            op_levels=op_levels[:6],
            feedback_depth=feedback_depth,
        )


# =============================================================================
# CARRIER CONFIG
# =============================================================================

@dataclass
class CarrierConfig:
    """Carrier oscillator configuration."""
    model: CarrierModel = CarrierModel.SAW
    role: CarrierRole = CarrierRole.AUDIBLE
    transition: TransitionKind = TransitionKind.RETRIGGER
    detune_cents: int = 0
    pulse_width_q15: int = Q15_HALF  # 0.5 for pulse wave

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CarrierConfig':
        """Parse carrier config from dict."""
        model_str = d.get('model', 'SAW').upper()
        role_str = d.get('role', 'AUDIBLE').upper()
        transition_str = d.get('transition', 'RETRIGGER').upper()

        # Map string to enum
        model_map = {
            'SINE': CarrierModel.SINE,
            'SAW': CarrierModel.SAW,
            'PULSE': CarrierModel.PULSE,
            'TRI': CarrierModel.TRI,
            'PARTIALS': CarrierModel.PARTIALS,
            'NOISE': CarrierModel.NOISE,
        }
        role_map = {
            'OFF': CarrierRole.OFF,
            'AUDIBLE': CarrierRole.AUDIBLE,
            'EXCITER_ONLY': CarrierRole.EXCITER_ONLY,
        }
        transition_map = {
            'RETRIGGER': TransitionKind.RETRIGGER,
            'GLIDE': TransitionKind.GLIDE,
            'BRIDGE': TransitionKind.BRIDGE,
        }

        return cls(
            model=model_map.get(model_str, CarrierModel.SAW),
            role=role_map.get(role_str, CarrierRole.AUDIBLE),
            transition=transition_map.get(transition_str, TransitionKind.RETRIGGER),
            detune_cents=d.get('detune_cents', 0),
            pulse_width_q15=float_or_q15(d, 'pulse_width', 'pulse_width_q15', Q15_HALF),
        )


# =============================================================================
# MATERIAL CONFIG
# =============================================================================

@dataclass
class MaterialConfig:
    """Material (dry/wet) blend configuration."""
    beta_start_q15: int = 0         # Start blend (0 = dry)
    beta_target_q15: int = Q15_HALF    # Target blend
    beta_ramp_blocks: int = 8       # Blocks to ramp

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MaterialConfig':
        """Parse material config from dict.

        Supports both raw Q15 keys (e.g., beta_target_q15: 16384)
        and human-readable float keys (e.g., beta_target: 0.5).
        Float keys take priority when both are present.
        """
        return cls(
            beta_start_q15=float_or_q15(d, 'beta_start', 'beta_start_q15', 0),
            beta_target_q15=float_or_q15(d, 'beta_target', 'beta_target_q15', Q15_HALF),
            beta_ramp_blocks=d.get('beta_ramp_blocks', 8),
        )


# =============================================================================
# LATCH DEFAULTS
# =============================================================================

@dataclass
class LatchDefaults:
    """Default values for VoiceFrame fields (latched per-voice)."""
    pattern_id: int = 0
    tail_mode: int = 0  # 0=FULL, 1=HALF, 2=MIN
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    carrier: CarrierConfig = field(default_factory=CarrierConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LatchDefaults':
        """Parse latch defaults from dict."""
        tail_mode_str = d.get('tail_mode', 'FULL').upper()
        tail_mode_map = {'FULL': 0, 'HALF': 1, 'MIN': 2, 'EMERGENCY': 3}

        carrier_dict = d.get('carrier', {})

        return cls(
            pattern_id=d.get('pattern_id', 0),
            tail_mode=tail_mode_map.get(tail_mode_str, 0),
            tree_a_id=d.get('tree_a_id', 0),
            tree_b_id=d.get('tree_b_id', 0),
            tree_blend_q15=float_or_q15(d, 'tree_blend', 'tree_blend_q15', 0),
            carrier=CarrierConfig.from_dict(carrier_dict),
        )


# =============================================================================
# COMPILED PRESET
# =============================================================================

@dataclass
class CmmmDefaults:
    """
    CMMM runtime control defaults (WSE-STD-070 §10).

    These values are emitted to VoiceFrame when the voice is active.
    They control the executor latch for K degradation and tail operators.
    """
    node_mask_u32: int = 0xFFFFFFFF      # Target node mask (K=32 default)
    mask_ramp_blocks: int = 5             # Crossfade duration
    tail_mode: int = 0                    # 0=FULL, 1=MID, 2=MIN, 3=EMERGENCY
    traction_q15: int = Q15_ONE           # Projection hardness

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CmmmDefaults':
        """Parse from dictionary."""
        if d is None:
            return cls()

        # Pattern table for resolving pattern:N references - use primitives.K_LADDER_MASKS
        pattern_table = K_LADDER_MASKS
        # Tail mode map - use primitives.TailMode values
        tail_mode_map = {
            "FULL": _TailMode.FULL,
            "MID": _TailMode.MID,
            "MIN": _TailMode.MIN,
            "EMERGENCY": _TailMode.EMERGENCY,
        }

        # Resolve node_mask (can be int, hex string, or "pattern:N")
        node_mask = d.get('node_mask', 0xFFFFFFFF)
        if isinstance(node_mask, str):
            if node_mask.startswith("pattern:"):
                pattern_id = int(node_mask.split(":")[1])
                node_mask = pattern_table.get(pattern_id, 0xFFFFFFFF)
            elif node_mask.startswith("0x"):
                node_mask = int(node_mask, 16)
            else:
                node_mask = 0xFFFFFFFF
        node_mask_u32 = int(node_mask) & 0xFFFFFFFF

        # Resolve tail_mode (can be int or name)
        tail_mode = d.get('tail_mode', 0)
        if isinstance(tail_mode, str):
            tail_mode = tail_mode_map.get(tail_mode.upper(), 0)

        return cls(
            node_mask_u32=node_mask_u32,
            mask_ramp_blocks=d.get('mask_ramp_blocks', 5),
            tail_mode=int(tail_mode),
            traction_q15=float_or_q15(d, 'traction', 'traction_q15', Q15_ONE),
        )


@dataclass
class CompiledPreset:
    """
    Output of preset compilation.

    Contains all data needed by VoiceFrameEmitter to produce VoiceFrames.
    """
    # Metadata
    preset_id: str
    preset_version: str
    author: str
    description: str

    # Compatibility
    voiceframe_std: str
    vm_std: str
    operator_lib_id: str

    # Latch defaults
    latch: LatchDefaults

    # Envelope
    amp_env: Optional[ADSRSpec]

    # Material ramp
    material: MaterialConfig

    # Backbone operators (ordered)
    backbone: List[BackboneOp]

    # Harmonic amplitudes for saw/partial carriers
    harmonic_amps_q15: List[int] = field(default_factory=lambda: [32767, 16383, 10922, 8191, 6553, 5461, 4681, 4096])

    # FM6 configuration (optional - if present, enables FM6 mode)
    fm6: Optional[FM6Config] = None

    # Oscillator mode (0=ModeA, 1=ModeB, 2=Sample, 3=FM6)
    # Auto-set to 3 when fm6 section is present
    osc_mode: int = 0

    # CMMM runtime control defaults (WSE-STD-070 §10)
    cmmm: CmmmDefaults = field(default_factory=CmmmDefaults)

    # PWM width default (for Mode B pulse)
    pwm_width_q15: int = Q15_HALF

    # LFO modulation state (Phase 2 Mod Matrix)
    lfos: List[LFOState] = field(default_factory=list)

    def backbone_ops(self) -> List[str]:
        """Get list of enabled operator names."""
        return [op.op for op in self.backbone]

    def backbone_families(self) -> Dict[str, float]:
        """Get operator name -> strength (0.0-1.0) mapping."""
        return {op.op.lower(): op.strength_q15 / Q15_ONE for op in self.backbone}


# =============================================================================
# PRESET COMPILER
# =============================================================================

class PresetCompileError(Exception):
    """Raised when preset compilation fails."""
    pass


class PresetCompiler:
    """
    Compiles preset YAML/JSON into executable form.

    Example:
        compiler = PresetCompiler()
        compiled = compiler.compile_file("brass_stab_v0.yaml")
    """

    def compile(self, preset_dict: Dict[str, Any]) -> CompiledPreset:
        """
        Compile a preset dictionary into CompiledPreset.

        Args:
            preset_dict: Parsed preset data

        Returns:
            CompiledPreset ready for VoiceFrameEmitter

        Raises:
            PresetCompileError: If preset is invalid or incompatible
        """
        # Validate structure
        errors = self.validate(preset_dict)
        if errors:
            raise PresetCompileError(f"Preset validation failed: {errors}")

        # Extract sections
        preset_section = preset_dict.get('preset', {})
        compat_section = preset_dict.get('compat', {})
        latch_section = preset_dict.get('latch_defaults', {})
        env_section = preset_dict.get('env', {})
        material_section = preset_dict.get('material', {})
        backbone_section = preset_dict.get('backbone', [])

        # Parse metadata
        preset_id = preset_section.get('preset_id', 'UNKNOWN')
        preset_version = preset_section.get('preset_version', 'v0')
        author = preset_section.get('author', 'unknown')
        description = preset_section.get('description', '')

        # Parse compatibility
        voiceframe_std = compat_section.get('voiceframe_std', 'WSE-STD-020 v1')
        vm_std = compat_section.get('vm_std', 'WSE-STD-VM-HYBRID v0')
        operator_lib_id = compat_section.get('operator_lib_id', 'WSE-OPLIB-24 v0')

        # Parse latch defaults
        latch = LatchDefaults.from_dict(latch_section)

        # Parse envelope
        amp_env = None
        if 'amp' in env_section:
            amp_dict = env_section['amp']
            amp_env = ADSRSpec(
                attack_ms=amp_dict.get('attack_ms', 10.0),
                decay_ms=amp_dict.get('decay_ms', 100.0),
                sustain=amp_dict.get('sustain', amp_dict.get('sustain_q15', Q15_HALF) / Q15_ONE),
                release_ms=amp_dict.get('release_ms', 100.0),
            )

        # Parse material
        material = MaterialConfig.from_dict(material_section)

        # Parse backbone
        backbone = []
        for op_entry in backbone_section:
            if isinstance(op_entry, dict):
                op_name = op_entry.get('op', 'IDENTITY').upper()
                strength = float_or_q15(op_entry, 'strength', 'strength_q15', Q15_ONE)
                backbone.append(BackboneOp(
                    op=op_name,
                    strength_q15=strength,
                    param0_q15=float_or_q15(op_entry, 'param0', 'param0_q15', 0),
                    param1_q15=float_or_q15(op_entry, 'param1', 'param1_q15', 0),
                ))
            elif isinstance(op_entry, str):
                backbone.append(BackboneOp(op=op_entry.upper()))

        # Parse harmonic amplitudes (for saw carrier)
        harmonic_amps = preset_dict.get('carrier', {}).get('harmonic_amps_q15', None)
        if harmonic_amps is None:
            # Default saw harmonics: 1/n falloff
            harmonic_amps = [Q15_ONE // n for n in range(1, 9)]

        # Parse FM6 section (optional - enables FM6 mode when present)
        fm6_section = preset_dict.get('fm6', None)
        fm6_config = None
        osc_mode = 0

        if fm6_section is not None:
            fm6_config = FM6Config.from_dict(fm6_section)
            osc_mode = 3  # OSC_MODE_FM6

        # Parse CMMM runtime control (WSE-STD-070 §10)
        cmmm_section = preset_dict.get('cmmm', None)
        cmmm = CmmmDefaults.from_dict(cmmm_section)

        # Parse PWM width from carrier section
        carrier_section = preset_dict.get('carrier', {})
        pwm_width_q15 = float_or_q15(carrier_section, 'pwm_width', 'pwm_width_q15', Q15_HALF)

        # Parse LFO modulators (Phase 2 Mod Matrix)
        lfos = self._parse_lfos(preset_dict)

        return CompiledPreset(
            preset_id=preset_id,
            preset_version=preset_version,
            author=author,
            description=description,
            voiceframe_std=voiceframe_std,
            vm_std=vm_std,
            operator_lib_id=operator_lib_id,
            latch=latch,
            amp_env=amp_env,
            material=material,
            backbone=backbone,
            harmonic_amps_q15=harmonic_amps,
            fm6=fm6_config,
            osc_mode=osc_mode,
            cmmm=cmmm,
            pwm_width_q15=pwm_width_q15,
            lfos=lfos,
        )

    def _parse_lfos(self, preset_dict: Dict[str, Any]) -> List[LFOState]:
        """
        Parse LFO definitions from modulators section.

        Expected YAML format:
            modulators:
              lfos:
                - id: "vibrato"
                  waveform: "sine"
                  rate_hz: 5.5
                  depth_q15: 8192
                  delay_ms: 300
                  ramp_ms: 200
                  sync_mode: "free"
                  targets:
                    - type: "pitch"
                      amount: 1.0

        Returns:
            List of LFOState instances
        """
        lfos = []
        modulators_section = preset_dict.get('modulators', {})
        lfos_list = modulators_section.get('lfos', [])

        # Waveform name to enum mapping
        waveform_map = {
            'sine': LFOWaveform.SINE,
            'triangle': LFOWaveform.TRIANGLE,
            'saw_up': LFOWaveform.SAW_UP,
            'saw': LFOWaveform.SAW_UP,
            'square': LFOWaveform.SQUARE,
        }

        # Sync mode name to enum mapping
        sync_mode_map = {
            'free': LFOSyncMode.FREE,
            'gate_reset': LFOSyncMode.GATE_RESET,
            'legato_reset': LFOSyncMode.LEGATO_RESET,
        }

        # Target name to enum mapping
        target_map = {
            'pitch': LFOTarget.PITCH,
            'gain': LFOTarget.GAIN,
            'pwm_width': LFOTarget.PWM_WIDTH,
            'pwm': LFOTarget.PWM_WIDTH,
            'tilt': LFOTarget.TILT,
            'brightness': LFOTarget.TILT,
            'harmonic_rolloff': LFOTarget.TILT,  # Alias for tilt
            'filter_cutoff': LFOTarget.FILTER_CUTOFF,
            'cutoff': LFOTarget.FILTER_CUTOFF,
            'filter_reso': LFOTarget.FILTER_RESO,
            'resonance': LFOTarget.FILTER_RESO,
            'band_scale': LFOTarget.BAND_SCALE,
        }

        # Assign arc slots sequentially (MAX_ARCS = 8)
        arc_slot = 0

        for lfo_def in lfos_list:
            # Parse waveform
            waveform_str = lfo_def.get('waveform', 'sine').lower()
            waveform = waveform_map.get(waveform_str, LFOWaveform.SINE)

            # Parse sync mode
            sync_mode_str = lfo_def.get('sync_mode', 'free').lower()
            sync_mode = sync_mode_map.get(sync_mode_str, LFOSyncMode.FREE)

            # Parse targets
            routings = []
            for target_def in lfo_def.get('targets', []):
                target_str = target_def.get('type', 'pitch').lower()
                target = target_map.get(target_str, LFOTarget.PITCH)
                amount = target_def.get('amount', 1.0)
                amount_q15 = int(amount * Q15_ONE)
                routings.append(LFORouting(
                    target=target,
                    amount_q15=amount_q15,
                ))

            # Resolve depth with float alternative
            depth = float_or_q15(lfo_def, 'depth', 'depth_q15', Q15_HALF)

            # Create LFO state with arc_slot for unified phase storage
            lfo = LFOState(
                id=lfo_def.get('id', ''),
                waveform=waveform,
                sync_mode=sync_mode,
                routings=routings,
                depth_q15=depth,
                arc_slot=arc_slot,
            )

            # Configure timing (includes arc_slot)
            lfo.configure_lfo(
                rate_hz=lfo_def.get('rate_hz', 5.0),
                depth_q15=depth,
                waveform=waveform,
                sync_mode=sync_mode,
                delay_ms=lfo_def.get('delay_ms', 0.0),
                ramp_ms=lfo_def.get('ramp_ms', 0.0),
                routings=routings,
                arc_slot=arc_slot,
            )

            lfos.append(lfo)
            arc_slot += 1

            # Warn if exceeding MAX_ARCS (8 slots available)
            if arc_slot > 8:
                import warnings
                warnings.warn(f"LFO '{lfo.id}' exceeds MAX_ARCS=8 slots, phase storage may conflict")

        return lfos

    def compile_file(self, filepath: str) -> CompiledPreset:
        """
        Compile a preset from a YAML or JSON file.

        Args:
            filepath: Path to preset file

        Returns:
            CompiledPreset

        Raises:
            PresetCompileError: If file cannot be loaded or compiled
        """
        path = Path(filepath)

        if not path.exists():
            raise PresetCompileError(f"File not found: {filepath}")

        try:
            content = path.read_text(encoding='utf-8')

            if path.suffix in ('.yaml', '.yml'):
                if not HAS_YAML:
                    raise PresetCompileError("PyYAML not installed. Install with: pip install pyyaml")
                preset_dict = yaml.safe_load(content)
            elif path.suffix == '.json':
                preset_dict = json.loads(content)
            else:
                raise PresetCompileError(f"Unsupported file format: {path.suffix}")

            return self.compile(preset_dict)

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise PresetCompileError(f"Failed to parse {filepath}: {e}")

    def validate(self, preset_dict: Dict[str, Any]) -> List[str]:
        """
        Validate preset dictionary structure.

        Uses CentralRegistry to validate resource references.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check required sections
        if 'preset' not in preset_dict:
            errors.append("Missing 'preset' section")

        # Check compatibility
        compat = preset_dict.get('compat', {})
        voiceframe_std = compat.get('voiceframe_std', 'WSE-STD-020 v1')
        vm_std = compat.get('vm_std', 'WSE-STD-VM-HYBRID v0')

        if voiceframe_std not in SUPPORTED_VOICEFRAME_STD:
            errors.append(f"Unsupported voiceframe_std: {voiceframe_std}")

        if vm_std not in SUPPORTED_VM_STD:
            errors.append(f"Unsupported vm_std: {vm_std}")

        # Validate backbone operators using registry
        registry = get_central_registry()
        backbone = preset_dict.get('backbone', [])
        for i, op_entry in enumerate(backbone):
            if isinstance(op_entry, dict):
                if 'op' not in op_entry:
                    errors.append(f"Backbone entry {i} missing 'op' field")
                else:
                    op_name = op_entry['op']
                    if registry.operators.get_by_name(op_name) is None:
                        # Not a hard error - custom operators may be defined elsewhere
                        pass
            elif not isinstance(op_entry, str):
                errors.append(f"Backbone entry {i} must be string or dict")

        # Validate tree references using registry
        latch = preset_dict.get('latch_defaults', {})
        tree_a_id = latch.get('tree_a_id')
        tree_b_id = latch.get('tree_b_id')

        if tree_a_id is not None and registry.trees.get(tree_a_id) is None:
            errors.append(f"Tree A ID {tree_a_id} not found in registry")

        if tree_b_id is not None and registry.trees.get(tree_b_id) is None:
            errors.append(f"Tree B ID {tree_b_id} not found in registry")

        # Validate pattern references
        pattern_id = latch.get('pattern_id')
        if pattern_id is not None and registry.patterns.get(pattern_id) is None:
            # Not a hard error - K ladder patterns are in primitives.K_LADDER_MASKS
            if pattern_id not in K_LADDER_MASKS:
                errors.append(f"Pattern ID {pattern_id} not found in registry or K ladder")

        return errors

    def validate_compatibility(self, preset_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if preset is compatible with current runtime.

        Returns:
            (is_compatible, list of warnings)
        """
        warnings = []
        compat = preset_dict.get('compat', {})

        voiceframe_std = compat.get('voiceframe_std', 'WSE-STD-020 v1')
        vm_std = compat.get('vm_std', 'WSE-STD-VM-HYBRID v0')
        oplib_id = compat.get('operator_lib_id', 'WSE-OPLIB-24 v0')

        is_compatible = True

        if voiceframe_std not in SUPPORTED_VOICEFRAME_STD:
            warnings.append(f"VoiceFrame standard '{voiceframe_std}' not supported")
            is_compatible = False

        if vm_std not in SUPPORTED_VM_STD:
            warnings.append(f"VM standard '{vm_std}' not supported")
            is_compatible = False

        if oplib_id not in SUPPORTED_OPLIB_ID:
            warnings.append(f"Operator library '{oplib_id}' may not be fully supported")
            # Don't fail on this - may still work

        return is_compatible, warnings


# =============================================================================
# PRESET VALIDATION (instrument_ir_v0 format)
# =============================================================================

def validate_preset(data: Dict[str, Any]) -> List[str]:
    """
    Validate a preset dictionary for common authoring errors.

    Works with both the flat 'preset:' format (delegates to PresetCompiler.validate)
    and the 'instrument_ir_v0:' format (checks node/edge consistency, FM6, macros).

    Returns:
        List of error/warning strings (empty if valid)
    """
    errors = []

    # Flat preset format
    if 'preset' in data:
        compiler = PresetCompiler()
        return compiler.validate(data)

    # instrument_ir_v0 format
    ir_data = data.get('instrument_ir_v0', data)
    layers = ir_data.get('layers', {})
    l0 = layers.get('L0_graph', {})

    # Check nodes and edges consistency
    nodes = l0.get('nodes', [])
    edges = l0.get('edges', [])
    node_ids = {n.get('id', '') for n in nodes if isinstance(n, dict)}

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        src = edge.get('src', '')
        dst = edge.get('dst', '')
        if src and src not in node_ids:
            errors.append(f"Edge {i}: src '{src}' not found in nodes")
        if dst and dst not in node_ids:
            errors.append(f"Edge {i}: dst '{dst}' not found in nodes")

    # Check macro defaults in range
    macros = l0.get('macros', [])
    for macro in macros:
        if not isinstance(macro, dict):
            continue
        default = macro.get('default')
        name = macro.get('id', macro.get('name', '?'))
        if default is not None:
            if not isinstance(default, (int, float)):
                errors.append(f"Macro '{name}': default must be a number")
            elif default < 0.0 or default > 1.0:
                errors.append(f"Macro '{name}': default {default} outside [0.0, 1.0]")

    # Check FM6 operator count
    for node in nodes:
        if not isinstance(node, dict):
            continue
        params = node.get('params', {})
        if node.get('type') == 'FM6':
            # Check operators format
            if 'operators' in params:
                ops = params['operators']
                if len(ops) != 6:
                    errors.append(f"FM6 node '{node.get('id', '?')}': expected 6 operators, got {len(ops)}")
            # Check ratios/levels count
            for key in ('op_ratios', 'op_levels'):
                vals = params.get(key)
                if vals is not None and len(vals) != 6:
                    errors.append(f"FM6 node '{node.get('id', '?')}': {key} has {len(vals)} entries, expected 6")
            # Check ratios are positive
            ratios = params.get('op_ratios', params.get('operators', []))
            if isinstance(ratios, list):
                for j, r in enumerate(ratios):
                    val = r.get('ratio', 1.0) if isinstance(r, dict) else r
                    if isinstance(val, (int, float)) and val < 0:
                        errors.append(f"FM6 node '{node.get('id', '?')}': op_ratios[{j}] is negative")

        # Check ADSR values are non-negative
        adsr = params.get('adsr', {})
        if isinstance(adsr, dict):
            for adsr_key in ('a_ms', 'd_ms', 's', 'r_ms'):
                val = adsr.get(adsr_key)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    errors.append(f"Node '{node.get('id', '?')}': adsr.{adsr_key} is negative ({val})")

    return errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Q15_ONE',
    'BackboneOp',
    'FM6Config',
    'CarrierConfig',
    'MaterialConfig',
    'LatchDefaults',
    'CompiledPreset',
    'PresetCompileError',
    'PresetCompiler',
    'validate_preset',
]
