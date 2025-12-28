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

from .envelope_scheduler import ADSRSpec
from .frames import CarrierModel, CarrierRole, TransitionKind


# =============================================================================
# CONSTANTS
# =============================================================================

Q15_ONE = 32767

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
# CARRIER CONFIG
# =============================================================================

@dataclass
class CarrierConfig:
    """Carrier oscillator configuration."""
    model: CarrierModel = CarrierModel.SAW
    role: CarrierRole = CarrierRole.AUDIBLE
    transition: TransitionKind = TransitionKind.RETRIGGER
    detune_cents: int = 0
    pulse_width_q15: int = 16384  # 0.5 for pulse wave

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
            pulse_width_q15=d.get('pulse_width_q15', 16384),
        )


# =============================================================================
# MATERIAL CONFIG
# =============================================================================

@dataclass
class MaterialConfig:
    """Material (dry/wet) blend configuration."""
    beta_start_q15: int = 0         # Start blend (0 = dry)
    beta_target_q15: int = 16384    # Target blend
    beta_ramp_blocks: int = 8       # Blocks to ramp

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MaterialConfig':
        """Parse material config from dict."""
        return cls(
            beta_start_q15=d.get('beta_start_q15', 0),
            beta_target_q15=d.get('beta_target_q15', 16384),
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
            tree_blend_q15=d.get('tree_blend_q15', 0),
            carrier=CarrierConfig.from_dict(carrier_dict),
        )


# =============================================================================
# COMPILED PRESET
# =============================================================================

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
                sustain=amp_dict.get('sustain', amp_dict.get('sustain_q15', 16384) / Q15_ONE),
                release_ms=amp_dict.get('release_ms', 100.0),
            )

        # Parse material
        material = MaterialConfig.from_dict(material_section)

        # Parse backbone
        backbone = []
        for op_entry in backbone_section:
            if isinstance(op_entry, dict):
                op_name = op_entry.get('op', 'IDENTITY').upper()
                strength = op_entry.get('strength_q15', Q15_ONE)
                # Also check for float strength (0.0-1.0)
                if 'strength' in op_entry:
                    strength = int(op_entry['strength'] * Q15_ONE)
                backbone.append(BackboneOp(
                    op=op_name,
                    strength_q15=strength,
                    param0_q15=op_entry.get('param0_q15', 0),
                    param1_q15=op_entry.get('param1_q15', 0),
                ))
            elif isinstance(op_entry, str):
                backbone.append(BackboneOp(op=op_entry.upper()))

        # Parse harmonic amplitudes (for saw carrier)
        harmonic_amps = preset_dict.get('carrier', {}).get('harmonic_amps_q15', None)
        if harmonic_amps is None:
            # Default saw harmonics: 1/n falloff
            harmonic_amps = [Q15_ONE // n for n in range(1, 9)]

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
        )

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

        # Validate backbone operators
        backbone = preset_dict.get('backbone', [])
        for i, op_entry in enumerate(backbone):
            if isinstance(op_entry, dict):
                if 'op' not in op_entry:
                    errors.append(f"Backbone entry {i} missing 'op' field")
            elif not isinstance(op_entry, str):
                errors.append(f"Backbone entry {i} must be string or dict")

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
# EXPORTS
# =============================================================================

__all__ = [
    'Q15_ONE',
    'BackboneOp',
    'CarrierConfig',
    'MaterialConfig',
    'LatchDefaults',
    'CompiledPreset',
    'PresetCompileError',
    'PresetCompiler',
]
