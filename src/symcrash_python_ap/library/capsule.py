"""
Capsule Object Model (WSE-STD-CAPSULE-COMPAT v0 §2)
====================================================

Portable capsule objects for PresetLibrary storage and runtime loading.

Capsule Types (v0):
1. CapsuleInstrument - ProgramPreset + optional InstrumentIR references
2. CapsuleSynth - ProgramPreset + synth-focused defaults
3. CapsuleAsset - References only; no control semantics
4. CapsuleMacro - Performance macro (vibrato, legato, tremolo, portamento)

Key design choices:
- Library stores capsules
- Pipeline owns morphing
- Capsules include control/macro defaults and engine defaults
- Capsules MUST NOT contain runtime histories (coeff_state, phase, PRNG, etc.)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


# =============================================================================
# ENUMS
# =============================================================================

class CapsuleKind(IntEnum):
    """Capsule type identifier (§1)."""
    INSTRUMENT = 0
    SYNTH = 1
    ASSET = 2
    MACRO = 3  # Performance macro (vibrato, legato, tremolo, portamento)


class AssetType(IntEnum):
    """Asset reference types (§2)."""
    IR = 0
    WAVETABLE = 1
    MATERIAL = 2
    TABLE_ATLAS = 3
    OTHER = 255


class MacroType(IntEnum):
    """Macro type for CapsuleKind.MACRO capsules."""
    VIBRATO = 0      # Pitch/amplitude/timbre LFO modulation
    LEGATO = 1       # Smooth pitch transition mode (preserves phase)
    TREMOLO = 2      # Amplitude-only LFO modulation
    PORTAMENTO = 3   # Pitch glide between notes


class SystemMacroID(IntEnum):
    """
    System macro IDs per WSE-STD-MACRO v0 §2.

    These are semantic control axes for CMMM (Controlled Markov Mealy Machine).
    They represent expressive dimensions of instrument performance.

    Ownership:
    - ENERGY_STATE (0): Envelope-owned, NEVER set by controllers
    - AGE (7): CMMM integrator, evolves deterministically
    - All others: Controller → CMMM
    """
    # Envelope-owned state (macro 0)
    ENERGY_STATE = 0     # Envelope phase (0=silent, 32767=peak)

    # Controller → CMMM axes (macros 1-6, 8-10)
    ENERGY_INTENT = 1    # Playing intensity / attack character
    MOTION = 2           # Movement / modulation amount (gesture tracking)
    FOCUS = 3            # Attention / emphasis (spectral concentration)
    BRIGHTNESS = 4       # Timbral brightness (harmonic content)
    DENSITY = 5          # Body / thickness (low-mid presence)
    RISK = 6             # Chaos / instability (stochastic behavior)

    # CMMM-owned integrator (macro 7)
    AGE = 7              # Voice age (accumulates over note duration)

    # Controller → CMMM axes continued (macros 8-10)
    SPACE = 8            # Spatial / reverb character
    MATERIAL = 9         # Surface texture (wood, metal, glass)
    UNITY = 10           # Coherence / blend (how much voices blend)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EngineCompat:
    """
    Engine compatibility requirements (§2).

    Capsule MUST declare ABI versions for compatibility checking.
    """
    control_registry_abi: int = 0
    voice_registry_abi: int = 0
    voiceframe_abi: int = 0
    sample_rate_hz: int = 48000
    block_size: int = 64


@dataclass
class CapsuleIdentity:
    """
    Capsule identity metadata (§2).

    Unique identification and human-readable info.
    """
    capsule_id: str
    kind: CapsuleKind
    name: str
    author: str = ""
    tags: List[str] = field(default_factory=list)
    engine_compat: EngineCompat = field(default_factory=EngineCompat)


@dataclass
class ProgramReference:
    """
    Program identity (§2).

    Either program_id OR instrument_ir_id must be set.
    """
    program_id: Optional[int] = None
    instrument_ir_id: Optional[str] = None
    regime_graph_id: Optional[int] = None
    tree_family_id: Optional[int] = None


@dataclass
class AssetRef:
    """
    Asset reference (§2).

    Points to external resources like IRs, wavetables, materials.
    """
    asset_type: AssetType
    asset_id: str
    asset_hash: Optional[str] = None
    asset_path: Optional[str] = None


@dataclass
class SeedConfig:
    """
    Seed configuration for determinism (§2).

    Controls PRNG initialization for deterministic playback.
    """
    master_seed: int = 0x12345678
    derivation_rule_id: int = 0  # 0 = deterministic


@dataclass
class MacroProgram:
    """
    Macro operator program for CapsuleKind.MACRO capsules (§2.4).

    Defines a performance macro (vibrato, legato, tremolo, portamento)
    as a portable capsule that can be applied to any instrument.

    MACRO capsules do NOT require program_id or instrument_ir_id.
    They define modulation behavior applied at runtime.
    """
    macro_type: MacroType = MacroType.VIBRATO

    # LFO parameters (VIBRATO, TREMOLO)
    rate_hz: float = 5.5              # LFO rate in Hz
    depth_pitch_cents: int = 0        # Pitch modulation depth in cents
    depth_amp_q15: int = 0            # Amplitude modulation depth (Q15)
    depth_timbre_q15: int = 0         # Timbre modulation depth (Q15)

    # Envelope parameters
    delay_ms: float = 0.0             # Delay before modulation onset
    ramp_ms: float = 0.0              # Ramp time to full depth

    # Pitch transition parameters (LEGATO, PORTAMENTO)
    glide_time_ms: float = 0.0        # Pitch glide duration
    glide_curve: int = 0              # 0=linear, 1=exp, 2=log

    # Phase behavior (LEGATO)
    preserve_phase: bool = False      # Keep oscillator phase on note change


@dataclass
class ControlProgram:
    """
    Control axis defaults per WSE-STD-MACRO v0 §2.

    Defines default values for semantic control axes that presets can specify.
    These values are pushed to MacroRuntime at preset load time.

    All values are Q15 unsigned (0-32767) representing 0.0-1.0 normalized.

    Usage:
        control = ControlProgram(brightness_q15=24576, density_q15=16384)
        # Push to runtime at preset load
        runtime.apply_control_defaults(control)

    Semantic meanings:
    - energy_intent: Playing intensity / attack character
    - motion: Movement / modulation amount (gesture tracking)
    - focus: Attention / emphasis (spectral concentration)
    - brightness: Timbral brightness (harmonic content)
    - density: Body / thickness (low-mid presence)
    - risk: Chaos / instability (stochastic behavior)
    - space: Spatial / reverb character
    - material: Surface texture (wood=low, metal=mid, glass=high)
    - unity: Coherence / blend (how much voices blend together)
    """
    # Controller → CMMM axes (Q15 unsigned, 0=minimum, 32767=maximum)
    energy_intent_q15: int = 16384    # 0.5 = neutral intensity
    motion_q15: int = 0               # 0.0 = no modulation
    focus_q15: int = 16384            # 0.5 = balanced emphasis
    brightness_q15: int = 16384       # 0.5 = neutral brightness
    density_q15: int = 16384          # 0.5 = neutral body
    risk_q15: int = 0                 # 0.0 = stable, no chaos
    space_q15: int = 8192             # 0.25 = subtle spatial
    material_q15: int = 16384         # 0.5 = neutral (neither wood nor glass)
    unity_q15: int = 16384            # 0.5 = balanced blend

    def to_macro_dict(self) -> Dict[int, int]:
        """
        Convert to macro ID -> value dictionary for MacroRuntime.

        Returns:
            Dict mapping SystemMacroID -> Q15 value
        """
        return {
            SystemMacroID.ENERGY_INTENT: self.energy_intent_q15,
            SystemMacroID.MOTION: self.motion_q15,
            SystemMacroID.FOCUS: self.focus_q15,
            SystemMacroID.BRIGHTNESS: self.brightness_q15,
            SystemMacroID.DENSITY: self.density_q15,
            SystemMacroID.RISK: self.risk_q15,
            SystemMacroID.SPACE: self.space_q15,
            SystemMacroID.MATERIAL: self.material_q15,
            SystemMacroID.UNITY: self.unity_q15,
        }


@dataclass
class CapsuleDefaults:
    """
    Default value overrides (§3-4).

    Overrides ONLY default values, never:
    - control ranges
    - value types
    - slew policy / slew param
    - binding targets
    """
    # control_id -> default value (§3)
    control_defaults: Dict[int, int] = field(default_factory=dict)

    # param_id -> default value (§4)
    voice_param_defaults: Dict[int, int] = field(default_factory=dict)


# =============================================================================
# CAPSULE V0
# =============================================================================

@dataclass
class CapsuleV0:
    """
    Complete capsule object per WSE-STD-CAPSULE-COMPAT v0 §2.

    Portable "instrument identity" that can be:
    - stored in a PresetLibrary
    - compiled-to by tools
    - loaded by the runtime pipeline
    - applied without ambiguity across engines

    MUST NOT contain runtime histories (coeff_state, phase, ADSR state, PRNG state).
    """
    identity: CapsuleIdentity
    program: ProgramReference = field(default_factory=ProgramReference)
    defaults: CapsuleDefaults = field(default_factory=CapsuleDefaults)
    assets: List[AssetRef] = field(default_factory=list)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    macro: Optional[MacroProgram] = None  # Only for CapsuleKind.MACRO
    control: Optional[ControlProgram] = None  # Control axis defaults for INSTRUMENT/SYNTH

    def validate(self, strict: bool = False) -> List[str]:
        """
        Validate capsule structure.

        Args:
            strict: If True, unknown IDs cause errors. If False, warnings only.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Identity validation
        if not self.identity.capsule_id:
            issues.append("ERROR: capsule_id is required")

        if not self.identity.name:
            issues.append("ERROR: name is required")

        # Program validation (not required for MACRO or ASSET kinds)
        if self.identity.kind in (CapsuleKind.INSTRUMENT, CapsuleKind.SYNTH):
            if self.program.program_id is None and self.program.instrument_ir_id is None:
                issues.append("ERROR: Either program_id or instrument_ir_id must be set")

            if self.program.program_id is not None and self.program.instrument_ir_id is not None:
                issues.append("WARNING: Both program_id and instrument_ir_id set; program_id takes precedence")

        # Macro validation
        if self.identity.kind == CapsuleKind.MACRO:
            if self.macro is None:
                issues.append("ERROR: MACRO capsule requires macro field")
        elif self.macro is not None:
            issues.append("WARNING: macro field set on non-MACRO capsule; will be ignored")

        # Asset validation
        for asset in self.assets:
            if not asset.asset_id:
                issues.append("ERROR: Asset reference missing asset_id")

        return issues

    @property
    def is_instrument(self) -> bool:
        """Check if capsule is an instrument type."""
        return self.identity.kind == CapsuleKind.INSTRUMENT

    @property
    def is_synth(self) -> bool:
        """Check if capsule is a synth type."""
        return self.identity.kind == CapsuleKind.SYNTH

    @property
    def is_asset(self) -> bool:
        """Check if capsule is an asset type."""
        return self.identity.kind == CapsuleKind.ASSET

    @property
    def is_macro(self) -> bool:
        """Check if capsule is a macro type."""
        return self.identity.kind == CapsuleKind.MACRO


# =============================================================================
# INTERNAL FACTORY FUNCTIONS
# =============================================================================
#
# NOTE: These functions are for internal use by the compiler and library.
# For human authoring, use InstrumentIR instead:
#
#   from symcrash_python_ap.instrument_ir import InstrumentIR
#
#   # Quick preset:
#   ir = InstrumentIR.quick_synth("My Synth", brightness=0.7)
#   capsule = ir.to_capsule()
#
#   # From YAML:
#   ir = load_instrument("preset.yaml")
#   capsule = ir.to_capsule()
#
# =============================================================================

def _create_instrument_capsule(
    capsule_id: str,
    name: str,
    program_id: int,
    author: str = "",
    tags: List[str] = None,
    control_defaults: Dict[int, int] = None,
    voice_param_defaults: Dict[int, int] = None,
) -> CapsuleV0:
    """
    INTERNAL: Create a CapsuleInstrument.

    For human authoring, use InstrumentIR.quick_synth() instead.
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.INSTRUMENT,
            name=name,
            author=author,
            tags=tags or [],
        ),
        program=ProgramReference(program_id=program_id),
        defaults=CapsuleDefaults(
            control_defaults=control_defaults or {},
            voice_param_defaults=voice_param_defaults or {},
        ),
    )


def _create_synth_capsule(
    capsule_id: str,
    name: str,
    program_id: int,
    author: str = "",
    tags: List[str] = None,
    control_defaults: Dict[int, int] = None,
    voice_param_defaults: Dict[int, int] = None,
) -> CapsuleV0:
    """
    INTERNAL: Create a CapsuleSynth.

    For human authoring, use InstrumentIR.quick_synth() instead.
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.SYNTH,
            name=name,
            author=author,
            tags=tags or [],
        ),
        program=ProgramReference(program_id=program_id),
        defaults=CapsuleDefaults(
            control_defaults=control_defaults or {},
            voice_param_defaults=voice_param_defaults or {},
        ),
    )


def _create_asset_capsule(
    capsule_id: str,
    name: str,
    assets: List[AssetRef],
    author: str = "",
    tags: List[str] = None,
) -> CapsuleV0:
    """
    INTERNAL: Create a CapsuleAsset (references only, no control semantics).

    For human authoring, use InstrumentIR instead.
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.ASSET,
            name=name,
            author=author,
            tags=tags or [],
        ),
        assets=assets,
    )


def _create_macro_capsule(
    capsule_id: str,
    name: str,
    macro: MacroProgram,
    author: str = "",
    tags: List[str] = None,
) -> CapsuleV0:
    """
    INTERNAL: Create a CapsuleMacro (performance macro: vibrato, legato, etc.).

    For human authoring, use YAML macro presets.
    """
    return CapsuleV0(
        identity=CapsuleIdentity(
            capsule_id=capsule_id,
            kind=CapsuleKind.MACRO,
            name=name,
            author=author,
            tags=tags or [],
        ),
        macro=macro,
    )


# Backwards compatibility aliases (deprecated)
create_instrument_capsule = _create_instrument_capsule
create_synth_capsule = _create_synth_capsule
create_asset_capsule = _create_asset_capsule
create_macro_capsule = _create_macro_capsule


__all__ = [
    # Core types (public API)
    'CapsuleKind',
    'AssetType',
    'MacroType',
    'SystemMacroID',
    'EngineCompat',
    'CapsuleIdentity',
    'ProgramReference',
    'AssetRef',
    'SeedConfig',
    'MacroProgram',
    'ControlProgram',
    'CapsuleDefaults',
    'CapsuleV0',
    # Factory functions (deprecated - use InstrumentIR or YAML instead)
    'create_instrument_capsule',
    'create_synth_capsule',
    'create_asset_capsule',
    'create_macro_capsule',
]
