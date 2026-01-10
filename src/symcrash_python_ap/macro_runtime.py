"""
Macro Runtime
=============

Runtime macro state with slew filters per WSE-STD-MACRO v0.

MacroMap Lite v0 provides 3 preset macros:
- BRIGHT → brightness (macro 4)
- BODY → density (macro 5)
- IMPACT → energy_intent (macro 1)

All macros use Q15 unsigned values (0-32767).
Slew policies are applied at block boundaries.

Usage:
    runtime = MacroRuntime()
    runtime.set_macro("BRIGHT", 24576)  # Set brightness

    for block in range(total_blocks):
        runtime.tick()
        brightness = runtime.get_system_macro(4)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum


# =============================================================================
# CONSTANTS
# =============================================================================

Q15_ONE = 32767

def clamp_q15(x: int) -> int:
    """Clamp to valid Q15 range [0, 32767]."""
    return max(0, min(Q15_ONE, x))


# =============================================================================
# SYSTEM MACRO IDS (from WSE-STD-MACRO v0)
# =============================================================================

class SystemMacro(IntEnum):
    """System macro IDs per WSE-STD-MACRO v0."""
    ENERGY_STATE = 0     # Envelope-owned, never set by controllers
    ENERGY_INTENT = 1    # Controller → CMMM
    MOTION = 2           # Controller → CMMM
    FOCUS = 3            # Controller → CMMM
    BRIGHTNESS = 4       # Controller → CMMM (BRIGHT target)
    DENSITY = 5          # Controller → CMMM (BODY target)
    RISK = 6             # Controller → CMMM
    AGE = 7              # CMMM integrator (not set directly)
    SPACE = 8            # Controller → CMMM
    MATERIAL = 9         # Controller → CMMM
    UNITY = 10           # Controller → CMMM
    VIBRATO = 11         # Controller → Mode B (pitch modulation depth, 0-32767)
    LEGATO = 12          # Controller → Mode B (0=retrigger, >0=legato mode)


# =============================================================================
# SLEW POLICIES
# =============================================================================

class SlewPolicy(IntEnum):
    """Slew policy types per WSE-STD-MACRO v0 §5."""
    NONE = 0            # Instant update
    NBLOCK = 1          # Linear ramp over N blocks
    ONE_POLE = 2        # Exponential smoothing


@dataclass
class SlewConfig:
    """Slew configuration for a macro."""
    policy: SlewPolicy
    param: int = 0      # N for NBLOCK, alpha_q15 for ONE_POLE


# Default slew table from WSE-STD-MACRO v0 §6
DEFAULT_SLEW_TABLE: Dict[int, SlewConfig] = {
    SystemMacro.ENERGY_STATE: SlewConfig(SlewPolicy.NONE),          # Envelope-owned
    SystemMacro.ENERGY_INTENT: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.MOTION: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.FOCUS: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.BRIGHTNESS: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.DENSITY: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.RISK: SlewConfig(SlewPolicy.NBLOCK, 16),
    SystemMacro.AGE: SlewConfig(SlewPolicy.NONE),                   # Integrator
    SystemMacro.SPACE: SlewConfig(SlewPolicy.ONE_POLE, 2048),
    SystemMacro.MATERIAL: SlewConfig(SlewPolicy.NBLOCK, 32),
    SystemMacro.UNITY: SlewConfig(SlewPolicy.NBLOCK, 32),
}


# =============================================================================
# MACRO BINDING
# =============================================================================

@dataclass
class MacroBinding:
    """
    Single macro binding from preset space to system macro.

    Preset macros (BRIGHT, BODY, IMPACT) are mapped to system macros
    via affine transformation: out = (a * in) >> 15 + b
    """
    preset_macro: str       # "BRIGHT", "BODY", "IMPACT"
    system_macro_id: int    # 1-10 (never 0, envelope-owned)
    map_type: str           # "AFFINE_Q15"
    a_q15: int              # Scale factor (32767 = 1.0)
    b_q15: int              # Offset


# MacroMap Lite v0: 3 preset macros
MACROMAP_LITE_V0: List[MacroBinding] = [
    MacroBinding("BRIGHT", SystemMacro.BRIGHTNESS, "AFFINE_Q15", Q15_ONE, 0),
    MacroBinding("BODY", SystemMacro.DENSITY, "AFFINE_Q15", Q15_ONE, 0),
    MacroBinding("IMPACT", SystemMacro.ENERGY_INTENT, "AFFINE_Q15", Q15_ONE, 0),
]


# =============================================================================
# MACRO RUNTIME
# =============================================================================

@dataclass
class MacroRuntime:
    """
    Runtime macro state with slew filters.

    Manages the mapping from preset macros to system macros,
    and applies slew filters per WSE-STD-MACRO v0.

    AGE Integrator (per WSE-STD-MACRO v0 §7):
    - AGE accumulates based on energy_state each block
    - inc = mul_q15(k_inc_q15, energy_state)
    - dec = mul_q15(k_dec_q15, (Q15_ONE - energy_state))
    - age = clamp_q15(age + inc - dec)
    """
    bindings: List[MacroBinding] = field(default_factory=lambda: list(MACROMAP_LITE_V0))
    slew_table: Dict[int, SlewConfig] = field(default_factory=lambda: dict(DEFAULT_SLEW_TABLE))

    # Macro state (per system macro ID)
    macro_target_q15: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(11)})
    macro_curr_q15: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(11)})

    # AGE integrator parameters (per WSE-STD-MACRO v0 §7)
    # These control how fast AGE accumulates during note-on vs decays during release
    age_k_inc_q15: int = 64      # ~0.2% per block at full energy (slow accumulation)
    age_k_dec_q15: int = 32      # ~0.1% per block at zero energy (slow decay)

    # Binding lookup
    _binding_map: Dict[str, MacroBinding] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Build binding lookup map."""
        self._binding_map = {b.preset_macro.upper(): b for b in self.bindings}

    def set_macro(self, preset_macro: str, value_q15: int) -> None:
        """
        Set a preset macro value (will be mapped and slewed).

        Args:
            preset_macro: Preset macro name ("BRIGHT", "BODY", "IMPACT")
            value_q15: Value (0-32767)
        """
        binding = self._binding_map.get(preset_macro.upper())
        if binding is None:
            return  # Unknown macro, ignore

        # Apply affine mapping: out = (a * in) >> 15 + b
        if binding.map_type == "AFFINE_Q15":
            mapped = ((binding.a_q15 * value_q15) >> 15) + binding.b_q15
        else:
            mapped = value_q15

        self.macro_target_q15[binding.system_macro_id] = clamp_q15(mapped)

    def set_system_macro(self, macro_id: int, value_q15: int) -> None:
        """
        Set a system macro directly (bypassing preset mapping).

        Note: energy_state (0) should not be set by controllers.

        Args:
            macro_id: System macro ID (1-10)
            value_q15: Value (0-32767)
        """
        if macro_id == SystemMacro.ENERGY_STATE:
            return  # Envelope-owned, reject

        self.macro_target_q15[macro_id] = clamp_q15(value_q15)

    def tick(self) -> None:
        """
        Advance slew filters by one block.

        Call once per block to update all macro current values.
        """
        for macro_id in self.macro_target_q15.keys():
            if macro_id == SystemMacro.ENERGY_STATE:
                continue  # Envelope-owned
            if macro_id == SystemMacro.AGE:
                continue  # Integrator, handled separately

            target = self.macro_target_q15[macro_id]
            curr = self.macro_curr_q15[macro_id]

            slew = self.slew_table.get(macro_id, SlewConfig(SlewPolicy.NONE))

            if slew.policy == SlewPolicy.NONE:
                # Instant
                self.macro_curr_q15[macro_id] = target

            elif slew.policy == SlewPolicy.NBLOCK:
                # Linear ramp over N blocks
                n = max(1, slew.param)
                d = target - curr

                if d == 0:
                    pass
                else:
                    # step = ceil(abs(d) / n)
                    step = (abs(d) + n - 1) // n
                    if d > 0:
                        self.macro_curr_q15[macro_id] = min(curr + step, target)
                    else:
                        self.macro_curr_q15[macro_id] = max(curr - step, target)

            elif slew.policy == SlewPolicy.ONE_POLE:
                # Exponential: curr = curr + alpha * (target - curr)
                alpha = slew.param  # alpha_q15
                d = target - curr
                delta = (alpha * d) >> 15
                if delta == 0 and d != 0:
                    delta = 1 if d > 0 else -1
                self.macro_curr_q15[macro_id] = clamp_q15(curr + delta)

        # Update AGE integrator (per WSE-STD-MACRO v0 section 7)
        # AGE accumulates during note-on (energy_state > 0) and decays during release
        # inc = mul_q15(k_inc_q15, energy_state)
        # dec = mul_q15(k_dec_q15, (Q15_ONE - energy_state))
        # age = clamp_q15(age + inc - dec)
        energy = self.macro_curr_q15.get(SystemMacro.ENERGY_STATE, 0)
        age = self.macro_curr_q15.get(SystemMacro.AGE, 0)
        inc = (self.age_k_inc_q15 * energy) >> 15
        dec = (self.age_k_dec_q15 * (Q15_ONE - energy)) >> 15
        self.macro_curr_q15[SystemMacro.AGE] = clamp_q15(age + inc - dec)

    def get_system_macro(self, macro_id: int) -> int:
        """
        Get current value of a system macro (after slew).

        Args:
            macro_id: System macro ID (0-10)

        Returns:
            Current Q15 value
        """
        return self.macro_curr_q15.get(macro_id, 0)

    def get_preset_macro(self, preset_macro: str) -> int:
        """
        Get current value of a preset macro (mapped to system macro, after slew).

        Args:
            preset_macro: Preset macro name

        Returns:
            Current Q15 value
        """
        binding = self._binding_map.get(preset_macro.upper())
        if binding is None:
            return 0
        return self.macro_curr_q15.get(binding.system_macro_id, 0)

    def set_energy_state(self, value_q15: int) -> None:
        """
        Set energy_state from envelope (internal use only).

        This is the only way to set macro 0.
        """
        self.macro_target_q15[SystemMacro.ENERGY_STATE] = clamp_q15(value_q15)
        self.macro_curr_q15[SystemMacro.ENERGY_STATE] = clamp_q15(value_q15)

    def reset(self) -> None:
        """Reset all macros to zero."""
        for i in range(len(self.macro_target_q15)):
            self.macro_target_q15[i] = 0
            self.macro_curr_q15[i] = 0

    def register_macro(
        self,
        macro_id: int,
        preset_name: str,
        slew_config: Optional[SlewConfig] = None,
        default_q15: int = 0,
    ) -> None:
        """
        Register a new macro dynamically (e.g., from a MacroProgram capsule).

        This allows capsules to extend the macro system without hardcoding.

        Args:
            macro_id: System macro ID (should be >= 11 for user macros)
            preset_name: Preset macro name (e.g., "VIBRATO", "LEGATO")
            slew_config: Slew configuration (defaults to ONE_POLE with alpha=2048)
            default_q15: Initial value
        """
        # Ensure state dictionaries are large enough
        if macro_id not in self.macro_target_q15:
            self.macro_target_q15[macro_id] = default_q15
            self.macro_curr_q15[macro_id] = default_q15

        # Add slew config
        if slew_config is None:
            slew_config = SlewConfig(SlewPolicy.ONE_POLE, 2048)
        self.slew_table[macro_id] = slew_config

        # Add binding
        binding = MacroBinding(preset_name.upper(), macro_id, "AFFINE_Q15", Q15_ONE, 0)
        self.bindings.append(binding)
        self._binding_map[preset_name.upper()] = binding

    def apply_macro_capsule(self, macro_program: 'MacroProgram') -> None:
        """
        Apply a MacroProgram capsule to register its macro and set defaults.

        This is the integration point between the capsule library and runtime.

        Args:
            macro_program: MacroProgram from a CapsuleMacro
        """
        from symcrash_python_ap.library.capsule import MacroType

        if macro_program.macro_type == MacroType.VIBRATO:
            # Register VIBRATO macro with smooth slew
            self.register_macro(
                macro_id=SystemMacro.VIBRATO,
                preset_name="VIBRATO",
                slew_config=SlewConfig(SlewPolicy.ONE_POLE, 4096),
                default_q15=Q15_ONE,  # Full depth by default
            )
        elif macro_program.macro_type == MacroType.LEGATO:
            # Register LEGATO macro with instant toggle
            self.register_macro(
                macro_id=SystemMacro.LEGATO,
                preset_name="LEGATO",
                slew_config=SlewConfig(SlewPolicy.NONE),
                default_q15=Q15_ONE if macro_program.preserve_phase else 0,
            )
        elif macro_program.macro_type == MacroType.TREMOLO:
            # Tremolo uses VIBRATO macro but amplitude-only
            self.register_macro(
                macro_id=SystemMacro.VIBRATO,
                preset_name="TREMOLO",
                slew_config=SlewConfig(SlewPolicy.ONE_POLE, 4096),
                default_q15=Q15_ONE,
            )
        elif macro_program.macro_type == MacroType.PORTAMENTO:
            # Portamento uses LEGATO macro
            self.register_macro(
                macro_id=SystemMacro.LEGATO,
                preset_name="PORTAMENTO",
                slew_config=SlewConfig(SlewPolicy.NBLOCK, 8),
                default_q15=Q15_ONE,
            )

    def apply_control_defaults(self, control: 'ControlProgram') -> None:
        """
        Apply ControlProgram defaults to set initial macro target values.

        This is called at preset load time to configure the semantic control axes
        (BRIGHTNESS, DENSITY, MOTION, etc.) from the preset's control section.

        Args:
            control: ControlProgram from a CapsuleV0.control field

        Example:
            capsule = load_preset("violin_ir_v1.yaml")
            if capsule.control:
                runtime.apply_control_defaults(capsule.control)
        """
        macro_dict = control.to_macro_dict()
        for macro_id, value_q15 in macro_dict.items():
            # Skip envelope-owned and integrator macros
            if macro_id == SystemMacro.ENERGY_STATE:
                continue
            if macro_id == SystemMacro.AGE:
                continue
            # Set both target and current to apply immediately
            self.macro_target_q15[macro_id] = clamp_q15(value_q15)
            self.macro_curr_q15[macro_id] = clamp_q15(value_q15)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Q15_ONE',
    'clamp_q15',
    'SystemMacro',
    'SlewPolicy',
    'SlewConfig',
    'DEFAULT_SLEW_TABLE',
    'MacroBinding',
    'MACROMAP_LITE_V0',
    'MacroRuntime',
]
