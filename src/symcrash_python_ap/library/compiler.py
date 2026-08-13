"""
Capsule Compiler (WSE-STD-IPLC v0 §4)
======================================

Converts authoring inputs (InstrumentIR, etc.) into COMPILED capsule bundles.

Compiler Contract (§4):
1. All overrides are clampable against registries
2. All IDs are resolved (program_id, graph_id)
3. All assets are referenced or embedded deterministically
4. All ABI requirements are declared in manifest

Versioning Rules (§7):
- capsule_id identifies "the instrument identity" across versions
- Same identity = same capsule_id across rebuilds
- Changed identity = new capsule_id

Natural IR Compilation:
- InstrumentIR.to_capsule() compiles the IR to CapsuleV0
- CapsuleCompiler.compile_from_ir(ir) handles full compilation with validation
- L0 macros map to control_defaults
- L2 regime graph provides regime_graph_id
- L3 emit templates provide tree family defaults
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from .capsule import (
    CapsuleV0,
    CapsuleKind,
    CapsuleIdentity,
    EngineCompat,
    ProgramReference,
    CapsuleDefaults,
    SeedConfig,
    AssetRef,
    AssetType,
    ControlProgram,
    create_instrument_capsule,
    create_synth_capsule,
)
from .lifecycle import LifecycleState, CapsuleRecord, create_compiled_record
from .central_registry import get_central_registry
from ..control_registry import ControlID
from ..voice.voice_engine_registry import VoiceParamID
from ..primitives import Q15_ONE

if TYPE_CHECKING:
    from ..instrument_ir import InstrumentIR
    from ..macro_ir import MacroIR


# =============================================================================
# COMPILE RESULT
# =============================================================================

@dataclass
class CompileResult:
    """Result of capsule compilation."""
    success: bool
    capsule: Optional[CapsuleV0] = None
    record: Optional[CapsuleRecord] = None  # In COMPILED state
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    capsule_id: Optional[str] = None  # Generated stable ID


# =============================================================================
# CAPSULE COMPILER
# =============================================================================

class CapsuleCompiler:
    """
    Converts authoring inputs into COMPILED capsule bundles.

    Per WSE-STD-IPLC v0 §4.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        block_size: int = 64,
    ):
        """
        Initialize compiler with central registry.

        Args:
            sample_rate: Target sample rate
            block_size: Target block size
        """
        central = get_central_registry()
        self.control_registry = central.controls
        self.voice_registry = central.voice_engine
        self.sample_rate = sample_rate
        self.block_size = block_size

        # ABI versions
        self.control_registry_abi = 0
        self.voice_registry_abi = 0
        self.voiceframe_abi = 0

    def compile_instrument(
        self,
        program_id: int,
        name: str,
        author: str = "",
        tags: List[str] = None,
        control_defaults: Dict[int, int] = None,
        voice_param_defaults: Dict[int, int] = None,
        regime_graph_id: int = None,
        tree_family_id: int = None,
        instrument_ir: dict = None,
        master_seed: int = 0x12345678,
        assets: List[AssetRef] = None,
        capsule_id: str = None,
        control_program: 'ControlProgram' = None,
        envelopes: List[dict] = None,
        amp_dynamics: dict = None,
        sound_extras: dict = None,
    ) -> CompileResult:
        """
        Compile inputs to CapsuleInstrument.

        Args:
            program_id: Voice engine program ID (required)
            name: Human-readable name (required)
            author: Optional author name
            tags: Optional list of tags
            control_defaults: Optional control default overrides
            voice_param_defaults: Optional voice param default overrides
            regime_graph_id: Optional regime graph ID
            tree_family_id: Optional tree family ID
            instrument_ir: Optional InstrumentIR data (dict)
            master_seed: PRNG seed for determinism
            assets: Optional asset references
            capsule_id: Optional explicit capsule_id (generated if None)
            control_program: Optional ControlProgram with system macro defaults
            envelopes: Optional matrix envelope binding dicts (id/adsr/targets/
                velocity_sens/key_scale_rate) carried into the capsule
            amp_dynamics: Optional amp dynamics dict (velocity_sens/key_scale_rate)
            sound_extras: Optional SOUND_EXTRAS dict (material ramp, FM6
                post-filter, partials, PWM, LFOs) closing capsule lossiness

        Returns:
            CompileResult with capsule in COMPILED state
        """
        result = CompileResult(success=True)
        tags = tags or []
        control_defaults = control_defaults or {}
        voice_param_defaults = voice_param_defaults or {}
        assets = assets or []

        # Generate capsule_id if not provided (§7)
        if capsule_id is None:
            capsule_id = self._generate_capsule_id(
                kind=CapsuleKind.INSTRUMENT,
                program_id=program_id,
                name=name,
                author=author,
            )
        result.capsule_id = capsule_id

        # Step 1: Validate and clamp control defaults (§4.1)
        clamped_controls, control_warnings = self._validate_control_defaults(control_defaults)
        result.warnings.extend(control_warnings)

        # Step 2: Validate and clamp voice param defaults (§4.1)
        clamped_voice_params, voice_warnings = self._validate_voice_param_defaults(voice_param_defaults)
        result.warnings.extend(voice_warnings)

        # Step 3: Validate program_id and other IDs (§4.2)
        id_errors = self._validate_ids(program_id, regime_graph_id, tree_family_id)
        if id_errors:
            result.errors.extend(id_errors)
            result.success = False
            return result

        # Step 4: Create capsule with ABI requirements (§4.4)
        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id=capsule_id,
                kind=CapsuleKind.INSTRUMENT,
                name=name,
                author=author,
                tags=tags,
                engine_compat=EngineCompat(
                    control_registry_abi=self.control_registry_abi,
                    voice_registry_abi=self.voice_registry_abi,
                    voiceframe_abi=self.voiceframe_abi,
                    sample_rate_hz=self.sample_rate,
                    block_size=self.block_size,
                ),
            ),
            program=ProgramReference(
                program_id=program_id,
                instrument_ir_id=instrument_ir.get('id') if instrument_ir else None,
                regime_graph_id=regime_graph_id,
                tree_family_id=tree_family_id,
            ),
            defaults=CapsuleDefaults(
                control_defaults=clamped_controls,
                voice_param_defaults=clamped_voice_params,
            ),
            assets=assets,
            seeds=SeedConfig(
                master_seed=master_seed,
                derivation_rule_id=0,  # deterministic
            ),
            control=control_program,  # System macro defaults for cockpit
            envelopes=envelopes,      # Matrix envelope bindings (packet E6)
            amp_dynamics=amp_dynamics,
            sound_extras=sound_extras,
        )

        # Validate capsule structure
        validation_issues = capsule.validate()
        for issue in validation_issues:
            if issue.startswith("ERROR"):
                result.errors.append(issue)
                result.success = False
            else:
                result.warnings.append(issue)

        if result.success:
            result.capsule = capsule
            result.record = create_compiled_record(capsule)

        return result

    def compile_synth(
        self,
        program_id: int,
        name: str,
        author: str = "",
        tags: List[str] = None,
        control_defaults: Dict[int, int] = None,
        voice_param_defaults: Dict[int, int] = None,
        master_seed: int = 0x12345678,
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile inputs to CapsuleSynth.

        Args:
            program_id: Voice engine program ID
            name: Human-readable name
            author: Optional author name
            tags: Optional list of tags
            control_defaults: Optional control default overrides
            voice_param_defaults: Optional voice param default overrides
            master_seed: PRNG seed for determinism
            capsule_id: Optional explicit capsule_id

        Returns:
            CompileResult with capsule in COMPILED state
        """
        result = CompileResult(success=True)
        tags = tags or []
        control_defaults = control_defaults or {}
        voice_param_defaults = voice_param_defaults or {}

        # Generate capsule_id if not provided
        if capsule_id is None:
            capsule_id = self._generate_capsule_id(
                kind=CapsuleKind.SYNTH,
                program_id=program_id,
                name=name,
                author=author,
            )
        result.capsule_id = capsule_id

        # Validate defaults
        clamped_controls, control_warnings = self._validate_control_defaults(control_defaults)
        result.warnings.extend(control_warnings)

        clamped_voice_params, voice_warnings = self._validate_voice_param_defaults(voice_param_defaults)
        result.warnings.extend(voice_warnings)

        # Validate IDs
        id_errors = self._validate_ids(program_id)
        if id_errors:
            result.errors.extend(id_errors)
            result.success = False
            return result

        # Create capsule
        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id=capsule_id,
                kind=CapsuleKind.SYNTH,
                name=name,
                author=author,
                tags=tags,
                engine_compat=EngineCompat(
                    control_registry_abi=self.control_registry_abi,
                    voice_registry_abi=self.voice_registry_abi,
                    voiceframe_abi=self.voiceframe_abi,
                    sample_rate_hz=self.sample_rate,
                    block_size=self.block_size,
                ),
            ),
            program=ProgramReference(program_id=program_id),
            defaults=CapsuleDefaults(
                control_defaults=clamped_controls,
                voice_param_defaults=clamped_voice_params,
            ),
            seeds=SeedConfig(master_seed=master_seed, derivation_rule_id=0),
        )

        validation_issues = capsule.validate()
        for issue in validation_issues:
            if issue.startswith("ERROR"):
                result.errors.append(issue)
                result.success = False
            else:
                result.warnings.append(issue)

        if result.success:
            result.capsule = capsule
            result.record = create_compiled_record(capsule)

        return result

    def _generate_capsule_id(
        self,
        kind: CapsuleKind,
        program_id: int,
        name: str,
        author: str,
    ) -> str:
        """
        Generate stable capsule_id from identity (§7).

        Same identity = same capsule_id across rebuilds.
        """
        # Create identity string
        identity_str = f"{kind.name}:{program_id}:{name}:{author}"

        # Hash to create stable ID
        hash_bytes = hashlib.sha256(identity_str.encode()).hexdigest()[:12]

        # Create human-readable ID
        safe_name = name.lower().replace(' ', '-')[:20]
        return f"{safe_name}-{hash_bytes}"

    def _validate_control_defaults(
        self,
        defaults: Dict[int, int],
    ) -> tuple[Dict[int, int], List[str]]:
        """
        Validate and clamp control defaults against registry (§4.1).

        Returns:
            (clamped_defaults, warnings)
        """
        clamped = {}
        warnings = []

        for control_id, value in defaults.items():
            if self.control_registry.has_control(control_id):
                clamped_value = self.control_registry.clamp(control_id, value)
                clamped[control_id] = clamped_value

                if clamped_value != value:
                    warnings.append(
                        f"Control {control_id} value {value} clamped to {clamped_value}"
                    )
            else:
                warnings.append(f"Unknown control_id {control_id} ignored")

        return clamped, warnings

    def _validate_voice_param_defaults(
        self,
        defaults: Dict[int, int],
    ) -> tuple[Dict[int, int], List[str]]:
        """
        Validate and clamp voice param defaults against registry (§4.1).

        Returns:
            (clamped_defaults, warnings)
        """
        clamped = {}
        warnings = []

        for param_id, value in defaults.items():
            entry = self.voice_registry.get(param_id)
            if entry is not None:
                clamped_value = self.voice_registry.clamp(param_id, value)
                clamped[param_id] = clamped_value

                if clamped_value != value:
                    warnings.append(
                        f"Voice param {param_id} value {value} clamped to {clamped_value}"
                    )
            else:
                warnings.append(f"Unknown param_id {param_id} ignored")

        return clamped, warnings

    def _validate_ids(
        self,
        program_id: int,
        regime_graph_id: int = None,
        tree_family_id: int = None,
    ) -> List[str]:
        """
        Validate IDs are resolved (§4.2).

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Program ID validation
        if program_id is None:
            errors.append("program_id is required")
        elif program_id < 0:
            errors.append(f"Invalid program_id: {program_id}")

        # Regime graph ID validation (if provided)
        if regime_graph_id is not None and regime_graph_id < 0:
            errors.append(f"Invalid regime_graph_id: {regime_graph_id}")

        # Tree family ID validation (if provided)
        if tree_family_id is not None and tree_family_id < 0:
            errors.append(f"Invalid tree_family_id: {tree_family_id}")

        return errors

    # =========================================================================
    # NATURAL IR COMPILATION
    # =========================================================================

    def compile_from_ir(
        self,
        ir: 'InstrumentIR',
        program_id: int = None,
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile InstrumentIR directly to CapsuleV0.

        This is the natural compilation path - the IR contains all metadata,
        control mappings, regime graphs, and emission templates needed.

        Args:
            ir: Complete InstrumentIR v0
            program_id: Override program_id (uses hash of instrument_id if None)
            capsule_id: Override capsule_id (generated from IR if None)

        Returns:
            CompileResult with capsule in COMPILED state

        Example:
            compiler = CapsuleCompiler()
            ir = load_instrument("my_synth.yaml")
            result = compiler.compile_from_ir(ir)
            if result.success:
                capsule = result.capsule
        """
        result = CompileResult(success=True)

        # Validate IR structure first
        valid, error_msg = ir.validate()
        if not valid:
            result.errors.append(f"IR validation failed: {error_msg}")
            result.success = False
            return result

        # Extract metadata from IR
        meta = ir.meta
        name = meta.name or meta.instrument_id or "Unnamed"
        author = meta.author or ""
        master_seed = meta.seed_u32 or 0x12345678

        # Generate program_id from instrument_id if not provided
        if program_id is None:
            if meta.instrument_id:
                # Hash instrument_id to get stable program_id
                hash_val = int(hashlib.sha256(
                    meta.instrument_id.encode()
                ).hexdigest()[:8], 16)
                program_id = hash_val & 0x7FFFFFFF  # Keep positive
            else:
                result.errors.append("Either program_id or instrument_id is required")
                result.success = False
                return result

        # Extract control defaults from L0 macros
        control_defaults = self._extract_control_defaults_from_ir(ir)
        result.warnings.extend(control_defaults.get('_warnings', []))
        control_defaults.pop('_warnings', None)

        # Extract voice param defaults from compile config
        voice_param_defaults = self._extract_voice_param_defaults_from_ir(ir)

        # Extract regime graph ID (hash of regime structure)
        regime_graph_id = self._extract_regime_graph_id(ir)

        # Extract tree family ID from emission templates
        tree_family_id = self._extract_tree_family_id(ir)

        # Build tags from IR
        tags = self._extract_tags_from_ir(ir)

        # Build asset references from compile.map_library
        assets = self._extract_assets_from_ir(ir)

        # Extract control program from IR (system macro defaults)
        control_program = None
        if ir.control is not None:
            control_program = ir.control.to_control_program()

        # Extract matrix envelope bindings (WSE-STD-ENVELOPE-MATRIX v0) so the
        # compiled capsule retains mod envelopes (velocity -> FM index etc.)
        envelopes = self._extract_envelopes_from_ir(ir)

        # Extract amp dynamics (velocity_sens / key_scale_rate) from Amp node
        amp_dynamics = self._extract_amp_dynamics_from_ir(ir)

        # Sound extras: close the capsule's documented lossiness vs the IR
        # (material ramp, FM6 post-filter, partials/PWM, LFO modulators)
        sound_extras = self._extract_sound_extras_from_ir(ir)

        # Use standard compile_instrument with extracted values
        return self.compile_instrument(
            program_id=program_id,
            name=name,
            author=author,
            tags=tags,
            control_defaults=control_defaults,
            voice_param_defaults=voice_param_defaults,
            regime_graph_id=regime_graph_id,
            tree_family_id=tree_family_id,
            instrument_ir={'id': meta.instrument_id},
            master_seed=master_seed,
            assets=assets,
            capsule_id=capsule_id,
            control_program=control_program,
            envelopes=envelopes,
            amp_dynamics=amp_dynamics,
            sound_extras=sound_extras,
        )

    def compile_from_macro(
        self,
        macro_ir: 'MacroIR',
        capsule_id: str = None,
    ) -> CompileResult:
        """
        Compile MacroIR to CapsuleV0 with CapsuleKind.MACRO.

        Args:
            macro_ir: Complete MacroIR v0
            capsule_id: Override capsule_id (generated from IR if None)

        Returns:
            CompileResult with macro capsule
        """
        result = CompileResult(success=True)

        # Validate macro structure
        valid, error_msg = macro_ir.validate()
        if not valid:
            result.errors.append(f"Macro validation failed: {error_msg}")
            result.success = False
            return result

        try:
            # Delegate to MacroIR.to_capsule()
            capsule = macro_ir.to_capsule(capsule_id)
            result.capsule = capsule
            result.capsule_id = capsule.identity.capsule_id
            result.record = create_compiled_record(capsule)
        except Exception as e:
            result.errors.append(f"Macro compilation failed: {e}")
            result.success = False

        return result

    def _extract_control_defaults_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Dict[int, int]:
        """
        Extract control defaults from IR L0 macros and compile config.

        Maps L0 macro defaults to ControlRegistry control IDs.
        """
        defaults: Dict[int, int] = {}
        warnings: List[str] = []

        # Map L0 macros to control IDs
        macro_to_control = {
            'M_BRIGHT': ControlID.TIMBRE,
            'M_ATTACK': ControlID.ATK_BLOCKS,
            'M_DECAY': ControlID.DCY_BLOCKS,
            'M_SUSTAIN': ControlID.SUS_Q15,
            'M_RELEASE': ControlID.REL_BLOCKS,
            'M_GLITCH': ControlID.CHAOS,
            'M_MORPH': ControlID.MORPH,
            'M_TEXTURE': ControlID.TEXTURE,
            'M_DETUNE': ControlID.DETUNE,
        }

        # Extract from L0 macros
        for macro in ir.layers.L0_graph.macros:
            macro_id = macro.id.upper()
            if macro_id in macro_to_control:
                control_id = macro_to_control[macro_id]
                # Convert macro default (0.0-1.0) to Q15 (0-32767)
                default_q15 = int(macro.default * Q15_ONE)
                defaults[control_id] = default_q15
            else:
                warnings.append(f"Unknown macro {macro.id} - no control mapping")

        # Extract ADSR from L0 Amp node if present (shared IR extraction)
        from ..primitives import ms_to_blocks
        from .ir_extract import extract_amp
        amp = extract_amp(ir)
        if amp is not None:
            defaults[ControlID.ATK_BLOCKS] = ms_to_blocks(amp.a_ms)
            defaults[ControlID.DCY_BLOCKS] = ms_to_blocks(amp.d_ms)
            defaults[ControlID.SUS_Q15] = int(amp.s * Q15_ONE)
            defaults[ControlID.REL_BLOCKS] = ms_to_blocks(amp.r_ms)

        defaults['_warnings'] = warnings
        return defaults

    def _extract_voice_param_defaults_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Dict[int, int]:
        """
        Extract voice param defaults from IR.

        Sources:
        - compile.frame_emitter_policy for frame emission params
        - L0 nodes for waveform/carrier params
        - L0 FM6 nodes for 6-operator FM synthesis params
        """
        defaults: Dict[int, int] = {}

        # Shared IR extraction (library/ir_extract.py). Only values the IR
        # actually declares are written; registry defaults stand otherwise.
        # (Osc waveform is not carried in voice params yet; when it is,
        # use instrument_ir.extract_waveform_type - the single canonical
        # name->id mapping - rather than a local table.)
        from .ir_extract import extract_fm6
        fm6 = extract_fm6(ir)
        if fm6 is not None:
            # Set OSC_MODE to 3 (FM6)
            defaults[VoiceParamID.OSC_MODE] = 3

            # Algorithm (1-32)
            if fm6.algorithm is not None:
                defaults[VoiceParamID.FM6_ALGORITHM] = fm6.algorithm

            # Feedback (0-7)
            if fm6.feedback is not None:
                defaults[VoiceParamID.FM6_FEEDBACK] = fm6.feedback

            # Operator ratios (Q16.16: 65536 = 1.0x), stored as U32
            if fm6.op_ratios:
                ratio_params = [
                    VoiceParamID.FM6_OP0_RATIO,
                    VoiceParamID.FM6_OP1_RATIO,
                    VoiceParamID.FM6_OP2_RATIO,
                    VoiceParamID.FM6_OP3_RATIO,
                    VoiceParamID.FM6_OP4_RATIO,
                    VoiceParamID.FM6_OP5_RATIO,
                ]
                for i, ratio in enumerate(fm6.op_ratios[:6]):
                    defaults[ratio_params[i]] = ratio

            # Operator levels (Q15: 32767 = full)
            if fm6.op_levels:
                level_params = [
                    VoiceParamID.FM6_OP0_LEVEL,
                    VoiceParamID.FM6_OP1_LEVEL,
                    VoiceParamID.FM6_OP2_LEVEL,
                    VoiceParamID.FM6_OP3_LEVEL,
                    VoiceParamID.FM6_OP4_LEVEL,
                    VoiceParamID.FM6_OP5_LEVEL,
                ]
                for i, level in enumerate(fm6.op_levels[:6]):
                    defaults[level_params[i]] = level

            # Topology morphing
            if fm6.topo_a is not None:
                defaults[VoiceParamID.FM6_TOPO_A] = fm6.topo_a
            if fm6.topo_b is not None:
                defaults[VoiceParamID.FM6_TOPO_B] = fm6.topo_b
            if fm6.topo_blend_q15 is not None:
                defaults[VoiceParamID.FM6_TOPO_BLEND] = fm6.topo_blend_q15

        return defaults

    def _extract_envelopes_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Optional[List[dict]]:
        """
        Extract matrix envelope bindings from IR as JSON-friendly dicts.

        These are carried through capsule serialization so a compiled
        capsule retains the preset's `envelopes:` section (mod envelopes
        routed to FM operators, filter sweeps, etc.).
        """
        bindings = getattr(ir, 'envelopes', None) or []
        if not bindings:
            return None

        result = []
        for b in bindings:
            adsr = b.adsr
            result.append({
                'id': b.id,
                'adsr': {
                    'a_ms': adsr.a_ms,
                    'd_ms': adsr.d_ms,
                    's': adsr.s,
                    'r_ms': adsr.r_ms,
                },
                'targets': list(b.targets),
                'velocity_sens': float(getattr(b, 'velocity_sens', 0.0)),
                'key_scale_rate': float(getattr(b, 'key_scale_rate', 0.0)),
            })
        return result

    def _extract_amp_dynamics_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Optional[dict]:
        """
        Extract amp dynamics (velocity_sens / key_scale_rate) from the Amp node.

        Defaults (1.0 / 0.0) are omitted so legacy capsules stay byte-identical.
        """
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Amp" and node.params.adsr is not None:
                velocity_sens = float(getattr(node.params, 'velocity_sens', 1.0))
                key_scale_rate = float(getattr(node.params, 'key_scale_rate', 0.0))
                if velocity_sens != 1.0 or key_scale_rate != 0.0:
                    return {
                        'velocity_sens': velocity_sens,
                        'key_scale_rate': key_scale_rate,
                    }
                return None
        return None

    def _extract_sound_extras_from_ir(
        self,
        ir: 'InstrumentIR',
    ) -> Optional[dict]:
        """
        Extract the SOUND_EXTRAS payload: everything the older chunks could
        not carry (material beta ramp, FM6 post-filter, resolved harmonic
        profile / phase spread, engine osc mode, PWM, LFO modulators).
        """
        from .ir_extract import (
            extract_fm6, extract_pwm_width_q15, extract_harmonics_q15,
            lfo_to_dict,
        )
        from ..instrument_ir import (
            extract_phase_spread_q15, extract_waveform_type,
        )

        extras: dict = {}

        # Material beta ramp (compile.material)
        mat = getattr(getattr(ir, 'compile', None), 'material', None)
        if mat is not None and (mat.beta_start_q15 or mat.beta_target_q15
                                or mat.beta_ramp_blocks != 1):
            extras['material'] = {
                'beta_start_q15': mat.beta_start_q15,
                'beta_target_q15': mat.beta_target_q15,
                'beta_ramp_blocks': mat.beta_ramp_blocks,
            }

        # FM6 post filter
        fm6 = extract_fm6(ir)
        if fm6 is not None and fm6.post_filter_hz is not None:
            extras['fm6_post_filter_hz'] = float(fm6.post_filter_hz)

        # FM6 phase-modulation depth scale (DX7-fidelity imports)
        if fm6 is not None and fm6.mod_index is not None:
            extras['fm6_mod_index'] = float(fm6.mod_index)

        # Resolved harmonic profile (custom partials win over the named
        # waveform's default profile) + phase spread (Mode B spectrum)
        harmonics = extract_harmonics_q15(ir)
        if harmonics:
            extras['harmonic_amps_q15'] = harmonics
        spread = extract_phase_spread_q15(ir)
        if spread:
            extras['phase_spread_q15'] = spread

        # Engine osc mode + waveform id (resolves the OSC_MODE=0 ambiguity:
        # 3 = FM6 via voice params; Mode B presets are explicit here)
        if fm6 is None:
            extras['osc_mode'] = 1  # Mode B (harmonic bank)
        extras['waveform_type'] = extract_waveform_type(ir)

        # PWM
        pwm_width = extract_pwm_width_q15(ir)
        if pwm_width is not None:
            extras['pwm'] = {'enabled': True, 'width_q15': pwm_width}

        # LFO modulators
        lfos = getattr(ir, 'modulators', None) or []
        if lfos:
            extras['lfos'] = [lfo_to_dict(lfo) for lfo in lfos]

        # Always emitted: the loader relies on osc_mode/waveform_type to
        # resolve the OSC_MODE=0 (unset vs Mode A) ambiguity. Chunk is
        # JSON/manifest-only; binary chunks stay byte-identical.
        return extras

    def _extract_regime_graph_id(self, ir: 'InstrumentIR') -> Optional[int]:
        """
        Generate regime graph ID from L2 regime structure.

        Creates a stable hash of the regime graph for identification.
        """
        regimes = ir.layers.L2_regimes
        if not regimes.regime_nodes:
            return None

        # Create identity string from regime structure
        regime_str = ""
        for node in sorted(regimes.regime_nodes, key=lambda n: n.id):
            regime_str += f"{node.id}:{node.kind}:{node.emit_template}|"
        for trans in sorted(regimes.transitions, key=lambda t: (t.from_regime, t.to)):
            regime_str += f"{trans.from_regime}->{trans.to}:{trans.when}|"

        # Hash to get stable ID
        hash_val = int(hashlib.sha256(regime_str.encode()).hexdigest()[:8], 16)
        return hash_val & 0x7FFFFFFF

    def _extract_tree_family_id(self, ir: 'InstrumentIR') -> Optional[int]:
        """
        Extract tree family ID from L3 emission templates.

        Looks at tree targets across templates to identify the family.
        """
        emit_templates = ir.layers.L3_cmmm.emit_templates
        if not emit_templates:
            return None

        # Collect unique tree references
        tree_refs = set()
        for template in emit_templates.values():
            tree_refs.add(template.trees.A)
            tree_refs.add(template.trees.B)

        if not tree_refs:
            return None

        # Create stable hash from tree references
        tree_str = "|".join(sorted(tree_refs))
        hash_val = int(hashlib.sha256(tree_str.encode()).hexdigest()[:8], 16)
        return hash_val & 0x7FFFFFFF

    def _extract_tags_from_ir(self, ir: 'InstrumentIR') -> List[str]:
        """
        Extract tags from IR structure.

        Infers tags from:
        - L0 node types (has filter, has noise, etc.)
        - L2 regime kinds present
        - Compile config features
        """
        tags = []

        # Check L0 nodes
        node_types = {node.type for node in ir.layers.L0_graph.nodes}
        if "Filter" in node_types:
            tags.append("filtered")
        if "Noise" in node_types:
            tags.append("noise")
        if "FXSend" in node_types:
            tags.append("fx")
        if "FM6" in node_types:
            tags.append("fm6")
            tags.append("fm")

        # Check waveforms
        for node in ir.layers.L0_graph.nodes:
            if node.type == "Osc" and node.params.waveform:
                tags.append(node.params.waveform.lower())

        # Check regime kinds
        regime_kinds = {node.kind for node in ir.layers.L2_regimes.regime_nodes}
        if "GlitchChatter" in regime_kinds:
            tags.append("glitch")
        if "NoiseTexture" in regime_kinds:
            tags.append("texture")

        return list(set(tags))  # Remove duplicates

    def _extract_assets_from_ir(self, ir: 'InstrumentIR') -> List[AssetRef]:
        """
        Extract asset references from IR compile config.

        Maps compile.map_library entries to AssetRef objects.
        """
        assets = []

        for asset_id, asset_path in ir.compile.map_library.items():
            # Infer asset type from path/id
            if 'wavetable' in asset_id.lower() or asset_path.endswith('.wtbl'):
                asset_type = AssetType.WAVETABLE
            elif 'material' in asset_id.lower():
                asset_type = AssetType.MATERIAL
            elif 'atlas' in asset_id.lower():
                asset_type = AssetType.TABLE_ATLAS
            else:
                asset_type = AssetType.OTHER

            assets.append(AssetRef(
                asset_type=asset_type,
                asset_id=asset_id,
                asset_path=asset_path,
            ))

        return assets


__all__ = [
    'CompileResult',
    'CapsuleCompiler',
]
