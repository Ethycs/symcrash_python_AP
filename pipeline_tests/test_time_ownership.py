"""
WSE-STD-060 Time Ownership Conformance Tests
=============================================

Tests for time ownership and temporal responsibility.
"""

import pytest
from src.symcrash_python_ap.time_ownership import (
    # Constants
    BLOCK_SIZE_SAMPLES,
    SAMPLE_RATE_48K,
    SAMPLE_RATE_44K,
    # Enums
    TimeDomain,
    EngineType,
    ProhibitedPattern,
    # Capabilities
    EngineCapabilities,
    ENGINE_CAPABILITIES,
    get_engine_capabilities,
    # BlockTime
    BlockTime,
    # Pattern detection
    PatternViolation,
    detect_prohibited_patterns,
    # Validation
    TimeOwnershipValidator,
    # Intent/Voice conversion
    IntentFrame,
    VoiceFrame,
    convert_intent_to_voice,
    # Determinism
    DeterminismChecker,
    # Compliance
    ComplianceReport,
    check_compliance,
    # Behavior tables
    get_engine_responsibilities,
)


# =============================================================================
# CONSTANT TESTS
# =============================================================================

class TestConstants:
    """Test normative constants per Section 2.1."""

    def test_block_size_is_64(self):
        """Block size MUST be exactly 64 samples."""
        assert BLOCK_SIZE_SAMPLES == 64

    def test_standard_sample_rates(self):
        """Standard sample rates should be defined."""
        assert SAMPLE_RATE_48K == 48000
        assert SAMPLE_RATE_44K == 44100


# =============================================================================
# TIME DOMAIN TESTS (Section 2.2)
# =============================================================================

class TestTimeDomains:
    """Test time domain classification per Section 2.2."""

    def test_all_domains_defined(self):
        """All three time domains should be defined."""
        assert TimeDomain.GESTURE_TIME is not None
        assert TimeDomain.VOICE_TIME is not None
        assert TimeDomain.BLOCK_TIME is not None

    def test_domain_values(self):
        """Time domains should have correct string values."""
        assert TimeDomain.GESTURE_TIME.value == "gesture_time"
        assert TimeDomain.VOICE_TIME.value == "voice_time"
        assert TimeDomain.BLOCK_TIME.value == "block_time"


# =============================================================================
# ENGINE CAPABILITIES TESTS (Section 4)
# =============================================================================

class TestEngineCapabilities:
    """Test engine capabilities per Section 4."""

    def test_voice_engine_is_time_aware(self):
        """Voice Engine should be allowed all time operations."""
        caps = get_engine_capabilities(EngineType.VOICE_ENGINE)

        assert caps.may_maintain_counters
        assert caps.may_maintain_clocks
        assert caps.may_track_phase
        assert caps.may_trigger_time_events
        assert caps.may_use_block_modulo
        assert caps.may_detect_regime_progress
        assert caps.may_contain_adsr
        assert caps.may_schedule_transitions
        assert not caps.must_be_stateless_temporal

    def test_symbol_engine_is_time_blind(self):
        """Symbol Engine MUST NOT perform time operations."""
        caps = get_engine_capabilities(EngineType.SYMBOL_ENGINE)

        assert not caps.may_maintain_counters
        assert not caps.may_maintain_clocks
        assert not caps.may_track_phase
        assert not caps.may_trigger_time_events
        assert not caps.may_use_block_modulo
        assert not caps.may_detect_regime_progress
        assert not caps.may_contain_adsr
        assert not caps.may_schedule_transitions
        assert caps.must_be_stateless_temporal
        assert caps.must_treat_frame_as_command

    def test_wavelet_engine_is_mechanical(self):
        """Wavelet Engine should be purely mechanical."""
        caps = get_engine_capabilities(EngineType.WAVELET_ENGINE)

        assert not caps.may_maintain_counters
        assert not caps.may_maintain_clocks
        assert not caps.may_track_phase
        assert not caps.may_contain_adsr
        assert not caps.may_schedule_transitions
        assert caps.must_be_stateless_temporal

    def test_safety_engine_is_block_shaped(self):
        """Safety Engine should be block-shaped only."""
        caps = get_engine_capabilities(EngineType.SAFETY_ENGINE)

        assert not caps.may_maintain_counters
        assert not caps.may_track_phase
        assert not caps.may_detect_regime_progress
        assert caps.must_be_stateless_temporal


# =============================================================================
# BLOCKTIME TESTS (Section 2.1)
# =============================================================================

class TestBlockTime:
    """Test BlockTime representation per Section 2.1."""

    def test_block_sample_offset(self):
        """BlockTime should calculate correct sample offset."""
        bt = BlockTime(block_index=10)
        assert bt.sample_offset == 10 * BLOCK_SIZE_SAMPLES

    def test_block_time_seconds(self):
        """BlockTime should calculate correct time in seconds."""
        bt = BlockTime(block_index=48000 // BLOCK_SIZE_SAMPLES, sample_rate=48000)
        # 48000 samples = 1 second at 48kHz
        assert abs(bt.time_seconds - 1.0) < 0.1

    def test_from_sample_quantizes(self):
        """from_sample should quantize to block boundary."""
        # Sample 100 is in block 1 (64-127)
        bt = BlockTime.from_sample(100)
        assert bt.block_index == 1

        # Sample 63 is in block 0
        bt = BlockTime.from_sample(63)
        assert bt.block_index == 0

        # Sample 64 is in block 1
        bt = BlockTime.from_sample(64)
        assert bt.block_index == 1

    def test_advance(self):
        """BlockTime should advance correctly."""
        bt = BlockTime(block_index=5)
        advanced = bt.advance(3)
        assert advanced.block_index == 8
        assert bt.block_index == 5  # Original unchanged


# =============================================================================
# PROHIBITED PATTERN TESTS (Section 6)
# =============================================================================

class TestProhibitedPatterns:
    """Test prohibited pattern detection per Section 6."""

    def test_detect_block_modulo_in_symbol_engine(self):
        """Block modulo should be detected in Symbol Engine code."""
        code = """
def process(block):
    if block % 4 == 0:
        do_something()
"""
        violations = detect_prohibited_patterns(code, EngineType.SYMBOL_ENGINE)
        assert len(violations) > 0
        assert any(v.pattern == ProhibitedPattern.BLOCK_MODULO for v in violations)

    def test_detect_phase_detection(self):
        """Phase detection should be detected in non-Voice engines."""
        code = """
def process(frame):
    if phase == 2:
        trigger_event()
"""
        violations = detect_prohibited_patterns(code, EngineType.SYMBOL_ENGINE)
        assert len(violations) > 0
        assert any(v.pattern == ProhibitedPattern.PHASE_DETECTION for v in violations)

    def test_detect_adsr_logic(self):
        """ADSR logic should be detected in non-Voice engines."""
        code = """
def process(frame):
    value = adsr(attack_time, decay_time)
"""
        violations = detect_prohibited_patterns(code, EngineType.WAVELET_ENGINE)
        assert len(violations) > 0
        assert any(v.pattern == ProhibitedPattern.ADSR_LOGIC for v in violations)

    def test_voice_engine_allows_all_patterns(self):
        """Voice Engine should be allowed all patterns."""
        code = """
def process(block):
    if block % 4 == 0:
        if phase == 2:
            value = adsr(attack_time, decay_time)
"""
        violations = detect_prohibited_patterns(code, EngineType.VOICE_ENGINE)
        assert len(violations) == 0

    def test_clean_code_no_violations(self):
        """Code without prohibited patterns should pass."""
        code = """
def process(frame):
    coeffs = frame.coefficients
    result = apply_operator(coeffs, frame.strength)
    return result
"""
        violations = detect_prohibited_patterns(code, EngineType.SYMBOL_ENGINE)
        assert len(violations) == 0


# =============================================================================
# TIME OWNERSHIP VALIDATION TESTS (Rule 060.1)
# =============================================================================

class TestTimeOwnershipValidation:
    """Test time ownership validation per Rule 060.1."""

    def test_voice_engine_with_counters_ok(self):
        """Voice Engine MAY maintain counters."""
        validator = TimeOwnershipValidator()
        violations = validator.validate_engine(
            EngineType.VOICE_ENGINE,
            has_counters=True,
            has_clocks=True,
        )
        assert len(violations) == 0

    def test_symbol_engine_with_counters_violation(self):
        """Symbol Engine MUST NOT maintain counters."""
        validator = TimeOwnershipValidator()
        violations = validator.validate_engine(
            EngineType.SYMBOL_ENGINE,
            has_counters=True,
        )
        assert len(violations) > 0
        assert "counters" in violations[0].lower()

    def test_wavelet_engine_with_adsr_violation(self):
        """Wavelet Engine MUST NOT contain ADSR."""
        validator = TimeOwnershipValidator()
        violations = validator.validate_engine(
            EngineType.WAVELET_ENGINE,
            has_adsr=True,
        )
        assert len(violations) > 0
        assert "ADSR" in violations[0]

    def test_safety_engine_with_event_scheduling_violation(self):
        """Safety Engine MUST NOT schedule events."""
        validator = TimeOwnershipValidator()
        violations = validator.validate_engine(
            EngineType.SAFETY_ENGINE,
            has_event_scheduling=True,
        )
        assert len(violations) > 0


# =============================================================================
# INTENT → VOICE FRAME TESTS (Section 5)
# =============================================================================

class TestIntentToVoiceConversion:
    """Test IntentFrame to VoiceFrame conversion per Section 5."""

    def test_intent_frame_is_time_free(self):
        """IntentFrame should not contain timing information."""
        intent = IntentFrame(
            note=60,
            velocity=0.8,
            gate=True,
        )
        # IntentFrame has no block_index, no gate_rise/fall, no regime
        assert not hasattr(intent, 'block_index')
        assert not hasattr(intent, 'gate_rise')
        assert not hasattr(intent, 'gate_fall')

    def test_voice_frame_is_time_full(self):
        """VoiceFrame should contain all timing information."""
        frame = VoiceFrame(
            voice_id=0,
            block_index=100,
            gate=True,
            gate_rise=True,
            gate_fall=False,
            regime_id=2,
        )
        assert frame.block_index == 100
        assert frame.gate_rise
        assert frame.regime_id == 2

    def test_convert_intent_to_voice(self):
        """Conversion should transfer intent and add timing."""
        intent = IntentFrame(
            note=72,
            velocity=0.9,
            gate=True,
            family_weights={'CALM': 0.8, 'TENSE': 0.2},
        )

        frame = convert_intent_to_voice(
            intent=intent,
            voice_id=5,
            block_index=200,
            gate_rise=True,
            gate_fall=False,
            regime_id=3,
            event_mask=0b00001111,
        )

        # Check identity
        assert frame.voice_id == 5
        assert frame.block_index == 200

        # Check intent transfer
        assert frame.note_u8 == 72
        assert frame.velocity_q15 == int(0.9 * 32767)
        assert frame.gate

        # Check timing (from Voice Engine)
        assert frame.gate_rise
        assert not frame.gate_fall
        assert frame.regime_id == 3
        assert frame.event_mask == 0b00001111

        # Check family weights
        assert frame.w_family_q15[0] == int(0.8 * 32767)  # CALM
        assert frame.w_family_q15[1] == int(0.2 * 32767)  # TENSE

    def test_timing_comes_from_voice_engine(self):
        """All timing decisions should come from Voice Engine parameters."""
        intent = IntentFrame(note=60, gate=True)

        # Voice Engine decides gate_rise/fall
        frame1 = convert_intent_to_voice(intent, 0, 0, gate_rise=True)
        frame2 = convert_intent_to_voice(intent, 0, 1, gate_rise=False)

        assert frame1.gate_rise
        assert not frame2.gate_rise


# =============================================================================
# DETERMINISM TESTS (Rule 060.3)
# =============================================================================

class TestDeterminism:
    """Test determinism guarantee per Rule 060.3."""

    def test_identical_frames_are_equivalent(self):
        """Identical VoiceFrames should be detected as equivalent."""
        checker = DeterminismChecker()

        frame1 = VoiceFrame(voice_id=0, block_index=10, note_u8=60, gate=True)
        frame2 = VoiceFrame(voice_id=0, block_index=10, note_u8=60, gate=True)

        assert checker.check_voice_frame_equivalence(frame1, frame2)

    def test_different_frames_not_equivalent(self):
        """Different VoiceFrames should not be equivalent."""
        checker = DeterminismChecker()

        frame1 = VoiceFrame(voice_id=0, block_index=10, note_u8=60)
        frame2 = VoiceFrame(voice_id=0, block_index=10, note_u8=72)

        assert not checker.check_voice_frame_equivalence(frame1, frame2)

    def test_reordering_preserves_content(self):
        """Reordering should not change frame content."""
        checker = DeterminismChecker()

        frames = [
            VoiceFrame(voice_id=0, block_index=0, note_u8=60),
            VoiceFrame(voice_id=1, block_index=0, note_u8=64),
            VoiceFrame(voice_id=0, block_index=1, note_u8=60),
            VoiceFrame(voice_id=1, block_index=1, note_u8=64),
        ]

        # Reorder (simulate parallel execution)
        reordered = [frames[1], frames[3], frames[0], frames[2]]

        assert checker.validate_scheduling_invariance(frames, reordered)

    def test_modified_reordering_detected(self):
        """Modified frames during reordering should be detected."""
        checker = DeterminismChecker()

        original = [
            VoiceFrame(voice_id=0, block_index=0, note_u8=60),
            VoiceFrame(voice_id=1, block_index=0, note_u8=64),
        ]

        # Reordered with modification (BUG!)
        reordered = [
            VoiceFrame(voice_id=1, block_index=0, note_u8=64),
            VoiceFrame(voice_id=0, block_index=0, note_u8=72),  # Changed!
        ]

        assert not checker.validate_scheduling_invariance(original, reordered)


# =============================================================================
# COMPLIANCE TESTS (Section 9)
# =============================================================================

class TestCompliance:
    """Test compliance checking per Section 9."""

    def test_voice_engine_compliant(self):
        """Voice Engine with time operations should be compliant."""
        report = check_compliance(
            EngineType.VOICE_ENGINE,
            has_counters=True,
            has_clocks=True,
            has_adsr=True,
        )
        assert report.is_compliant

    def test_symbol_engine_compliant_when_time_blind(self):
        """Symbol Engine without time operations should be compliant."""
        report = check_compliance(
            EngineType.SYMBOL_ENGINE,
            has_counters=False,
            has_clocks=False,
            has_adsr=False,
        )
        assert report.is_compliant

    def test_symbol_engine_non_compliant_with_counters(self):
        """Symbol Engine with counters should be non-compliant."""
        report = check_compliance(
            EngineType.SYMBOL_ENGINE,
            has_counters=True,
        )
        assert not report.is_compliant
        assert len(report.violations) > 0

    def test_compliance_with_code_analysis(self):
        """Compliance should detect prohibited patterns in code."""
        bad_code = """
def process(block):
    if block % 8 == 0:
        trigger()
"""
        report = check_compliance(
            EngineType.SYMBOL_ENGINE,
            code=bad_code,
        )
        assert not report.is_compliant
        assert len(report.pattern_violations) > 0

    def test_compliance_report_str(self):
        """Compliance report should have readable string format."""
        report = check_compliance(
            EngineType.SYMBOL_ENGINE,
            has_counters=True,
        )
        report_str = str(report)
        assert "NON-COMPLIANT" in report_str
        assert "symbol_engine" in report_str


# =============================================================================
# BEHAVIOR TABLE TESTS (Section 4)
# =============================================================================

class TestBehaviorTables:
    """Test engine behavior tables per Section 4."""

    def test_voice_engine_responsibilities(self):
        """Voice Engine should have correct responsibilities."""
        resp = get_engine_responsibilities(EngineType.VOICE_ENGINE)

        assert 'counters_clocks' in resp
        assert 'MUST maintain' in resp['counters_clocks']
        assert 'timing_quantization' in resp
        assert 'BlockTime' in resp['timing_quantization']

    def test_symbol_engine_responsibilities(self):
        """Symbol Engine should have correct responsibilities."""
        resp = get_engine_responsibilities(EngineType.SYMBOL_ENGINE)

        assert 'counters_clocks' in resp
        assert 'MUST NOT' in resp['counters_clocks']
        assert 'frame_treatment' in resp
        assert 'command' in resp['frame_treatment']

    def test_wavelet_engine_responsibilities(self):
        """Wavelet Engine should have correct responsibilities."""
        resp = get_engine_responsibilities(EngineType.WAVELET_ENGINE)

        assert 'envelopes' in resp
        assert 'MUST NOT' in resp['envelopes']
        assert 'inverse_transform' in resp
        assert 'deterministically' in resp['inverse_transform']

    def test_safety_engine_responsibilities(self):
        """Safety Engine should have correct responsibilities."""
        resp = get_engine_responsibilities(EngineType.SAFETY_ENGINE)

        assert 'idempotence' in resp
        assert 'phase_detection' in resp
        assert 'MUST NOT' in resp['phase_detection']


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for time ownership."""

    def test_full_pipeline_time_flow(self):
        """Test complete time flow from Intent to Voice."""
        # GestureTime: Async intent arrives
        intent = IntentFrame(
            note=60,
            velocity=0.75,
            gate=True,
            family_weights={'CALM': 1.0},
        )

        # VoiceTime: Voice Engine adds timing
        for block in range(5):
            gate_rise = (block == 0)  # Rise on first block only
            gate_fall = False

            frame = convert_intent_to_voice(
                intent=intent,
                voice_id=0,
                block_index=block,
                gate_rise=gate_rise,
                regime_id=2,
            )

            # BlockTime: Frame is executable
            assert frame.block_index == block
            assert frame.gate_rise == gate_rise

    def test_voice_engine_sole_owner(self):
        """Only Voice Engine should add timing to frames."""
        intent = IntentFrame(note=60, gate=True)

        # Same intent, different timing from Voice Engine
        frame1 = convert_intent_to_voice(intent, 0, 0, regime_id=1)
        frame2 = convert_intent_to_voice(intent, 0, 0, regime_id=2)

        # Timing differs based on Voice Engine state
        assert frame1.regime_id != frame2.regime_id

        # But base intent is same
        assert frame1.note_u8 == frame2.note_u8
        assert frame1.gate == frame2.gate
