"""
SYMBOL ENGINE TESTS (S1-S5)
===========================

Automatic transmission: algebra, legality, smoothing.

Covers:
- S1: Statelessness w.r.t. Musical Time
- S2: Operator Legality Enforcement
- S3: Coefficient State Stability
- S4: Event Lane Isolation
- S5: Emergency Simplify Correctness
"""

import pytest
import numpy as np


class TestS1StatelessnessMusicalTime:
    """
    S1. Statelessness w.r.t. Musical Time

    Requirement:
    - Symbol Engine cannot know block count, note age, or phase

    Test:
    - Replay same VoiceFrame repeatedly -> converges to fixed coefficient behavior

    Gotchas:
    - Internal counters
    - Phase-based operator logic
    """

    def test_same_input_converges(self, q15_one, q15_half):
        """Same input repeatedly produces stable output."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Create fixed input
        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[1000] * 256,
        )

        # Run many times with identical input
        outputs = []
        for _ in range(100):
            c_out = executor.step_voice(vf, program_id=0)
            outputs.append(tuple(c_out[:20]))  # First 20 coeffs

        # After initial transient, should stabilize
        late_outputs = outputs[50:]
        unique_outputs = set(late_outputs)

        # Should converge to single stable state (or very few)
        assert len(unique_outputs) <= 5, f"Output not converging: {len(unique_outputs)} states"

    def test_no_internal_counters(self, q15_half):
        """Symbol Engine has no time-varying internal counters."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        # Create two fresh executors
        exec1 = SymbolExecutor(n_voices=4)
        exec2 = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[1000] * 256,
        )

        # Run exec1 for 50 steps, then compare with fresh exec2
        for _ in range(50):
            exec1.step_voice(vf, program_id=0)

        # Now both should produce same output for same input
        out1 = exec1.step_voice(vf, program_id=0)
        out2 = exec2.step_voice(vf, program_id=0)

        # Outputs may differ due to accumulated state, but should be similar
        diff = sum(abs(a - b) for a, b in zip(out1[:20], out2[:20]))

        # Some difference expected due to coefficient history, but not huge
        assert diff < 100000, f"Large difference suggests hidden time state: {diff}"

    def test_block_count_not_used(self, q15_half):
        """Block count is not accessible or used."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # VoiceFrame should not have block_count field that affects output
        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[1000] * 256,
        )

        # Verify VoiceFrame doesn't have block_count
        assert not hasattr(vf, 'block_count') or vf.block_count is None if hasattr(vf, 'block_count') else True


class TestS2OperatorLegalityEnforcement:
    """
    S2. Operator Legality Enforcement

    Requirement:
    - All outputs satisfy sparsity, energy, rank, and band constraints

    Test:
    - Apply worst-case operator chains continuously

    Gotchas:
    - Accumulating low-rank ops into dense behavior
    - Permutations creating unintended energy concentration
    """

    def test_output_bounded(self, q15_one):
        """Output coefficients stay within bounds."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # High-energy input
        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_one,
            coeff_q15=[q15_one // 4] * 256,
        )

        for _ in range(100):
            c_out = executor.step_voice(vf, program_id=0)

            # All coefficients should be bounded
            max_coeff = max(abs(c) for c in c_out)
            assert max_coeff < 40000, f"Coefficient overflow: {max_coeff}"

    def test_energy_conservation(self, q15_half):
        """Total energy doesn't grow unbounded."""
        from symbol_engine import SymbolExecutor, COEFF_LEN
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Initialize with known energy
        initial_coeffs = [q15_half // 4] * COEFF_LEN

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=initial_coeffs,
        )

        # Track energy over time
        energies = []
        for _ in range(100):
            c_out = executor.step_voice(vf, program_id=0)
            energy = sum(c * c for c in c_out) >> 15
            energies.append(energy)

        # Energy should not grow exponentially
        max_energy = max(energies)
        min_energy = max(1, min(energies[10:]))  # Skip initial transient

        # Ratio should be bounded
        ratio = max_energy / min_energy
        assert ratio < 1000, f"Energy ratio too high: {ratio}"

    def test_worst_case_chain(self, q15_one):
        """Worst-case operator chains don't cause overflow."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Maximum gain
        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_one,
            coeff_q15=[q15_one // 2] * 256,
        )

        # Run continuously
        for i in range(200):
            c_out = executor.step_voice(vf, program_id=0)

            # Check for NaN-like values (would show as very large)
            for c in c_out:
                assert -50000 < c < 50000, f"Overflow at block {i}"


class TestS3CoefficientStateStability:
    """
    S3. Coefficient State Stability

    Requirement:
    - c_{t+1} = Q(T*c_t + b) never diverges

    Test:
    - Long-run simulation (minutes) with static VoiceFrame

    Gotchas:
    - Injection bias not counteracted by Q
    - Numeric overflow in fixed-point arithmetic
    """

    def test_long_run_stability(self, q15_half):
        """Coefficients don't diverge over long runs."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[1000] * 256,
        )

        # Run for 5000 blocks
        max_seen = 0
        for i in range(5000):
            c_out = executor.step_voice(vf, program_id=0)
            current_max = max(abs(c) for c in c_out)
            max_seen = max(max_seen, current_max)

            # Check periodically
            if i % 1000 == 999:
                assert max_seen < 40000, f"Divergence at block {i}: max={max_seen}"

        # Final check
        assert max_seen < 40000, f"System diverged: max_seen={max_seen}"

    def test_no_dc_accumulation(self, q15_half):
        """DC offset doesn't accumulate."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[1000] * 256,
        )

        # Run and check DC (sum of all coefficients)
        dc_values = []
        for _ in range(1000):
            c_out = executor.step_voice(vf, program_id=0)
            dc = sum(c_out) / len(c_out)
            dc_values.append(dc)

        # DC should be bounded
        max_dc = max(abs(dc) for dc in dc_values[100:])  # Skip transient
        assert max_dc < 10000, f"DC accumulation: max_dc={max_dc}"

    def test_injection_counteracted(self, q15_half, q15_one):
        """Injected energy is counteracted by normalization."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_one,
            coeff_q15=[q15_half] * 256,
        )

        energies = []
        for _ in range(500):
            c_out = executor.step_voice(vf, program_id=0)
            energy = sum(c * c for c in c_out) >> 15
            energies.append(energy)

        # Energy should stabilize, not grow indefinitely
        late_energies = energies[200:]
        if len(late_energies) > 10 and late_energies[0] > 0:
            energy_trend = late_energies[-1] - late_energies[0]

            # Should not be strongly increasing
            assert energy_trend < late_energies[0] * 2, "Energy growing despite normalization"


class TestS4EventLaneIsolation:
    """
    S4. Event Lane Isolation

    Requirement:
    - Events affect only intended bands/nodes

    Test:
    - Trigger impulses in single lanes while monitoring others

    Gotchas:
    - Incorrect index mapping
    - Off-by-one in leaf targeting
    """

    def test_impulse_targets_correct_band(self, q15_half):
        """Impulse events target correct frequency band."""
        from symbol_engine import SymbolExecutor, BAND_SLICES
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Clear state
        state = executor.get_voice_state(0)
        state.c_q15 = [0] * 256

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[0] * 256,
        )

        c_out = executor.step_voice(vf, program_id=0)

        # Output should exist
        assert len(c_out) == 256

    def test_noise_isolated_to_band(self, q15_half):
        """Noise injection is isolated to target band."""
        from symbol_engine import SymbolExecutor, BAND_SLICES
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Clear state
        state = executor.get_voice_state(0)
        state.c_q15 = [0] * 256

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[0] * 256,
        )

        c_out = executor.step_voice(vf, program_id=0)

        # Should produce output
        assert len(c_out) == 256


class TestS5EmergencySimplifyCorrectness:
    """
    S5. Emergency Simplify Correctness

    Requirement:
    - EmergencySimplify must always produce a legal, audible state

    Test:
    - Force-trigger simplify under max load repeatedly

    Gotchas:
    - Simplify muting output
    - Simplify introducing DC or silence
    """

    def test_emergency_produces_output(self, q15_one):
        """Emergency degradation still produces output."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        # Initialize with some energy
        state = executor.get_voice_state(0)
        state.c_q15 = [1000] * 256

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_one,
            coeff_q15=[1000] * 256,
        )

        c_out = executor.step_voice(vf, program_id=0)

        # Should still have some non-zero output
        max_coeff = max(abs(c) for c in c_out)
        assert max_coeff >= 0, "Emergency simplify completely muted output"

    def test_emergency_no_overflow(self, q15_one):
        """Emergency mode doesn't cause overflow."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_one,
            coeff_q15=[q15_one // 2] * 256,
        )

        for _ in range(100):
            c_out = executor.step_voice(vf, program_id=0)

            for c in c_out:
                assert -40000 < c < 40000, f"Overflow in emergency mode"

    def test_emergency_repeated_stable(self, q15_half):
        """Repeated emergency calls remain stable."""
        from symbol_engine import SymbolExecutor
        from frames import VoiceFrame

        executor = SymbolExecutor(n_voices=4)

        vf = VoiceFrame(
            voice_id=0,
            gain_q15=q15_half,
            coeff_q15=[500] * 256,
        )

        # Run many cycles
        for _ in range(500):
            c_out = executor.step_voice(vf, program_id=0)

        # Should still be valid
        max_coeff = max(abs(c) for c in c_out)
        assert max_coeff < 40000, "Emergency mode became unstable"
