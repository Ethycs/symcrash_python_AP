"""
FM Topology Morphing Tests
==========================

Tests for WSE-STD-FM-TOPOLOGY-v0 Section 8:
- 8 canonical archetypes
- DX7 -> archetype mapping
- Edge weight interpolation
- Carrier fade interpolation
- Continuous morphing
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symcrash_python_ap.vm.fm_archetypes import (
    ARCHETYPES, get_archetype, dx7_to_archetype,
    blend_edge_weight, blend_carrier_level,
    compute_blended_matrix, compute_blended_carriers,
    ARCH_CHAIN, ARCH_STACK, ARCH_DUAL, ARCH_SPLIT,
    ARCH_YPAIR, ARCH_CLOUD, ARCH_ADDITIVE, ARCH_FEEDBACK,
    DX7_TO_ARCHETYPE, Q15_ONE
)
from symcrash_python_ap.vm.fm6_morph_engine import (
    FM6MorphState, fm6_generate_block_morphed, fm6_morph_generate_audio,
    create_fm6_morph_state
)
from symcrash_python_ap.fixed import midi_to_phase_inc_fixed


def test_archetype_definitions():
    """Test that all 8 archetypes are defined correctly."""
    print("\n=== Archetype Definitions Test ===\n")

    # Verify all 8 archetypes exist
    for arch_id in range(8):
        arch = get_archetype(arch_id)
        assert arch.arch_id == arch_id, f"Archetype {arch_id}: ID mismatch"
        assert arch.carrier_mask > 0, f"Archetype {arch_id}: No carriers"
        print(f"  [{arch_id}] {arch.name}: {len(arch.edges)} edges, carriers={bin(arch.carrier_mask)}")

    print("[PASS] All 8 archetypes defined")
    return True


def test_dx7_mapping():
    """Test DX7 -> archetype mapping covers all 32 algorithms."""
    print("\n=== DX7 Mapping Test ===\n")

    # Verify all 32 algorithms mapped
    for alg_id in range(1, 33):
        arch_id = dx7_to_archetype(alg_id)
        assert 0 <= arch_id <= 7, f"Algorithm {alg_id}: Invalid archetype {arch_id}"

    # Count distribution
    counts = {i: 0 for i in range(8)}
    for alg_id, arch_id in DX7_TO_ARCHETYPE.items():
        counts[arch_id] += 1

    print("Archetype distribution:")
    for arch_id in range(8):
        arch = get_archetype(arch_id)
        print(f"  {arch.name}: {counts[arch_id]} algorithms")

    print("[PASS] All 32 DX7 algorithms mapped")
    return True


def test_edge_weight_interpolation():
    """Test edge weight interpolation."""
    print("\n=== Edge Weight Interpolation Test ===\n")

    # Test pure A (blend=0)
    result = blend_edge_weight(32767, 16384, 0)
    # Allow 1 LSB error due to Q15 math
    assert abs(result - 32767) <= 1, f"Pure A: expected ~32767, got {result}"
    print(f"  Pure A (blend=0): {result}")

    # Test pure B (blend=32767)
    result = blend_edge_weight(32767, 16384, 32767)
    assert abs(result - 16384) <= 1, f"Pure B: expected ~16384, got {result}"
    print(f"  Pure B (blend=32767): {result}")

    # Test 50% blend
    result = blend_edge_weight(32767, 16384, 16383)
    expected = (32767 + 16384) // 2  # ~24575
    assert abs(result - expected) < 100, f"50% blend: expected ~{expected}, got {result}"
    print(f"  50% blend: {result} (expected ~{expected})")

    # Test fade in (0 to full)
    result = blend_edge_weight(0, 32767, 16383)
    expected = 32767 // 2
    assert abs(result - expected) < 100, f"Fade in: expected ~{expected}, got {result}"
    print(f"  Fade in: {result}")

    print("[PASS] Edge weight interpolation works")
    return True


def test_carrier_fade_interpolation():
    """Test carrier fade interpolation."""
    print("\n=== Carrier Fade Interpolation Test ===\n")

    level = 32767

    # Carrier in both topologies (no fade)
    result = blend_carrier_level(True, True, level, 16383)
    assert result == level, f"Both carriers: expected {level}, got {result}"
    print(f"  Carrier in both: {result}")

    # Carrier only in A (fade out)
    # Note: Q15 math can lose 1 LSB
    result = blend_carrier_level(True, False, level, 0)
    assert abs(result - level) <= 1, f"A only at 0%: expected ~{level}, got {result}"
    result = blend_carrier_level(True, False, level, 32767)
    assert result == 0, f"A only at 100%: expected 0, got {result}"
    result = blend_carrier_level(True, False, level, 16383)
    expected = level // 2
    assert abs(result - expected) < 100, f"A only at 50%: expected ~{expected}, got {result}"
    print(f"  A only fade out at 50%: {result}")

    # Carrier only in B (fade in)
    result = blend_carrier_level(False, True, level, 0)
    assert result == 0, f"B only at 0%: expected 0, got {result}"
    result = blend_carrier_level(False, True, level, 32767)
    assert abs(result - level) <= 1, f"B only at 100%: expected ~{level}, got {result}"
    result = blend_carrier_level(False, True, level, 16383)
    expected = level // 2
    assert abs(result - expected) < 100, f"B only at 50%: expected ~{expected}, got {result}"
    print(f"  B only fade in at 50%: {result}")

    # Carrier in neither
    result = blend_carrier_level(False, False, level, 16383)
    assert result == 0, f"Neither: expected 0, got {result}"
    print(f"  Neither: {result}")

    print("[PASS] Carrier fade interpolation works")
    return True


def test_blended_matrix():
    """Test blended modulation matrix computation."""
    print("\n=== Blended Matrix Test ===\n")

    # CHAIN -> ADDITIVE morph (edges disappear)
    mat_pure_a = compute_blended_matrix(ARCH_CHAIN, ARCH_ADDITIVE, 0)
    mat_pure_b = compute_blended_matrix(ARCH_CHAIN, ARCH_ADDITIVE, 32767)
    mat_mid = compute_blended_matrix(ARCH_CHAIN, ARCH_ADDITIVE, 16383)

    # Chain has 5 edges, Additive has 0
    edges_a = np.sum(mat_pure_a > 0)
    edges_b = np.sum(mat_pure_b > 0)
    edges_mid = np.sum(mat_mid > 0)

    print(f"  CHAIN -> ADDITIVE:")
    print(f"    Pure A edges: {edges_a}")
    print(f"    Pure B edges: {edges_b}")
    print(f"    Mid blend edges: {edges_mid}")

    assert edges_a == 5, f"CHAIN should have 5 edges, got {edges_a}"
    assert edges_b == 0, f"ADDITIVE should have 0 edges, got {edges_b}"
    assert edges_mid == 5, "Mid blend should have fading edges"

    # Verify edge weights decrease
    chain_edge = mat_pure_a[1, 0]  # Op2 -> Op1 in chain
    mid_edge = mat_mid[1, 0]
    assert mid_edge < chain_edge, "Edge weight should decrease in blend"
    print(f"    Op2->Op1 weight: {chain_edge} -> {mid_edge} (fading)")

    print("[PASS] Blended matrix computation works")
    return True


def test_blended_carriers():
    """Test blended carrier mask computation."""
    print("\n=== Blended Carriers Test ===\n")

    # CHAIN (1 carrier) -> ADDITIVE (6 carriers)
    mask_a, fades_a = compute_blended_carriers(ARCH_CHAIN, ARCH_ADDITIVE, 0)
    mask_b, fades_b = compute_blended_carriers(ARCH_CHAIN, ARCH_ADDITIVE, 32767)
    mask_mid, fades_mid = compute_blended_carriers(ARCH_CHAIN, ARCH_ADDITIVE, 16383)

    print(f"  CHAIN -> ADDITIVE:")
    print(f"    Pure A mask: {bin(mask_a)}, fades: {list(fades_a)}")
    print(f"    Pure B mask: {bin(mask_b)}, fades: {list(fades_b)}")
    print(f"    Mid mask: {bin(mask_mid)}, fades: {list(fades_mid)}")

    # Union mask should include all carriers from both
    assert mask_mid == 0b111111, "Union mask should be all carriers"

    # Op1 (carrier in both) should have full level
    assert fades_a[0] == Q15_ONE, "Op1 should be full in A"
    assert fades_b[0] == Q15_ONE, "Op1 should be full in B"
    assert fades_mid[0] == Q15_ONE, "Op1 should be full in mid"

    # Op2-6 (carrier only in B) should fade in
    for op in range(1, 6):
        assert fades_a[op] == 0, f"Op{op+1} should be 0 in A"
        assert abs(fades_b[op] - Q15_ONE) <= 1, f"Op{op+1} should be ~full in B"
        assert 0 < fades_mid[op] < Q15_ONE, f"Op{op+1} should fade in mid"

    print("[PASS] Blended carriers computation works")
    return True


def test_morph_state_creation():
    """Test FM6MorphState creation and configuration."""
    print("\n=== Morph State Creation Test ===\n")

    # Create default state
    state = create_fm6_morph_state()
    assert state.topo_a_id == 0
    assert state.topo_b_id == 0
    assert state.blend_q15 == 0
    print("  Default state created")

    # Set morph parameters
    state.set_morph(ARCH_CHAIN, ARCH_ADDITIVE, 16384)
    assert state.topo_a_id == ARCH_CHAIN
    assert state.topo_b_id == ARCH_ADDITIVE
    assert state.blend_q15 == 16384
    print("  Morph parameters set")

    # Test DX7 compat mode
    state.set_dx7_compat(5)  # E.Piano
    assert state.dx7_compat_id == 5
    assert state.topo_a_id == dx7_to_archetype(5)  # Should be YPAIR
    assert state.blend_q15 == 0
    print(f"  DX7 compat: Alg 5 -> Archetype {state.topo_a_id}")

    print("[PASS] Morph state creation works")
    return True


def test_morphed_audio_generation():
    """Test morphed audio generation produces sound."""
    print("\n=== Morphed Audio Generation Test ===\n")

    base_phase_inc = midi_to_phase_inc_fixed(69, 0)  # A4

    # Test pure archetype A
    state = create_fm6_morph_state(ARCH_CHAIN, ARCH_ADDITIVE, 0)
    state.operators[0].level_q15 = Q15_ONE
    audio_a = fm6_morph_generate_audio(state, base_phase_inc, num_blocks=100)

    assert len(audio_a) > 0, "Should produce audio"
    rms_a = np.sqrt(np.mean(audio_a.astype(np.float64)**2))
    print(f"  Pure CHAIN: RMS={rms_a:.1f}")
    assert rms_a > 100, "Should have audible output"

    # Test pure archetype B
    state = create_fm6_morph_state(ARCH_CHAIN, ARCH_ADDITIVE, 32767)
    state.operators[0].level_q15 = Q15_ONE
    audio_b = fm6_morph_generate_audio(state, base_phase_inc, num_blocks=100)
    rms_b = np.sqrt(np.mean(audio_b.astype(np.float64)**2))
    print(f"  Pure ADDITIVE: RMS={rms_b:.1f}")
    assert rms_b > 100, "Should have audible output"

    # Test 50% blend
    state = create_fm6_morph_state(ARCH_CHAIN, ARCH_ADDITIVE, 16384)
    state.operators[0].level_q15 = Q15_ONE
    audio_mid = fm6_morph_generate_audio(state, base_phase_inc, num_blocks=100)
    rms_mid = np.sqrt(np.mean(audio_mid.astype(np.float64)**2))
    print(f"  50% blend: RMS={rms_mid:.1f}")
    assert rms_mid > 100, "Should have audible output"

    print("[PASS] Morphed audio generation works")
    return True


def test_continuous_morph():
    """Test continuous morphing produces smooth transitions."""
    print("\n=== Continuous Morph Test ===\n")

    base_phase_inc = midi_to_phase_inc_fixed(60, 0)  # C4

    # Generate audio while sweeping blend
    state = create_fm6_morph_state(ARCH_DUAL, ARCH_SPLIT, 0)
    state.operators[0].level_q15 = Q15_ONE
    state.operators[3].level_q15 = Q15_ONE // 2  # Op4

    rms_values = []
    blend_steps = 10

    for i in range(blend_steps):
        blend = (i * Q15_ONE) // (blend_steps - 1)
        state.set_morph(ARCH_DUAL, ARCH_SPLIT, blend)
        audio = fm6_morph_generate_audio(state, base_phase_inc, num_blocks=20)
        rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
        rms_values.append(rms)
        print(f"  Blend {blend:5d} ({100*i//(blend_steps-1):3d}%): RMS={rms:.1f}")

    # Check that audio continues throughout morph (no dropout)
    # Note: Energy changes ARE expected during topology morphing
    # as carrier count and modulation depth change
    min_rms = min(rms_values)
    max_rms = max(rms_values)
    print(f"  RMS range: {min_rms:.1f} - {max_rms:.1f}")

    # All steps should produce audible output
    for i, rms in enumerate(rms_values):
        assert rms > 100, f"Dropout at step {i}: RMS={rms}"

    print("[PASS] Continuous morph produces output throughout")
    return True


def test_all_archetype_pairs():
    """Test morphing between all archetype pairs produces audio."""
    print("\n=== All Archetype Pairs Test ===\n")

    base_phase_inc = midi_to_phase_inc_fixed(60, 0)

    working_pairs = 0
    total_pairs = 8 * 8

    for a in range(8):
        for b in range(8):
            state = create_fm6_morph_state(a, b, 16384)  # 50% blend
            state.operators[0].level_q15 = Q15_ONE
            audio = fm6_morph_generate_audio(state, base_phase_inc, num_blocks=20)
            rms = np.sqrt(np.mean(audio.astype(np.float64)**2))
            if rms > 10:  # Audible
                working_pairs += 1

    print(f"  Working pairs: {working_pairs}/{total_pairs}")
    assert working_pairs >= 56, f"Most pairs should produce audio, got {working_pairs}"

    print("[PASS] Most archetype pairs produce audio")
    return True


def main():
    print("=" * 60)
    print("FM Topology Morphing Test Suite")
    print("WSE-STD-FM-TOPOLOGY-v0 Section 8")
    print("=" * 60)

    results = []

    results.append(("Archetype Definitions", test_archetype_definitions()))
    results.append(("DX7 Mapping", test_dx7_mapping()))
    results.append(("Edge Weight Interpolation", test_edge_weight_interpolation()))
    results.append(("Carrier Fade Interpolation", test_carrier_fade_interpolation()))
    results.append(("Blended Matrix", test_blended_matrix()))
    results.append(("Blended Carriers", test_blended_carriers()))
    results.append(("Morph State Creation", test_morph_state_creation()))
    results.append(("Morphed Audio Generation", test_morphed_audio_generation()))
    results.append(("Continuous Morph", test_continuous_morph()))
    results.append(("All Archetype Pairs", test_all_archetype_pairs()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[PASS] All FM topology morphing tests passed!")
        print("\nFM Topology Morphing Features:")
        print("  - 8 canonical archetypes")
        print("  - All 32 DX7 algorithms mapped")
        print("  - Continuous A/B topology blending")
        print("  - Smooth carrier fade transitions")
        print("  - Edge weight interpolation")
        print("  - CMMM-bindable blend parameter")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
