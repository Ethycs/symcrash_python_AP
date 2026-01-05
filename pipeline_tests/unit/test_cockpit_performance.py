"""
CockpitEngine Performance Tests
================================

Validates that CockpitEngine tick performance is within budget.

Target: < 100 us per tick (leaves ~1200 us for audio rendering @ 1333 us/block)
"""

import time
import pytest
from dataclasses import dataclass, field as dc_field

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.control_registry import ControlID
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
from symcrash_python_ap.envelope_scheduler import ADSRSpec
from symcrash_python_ap.preset_compiler import MaterialConfig, LatchDefaults


@dataclass
class MinimalPreset:
    """Minimal preset for benchmarking."""
    preset_id: str = "bench"
    preset_version: str = "1.0"
    author: str = "Bench"
    description: str = "Benchmark preset"
    voiceframe_std: str = "WSE-STD-020-v1"
    vm_std: str = "WSE-STD-VM-HYBRID-v1"
    operator_lib_id: str = "WSE-STD-030-v2"
    latch: LatchDefaults = dc_field(default_factory=lambda: LatchDefaults(
        tree_a_id=0, tree_b_id=0, tree_blend_q15=0
    ))
    amp_env: ADSRSpec = dc_field(default_factory=lambda: ADSRSpec(
        attack_ms=10.0, decay_ms=42.0, sustain=0.75, release_ms=85.0
    ))
    material: MaterialConfig = dc_field(default_factory=lambda: MaterialConfig(
        beta_start_q15=0, beta_target_q15=24576, beta_ramp_blocks=16
    ))
    backbone: list = dc_field(default_factory=list)
    harmonic_amps_q15: list = dc_field(default_factory=lambda: [32767, 16384, 10923, 8192])

    def backbone_ops(self):
        return []

    def backbone_families(self):
        return [0] * 8


def test_cockpit_tick_performance():
    """Test that CockpitEngine.tick() is fast enough."""
    cockpit = CockpitEngine()
    cockpit.set_control(ControlID.TIMBRE, 16384)

    # Warmup
    for _ in range(100):
        cockpit.tick()

    # Benchmark
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        frame = cockpit.tick()
    elapsed = time.perf_counter() - start

    us_per_call = elapsed / iterations * 1000000

    print(f"\nCockpitEngine.tick(): {us_per_call:.2f} us per call")
    print(f"Target: < 100 us (leaves 1200+ us for audio rendering)")

    # Should be well under 100 us per tick
    assert us_per_call < 100, f"CockpitEngine.tick() too slow: {us_per_call:.2f} us (target: < 100 us)"


def test_emitter_performance():
    """Test that VoiceFrameEmitter.emit_frame() is fast enough."""
    preset = MinimalPreset()
    emitter = VoiceFrameEmitter(preset, voice_id=0)
    emitter.note_on(60, 32767)

    # Warmup
    for _ in range(100):
        emitter.emit_frame()

    # Benchmark
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        frame = emitter.emit_frame()
    elapsed = time.perf_counter() - start

    us_per_call = elapsed / iterations * 1000000

    print(f"\nVoiceFrameEmitter.emit_frame(): {us_per_call:.2f} us per call")

    # Should be fast (< 50 us)
    assert us_per_call < 50, f"VoiceFrameEmitter.emit_frame() too slow: {us_per_call:.2f} us"


def test_full_control_pipeline_performance():
    """Test full control pipeline performance (Cockpit -> Emitter)."""
    cockpit = CockpitEngine()
    preset = MinimalPreset()
    emitter = VoiceFrameEmitter(preset, voice_id=0)

    # Send note-on
    cockpit.midi_note_on(60, 100)

    # Warmup
    for _ in range(100):
        control_frame = cockpit.tick()
        emitter.consume_control_frame(control_frame)
        voice_frame = emitter.emit_frame()

    # Benchmark
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        control_frame = cockpit.tick()
        emitter.consume_control_frame(control_frame)
        voice_frame = emitter.emit_frame()
    elapsed = time.perf_counter() - start

    us_per_call = elapsed / iterations * 1000000

    print(f"\nFull control pipeline: {us_per_call:.2f} us per block")
    print(f"Target: < 200 us (leaves 1100+ us for audio rendering)")

    # Should be under 200 us total for control pipeline
    assert us_per_call < 200, f"Control pipeline too slow: {us_per_call:.2f} us (target: < 200 us)"


if __name__ == "__main__":
    print("=" * 70)
    print("COCKPIT ENGINE PERFORMANCE TESTS")
    print("=" * 70)

    test_cockpit_tick_performance()
    test_emitter_performance()
    test_full_control_pipeline_performance()

    print("\n" + "=" * 70)
    print("ALL PERFORMANCE TESTS PASSED")
    print("=" * 70)
