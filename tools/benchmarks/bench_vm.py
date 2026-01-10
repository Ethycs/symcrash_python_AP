#!/usr/bin/env python3
"""Benchmark VM components to identify bottlenecks."""

import time
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('tools', 1)[0] + 'src')

def bench_effect_mode():
    from symcrash_python_ap.wavelet.fast_lifting import effect_mode_fused, _get_fused_wpt

    carrier = [16000] * 64
    mask = [16384] * 64

    # Warmup
    _get_fused_wpt()
    for _ in range(100):
        effect_mode_fused(carrier, mask)

    t0 = time.perf_counter()
    for _ in range(1000):
        effect_mode_fused(carrier, mask)
    t1 = time.perf_counter()
    return (t1-t0) * 1000  # us per block


def bench_effect_mode_jit():
    from symcrash_python_ap.wavelet.fast_lifting import effect_mode_jit, _get_jit_processor, NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        return None

    carrier = [16000] * 64
    mask = [16384] * 64

    # Warmup (includes JIT compilation)
    _get_jit_processor()
    for _ in range(100):
        effect_mode_jit(carrier, mask)

    t0 = time.perf_counter()
    for _ in range(1000):
        effect_mode_jit(carrier, mask)
    t1 = time.perf_counter()
    return (t1-t0) * 1000  # us per block


def test_jit_correctness():
    """Verify JIT version matches pure Python version."""
    from symcrash_python_ap.wavelet.fast_lifting import effect_mode_fused, effect_mode_jit, NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        print("Numba not available, skipping JIT correctness test")
        return True

    # Test with various inputs
    test_cases = [
        ([16000] * 64, [16384] * 64),  # Uniform
        ([i * 500 - 16000 for i in range(64)], [16384] * 64),  # Ramp
        ([16000 if i % 2 == 0 else -16000 for i in range(64)], [16384] * 64),  # Alternating
        ([16000] * 64, [i * 512 for i in range(64)]),  # Varying mask
        ([16000] * 64, None),  # No mask
    ]

    all_pass = True
    for carrier, mask in test_cases:
        fused = effect_mode_fused(carrier, mask)
        jit = effect_mode_jit(carrier, mask)

        # Allow 1 LSB difference due to int32 vs Python int precision
        max_diff = max(abs(f - j) for f, j in zip(fused, jit))
        if max_diff > 1:
            print(f"FAIL: max diff = {max_diff}")
            all_pass = False
        else:
            pass  # OK

    return all_pass

def bench_carrier():
    from symcrash_python_ap.vm.lanes import lane_carrier_render
    from symcrash_python_ap.vm.state import VMState
    from symcrash_python_ap.vm.decode import DecodedPlan, CarrierParams, CarrierRole

    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    cfg = DecodedPlan(
        tree_a=0,
        tree_b=0,
        tree_blend_q15=0,
        mask_u32=0xFFFFFFFF,
        eta_prev_q15=16384,
        eta_curr_q15=16384,
        gain_prev_q15=24576,
        gain_curr_q15=24576,
        carrier_role=CarrierRole.AUDIBLE,
        carrier_params=CarrierParams(
            phase_inc=int((440.0 / 48000) * (2**32)),
            harmonic_amps_q15=[32767, 16384, 8192, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        idwpt_enable=True,
        backbone_instrs=[],
        event_instrs=[],
        tail_ops=[],
    )

    # Warmup
    for _ in range(100):
        lane_carrier_render(state, cfg)

    t0 = time.perf_counter()
    for _ in range(1000):
        lane_carrier_render(state, cfg)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_vm_step():
    from symcrash_python_ap.vm.executor import vm_step_block
    from symcrash_python_ap.vm.state import VMState
    from symcrash_python_ap.frames import VoiceFrame

    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        f0_hz=261.63,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
    )

    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=261.63)

    # Warmup
    for _ in range(100):
        vm_step_block(state, frame)

    t0 = time.perf_counter()
    for _ in range(1000):
        vm_step_block(state, frame)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_full_render():
    """Benchmark the full pico2_live_synth render path."""
    from symcrash_python_ap.voice import VoiceEngine, GestureBasis
    from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
    from symcrash_python_ap.frames import VoiceFrame

    engine = VoiceEngine(max_voices=4)
    stage = UnifiedOutputStage(block_size=64)

    # Allocate a voice
    vid = engine.allocate_voice(program_id=1001)
    gesture = GestureBasis(gate=True, pressure_q15=24576)
    engine.note_on(vid, gesture=gesture)
    stage.voice_on(vid, 60)

    # Warmup
    for _ in range(100):
        voice_frames = engine.step_all()
        # Create VoiceFrame
        vf = voice_frames[0]
        frame = VoiceFrame(
            voice_id=vf.voice_id,
            note=60,
            cents=0,
            f0_hz=261.63,
            gain_q15=vf.gain_q15,
            pan_q15=vf.pan_q15,
            tree_a_id=vf.tree_a_id,
            tree_b_id=vf.tree_b_id,
            tree_blend_q15=vf.tree_blend_q15,
            is_active=vf.is_active,
        )
        stage.render_block(frame)

    t0 = time.perf_counter()
    for _ in range(1000):
        voice_frames = engine.step_all()
        vf = voice_frames[0]
        frame = VoiceFrame(
            voice_id=vf.voice_id,
            note=60,
            cents=0,
            f0_hz=261.63,
            gain_q15=vf.gain_q15,
            pan_q15=vf.pan_q15,
            tree_a_id=vf.tree_a_id,
            tree_b_id=vf.tree_b_id,
            tree_blend_q15=vf.tree_blend_q15,
            is_active=vf.is_active,
        )
        stage.render_block(frame)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_voice_engine():
    """Benchmark just the voice engine step."""
    from symcrash_python_ap.voice import VoiceEngine, GestureBasis

    engine = VoiceEngine(max_voices=4)

    # Allocate a voice
    vid = engine.allocate_voice(program_id=1001)
    gesture = GestureBasis(gate=True, pressure_q15=24576)
    engine.note_on(vid, gesture=gesture)

    # Warmup
    for _ in range(100):
        engine.step_all()

    t0 = time.perf_counter()
    for _ in range(1000):
        engine.step_all()
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_unified_renderer_only():
    """Benchmark just UnifiedOutputStage.render_block."""
    from symcrash_python_ap.safety.unified_renderer import UnifiedOutputStage
    from symcrash_python_ap.frames import VoiceFrame

    stage = UnifiedOutputStage(block_size=64)
    stage.voice_on(0, 60)

    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        f0_hz=261.63,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
    )

    # Warmup
    for _ in range(100):
        stage.render_block(frame)

    t0 = time.perf_counter()
    for _ in range(1000):
        stage.render_block(frame)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_decode_frame():
    """Benchmark just the decode_frame function."""
    from symcrash_python_ap.vm.decode import decode_frame
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES
    from symcrash_python_ap.frames import VoiceFrame

    frame = VoiceFrame(
        voice_id=0,
        note=60,
        cents=0,
        f0_hz=261.63,
        gain_q15=24576,
        pan_q15=0,
        tree_a_id=0,
        tree_b_id=0,
        tree_blend_q15=0,
        is_active=True,
    )

    # Warmup
    for _ in range(100):
        decode_frame(frame, DEFAULT_TABLES, 16384, 24576)

    t0 = time.perf_counter()
    for _ in range(1000):
        decode_frame(frame, DEFAULT_TABLES, 16384, 24576)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_lane_idwpt():
    """Benchmark just the IDWPT lane."""
    from symcrash_python_ap.vm.lanes import lane_idwpt_render
    from symcrash_python_ap.vm.state import VMState
    from symcrash_python_ap.vm.decode import DecodedPlan, CarrierParams, CarrierRole

    state = VMState.create(0)
    state.reset_for_note_on(f0_hz=440.0)

    cfg = DecodedPlan(
        tree_a=0,
        tree_b=0,
        tree_blend_q15=0,
        mask_u32=0xFFFFFFFF,
        eta_prev_q15=16384,
        eta_curr_q15=16384,
        gain_prev_q15=24576,
        gain_curr_q15=24576,
        carrier_role=CarrierRole.AUDIBLE,
        carrier_params=CarrierParams(
            phase_inc=int((440.0 / 48000) * (2**32)),
            harmonic_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        idwpt_enable=True,
        backbone_instrs=[],
        event_instrs=[],
        tail_ops=[],
    )

    carrier_buf = np.array([16000] * 64, dtype=np.int16)

    # Warmup
    for _ in range(100):
        lane_idwpt_render(state, cfg, carrier_buf)

    t0 = time.perf_counter()
    for _ in range(1000):
        lane_idwpt_render(state, cfg, carrier_buf)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

def bench_lane_mix():
    """Benchmark just the mix+gain lane."""
    from symcrash_python_ap.vm.lanes import lane_mix_gain
    from symcrash_python_ap.vm.decode import DecodedPlan, CarrierParams, CarrierRole

    cfg = DecodedPlan(
        tree_a=0,
        tree_b=0,
        tree_blend_q15=0,
        mask_u32=0xFFFFFFFF,
        eta_prev_q15=16384,
        eta_curr_q15=16384,
        gain_prev_q15=24576,
        gain_curr_q15=24576,
        carrier_role=CarrierRole.AUDIBLE,
        carrier_params=CarrierParams(
            phase_inc=int((440.0 / 48000) * (2**32)),
            harmonic_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        idwpt_enable=True,
        backbone_instrs=[],
        event_instrs=[],
        tail_ops=[],
    )

    carrier_buf = np.array([16000] * 64, dtype=np.int16)
    texture_buf = np.array([8000] * 64, dtype=np.int16)

    # Warmup
    for _ in range(100):
        lane_mix_gain(cfg, carrier_buf, texture_buf)

    t0 = time.perf_counter()
    for _ in range(1000):
        lane_mix_gain(cfg, carrier_buf, texture_buf)
    t1 = time.perf_counter()
    return (t1-t0) * 1000

if __name__ == "__main__":
    print("VM Component Benchmarks")
    print("=" * 50)
    print(f"Budget: 1333 us/block (64 samples @ 48kHz)")
    print()

    # Test JIT correctness first
    print("Testing JIT correctness...", end=" ")
    if test_jit_correctness():
        print("PASS")
    else:
        print("FAIL")
    print()

    results = []

    t = bench_decode_frame()
    results.append(("decode_frame", t))

    t = bench_carrier()
    results.append(("lane_carrier", t))

    t = bench_lane_idwpt()
    results.append(("lane_idwpt", t))

    t = bench_lane_mix()
    results.append(("lane_mix_gain", t))

    t = bench_effect_mode()
    results.append(("effect_mode_fused", t))

    t_jit = bench_effect_mode_jit()
    if t_jit is not None:
        results.append(("effect_mode_jit", t_jit))
        speedup = t / t_jit
        print(f"\n*** JIT speedup: {speedup:.1f}x ({t:.0f} us -> {t_jit:.0f} us) ***\n")

    t = bench_vm_step()
    results.append(("vm_step_block", t))

    t = bench_voice_engine()
    results.append(("voice_engine.step", t))

    t = bench_unified_renderer_only()
    results.append(("render_block only", t))

    t = bench_full_render()
    results.append(("full_render_path", t))

    for name, t in results:
        pct = t / 1333 * 100
        print(f"{name:20s} {t:6.1f} us/block ({pct:5.1f}% budget)")

    print()
    if t > 1333:
        print(f"WARNING: Full render exceeds budget by {t - 1333:.0f} us!")
    else:
        print(f"Headroom: {1333 - t:.0f} us ({(1333 - t) / 1333 * 100:.0f}%)")
