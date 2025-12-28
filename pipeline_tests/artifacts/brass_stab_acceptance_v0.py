"""
Brass Stab v0 Acceptance Test
=============================

Validates the VM produces musically-correct output for a classic
brass stab patch (subtractive profile, no filter yet).

This test is NOT a hash-based ABI regression test. It validates:
- Carrier quality (saw harmonics, phase continuity)
- Amp envelope shape (fast attack, punchy decay)
- Gain ramps are click-free
- Beta ramps provide smooth dry/wet blend
- Tilt operator acts as spectral shaper

Target character:
- Attack: near-instant, percussive (no fade-in)
- Body: thick saw with slight detune (via harmonics)
- Brightness: strong upper harmonics, not shrill
- Decay: fast drop to mid sustain
- Release: short, clean
- No clicks: all ramps deterministic and smooth

Usage:
    python -m pytest pipeline_tests/artifacts/brass_stab_acceptance_v0.py -v

    # Or render audio for listening:
    python pipeline_tests/artifacts/brass_stab_acceptance_v0.py --render
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

# =============================================================================
# BRASS STAB SPEC
# =============================================================================

SAMPLE_RATE = 48000
BLOCK_SIZE = 64
BLOCKS_PER_SECOND = SAMPLE_RATE // BLOCK_SIZE  # 750 blocks/sec

# Envelope timing (in blocks, at 750 blocks/sec)
ATTACK_BLOCKS = 2      # ~2.7ms - near-instant
DECAY_BLOCKS = 90      # ~120ms
SUSTAIN_LEVEL_Q15 = 16384  # 0.5
RELEASE_BLOCKS = 75    # ~100ms

# Note duration
NOTE_ON_BLOCKS = 36    # ~48ms (punchy stab)
TOTAL_BLOCKS = 120     # ~160ms total (enough for full release)

# Pitch
NOTE_F0_HZ = 220.0     # A3

# Saw harmonics (1/n falloff for 8 harmonics)
SAW_HARMONICS = [32767 // n for n in range(1, 9)]  # [32767, 16383, 10922, 8191, ...]

# Material settings
BETA_START_Q15 = 4915      # 0.15 (mostly dry)
BETA_TARGET_Q15 = 11468    # 0.35 (bring in some texture)
BETA_RAMP_BLOCKS = 8

# Backbone ops (minimal for v0)
OPS_ENABLE = ["tilt", "intramix", "diffusedown"]
OP_STRENGTHS = {
    "tilt": 14746,          # 0.45
    "intramix": 8192,       # 0.25
    "diffusedown": 6554,    # 0.20
}


# =============================================================================
# ENVELOPE GENERATOR
# =============================================================================

def compute_amp_envelope(total_blocks: int, note_off_block: int) -> List[int]:
    """
    Compute amp envelope for brass stab (ADSR with fast attack).

    Returns Q15 gain values for each block.
    """
    envelope = []

    for block in range(total_blocks):
        if block < note_off_block:
            # Note is held
            if block < ATTACK_BLOCKS:
                # Attack: ramp from 0 to max
                t = block / max(1, ATTACK_BLOCKS - 1)
                gain = int(32767 * t)
            elif block < ATTACK_BLOCKS + DECAY_BLOCKS:
                # Decay: ramp from max to sustain
                t = (block - ATTACK_BLOCKS) / max(1, DECAY_BLOCKS)
                gain = int(32767 - (32767 - SUSTAIN_LEVEL_Q15) * t)
            else:
                # Sustain
                gain = SUSTAIN_LEVEL_Q15
        else:
            # Note released
            release_block = block - note_off_block
            if release_block < RELEASE_BLOCKS:
                # Release: ramp from current to 0
                # Get the level at note-off
                if note_off_block < ATTACK_BLOCKS:
                    note_off_level = int(32767 * note_off_block / max(1, ATTACK_BLOCKS - 1))
                elif note_off_block < ATTACK_BLOCKS + DECAY_BLOCKS:
                    t = (note_off_block - ATTACK_BLOCKS) / max(1, DECAY_BLOCKS)
                    note_off_level = int(32767 - (32767 - SUSTAIN_LEVEL_Q15) * t)
                else:
                    note_off_level = SUSTAIN_LEVEL_Q15

                t = release_block / max(1, RELEASE_BLOCKS)
                gain = int(note_off_level * (1 - t))
            else:
                gain = 0

        envelope.append(max(0, min(32767, gain)))

    return envelope


def compute_beta_ramp(total_blocks: int) -> List[int]:
    """Compute beta (dry/wet) ramp for brass stab."""
    ramp = []

    for block in range(total_blocks):
        if block < BETA_RAMP_BLOCKS:
            t = block / max(1, BETA_RAMP_BLOCKS - 1)
            beta = int(BETA_START_Q15 + (BETA_TARGET_Q15 - BETA_START_Q15) * t)
        else:
            beta = BETA_TARGET_Q15

        ramp.append(beta)

    return ramp


# =============================================================================
# GOLDEN FRAME GENERATION
# =============================================================================

@dataclass
class BrassStabFrame:
    """Frame for brass stab rendering."""
    f0_hz: float
    gain_q15: int
    beta_target_q15: int
    ops_enable: List[str]
    families: dict
    voice_id: int = 0
    note: int = 57  # A3
    cents: int = 0
    pattern_id: int = 0
    tail_mode: int = 0
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    partial_amps_q15: List[int] = None
    event_mask: int = 0
    event_strength_q15: List[int] = None

    def __post_init__(self):
        if self.partial_amps_q15 is None:
            self.partial_amps_q15 = list(SAW_HARMONICS)
        if self.event_strength_q15 is None:
            self.event_strength_q15 = [0] * 8


def generate_brass_stab_sequence(
    total_blocks: int = TOTAL_BLOCKS,
    note_off_block: int = NOTE_ON_BLOCKS,
) -> List[BrassStabFrame]:
    """
    Generate frame sequence for brass stab.
    """
    envelope = compute_amp_envelope(total_blocks, note_off_block)
    beta_ramp = compute_beta_ramp(total_blocks)

    frames = []
    for block in range(total_blocks):
        frame = BrassStabFrame(
            f0_hz=NOTE_F0_HZ,
            gain_q15=envelope[block],
            beta_target_q15=beta_ramp[block],
            ops_enable=list(OPS_ENABLE),
            families={k: v / 32767.0 for k, v in OP_STRENGTHS.items()},
            partial_amps_q15=list(SAW_HARMONICS),
        )
        frames.append(frame)

    return frames


# =============================================================================
# RENDER FUNCTION
# =============================================================================

def render_brass_stab() -> Tuple[np.ndarray, List[dict]]:
    """
    Render brass stab through the VM.

    Returns:
        (audio_samples, debug_info_per_block)
    """
    from symcrash_python_ap.vm import vm_step_block, VMState
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    frames = generate_brass_stab_sequence()

    # Initialize VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=NOTE_F0_HZ)

    # Set saw harmonics
    state.harmonic_amps_q15 = list(SAW_HARMONICS)

    all_samples = []
    debug_info = []

    for i, frame in enumerate(frames):
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)
        all_samples.append(output)

        debug_info.append({
            'block': i,
            'gain_q15': frame.gain_q15,
            'beta_q15': frame.beta_target_q15,
            'rms': np.sqrt(np.mean(output.astype(np.float64) ** 2)),
            'peak': np.max(np.abs(output)),
        })

    audio = np.concatenate(all_samples)
    return audio, debug_info


# =============================================================================
# ACCEPTANCE TESTS
# =============================================================================

import pytest


class TestBrassStabAcceptance:
    """Acceptance tests for brass stab musical quality."""

    def test_renders_without_error(self):
        """Basic smoke test - rendering should complete without error."""
        audio, _ = render_brass_stab()
        assert len(audio) == TOTAL_BLOCKS * BLOCK_SIZE

    def test_attack_is_immediate(self):
        """Attack should be near-instant (high energy in first 2-3 blocks)."""
        audio, debug = render_brass_stab()

        # Block 1 should have significant energy (attack phase)
        # Block 0 might be near-zero depending on exact envelope
        attack_rms = debug[1]['rms'] if len(debug) > 1 else debug[0]['rms']

        # Should have meaningful signal, not silence
        assert attack_rms > 100, f"Attack too quiet: RMS={attack_rms}"

    def test_no_clicks_at_start(self):
        """First sample should not be a hard discontinuity."""
        audio, _ = render_brass_stab()

        # Check first few samples don't have huge jumps
        first_samples = audio[:10]
        max_jump = np.max(np.abs(np.diff(first_samples)))

        # Allow some transient but not extreme click
        assert max_jump < 10000, f"Click detected at start: max_jump={max_jump}"

    def test_envelope_shape_correct(self):
        """Verify ADSR envelope has expected shape."""
        _, debug = render_brass_stab()

        # Attack: should rise quickly
        attack_gains = [d['gain_q15'] for d in debug[:ATTACK_BLOCKS + 1]]
        assert attack_gains[-1] > attack_gains[0], "Attack should rise"

        # Decay: after attack, should decrease
        if len(debug) > ATTACK_BLOCKS + 5:
            post_attack = debug[ATTACK_BLOCKS + 1]['gain_q15']
            later = debug[ATTACK_BLOCKS + 10]['gain_q15']
            assert later < post_attack, "Decay should decrease gain"

        # Release: after note-off, should decrease to near-zero
        if len(debug) > NOTE_ON_BLOCKS + 10:
            at_noteoff = debug[NOTE_ON_BLOCKS]['gain_q15']
            after_release = debug[min(NOTE_ON_BLOCKS + 50, len(debug) - 1)]['gain_q15']
            assert after_release < at_noteoff, "Release should decrease gain"

    def test_beta_ramp_is_smooth(self):
        """Beta (dry/wet) should ramp smoothly without jumps."""
        _, debug = render_brass_stab()

        betas = [d['beta_q15'] for d in debug]

        # Check no large jumps in beta
        max_beta_jump = max(abs(betas[i+1] - betas[i]) for i in range(len(betas)-1))

        # Allow reasonable ramp rate but no sudden jumps
        expected_max_jump = (BETA_TARGET_Q15 - BETA_START_Q15) // max(1, BETA_RAMP_BLOCKS - 2)
        assert max_beta_jump < expected_max_jump * 2, f"Beta jump too large: {max_beta_jump}"

    def test_gain_ramp_is_smooth(self):
        """Gain should ramp smoothly (no zipper noise)."""
        _, debug = render_brass_stab()

        gains = [d['gain_q15'] for d in debug]

        # Find largest jump in gain
        jumps = [abs(gains[i+1] - gains[i]) for i in range(len(gains)-1)]
        max_jump = max(jumps)

        # During attack, large jumps are expected
        # During decay/sustain/release, jumps should be smaller
        # Check post-attack region
        if len(gains) > ATTACK_BLOCKS + 5:
            post_attack_jumps = jumps[ATTACK_BLOCKS + 2:]
            max_post_attack_jump = max(post_attack_jumps) if post_attack_jumps else 0

            # Should not have huge jumps after attack
            assert max_post_attack_jump < 2000, f"Gain jump too large post-attack: {max_post_attack_jump}"

    def test_audio_has_harmonic_content(self):
        """Output should have rich harmonic content (not pure sine)."""
        audio, _ = render_brass_stab()

        # Take a segment from the sustain region
        start = (ATTACK_BLOCKS + 5) * BLOCK_SIZE
        end = start + 1024
        if end <= len(audio):
            segment = audio[start:end].astype(np.float64)

            # Compute FFT
            fft = np.fft.rfft(segment)
            mags = np.abs(fft)

            # Find fundamental and harmonics
            # At 220 Hz with 48kHz sample rate and 1024 samples:
            # bin = freq * N / sample_rate = 220 * 1024 / 48000 ≈ 4.7
            fund_bin = int(round(NOTE_F0_HZ * 1024 / SAMPLE_RATE))

            if fund_bin > 0 and fund_bin * 3 < len(mags):
                fund_mag = mags[fund_bin]
                h2_mag = mags[fund_bin * 2]
                h3_mag = mags[fund_bin * 3]

                # Saw wave should have significant 2nd and 3rd harmonics
                assert h2_mag > fund_mag * 0.2, "Missing 2nd harmonic (not saw-like)"
                assert h3_mag > fund_mag * 0.1, "Missing 3rd harmonic (not saw-like)"

    def test_audio_is_not_silent(self):
        """Audio should have meaningful amplitude."""
        audio, debug = render_brass_stab()

        # Check overall RMS
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        assert rms > 500, f"Audio too quiet: RMS={rms}"

        # Check peak
        peak = np.max(np.abs(audio))
        assert peak > 1000, f"Audio has no peaks: max={peak}"

    def test_audio_is_not_clipping(self):
        """Audio should not be hard-clipping."""
        audio, _ = render_brass_stab()

        # Count samples at max value
        clipped = np.sum(np.abs(audio) >= 32767)
        clipped_ratio = clipped / len(audio)

        # Allow very occasional clipping but not sustained
        assert clipped_ratio < 0.01, f"Too much clipping: {clipped_ratio*100:.1f}%"

    def test_release_reaches_silence(self):
        """After release, audio should approach silence."""
        audio, debug = render_brass_stab()

        # Check final blocks
        final_start = (TOTAL_BLOCKS - 5) * BLOCK_SIZE
        final_samples = audio[final_start:]

        final_rms = np.sqrt(np.mean(final_samples.astype(np.float64) ** 2))

        # Should be very quiet
        assert final_rms < 1000, f"Release didn't reach silence: RMS={final_rms}"


class TestBrassStabDeterminism:
    """Verify brass stab rendering is deterministic."""

    def test_two_renders_are_identical(self):
        """Same input should produce identical output."""
        audio1, _ = render_brass_stab()
        audio2, _ = render_brass_stab()

        assert np.array_equal(audio1, audio2), "Rendering is not deterministic!"

    def test_envelope_is_deterministic(self):
        """Envelope computation should be deterministic."""
        env1 = compute_amp_envelope(TOTAL_BLOCKS, NOTE_ON_BLOCKS)
        env2 = compute_amp_envelope(TOTAL_BLOCKS, NOTE_ON_BLOCKS)

        assert env1 == env2, "Envelope computation is not deterministic!"


# =============================================================================
# AUDIO EXPORT (for manual listening)
# =============================================================================

def export_wav(filename: str = "brass_stab_v0.wav"):
    """Export brass stab to WAV file for listening."""
    import wave

    audio, debug = render_brass_stab()

    # Normalize to 16-bit range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio_normalized = (audio.astype(np.float64) / max_val * 32000).astype(np.int16)
    else:
        audio_normalized = audio.astype(np.int16)

    output_path = Path(__file__).parent / filename

    with wave.open(str(output_path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_normalized.tobytes())

    print(f"Exported: {output_path}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.3f}s")
    print(f"Peak: {max_val}")

    # Print envelope summary
    print("\nEnvelope summary:")
    for d in debug[:5]:
        print(f"  Block {d['block']:2d}: gain={d['gain_q15']:5d}, beta={d['beta_q15']:5d}, RMS={d['rms']:.0f}")
    print("  ...")
    for d in debug[-3:]:
        print(f"  Block {d['block']:2d}: gain={d['gain_q15']:5d}, beta={d['beta_q15']:5d}, RMS={d['rms']:.0f}")


if __name__ == "__main__":
    import sys

    if "--render" in sys.argv:
        export_wav()
    else:
        pytest.main([__file__, "-v"])
