"""
Golden Trace SUBTRACTIVE v0 - ABI Regression Artifact
======================================================

This file defines a deterministic VoiceFrame sequence using Mode B (Harmonic Bank)
oscillator for subtractive synthesis, and the expected output hashes for the
WSE-STD-VM-HYBRID v0 VM implementation.

Purpose:
- Freeze the SUBTRACTIVE synthesis ABI boundary
- Verify Mode B (Harmonic Bank) generates consistent output
- Catch changes that affect waveform generation, PWM, harmonic weights

Usage:
    python -m pytest tests/artifacts/test_golden_trace_subtractive.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python tests/artifacts/golden_trace_subtractive_v0.py --regenerate
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess

# =============================================================================
# VERSION METADATA
# =============================================================================

GOLDEN_TRACE_VERSION = "subtractive-v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
OSC_MODE = "MODE_B (Harmonic Bank)"

VERSION_INFO = {
    "golden_trace": GOLDEN_TRACE_VERSION,
    "vm_std": VM_STD_VERSION,
    "osc_mode": OSC_MODE,
    "table_size": 1024,
    "q_bits": 15,
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

VM_VERSION = VM_STD_VERSION
TRACE_NAME = "subtractive_saw_sweep"
TRACE_DURATION_BLOCKS = 48  # ~64ms * 48 = ~3 seconds
SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def generate_golden_sequence() -> List['VoiceFrame']:
    """
    Generate the deterministic frame sequence for SUBTRACTIVE golden trace.

    This represents a "sawtooth filter sweep" sound:
    - Attack: Quick ramp up with full brightness
    - Sustain: Sweep through waveform types (saw -> square -> pulse)
    - Release: Fade out with PWM modulation
    """
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.primitives import Q15_ONE
    from symcrash_python_ap.vm.state import OSC_MODE_B

    frames = []

    # Attack phase (blocks 0-7): Ramp up gain with sawtooth
    for i in range(8):
        gain = (i + 1) * 4096
        frames.append(VoiceFrame(
            note=69,  # A4 = 440 Hz
            coeff_osc_mode=OSC_MODE_B,  # Mode B: Harmonic Bank
            gain_q15=min(32767, gain),
            H_request=16,
            waveform_type=3,  # SAW
            pwm_enabled=False,
            pwm_width_q15=16384,
        ))

    # Sustain phase (blocks 8-39): Waveform morphing
    for i in range(32):
        # Morph through waveforms: saw(0-10) -> square(11-20) -> pulse(21-31)
        if i < 11:
            waveform = 3  # SAW
            pwm_enabled = False
            pwm_width = 16384
        elif i < 21:
            waveform = 1  # SQUARE
            pwm_enabled = False
            pwm_width = 16384
        else:
            waveform = 2  # PULSE
            pwm_enabled = True
            # PWM sweep from 50% to 10%
            pwm_width = 16384 - (i - 21) * 1200

        frames.append(VoiceFrame(
            note=69,
            coeff_osc_mode=OSC_MODE_B,
            gain_q15=Q15_ONE,
            H_request=16,
            waveform_type=waveform,
            pwm_enabled=pwm_enabled,
            pwm_width_q15=max(3276, pwm_width),  # Min 10% duty
        ))

    # Release phase (blocks 40-47): Fade out
    for i in range(8):
        gain = 32767 - (i + 1) * 4096
        frames.append(VoiceFrame(
            note=69,
            coeff_osc_mode=OSC_MODE_B,
            gain_q15=max(0, gain),
            H_request=16,
            waveform_type=5,  # TRIANGLE for softer release
            pwm_enabled=False,
            pwm_width_q15=16384,
        ))

    return frames


# =============================================================================
# FROZEN HASHES (Generated 2026-01-21; PWM blocks 29-39 regenerated 2026-06-09:
# VoiceFrame.partial_amps_q15 default became empty, so the PWM segment now
# honors H_request=16 instead of being clamped to 8 by the old default list.
# PWM blocks 30-39 regenerated 2026-06-11: compute_pwm_weights /
# compute_pwm_harmonics no longer emit the pulse DC term (2d - 1); the
# oscillator is AC-coupled, so the duty-dependent DC offset (which became LF
# rumble under width LFOs, e.g. analog_strings_v0) was removed by design.
# Block 29 is the 50%-duty pulse (dc was already 0) and is unchanged; all
# non-PWM blocks are byte-identical. Regenerated via:
#   pixi run python tests/artifacts/golden_trace_subtractive_v0.py --regenerate)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "ce8222b6286407fbb404027a7b1470bb5d65ce094f0944d6e8ffc2afa29c692c",
    "0fbb4d19137c2db9953294ebd76d6d934da273ace2c156973f2bef57e5c46d88",
    "eb4c287a712866baa955a5c3fc5e0ee5eadf6cfe70bb7a74ffc0f9ab0aaa7040",
    "1c37850ae63260b8c0a8f852fd8a8ea27ac68442b44a27f6abbe683d96240c29",
    "1804a9f4d0ec9d42ce89d7fe31647e8c357f09a91a081b422b5b19994ca47f3a",
    "2169193e9cb516b006e796b0415a0178e85234fa8bffaa8d64a4f2f7d2c9310b",
    "81fe630ca6b43124418482c378a041adc8f53e0176049442aed546946cbed85d",
    "cf5192fbb7145bf45306bb645f3844412f80e8483f042a6847d84b37cdca2fd4",
    "a0b31e99ce8ea8d319bee4d2ecce8c132050dc843402906179d72381ea4e10f3",
    "1f25ac41d154845da5601a21d3069911a041cd15e56ac493eacb292807a3e04e",
    "f6c46cafaaaadb63a637aee0543c2d096578f08349f2afa63d470c4ffe0cc423",
    "e2a50882399a9c18b5360855f76496e30ceacad2b189f4b1afac4301db3d84c8",
    "d30c469837ef41673bf99a51a4d1c64f4e6d2602225fc98d5e3164be57e29fdd",
    "f5ae00b833b61414b34be7551d0f6aba9b1875c0861923d73109aa7073d486e2",
    "1a36a632e9710023c8e5357581ed8cb43c5bb3bd8a9a8505c6448440ab380e8f",
    "b534db8ca93887c082af10b19813918e4e867fe3da77675b9847c968d477634c",
    "4c40e188b416d53ee5af57ec0171123c7fd4be826e56933c2a006696728b0947",
    "9cb2cbe956b9b834b7f661dc55523ff83784f7c6fc25da7d09cff8a6bf4efe73",
    "432fbc5275df4c3a6ca8e765e8254d188ecec9644c19c5981a89de3dbf3a1a89",
    "39ab8ffce90ac74c65995f38e33f7653ea3a353216d8f6100bda6be7745e4fba",
    "7df2675c38dcd76a306cad174f72ccf1c7e1710041363ed1776bd724ac4ca44c",
    "01783d4453738959ca35cd92efaa28a85ac5d0b97218e7a77c5f6631dfd0e8c8",
    "1c35be70bb0fe591dd7223aa53353bce7dab2d74e6e5fadb3fe6ce0d671a545c",
    "ba05f221e6c21bf66452918fcab8135d25495f410d6cfc170dc7e27648815ce6",
    "969f4dda80ae20733e3d8c7776db336c6f0a3932db41173dc6ee21df6abbbfac",
    "be0ad5aab249c15e213c0cb04f96fe1e911ae6e4a9b4d65b05130a07c72e42d3",
    "6fd71fdd7e036c59dac70973c2d76d4baefd1c1e02a34dce971c1530a3c829b0",
    "670b6abe8b7aa3cdcda1d1856a5cfc8177557cf599fa337268b38cbf071ba89d",
    "ee03322dea9260416de7db1445611dce175db6ff1faf58d4308313175359a131",
    "bf46eecb7e9924c7b69331d76c5208199dfb8d9393519218c4d91937e0d8dac3",
    "56832602e713220ba2ab50cf7e669bd0de158741c9f95a0352f19d24ba90a60e",
    "90dc9122cb441c70b3f82ab62cba9e67986df8f9af248867f9e2f9ef1ec01975",
    "2b1f62318cd2f55e8e6bd67fd20ebea3ee87754f40004493551b386097de0f5d",
    "2a2c58f79c42bb6808923df6f83fdf777d623acf0502710cd798f3818ae2d6ab",
    "dc599cbefa2ccf437ca410529ec7c4d77916494a00e0551758628adef7b634ec",
    "706fa847fffa37c9074884cc56408f995945831e0852bb69a3802ec4c5bd1081",
    "c68a00cfa65f67602c5a747e9128309237ea3ed89dda941bde48e76e6aaa2699",
    "dbe57d8a4bc11039345d18b20a854e7de6b4f35a0bc804e16b7cfe9b808e188c",
    "39b5048da21d4d47737bc70ae9cf3f98428584e89e6c86ffa7b09873e004e21e",
    "8acf169bfa02bdb100c514d8065f71d5d32b2a77e6b2c7948f9aed4eee521598",
    "3340cc7e724003b2b6e916bec562302ea3441e364acf631496de7581d23baf6e",
    "90560c99816a91bb588b72155ba14fec51a040fe61f2d04d07846d0eb45b6071",
    "b23ac59f62a06a38783d05eb1a1641ef6cb861532ec2997d6ab7fe30d980cd17",
    "4687096e1bca00ab42662646ced0b2ce6e04a173836759bfedba0d153fc6eee1",
    "29d0abced97440196200dd2710da0537d12246c983dd0122efe6e306ce4ce9b8",
    "fd997b7259cb3dc0bfe64f8dad135f91cffaf6c792b394976bbf5e1946b968f4",
    "1471778763b4cc4a6f19d09ae99569e484eb0654e3548da3376195eaed85225d",
    "58f2ae95df07e8c3314434aea4ab5c55537c81bae0e1793bedb7e1bd63c64efc",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "190cb6fb48f7edd88393ca0f6d71f5f121106a3e6be694220c5e122aae12330a"


# =============================================================================
# HASH GENERATION
# =============================================================================

def compute_trace_hashes() -> Tuple[List[str], str]:
    """
    Run the golden trace through the VM and compute hashes.

    Returns:
        (block_hashes, trace_hash)
    """
    import numpy as np
    from symcrash_python_ap.vm import vm_step_block, VMState
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    # Create VM state (A4 = MIDI note 69)
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=69, cents=0)

    # Generate frame sequence
    frames = generate_golden_sequence()

    # Run through VM and hash each block
    block_hashes = []
    all_bytes = b''

    for frame in frames:
        state, output = vm_step_block(state, frame, DEFAULT_TABLES)

        # Hash this block
        block_bytes = output.tobytes()
        block_hash = hashlib.sha256(block_bytes).hexdigest()
        block_hashes.append(block_hash)

        # Accumulate for final hash
        all_bytes += block_bytes

    # Compute final trace hash
    trace_hash = hashlib.sha256(all_bytes).hexdigest()

    return block_hashes, trace_hash


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def regenerate_hashes():
    """
    Regenerate and print the frozen hashes.

    Run this when intentional behavior changes require updating the golden trace.
    """
    print("=" * 60)
    print("GOLDEN TRACE SUBTRACTIVE REGENERATION")
    print("=" * 60)
    print()

    # Print version info
    print("VERSION INFO:")
    for k, v in VERSION_INFO.items():
        print(f"  {k}: {v}")
    print()

    # Print trace info
    print(f"Trace: {TRACE_NAME}")
    print(f"Duration: {TRACE_DURATION_BLOCKS} blocks ({TRACE_DURATION_BLOCKS * BLOCK_SIZE / SAMPLE_RATE:.2f}s)")
    print()

    # Show old hash
    print(f"OLD TRACE HASH: {FROZEN_TRACE_HASH or '(not set)'}")

    # Compute new hashes
    block_hashes, trace_hash = compute_trace_hashes()

    # Show comparison
    print(f"NEW TRACE HASH: {trace_hash}")
    print()

    if trace_hash == FROZEN_TRACE_HASH:
        print("STATUS: UNCHANGED [OK]")
    else:
        print("STATUS: CHANGED [!]")
        print("  The VM output has changed. Review carefully before updating.")
    print()

    # Print hashes for copy-paste
    print("-" * 60)
    print("COPY-PASTE VALUES (if updating frozen hashes):")
    print("-" * 60)
    print()

    print("# Block hashes (first 5, last 5):")
    for i, h in enumerate(block_hashes[:5]):
        print(f"    # Block {i:2d}: {h}")
    print("    # ...")
    for i, h in enumerate(block_hashes[-5:], start=len(block_hashes)-5):
        print(f"    # Block {i:2d}: {h}")
    print()

    print(f"# Trace hash (all {len(block_hashes)} blocks):")
    print(f"FROZEN_TRACE_HASH = \"{trace_hash}\"")
    print()

    print("# Python list for FROZEN_BLOCK_HASHES:")
    print("FROZEN_BLOCK_HASHES = [")
    for h in block_hashes:
        print(f"    \"{h}\",")
    print("]")

    return block_hashes, trace_hash


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_hashes()
    else:
        print("Golden Trace SUBTRACTIVE v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_subtractive_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Print new hashes if you need to update frozen values")
