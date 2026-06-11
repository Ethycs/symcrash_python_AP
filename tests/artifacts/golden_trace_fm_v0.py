"""
Golden Trace FM/PM v0 - ABI Regression Artifact
================================================

This file defines a deterministic VoiceFrame sequence using Mode FM6
oscillator for FM/PM synthesis, and the expected output hashes for the
WSE-STD-VM-HYBRID v0 VM implementation.

Purpose:
- Freeze the FM/PM synthesis ABI boundary
- Verify FM6 operator graph generates consistent output
- Catch changes that affect algorithm routing, feedback, operator levels

Usage:
    python -m pytest tests/artifacts/test_golden_trace_fm.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python tests/artifacts/golden_trace_fm_v0.py --regenerate
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import subprocess

# =============================================================================
# VERSION METADATA
# =============================================================================

GOLDEN_TRACE_VERSION = "fm-v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
OSC_MODE = "MODE_FM6 (6-Operator FM)"

VERSION_INFO = {
    "golden_trace": GOLDEN_TRACE_VERSION,
    "vm_std": VM_STD_VERSION,
    "osc_mode": OSC_MODE,
    "fm_std": "WSE-STD-FM-TOPOLOGY-v0",
    "q_bits": 15,
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

VM_VERSION = VM_STD_VERSION
TRACE_NAME = "fm_bell_evolve"
TRACE_DURATION_BLOCKS = 48  # ~64ms * 48 = ~3 seconds
SAMPLE_RATE = 48000
BLOCK_SIZE = 64


def generate_golden_sequence() -> List['VoiceFrame']:
    """
    Generate the deterministic frame sequence for FM golden trace.

    This represents an "evolving FM bell" sound:
    - Attack: Quick attack with bell-like algorithm (high mod index)
    - Sustain: Modulation index decay (bell-like envelope)
    - Release: Fade out with algorithm morph
    """
    from symcrash_python_ap.frames import VoiceFrame
    from symcrash_python_ap.primitives import Q15_ONE
    from symcrash_python_ap.vm.state import OSC_MODE_FM6

    frames = []

    # FM operator ratios: classic bell (1:1, 14:1 for inharmonic partials)
    # Q16 format: 65536 = 1.0x
    RATIO_1X = 65536      # 1.0x fundamental
    RATIO_14X = 917504    # 14.0x (inharmonic for bell)
    RATIO_2X = 131072     # 2.0x octave
    RATIO_3X = 196608     # 3.0x fifth above octave

    bell_ratios = [RATIO_1X, RATIO_14X, RATIO_1X, RATIO_2X, RATIO_3X, RATIO_1X]

    # Attack phase (blocks 0-7): Quick gain ramp with high modulation
    for i in range(8):
        gain = (i + 1) * 4096
        # High modulation on attack
        mod_level = 32767 - i * 1000  # Decay from max

        frames.append(VoiceFrame(
            note=72,  # C5 = 523 Hz (good for bell)
            coeff_osc_mode=OSC_MODE_FM6,
            gain_q15=min(32767, gain),
            fm_algorithm_id=1,  # Algorithm 1: simple 2-op FM
            fm_op_ratios=bell_ratios,
            fm_op_levels=[Q15_ONE, mod_level, 0, 0, 0, 0],
            fm_feedback_depth=0,
        ))

    # Sustain phase (blocks 8-39): Modulation decay (bell envelope)
    for i in range(32):
        # Exponential-ish decay of modulation (bell characteristic)
        # Start at ~24000, decay to ~4000
        mod_level = max(4000, 24000 - i * 625)

        frames.append(VoiceFrame(
            note=72,
            coeff_osc_mode=OSC_MODE_FM6,
            gain_q15=Q15_ONE,
            fm_algorithm_id=1,
            fm_op_ratios=bell_ratios,
            fm_op_levels=[Q15_ONE, mod_level, 0, 0, 0, 0],
            fm_feedback_depth=0,
        ))

    # Release phase (blocks 40-47): Fade out with minimal modulation
    for i in range(8):
        gain = 32767 - (i + 1) * 4096
        mod_level = 4000 - i * 400  # Continue decay

        frames.append(VoiceFrame(
            note=72,
            coeff_osc_mode=OSC_MODE_FM6,
            gain_q15=max(0, gain),
            fm_algorithm_id=1,
            fm_op_ratios=bell_ratios,
            fm_op_levels=[Q15_ONE, max(0, mod_level), 0, 0, 0, 0],
            fm_feedback_depth=0,
        ))

    return frames


# =============================================================================
# FROZEN HASHES (Regenerated 2026-06-11)
# =============================================================================
#
# REGENERATION JUSTIFICATION (2026-06-11, Packet E2 - FM6 gateway headroom):
#   The FM6->FWPT gateway now auto-normalizes carrier levels when their sum
#   exceeds 0.5 (Q15 16384), because the FWPT analysis stage saturates above
#   that and rendered ~23% H3 distortion at carrier sum 1.0. This trace uses
#   carrier level Q15_ONE (sum 1.0, non-compliant), so the gateway engages
#   and the output intentionally changed: same loudness (IDWPT's built-in 2x
#   makeup gain), H3 reduced from ~23% to <0.01%. Presets with carrier sums
#   <= 0.5 are bit-exact with the previous behavior (gateway is a no-op).
#   See vm/fm6_engine.py::fm6_gateway_level_overrides.
#
# Previous hashes (generated 2026-01-21, pre-gateway):
#   trace: acf33e79c764f7babba00fd75468fb7f807cc3033bc7495c1634de0e91486c44

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "cf6907f77d6ce4790010ee71afd553d89e74a6acb152cfea42db13fe36e232d6",
    "c3566e1f8b0f99aef0fad37d96b3099a9322cd8f0d501ac2d4a11786e6fe6c5d",
    "5e23869692d22ae96b26abe6142524a69a5d2bc20eca721b7ef1ddf6a64b3055",
    "3083b8d3eeceeadd2f876bcfe6504db4c18c530039b3777874548edda55b98c3",
    "239caa225f144cf9e7d240f67be6a6dcaf0b83eca0a7a784b6e7a87fb343c257",
    "1aaca8ce896fbe950d85beacb59a1322234cadfdcac6b36d28626f647b401f9b",
    "f9d9f7da849eaa37e7f353c0d0db1a14d5ac2bebd591bbd2b5d66018e2cb1340",
    "e0167c9ca3dd70a2ee0e9e597a851cd588c272f9a09511b95243b3e55ea8327f",
    "622be721f6f846317916598a224ddfba9adae42d10785d43d82d2593eebe64c8",
    "2a1e35a21f1a13ada9451303bd806f624401f41c49d9ad0313d4ce4e41c624c0",
    "4077f80874a1a98274b4530234ae92e1769b9bab7e41508b973c56d8c0d4ed63",
    "d090e890f7ff01fedadf8bf90a1f9e93a55d5611e7195b2eac5085a48b75019f",
    "224ad6ed63f8ebb610a7e532d9fde47975dfb41409293cc66a9fe4fd3d0622c8",
    "6daf530b26ed9be2e77270635a7e94f271ee7098f8b08ad9691cbab32a18bb7e",
    "d9434a83b97fe381439804a852dab06fe6a351e4e0f0f524cf2d6d83eaa9e3f3",
    "725dc5ed8cb50a53f5bd14e67c3db1c453796a74fd887af1fefc505540ad4b9e",
    "f94ec473c072d6fda60870c1940bd7aaaa91f8220f9d0684011411716c449196",
    "51fb6b58538409a9f8287283acc7079becca4eae157a7dba6d66e43cf8400dcc",
    "8b8f6c389abdf94c476c10deae2d512627421646d14eee59781dc44027355fcb",
    "9f034bbc7be02e1d37bb6cfa72fa48b2194d07150cd020317ae479b97766405c",
    "d1d285ed7920e57ef6efe75abdd6f3f8c5606d6cdffd0b24fdbb6bb27d76ae73",
    "001b629edc1dc140c3730b81f247e58ff776a61ccc49fb332ee5e38adf88b18d",
    "f5b308c6e106649518c2ada05be35f0b164dd97a28807432ce7b2accb3599aa7",
    "e08b4d28ec9efd5c31b1b31f1175d9916e186cd0ade2b3f6888d3bcf4a6da141",
    "dcf612b52336e1b4c17ea76a00ced181bb7e712c45af4377ece37a65913bd87f",
    "f2e0a6da4acc36632b2c4a60c9cb57a49db825f7c75da91aa94762e274d6e7d5",
    "99edfe3a7af6e8d7624fef7322b79f0e27d020ea4472ba343e8a88d7f92415c3",
    "6d52f28a983727f83b1851446837cab227901e09f9a6680b27b9314934b654c1",
    "2ccbd10f9799323d068765725f443e110292d28915c6d6830e48da11d2c37465",
    "8c7771e5bb92dc768a224cbe5978bac2c84722ce71bb9c8029a0c9031472d3d8",
    "9f9d37ca154f1b75263442385fe3c3b7375f58cba497e666165590150e4ce58d",
    "17cc9c4cc785c04bffd2685c8802280a8067a1076970718732086dd8fbde3a36",
    "09c30d56fa477657c735d36386f2d6eb72d6e2563be1a02b043a8c587d745ad0",
    "c7b6f939ca4e18c2df0d26fceff057cee10d7569435016883adb45c439755b48",
    "444110d98cac6a50263b4427fe0108ee77fc7100cff516275b8c00440e8d31f2",
    "da640f2863e9f123ebbdde2730df3a8c20c8d15186a60d07e74d288a6d387f9d",
    "2f0b70439740b416fa9febbdcb3fe0e01b1eae9a3eaaa0afabcc0c982856de61",
    "5a7b4080a40dc466b5b092d529a7dbe02e6a3cb8ef3617725b8e916e592f3e35",
    "6e411eaa79617815471a9163fb620cbb556a425582bc8a08136b9c0698ddb188",
    "6d28bd2b690c591950a1cf19b8e56f5d81fb9ec946d3fb50a6f384bd33bf1dfb",
    "5cc6a66ac3124bc080271a95182a7e4669565af0bf4d345cf86a89480c9f66f6",
    "087674431478a9c0058f9f8e44760b88383bb7f853c5e183cabc3c425bd9afac",
    "abee94536b019cd804328fdf29d3d743f582b15527844a1ac1d05eee8f49d5fd",
    "1097fef5c965e82f6c0f2a63ce987b8215fa54b605ddd03602d843a12b2681dd",
    "dc28fef9b2968ab2cce7fce69ff509c74e3fdeb334a464655b01563d28701540",
    "720895043136ea08c1bc1ae9ab580169ea568cc8ce5975f783a433bca6aeb0a3",
    "e75578124971900678c3adbffe67ec4533ef7b60c4ce21d5f685607b54f0c66e",
    "6cfbfa344893af6844073d144796926ea67f7035eec9408f531cf8f6a5491897",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "05d660651d9543c42cc343e15754b4c54c013c26b77bffbaab861fb60d55b654"


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

    # Create VM state (C5 = MIDI note 72)
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(note=72, cents=0)

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
    print("GOLDEN TRACE FM/PM REGENERATION")
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
        print("Golden Trace FM/PM v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_fm_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Print new hashes if you need to update frozen values")
