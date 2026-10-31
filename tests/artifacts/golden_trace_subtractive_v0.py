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
# Block 34 regenerated 2026-10-30 for exact ties-to-even coefficient products.
# Ten of 3072 samples change, by at most four PCM units. The former half-up
# reductions reproduce every old hash; an independent Fraction generator
# reproduces the corrected Python/JIT audio. See the evidence in
# regen_reports/subtractive_arithmetic_20261030.txt. Other blocks are unchanged.
# 2026-10-30: blocks except 37 regenerated for consistent Mode B2 spectrum
# selection, exact-pitch bases and enforced harmonic caps. The former matrix
# shortcut ignored selected weights/caps. Independent authored-weight laws
# and Fraction coefficient arithmetic reproduce the corrected audio in both
# Python and JIT. See regen_reports/carrier_spectrum_20261030.txt. The other
# three frozen traces and this input frame sequence remain unchanged.
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
# Regenerated 2026-10-30: signed-RNE lifting; dimensions/headroom unchanged.
# Prior commit reproduced; Fraction oracle verifies new audio.
# Audit: regen_reports/lifting_arithmetic_20261030.txt
FROZEN_BLOCK_HASHES: List[str] = [
    "3f139e37ee545d873b5edf9879706697c3eb1094add3ef555b798d7d707a2594",
    "e830855600dd591e517424283a39e19928c0ac59e6e969163cc2ff48c792fa14",
    "ab48425b205ea337cc75cd53744deadccf6d64d3aefd3eb7af62d9e907c5cdc7",
    "9aeea9391fb1c35b3044b12d609724c656504724f8014484299ce6b920d8b3c4",
    "b0149f310e54cd754cf59ea6ade9684b38c1f91d4d88b530b7a0709cd9ebd1ce",
    "166d0c1d500c73d883f7ba73575cdbf820ba31ab4590cfce819c78bd954ac992",
    "e50983e5c700c72f1d2a960a53192985321de05bd4b6a0d38358d30ca146299e",
    "5f3aa03da43202a10811075a50a86f8f95a04e1b4f4c9be6205ef5185d2c6618",
    "1891a82f4d5a9d4d358797a0243036537641ab19251b35a7d19d31e6cc399709",
    "6ecab082f3a1cb642d7886ddbb6446e68eb386aa02d1a3b22105dc6c908bdd6f",
    "5b29423c7451ce231477bbccf0178f8c35b7c2e22d1cf66f74a7835a91141db6",
    "707087a15310366d9fc9871e8c8114ad42c64d8bd298dab607bd5911c1dc3843",
    "ac4d685ca13224823248a377c4e7b156b04364dcdaf281cf53243daf6363bbb4",
    "4c81fab6fb13a41f3801340945c7ead76e47a7376ede4ba87a228ded182318c3",
    "5445bbb0a2f93c0fb50509986c3f572ec24ad894df5195e211919f6625e0b752",
    "02505436e4fe9aa0a7856bde0b63bdad32fbeb80f8c8de36120eab155d829ac5",
    "d3e8e6619dfffb3220e83f8f0c6dfc94f5674f292ed96a68dba4895c793d27ce",
    "38805ab84a45e5b02382f81104cac831b694c3df3bf2271517635cf26a8b316d",
    "79883d35f26ac57e1be9f7d930f9301aab52dd62fa98260e91affe80cfc3729f",
    "e2a6c2923345d53f716d80e870cfd09d1d0a34a0e7708a985fb4581345d9919f",
    "2d024de47428dba13ff2a76be807bd4a49a6929443faf95292196c1b29e8fd4d",
    "c069b20c3aa92561820b28e7c0dccacdb72857f734f3a68b94732c15d3b4c4ea",
    "2b5518ec1a4727aae3e0090f3f8c9559ddfe7dd3f76fe65c61b6d234418dfb6b",
    "4a0394c34a51dd9e16914bddb4e71828704e51c5b268625236f97f1bbd786ce7",
    "d81b6533032e77f23c7662180572e260d21df9e66b3e73319d5e85a485b131bd",
    "dc66d4d657e7f21c63a7e801bee4e13d01d760efcd320877c95c6c7ff72f1f5e",
    "eccc6d3571b71dcfd2b008921d76d063fd0da4af884502f9eb913ea4b54c666b",
    "2958d8ed1ca1f3f386062ebf192d9ce49f9b2240fa45f76ccc0d66198400a3e1",
    "16d11cf790e42b102e350ce394ef56b4e98d3499e4f959316cb27ffbd1eb2e9e",
    "0dcdfdc78eecf814fc72ae2c077e7a16b6eb2a809d03002a800720684a0c1bf6",
    "ccdd1282083df902c3318e2877f55951e43a8f5b808a03a550c0e17cf251c274",
    "96a140ec1ddc84fe99dd23a454fd04a3f52291a73f645d36be10fcb46cb3b13e",
    "7c92f08c38e6987424a78d6ac30b0e99d11bb1c8b7ae114ff75f573bde7c6218",
    "0d42ca39657e59d0dc3e2612835989bdb61b74a514689152864f2db24efcd7d3",
    "10eaf6b678173e6907abb055965a7d8dccfc79aaca0580005c205c0dfd559479",
    "cf7f412a55cef967d1ebf82a883d861eb4e352e5e398db128b98f7b7de93bd40",
    "dbdde937b840c3cf9cd32b4c24750aad9a55fc95b4f29aaa37563aa94420fe21",
    "bb0a94837da158f0dc9c924a53075b969efb22a1a12d4210d249a0a357362d8c",
    "8c172542c2bb08b8b30cec14c9513f0127a03eb680d463c0b3e4d35d4471a79d",
    "f21146a2c24f760ae0f9a92b2e075ede52aad3097f9321695fb4be31157a12e8",
    "47a7ad5f8f01bfb61eab0fd7a92b3e43b7195881a2aaafbf50a84826feed63ca",
    "537f2a5e3ba1677caf576da608c2e57de5b732e531ac2440a5c0da7394750c3f",
    "d91a1d4f1ae36c00abf316d03956dd82cda51d8872a0332893f96fee38a26006",
    "f99c7c6e8cb419a06dc3ed8cc9847d818a8dd234ca07b9606a52d1e08550aee0",
    "7004feac1599bd2dfba4da7d898d42558901b68e2f0d1070d8b4ae08f2550946",
    "5b3cd0e235591bfd953f01d6f99bae1255a1dc3c37972fc025c49acef3531140",
    "13da61f427ec5d51bca5edc558c8fbace1a3970a7ad8a3a04972d07d2f368b5a",
    "a871acdfc116646ae58c79270abc4abe448b490b644635a12a00bedebe43dad8",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "40de4aec84d47b7c2b691a46e02519456e6a021d6e84a40ced1cbdeef7493c7e"


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
