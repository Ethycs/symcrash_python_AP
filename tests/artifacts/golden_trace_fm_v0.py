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
    "7f8c48aa5743c92ca36664672cb816cb71111343098c16089ccae83033aef080",
    "372742bc8521d9a156e345c08c931f1a7b22b2da9b1d7ecc7eef65fd22eedbaf",
    "3a3d330c4123b19cea33232796c5d7e9448d879ff99b23a843b1ee94ba744b8e",
    "505348305a823b870fbfcd37f64ea93c66deece71234f6774861175bee318afc",
    "b2523049cd3438228203233b46284256aa04693a6970090d9c21a691e6700f66",
    "6b34e1e2299297bb51e718b530a603a70da7e62f4cc3c745c751c7749884abc9",
    "e884409eb57eb7b1c44c55d2c6d2084d7287d895e8be8aca8540e9aac8029522",
    "75d0f2d9224118535d9989fd25d5e19cfa8e0fb7b75b89286cc87f73f0bda1e4",
    "1475e6700d7b463ea768d878ba24bb8975fc1ff5fa03ecf3ae74ad8cfeb0c3b2",
    "b53868050abcce87a90b52bb4a3cac1b7e5c06e13b12d0d4b0a7947ec2221e93",
    "1edc406131ea0482fa2fe62924a7ccd27d67315d1e561be8ca4364d71de6e3bb",
    "6c38f2f3e29e2de2daf306cdd4d44863d1d88ee95c47cb897a1c497b7dfa440f",
    "01dbc66c6b7303b919afca9a0706e76adb92029edb8eede1a5514e4f1f6d726c",
    "36a8f271f4316404b700f61ce1489ab27470963f721c4f71feb86c67560f4a0a",
    "81047dbf773dd52e6fe319e49e166b523cdfdbb5325dc56fa21a4f60b00b71d6",
    "bdb6c9b7868a6b0eb20ffbf613c246b46d972932a827adf265a22c1aeaf1b263",
    "823a07f61510a2017e58fa7aca0a03d2a5338eb677fce2a8dfb93009bc151a56",
    "2e2196dbfa489160d20cdca49c58c5d80872b4b7943a67e039f9d1a7eb74e576",
    "587400ad49932c3dc75f312e8a65fd1f08bf8474a1e14d24949311d7e0dd760a",
    "54a28586bba46b533b712035ecaaf52d83b2f47dcc7ba9b7d59a6832f7d95ef6",
    "16167367e18a0f240800b3d97c520e3bf434a50d98e40a87a584f8ef5807d511",
    "a7a25b45b28a6790d12709fa57aa019734f3ea5d67f13c5603830cac0ec7ceaa",
    "65fab3c45bc3ff48cfb937bd89e35b2399667c6c2fb514a7524804cd5f12042d",
    "41f50d0f4be09e85d94beb4d5bc16a3ced03bb139439ad05410e5e4762d911ab",
    "0da31d84d76645282ee4fb05891c1d3c45001ca994d8cd49ef850d89d23e0a54",
    "75da2d5823a50d927f963b9c3e93197127252ded4ce6e8268dfae6fcc6b8d281",
    "999e8253008dea41f36a5731114a5d2931e8a874baf7d4a836bac49e5a9fe74f",
    "db3a16c9124610df78e6f2231db6048dad40514fd32315aa6b744a06ceb36fbe",
    "770cbd5247ce754d9d0ff22efce2a4f0b7814947a3c621f6a1224fb87ca41ded",
    "13cb7087917cc64d31a26faa5ff29d77c2e19af2a8e7b94808f555ad181e7900",
    "3d62741a1b83e292b4084e79685accbcf2dcbba846c7f926d0e94bf4ee935205",
    "d0415025142a8eaafb53758de6cf901143cce6070f6674fc9ecee8bd161f079e",
    "3ebfd4ea28ef179f0d5b1ed0c0747bf61a20ea5b1d761c43e9e04c2c24d8dff5",
    "b6687cd2071b9871af45d39a583d4f3796cedac40d66a56dd656058fd86faaaf",
    "3d0fafee938e9e7d08bb2ea36f219fb962b37bf50a50794552d568997f1a9a8b",
    "be9af527319b32b429f76e30199f49a45b1db765cd539ef34f2ed96bdeff0c2f",
    "45294f152923eb2cb62c4102f48f0f037ec6c79fe2e69805047cb8f08af09c9d",
    "2a88b630df9afd041ce9242daa8a98c699f21709d922dacded3b4b0cab939d6b",
    "1feedbc7643d8e959405c39d4a17875e9108c287d2d4307e9e8d0aa9f63999a9",
    "f8c0ffada47cb0201e3702634cd63f978ad239f3712630fb6283430083d6a8b1",
    "d86f6abc887b415aae0376d1ba2422cc4c8377617b4898eaf2ae131d70ebf266",
    "565437eb026893c845b47a9c3b8db2faa188b433f5a5adeb256eebfa083f40d8",
    "4efff78c6356a90e63bfebd782b2f49df6ce0ecfdfe2e97385b6a93437906bb0",
    "180f6627a5bedc316777bebb49591b6e71e50082df480e752aa0d6b3c3c72a98",
    "d0607f96e18f006303a4f0f979935d07a89eb684dc1e18777ba94252fc434e80",
    "d3b86d958c978bf794d7ec9283e8bbdae364c26e19feb7791a117f861156672b",
    "99938f53bb3bf9c6f6e26a3de68876dc908ca5ac02dc8c1b72e2eb35a8a3c61c",
    "9e9374058fd5c9bc85b2022e1e509ffa08d18da093c928139d1ded5b97fe42b0",
]

# Final trace hash (hash of all block hashes concatenated)
# Re-frozen 2026-06-25: FM algorithm table corrected to the canonical DX7
# chart (28/32 were structurally wrong; audited vs Dexed/msfa, see
# tests/test_fm_algorithms_dx7_conformance.py). This trace uses alg 1,
# whose topology legitimately changed (old 6-op chain -> canonical pair+stack).
FROZEN_TRACE_HASH = "18eb89d33820df88023d5ec98fd75af713dc13c3ba1a23d541b5a716394b73a0"


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
        print(f'    "{h}",')
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
