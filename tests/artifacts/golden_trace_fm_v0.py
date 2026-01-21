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
# FROZEN HASHES (Generated 2026-01-21)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "1e0dcf900b257b7ee417cc6c8f6d1679dcc9793303f2a8dcb993160151949af3",
    "d27228053eec9675d3996114ce4a951687b0382e1f96ee1102de87af7c0cfe79",
    "7aea42a6abe23801684961cb8abd7b52fa6fbae54a49ebdbcc64fdb242edbe32",
    "a326962928c68dcb8c018c2a7e8b5e4bca22bd48d8da589c04ef3f54034328ec",
    "bd224184b3b4bde463cae78b844965cf62aecac2988e58f3139d973444f5692c",
    "abe1eeba17d7c09f90e6b56c6f5776b9dcb62ef42bd7a9764ff0000c5f6f58e3",
    "63ab9173b6e94b46cd8074cd757df4e704bef7074525cf8f12cb1b8ed4598c18",
    "f0dd8e1690ca68db8505f4c66f9fb694d39b11d4827852fc175ac9796031033f",
    "c3b3a7d24fab7a1eb6da05a504a3138ca3e0b7363aa7d09f951d0694092f91c9",
    "46a43859c77d2786b4746eba96447779c3ec49385ae58ae6ff3d17de1a6d54f4",
    "8087f7fe77bcd1295299f5f67cfa67900f01e2f239319c86e023d771ccafadb8",
    "ae12558813f517f4baed30c4bc240065a63b7600b6b7be9eaaad939e9cced544",
    "3058fd45c01371bbcdc6f6097352a15d55640607aa33db30f2a9604e821292bf",
    "48f9053716b866b74b398daee3128d086b17b126051d685354ac33536e9a485c",
    "5c89ea590a323efec24c33fc6523c6dc8def3e1d392ff29939ece3780c082772",
    "ab895269c26d63590d71626f676fd6cc92b23193553a2a49c02ea5678be020fc",
    "9fa4e5d175a8d1700f491374e8ad717472470f0a8cae4db124647a4fa8eafcac",
    "e44cbfc21b049b418bac14e0ae34ce455f2ce46452ce0118e4e535b1ba3cb4de",
    "ac5c26ab158554a4f0ca06708cb3c7fec07abfc5392b21c5ab145f2544e60b75",
    "9efc91e6b7d0ff4b19fd7ddb883dc98c861f4d59559a2cdf751bc9eef2c8be20",
    "bb7313bff9eb3232d179182eec3479efcc0c47207deccc35aaa8f9654416e57a",
    "b1e7dc190455e232b633a9520949ecd84a8c19a3698225a8c32c8b754dfe30c1",
    "3c215857d927c38e100fceadcdb272e780103b3723043344f56a9fdaf5e2f7e4",
    "ef2b7ad6e94c6dd09f2257ee4ec30a28e12168d9bcbbb4787019e7f72812b33f",
    "10d54447acf14e11823c3f57f748536c0a029f887a125dd3609d35c178afab56",
    "bbe5c13bd83fa0a98a948a701dba661487bc7e36111c26f7275ac214bce75834",
    "661cfd5e3ff3c8ff8729d8aec84aac631410e13e8e18732489c00ec13d89d733",
    "bedf1dcdf7b880932f89b64c2b6782de40edeecd494d59f181515653568cab3c",
    "00a235620d7b5cbf90abdb71da5febc39715b6bdb09f228777285a168a788832",
    "caee057d78ed068080fe7ff01a30cf70cd6963b7d805a668cab5b30899005b8c",
    "f81f4184e030586194dbb6c766e6c03c7014802befbb84aa7551e1b0070b4ddb",
    "3703c0dfa870b933a801b93886a4ff28b823ff73447ca0d8c535d0ad6d6ae63c",
    "f0271ab22e01543d252052c3e7cb8cfc84899fdcd33a60fbf2b67c13e15c7682",
    "f48613b7e8b5c5457f3acfed7893c08ebcd1015514442a954cfc9fc4934263b1",
    "4809568c5f8957748153176d1c40041e25d414c32c655b55c88ef9ae60c167bc",
    "4b7eeaad8cb740b892a9e22c1d8231783bceb2621e60875a69af08c0bf0ff989",
    "d514608fa11b174d7abc559d3d3cced9dbc2ae10ccbad9a274b463bbdc9dc4eb",
    "596000503430a1286acf56ad8d7f2e08fd02bd78c20aa494837c513c87dcf7fa",
    "06359bcb133e1dc3ff88896ed0f874a2a687a5639dc91793a398f97b1b62a90b",
    "9fdaf088827be38814e69e3dc895a426fc3aa6eb0077cb5991d24e912cd529c9",
    "abd4e9a9cef43e2a06b713e65b9e4d649ded146605b16d6ca5eca0c56cfb7309",
    "e99374d4609cf496a68c5c9fbe82780e13a4087a37044341c27a93e015e2c6ed",
    "1025ea4d92fffc59ce6956496198ff117b1552d32915aea97c8c7fdab3b4e343",
    "a851845615cf1ff5ffd4d66006681338753e14864378c270416b9e53b4f042df",
    "c4c944d232d00c9f92cdd82a644e2530e5ce1d16d6dd22af7ef18ed45ebbbeea",
    "1871e0876a03e74d76bd62df3b9d5f49335a50a558ef74be5e64763fc1175c82",
    "3fca1c30cd2608cc08821a10d5c86dc99d792196bcefd163b60505ba390882dd",
    "c1aaa867813d26319628cee5eafed7b6132f95f9c46ca25dc16d08b9dd80c3c1",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "acf33e79c764f7babba00fd75468fb7f807cc3033bc7495c1634de0e91486c44"


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
