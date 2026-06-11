"""
Golden Trace v0 - ABI Regression Artifact
==========================================

This file defines a deterministic VoiceFrame sequence and the expected
output hashes for the WSE-STD-VM-HYBRID v0 VM implementation.

Purpose:
- Freeze the ABI boundary after implementation is stable
- Catch changes that are "numerically valid" but musically different
- Ensure bit-exact reproducibility across versions

Usage:
    python -m pytest pipeline_tests/artifacts/test_golden_trace.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python pipeline_tests/artifacts/golden_trace_v0.py --regenerate
"""

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# =============================================================================
# VERSION METADATA (Prevents silent ABI drift)
# =============================================================================

GOLDEN_TRACE_VERSION = "v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
VOICEFRAME_STD_VERSION = "WSE-STD-020 v1"
SINE_CORE_LUT_VERSION = "1024-Q15-12bit-interp"

VERSION_INFO = {
    "golden_trace": GOLDEN_TRACE_VERSION,
    "vm_std": VM_STD_VERSION,
    "voiceframe_std": VOICEFRAME_STD_VERSION,
    "sine_core_lut": SINE_CORE_LUT_VERSION,
    "table_size": 1024,
    "q_bits": 15,
    "frac_bits": 12,
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

# VM version this trace is frozen for (legacy alias)
VM_VERSION = VM_STD_VERSION

# Trace metadata
TRACE_NAME = "brass_stab_440hz"
TRACE_DURATION_BLOCKS = 48  # ~64ms * 48 = ~3 seconds
SAMPLE_RATE = 48000
BLOCK_SIZE = 64


@dataclass
class GoldenFrame:
    """Minimal frame definition for golden trace."""
    f0_hz: float
    gain_q15: int
    beta_target_q15: int
    ops_enable: List[str]
    families: dict
    # Required fields with defaults
    voice_id: int = 0
    note: int = 69  # A4
    cents: int = 0
    pattern_id: int = 0
    tail_mode: int = 0  # FULL
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0
    partial_amps_q15: List[int] = None
    event_mask: int = 0
    event_strength_q15: List[int] = None

    def __post_init__(self):
        if self.partial_amps_q15 is None:
            self.partial_amps_q15 = [32767, 16384, 8192, 4096]
        if self.event_strength_q15 is None:
            self.event_strength_q15 = [32767] * 8


def generate_golden_sequence() -> List[GoldenFrame]:
    """
    Generate the deterministic frame sequence for the golden trace.

    This represents a "brass stab" sound:
    - Attack: Quick ramp up with tilt
    - Sustain: Full gain with intramix
    - Release: Fade out
    """
    frames = []

    # Attack phase (blocks 0-7): Ramp up gain, apply tilt
    for i in range(8):
        gain = (i + 1) * 4096  # 4096 -> 32768
        frames.append(GoldenFrame(
            f0_hz=440.0,
            gain_q15=min(32767, gain),
            beta_target_q15=0,  # Start dry
            ops_enable=['tilt'],
            families={'tilt': 0.7},
        ))

    # Sustain phase (blocks 8-39): Full gain, wet blend in, intramix
    for i in range(32):
        # Wet blends in from 0 to ~75% over first 16 blocks
        if i < 16:
            beta = i * 1536  # 0 -> 24576
        else:
            beta = 24576

        frames.append(GoldenFrame(
            f0_hz=440.0,
            gain_q15=32767,
            beta_target_q15=beta,
            ops_enable=['tilt', 'intramix'],
            families={'tilt': 0.5, 'intramix': 0.8},
        ))

    # Release phase (blocks 40-47): Fade out
    for i in range(8):
        gain = 32767 - (i + 1) * 4096
        frames.append(GoldenFrame(
            f0_hz=440.0,
            gain_q15=max(0, gain),
            beta_target_q15=24576,  # Keep wet
            ops_enable=['tilt'],
            families={'tilt': 0.3},
        ))

    return frames


# =============================================================================
# FROZEN HASHES (Generated 2026-01-01 - Audio path math fixes)
# Regenerated 2026-06-11: hashes were stale since the engine commits
# d84ffc5/74bc979 (partials fix, vibrato phase continuity, waveform
# dispatch, expressive dynamics) intentionally changed VM output but only
# the subtractive trace was regenerated at the time. Verified identical
# on the JIT and pure-Python paths (SYMCRASH_VM_REFERENCE=1) before
# freezing. Regenerated via:
#   pixi run python tests/artifacts/<this file> --regenerate
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
# Updated 2026-01-02: Orthogonal residual projection for wet/dry blend (linear residual instead of equal-power)
# Updated 2026-01-02: Phase-coherent wet path (mask smoothing + C⁰ seam stitching, no windowing)
# Updated 2026-01-03: Disable DC removal for VM path (CCS is phase-continuous with zero DC)
# Updated 2026-01-05: Skip SheafProject in CCS mode (was corrupting polyphase coefficients)
FROZEN_BLOCK_HASHES: List[str] = [
    "a70987fa5b7816a7b4569a152f0ba8efb19b4f15acadb3504281e05e2e2a95bf",
    "3a7cb241d090db26ae8e791451a34331f104f0f842eb25b57f8c3fde7abf8506",
    "bfe0100ff1c8f8c0077ccd6a9db1f1230009d5ce193ddcab5e52cfc42047dd92",
    "8df4823a6a45d7f8f07d60c45c7e20e0f45aca5e366fbd3e24966fdbebaad450",
    "278c3a3623e5925f6cabc2b388ffb27002a6a49b1d7af4afc820b94716f3db0d",
    "c5305df53012e13d965d4a887a0ffb67ef52face98391b6e0adaa8cfe244da59",
    "e9a72189a60662a980de5f1147d089a6b44140fbcd63a342f3c71e294b1c63bc",
    "9afc5c04466ad0135ce5c397e500139c9ccd2ef215254087afde7a31072f47e4",
    "dfe0ebf7613bfa81922a9bd45f94e6f99fafefc8c518038cd94ae4e85c94910e",
    "ce005c4dd520a767b24e1326ff4078635d0ca0bc0d2e014d9a590108abaf9484",
    "55d8a1e7778d4938154d8f7090c86e63f3d53d091fedd240e35e962bff7a9fe7",
    "a11a5f1b66333e82f52b5de6c2ab3f08726d6c2981bc08c0d37fbf48719b0001",
    "87ba6e4e71b09ab1227ab2519d28e23488f7e8bed94fec451cb27f8828fe4a3b",
    "48040c0cc013653a1dc4333be33dec7ba3fc53ca98ba74e48e514d724f65c2d8",
    "40e434b7cb7d763db6305f38364f04569ee7649ce00fc10b394b8d165caa8236",
    "a5d9ad8e754962c240d480e04690f54d1a41b2b7ee8dab4c12dcd7a8650707ba",
    "b34321c907e2db0848b032ce7ed88937e0da1432e71cf9f0479d6baa330dede1",
    "b4546a48034134af3464c8d839c080235b080163d755212c01a4bb2bc5f29d09",
    "73e67e420b13dc078463a7b240a1ec704563d6bbfa81d63d52196a9f7bf33613",
    "4f0eaaa93d36286b5beaa8bacc3eec67e32c0a35f18f8963ad759ffb3b33e77d",
    "3845e628abbf210794b6ff50fdbbb76b7ffd87664ce264741ebfa9ba60d46952",
    "0aa7797095710641d5f2fc273c0bcf33ed7986e67104105ab584fd014dc96adb",
    "8620d1163824f1bcb629aa4eb77da216628337350df70dccedff8dd3104171de",
    "0770ab2f1d4a90400cf77c27c3a58030f23f233c0fb7741662f3c9a18084cd9f",
    "6e8c8bca00eb40bbbb259896aa8dbd0ae939606152da774ea8b2bd2a565b4c27",
    "d52da4849c624e9dc59e2f7688d054e8013d257aef7d37efcf35e0dcc58cf3c7",
    "d03abde9b918fb2cf860ca56f411ba33f78b9f60c10259a4712f4f18220b7e7a",
    "e56bb386a1aa8d6196b2ddca4e0522cd74184140b3acf26303242adee4880664",
    "a9794825967921db63d0906239f232c8e85ea61151163404578f08f0a0bc5db7",
    "3bbd070edccdf4cc60289e7f6e9decb3ef1138b1bca9c6a6453e5f39b6746bb3",
    "7b79ca5ffffb829bae596ae61221fc270dc4dff49c918736f15a7cf6aaa7771e",
    "361b7e3fcdccf7744af774383b54655341bd610998b1d73104e89defc72ff17c",
    "5e5b5d7d2d7e9c598366f3b8506b7c784faaead3af73e7bf3da68bdb6721b9e2",
    "b88a5093bee8caee2ebc494fc46ae80f416b9f289f27dc16b3ea331040bba314",
    "38423e49bea16b0b46d88c14fc2bc5dec983dd0b278666e6b07cc9b87156d839",
    "914d104bd6626becd1bab9beb1df5f002629eea251bbf0b8045af4a9875261af",
    "27a98c833ae8888fffef8e62c70f6e5cf0a12c72ef8c75a6e24ad7ab67ee5eb7",
    "86b99dabbc5d1a95becc3f8847c674c46cb16e1bc715369a6e595e7e7caa8d31",
    "0dbf0bb386066ad22e9a375c8778aff0f9be4cc7f66edf187b970f44822c41af",
    "b98b846690711ef13055b6ad1df0dfba346c9b95bf5b1ad7b5ed3c0678320c59",
    "d237c2e94f3e211181b0680c8a9712ba768f932c745395ed82611de4f0179382",
    "ac44a500748448b547be130e372aead4c27f49dfa07a613dc7eaf738fae84545",
    "1117cd7338952be464ee47f72066faed2dea5f9939d6fa5f35c50a56380a3440",
    "f3da270666e8d4076980e751fd175313590f4a8d9df612c49f459d572c6e7d9e",
    "c0a9336d26aaca45e31fe27820c8a3c8ca1b3c574917e81f052d71bee39be7d3",
    "71c7ff75834c805d71ece72c51c6de1a26e14c4ebf2ba3642f27762895b03c97",
    "293d2c0d95264e399d865bde993f493bdaf31c23000e38b8dfe0b53ed9957e4b",
    "c61ccf9322974d85e4b94939970abe9b36cda4b23ec87170a7d0484fb7490602",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "0617373e40ea379e14db73329feb8dc6cf85a249d05aa0a1cec182bf944bfd23"


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

    # Create VM state (A4 = MIDI note 69, no cents)
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
            return result.stdout.strip()[:12]  # Short hash
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _write_regen_report(
    old_hash: str,
    new_hash: str,
    block_hashes: List[str],
) -> Optional[str]:
    """
    Write a regeneration report file for audit trail.

    Returns:
        Path to the report file, or None if failed.
    """
    # Create reports directory
    reports_dir = Path(__file__).parent / "regen_reports"
    try:
        reports_dir.mkdir(exist_ok=True)
    except OSError:
        return None

    # Generate timestamp
    timestamp = datetime.now()
    filename = f"regen_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = reports_dir / filename

    # Build report
    git_commit = _get_git_commit()
    status = "UNCHANGED" if old_hash == new_hash else "CHANGED"

    report_lines = [
        "=" * 60,
        "GOLDEN TRACE REGENERATION REPORT",
        "=" * 60,
        "",
        f"Timestamp:    {timestamp.isoformat()}",
        f"Git Commit:   {git_commit or 'N/A'}",
        f"Trace Name:   {TRACE_NAME}",
        f"Trace Ver:    {GOLDEN_TRACE_VERSION}",
        "",
        "--- VERSION INFO ---",
    ]
    for k, v in VERSION_INFO.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.extend([
        "",
        "--- HASH COMPARISON ---",
        f"Old Hash: {old_hash}",
        f"New Hash: {new_hash}",
        f"Status:   {status}",
        "",
        f"Block Count: {len(block_hashes)}",
        "",
    ])

    # Find first diverging block if changed
    if old_hash != new_hash:
        report_lines.append("--- DIVERGENCE ANALYSIS ---")
        report_lines.append("(Cannot determine exact block without previous block hashes)")
        report_lines.append("")

    report_lines.append("=" * 60)

    # Write report
    try:
        report_path.write_text("\n".join(report_lines))
        return str(report_path)
    except OSError:
        return None


def regenerate_hashes():
    """
    Regenerate and print the frozen hashes.

    Run this when intentional behavior changes require updating the golden trace.
    Writes an audit report to regen_reports/ for traceability.
    """
    print("=" * 60)
    print("GOLDEN TRACE REGENERATION")
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
    print(f"OLD TRACE HASH: {FROZEN_TRACE_HASH}")

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

    # Write audit report
    report_path = _write_regen_report(FROZEN_TRACE_HASH, trace_hash, block_hashes)
    if report_path:
        print(f"Audit report written to: {report_path}")
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
        print("Golden Trace v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Write an audit report to regen_reports/")
        print("  4. Print new hashes if you need to update frozen values")
