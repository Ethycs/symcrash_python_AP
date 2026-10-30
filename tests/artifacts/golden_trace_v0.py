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

# Regenerated 2026-10-30: corrected interpolation/gate arithmetic.
# Audit: regen_reports/operator_math_20261030.txt
# Old arithmetic reproduces the prior hashes; corrected Python/JIT agree.
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
    "a1e36338defeb9fdf2d56c339f3c5c673ffa7ff7803b2eb8177345fd7485f2b9",
    "d6c4e36481a7c01607256a96a810342945bb202c7661097e3b5fe06841f5e2d9",
    "18de4970553bac679df275184df0c99397802a163523d1fc94119321f1d5253d",
    "89a79c835324db1d81d242d5ece7c031560228488278ca89bcbf6c755fdd3028",
    "3691604b309a69fbd7fa2c8f95071eed0a869385770457507dec7e07572b9b38",
    "86780e42ac94059c93cbd4e7091397198b0c477b34b1e1724012b655f2664a46",
    "49716f5b24ae9ef76bb22d019a084dc3f1074252e0e5101dae61ad89e08c0981",
    "311a1d28c1b38b5c7a24c4d8d4beef28d38ec976f9e38ee93dc92b8537506d60",
    "8a75dab466c0d014c221ee58869c307740184c3ea2876732a71ec3a7738134cf",
    "73d1af61751de1135cb65a419b8725190837e34594582e42df0f9bf86bc1853d",
    "676d8eaf0f00139c0a43fc1a8b56d2525d5ab1b05fcfaea7b82c2ceaaf5dc831",
    "640e6b9c28082361401e8985aa97055a56bd7c04db87a450dc4744c457d7c061",
    "40dd8ce2c09f3903b0bc8c8492240a5f43a1755b2e7d80db802196a2abf5785b",
    "e41ed0b40e9c1f801951d53d6a8078a82cf0a6b75f18689aad4f8b18d1763150",
    "f086c80636a98ad0e9c577e9946d47ead0b63d615ccea8f6f5ec9c615200bc73",
    "32a3622dc59cb419222290c0e2391e8b81450d2b4b4f9acdfdef1577009d05ed",
    "578aa37c44398717902ac8e4de6d711aa0cbca1508df8d69f163ffafcaacf5e1",
    "fd5c2894ddf48efddf30fa4e4543500b055a9afcf439e2fc277984f491ed3a0c",
    "8e2b567284897f5651be110ffdd02f43ce1392923e02b52867a167ba0dfe7553",
    "a49bc5353e9fce35c891c37beb9459e1bc247af759dcb9684640c16705a6e54f",
    "2721d072d3c8c4e2d12a5fb8672fc9847c2c56b46a699db7e7416eb1243740aa",
    "d5575bc60c9d425ea9a7bc4912f713de01c0b515ca4236e85a46f71c1a88f324",
    "caaff3780b2b805cf7a91113f97632b510655b697c5b1c68554e5113cd1f1b06",
    "5924668ca89a2c43eefd6bfbbfe6dd634b0222e73189c0794f4a151f98b82d7f",
    "39b67d881470a65c1833b882683de24ba2c59e6839b2bec0976daa0505f32da6",
    "e21d78d52ffb123a8ac5bde6e3f5909eb1a933a3fbf92196afb734e6fbd3d065",
    "c9479c6c3b7c7c4dc33404c4b99223f89da6597c0df4ff6720e0e0901b89180d",
    "cb6aba4b216b56910723559410512ba3334f27fec6958b1e817c4a3cb27ba83b",
    "c039ae59509e291aaf5b17aa067fd60168788ffdb63d4fab363f22996c228f41",
    "c5e1023c21ad164d76c36575648f92edb490eaefba83045eeebacb52ea019d86",
    "a3a13e5940400658a711094f91fe4bfeaf14f6aa871a2ee6ec99d556ffbc724e",
    "92d3a2e19fac6e3ba60cc3807d9764f15c2e3fb3f45e3000255166299ad30d3e",
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
FROZEN_TRACE_HASH = "0227619e63795fbafddc6ae9a697c1074a90d47fb7108489472acd4974e5becd"


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
