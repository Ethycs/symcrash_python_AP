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
# FROZEN HASHES (Generated 2025-12-26)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "e2e8cf00981098a7eaff7151c1df8c7b8be70a71911ccf7a6a78c786978cc38f",
    "59d79d892f29ce2e76592be2c0b9e40345508ad6a569df7e612cf14983d0ee83",
    "40e961f4234164706ecfb49317d1f0d4fafbc43f5cbd0745c804420d47417e2f",
    "a5a9e6d29e92b1d6ac294152f09b82830a3ee92317c3379421ef60b3425a6faa",
    "ec2ce51a245ca4aac5ef5848c861baf6e910f5b45c420637cba3be7554675b8a",
    "899b56ed63710239c4365b70b8144cd11dd2bac10ae1504415843c8382ec16d1",
    "e86712fa76332649fd3bdc350b7524613c55ea5db2237fad25f0fdd23097f370",
    "01911c07407e4f78bebcd151aa6245bebb957b1436fcf0b540cce20c0dbbe8b9",
    "bece7d7bd09205cd57f05f0e7b300fd7afc980d1d15dfe381b7faee9fd53fd26",
    "c29cd6ff6f152a88dceda70094cc63814927bb2c0cd3ee0b86812d459c6f48cb",
    "9ed95f8349a4b476da343985de929bb2526545925becdd73d734f395b8c298c9",
    "90d961ad2c2cd104d289746caf35778e32c21dd82f91774e10cb9450f8344365",
    "1a7b47a051a33bdae278639e93853a7318686d47591d7cd972c4a9bbd2eb9cc9",
    "7e0e4a95d3a1e8cb996f2fa6990ba3b38bdc67cf8b130cd7ab91fe8ea5b9d878",
    "983ff5fb287088bf355e0133422089932d886e3b95c30083a83eeb60e02e7d29",
    "211d60160242a6fbee371ae28913f6c681708035d6bea3e4040b5476e89734ef",
    "eecb2044dd43ee628cc2d086718fc22ca373f88600c68737d0ec8776da13b1a9",
    "5a8854ed8699c8a124292fce8c619f9f102ddc9b61f6917734c9fefe19cb27b2",
    "eb6a5091b86d5769b929ff8daf11022ee83eb25843e6f616bc4f2536a45822d3",
    "27fb6208577286bf62d7a2f1127bce1059edd8ead18650bae0b33913ecf813c5",
    "602b7bcdeac3f343d868ab6fb6903d0a190856299f2c3e6e2b13df12abf23a8a",
    "ce96f10ce4c068a3268a622207973cacf09fa7a5f2b4398bdbf9cce11f1a8942",
    "a690666ce0868dddbf109d1657bc037513f78b626c3a590b97c5cbe1ff1d1345",
    "9554bb5e4ee59b3da046732761096713340e89cd9d434e05d963d9135a7c1cd8",
    "599f4cf8a2a3c5ec6937396a2ae4cb638ac4b44a758aab3d5cee4fea0b6e76cc",
    "fff9941321ee84e2eeb67b7f1f23327f212cb70731a25fa21d87d45ffa02e7f8",
    "03569a8806b05636207cf31eff3c31d53b8d19a782fadd9ed7a08e5222c07b90",
    "279fcd4534a961c95b47b4802c6d7bd6fa6014e8f5a4eb2c09e322768295ccd1",
    "5af1f5b8afb30d576721101dd38cb3a5c07a2b7b004b5abc0c2d37eab2c21259",
    "f512eb6d296f427649989543fc3deffab57ef3f478eff9a2daa2fcc827510142",
    "04b4ed641ec51a72cfa3a19f5c695dd448c297b1f22e06ec2943c79483df82ae",
    "3ee7dc4e662ccb360beaa67e8b3cd98c36c5127a3cc9e7dcd603a94643c6a183",
    "84ca90b2961317b53f873dffbd65dd7fb02e5c22e47e7abc10dee137784e7a49",
    "95dae6ab6ce1981219945878e90770603c84ceb8c50fc048055d7c0daaacbdc4",
    "cf3344c9cafdfd94997cdc5c135891367c539b5651a78c5bd788dcae2a63ccc5",
    "f3192155333c4e52f3da3ae31cb4306081bfd95e197468c9f6abdad7f0f32409",
    "b581bc59bc4dfe5ead2705b261d78d6db534c84fd3c2f7f39bbcca71bfa80425",
    "5a382bbe239b44e5bbaec37f75c312a97b1396362c59d2550cd25e7708a390f4",
    "f640734402c95e6bc884f29062f1c79cc356fa00a091985343103be62d508d82",
    "3e8801373212f551ec47008edf9d4388e1438eef6e989514af6213d4ac6baa29",
    "c71e9fe25e4c93f9ab587a2cfcdf88871c6a98a53e3d8f57e707390b6b40bcb5",
    "14e11f42368c3d9e1606781fa6ab84e5835c549bb72ab22b4d083cbb0a3aff27",
    "46c9deac31f7c77f69d6df0b79e6ce8eef6998d0f52c999b8e4b90572c92f0f3",
    "9924028282198002a578d3c9fc335f3e70764c8ad5e015a321e8318d0021c487",
    "cbdea5193f5a3b44f50f536bf1132f5392e26af9ebe58f8a5d8d7d1ca6b01263",
    "a52b53784999eb001733325c2881b88cf8ed8e43b2088d2f9ec6f6cf8a544386",
    "a6c03667d07e8d1b7c90938c79ea4853e39f33ecc5e22270204295874cac5277",
    "8cb9004cd7bc382f8b9fa9ee240c31609c50fdc5e2c40dbb5a03b6f177a36bda",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "999e7b6be0bc68ecc7d2e416e28b36ca7171d9f121d75b110982f2ed07aacab0"


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

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=440.0)

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
