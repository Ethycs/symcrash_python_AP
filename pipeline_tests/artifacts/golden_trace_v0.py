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
# FROZEN HASHES (Generated 2026-01-01 - IDWPT now uses effect_mode)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "e2e8cf00981098a7eaff7151c1df8c7b8be70a71911ccf7a6a78c786978cc38f",
    "59d79d892f29ce2e76592be2c0b9e40345508ad6a569df7e612cf14983d0ee83",
    "40e961f4234164706ecfb49317d1f0d4fafbc43f5cbd0745c804420d47417e2f",
    "a5a9e6d29e92b1d6ac294152f09b82830a3ee92317c3379421ef60b3425a6faa",
    "c74c5114cff55790eaf9fc4a08ff452b5f793d2831e4f7305fbfd31b23efda36",
    "c93232ffeac66b56b7d7c24b2f68fecea941a1da68bd6e9e3a833858e6062be5",
    "0f10a70daefeeea33ba9a9553b924b39b3893f1cd9d4e7dfe0113bac80408e91",
    "9f86e8cbf66fe60d98683ec2133af8d9bca9e14d7ca7075b5e5f199671ee7c47",
    "8f450b3521fd6c8177a4889f90f87d86e9474e629f59e826b84a56af03b31326",
    "13c50259dd668c60792afe45d6203eb0a804552be5a51fde705b3af03a07f5cc",
    "1a18aedce08a4bc2d1558c9ccd1d3fa627825913db699d8bf500f794dbd1a03a",
    "f8e910a7497fb82b719f42c219ed7ff692e409748b8f206a9dc0b50713e00282",
    "9a565dbaec466c4f0b8b55db1aa5d8014859d7ea8b30d9a94215d19ca8a0de0f",
    "2d0dd71be7409f07a0b34e44b3a9d3df2b829946440a4004bec22305a4ca60ab",
    "f26baf02de1892ad8b9196712d33eb3168458a43f63da5be0c9a283005efd99a",
    "4c59b85f26867765a6a3a80fdab179ab2a1d7f6667d2b7fdfb1440c2629fc591",
    "24d411a4511afc2958a1889f9e87760afafba5aecb730ab364917fd75633dc9e",
    "91b2ee62707d77fbd4b70d571a72371337b6e02a968254b12fb27b560d2e4af1",
    "297980c4d4c731bed1eac51ae18457d201415283a1a6ff46b7dfca1b0f5e95e8",
    "1f4e317c53937542d4260da90123f2043e514986f438d5a468791c05c89a54c9",
    "5329d8f5b211f44803c4faa0997ef4a3cf86b30f21def17cde865e5b20738ce6",
    "29123558e83a9bf3bf7c2513d9f0993b1f899c6c8a5200b193fceecb73a11952",
    "1891d0230c978e9432dc0bb0898bdfa734bbf865257e0772795b95a5b16224fb",
    "7787e66a9cabfae7393479dd6ab32b1bfde3696ca22945a045d1acf8af7a43f0",
    "18da2b8236f132a3e6c441a407b748f69c075a09c939e90eb8953fdb46182112",
    "ebcf2d3342c33d6cf4790807ed2939948c70692386625f3899b02993170d8ee0",
    "7b865ec0b2917f18177aef17274394abb1576958beaef49571383dc49adb9ae1",
    "5637548b37b2ec5a0327cd78c13f2dd5b3fb39bf95c3e7667fc6e3f6d92678e7",
    "62c0f382c6843046e4a751b7fb7134b3726e0b9b81c40ff7a3317dd43140ef93",
    "0d8a565584fec8c0327d37797427b7378dabd65ab248e5165d95a77ec57a688c",
    "a29112e624f8ed58caeafc4d4afb05907d1a74488d1b1bdd5cabd75390061234",
    "22abe57a2dc1b412f95b9819324650f33c89c346b1f1cd8f96895625e6298f4c",
    "49324f4206e1f136086291379628ce70da1f63efbbcb59a76a1a6b40ef0ffff5",
    "20780e681fc9914fb2623a47909d2450c5a83c49afcb2cc76ae8478bb4208de4",
    "565be549f89047c66ab4906ea8b5c3e213adf74ea41762157b4a4218777a8329",
    "4cf2a544d7a1f508865f780953edb8c8b54a3a591c2bbe1cd9f82cdb8f88353c",
    "f967e66e1a01f8b4ef1fcdc619d9b3aed8692470dd71605d93bce45eac217e6b",
    "3b89ffc3694e0b6064ea6d6a3a5a1209c58e109dcffab9e8bcd063ac98f86505",
    "8d089ea55c68c4a066e6ba2b69bd15574440e54ddc52351b245e71e40d7a2412",
    "19904b12f5b2cb1778787cf370750749a837be6974af0fa7425dd5baee6f9f6a",
    "cb0a8081870a14d1db7d3e5cbcef62bdfa809db44c66b60af3ac592acbfa2114",
    "b149d4a9b15522434aca4e32386043e64e5016e07c2cbc2beaa2a0ece81f7f0f",
    "ee13bc8a4e4b46c4d317368391f23d66bb49c1b00b28bbcd0b0bcf35cdaf89a4",
    "2354b504e327cb359680e7c677013189ac4d22568f7924965a5a19ad01d8bdee",
    "fa3a291a23fd3f62e6e8ee584e73590a2287b36dab6beb17aacb8b782191ee9f",
    "8862bca8d1ceb63d4a9037f13ac217050fedf97fcf875949dfc7550f2a530c66",
    "1b103d8d4b2c360be3a034ae24919fde7eb673b932ae6f82f222c0550781b738",
    "11a41815b9ee37efbfd7baff5681ac9a0268c138b14c702d997388ce10325e5f",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "a7177b30ac30a53d43bf11e462e0db8acc99987d43237680c301ba4ca9dcaad6"


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
