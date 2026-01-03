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
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
# Updated 2026-01-02: Orthogonal residual projection for wet/dry blend (linear residual instead of equal-power)
# Updated 2026-01-02: Phase-coherent wet path (mask smoothing + C⁰ seam stitching, no windowing)
# Updated 2026-01-03: Disable DC removal for VM path (CCS is phase-continuous with zero DC)
FROZEN_BLOCK_HASHES: List[str] = [
    "68edca572746700e74752fa18c80498802942e9c8aebd6420ef9d20872483b06",
    "ac6d2b15c562bb61685995c7147835e5efd2dabf99b80475c06bc5d266cc32c3",
    "3f0152ab94961f081b87065c2b8e08867e02227fa9bc147a5ada12fd83eb5ac8",
    "da42d7ccd6b594054e5013c9ded57dd799d90d57fda02a919ef2da64d5717a3e",
    "722aeca3e4fe316ea75f798006118f40b1673f24df26a3dce197596e8e618f25",
    "ec180bf81628bbd6d2505dc4a2bff01136d0394801f7eabf5585a1d1d4215129",
    "10f0c33673afa1a5f647d5dc1a4e7ca3bd8af53dd747e3a53aa7fc73ccfa49f7",
    "f60917bfef179c0733caaf2cdf06db93bcb303eb3c163b59558bcb0b940cf57d",
    "2f519bf6439d7bdf99998d064adbf48bf0d97dae61ee66f1aa27e06df3d692ce",
    "cc8462fe928a1bfc0faa7116a0cdd8b70392373ec533e79d38eee51fcb035450",
    "677677d9694de82da4a742f9bb1aaabf398ebf44ff01470b34bd7de24dd4d287",
    "3d9e4f26cf522e11327e3d4cef0b2dbf9ecd5ffc2561e2305bfc38815faf588e",
    "02f363195f5b63c40d3ad358aa4fe36394bed4b29a0c68f2dae772b347bbc45b",
    "51a420eb3176a53ceeae4b717ad8073e228fa24e915c02c58f4850cf2bbd1244",
    "577bf4d6f9f5faacc60baa58346591b7c40d5a5e54c332c037ab432688191ab4",
    "b19257ddd48fb05f62b1e6a7878ae905eee11f005b800e1ded3000a5770cc47e",
    "7b2e2b2eb0a9f6e062971a954b82e209ab43918a653d9ba86705c1c8b7e075cb",
    "bf8968821893e32c5520c20a4b49993e6b09721395c6855f27658618f04b15d6",
    "160c175c20bdcb2895abfe8423478e961fa606b8b84cc8a7d0afc88fd32e4036",
    "efc090fe877374b183519f88e8219f9f8bfb384363d00c7de37e3ed4d3b756ac",
    "cfe48ab486a8a64cd144f99dcf501bedb734711c652c14d4d357a9c82e30bfe7",
    "b56324b68c6a369448e7ed7d3464be823d8cc4ceac771d505969415c32560568",
    "ae70736dd7a57e1577f1d10619e90724733630421d2b674deb577b345b96d96e",
    "b5d5e12c4c8b653510ce05de22d0702a4d4a67d3b8a5c41c6ca74f6d969435c9",
    "964b3f19f5e0ad26c6347ff55e71b86160554becfcc1870d3c1d2086cb844e9d",
    "19843e15156d29201642c6531767b33d5c23ed5603f4cd6ce82b2cb1f1d8e2e7",
    "79eb3bfaa6fc10922f894dea1b4cebb94d2669fb3060126147a9762452858bcf",
    "97ef462b123fea8e340b1d5db0da4a3276059f3c1c263bdbe4542a182cb550cd",
    "9ffa8991e6bd243bcc49ac3628179d2f64ef6a600ce5d690f21ae8c51ea0a601",
    "fecb869ffb31012afd84595407ffe87785e0b4ba62fb889b6ddd0379fedeafde",
    "f05b7f1a959395e78dc1f13f33502309a9920f4ecd42b18b848e2ff4c3d00e8e",
    "4efb37a37eb54fce6bbf2c8e6c2701724170b911b5d18e87b431bd170a96ddae",
    "a422536b69cba036a1693b9834b9552d4c4180bf09e1a3c5afba5eb98f8956d3",
    "d86f6dc2a42aab15a8b2bf163400d1e104a0bb3698a62ec46dd44ef920ba9957",
    "69207df956d3ec21a6c797ae79c067fed1a1ea51a25fb87a14b61ccd617d17e5",
    "19306d1de263d15328ca50518bcde469bc8f850f0cff8f57fcffe72b9d5ce9c9",
    "f9af896f068d94f85c941de726a380e722cfcb7de8cfedf5753e923a874f13f8",
    "94b29c7f1730f65c3b9fc63211a841938ba08ae22ccda6a309b743477a3b901b",
    "220373fdcc6e29c8881fc0ee9cb1acb444cef669542d63443b080b07106d9d36",
    "71226c144c7649e5f23f023fbf07e41f13d8f8b9e78cfae1a52e0cb9d19e5452",
    "eb72b78b1b2f0a46ecd5f500aa476aa420c51d38b4d196b0f0fb9b58c4f07e7f",
    "68a6150faf30b1df4926c022867df672646f35ea8edadeb8f846b8082ec0e438",
    "aed5cc18c270165b33830cc9711621adedb44d74639e97cf9e257ac3162115bb",
    "a716bd5688dad0859d276e3937a769d59332b4fb5495b5bfc5fd22e8b0d8f6eb",
    "a415acd18146d87d934cda40dbaf3812a0e3b4e5810bf11266103070ad550eb7",
    "7dd08c77d690aa3cfa66625143538cd5523e95320a74346a24281398e808cd73",
    "f8eb325acfd10945dc1e5dd0e23e2f465e5d80153927ab376d66c299d3131ba5",
    "449956c400a52a1df1ca9a395c07093d3212aea88dc134438a7823059a4f54ca",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "3ec436a05298dbb99d56895ec276d4894b61f556b170de3f11af2448fb110012"


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
