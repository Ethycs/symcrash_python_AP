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
# Updated 2026-01-05: Skip SheafProject in CCS mode (was corrupting polyphase coefficients)
FROZEN_BLOCK_HASHES: List[str] = [
    "8eac3e070aeeea40d12dc8b72a1a071bdd3bc432c327d616b15dfe7a6a81793c",
    "307de78e0edf6fb0c1c3ac7f885e33983143b7fd19be264a58c4342d4b362f0b",
    "b9fb43cc5cbc4c3326116f50294c077cdf88da3f5f6aa130a7959f3acd404b63",
    "9c662812ccb973e89ea6882fbd4e0172381348000562c13e5a9e367706f9a61f",
    "136daa0d23e8cd43cc86a36e48d81e9c544aea1380d9ea833f0d8efac599b1ad",
    "0c444986919c01573626b5d1bd9ba232042decd07f76c67218deb5361b07c3ef",
    "03244b4e42fb6f1ed7ed01c026577a20269bc0802411d28af848867281e016ab",
    "cb4b0563fe47589b67de3109fb5fe215ee685d454e15b5df3c55ddc2a99aeb9d",
    "be5bbc628c926c14ed1d27e4ec3e12a32d861eeb4640e1d13f20885c958f815d",
    "f59fdc87e89f09fde04df69fe44cbffc2fbeaa2812e7ae549d61bc2c01012c13",
    "4860899c2d3430733c20e23428d469c043f0eefb3d4dd062a0eb91ab26048fe0",
    "b7522c366becf4ee7229a260b86b6c7ca8bafa0027c414d12fab09ac8a518d02",
    "3ba97d2600b65e479166782267f0a88728d2a056dba05f8be893a222d9fdbace",
    "16dd84d1bd18fadd706d9ac8e47706292379b0f97c853aa1d5020ac07819a913",
    "7ae13ef0c3eef3539d25b6018e02fba36e8f8f99af1e74c80bfaa330d3128728",
    "61f7315160e3b8de9012b79fce3d00290d5208adba0460db4ef7fe1451a40a05",
    "31d675f8bdb4b5bea6be4963cbca3af52b81b21afc15659ac005350e0ca76e8e",
    "41d43c552bde11bfea471255dedfa8ca7c4f19bddc30375a4548155557adfae6",
    "356a735a43e5155d430c4f1f6a4b6221b376a993a160b8e8e4a86cca4a3ff174",
    "0fb745e34b0462c3e4c0777eef237a1aad3ed8085699f5edf305b65db4aef919",
    "4778622ba04cf9c4cf7d90abc83912e97a67b9016a62a0eb81ab1a3a0a2c241c",
    "95057f35235913dbfa6105abe3d21ee4c271e94bd6fb154edc28e6e8215ae30b",
    "b862f7d49d0e930ecce6a1dd048842544b9855d002f881eb8c523f2dd7a54c95",
    "2428f4514656b17c8af57e5b73ada6ccdc21a43d8f05912c47d67739dd430dc5",
    "7d380390fd4c4c14d66f07a885e3f0a954df4c9b1de309038344ef0d30a120cb",
    "48739d87737988bad0b9707a1c1dcbc5bec0e685b62a5b94c770ea7d190258f0",
    "68f9a1ab99f8ade436a844a2bc8716dcb84516485775f226815c7f033098d2fe",
    "8c9cf4c5ce3057f35651732d4aaabbe3996c9380739359b617a41d1c09c8c11a",
    "0244fdd6cb53a1da62cfa920898302d3fc532f0e3fd8ed2c5e331369fe179426",
    "53c70ce727f3f6b7ad9b3207a34ce7dc2cbc7d55ac15388dbcfa4139b4831682",
    "d6c356eb03ae36eb017539147468dd5d93e6b1587d520c1926cb5ab41c946daa",
    "37bc0fcbb294bb3f0777159ac8d7039c246a65db80e9029595e69837d5999823",
    "934f91f5a85e2442913130e3f6e6bf131d5afe20fabd50238c1f916cc940ec08",
    "117972e3640a22b1f3c39b3d884718622673a5322d5885a8f09266e0babb4d11",
    "c088e59ea622b4dc3a53b9f94a88c5b1c735ba69c2f2cedc616deb04946fe48c",
    "7324adac6993dbbe24e4b56ce749ac7c9a3d1d948b0df74fdaf3fd889c25505b",
    "cba944994993ae46bf3c3510b9f8501e047958d6acdf2a02bb2631d8a588ea03",
    "30fd208e474d690cff0b2d26aebc683f0f39745e8a2aed93c497b7ec4bf1c470",
    "e61cb23db737155364c43000c9ec7a55a3994b2e9d5a316cbeb94b767c3b96f7",
    "7078b2315fe3b2e83f0f6a7ff7adbfbd490254d865e01516cb50448af948f399",
    "6ce928837c1856cf4cd229ea731b650221af65e68701cad3874ca99bbe9f494b",
    "ffc41a9e0ba25c2be2bb9757214dc7dc656bc8f5f248c52dfb82271a1329b843",
    "f52bc28f80284a37f3b2c1bf56ea830926069b897cbbacbb931395b38c2d82e7",
    "d8b782451e5ac6d1dbb38c34074672ff44d7ac4978d561cb8602bd561fcb8ed2",
    "7ecbb44e05315657561b9f2c78c068cbb2fed08296af3b1d75be30a0240d40c3",
    "701b2ff08e38278302f398010bb15494c76c8f51f75a0144ffab7e9ef6e3c6e3",
    "78d1339613e46eed2d243699c4612e8ce866bb2a735196224a5f344ad631d00e",
    "b6d5b13c93fdd4ae57129b0150cdd5c6dc90b1033c1b9d7b0e6928969e2e3124",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "be2aa6755fc127c363a94811db87fc344c9f9e5108e235dc65794cdc4e074412"


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
