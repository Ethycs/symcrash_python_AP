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
# Regenerated 2026-10-30: signed-RNE lifting; dimensions/headroom unchanged.
# Prior commit reproduced; Fraction oracle verifies new audio.
# Audit: regen_reports/lifting_arithmetic_20261030.txt
FROZEN_BLOCK_HASHES: List[str] = [
    "27a86a8715f597c78cd0ccf7d1466e7b9decd328f58809c08af4cf0167467fa2",
    "10d2593cddf917d784f568dc5093adb760f246a3628746f52552f0bf3e22ea94",
    "c4327372443fa6796cbb71f5f5691ea07066d90999b70e81471356a4c0966958",
    "52bf1c01e11e7d335af65cdf6607fd6a6ef60a2824e708247bae4fedf703cc39",
    "288f7db5810420d19ed057ec222b221d43e259aeb822d84bbec4ef10c7ec2345",
    "3f735c16e2bcedff6e4ab17a949024578237198deebe008b8692d672d600fa3b",
    "683efc62ce87552ef47ec93cf43e1e4a695f5b8d46e067991c3a0e320cbbcf6d",
    "b48697ebe37e1be553bf665be0dc4ddc8f8dda99cf2650236515fc5ad20cdf2b",
    "3e127ff966d950cff8bf680252c20d31295fa32b0bcaa09b185178280e6462e1",
    "960f264444a3250b10b4235ef8f867d2fcbe33ead7769b75d6e0da5d30acb1f6",
    "c9ed7f4999314bef58a038697b732a33f3aaad3ade42fb5ded4ef225ff5cf4c3",
    "d04ef70bea277379a71e49d322d4d58c89e4661fcc19e0a94266feca307ab847",
    "00caae551cd3a75c6cbf473e213404e1e41d2792fd6c8de7a3039e3f241299a7",
    "2f8d14e1e46d41d279a949134a9cd5c0ae049fc536a61c05d3b0eca57b9cddba",
    "c443118a810b7dc6ca51461e42c3bd12eb6cfa9d08c6824136f2454beaea6437",
    "44cf557907c80f8e979d14c309589ad52f091b23fc590d0f5d6c43ef7d4c9734",
    "ad2531dcce38d5e5c0d9869d088fc5eb419afc966584bdd651ba0a8ba1520b8a",
    "88a5416ade961fe74ec9005e1fc88f9d7df044e7e6483215fefa1b68a6bdf998",
    "34e7aedc2170fe645e5e689164c2e4fa4c26230464f459348cf5951bbd4615e9",
    "e19549ecfb9b0d4668d35a1355c06736ade5f4ed5b0edf6c058184c234123943",
    "fa34185b32066ffe9902c97c87c72e7522fd46e6bb22c0762a35cd8c3ab459a9",
    "e6ea89f6275bacca70cd5dd29a4e780ad0d3a4f6c755ccd74a20df5912cba68d",
    "1a6abc477c6148f3b5d2326178e1130456693ac7dbd15a4055670ce379e1bd68",
    "a77b3e363e2a7bfb6f09164cd54556aaf16f69aa88956d77bc87f27d4f35e5d9",
    "290e9b04b6f7b6f2dd85883309a8d3c0940a3b146655b7ad66e52fda5a55a24a",
    "abd6227be2dbf7470f8420c1d2943cb580037fd25de93a7d5e64ff1877f308a9",
    "3c9d3327889ace651b553b4a3d26c6df0415b9000cc26893cf57979e8f71bd7f",
    "4ba21639d2777bb8e4534371278b44b70e2e7fdea90e8b701508a877121b7315",
    "c3ed28680b4cf088a696d8adc06cb4858311ebccfdb31a8ecf2c2cd73ba87951",
    "141fe0669b4b8dacbd1d575aa3482677fdcb028b9fde1a7e84139b1e385a02ce",
    "4a273b707ef90cd1969bff390abb1969eb4aac66f435f39d2e9bd8b0cd822973",
    "b5c8b610cf36e52cd5dcc3ba691e268690b28fc58221b47d4bca0e1c62c5d3ec",
    "8d4f360b890aacac87f55a5b0f9e4ac3849b7dc1facdd1057a3a8663747c0c25",
    "896b5226b9b64449ff616ec1d97eb91341d30078ceeeb95a67b45877c61eece5",
    "2accb3472a05100c61c2a47214f44d489739ac89f9146114f9c8190ba2e7d447",
    "a8925dca296329259da9d2f0f7780dc515d267c1c511e8a297db3035dd7056bd",
    "54b2bfd4a3c81aea87c233f6befe464ce421fd3dfa7954024f1f67eec5f5f5fb",
    "084a3896e9d0b3a4fb8c4820da84d55141676ed71fa91dcd6dd749d16e6ccfcc",
    "4220fc2b7cc97839c29d68a0b3cc0c64f867fdeaf1f8a3c6e3637a314f4729aa",
    "635f19a9d3abdddc13be01cd462a61f7f30abda482ac8e065eb964028dc904eb",
    "5f45b68a38f43a20f384eccddedd4eabc91da43680c08bec988b9b50bb5980f8",
    "8cd9b1e1c587269d2e1f22ddedaa58a967ace49f76d9349ad08fa4aaad9e70ae",
    "4a1536e599d3e472e04c2b4b7b6d826e437ce43bae9e5d7d30a16fe861ae9dac",
    "8b29368799c604899aa59301d6da623835cc4e9bf7ee6dfc93acefb06544e5be",
    "283d0bf5b3169d3235495301ce759479f7a1e76b7130b9e9ec407974a7745400",
    "af5bf6bd76bcef24e01612347e24738e73252a39204a8f72f55673e055a7b18d",
    "595069f103c2ea805d54fa0d05818f12032f89184bbb47807c0506c0990ae0bf",
    "5c275976a1c146f83a0f8d1d30c9a49c498a7bb959b9541f82ee84777dfb237e",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "eba64067358f8fe4d24f9702ba1180347562ea25b3b478e7ac62e67655825863"


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
