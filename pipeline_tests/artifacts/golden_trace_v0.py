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
FROZEN_BLOCK_HASHES: List[str] = [
    "6dc6b92841a93565903f33d7385b241ef4bca2510bb1e6c4b4065d34ff28104f",
    "b8f52dc1ae0d4a66d870ec08fb41ec7c9dffdd60fcc9f6075d726916c783e7ff",
    "51e085715a4cb3f08f4921c31710f746e049fd4b9041c2d6c5177ae621f86d40",
    "cf9ecbeab1595905a870ce570c99d82f216ceecf5b71040736a4c86703979091",
    "c2021469d1c0d793eb6ba514fbaca3b1f2c792c827ac6b43dbd843b47be22df1",
    "3e8aebac1eccc6c53f69cf2cd068a913c852598ba3a154208a4bae859b191346",
    "7d6621d76d56965456c891a04a261a47872bb83702533cadaa448c00ec381848",
    "ce508e050c3f4de2c9dd08b804850cfdb71f541452e993c2028d232cf0bfb4cb",
    "ad75d2f989436711522f8bffb806e2739ff2e22c6d73af5e46df46bbe8a2aef9",
    "b58c8ca53567db1cdcdbe2f357df5efcba166048ee81d35e8f9e40f2523e3f01",
    "709c8fc4142fe7cb9ab28bc157b896193ccc28a9ce556a269b36a05165aa8d3f",
    "bc572d6b3f9e6fd0d4de33e9fa7f253468b7581003fdf6ef9d08291a19b6ea1c",
    "0c4cfcf5f2404eb4181ff100007cdd73f597462806e44cd65cff66d0c95e90e9",
    "7eb88ec389a4642f36202879e62afd5df9e2a4d3da82d5fcdaf536da75362791",
    "11ebf57b345c419da1fb24e915e5596e2e944fa5198f0095e7e7df0ee50b310c",
    "328c34914e489c92cd1bc89ab10b8ab44132e9e49288a5ce4a29b4dd750a7ae6",
    "126cea59fc6a52aaf5b90ec9c97f8ffbd28a8df1e6958dd315ec21a5c969765f",
    "149572ab3e550a8b3d516c69cb8b4442e1f61198d7b1e8bc60a3848ed5ad32e5",
    "48cea582e25bc7518d11915529fa0039ae47d8253e1ab353acef5a681a8ac0d6",
    "9ab5920f36cc2ba42965bed08ba01e98fe93e166c969a1b8822f388cd942010b",
    "e0ef6c6bad0ef8d37c2d5deffba9db9829afe9c6d28b47196741bd201c6e46ca",
    "0b24e388db68aac94b1f222e0db49fd097476f3b81ae8446d4025c8a10d8b8d4",
    "08dd82abb339cec0dfb7476509c1737b492dda78bb842b1faabeab3c933a0890",
    "c67327983bbba88838e7f762cb67afb6595fa0e30d64d40a088123ba2edaed85",
    "a2d15201d9b83ca62805e200ba8177446a13e2fbc9dcfee68fc5dd28cb38aa70",
    "012ee39125f9ff0dd9e59188f94fade422621cf132722d382f04fe1140909d2f",
    "b2a4686cadb4de996b5956025cdc0a6f4d87946f9af8118beaaa8846d6d03875",
    "4d0e0c3f874616a7fd6857f4c081a91de428edcd1a2664769647d30b15dd42ae",
    "4248f7d8774068ed193ffc10ca778c2a7a7d88784b3be1907a8342fcea785af0",
    "5bf16bba16557fc47538846c7a314cbb644f3b591acf640104dab30473ada00f",
    "5849a0149d75297628b3593f2cee49c864bb5d6d3306af3bfaa425e57edcbe00",
    "a1399d8b2c8516bebf6cb51860a0e74f19846721b87ef4303c0748daf5c389ea",
    "e207ee059da6fd625a36cfc7b9a476a5a4577396b67e9f0b83272a9c322ff22e",
    "d569c66dfcc5b9fd492bf978519e39eb3c408887c511d98ca59f986c6679cd00",
    "588f1cc0f6ff3da6f3c6eb171cf305bd2893a35e9e05b51f853951df3bcb1dd1",
    "2c08e0ab829af97593e0c7946a9d99b730bcbeb7153e246f574132f476d55f02",
    "d812006339f11ea9c6d2ad0b848f3bae9ca1e9aa3f41d71986d1deadef3089b2",
    "a0590950a20861378d4ccb3de542fe0c8f4025f8256c80fa0bfb31656ffc9af4",
    "20197fc5d6aaf9b5897a2a8e0f73f3e2f086d08cabd3ee4584a1a89ee57af004",
    "e2d93ebe5bd0d8602b7b1686f7dc9a45686bb0abb6e49c32d66488e4f507456a",
    "16a5f10a6ff8f19f92191180bf208ea67a0481c568bc20dcaff6f12782bbfc05",
    "1076b0c4304b925fd0a57a7dbca171127022030b71718a4a62f417d4ff72f41d",
    "d6ceb419c6159935bc5c84d12108998c39205c094ea44565bdaa4417fafd4c41",
    "76546f7dfec79f6841712aa58441c6807a8c98462fb393dfb2646bc17d57151d",
    "2e2898d7156ac17fd1d958ba812e50ed472a4cbfd8344e927e3a505cc70fdc6f",
    "02825e228218ed6663178dc970fb1088a4b43ff2751778fa58fa7337201b4463",
    "65856866a8f2e7b55f14212dd5093de59627cc77fabb278c27c43f2ba4a7b560",
    "24bc112ad343db0bce27e7bb0a8a86ab4dd34cf114393c4451bdebd1779b9556",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "1f6fbc168511c0e1c17b7099ede032ac6461fe227c98c281f577b1eda069a917"


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
