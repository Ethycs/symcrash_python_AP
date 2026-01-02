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
# Updated 2026-01-02: Fixed-point pitch LUT (WSE-STD-PITCH v0) - eliminates floating-point noise
FROZEN_BLOCK_HASHES: List[str] = [
    "832361b55e2190865960a16dc69c7ff94901db504236d38cb7e57ffec0337504",
    "5dded3914b73f1679e59b3bc1e902c0557316ae83f070d1b56fe6a2c86195596",
    "ad3dd15b6fd0e634626c17989dfe970331e272552da03cfde41a1ee146612df6",
    "a3ddec556600c4352c4a01b76b2f6d0457a953c6aeaa2b73953db2798a3b9a33",
    "fd01947d6133ae643fd013e52a8d13a0303ecb18bef5ddb8b14fe8150d9c5756",
    "4a10d1e9e46575007b4da4875e8ac0d556dd770a5a459091ca191621482e8292",
    "6df808b4c04983379c8946332c008717e7902b3dd97087296a2eeab6028c3e52",
    "a8a71c9b7a69857ed4cfaacbb89a0d565e56eb5137533ac9cbdad16ecdd2180e",
    "346c3d08b32898d527078ab60a0efc0d450962ebba3a9d649040ae1259b10d36",
    "ffc539c16971e65aa4870b730b5e6cdbfa7f3c1aa61ae06be6e0c4ffa1ddfc1a",
    "a61e7e5a3512ab1777f21e8c4b6acb4288e22629dd5bb520017f22c7e613ae7d",
    "b040d1234c0d493ea4effe4094de9dcab3066c49ad7f4be3e7776b00a197d835",
    "71d37102cd1b14c7fad2266f9f0c63801e1b05d30aa53cfcd4040912ac229bd1",
    "7eb32f666e5bc4896dd491e69a0a1b8833f8c04e218a3a4ff69f12c3040d72a7",
    "3633ec682b2f3c650835070155d7e5a693ec68464096281ffb1f5a312505a414",
    "71eb24a771aca00829de7e7d03b5519258b64057e01f2b60d76c689fc503f77b",
    "65c0505a3582f9bea64e9c50da4a5fe17806b93f7cda6c10944bbd6c3070b957",
    "fcb04bb2bbeb019f3fc49c1bbfc5958ec5ac209ea2b512be21d7c6579bbe93d1",
    "e001640ea989ecbc71fde235573ebc165ef0278ffc27bd8c120fed540741e07b",
    "c0c49ca66591ea755b5e90f6cdf80d31c7ad968ccf54c4ec19ba0903832eef79",
    "cbba6e3ef0158c67fe87152267dac01430df2ca1ee7da8236839cbbf9175ac06",
    "afecdcc8a142dfe1637905cd5a5ddc5b29d09ea92980e9f0fb710dfea60bc48a",
    "816a5762168503d2f4db80494212ed6710fad372d145adf3015a3c440543026c",
    "9727ef65f83f060284b788fa0aa420e21383bb51a05582003a73707d962da4d1",
    "c5e331c072af5ab208d72b7a7550080dc2e47bdafe2e69ef291cd0bd590cee10",
    "24e43a6e0960252fcfd6236a0604665be67951cb2df66d4b6fd86320c71f5dd2",
    "e2980b0808700b30535cf1ff8ecb329a3d226feda1e6df6924ca12b5e3e30e23",
    "54417dbab67e6c1c5b6ed7b3280cf1c7138882974bbfacdf7c90c4e001314e27",
    "913cf649ca258d7c632480ef256e52ba6368a3b8d2fbb194d42f7c4783a7f6c0",
    "214184d615133a728db403c1beb6c02a37d9bfbe5fb3be2aa67b68dab50ce498",
    "dc2e28919b5eba981a9a75a1c60bf3c5dcb65c0f70c8f6fb97c59d79df029f40",
    "905e14660082fe4b6a0648bacab039d1699939e38480cdac90ec53838ed0f68c",
    "ad0ffd6f8fbbfd479f45bf5b782df868bebb126d48e19de162fc483501159054",
    "994982c0ae9a99c3fd8bc9d5507d228871415d9f0afccd701e88bac80e254121",
    "279a4e51de422b21a4bcca6de3dd7d529a43385cda22e424c40c4d30ae6e7510",
    "983438f9cf99ca99317f10dc19fd7b7aa593fd3fccddf543c0e31188eee92858",
    "952597fe9c31e7e44e15c7e0280f13a3743409c36eb0126efd22cb230429f40c",
    "ab72e2185b262e5a53dd5f740178ee509c66ff75ee6701678d27ce37e46f178b",
    "57f1478d194d31bbcf376765fbb41a3d738838be68a7d1f49349c15c8e09f3cb",
    "1333b608085b60406727909ad5517cb20d1f48c7c39c37242102a94302b0cdc3",
    "6cf6d9c6a6f3cc15609013bd59316f01906939186ecd604faf40986381d86687",
    "bc7cbc05e5245101e71b7a7b294182ce9c060d918331e965a8ebf2b2721bd84b",
    "551b7b129348c2dd1141d31fa29fcd208393f209da6926e998711c456497060d",
    "db9cc069c56850b42de5e13a8427f7e3cd7da61d1de71d210b3c1e29dd4f0231",
    "fb85a289f0ea98cef25c5aa5a1aa11fa18d050d9be507ad7f47b4ea4605f9034",
    "a1ee011fee4000cd530cb0de50b1a6664c12ef93fa82d3f733f4397cbb1e0617",
    "10f056e37e2a6a66fe14aa1bff6954805623ea22b607aa51115b05296aefd4c6",
    "bc72f3a9eac55e6bb3a67fbfaf18d115bd8daabc43bab58db863f5c7536c4500",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "8496a4f80bc3783e468b81d852957fb8c147a91e4c4d15be84951f8a69c6f97c"


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
