"""
Golden Trace Stress v0 - ABI Regression Artifact (Stress Test)
===============================================================

This file defines a stress-test VoiceFrame sequence that exercises
VM semantics not covered by the brass_stab trace:

Coverage targets:
- PRNG advancement: NoiseBand via event lane 2 (advances ONLY on those blocks)
- Permute ops: RotateNodes enabled window
- K ladder: pattern_id 0 (K=32) -> 2 (K=16) -> 0 (K=32)
- Tail mode ladder: FULL -> MIN -> FULL
- Event lanes: Impulse (0), NoiseBand (2), GateMask (4), CuspInjection (7)
- Smooth ramps: gain and beta changes within and across regions

Musical concept: "riser -> airy noise -> topology motion -> hit -> resolve"

Usage:
    python -m pytest pipeline_tests/artifacts/test_golden_trace_stress.py -v

Regenerating hashes (ONLY when intentional behavior change):
    python pipeline_tests/artifacts/golden_trace_stress_v0.py --regenerate
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

GOLDEN_TRACE_STRESS_VERSION = "v0"
VM_STD_VERSION = "WSE-STD-VM-HYBRID v0"
VOICEFRAME_STD_VERSION = "WSE-STD-020 v1"
SINE_CORE_LUT_VERSION = "1024-Q15-12bit-interp"

VERSION_INFO = {
    "golden_trace_stress": GOLDEN_TRACE_STRESS_VERSION,
    "vm_std": VM_STD_VERSION,
    "voiceframe_std": VOICEFRAME_STD_VERSION,
    "sine_core_lut": SINE_CORE_LUT_VERSION,
    "table_size": 1024,
    "q_bits": 15,
    "frac_bits": 12,
    "intent": "noise PRNG gating, permutes, K ladder, tail ladder, event lanes, ramps",
}

# =============================================================================
# GOLDEN TRACE DEFINITION
# =============================================================================

TRACE_NAME = "stress_riser_hit_resolve"
TRACE_DURATION_BLOCKS = 64
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
    note: int = 57  # A3 (220 Hz)
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
            self.event_strength_q15 = [0] * 8


# =============================================================================
# HELPERS
# =============================================================================

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_i(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _q15_from_float(x: float) -> int:
    """Convert 0.0-1.0 float to Q15 (0-32767)."""
    return _clamp_int(int(round(x * 32767.0)), 0, 32767)


def _event_strengths(default: int = 0) -> List[int]:
    return [default] * 8


# =============================================================================
# STRESS GOLDEN SEQUENCE
# =============================================================================

def generate_golden_stress_sequence() -> List[GoldenFrame]:
    """
    64-block stress trace: "riser -> airy noise -> topology motion -> hit -> resolve"

    Coverage targets:
      - NoiseBand via event lane 2 (PRNG advances ONLY on these blocks)
      - Permute op enabled window (rotatenodes)
      - K ladder change via pattern_id (0=K32, 2=K16, back to 0)
      - Tail mode change FULL -> MIN -> FULL
      - Event lanes: impulse (lane 0), noise (lane 2), gatemask (lane 4), cusp (lane 7)
      - Smooth ramps: gain and beta_target_q15 changes within and across regions
    """
    frames: List[GoldenFrame] = []

    voice_id = 0
    note = 57  # A3
    f0_hz = 220.0  # Slightly lower than A4; stable carrier reference
    gain_q15 = 26000

    # =========================================================================
    # Region A: Blocks 0-15 (stable tone, brightness rises, beta ramps 0->0.6)
    # =========================================================================
    for i in range(16):
        t = i / 15.0
        beta = _lerp_i(0, _q15_from_float(0.60), t)

        # Backbone: classic-ish glue stack (no PRNG, no permutes)
        ops = ["tilt", "intramix", "diffusedown"]
        fam = {
            "tilt": _lerp(0.20, 0.70, t),  # brightness macro rising
            "intramix": 0.25,               # body/glue
            "diffusedown": 0.12,            # add detail from coarse->fine
        }

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=gain_q15,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=0,            # K=32
            tail_mode=0,             # FULL
            event_mask=0,
            event_strength_q15=_event_strengths(0),
        ))

    # =========================================================================
    # Region B: Blocks 16-23 (K->16, intermittent noise via event lane 2)
    # Noise on even blocks only => PRNG advances only on those blocks.
    # =========================================================================
    for j in range(8):
        # Keep beta ~0.6; slight gain trim to avoid energy jump with K drop
        beta = _q15_from_float(0.60)
        g = gain_q15 - 1200

        ops = ["tilt", "intramix"]  # still no permutes; keep it controlled
        fam = {"tilt": 0.65, "intramix": 0.22}

        event_mask = 0
        strengths = _event_strengths(0)

        if (j % 2) == 0:
            # Lane 2 = NoiseBand (advances PRNG)
            event_mask |= (1 << 2)
            strengths[2] = 9000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # K=16
            tail_mode=0,             # FULL (still; tail ladder exercised later)
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region C: Blocks 24-31 (permute window + tail downgrade to MIN)
    # This is the "topology motion" region.
    # =========================================================================
    for j in range(8):
        t = j / 7.0

        beta = _q15_from_float(0.60)
        g = gain_q15 - 1500

        # Enable a permute op (RotateNodes) plus mild Tilt
        ops = ["tilt", "rotatenodes"]
        fam = {
            "tilt": 0.60,
            "rotatenodes": _lerp(0.20, 0.70, t),
        }

        event_mask = 0
        strengths = _event_strengths(0)

        if j == 3:
            # Lane 4 = GateMask (deterministic; no PRNG)
            event_mask |= (1 << 4)
            strengths[4] = 14000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # still K=16
            tail_mode=2,             # MIN (FineClamp only)
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region D: Blocks 32-39 (hit/burst; impulse + cusp event, beta dip/recover)
    # =========================================================================
    for j in range(8):
        # Beta dip and recover: 0.60 -> 0.30 -> 0.60
        if j <= 3:
            beta = _lerp_i(_q15_from_float(0.60), _q15_from_float(0.30), j / 3.0)
        else:
            beta = _lerp_i(_q15_from_float(0.30), _q15_from_float(0.60), (j - 3) / 4.0)

        # Slight gain "punch" at onset
        g = gain_q15 - 800 if j == 0 else gain_q15 - 1400

        ops = ["tilt", "intramix"]
        fam = {"tilt": 0.72, "intramix": 0.28}

        event_mask = 0
        strengths = _event_strengths(0)

        if j in (0, 1, 2):
            # Lane 0 = Impulse (attack transient)
            event_mask |= (1 << 0)
            strengths[0] = 16000

        if j == 0:
            # Lane 7 = CuspInjection (dramatic but deterministic)
            event_mask |= (1 << 7)
            strengths[7] = 12000

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=2,            # still K=16 for this "constrained" moment
            tail_mode=2,             # MIN continues
            event_mask=event_mask,
            event_strength_q15=strengths,
        ))

    # =========================================================================
    # Region E: Blocks 40-63 (resolve: restore K=32, restore FULL tail, lowrank)
    # =========================================================================
    for j in range(24):
        t = j / 23.0

        beta = _q15_from_float(0.55)  # settle slightly less wet than before
        g = gain_q15

        # Add LowRankMorph gently to prove LOWRANK path is stable + deterministic.
        ops = ["tilt", "lowrankmorph", "diffuseup"]
        fam = {
            "tilt": 0.55,
            "lowrankmorph": _lerp(0.10, 0.30, t),
            "diffuseup": 0.10,
        }

        frames.append(GoldenFrame(
            f0_hz=f0_hz,
            gain_q15=g,
            beta_target_q15=beta,
            ops_enable=ops,
            families=fam,
            voice_id=voice_id,
            note=note,
            cents=0,
            pattern_id=0,            # back to K=32
            tail_mode=0,             # FULL restored
            event_mask=0,
            event_strength_q15=_event_strengths(0),
        ))

    assert len(frames) == 64
    return frames


# =============================================================================
# FROZEN HASHES (Generated 2026-01-01 - Audio path math fixes)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
# Updated 2026-01-02: Phase-coherent wet path (mask smoothing + C⁰ seam stitching, no windowing)
# Updated 2026-01-03: Disable DC removal for VM path (CCS is phase-continuous with zero DC)
FROZEN_BLOCK_HASHES: List[str] = [
    "6fa4c58dc1f056586541d104bd33ba489533c01fb1c755bc549c5141c8e53446",
    "65e39c07d12002e3fbdf62af2ab2e3a3608bfb10942d073e5a90c31df11443c6",
    "296230fef1d7e2c3f3d3938c00dd669f526fa5a1cd7afa180777ad3164584867",
    "3ffc20d639df134f009ccd357e8d3b934c71b155de4e0e0e70f2affa8fa7ec8f",
    "48540fbb2a6bdbbc82fc5338d5c3000e5627d20f18d267470baead07aab34d81",
    "25f0ed236c24eee4e91b122d1e9e792a831f36e83db363f5ac5bb30059a8ee7b",
    "b02f1d89b8e13ab6a71d7bc4bdf1cea2e5e4cc8d9e94a85f92bf6cc542bb7291",
    "f77ba6f7f2165a9c5cb6987eaf4430360c77ec7b604535e0bd0944084e923d5d",
    "33078b9bcec8d36370b74d3b62c65b06ba8132d3afda10a6814350ee1900b071",
    "038ff60f760b184699e5b5c9bfde5a84c411659bafdae8ccd4e04a6cb59828b7",
    "95b5c007b426e7df51f56b4c9cc4ff675e1ec643e0510b94cec0205343b8ab87",
    "8be7b636382a222895d76d161cdb235134dc18363ce03d5a0be764fe481b3b2a",
    "f86cf031b830b5b8b8dd448ee054fae98d59364c16e81a6c6cb0d0f5b18c6b21",
    "f8f30edde9e943e819bedcfc772bb4ef0284a44c5e63704542ee451424bc30f8",
    "143b9fc6e1df31cb3b5967256d4151e5bd77bdc478bab0eb57b4e0730f0662f3",
    "f4fb3c0eca8089d118d441404cc74ccedad87b67f4446f3e37348bbc0c73d31c",
    "a9142af08d18ff2e03e120d7cc2ec6e0625f2618342ec7335b98b1cfffaa07d5",
    "44b1492838a7e8605be33f47691787c5ad596f2c74a7b447936aacb015b988f8",
    "f6896919600b184d770a48e35bddaa8b96661980bee19ba9b7ec48e794f17c2b",
    "c441d6201d9860cc81439fb31b4f9402eaa12a7e411befd874025cc90cfd0593",
    "b0bde4a523dc7ef354109d258325a863694c3ec97e83fee3df90bc3bc1c042a7",
    "fab151b9d8b2a091e65f124fb5dd4b6b60d1987e7aad336904f6bef6aff0e3e8",
    "3ec68846e8275a41aefb09a90d822b87a0470b7964c1f955f85346a12356a7db",
    "13c3f788daa9cd800377af610bf8fbde1ea1337d9845daadad7438b01bec3f43",
    "ff7401ab919eb80d674402a2ca65ec2b4e1a3b876393a662433d1bbfe3979831",
    "a226e44507690cbee483929074fb2e313d51efdc83a8ed0ec65f879bd698714b",
    "1a27013da805884455698f998a1a41ca78da8e147ed25bf76462c31f8af33b02",
    "c628f56bf557cdf39989182310e3b361752c3753eae05be24671da70babb8607",
    "8d58225e3b91fa19aa177a5746232efc3e72a4eae9fb31416615763648a6be34",
    "9efbf70295c4209d224ce11c09aa194d0ff185c01a86050a3e984f0d0c7ad11b",
    "68da0c15ed97c7e9544c21aba220046088f6e8828e6e3f98c54b63ce10a27df6",
    "ce007482992a715d8ca4cb86191046ade202b3166bcf4fd6fd8ff314a93b9eea",
    "69b0e2361e75de6ce599035d0712bc4a58c69568572cc17fb8c5afd07bb21495",
    "720f7558ab3777e943b83a28e869f8b0e9bbb0eeb81e0c082722cb2d1fb22b63",
    "5e70420e40e024814b06122008a614bb318da97023592dd7c41e5ae62da8c673",
    "ee1fa22b9f01d6c6134d17d67472feacbe62951b79ecf6043521f92f25ea26e2",
    "b878c72c251b4d6af63ba4c3c28e85d18cdda45ee7a23ccc77371476a5662a1b",
    "950d97682bbf22b6b7d7ae97404f0575484ff1f4b06d2e7588f339c7310a2ab5",
    "c84cb2c7246e2c5d6dbf46e0e5f7e5fa1abcb3872e65ca0b40f350a44a89820d",
    "d0d26c94af9bc6e71df80b953bfd71819bb81a3bf9bade655202dadf207b9a65",
    "46851f32f370f180828a02f92991a42a6707e1b76e16e3eebbb0788f4bdbe400",
    "0136377edc0c865ce054787dfae1c615f346e4dcd4707c10d7e9dbf672d64230",
    "03dc90f23bd649c2273af2beb697ad12c95738a068e0a19f0f9e3a56bd1aed24",
    "fca870e1a7fe972770765dbfd667ea64b128ce0c4d0652ee05878775dd6dac44",
    "6b1483dbe1d88534c6e1ffe18bd14e6b3621fd0e2ff215c70cf633f7f7784fdf",
    "90acee3847c9f0ac2b32008f67c60990f6468905c2f755a5625f601a7ebeead3",
    "d45ddffa82ba195cc54a0b9d769fff2eefd5187fc48f4e9da2ef09e323707c3e",
    "471be7a5c2e735ff3fe86cd979edfb09d51ab773bd3b519a65783b71a7c627e4",
    "ed30ad800bb83761a5e2ad438d1f40e50268c9d8869b2ffc332e1df02670db4a",
    "bb358addfd9d63e5ad6b38b6b4746da04ae56711da334838e5e92649a7ed2656",
    "5bb8c8d549adeb91e625a8bd66f4603f486bb93a2578924f8fcc3634f76e4451",
    "7d1b3d7f881318c8adaf875ca2ad42bf3f605bf3013ec0b083266663b216af02",
    "d0a5328e65384aaf6870ec249e087cc553c8b0607c4da045f735a13dfbb6e86e",
    "25e3ea28ead75be0bcea00ca275725a086e3083946b6f926efbe8dab2600a7de",
    "3665c67eb9aa175d2d069f3c8bbd38850dc67c3c7ff61cdecb0af3e47e8b6c81",
    "a270b035fab103d9f86179eb1d71f33c942ff344df9dfb79b35b8e3c0dbd876e",
    "bf013d94d320c485e83a4b837b4559031f928ba8d6232bfae584f27abf5d94ad",
    "278b282f5c648586534c5a591cf735d668ad88effaaab22d9a8f68999a954f1e",
    "f07dd53f2b83dd8b68efd35beaa5104231ea8a1a776b1636ca77eef7c173de4d",
    "8c92a0308b8390f3abc71344f7032f1021c8932b27159012829b1d352880c601",
    "60fc168bd97679a8568484cae840664eefd74cf34d906f7e207199d41690edea",
    "0b03cf4da6388a1e888a5cbfc3d6b648bdbf19135e1c9332f1d12aa05c0bb9e2",
    "fa0c7408e97692c497c3a71d7b4e9608350c62db00f6d203440c3a935ba275dd",
    "4de1b91c65e36dd63f75f3f0ca1abaae8d3ded465dd0a439eb232efdfeb3a1b1",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "ed8a0aae46ebe1caf753893ef2a44564427875ec6c46b73026033bae2f60b44a"


# =============================================================================
# HASH GENERATION
# =============================================================================

def compute_trace_hashes() -> Tuple[List[str], str]:
    """
    Run the golden stress trace through the VM and compute hashes.

    Returns:
        (block_hashes, trace_hash)
    """
    import numpy as np
    from symcrash_python_ap.vm import vm_step_block, VMState
    from symcrash_python_ap.vm.tables import DEFAULT_TABLES

    # Create VM state
    state = VMState.create(voice_id=0)
    state.reset_for_note_on(f0_hz=220.0)

    # Generate frame sequence
    frames = generate_golden_stress_sequence()

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
    """Write a regeneration report file for audit trail."""
    reports_dir = Path(__file__).parent / "regen_reports"
    try:
        reports_dir.mkdir(exist_ok=True)
    except OSError:
        return None

    timestamp = datetime.now()
    filename = f"regen_stress_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = reports_dir / filename

    git_commit = _get_git_commit()
    status = "UNCHANGED" if old_hash == new_hash else "CHANGED"

    report_lines = [
        "=" * 60,
        "GOLDEN TRACE STRESS REGENERATION REPORT",
        "=" * 60,
        "",
        f"Timestamp:    {timestamp.isoformat()}",
        f"Git Commit:   {git_commit or 'N/A'}",
        f"Trace Name:   {TRACE_NAME}",
        f"Trace Ver:    {GOLDEN_TRACE_STRESS_VERSION}",
        "",
        "--- VERSION INFO ---",
    ]
    for k, v in VERSION_INFO.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.extend([
        "",
        "--- HASH COMPARISON ---",
        f"Old Hash: {old_hash or '(not yet frozen)'}",
        f"New Hash: {new_hash}",
        f"Status:   {status}",
        "",
        f"Block Count: {len(block_hashes)}",
        "",
        "=" * 60,
    ])

    try:
        report_path.write_text("\n".join(report_lines))
        return str(report_path)
    except OSError:
        return None


def regenerate_hashes():
    """
    Regenerate and print the frozen hashes.

    Run this when intentional behavior changes require updating the golden trace.
    """
    print("=" * 60)
    print("GOLDEN TRACE STRESS REGENERATION")
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
    old_hash = FROZEN_TRACE_HASH if FROZEN_TRACE_HASH else "(not yet frozen)"
    print(f"OLD TRACE HASH: {old_hash}")

    # Compute new hashes
    block_hashes, trace_hash = compute_trace_hashes()

    # Show comparison
    print(f"NEW TRACE HASH: {trace_hash}")
    print()

    if FROZEN_TRACE_HASH and trace_hash == FROZEN_TRACE_HASH:
        print("STATUS: UNCHANGED [OK]")
    elif not FROZEN_TRACE_HASH:
        print("STATUS: INITIAL GENERATION")
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
    print("COPY-PASTE VALUES (update FROZEN_BLOCK_HASHES and FROZEN_TRACE_HASH):")
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
    print(f'FROZEN_TRACE_HASH = "{trace_hash}"')
    print()

    print("# Python list for FROZEN_BLOCK_HASHES:")
    print("FROZEN_BLOCK_HASHES: List[str] = [")
    for h in block_hashes:
        print(f'    "{h}",')
    print("]")

    return block_hashes, trace_hash


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        regenerate_hashes()
    else:
        print("Golden Trace Stress v0 - ABI Regression Artifact")
        print()
        print("Usage:")
        print("  python golden_trace_stress_v0.py --regenerate")
        print()
        print("This will:")
        print("  1. Print version info for auditability")
        print("  2. Compare old vs new trace hashes")
        print("  3. Write an audit report to regen_reports/")
        print("  4. Print new hashes if you need to update frozen values")
        print()
        print("Coverage targets:")
        print("  - PRNG advancement (NoiseBand on event lane 2)")
        print("  - Permute ops (RotateNodes)")
        print("  - K ladder (K=32 -> K=16 -> K=32)")
        print("  - Tail mode ladder (FULL -> MIN -> FULL)")
        print("  - Event lanes (Impulse, NoiseBand, GateMask, CuspInjection)")
        print("  - Smooth ramps (gain, beta)")
