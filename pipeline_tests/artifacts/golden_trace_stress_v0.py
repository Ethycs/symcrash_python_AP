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
# FROZEN HASHES (Generated 2025-12-26)
# =============================================================================

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
FROZEN_BLOCK_HASHES: List[str] = [
    "7bed9cc8a93ffbf92e9ab94dba9f7a0d679e2557684041598face9c3aa7cf4b4",
    "7b54d71896c2cc04805df2af22ca39d70ddc941924af009aac909c15b29262f2",
    "b4940c9510ff7255ffbb0198c2efe5e1d4150cf53ba996b581afb2de5b5f4b69",
    "09d6aba497d0a269e9b2de2f0c805c5ed83e9a9685d47a5c12696472c2ab0f7f",
    "ffd0855205dca55f7d1d8f8b52d9baf522206db147a18b51b64e7826588ec92e",
    "887e8270db23f3d85fc89112a0399ba59c0cb9f3813df52bbc46832b265b3616",
    "4ac7127afe3ad016921cd52b40fd9c3149d9972ac2b0a17876c418c81705d7c2",
    "3c1213c0187a15d6af909d096ee59c02e1cd3c2fddfe1988108ccef1bb2dd2a8",
    "bc124231105c7a99eb1051a74c0f5aec303dcd2992948e4ea2f07b54e633e14e",
    "0db175d1e14c62e853f476381b333ce8660e6c6307d301865bcedf95c5ad9d45",
    "a265e947e51cd15ff6b7354c2881bc19b14f2a66629fe96a22e3d09085055241",
    "213d9303bb936d6946ae43c827bcd9a67022e7a8c01ad3428c5b16bfe21379a1",
    "fad9598d2901f4fa0571e16d7cf060834b50fe44b78e2f57750959677bcaf7e4",
    "f2170629526718807183334f53988b3cd108da5ea8814521ae71828ba833ec03",
    "d9271a01d7babaa2726fc8a52baac54bd62da34bb5df590614b2913410e5e684",
    "6408c316330a945a495c4757c7a42873d4aabd2203885726336c02faf3a66943",
    "444e4a2d000d2188d10c3e82b60feddd486a53a75e599056d769e059ef93dbe7",
    "532fc5d619ee186c5435c7953276957056728094f1841195da5955717d3bcaa4",
    "1c7efac82f501848cc99d802aa80e76e6d4509c468eda66b0ac44ca7ae882cc7",
    "62e32d9da94ebca39489b80ebdbc7a70239b74e71c1ce274c48083b23750a9cf",
    "e152f0bf8c328623c7dba3ddf7f5ecbb6905a5e7ed8339d2cbe6f0d7daa6495c",
    "918851e6682f558fbc040174db41a7685880bb4124e44dd35a614c744f315c61",
    "b475e5f151e1a40fd10d9deea11d1e0b04878fa77195f695eec6eaec9793230f",
    "a7a01ce7857ab59ded2e2d85a33931cad011e2a3a943fcfdec44e15f8775fd63",
    "76ae2b1d76c696ed00dcfdb5512cc14b390f91072284e0feb190d3d8c6de227f",
    "4e251093fe6ca3edc95161ff619619e5b16f1e96f96f76482f89c272914a20a8",
    "b6a54aaaf7d4dae3fa240aaf85334b2db995ce3b861fe3f43bb6939c48682add",
    "46bfb8670820055f091e039afa57ee3f23e0e1d643b57478ad44ec7f26967d0b",
    "4d8c397cc3bd55dcd7dbbcf7479f1915dfec015d437927895785dc6f927ce9d6",
    "a51025e2ab1fd8c0ec6540a0ad8ed09a71626551fa30f0e619aabfaaffbb3420",
    "11a0e983158cb0f69122314766e57038472bd56e4dce8cd701947e82b210b463",
    "e9972b171941f42536c5bb8def82a6014ef4b1775e720a1e6a73761f97bb6bb1",
    "80e432e12254fab343b064303dbe45d6eadc504f39b64c71d0cb85e8be601665",
    "de4e71d102ed1656d5935b6394086dd83bd87c6bdee368c56076fa4592ef922f",
    "a722509bcf478f525a23362d1ea495eb92e159d12b08fff7f52374f22183b00f",
    "3b5dc86a832874c0a6c4a2b904f733d4f22b00b8538d261204ee4664935a87e6",
    "1e0c7f7cb5f34eb36f4eadcf13b6e086be6d5180ed31abb89c356aa9843bce05",
    "6f7822d0cb69b47474b5b6c991ee54435e5b6bdaaa6b91ba32905cdeb35879f0",
    "e743c9907f4a44377396705fe876ea15b58202224241509651bdb4fff983a15b",
    "f51699f5ff99d23a1b70bd9cfb140c413b19253bffabfc0cecb345cacf34f568",
    "60e392848642b791f575de91279663307f3892dbfbe87399f6475a7316f51775",
    "eae0b71b0de335ebeaecd1f6e8984132aea857472649b5301079b5ebd7cf2165",
    "d8e0f74c2beb923aa6e1f3b8de6ed1069f82a57b53829dc7e4fcd1045635bbbd",
    "a37a7e3bff6acd663cf815c2e617c3daa6edd36f12263d4f2eee5de39c69e64e",
    "e591221aa191177ffb73ad775dec4d6602903b2c8ff19922b4ce2de0e0164ef1",
    "accf77eb4617cc06ff5a6b25dc41f0a9ca44fe4b556b0880516863ccfa23f419",
    "e05c94efab9db80edbaa4f0522bfdd05b6e2e0150bce9b077841de77841f5aaf",
    "39479b5b7435033823e3f0f8bf71a0cf15953df2d023287f9a42a204b4e06679",
    "9d5a5fcabf293adb9e52894e07ac6c27c31d930ebeb9d9860ee3fd024d5144a7",
    "c5894cda13cdf7698f9a2fcbc523efae2e1796cb354b340872982eed0c5dd42e",
    "ad0e17f8ce37587c05ff7ccd2e068a5d4355f6d0f6b485fc5a228bdac58afea7",
    "73348d8ddb42c60d61f0f7c3a43950cb78679eca09909fa8e0689df73272d833",
    "bc1c1d0f28f058bde130e2b82605cae8421167e40ef2022429a6117033360e09",
    "5613de37b92cd19f7527b84983a39fffc5c8c410fedc53eabce8a154f17002ec",
    "2f017e4578ce6c62a47eea20cf6bca1f333c7cb1e166435a22bb9a3689ce946c",
    "d7978e94de475f2fbbe8578069fc77f7249b978f4256fc805e978fda4a910537",
    "be5abfc1d183e185cbe89775bb3870a01b0381d6ecc4240743a94955fb2feabf",
    "34c8d5599395dd7922b048cbe220cdf6d9b1961bc0a1bd6470d47acdba237e09",
    "a53430c2720124299b35a1ff2a8ef23be3485fb80904992a24192de9dddd1a9d",
    "808ce86b49c9fcba2eb7f5b8d99f016cbe4f42471bf31ec80d9a802067fbc369",
    "ffe9913ee370c77ddb77e39b5a6ccc74ba806a2b616f14277e5385a33be31c1b",
    "94cdbdfe145642517e8d3983ed80f27d9e7cb3b3f5f9041ee7d0c7d89bd09182",
    "6ca09ff13227970581bbc56c8977a052c121c387602f0b34bc3585a587ef8b41",
    "a9cb887867f52d6e1919e41d4338ac63ad1f21edd73303cb1b318e50e217431a",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "cb54120a25178f95ef657f6f054686ca0b08bb8e468772d3a3061268a9141f0d"


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
