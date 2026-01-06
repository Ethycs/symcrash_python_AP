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
# Updated 2026-01-05: Skip SheafProject in CCS mode (was corrupting polyphase coefficients)
FROZEN_BLOCK_HASHES: List[str] = [
    "2590ee8362ee7e6bc686aa00a2b3a4c2b5beddc87b33578f243ed3a881b810ac",
    "2b98b500043673f6f73594dde96363e6e138911dcec566affac728f11cdb468e",
    "d77951947bad21c664b28ba06e29309913e7f691dd0b2ef4f668b3a61ad2ecbc",
    "3c1cbf3625f0778978a29c53c2b90c6eb8f74302a756bfdf74ad5c872eb5d204",
    "a639146389f865c8d209d68facc992c48d88dfe6a62669ba569a852a3279442e",
    "5835f80fdfb60b3504b6f73d092b60c3402985c8c5dcf3470268d6be1a338039",
    "2fb2a84d5a98d9d0f0ecc3836109c9fd7be602fff70e7bcb8c239a8a133280dd",
    "da5f1903c310276f02209110238f86bab70f4b9c9cb722d80fc3e79d814d707d",
    "6eaf6f934c5ea3e607e6187361538d31223ca75f8c558dc3897f039ebfb9aaa1",
    "43e7fd3321206d86151a0420372a4bed7cd2c857fb49c8fceaf297f8eaf7e59d",
    "8bf1a45d0fcc7c3a58ecf2045f067bb9ac681b4384bfee1c9f66eb777c2d5553",
    "f53406294a52453b0f04acd706be533f87ee49838acb284c57919d62c6fcd345",
    "84ba6b38a0776fcff5660127dd24c8d3d5618b8d473142fd0092c82495abc283",
    "fe808715b32ab9280815712d6a79d23aa8f1ba3fab9e3daf12f39b4b4665aeaa",
    "2f006861830746d893ba628e1dede400ec117b338a556df034df35acf416caee",
    "a0afd384beb4c33c86ee67da8ffa646b24b74f71641f6f696fbd176d70baed35",
    "8aa9caa141134241773370cca94891e48cdf5e61fbaba1b69689db67555f64a6",
    "d2945ad1ba95e5d6872d2867f2e9d5546473f92800f148f81af5a423b0d00142",
    "2e6472ef55f532d2caab8deefcc3f20aadc7502f9264ef8fb91addd7f2302de1",
    "350d519210bc3154f3289852ac5546e401a23234a78588b3d35049e49d66064d",
    "4656bb1bfd489854d26685b725ffe18b32cd14828d3068b0a79b5daf66fab8f3",
    "5bf16a7057c765600e622324a7be7618ffd4deda4c84a53f1b939584fd771bf2",
    "8e13cc359db07ec9bf098bb103c52e778ce44977a6fdb324a057d410e9744325",
    "77fc9cfe8e4bdb4264d53a026ead44384bf9f1b2df44b052b3b2589cf47c8914",
    "66657014f1e08fd8510f8d3cf8b3c52bd2d7f2938b077694a33e2207496eabc6",
    "1a13ac0fa91b45236ef0f8ab494852318a5a63e7bf5215b72a2fc4d40043ff4a",
    "c10339ac56db058cf6068028c39df40adb975a73ca95aafe2e29d8deafca10f7",
    "6dd0b2e059138a460dd71cca5f2fc22ac82f8dc36a386dc5379f3190f8484097",
    "285a3a559b831aa3367e17a2cd55529cb725ac254ec7a4a3c63f2594f6c88c0b",
    "f64f2ca2c3b81343af609ce4161bf7228b21f5994f3d4b870565e93e9ee7e3a3",
    "93e9989daa898bf2068473a66a79d24750fb8405907ca505f3e55c6abdc79480",
    "8cc41c4831c98674072b2c0037f0ccff31887f074ca886e0d581faaa522684b9",
    "65edab8018fca4b386a6656f8690eef971a1d84fdc9057ddbf1583905e8dbfd0",
    "722a439ba0927f22ec95bffb1ab0821b8d367de697da0bc87a062aee7a835807",
    "9584751ea87f560418d81745328b580249c8b07ec351dc889b7ee12b5320b37e",
    "cfa1bd3722742816429972410bd16d2a5dca31d63ad1c63ffe7eafb60561850a",
    "55157e49e356fd4675c4b9a222c83d6de0e7ac2c1a167635988d60a50e49b81a",
    "8fb8b6710e3d6e2df9b2c4c239048a16aba60568c60ca04fcae7c7848a764413",
    "d831c4e5f8cb71f0bb3a259a422550b92bf17f1be15a4c5288ce96ad42637bbb",
    "b614df406633ce413e0e0a5f22c5b0a8318aaaadaea135e5b150e8b72a164453",
    "88374c5f83c8d76c21f7064df3ad7eaa883a72e0d92262c7d2aff3039b66ff5a",
    "a106d6e81b09a4d9e21220924cd1061cd66f7194aa5e258de5d8a95e2ffb2c96",
    "957d3c1dedb5ff7431846b62109d31ca023da0446dd47d4c2fb90e36234020d2",
    "5d5bd169949b9ca80ad62b467d0f63b5e1085d8252cc6e937fdd89a9e1072e14",
    "7199eb84aa7fb04708daefdc21059dea3cc815c8c46d255abe45c7f0e5eae076",
    "a43296029ba1f76047bf81b93e88c33f76c32eea2eface3e61746417f16291bb",
    "f073183692dbe2f329c12c4f32433634bb260f342165c4c6becaf9cc8cc96e7d",
    "ac939e7d62f009f42ac2ca7247ce3eb5519fe9e137859f9319984cc6bed8123c",
    "aa6c4668694d0a200edfab85ed2355ae4cc8f12d8d67a5c6cd67c168e61e8ea3",
    "f21357e3ca0aadbf956d0219fd60ba1caa61698ee5523d81e20106b486dbb0c7",
    "d3d70ca2ce60428e8cc6e1d95abdb00cb98e7149a4800cd996971408416e7d66",
    "a111d0398eb80899565c3bbf91782718ce523c60c6a8dacfa06089ec7c5e4eca",
    "c86e21166a92b641646572093a85a42c986ea5d68685e3aae16afc390deae5ca",
    "931e08b0aeab2c2ef8ff29e3dea7efbade5980293cf493bb157303b8ea5acd22",
    "86ffcbe3d4a34aa6b45e12906a2e78eef4a714da76956f1e1699ffd254b308a7",
    "944ce8fb35e5d6040e3fc89983f44fc0e954fa9ab3fb6feb17de2e81825743f6",
    "389082ca7810449bacb175cbebd032baf67b446b03d232b1208c3faa6c06af81",
    "41f4c823288107f064aa4f418668dc3d81d32d0d2cb5270e848bf0725b5fd585",
    "4c67f76818368b729260296c4518869be8c19f7de07b550f9df1290dd8cc228b",
    "b90d1b687a0e916290919dd6844409cd99fcc95fb750a6a739575c1ade342289",
    "8e30883a9c6c4866ec72f5c75ddbacb20b2f1c8c421fb57757aa5f071b912055",
    "25a8185f51df113e25ae07c1f87d820b4752cc3a4752d174ff87c35df918eb2b",
    "272c7bc890b7e548f33e6a886ea1886811de47cd41120fd14e2f6b95ff63d4e5",
    "79420328e1f83f0e9c1d1df09d0287013c847870f9bced9957eb0748c55daf59",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "e7914a83f04df14f976c24eebdc3b40f20223dd0eafbb8babc46dab029391128"


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
