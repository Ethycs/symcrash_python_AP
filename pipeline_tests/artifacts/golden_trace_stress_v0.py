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
FROZEN_BLOCK_HASHES: List[str] = [
    "83a7ec68d71a11d184574263a4078f1415673ce4daee94d69180180f407274b8",
    "5a477f6c6ddd89e207bee0cd95a4d9ca5ea3caebece26717c5b31c371c0deb93",
    "065537c3338aaaf95be83f4791623430a84b1388bf37a6128d219846b9eb3fc3",
    "bc3945602522883ac3c8e064d099f99907f9e66cb456b6cf2ca1e10ae4e513f3",
    "1c7b4e1d9c2a57ee6d9385f7d5c654bf11f44b5397188f27c2aa97d764445b09",
    "10905d6e37f78c20fc093c7dd8dd08d4f445d2902be9d68f268b34578034eeae",
    "a96ff8032fe831ff8c58317355cf5947b2ae73b334044b15ad9200f8e65e7998",
    "9d982c147b06938839a773c5bdc4fad68972546bf77efd851febfdd07d4e1c15",
    "8f5c0c66abe17598d5e1ad7974321c995ea17846a11943dd09055e0ff95c38c0",
    "97eb940b6097eb6c415caa5b5a2c092da1b72913f8d150799a275a94cca82377",
    "7ae1d48826bda73f09e0041c4f52f97a8d787c678c9f5776b4b4d3627e5c48f0",
    "0e8c42ebe5263e05cfa74bea9f3fadd6c636c9c19de3083a1357024abb73306b",
    "d320b0a37595c96e688eb1cd576d0b13fb01e21b45aa439827c3eebee9a4f56d",
    "a212ebdfdb4a3f58977e2327489c89efe6c10cb7705ac0b3e615eb2412731189",
    "41c105e5c1739b27fb9d3f7ea584457664ef1a77197e10a3fbaed554b09853b9",
    "654631f202a6af2f8d57ba337ebc3cccd9f1a4d3f817e4d6e230a8c5a396164e",
    "678e546f4205c3ae777346e92499ca8530e073793bdb7dc0556d0b7501bdec90",
    "d6d6cdd9455c4028d45dcce634676e67ae55020cc23cb6ee0cd1a81c65d0be92",
    "0297bbe69f2c53bdbbdbc29e558106473a138b6773049a447083d6603fa6b66e",
    "7dd7f582a5cd66260bdf9197f9d335d5a3fa729810177c713ef5466315d909a6",
    "02db87f37bb4076ed4eaae86f8c593ec0b265cf62864f283052c4771cf6f2ec3",
    "8c8aa4bf3a465ee0de5efd18759cfdf7afa8c6f563b74cd80c1b54816fdf2fdc",
    "5b4ba9ebc7f2ef8642f669e09c32b79f6cefe47f04a25aa88d4e95322ce0ff21",
    "c8158bd3b166848d14f510533eeaeff34ba029d1cecf7d08dcbfc68df4aac455",
    "ea1ce951e3cfeb4a450a585a17c35e0ba7924237fc6ce87604de83e66c21b8c7",
    "4fafb3dba457be0020280b07d6cdcc1cf34c7ce6b2cccc9b6834721956829a5b",
    "ca5fb5b60e3a6b8f1918b749dcdf1c6759f3d90b824a07fd802d93972695451a",
    "fb4598dc6dbe49aab0639ceb125e40de8bbc18586b2e735c6e8cab6dd19af9f4",
    "533dade2f78b2f4a3ed2c678cb58abb7f913306c3cd9a3d39ca34f765aac2628",
    "050a3147356023e28eafd99d8c97cb29f8cb4962e658d223aae8de4ef3bb7cf8",
    "b93a08caa104f6af208a197fd5c702aea8fa092ce7b2831f3bdf25b1e928505a",
    "c9b6878346bbf1d2190701c453a10a4220fc1fb51cd9acd2078dbe39d2b0c823",
    "de3d397ceedc9d17b9d0d0bed9b70f712ec6ef8dd6c3a0678507d4b34c9db7f1",
    "be25bf19858becc6be29c274fc10bda4bd6c98e6cfd5f3caebe5b2f591671ac0",
    "bb0199396ea89402032e2b2067bfb42599b434363be19e7398c9d17c2d84cde0",
    "7e76ec593612f4362917969fd80768c2ab815344f4a947d2ae35e3501ef798cf",
    "b6b5ac78f57d6b3a5ecaef337e37269ee011848f4dbd59cbe6d222863efec758",
    "79033930fbd97c6fe8c1c673c040c2973ed68ac762935b1d708413d0bf75265e",
    "0cc499835bb934c90565a87d62ca3c96533d5196641fa669985b00db3f28c6c9",
    "bc9130542b8d300cd12f4cb80fcb89f0f246275fccaac385ce88f159ad6b83ca",
    "c8a6501a88e628ec95a87de6c3cac0e17236b7df0c6741dcd3b3e3a2fb9fbf57",
    "b7c984515f65e16e171b6c9d532e652ef910ddd77357bd820c0783988f07fceb",
    "0b10c177b94a540cb1f6e1bc1130bdebf1558b4d429404ddd0ac228bb1a40dc3",
    "a5e7487ef8cd7adc7e087981ee86d7c4f756e84de40923d29d39009441b357f5",
    "827d37821d8dc42abea9e0bcd8cae751a2c4e86135e5843551b701783af49823",
    "b80a786cb7d1408d6ea97013157d9ddf805c4551c32eca00635ffb59427a54ab",
    "4b8f60f0842cf4da0b4832e3080760830e59e60dba4513b3540719971c636c2a",
    "059ce866ffb4a8853ba0dd3c17152b4d9494ee8d1d2b6d4cfa4776dfba91e978",
    "95255d858831482e62a84bd2545286667bdc98ce83fa6f8598f8a568e523ea62",
    "ade69cc366f83c8c66a17b524e5d0d11e8b72a99d58279ea9d571fcf7f171750",
    "2a4d6f6a7029792a13db93d70eca6e59d04e6c61f09ae35581df8a196d80b168",
    "b6c9fe40cc85a21101393cb7740c28a15c25d39939920cd7411d293af7fc1a77",
    "33af1f29ada6591ef472ef1c7713a9bd57b947b1028115edd636780e22600b27",
    "cafd572a5c544ba82c9ba3371ff580f68875b777d327d461d7fb5374437415ef",
    "97395a69d0794b4d8c61841556ffef93430d10a6017b8b025f98df9ab865e5d0",
    "94b2f82ba8c36910fcaaa68c4113a4016b21b279912e2879fe8ec7797b20659a",
    "9d3f1f2afc964f1af72b9d29f3e450692fba2d2d5acaabc1c5fdca27b432e139",
    "09ed8075b90eaf17622ee97aa0c18809b8e15aab4892de7932fa6ca55ad20588",
    "1da67adb95871e78eb4dd1b8c31716b0053f909a83bb847814306fae5b94f511",
    "2b91a8d53e8065d18623166240363c008a0b3c0113f8bbd095696d8497e4a04f",
    "f04b1a2bf522e061d364a0e1d5cba030fd7e840a7497eb5ed5687d38ba479c7e",
    "8a1fc21ba933062c65e32d809a112a8c47d26acc00164c36ae4b2280962b57a2",
    "07b0728c24f3c480225f6602407f59e28a6de2301ba411269d276bc583084bec",
    "912eddba0168140c3ed6681f99121ff575c2ec42576b9fe9c5d2af8f8f39fb05",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "ff3d9da359d88e9990ff6099ef96f3cc4057facdf948124d5b23946ee2029790"


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
