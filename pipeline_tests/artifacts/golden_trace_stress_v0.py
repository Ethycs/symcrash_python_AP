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
# Updated 2026-01-01: Added 1/sqrt(2) headroom to prevent clipping on coherent mix
FROZEN_BLOCK_HASHES: List[str] = [
    "6e218c163320f04c885cbe72e91e297fd19a7edc9e497a2f1473f4d3d740fb77",
    "43cc508088b5eefe57be16d1d8296fa1f1379c07fca6c5f7f10627a22d0e4c41",
    "ef0311fc8a37bd0f8028de5732ef400a496f8636e91995c85731efaeaddf54b8",
    "9336b60acd5c51d8ae1af4c7ee791b6197bf0a035ee4628be1142ec6d70c4b8b",
    "592750953bda193ce02d200eb66dd8f87554af9ebdc1b5cc4c7781e451ae01a4",
    "6dafcc59972376b8f03d20975dc0dc8cc484a6965ccefc8c9518e504af10297d",
    "10a8357f5adca320d60ee5514ada79f1b35c9f0362e60db99b11e412f2de4849",
    "5207dedb06b3a2db70b4ab76e804a756df5fa62d08b4ff141f7e60f9dd14715c",
    "79c5858a0f497399652a299ab8b9d2d1b5a727fec1f7411e2216d7ab835a1a69",
    "626c0a74149e863a703fddaa9a8cc3994951451fb05c57a1b6a329734de4dfa4",
    "7d9d6b9dfce8f2212a8499278c626a23db67e7cf39ca0539ccf6fd5f4d3659f5",
    "947fc22b524157f29bbf42b89ec1801d87bb63104ad81e64c79e1a6ad572dacb",
    "372f4561730dac68fa88a321a395fb56008efec9f667861809cdf29259f5eaf4",
    "af1897257be27047978566846d93b104186d6941467cb14f244ea985ef7f11ca",
    "41e18a1f4a46c0f481241dda1a6c62b55d8908cd1ed0bf01c2a4d6cab6db91fa",
    "cebc74a0793f9dd23a360703618f3ef87312fc85499b9180fd2049262f6bf82e",
    "ab80e1c24368f4877ce796fd4296b2889c134d1a4512d078f9b631b4f3f61457",
    "979a974ba5712232e208c823da240f786444bd497097468d6e01633365cf870e",
    "cad723797d953ad3f1a1a9a30e61191bf85d4958cbdfbccedb5cb7ba7e728298",
    "50889c594e4d17bc38a7995fb52d2bb911b5998cbace658c6d5a50a66e453c10",
    "1252f63e1a3ae65abd3997ebfa89817c58d4aaf64d74a78251bf14664a8009ce",
    "9495f570d033399192061c839806964182ef62ffa9e3652fa6af6af17c12e53e",
    "2a16af0f0de2495962cce14875dfa18e60c16b7b2f45da95c91fc0a5800682d3",
    "b977f33a08e8ac26374447e487970167c01e4493e8cc177c7c374e6da99ebc2f",
    "7f2747fadbb2a993ad390b2bba561d0acfdf21536c3810de087a15a209b44bcb",
    "384315dd37ea37deb2d7ebec214081e38b2efd4261fb93af6a2eb53b0af1d785",
    "f23f5434e9e5de5c31d4b1f343c61826e6f1e8d006dac6437cb148324cbff45d",
    "9695f8ec3dddc5e6c73326b4702ff6cb0e9fdad2a5cac406c426b063a4c55a80",
    "c9a4abcc7ba872fc9ba08bb3eddd7f8ddea540891e6e67a24f74e6fa9236c15c",
    "86780f3ac48ac4ee29228309ffb442e430f87e9f749b6e5cb1944a8ff39aa35c",
    "413073e914790a8baf22dc76941f0b3d71a3bd3b9824416278fd9a3d04a5dd66",
    "b9fe8ebb6ad8916c5bcb0c70d5f0f8ad0f870d39776da2cbce1854db1804318b",
    "78eb587a1ead974ca6d545d5ee7466c5bf25b09f943c92e7f1358e623afefa24",
    "4c0a3791e2667407725d4d5bd14b4cce9218c1f8564a4bc8bbfc74aefcc243cb",
    "f4cc8c0bb7492e1d5d36759ad83d6477dc7afaf44d5ec038c0a6ed15af1cb74d",
    "aa6992a886c04886c5985b91423edacc24697b474244e44d9599551a675db5d3",
    "d2c3fa157209a5839c18629cf80cfac0c7032b17be91ca5ef5c274016c92cc5c",
    "180d46838c80cc4a8e5c1497856ddcad11db599941010b60c33e40069deea988",
    "1de4ea9b5f040c08e97dd574d5d19ba9ce0a8372d0080a5de931579b88867a39",
    "69af382e4e13f6064a7ddfb204ce5e70ce8785a426806ab14338b9152d89b0a1",
    "0131e7154978dee9688dc1cc8976df4191e8161f90b00129995db793560c35aa",
    "e278c007a0be7798bce74bf568d85b71003b8f5e153bc9d1143aa7a92358b4c3",
    "aebb39881367fb713067e02d8348966dc6d0560352b3105344f95d9092fce814",
    "603cd784419682c25ab78cf2abb6109305f3cba991ad75acfc84c6120c7c8b8b",
    "32c3eb9194c85a2a9ea0e99caac39fd0bf83643d7636fbbff18163d4e4828eb6",
    "7eed1b85dace9ee2ba29ba61bcfe49e2021ce044589f527ad68cefdc03a22f78",
    "31d344bd90baf26825d5aaa770ca8548e1e369ea082bf25b0244554f883fd957",
    "779a04efc0905243d6228cfe1ba11fd91d021453edd5f405d89158c4cbb8ae30",
    "e7aa9c48bd77ba908d0a8f6db1d736aa664f4778be7a6240aaa194205facf597",
    "b1e9a06a18368507a97032d46690c577b8532e381453cec6bf9b97e6033ff9df",
    "34a388a776f08df8855692300141ec46308f276818098e8e0d44a6987279054c",
    "3a8da573bd1685773292d7647cb90227d782e3f95ccc3042cad34c1b225c22f6",
    "9f51c8cb298fae46ba310cbc758347ed3de043ec5da166fc016d3f9881900c62",
    "4bdbfaeeae3fad468c26d9556baa894fc7d5f74b40d14fcedcad499810003a03",
    "aa5fb90b70b7c75426f8b6b213c31c2fbbb13f7f25c7ac441e3bdd9af8de8da9",
    "f8784477e23df19ecf85921a60f69d08ba6c076927b18ff7a9713de5c611b84a",
    "f656e110b24f7eade128d04493170bd13533e7f4093fb00ce72aace4d1a19d9c",
    "1b3397b0bf1596217f665773758e86c23f712b394a00bb6ab68633d5be81e23e",
    "31f97280f422561160802d2ac0fee3227ae1bade92304066d22fafb12eca0110",
    "d3137a22be211bfada3eda26d50cee676390245838d304f78560a805d1da4ac1",
    "715ae7a0f8fb2845b0fc13791abb4464998e2d4ae3c68a344f0998771541d8d8",
    "32e2c95586212764683fb4fadd5ee4728651fff77a1b125aa5e92e5e79efe2e1",
    "152047270d738599d9040bf2556e95fe770d371c2b03596b25ca59b524fc77d6",
    "56ea63812c046550b3de4b48f6cba9a415c5edb486a03e3294ec59d221874db5",
]

# Final trace hash (hash of all block hashes concatenated)
FROZEN_TRACE_HASH = "e908babe46700c0f9765ad3e5cdd6b6f52b67d29632ec85926e778140ba53f9a"


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
