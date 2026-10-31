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
# Updated 2026-01-02: Phase-coherent wet path (mask smoothing + C⁰ seam stitching, no windowing)
# Updated 2026-01-03: Disable DC removal for VM path (CCS is phase-continuous with zero DC)
# Updated 2026-01-05: Skip SheafProject in CCS mode (was corrupting polyphase coefficients)
# Refreshed for exact Q32 Mode A tables and RNE interpolation, 2026-10-30.
# Original implementation reproduces the previous hash; a scalar oracle
# reproduces the new trace. See regen_reports/mode_a_pitch_20261030.txt.
# Regenerated 2026-10-30: signed-RNE lifting; dimensions/headroom unchanged.
# Prior commit reproduced; Fraction oracle verifies new audio.
# Audit: regen_reports/lifting_arithmetic_20261030.txt
FROZEN_BLOCK_HASHES: List[str] = [
    "e471fd5bd3dd7d95f811f84343dde74cd005aa2315f336f1e56a0a58d2a83b5e",
    "08af91add8dcadea80b2cb4c787e512a5bcb85751f9502d9ce26089cbc4efc1d",
    "e418ebc395a97f92ff3f5d46f284d7f6f20d724030c3b2c830deebb979a3b013",
    "2c43746a82c82815653a346efeae409726bf347ad8980d4544d3db14814c9e67",
    "c2fded87c23eb33c4576121149068e5224cee9dd849ab279323abc159c7a02b9",
    "c8708839da0473e9ae5262d6a81add27bc7fe5bee74d8a55bf7cc912180f4695",
    "89239f450ad5eceadc9259fcde308fb5486e1d2b47a3b6aa49c6f8dde09629f1",
    "9eb33129bb0b2b75f03b0eed024d4664e3685716310401621d1ac15bbaf792c8",
    "90c84921a55802206ec76f135be57611dfdf52d5f6b2e96b0fc1c41d5f37037b",
    "36e960d60a9c3db739b32104f64b2802cf10c3eb21ee2b162857678eb41ebff3",
    "d0326be8827873fa5f1d91f715f83ee75bba7a7fd8088852e9b95ea84ca2866e",
    "e3a886b6024d9f742a94780b7691daf9bb183f8e5d28f72350b8a357c27e0405",
    "504fd6dc68e3073944f03b09c07ea951173e97410472d489354a47fb47fcfc06",
    "872449f36ff9d00d14e6488b3c8b7f1fb1c31da1da152798af0e8d5becc85b3e",
    "474b55aa655c996c7fd6b7573553ef425d92c99f2dbb29ed2012eaff87d168ae",
    "bbd2461cf9b06701e73e1eccc41f2d34ef49151c28f57514a7085269623d68a7",
    "35ee36d96dc6efadbe4a5a68c4ac3bef10540ca9f953d7b90b75691434da6885",
    "891d636f4fa68100b8ccfc23163ca253fc58bf738031e6ace7649cf42805bb64",
    "c11b585dc5d83b7e8ac3c908498469f4b6be026236dbb334ce89439ac11cd5a7",
    "cf9c55fb6204bbbc633091ebf798f64058e1fcc5199433185373bc5091620b8a",
    "d62acae0e0a8b14b582c9dc21751b67e1f5957db3c063faf7d2dc144c423b21e",
    "1685e356951ede3bd00bde8e3e4a701bdbf03f02b1de8b80e94b00dddc3c9da5",
    "788d455450c77edb758ffe6407ce26ab07781ff8b89ea12b2477694b288f4897",
    "fcd267006183def8780118ae51fb7af421e6cac534700afe20e8882b5b5a6d9e",
    "0107463be7719db3d2831727cc7a24d4bcd13a34691e7f2db7cfb8ea9b7ec0ff",
    "e2e417cc8683ce85c6f9df8b91e1f1c511aa7c966bb413a223025f3b0e8649ad",
    "025530d0bce9b5da6aa3d0b797bc12dce9e94489b9e5b5e0fb530e88605d596b",
    "2be0ff0cff65a7832423cb03f5247af296cecd8043bd88b84ebd32149d19aca4",
    "0cb40056d3efee11b7b704660f48b41f1c237161046995516a8af728e84097f7",
    "37081b305abaeb24de0dc4fe63e36e2ccfcc144bedce8a33ec152b72dd84c2ce",
    "7abb5c492f4de40bc688f437f1e006f7df31071992eeffeefdcd53887e05ff90",
    "49adba99ab39d469f9412ac26f1ac25a06fda3cb9e5d61980e40cb89d6a36347",
    "3fbe4fa946423638a7b5ea05570acf187eba505b38d7bf5d3a7a801b3c45f282",
    "f5b8835c730f89320bf2a4ee4cd9928a6317fa4039a2eb3f9a277ad3715b4569",
    "ae8ae0688520db075ff8bb5ca67f83e9ee7b4780478feff63d76be13ff13960e",
    "d4016166ef3ea51095fc6dcbc2173794e1699551c77ad307f11bc41e6919077a",
    "b4aeb49ac250396503a86830df803e5929bb45484832bbcc4bea3b745272601f",
    "91b8a5523566f86ed076c481b2940e2df10b04392d01d006c463d379e6ff3afb",
    "1fd0e14b7a477ef4872772690bf8e13382de0a1c625daaa5740f961ea12e9656",
    "02678fa65ac2124ef31daf11f0767251cc08118bf33a2dd726103645994e9af2",
    "aee31b5929d52ebf8e5bb5de27ebc466374c9f4eaf48e3245b4489c22e15a6d6",
    "a3ed1b723941c43ddf787fc901c4802b2ee23673f744116047d2ab58d4c2fd36",
    "e71f58e21215d3624edf321c08a2e5e56e85993a70c393f40f51925ec7ecbe67",
    "02bab06c66e928999ab40d0ac2a262ad860b21db748d67eb8414534d3dea0b05",
    "149517bbc81797efddcb39f318e25502a1227ed2128056d26297183d0d66e466",
    "89e454e48452a7b03da65e5dbd4a09b7b40cfa9dd9ecfefb08c6f07f91b5c0ef",
    "8fe0a58569c55cf9f36bddc86ae58ceae03e6c782eeb5487d1a5b6f3fa55b1de",
    "eb98463e1e60789d0978e9e73d585a8b3aec7b70b2fea9509f01a54b4d1e43d3",
    "a5304f1a436c8c158177ed9759347895970cc5a2928ca986f4b30d2ce28686fa",
    "8e08c9003fdfe4b21bf199eca6108685c588664a0c547c113009dd0efff2d7f4",
    "2a27f623b68963df3f6c2cf28898dd0c0006f607945c4f1abb1fc445b48911a9",
    "c643d3be4ee5361f3d67dff258db147cccef6d5ed69f4f40e57b22a53ae72620",
    "75e7bf0fadef79cb5520334d71e5fbf199d73dc79297eacb4f0a2cbb412c222e",
    "9a36e4f519d4096a4d71c518ec7a14b2a38875bde47d4a38836fbb32e172ccbf",
    "f24e6e9a9a9621e2830b926329d747af9b5fb0ab95f88a28c3682ac395bf6203",
    "dcde58a541a56e2a6d37d584c40d771e368f4c6ad56228506b50a1fb701c2d7d",
    "e37c481b0128e48ce3a38f330a2f0d41eddb2861824c93e60b5a614c63d0e850",
    "76a3905520d2a8581f5644ae0c99548c560697e8bc88335a8836c7c67a3b57a8",
    "2de4ed25a6abfc24f41ccd5b737b1dfb958d661afbdaddadca1273fb4fdf3ecd",
    "954f7393c2060dc41f80b64f376cb03f822389ec71cee792dcfa99672f875588",
    "ec77524b911a801e51849f728d8fae67adef2591f86524b971d7a380b5308cca",
    "c4b587844209df83d86b561faaadd04cb071c7d93d5ce9aab1fe056f03e0a56d",
    "688011bd7c9da85b643e8b017b34a83186f7f94d599e65304c8b47416ed82bef",
    "3bd0157cadf88206f30f6208cc458b136ae8b62dee48dfcf84bc5ada44cae108",
]

# Final trace hash (hash of all block hashes concatenated)
# Reviewed explicit mask delivery; audit: regen_reports/scheduled_vm_20261030.txt
FROZEN_TRACE_HASH = "2b41d4ab4560def2d2c9815ed6b889dbf082e2dae9970e3281da45ad683d9ad6"


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
