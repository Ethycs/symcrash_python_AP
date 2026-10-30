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
FROZEN_BLOCK_HASHES: List[str] = [
    "5856ebdb6ce150e0e2fe5daff34715f29927de1386e465b455e453a69f66a604",
    "c267cdea2eef2c2053cb4b4ccebbe6a687caa48889484b7b633f1c65fa4a2c25",
    "37852815a4420e72a22ec7d529bfd06b351ea304aaeb65a9fc75e12a03077d64",
    "613ba426bc6729a7c8683ca4d29c88b8348d5e7290eb4b182a9c35cac97c72e0",
    "17e4b2ebcc1fc99197e8179bcf65f02a742c1963565090aa9a01be6e7a1b8f7c",
    "1888a62cb16dae9204ea0a6af4652797b50a1c2eab4c8db1f166ec98a4f4c1e3",
    "ddd390390afa6d9918745e46d60bc3d251cdee10117d8b8d686775ca8d40dafd",
    "b00e38aa3ea45b61daeedbaecb155823a8381ee1714d3be419f157a6249fa2f1",
    "16b8e3638fd0924d41d578a2df0dad110366240e71cbca8e08385ea1a2cc14dc",
    "6b7957f1351e134120ba9d60ab97a12e7759f67b89f7727cf70190abeaef3487",
    "9e117be16b95901e3a93c503bda6f554bc8965cedeb888c9149b258ca35d862e",
    "7941dd3acf8c99dfbd8ed1e10c47068fec974f35a18b6b9cf27f79216e95d870",
    "5d4fcfa85113d816a0f76b98fc3370eba3984613a0ffedfc3be7f93bba0136f4",
    "4a991434d3c9bf044314dc254aec686300f5a33318693270d23dfd251b15a8bd",
    "56675c5e5c338b44a969d1e437b305590cb7f96d225cc37686f5e51ac53b1f00",
    "42799afd8c4428585b12c7fa63879223264457d9721a0f26c128528d8271e039",
    "615007be9238b3632f93243c3d2d3f9c28de398bf849bf6e40658a2616bc182b",
    "654aa561f0ba067187bc63074379ba2d737b3bf3f1f1cccc49547c8dc16fd7df",
    "25ab9e63c782802179ec22f3cd78cc2c2658b995941b0642b83d173d98564eb3",
    "84bd1470e6c6f47dbd807057047553be1740996f03bad318739efc3d0df6c214",
    "388ecd6b118616142db50573e16721fcf55ae4dc1c9115550271c2306b97f1dc",
    "36d42cabea65075eaca2aa3a44969927ed1afda88f10f403b7f38093b5dfa3ba",
    "366162a2dcdadc0419acbe017d86f6ade0c10561a2dcc3d845a8a170345ace98",
    "4500a251d36f480736e18488edad592c74f8f97ab5000b93c98ed2ba064c153f",
    "65a929a0737d8dac0b931e8e043a2f05a1ea2d6a58eff9a72b13ae4686d95862",
    "3a857a5affb9c41ed7238f850ca75e1f64589a690c3a1213cd9eb4bb7584e375",
    "d56d84b384a1c4c324d94efeda2971411edb9deae08b199ef5fa4d6841c6393d",
    "4f03537523fd8bcbbb0ee1b0850431b02a35199d484f14568cc9f32786b2a6a9",
    "172ecb113ba1b03321fb29d1e4bc40b86ab151f957f76df1ed6699ee5ab7276f",
    "d5571c6807add5a71fc084c54d3d16ae309c61d45306444124fb5677dbff6e91",
    "bb6e568ee81bec43732fe55ce8f3b0c6d81a626817978365daf7aad0d1ed1981",
    "2ab3ce99c3c7af087e92850d57d948d02014b3f3f8f4c6be05345d89b3ce279a",
    "16aaf25c73011911f562bb4cd235830b9da3d2c0f13130a746d8a2ae9bf89fa2",
    "0321eb1ba8cd41483d342325ac4bce337ad89cf7596a67afc7a5206c6e5a5e29",
    "20d331456d82c6c16371297920a9de1f82d8837b5907c4325804282456d4baf5",
    "8ce946fc933bfc2b59063ae54460a060e35e324249d587231f1bb7476aad1955",
    "930d435547df8d5330e92b9daca57df3b82cc5f7a3bcbef1652b7a2e9b135b53",
    "87b7e268799156061e5f9272b455b66d8dee06658f2c1101c872baceb8bde86e",
    "cea54429bee508058291919610f6dcfc643bcfa92200cdacc842adf75dd990ff",
    "8c1e6586225db938e5c260d14c977124b69aac5e1a2f24f45449232f6dbf3476",
    "41c847b7d245b91e349fbfff01c024d5fc7f3d114d06ce28513ef50b9f990f82",
    "f38decfc748a25144fa371bc5ad4da3ab37dcb4ce8255001425b3f9dae86709a",
    "2fcbf5ce967f0fddc0fa1d6069f418965458f612c8d879900b806863b090bbd5",
    "7bb0bb2cd19886a7f39cb2e579c2453c3fb415ef2aa9564464da1caf598209f3",
    "a64fc66d0f8ec64dcb1246ac5041931814b3eef7a14abf4cf59dd2d5c1d427ca",
    "be6021f8efcb2db74b1cf9acd8b62cc37729ce6e4584d17b90bd5d9be69e1f9f",
    "5879368da67c4514e0d8aa32e5775e6aa9139794df8c6a92d09e299044bb9e86",
    "47b1d16324f0ff145973e38e7652886a459ce1c15afe778edd77fd4b8657cf53",
    "57ce42f2c20d523457aa582872b4084f1679d66aee2236b45c0141c08fc64c1c",
    "bfeb0d2c41fa2a3449bcb5fb579610c0c59c865567ceae091fe2c557f8dff2d5",
    "b9888710db3680a4afa582179d9adab49f3d47c986cb385f732e927109587d9b",
    "7ff233b04e5a23644f366d9591a14d5d30b5b6d8b4f3ca056c9838d089d4f52d",
    "afaad99629cc4309f9f5149eef86d00081a84865425b793bd8dbc5e7b636f29f",
    "cf5e79b177dc0aeafd9a94504a159bd239f70f4adc700fe64a250954f547bcdc",
    "94f0aad54c41aa48ab450ebf31c758b87d12253ad8103142dd4fd7ac32edfa70",
    "9adfac73909a652b85c595f07668c6ec683f3d6e5a5175fa9eefba21144ba4b4",
    "fff71d77007758be33e1244d3f42e1f1739b37674afdb0298c0b8e073c107e2b",
    "582100405b7a805f0cf3d27f8db17b423cf9cfd5dbbc652663f514959c70cbf1",
    "1916be1a0f59d33c3c8b16eceb90d28cee44064f0091c105e07dbc9f12d63e5f",
    "0317c13f79a6bca172590350f0e65efbf8b7e161e1ffa6dfb73420baf3267a59",
    "c89a05ff1380503bd599af8f216af06bb746a3aa6406a974d9fdf81ca9744e06",
    "a8f9b0bfc91e1161d94ed0d23830500cb835d10713d70ef894da867e33171cca",
    "313b9f543465c714567ab754f900966db6d2c0f2547c741545625bdf757c1148",
    "854a49d0de3dfdbd5b5846c6d32a7777a0c1c9ba6bfa16ac0d545dbfb8a4a642",
]

# Final trace hash (hash of all block hashes concatenated)
# Reviewed explicit mask delivery; audit: regen_reports/scheduled_vm_20261030.txt
FROZEN_TRACE_HASH = "a313b041a6b4d2c05bdfc31cb5b2c90bc2f7270396c0b557bf9a61d52bb937a6"


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
