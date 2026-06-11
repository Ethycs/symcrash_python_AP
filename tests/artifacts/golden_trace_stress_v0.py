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

# SHA-256 hash of concatenated block outputs
# IMPORTANT: Update these ONLY when intentional behavior change occurs
# Updated 2026-01-02: Phase-coherent wet path (mask smoothing + C⁰ seam stitching, no windowing)
# Updated 2026-01-03: Disable DC removal for VM path (CCS is phase-continuous with zero DC)
# Updated 2026-01-05: Skip SheafProject in CCS mode (was corrupting polyphase coefficients)
FROZEN_BLOCK_HASHES: List[str] = [
    "8404dbeb1822c768c1aaef45bedbe71115e52ebaddbad185dbac0e7efa89c74f",
    "3ca9feef7ee57c0e132a22c4a2ce099b75ca5d12763b5c1e3d2b4b2516f9a9ad",
    "fea35ead5ba48b95ae9444695e7e84f2d2fa770261b6a772cf62f40cc9bb905d",
    "913fa8156b71080c8e4ced7272dcddce2b0cabee5ead8c381cfadd58a1a882de",
    "3ae100a8395a0f7c8444111bda269b9aec704267afed7b324522e8aae6ba6c56",
    "cf6cb0c27c875db62c1a8ddd7f9322495692349746f5eb2f80a286bf7d338342",
    "83811c91a459a7c4aa4b4f67b011b83166b652c0e1bd16faf33e5fa63b9c6b1f",
    "f38fd993e6097490d6a07f1138f5e1a87d3399df8e2a0b4c35e6e742fd3b0caa",
    "9fcccec472bbd2eca0f8a91f1e704a2b40f1206342c7e7985abe41d591aff5df",
    "1d7544abc3bb23e1536abf6869c4596eebbb2d33f1c7e8a6c9830cdae36e4f1a",
    "cff393d33e60b08c24af2bdd1ec452c1884c6c05112bccbd118583e51146fab2",
    "59035d7063689f376c044d82ee0c8b5ec1e5e6c7df02b89b6676e3f20a18635a",
    "517be88e1eea39eaf46348e06c02637a547fa084415ec92a4488703b13334377",
    "263393e10438e95ce3d92c150ee424316caaf8dcee56e6621043b09eb7290164",
    "3832fdf7d930d034c0ea459529c8a61ef36128b1b27db5893aa7a20587416cd1",
    "09f9c1386963121b5b26625100ad7acdd8d48e90edae040929a17fcce085ca7d",
    "e4e1ace24b8eb77c59f31f843c5c8a2298d6a3936be3465156808f3898e13312",
    "8b7da14eda204bba72cdde80e55b638c1b949930323e0f9b79bb6d1c46d53721",
    "1506d8420ed5dd77797358ee339f5c2bd27432ae3b98bdd1adf987fde8315467",
    "fa018d08bddec3152465b77f95d91d03f71c488f9721293a92c8cac812ca96fd",
    "3b7924b5ec9bcf76db18a0c4187bdff52ca56b49de30cbb0f5cf918f4b71cf90",
    "3c0678e64e4c4242ee18f58ed9e48e60ece8d459773ca9ba0b8630a966344ba6",
    "fc7653fc1b8658593e64ef49c11c6518a968a5a9f168057cd74f2328e597fbea",
    "cfd6df865ccd483293ead0b95a0f3004838896399bcb5ddc641ef458568523e9",
    "e1ef35545c99c579e787031987fcdb03ca2dd44e7fc4b52d0f1dc75d9b1e5857",
    "3f41d733b0ef82e275fcc843b651dc0e99636213c99ea18ac59ebc47e44b3171",
    "b146919f08c01994eb3be1a857b5aba530d09e1e5223000c4511bb115d33176f",
    "12a609d37f71a0aa1c76098a16801daab683fd8be0cf72f47b50953b2a1aa4f3",
    "029452cabd4e99a2c965a6c2ca0040df319ec0f15990c4b8a6341a1fdd7c79d9",
    "70de569e007d28123c24fe8d295316bee588589fd51c45a5861ab7da88039d82",
    "d7efb4b24566acfd7bcea94d9ce3c0f940770cceaf24ab4181962af73c7aff41",
    "34b91961d85ce7e0d98b48a70bc3f1074695c7acba525fe041d8e7ba3224a284",
    "cd037c546c74b2e06dc1be9ddcb36ee9ddb32c11ba431d3d9cdfa10a9c3ae431",
    "9dea57cb6490aedffbd197161fc5daacd2617ccce2a4d4f756de513b2b5f4406",
    "6623f4ef5855baf85ced976848d0fe61a0153db715e99843b13f8c8c67803ef6",
    "b0ec1f14a4429df325a413e1c81ad6c06c0f5124434450b7bb0ebc77cbd9280b",
    "9ea6f6cfc2e9f69bb9a2c58653d07bf23decab679fe9f1374ad5660a92446e6c",
    "1bb53e865b80dd536d35fc2475ac17e79d7ddc6f66ce6deeacfc98775644e59f",
    "24ccc362bdc49f9984b04dae495e4a451428b12749d0a5e02ad489b4b934e5e7",
    "d9113b757635493b45dc00ce196615d19284b1a5db09f5f60e441409d4f17382",
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
FROZEN_TRACE_HASH = "c0528cbd5b1c0932a8a3a44f004262e3548f489acea6e22266ac324af11ad455"


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
