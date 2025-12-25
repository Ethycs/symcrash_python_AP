"""
Tree Compression Pipeline
=========================

Complete pipeline for:
1. Tree data structures (authoring format)
2. Tree → β analysis and compression
3. β → schedule generation (runtime)
4. Wavelet-based tree compression (for complex trees)
5. Tree ADSR envelopes (temporal evolution)

This implements the "wavelets describing wavelets" architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np

from ..fixed import sat16, shift_round


# =============================================================================
# CONSTANTS
# =============================================================================

N_LEAVES = 32
N_LEVELS = 5
N_OPS = 31  # Total operations in schedule (2^5 - 1)


class Op(Enum):
    """Decomposition order for a tree node."""
    LOW_FIRST = 0   # Decompose low-frequency child first (bass emphasis)
    HIGH_FIRST = 1  # Decompose high-frequency child first (treble emphasis)


# =============================================================================
# TREE DATA STRUCTURES
# =============================================================================

@dataclass
class TreeNode:
    """One node in the wavelet packet tree."""
    id: int
    level: int              # 0 = root, 4 = one level above leaves
    position: int           # Position at this level (0 to 2^level - 1)
    freq_low: float = 0.0   # Hz (computed from position)
    freq_high: float = 0.0  # Hz
    is_leaf: bool = False
    children: Optional[Tuple[int, int]] = None  # (low_child_id, high_child_id)
    decomp_order: Op = Op.LOW_FIRST


@dataclass
class TreeStructure:
    """Complete tree definition - the authoring format."""
    name: str
    description: str = ""

    # The schedule: 31 operations defining decomposition order
    # schedule[i] = Op.LOW_FIRST or Op.HIGH_FIRST
    schedule: List[Op] = field(default_factory=list)

    # Metadata
    author: str = "unknown"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.schedule:
            # Default to balanced tree
            self.schedule = [Op.LOW_FIRST] * N_OPS


@dataclass
class TreeSchedule:
    """Runtime schedule format - minimal representation."""
    ops: List[bool]  # True = LOW_FIRST, False = HIGH_FIRST

    def __len__(self):
        return len(self.ops)

    def to_bits(self) -> int:
        """Pack schedule into 31-bit integer."""
        result = 0
        for i, op in enumerate(self.ops):
            if op:
                result |= (1 << i)
        return result

    @classmethod
    def from_bits(cls, bits: int) -> 'TreeSchedule':
        """Unpack from 31-bit integer."""
        ops = [(bits >> i) & 1 == 1 for i in range(N_OPS)]
        return cls(ops=ops)


# =============================================================================
# β → SCHEDULE GENERATOR
# =============================================================================

def generate_schedule_from_beta(beta: float) -> TreeSchedule:
    """
    Generate tree schedule from single β parameter.

    β = 0.0: Maximum treble resolution
    β = 0.5: Balanced (standard dyadic)
    β = 1.0: Maximum bass resolution

    Algorithm: At each level, threshold determines the cut point.
    All nodes below threshold get LOW_FIRST, above get HIGH_FIRST.
    """
    ops = []

    for level in range(N_LEVELS):
        nodes_at_level = 2 ** level
        threshold = beta * nodes_at_level

        for pos in range(nodes_at_level):
            if pos < threshold:
                ops.append(True)   # LOW_FIRST (bass emphasis)
            else:
                ops.append(False)  # HIGH_FIRST (treble emphasis)

    return TreeSchedule(ops=ops)


def op_to_level_pos(op_idx: int) -> Tuple[int, int]:
    """Convert operation index to (level, position)."""
    # Level 0: ops 0
    # Level 1: ops 1-2
    # Level 2: ops 3-6
    # Level 3: ops 7-14
    # Level 4: ops 15-30

    level = 0
    start = 0
    while start + (2 ** level) <= op_idx:
        start += 2 ** level
        level += 1

    pos = op_idx - start
    return level, pos


def level_pos_to_op(level: int, pos: int) -> int:
    """Convert (level, position) to operation index."""
    start = (2 ** level) - 1  # Sum of 2^0 + 2^1 + ... + 2^(level-1)
    return start + pos


# =============================================================================
# TREE ANALYZER (Tree → β)
# =============================================================================

@dataclass
class TreeAnalysis:
    """Results of analyzing a tree structure."""
    beta: float
    is_exact_fit: bool
    differing_ops: int
    weighted_error: float
    reconstruction: TreeSchedule


class TreeAnalyzer:
    """Analyze tree structure and extract parameters."""

    def analyze(self, tree: TreeStructure) -> TreeAnalysis:
        """Extract β from tree structure."""
        # Convert to schedule if needed
        schedule = self._tree_to_schedule(tree)

        # Find best-fit β
        best_beta, best_diff = self._fit_beta(schedule)

        # Check if exact
        reconstructed = generate_schedule_from_beta(best_beta)
        is_exact = best_diff == 0

        # Weighted error (root differences matter more)
        weighted = self._weighted_error(schedule, reconstructed)

        return TreeAnalysis(
            beta=best_beta,
            is_exact_fit=is_exact,
            differing_ops=best_diff,
            weighted_error=weighted,
            reconstruction=reconstructed
        )

    def _tree_to_schedule(self, tree: TreeStructure) -> TreeSchedule:
        """Convert TreeStructure to TreeSchedule."""
        ops = [op == Op.LOW_FIRST for op in tree.schedule]
        return TreeSchedule(ops=ops)

    def _fit_beta(self, schedule: TreeSchedule) -> Tuple[float, int]:
        """Find best-fit β by exhaustive search over quantized values."""
        best_beta = 0.5
        best_diff = N_OPS

        # Search all distinct β values (33 total for 32 leaves)
        for k in range(33):
            beta = k / 32.0
            generated = generate_schedule_from_beta(beta)
            diff = self._count_differences(schedule, generated)

            if diff < best_diff:
                best_diff = diff
                best_beta = beta

            if diff == 0:
                break  # Found exact match

        return best_beta, best_diff

    def _count_differences(self, a: TreeSchedule, b: TreeSchedule) -> int:
        """Count differing operations."""
        return sum(1 for x, y in zip(a.ops, b.ops) if x != y)

    def _weighted_error(self, a: TreeSchedule, b: TreeSchedule) -> float:
        """Weighted error (root-level differences count more)."""
        error = 0.0
        for i, (x, y) in enumerate(zip(a.ops, b.ops)):
            if x != y:
                level, _ = op_to_level_pos(i)
                error += 2 ** (N_LEVELS - 1 - level)  # Root = highest weight
        return error


# =============================================================================
# WAVELET TREE COMPRESSION
# =============================================================================

@dataclass
class WaveletTreeDescriptor:
    """Tree structure encoded as wavelet coefficients."""
    coeffs: np.ndarray  # Variable length, typically 4-8

    def to_threshold_curve(self, n_points: int = 32) -> np.ndarray:
        """Reconstruct threshold curve from coefficients."""
        # Simple basis: c0 = DC, c1 = linear, c2 = quadratic, etc.
        # This is a polynomial-like basis for simplicity

        x = np.linspace(0, 1, n_points)
        curve = np.zeros(n_points)

        for i, c in enumerate(self.coeffs):
            if i == 0:
                # DC component
                curve += c
            elif i == 1:
                # Linear tilt
                curve += c * (x - 0.5) * 2
            elif i == 2:
                # Quadratic (mid-scoop/boost)
                curve += c * (4 * (x - 0.5) ** 2 - 1)
            elif i == 3:
                # Cubic (asymmetry)
                curve += c * (x - 0.5) * (4 * (x - 0.5) ** 2 - 1) * 2
            else:
                # Higher order Chebyshev-like terms
                curve += c * np.cos(i * np.pi * x)

        return np.clip(curve, 0, 1)

    def to_schedule(self) -> TreeSchedule:
        """Generate tree schedule from descriptor."""
        # For simple beta (DC-only), use direct generation
        if len(self.coeffs) == 1:
            return generate_schedule_from_beta(self.coeffs[0])

        # For multi-coefficient descriptors, use curve-based generation
        curve = self.to_threshold_curve(32)
        ops = []

        for level in range(N_LEVELS):
            nodes = 2 ** level
            threshold = np.mean(curve) * nodes  # Use average as threshold

            for pos in range(nodes):
                # Use position relative to threshold
                ops.append(pos < threshold)

        return TreeSchedule(ops=ops)

    @classmethod
    def from_beta(cls, beta: float) -> 'WaveletTreeDescriptor':
        """Create descriptor from simple β (DC only)."""
        return cls(coeffs=np.array([beta]))

    @classmethod
    def from_extended(cls, beta: float, gamma: float, delta: float) -> 'WaveletTreeDescriptor':
        """Create descriptor from extended (β, γ, δ) parameters."""
        # β = DC bias
        # γ = mid-scoop (quadratic term)
        # δ = asymmetry (cubic term)
        c0 = beta
        c1 = 0.0  # No linear tilt in this parameterization
        c2 = (gamma - 0.5) * 2  # Map [0,1] to [-1,1]
        c3 = (delta - 0.5) * 2
        return cls(coeffs=np.array([c0, c1, c2, c3]))


def compress_tree_wavelet(schedule: TreeSchedule, max_coeffs: int = 8) -> WaveletTreeDescriptor:
    """Compress tree schedule to wavelet descriptor."""
    # Extract implicit threshold curve from schedule
    curve = schedule_to_curve(schedule)

    # Fit coefficients to curve
    coeffs = fit_curve_coefficients(curve, max_coeffs)

    return WaveletTreeDescriptor(coeffs=coeffs)


def schedule_to_curve(schedule: TreeSchedule) -> np.ndarray:
    """Extract threshold curve implied by schedule."""
    # Simple approach: for a beta-generated schedule, the curve is just beta
    # Count LOW_FIRST ops at each level to estimate effective threshold

    curve = np.zeros(32)

    # For each leaf position, determine the effective bias
    for leaf_pos in range(32):
        # The bias at this position depends on which ops affect it
        local_bias = 0.0
        count = 0

        for level in range(N_LEVELS):
            nodes_at_level = 2 ** level
            # Which node at this level covers this leaf?
            node_pos = (leaf_pos * nodes_at_level) // 32
            op_idx = level_pos_to_op(level, node_pos)

            if op_idx < len(schedule.ops):
                if schedule.ops[op_idx]:
                    local_bias += 1.0
                count += 1

        curve[leaf_pos] = local_bias / count if count > 0 else 0.5

    return curve


def fit_curve_coefficients(curve: np.ndarray, max_coeffs: int) -> np.ndarray:
    """Fit polynomial-like coefficients to threshold curve."""
    n = len(curve)
    x = np.linspace(0, 1, n)

    # Build basis matrix
    basis = np.zeros((n, max_coeffs))
    for i in range(max_coeffs):
        if i == 0:
            basis[:, i] = 1.0
        elif i == 1:
            basis[:, i] = (x - 0.5) * 2
        elif i == 2:
            basis[:, i] = 4 * (x - 0.5) ** 2 - 1
        elif i == 3:
            basis[:, i] = (x - 0.5) * (4 * (x - 0.5) ** 2 - 1) * 2
        else:
            basis[:, i] = np.cos(i * np.pi * x)

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(basis, curve, rcond=None)

    return coeffs


def reconstruction_error(original: TreeSchedule, descriptor: WaveletTreeDescriptor) -> Dict:
    """Measure reconstruction quality."""
    reconstructed = descriptor.to_schedule()

    diff_ops = sum(1 for a, b in zip(original.ops, reconstructed.ops) if a != b)

    # Weighted error
    weighted = 0.0
    for i, (a, b) in enumerate(zip(original.ops, reconstructed.ops)):
        if a != b:
            level, _ = op_to_level_pos(i)
            weighted += 2 ** (N_LEVELS - 1 - level)

    return {
        'differing_ops': diff_ops,
        'weighted_error': weighted,
        'is_exact': diff_ops == 0,
        'num_coeffs': len(descriptor.coeffs)
    }


# =============================================================================
# TREE ADSR ENVELOPES
# =============================================================================

@dataclass
class TreeEnvelope:
    """ADSR envelope for a single tree coefficient."""
    attack_level: float = 0.5
    attack_time: float = 0.01   # Seconds
    decay_time: float = 0.1
    sustain_level: float = 0.5
    release_time: float = 0.3

    # Curve shapes: -1 = log, 0 = linear, 1 = exp
    attack_curve: float = 0.0
    decay_curve: float = 0.0
    release_curve: float = 0.0


@dataclass
class TreeADSR:
    """Complete tree evolution specification."""

    # Coefficient targets at each stage
    attack_coeffs: np.ndarray   # Shape at attack peak
    sustain_coeffs: np.ndarray  # Shape during sustain
    release_coeffs: np.ndarray  # Shape at release end

    # Timing
    attack_time: float = 0.01   # Seconds
    decay_time: float = 0.1
    release_time: float = 0.3

    # Curves
    attack_curve: float = 0.0
    decay_curve: float = 0.0
    release_curve: float = 0.0

    def get_coeffs(self, t: float, note_on: bool, note_off_time: Optional[float] = None,
                   sample_rate: float = 48000.0) -> np.ndarray:
        """Get interpolated coefficients at time t."""

        if note_off_time is not None and t >= note_off_time:
            # Release phase
            release_progress = (t - note_off_time) / max(self.release_time, 0.001)
            release_progress = min(1.0, release_progress)
            release_progress = self._apply_curve(release_progress, self.release_curve)

            # Interpolate from sustain to release
            return self.sustain_coeffs * (1 - release_progress) + self.release_coeffs * release_progress

        elif t < self.attack_time:
            # Attack phase
            attack_progress = t / max(self.attack_time, 0.001)
            attack_progress = self._apply_curve(attack_progress, self.attack_curve)

            # Interpolate from zero to attack
            return self.attack_coeffs * attack_progress

        elif t < self.attack_time + self.decay_time:
            # Decay phase
            decay_progress = (t - self.attack_time) / max(self.decay_time, 0.001)
            decay_progress = self._apply_curve(decay_progress, self.decay_curve)

            # Interpolate from attack to sustain
            return self.attack_coeffs * (1 - decay_progress) + self.sustain_coeffs * decay_progress

        else:
            # Sustain phase
            return self.sustain_coeffs.copy()

    def _apply_curve(self, t: float, curve: float) -> float:
        """Apply curve shaping to linear 0-1 progress."""
        if curve == 0:
            return t
        elif curve > 0:
            # Exponential (slow start, fast end)
            return t ** (1 + curve)
        else:
            # Logarithmic (fast start, slow end)
            return 1 - (1 - t) ** (1 - curve)

    def to_descriptor_at(self, t: float, note_on: bool = True,
                         note_off_time: Optional[float] = None) -> WaveletTreeDescriptor:
        """Get tree descriptor at time t."""
        coeffs = self.get_coeffs(t, note_on, note_off_time)
        return WaveletTreeDescriptor(coeffs=coeffs)


# =============================================================================
# PRESET FORMAT
# =============================================================================

@dataclass
class TreePreset:
    """Runtime preset format for trees."""
    name: str = ""
    category: str = ""

    # Tree specification (one of these)
    mode: str = "beta"  # "beta" | "extended" | "wavelet" | "explicit"

    # Mode: beta (2 bytes)
    beta: Optional[float] = None

    # Mode: extended (6 bytes)
    gamma: Optional[float] = None
    delta: Optional[float] = None

    # Mode: wavelet (variable)
    wavelet_coeffs: Optional[np.ndarray] = None

    # Mode: explicit (62 bytes)
    explicit_schedule: Optional[TreeSchedule] = None

    # ADSR (optional)
    adsr: Optional[TreeADSR] = None

    def to_descriptor(self) -> WaveletTreeDescriptor:
        """Convert preset to wavelet descriptor."""
        if self.mode == "beta":
            return WaveletTreeDescriptor.from_beta(self.beta or 0.5)
        elif self.mode == "extended":
            return WaveletTreeDescriptor.from_extended(
                self.beta or 0.5,
                self.gamma or 0.5,
                self.delta or 0.5
            )
        elif self.mode == "wavelet":
            return WaveletTreeDescriptor(coeffs=self.wavelet_coeffs)
        elif self.mode == "explicit":
            return compress_tree_wavelet(self.explicit_schedule, max_coeffs=32)
        else:
            return WaveletTreeDescriptor.from_beta(0.5)

    def to_schedule(self) -> TreeSchedule:
        """Get runtime schedule."""
        if self.mode == "explicit" and self.explicit_schedule:
            return self.explicit_schedule
        else:
            return self.to_descriptor().to_schedule()


# =============================================================================
# LEGACY TREE BUILDERS
# =============================================================================

def build_balanced_tree() -> TreeStructure:
    """Tree 0: Balanced (β = 0.5)."""
    schedule = generate_schedule_from_beta(0.5)
    return TreeStructure(
        name="Balanced",
        description="Uniform frequency resolution",
        schedule=[Op.LOW_FIRST if op else Op.HIGH_FIRST for op in schedule.ops],
        tags=["balanced", "neutral"]
    )


def build_bass_heavy_tree() -> TreeStructure:
    """Tree 1: Bass-heavy (β = 0.75)."""
    schedule = generate_schedule_from_beta(0.75)
    return TreeStructure(
        name="Bass Heavy",
        description="More resolution in low frequencies",
        schedule=[Op.LOW_FIRST if op else Op.HIGH_FIRST for op in schedule.ops],
        tags=["bass", "sub", "electronic"]
    )


def build_treble_heavy_tree() -> TreeStructure:
    """Tree 2: Treble-heavy (β = 0.25)."""
    schedule = generate_schedule_from_beta(0.25)
    return TreeStructure(
        name="Treble Heavy",
        description="More resolution in high frequencies",
        schedule=[Op.LOW_FIRST if op else Op.HIGH_FIRST for op in schedule.ops],
        tags=["treble", "bright", "percussion"]
    )


# Tests moved to pipeline_tests/test_tree_compression.py
