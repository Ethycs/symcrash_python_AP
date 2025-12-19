"""
Equipment Harmonic Basis
========================

Maps wavelet leaves to characteristic frequencies found in analog
audio equipment: powerline harmonics, transformer saturation,
tube harmonics, tape artifacts, motor noise, etc.

This turns coefficient algebra into gear modeling:
- DIAG operators -> per-artifact gain (add/remove hum, saturation)
- PERMUTE operators -> swap artifact signatures between gear types
- LOWRANK operators -> "gear character" templates
- BANDED operators -> harmonic coupling (like transformer saturation)

Basis organized by artifact source, not musical pitch.
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from fixed import sat16, mul_q15_sat, Q15_ONE


# =============================================================================
# EQUIPMENT FREQUENCY CONSTANTS
# =============================================================================

# Power line fundamentals (region-dependent)
MAINS_60HZ = 60.0    # North America, Japan (east)
MAINS_50HZ = 50.0    # Europe, UK, Australia, Japan (west)

# Common sample rate
SAMPLE_RATE = 48000.0
NYQUIST = SAMPLE_RATE / 2


@dataclass
class EquipmentBasis:
    """Definition of an equipment-based frequency mapping."""
    name: str
    description: str
    frequencies: List[float]  # Hz for each of 32 leaves
    categories: List[str]     # Category label per leaf


# =============================================================================
# 60 Hz WORLD BASIS (North American/Japanese gear)
# =============================================================================

def build_60hz_basis() -> EquipmentBasis:
    """
    Equipment basis for 60 Hz mains regions.

    Leaf allocation:
      0-7:   Power supply harmonics (60, 120, 180, 240, 300, 360, 420, 480 Hz)
      8-11:  Transformer saturation (odd harmonics: 180, 300, 420, 540 Hz)
      12-15: Tube even harmonics (2nd, 4th, 6th, 8th of ~200 Hz reference)
      16-19: Tape head bump / bias artifacts
      20-23: Motor/mechanical (wow, flutter, capstan)
      24-27: "Character" regions (warmth, presence, air, sizzle)
      28-31: High frequency artifacts (CRT whine, switching noise, clock bleed)
    """
    freqs = []
    cats = []

    # 0-7: Power supply harmonics (60 Hz and overtones)
    # Full-wave rectifier doubles frequency, saturation adds harmonics
    for n in range(1, 9):
        freqs.append(60.0 * n)
        cats.append("power")

    # 8-11: Transformer saturation (primarily odd harmonics)
    # Magnetic core saturation creates odd harmonics of mains
    for n in [3, 5, 7, 9]:
        freqs.append(60.0 * n)
        cats.append("transformer")

    # 12-15: Tube amplifier harmonics
    # Referenced to ~200 Hz (typical guitar/vocal fundamental)
    # Triodes emphasize even, pentodes emphasize odd
    tube_ref = 200.0
    for n in [2, 3, 4, 5]:  # 2nd through 5th harmonics
        freqs.append(tube_ref * n)
        cats.append("tube")

    # 16-19: Tape artifacts
    # Head bump (low freq boost), bias bleed, high freq loss
    freqs.extend([
        80.0,    # Head bump low
        120.0,   # Head bump peak
        3150.0,  # NAB/IEC crossover region
        10000.0, # High frequency rolloff onset
    ])
    cats.extend(["tape"] * 4)

    # 20-23: Mechanical/motor artifacts
    freqs.extend([
        0.5,     # Wow (slow pitch variation)
        4.0,     # Flutter low
        10.0,    # Flutter high
        29.97,   # NTSC frame rate (video gear)
    ])
    cats.extend(["motor"] * 4)

    # 24-27: Tonal "character" regions
    freqs.extend([
        250.0,   # Warmth/body
        800.0,   # Midrange honk
        3000.0,  # Presence/bite
        12000.0, # Air/shimmer
    ])
    cats.extend(["character"] * 4)

    # 28-31: High frequency artifacts
    freqs.extend([
        15734.0,  # NTSC horizontal sync (CRT whine)
        16000.0,  # Switching power supply
        19000.0,  # Stereo pilot tone
        22050.0,  # Digital aliasing edge (44.1k/2)
    ])
    cats.extend(["hf_artifact"] * 4)

    return EquipmentBasis(
        name="60hz_equipment",
        description="North American/Japanese 60Hz mains equipment harmonics",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# 50 Hz WORLD BASIS (European/UK gear)
# =============================================================================

def build_50hz_basis() -> EquipmentBasis:
    """
    Equipment basis for 50 Hz mains regions.
    Same structure as 60 Hz but different fundamentals.
    """
    freqs = []
    cats = []

    # Power harmonics (50 Hz)
    for n in range(1, 9):
        freqs.append(50.0 * n)
        cats.append("power")

    # Transformer saturation
    for n in [3, 5, 7, 9]:
        freqs.append(50.0 * n)
        cats.append("transformer")

    # Tube harmonics
    tube_ref = 200.0
    for n in [2, 3, 4, 5]:
        freqs.append(tube_ref * n)
        cats.append("tube")

    # Tape artifacts
    freqs.extend([70.0, 100.0, 3150.0, 10000.0])
    cats.extend(["tape"] * 4)

    # Mechanical
    freqs.extend([0.5, 4.0, 10.0, 25.0])  # 25 Hz = PAL frame rate
    cats.extend(["motor"] * 4)

    # Character regions
    freqs.extend([250.0, 800.0, 3000.0, 12000.0])
    cats.extend(["character"] * 4)

    # HF artifacts
    freqs.extend([
        15625.0,  # PAL horizontal sync
        16000.0,
        19000.0,
        22050.0,
    ])
    cats.extend(["hf_artifact"] * 4)

    return EquipmentBasis(
        name="50hz_equipment",
        description="European/UK 50Hz mains equipment harmonics",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# TUBE AMPLIFIER BASIS
# =============================================================================

def build_tube_basis(fundamental: float = 110.0) -> EquipmentBasis:
    """
    Basis focused on tube amplifier harmonic series.

    Maps leaves to harmonic overtones of a reference fundamental,
    with different regions for triode vs pentode character.

    Leaf allocation:
      0-15:  Harmonics 1-16 of fundamental
      16-23: Intermodulation products (sum/difference tones)
      24-27: Power supply modulation frequencies
      28-31: Output transformer resonances
    """
    freqs = []
    cats = []

    # 0-15: Harmonic series
    for n in range(1, 17):
        freqs.append(fundamental * n)
        cats.append("harmonic")

    # 16-23: Intermodulation products
    # Common IM products from two-tone test (fundamental + fifth)
    f1 = fundamental
    f2 = fundamental * 1.5  # Perfect fifth

    im_products = [
        f2 - f1,           # Difference tone
        2*f1 - f2,         # 2nd order IM
        2*f2 - f1,         # 2nd order IM
        f1 + f2,           # Sum tone
        3*f1 - 2*f2,       # 3rd order IM
        3*f2 - 2*f1,       # 3rd order IM
        2*f1 + f2,         # Sum
        f1 + 2*f2,         # Sum
    ]
    freqs.extend(im_products)
    cats.extend(["intermod"] * 8)

    # 24-27: Power supply modulation
    # B+ ripple modulates signal at mains harmonics
    freqs.extend([60.0, 120.0, 180.0, 240.0])
    cats.extend(["supply"] * 4)

    # 28-31: Output transformer resonances
    # Typical transformer resonances and rolloffs
    freqs.extend([40.0, 80.0, 8000.0, 15000.0])
    cats.extend(["transformer"] * 4)

    return EquipmentBasis(
        name="tube_amp",
        description=f"Tube amplifier harmonics (fundamental={fundamental}Hz)",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# TAPE MACHINE BASIS
# =============================================================================

def build_tape_basis(speed_ips: float = 15.0) -> EquipmentBasis:
    """
    Basis for tape machine artifacts.

    Speed affects head bump frequency and HF response.
    Common speeds: 7.5, 15, 30 ips

    Leaf allocation:
      0-7:   Head bump frequencies (speed-dependent)
      8-15:  Bias/erase oscillator harmonics
      16-23: Mechanical frequencies (capstan, reel motors)
      24-27: Saturation harmonics
      28-31: Noise spectrum peaks
    """
    freqs = []
    cats = []

    # Head bump frequency scales with speed
    # Approximately: bump_freq = 20 * sqrt(speed)
    bump_center = 20.0 * np.sqrt(speed_ips)

    # 0-7: Head bump region
    bump_freqs = [
        bump_center * 0.5,
        bump_center * 0.75,
        bump_center,
        bump_center * 1.25,
        bump_center * 1.5,
        bump_center * 2.0,
        bump_center * 3.0,
        bump_center * 4.0,
    ]
    freqs.extend(bump_freqs)
    cats.extend(["head_bump"] * 8)

    # 8-15: Bias oscillator harmonics
    # Typical bias frequencies: 80-150 kHz
    # We care about audible subharmonics and beating products
    bias_fundamental = 100000.0  # 100 kHz typical

    # Subharmonics that beat into audio range
    bias_audible = [
        bias_fundamental % SAMPLE_RATE,  # Aliased fundamental
        (bias_fundamental * 2) % SAMPLE_RATE,
        100.0,   # Low frequency bias bleed
        200.0,
        1000.0,  # Mid bias artifact
        5000.0,
        10000.0,
        15000.0, # HF response limit
    ]
    freqs.extend(bias_audible)
    cats.extend(["bias"] * 8)

    # 16-23: Mechanical frequencies
    # Capstan motor, supply reel, takeup reel
    capstan_rpm = speed_ips * 60 / (np.pi * 0.25)  # Assuming 0.25" capstan
    capstan_hz = capstan_rpm / 60

    mech_freqs = [
        0.3,           # Wow (slow)
        0.7,           # Wow (fast)
        capstan_hz,    # Capstan fundamental
        capstan_hz * 2,
        6.0,           # Flutter low
        10.0,          # Flutter mid
        14.0,          # Flutter high
        capstan_hz * 3,
    ]
    freqs.extend(mech_freqs)
    cats.extend(["mechanical"] * 8)

    # 24-27: Saturation harmonics
    # Tape saturation creates primarily odd harmonics
    sat_ref = 1000.0  # Reference tone
    freqs.extend([sat_ref * 3, sat_ref * 5, sat_ref * 7, sat_ref * 9])
    cats.extend(["saturation"] * 4)

    # 28-31: Noise spectrum
    # Tape hiss has characteristic spectrum
    freqs.extend([2000.0, 4000.0, 8000.0, 14000.0])
    cats.extend(["noise"] * 4)

    return EquipmentBasis(
        name=f"tape_{speed_ips}ips",
        description=f"Tape machine at {speed_ips} ips",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# VINYL/PHONO BASIS
# =============================================================================

def build_vinyl_basis(rpm: float = 33.33) -> EquipmentBasis:
    """
    Basis for vinyl playback artifacts.

    Leaf allocation:
      0-7:   Rumble frequencies (turntable motor, bearing)
      8-15:  RIAA curve pivot points
      16-23: Stylus/cartridge resonances
      24-27: Surface noise spectrum
      28-31: Inner groove distortion frequencies
    """
    freqs = []
    cats = []

    # Rotation frequency
    rot_hz = rpm / 60.0  # ~0.556 Hz for 33 RPM

    # 0-7: Rumble (motor, bearing, acoustic feedback)
    rumble = [
        rot_hz,           # Once per revolution
        rot_hz * 2,       # Twice per revolution
        rot_hz * 3,
        8.0,              # Tonearm resonance low
        12.0,             # Tonearm resonance mid
        20.0,             # Subsonic content
        30.0,
        40.0,             # Rumble filter cutoff region
    ]
    freqs.extend(rumble)
    cats.extend(["rumble"] * 8)

    # 8-15: RIAA equalization pivots
    riaa_points = [
        50.0,    # Bass shelf onset
        500.0,   # RIAA turnover
        1000.0,  # Reference level
        2122.0,  # RIAA high shelf
        3000.0,
        5000.0,
        7500.0,
        10000.0,
    ]
    freqs.extend(riaa_points)
    cats.extend(["riaa"] * 8)

    # 16-23: Cartridge/stylus resonances
    cart_resonances = [
        15.0,     # Tonearm/cartridge compliance resonance
        800.0,    # Cantilever resonance
        2000.0,
        5000.0,
        8000.0,   # Stylus tip mass resonance
        12000.0,
        16000.0,  # HF limit (stylus geometry)
        20000.0,
    ]
    freqs.extend(cart_resonances)
    cats.extend(["cartridge"] * 8)

    # 24-27: Surface noise spectrum
    freqs.extend([1000.0, 3000.0, 6000.0, 10000.0])
    cats.extend(["surface_noise"] * 4)

    # 28-31: Inner groove distortion (tracking error increases)
    # Higher harmonics become more prominent toward label
    freqs.extend([3000.0, 5000.0, 7000.0, 9000.0])
    cats.extend(["tracking"] * 4)

    return EquipmentBasis(
        name=f"vinyl_{rpm}rpm",
        description=f"Vinyl playback at {rpm} RPM",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# DIGITAL ARTIFACTS BASIS
# =============================================================================

def build_digital_basis(sample_rate: float = 44100.0) -> EquipmentBasis:
    """
    Basis for digital audio artifacts.

    Leaf allocation:
      0-7:   Aliasing frequencies (near Nyquist)
      8-15:  Jitter/clock spectrum
      16-23: Quantization noise spectrum
      24-27: DAC reconstruction filter artifacts
      28-31: Power supply noise (switching)
    """
    nyq = sample_rate / 2

    freqs = []
    cats = []

    # 0-7: Aliasing products (frequencies that fold back)
    # When signal > Nyquist, it aliases to Nyquist - (signal - Nyquist)
    alias_freqs = [
        nyq - 1000,
        nyq - 500,
        nyq - 100,
        nyq - 50,
        nyq,
        nyq * 0.9,
        nyq * 0.8,
        nyq * 0.7,
    ]
    freqs.extend(alias_freqs)
    cats.extend(["aliasing"] * 8)

    # 8-15: Jitter spectrum
    # Clock jitter creates sidebands around signal
    jitter_freqs = [
        100.0,    # Low freq jitter (power supply)
        1000.0,   # Mid jitter
        sample_rate / 1000,  # ~44 Hz (buffer boundaries)
        sample_rate / 256,   # ~172 Hz (block boundaries)
        sample_rate / 64,    # ~689 Hz
        sample_rate / 16,    # ~2756 Hz
        sample_rate / 4,     # ~11 kHz
        nyq * 0.95,
    ]
    freqs.extend(jitter_freqs)
    cats.extend(["jitter"] * 8)

    # 16-23: Quantization noise (shaped by dither/noise shaping)
    quant_freqs = [
        200.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        12000.0,
        16000.0,
    ]
    freqs.extend(quant_freqs)
    cats.extend(["quantization"] * 8)

    # 24-27: Reconstruction filter
    freqs.extend([nyq * 0.8, nyq * 0.9, nyq * 0.95, nyq * 0.99])
    cats.extend(["reconstruction"] * 4)

    # 28-31: Switching power supply noise
    # Typical SMPS frequencies: 50-500 kHz, but harmonics in audio
    freqs.extend([100.0, 200.0, 400.0, 800.0])
    cats.extend(["switching"] * 4)

    return EquipmentBasis(
        name=f"digital_{int(sample_rate)}",
        description=f"Digital artifacts at {sample_rate}Hz",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# COMBINED "STUDIO CHAIN" BASIS
# =============================================================================

def build_studio_chain_basis() -> EquipmentBasis:
    """
    Combined basis representing a typical analog studio signal chain:
    Mic preamp -> EQ -> Compressor -> Tape -> Mix bus -> Mastering

    Most useful frequencies from each stage.
    """
    freqs = []
    cats = []

    # Mic preamp / console (transformer + tube/transistor)
    freqs.extend([
        60.0,    # Mains hum
        120.0,   # Rectifier
        200.0,   # 2nd harmonic (tube warmth)
        400.0,   # 2nd harmonic upper
    ])
    cats.extend(["preamp"] * 4)

    # EQ pivot points (Pultec-style)
    freqs.extend([
        30.0,    # Low shelf
        60.0,    # Kick fundamental
        100.0,   # Bass body
        1000.0,  # Reference
        3000.0,  # Presence
        10000.0, # Air
        12000.0, # HF shelf
        16000.0, # Ultra HF
    ])
    cats.extend(["eq"] * 8)

    # Compressor artifacts
    freqs.extend([
        0.5,     # Slow pump
        2.0,     # Fast pump
        10.0,    # Attack transient
        50.0,    # Sidechain filter
    ])
    cats.extend(["compressor"] * 4)

    # Tape (see tape basis)
    freqs.extend([
        80.0,    # Head bump
        3150.0,  # EQ crossover
        10000.0, # HF rolloff
        14000.0, # Bias limit
    ])
    cats.extend(["tape"] * 4)

    # Mix bus / summing
    freqs.extend([
        180.0,   # Transformer odd harmonic
        250.0,   # Mud frequency
        800.0,   # Nasal/honk
        5000.0,  # Harshness
    ])
    cats.extend(["summing"] * 4)

    # Mastering (final limiting, EQ)
    freqs.extend([
        40.0,    # Sub protection
        6000.0,  # De-essing
        8000.0,  # Brightness
        15000.0, # Sparkle
    ])
    cats.extend(["mastering"] * 4)

    return EquipmentBasis(
        name="studio_chain",
        description="Typical analog studio signal chain",
        frequencies=freqs,
        categories=cats,
    )


# =============================================================================
# SYNTHESIS ENGINE FOR EQUIPMENT BASIS
# =============================================================================

class EquipmentSynthesizer:
    """
    Synthesize audio using equipment-based frequency mapping.

    Instead of musical notes, coefficients represent artifact levels.
    """

    def __init__(self, basis: EquipmentBasis):
        self.basis = basis
        self.sample_rate = 48000.0

        # Precompute angular frequencies
        self.omegas = [2 * np.pi * f / self.sample_rate for f in basis.frequencies]

        # Phase accumulators
        self.phases = [0.0] * 32

        # Category masks for easy manipulation
        self.category_masks = self._build_category_masks()

    def _build_category_masks(self) -> Dict[str, List[int]]:
        """Build masks for each category."""
        masks = {}
        for i, cat in enumerate(self.basis.categories):
            if cat not in masks:
                masks[cat] = []
            masks[cat].append(i)
        return masks

    def get_category_leaves(self, category: str) -> List[int]:
        """Get leaf indices for a category."""
        return self.category_masks.get(category, [])

    def synthesize(self, coeffs: List[int], block_size: int = 64) -> List[int]:
        """
        Synthesize audio from equipment coefficients.

        Each coefficient controls the amplitude of its artifact frequency.
        """
        output = [0] * block_size

        for leaf_idx in range(32):
            amp = coeffs[leaf_idx * 8]  # Use first coeff as amplitude

            if abs(amp) < 10:
                continue

            omega = self.omegas[leaf_idx]
            phase = self.phases[leaf_idx]

            for i in range(block_size):
                sample = int(amp * np.sin(phase))
                output[i] = sat16(output[i] + sample)
                phase += omega

            self.phases[leaf_idx] = phase % (2 * np.pi)

        return output

    def reset(self):
        """Reset phase accumulators."""
        self.phases = [0.0] * 32


# =============================================================================
# PRESET COEFFICIENT PATTERNS
# =============================================================================

def make_clean_signal(amplitude: int = 8000) -> List[int]:
    """Coefficients for a 'clean' signal (character frequencies only)."""
    coeffs = [0] * 256

    # Only character region (leaves 24-27 in 60Hz basis)
    for leaf in [24, 25, 26, 27]:
        coeffs[leaf * 8] = amplitude

    return coeffs


def make_tube_warmth(amplitude: int = 6000) -> List[int]:
    """Coefficients for tube warmth (even harmonics)."""
    coeffs = [0] * 256

    # Tube harmonics (leaves 12-15) + some transformer
    for leaf in [12, 13, 14, 15]:
        # Even harmonics stronger (triode character)
        coeffs[leaf * 8] = amplitude if leaf % 2 == 0 else amplitude // 2

    return coeffs


def make_hum_60hz(amplitude: int = 2000) -> List[int]:
    """Coefficients for 60Hz hum (power supply noise)."""
    coeffs = [0] * 256

    # Power supply harmonics (leaves 0-7)
    for leaf in range(8):
        # Fundamental strongest, harmonics decay
        coeffs[leaf * 8] = amplitude // (leaf + 1)

    return coeffs


def make_tape_saturation(amplitude: int = 5000) -> List[int]:
    """Coefficients for tape saturation character."""
    coeffs = [0] * 256

    # Tape leaves (16-19) + head bump
    coeffs[16 * 8] = amplitude      # Head bump low
    coeffs[17 * 8] = amplitude * 2  # Head bump peak
    coeffs[18 * 8] = amplitude // 2 # HF rolloff
    coeffs[19 * 8] = amplitude // 4 # HF limit

    return coeffs


def make_vinyl_warmth(amplitude: int = 4000) -> List[int]:
    """Coefficients for vinyl playback character."""
    coeffs = [0] * 256

    # Rumble (subtle)
    coeffs[0] = amplitude // 4
    coeffs[1 * 8] = amplitude // 8

    # RIAA curve emphasis
    coeffs[8 * 8] = amplitude       # Bass
    coeffs[10 * 8] = amplitude      # Mid
    coeffs[12 * 8] = amplitude // 2 # Presence

    # Surface texture (very subtle)
    coeffs[24 * 8] = amplitude // 8

    return coeffs


# =============================================================================
# DEMO
# =============================================================================

def print_basis_info(basis: EquipmentBasis):
    """Print basis frequency mapping."""
    print(f"\n{basis.name.upper()}")
    print(f"  {basis.description}")
    print(f"\n  Leaf  Freq (Hz)    Category")
    print(f"  ----  -----------  ----------")

    for i, (freq, cat) in enumerate(zip(basis.frequencies, basis.categories)):
        print(f"  {i:3d}   {freq:10.1f}   {cat}")
