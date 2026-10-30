"""Frozen approximation data, exact indexing/rounding and cold-start guards."""
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from symcrash_python_ap import primitives as p
from symcrash_python_ap.fixed_rom import TANH_Q15, SQRT_Q15, HANN_HALF_Q15, BLEP_KERNEL_Q15
from symcrash_python_ap.safety.safety_engine import (
    hann_fade_in, hann_fade_out, _build_hann_lut, _HANN_FADE_LUT, _HANN_OLA_LUT,
)


def digest(values):
    return hashlib.sha256(b''.join(struct.pack('<h', int(x)) for x in values)).hexdigest()


@pytest.mark.parametrize('rom,size,expected', [
    (TANH_Q15, 256, 'e22d99115ec057abe6e92ed55e25c7112c0a74b9a2797a111b2c7b692b3e0e13'),
    (SQRT_Q15, 256, '7aef402968958d46cda18ac668c4a3db1221f1a6ae6e11888ce5a16203ed8ea9'),
    (HANN_HALF_Q15, 1025, '6609d7163d476ba1c2085b795c86b9ca6b19b8ce2c6b0f711ea15e206222dfe9'),
    (BLEP_KERNEL_Q15, 64, '34f344c8dc6e5907d5faeaebd30efa4580f53ee9568751e1a2679e835c30f733'),
])
def test_frozen_rom_fingerprint(rom, size, expected):
    assert type(rom) is tuple
    assert len(rom) == size
    assert all(type(x) is int and -32768 <= x <= 32767 for x in rom)
    assert digest(rom) == expected


def test_all_sqrt_entries_match_exact_integer_rounding():
    for i, value in enumerate(SQRT_Q15):
        # Compare the exact radicand with (floor + 1/2)^2. No float sqrt or
        # shared production quantizer participates in this oracle.
        numerator, denominator = 32767 ** 2 * i, 255
        floor = math.isqrt(numerator // denominator)
        midpoint = denominator * (2 * floor + 1) ** 2
        expected = floor + int(4 * numerator > midpoint or
                               (4 * numerator == midpoint and floor % 2 == 1))
        assert value == expected, i


def test_tanh_nodes_are_within_half_lsb_of_the_authored_curve():
    for i, value in enumerate(TANH_Q15):
        assert abs(value - 32767 * math.tanh((i - 128) / 32)) <= 0.500000001


def test_shipped_noninterpolated_functions_are_unchanged_for_every_int16():
    assert digest(p.tanh_q15(x) for x in range(-32768, 32768)) == (
        '1cc2e59ba38c06c6fc4a6ca417cb2020c2a776db9e36bbc0b503832b756ac759')
    assert digest(p.sqrt_q15(x) for x in range(-32768, 32768)) == (
        '5c5a0ae5f7d86442a810515f3da21f031c9a3f100e4f01e34f09beacae4aba48')


def test_tanh_interpolation_matches_rational_oracle_over_entire_input_domain():
    previous = TANH_Q15[0]
    for x in range(-32768, 32768):
        index, remainder = divmod(x + 32768, 256)
        expected = TANH_Q15[index]
        if index < 255:
            expected += round(Fraction((TANH_Q15[index + 1] - expected) * remainder, 256))
        actual = p.tanh_q15_interp(x)
        assert actual == expected, x
        assert actual >= previous, x
        previous = actual


@pytest.mark.parametrize('x,expected', [(-10**100, TANH_Q15[0]),
                                      (-32769, TANH_Q15[0]),
                                      (32512, TANH_Q15[-1]),
                                      (32768, TANH_Q15[-1]),
                                      (10**100, TANH_Q15[-1])])
def test_tanh_interpolation_clamps_instead_of_wrapping_fraction(x, expected):
    assert p.tanh_q15_interp(x) == expected


@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('x', [-32768, -30001, -1, 0, 1, 127, 10000, 32767])
def test_numpy_int16_inputs_are_widened_before_arithmetic(x):
    for fn in (p.tanh_q15, p.tanh_q15_interp, p.sqrt_q15):
        assert fn(np.int16(x)) == fn(x)


def test_mutable_compatibility_loaders_cannot_modify_the_rom():
    tanh = p._build_tanh_lut()
    sqrt = p._build_sqrt_lut()
    tanh[0] = sqrt[0] = 12345
    assert p.tanh_q15(-32768) == -32745
    assert p.sqrt_q15(0) == 0
    assert p._TANH_LUT is TANH_Q15
    assert p._SQRT_LUT is SQRT_Q15


def test_default_safety_windows_retain_the_shipped_samples():
    expected = (0, 1247, 4799, 10114, 16383, 22653, 27968, 31520, 32767)
    assert _HANN_FADE_LUT == _HANN_OLA_LUT == expected
    assert _build_hann_lut(8) == list(expected)


@pytest.mark.parametrize('length', [1, 2, 3, 7, 8, 9, 13, 16, 63, 64, 128, 1000, 4096])
def test_hann_lengths_have_bounded_error_and_exact_integer_interpolation(length):
    values = [hann_fade_in(n, length) for n in range(length + 1)]
    assert values[0] == 0 and values[-1] == 32767
    assert values == sorted(values)
    assert hann_fade_in(-999, length) == 0
    assert hann_fade_in(length + 999, length) == 32767
    for n, actual in enumerate(values):
        index, fraction = divmod(n * 1024, length)
        expected = HANN_HALF_Q15[index]
        if index < 1024:
            expected += round(Fraction((HANN_HALF_Q15[index + 1] - expected) * fraction, length))
        assert actual == expected
        ideal = 32767 * (1 - math.cos(math.pi * n / length)) / 2
        assert abs(actual - ideal) <= 1.1
        assert hann_fade_out(n, length) == values[-n - 1]


@pytest.mark.parametrize('length', [0, -1, -100])
def test_invalid_hann_length_is_rejected(length):
    with pytest.raises(ValueError, match='positive'):
        hann_fade_in(0, length)
    with pytest.raises(ValueError, match='positive'):
        _build_hann_lut(length)


def test_offline_generator_reproduces_checked_in_rom():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / 'tools/generate_fixed_rom.py'), '--check'],
                   check=True, capture_output=True, text=True, timeout=30)


def test_blep_default_table_is_bit_exact_and_uses_integer_fractional_offsets():
    from symcrash_python_ap.wavelet.oscillator import (
        BLEP_TABLE, generate_blep_kernel, generate_blep_table,
    )
    assert generate_blep_kernel() == list(BLEP_KERNEL_Q15)
    assert digest(x for row in BLEP_TABLE for x in row) == (
        '583e13413e1ba72c1fc491085e54ed261f196bc8f993effda245ce0ba8adeaf3')
    for oversampling in (1, 3, 8, 13):
        actual = generate_blep_table(oversampling)
        for phase, row in enumerate(actual):
            for index, value in enumerate(row):
                lower = BLEP_KERNEL_Q15[index]
                upper = BLEP_KERNEL_Q15[min(index + 1, 63)]
                assert value == int(Fraction(lower * (oversampling - phase) +
                                             upper * phase, oversampling))


def test_legacy_leaf_envelope_uses_the_frozen_sine_rom():
    from symcrash_python_ap.wavelet.wavelet_engine import _LEAF_ENVELOPE_Q15
    assert _LEAF_ENVELOPE_Q15 == (12539, 30273, 30273, 12539)


def test_all_fast_sine_phases_have_identical_scalar_and_vector_rounding():
    from symcrash_python_ap.wavelet import sine_core as sine
    phases = np.arange(65536, dtype=np.int32)
    scalar = [sine.sine_q15_fast(int(p)) for p in phases]
    np.testing.assert_array_equal(sine.sine_q15_vectorized(phases), scalar)
    np.testing.assert_array_equal(sine.cosine_q15_vectorized(phases),
                                  [sine.cosine_q15_fast(int(p)) for p in phases])
    # Independent real-number accuracy and symmetry checks across the cycle.
    ideal = np.rint(np.sin(phases * (2 * np.pi / 65536)) * 32767).astype(np.int32)
    np.testing.assert_array_equal(scalar, ideal)
    np.testing.assert_array_equal(scalar[32768:], -np.array(scalar[:32768]))
    assert not sine._get_lut_64k_np().flags.writeable


def test_high_resolution_sine_preserves_existing_quarter_wave_nodes():
    from symcrash_python_ap.tables.sine_hires import SINE_QUARTER_HIRES_Q15
    from symcrash_python_ap.tables.sine_quarter import SINE_QUARTER_Q15
    assert len(SINE_QUARTER_HIRES_Q15) == 16385
    assert SINE_QUARTER_HIRES_Q15[::16] == SINE_QUARTER_Q15


def test_batch_sincos_matches_scalar_at_edges_and_interpolation_ties():
    from symcrash_python_ap.vm.lanes import _sincos_batch_q15
    from symcrash_python_ap.wavelet.sine_core import sine_q15, cosine_q15
    # Every 12-bit fractional position at representative low/mid/high nodes,
    # plus ignored low phase bits and both quarter-wave endpoints.
    phases = [0, 1, 255, 0x40000000 - 1, 0x40000000]
    phases += [((node << 12) + fraction) * 256 + 127
               for node in (0, 1, 31, 127, 512, 768, 1000, 1023)
               for fraction in range(4096)]
    cosines, sines = _sincos_batch_q15(np.array(phases, dtype=np.uint32))
    np.testing.assert_array_equal(sines, [sine_q15(p) for p in phases])
    np.testing.assert_array_equal(cosines, [cosine_q15(p) for p in phases])
    assert cosines.dtype == sines.dtype == np.int32


def test_vm_wet_ola_keeps_its_legacy_gain_quantization():
    from symcrash_python_ap.vm.lanes import _WET_OLA_FADE_IN, _WET_OLA_FADE_OUT
    np.testing.assert_array_equal(_WET_OLA_FADE_IN,
                                  [0, 1247, 4798, 10114, 16383, 22653, 27969, 31520])
    np.testing.assert_array_equal(_WET_OLA_FADE_OUT,
                                  [32768, 31520, 27969, 22653, 16384, 10114, 4798, 1247])


def test_cold_import_and_fade_calls_do_not_evaluate_transcendentals():
    script = '''
import math
import numpy as np
# Initialize third-party numerical packages before installing guards. The
# synthesizer itself must still be completely cold.
import numpy.random
try:
    import numba
except ImportError:
    pass
import sys
assert 'symcrash_python_ap' not in sys.modules
def forbidden(*args, **kwargs):
    raise AssertionError('Runtime transcendental used to initialize or evaluate a ROM')
np.tanh = np.sqrt = np.sin = np.cos = forbidden
math.cos = math.sin = math.sqrt = math.tanh = forbidden
from symcrash_python_ap.primitives import tanh_q15, sqrt_q15
from symcrash_python_ap.safety.safety_engine import hann_fade_in, hann_fade_out
from symcrash_python_ap.wavelet.sine_core import sine_q15_fast, sine_q15_vectorized
from symcrash_python_ap.vm.lanes import _sincos_batch_q15
assert tanh_q15(0) == sqrt_q15(0) == 0
assert sine_q15_fast(16384) == 32767
assert sine_q15_vectorized(np.array([0, 16384])).tolist() == [0, 32767]
assert _sincos_batch_q15(np.array([0, 0x40000000]))[1].tolist() == [0, 32767]
for length in (7, 8, 16, 127, 1025):
    for n in range(length + 1):
        assert 0 <= hann_fade_in(n, length) <= 32767
        assert 0 <= hann_fade_out(n, length) <= 32767
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
