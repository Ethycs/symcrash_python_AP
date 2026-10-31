"""STD-001 signed RNE, full-depth CCS, and production saturation boundaries."""
from fractions import Fraction

import numpy as np
import pytest

from symcrash_python_ap.wavelet import fast_lifting as lift
from symcrash_python_ap.vm import hotpath_jit as hot
from tools.audit_lifting_arithmetic import forward, inverse, forward_step, inverse_step, vectors


@pytest.mark.parametrize('bits', [1, 2])
def test_bounded_rne_exhausts_the_signed_18_bit_lifting_sum_domain(bits):
    for value in range(-2**17, 2**17):
        expected = round(Fraction(value, 2**bits))
        assert lift._rshift_lift_rne(value, bits) == expected


def test_mask_reduction_signed_ties_and_product_extrema():
    for left in (-32768, -32767, -16384, -3, -1, 0, 1, 3, 16384, 32767):
        for right in (-32768, -32767, -16384, -1, 0, 1, 16384, 32767):
            product = left * right
            assert lift._rshift_lift_rne(product, 15) == round(Fraction(product, 32768))


@pytest.mark.parametrize('name,values', list(vectors()), ids=lambda x: x if isinstance(x, str) else None)
def test_ccs_transforms_against_rational_oracle(name, values):
    for headroom in (0, 1, 2, 3):
        expected_forward = forward(values, headroom)
        expected_inverse = inverse(values, headroom)
        assert lift.fwpt_jit(values, headroom) == expected_forward, (name, headroom, 'forward')
        assert lift.idwpt_jit(values, headroom) == expected_inverse, (name, headroom, 'inverse')
        out = np.empty(64, dtype=np.int16)
        hot.fwpt_64(np.asarray(values, dtype=np.int16), headroom, out)
        np.testing.assert_array_equal(out, expected_forward)
        hot.idwpt_polyphase(np.asarray(values, dtype=np.int16), out, headroom)
        np.testing.assert_array_equal(out, expected_inverse)


@pytest.mark.parametrize('slots', [2, 4, 8])
def test_all_polyphase_slots_use_independent_rational_reconstruction(slots):
    rng = np.random.default_rng(154)
    data = rng.integers(-32768, 32768, (64, slots), dtype=np.int16)
    expected = np.empty(64, dtype=np.int16)
    for slot in range(slots):
        expected[slot::slots] = inverse(data[:, slot], 2)[0::slots]
    out = np.empty(64, dtype=np.int16)
    hot.idwpt_polyphase(data.ravel(), out, 2)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('values', [[0, 1, 0, 0], [0, -1, 0, 0], [-32768, 32767]*2])
def test_scalar_lift_retains_its_explicit_step_saturation(values):
    expected = forward_step(values, step_saturation=True)
    assert lift.FastCDF53.forward_step(values) == expected
    assert lift.FastCDF53.inverse_step(*expected) == inverse_step(*expected, step_saturation=True)


@pytest.mark.parametrize('headroom', [0, 1, 2, 3])
def test_no_numba_path_preserves_headroom_and_node_saturation(monkeypatch, headroom):
    values = [32767, -32768, -32768, 32767] * 16
    monkeypatch.setattr(lift, 'NUMBA_AVAILABLE', False)
    assert lift.fwpt_jit(values, headroom) == forward(values, headroom)
    assert lift.idwpt_jit(values, headroom) == inverse(values, headroom)


@pytest.mark.parametrize('headroom', [0, 1, 2, 3])
@pytest.mark.parametrize('masked', [False, True])
def test_fused_paths_use_same_node_saturation_and_rne_mask(headroom, masked):
    values = [0, 1, -32768, 32767] * 16
    mask = [16384, -32768, 32767, -1] * 16 if masked else None
    coeffs = forward(values, headroom)
    if mask is not None:
        coeffs = [max(-32768, min(32767, round(Fraction(c*m, 32768)))) for c, m in zip(coeffs, mask)]
    expected = inverse(coeffs, headroom)
    assert lift.effect_mode_fused(values, mask, headroom) == expected
    assert lift.effect_mode_jit(values, mask, headroom) == expected


@pytest.mark.parametrize('kind,old_recipe', [
    ('mode_a', '8ae2ef3aed32a634f4377d803cabd2c7c121e1b39243e82c6e799bbfb7aecb36'),
    ('basis', '9f5faf5e008b202ae2c17db238993c185a64a2dc00f579aace55d8ea419db8a0'),
])
def test_old_half_up_resource_recipe_is_rejected(tmp_path, kind, old_recipe):
    from symcrash_python_ap.vm import mode_a_resource as a, basis_resource as b
    from symcrash_python_ap.vm.mode_a_resource_builder import build_mode_a_resource
    from symcrash_python_ap.vm.basis_resource_builder import build_basis_resource
    path = tmp_path / 'prior-recipe.bin'
    if kind == 'mode_a':
        resource, loader = a, a.PreparedModeAAtlas
        keys = [0]
        build_mode_a_resource(path, table_keys=keys, workers=1)
    else:
        resource, loader = b, b.PreparedBasisAtlas
        keys = [b.supported_pitch_increments()[-1]]
        build_basis_resource(path, pitch_increments=keys)
    image = bytearray(path.read_bytes())
    magic, version, count, recipe, checksum = resource.HEADER.unpack_from(image)
    assert recipe != bytes.fromhex(old_recipe)
    image[:resource.HEADER.size] = resource.HEADER.pack(magic, version, count, bytes.fromhex(old_recipe), checksum)
    path.write_bytes(image)
    with pytest.raises(ValueError, match='recipe'):
        loader(path, expected_keys=keys)


@pytest.mark.parametrize('wave', range(3))
def test_legacy_tables_are_generated_from_rational_lifting(monkeypatch, wave):
    from symcrash_python_ap.vm import coeff_lut_streaming as lut
    from symcrash_python_ap.vm.mode_a_resource import prepare_mode_a_resources, legacy_address
    atlas = prepare_mode_a_resources()
    monkeypatch.setattr(lut, 'fwpt_jit', lambda values, headroom_bits: forward(values, headroom_bits))
    expected = getattr(lut.StreamingCoeffLUT, 'generate_' + ('sine', 'saw', 'square')[wave])()
    np.testing.assert_array_equal(atlas.table(legacy_address(wave)).data, expected.data)
