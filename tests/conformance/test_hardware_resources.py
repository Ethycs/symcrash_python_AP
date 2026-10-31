"""Memory gates enforce the paper's allocations, including shared totals."""
import json
import subprocess
import sys

import pytest

from symcrash_python_ap.budgets.hardware_resources import (
    ResourceAllocation, check_allocations, oscillator_allocations,
    COEFFICIENT_STORE_BYTES, COEFFICIENT_WORKING_BYTES, EXTERNAL_RAM_BYTES, BRAM_BYTES,
    BLOCK_CYCLES, MIN_PREFETCH_BLOCKS,
)


def test_normative_target_and_exact_partition_boundaries():
    assert (COEFFICIENT_STORE_BYTES, COEFFICIENT_WORKING_BYTES) == (4194304, 32768)
    assert (EXTERNAL_RAM_BYTES, BRAM_BYTES, BLOCK_CYCLES, MIN_PREFETCH_BLOCKS) == (
        134217728, 131072, 133333, 2)
    allocations = [ResourceAllocation('tables', 4194304, 'coefficients'),
        ResourceAllocation('samples/effects/voices', 130023424, 'external_other'),
        ResourceAllocation('working', 32768, 'coefficient_working'),
        ResourceAllocation('other working', 98304, 'bram_other')]
    assert check_allocations(allocations) == ()


@pytest.mark.parametrize('region,limit,expected', [
    ('coefficients', 4194304, 'coefficients'),
    ('coefficient_working', 32768, 'coefficient_working'),
    ('external_other', 134217728, 'external_total'),
    ('bram_other', 131072, 'bram_total'),
])
def test_one_byte_over_limit_fails(region, limit, expected):
    failures = check_allocations([ResourceAllocation('resource', limit+1, region)])
    assert len(failures) == 1
    assert failures[0].region == expected
    assert failures[0].excess_bytes == 1


def test_images_must_fit_together_even_if_each_fits_alone():
    a = ResourceAllocation('A', 3*1024*1024, 'coefficients')
    b = ResourceAllocation('B', 2*1024*1024, 'coefficients')
    assert not check_allocations([a]) and not check_allocations([b])
    assert check_allocations([a, b])[0].excess_bytes == 1024*1024


def test_subregion_and_physical_overflow_are_both_reported():
    failures = check_allocations([ResourceAllocation('oversized', 200*1024*1024, 'coefficients')])
    assert [f.region for f in failures] == ['coefficients', 'external_total']


@pytest.mark.parametrize('size', [-1, True, 1.5, '32', None])
def test_invalid_sizes_cannot_bypass_gate(size):
    with pytest.raises(ValueError):
        ResourceAllocation('invalid', size, 'coefficients')


def test_unknown_regions_and_duplicate_names_are_rejected():
    with pytest.raises(ValueError):
        ResourceAllocation('hidden mmap', 9999999999, 'virtual')
    resource = ResourceAllocation('same storage', 512, 'coefficients')
    with pytest.raises(ValueError, match='Duplicate'):
        check_allocations([resource, resource])


def test_inventory_uses_recipe_sizes_without_resource_preparation(monkeypatch):
    from symcrash_python_ap.vm import mode_a_resource as a, basis_resource as b
    def forbidden(*args, **kwargs):
        pytest.fail('Storage inventory must not load or build resources')
    monkeypatch.setattr(a, 'prepare_mode_a_resources', forbidden)
    monkeypatch.setattr(b, 'prepare_basis_resources', forbidden)
    inventory = oscillator_allocations()
    assert len(inventory) == 2 and all(item.bytes > 0 for item in inventory)
    # If default images are already present, compare against their actual extents.
    for item, module in zip(inventory, (a,b)):
        path = module.default_resource_path()
        if path.exists():
            assert path.stat().st_size == item.bytes


def test_cli_exit_status_matches_actual_recipe_assessment():
    result = subprocess.run([sys.executable, 'tools/check_hardware_resources.py', '--json'],
                            capture_output=True, text=True, check=False)
    report = json.loads(result.stdout)
    failures = check_allocations(oscillator_allocations())
    assert result.returncode == int(bool(failures))
    assert report['storage_passed'] == (not failures)
    assert len(report['violations']) == len(failures)
