"""Executable OrangeCrab resource limits from the normative RAM standards.

This checks declared packed storage, not Python object overhead or FPGA timing.
An mmap is storage backed by the target's memory; its size is not zero merely
because a desktop operating system can page it. Resources sharing an address
space must be assessed together.
"""
from dataclasses import dataclass


KIB = 1024
MIB = KIB * KIB
EXTERNAL_RAM_BYTES = 128 * MIB
# WSE-STD-BRAM-RAM §2.2 gives the physical inventory as 56 x 18 Kbit.
# Its "128 KB" label is rounded; it is not another 2 KiB of storage.
BRAM_BLOCKS = 56
BRAM_BLOCK_BITS = 18 * KIB
BRAM_BYTES = BRAM_BLOCKS * BRAM_BLOCK_BITS // 8
COEFFICIENT_STORE_BYTES = 4 * MIB
COEFFICIENT_WORKING_BYTES = 32 * KIB
MIN_PREFETCH_BLOCKS = 2
SAMPLE_RATE = 48000
BLOCK_SAMPLES = 64
REFERENCE_CLOCK_HZ = 100_000_000
BLOCK_CYCLES = REFERENCE_CLOCK_HZ * BLOCK_SAMPLES // SAMPLE_RATE


def _size(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer byte count')
    return value


@dataclass(frozen=True)
class ResourceAllocation:
    """One unique packed allocation; include directories and staging buffers."""
    name: str
    bytes: int
    region: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name.strip():
            raise ValueError('Resource allocation needs a name')
        _size(self.bytes, self.name)
        if self.region not in ('coefficients', 'external_other', 'coefficient_working', 'bram_other'):
            raise ValueError(f'Unknown hardware memory region: {self.region!r}')


@dataclass(frozen=True)
class BudgetViolation:
    region: str
    used_bytes: int
    limit_bytes: int

    @property
    def excess_bytes(self):
        return self.used_bytes - self.limit_bytes


def check_allocations(allocations):
    """Return every exceeded subregion/physical limit without truncation.

    WSE-STD-BRAM-RAM sections 2.2/6.1: 128 MiB LPDDR, 56 x 18 Kbit BRAM,
    32 KiB coefficient working partition. WSE-STD-RAM-SUBSYSTEMS section
    2.2: 4 MiB maximum for the wavelet LUT/basis allocation. The two
    subregion limits do not replace the whole-device totals. This is a byte
    capacity gate; bank geometry, width waste and address overlap need their
    own checks. Passing it does not establish physical packing.
    """
    totals = {name: 0 for name in
              ('coefficients', 'external_other', 'coefficient_working', 'bram_other')}
    seen = set()
    for item in allocations:
        if type(item) is not ResourceAllocation:
            raise ValueError('Expected a ResourceAllocation')
        if item.name in seen:
            raise ValueError(f'Duplicate resource allocation: {item.name}')
        seen.add(item.name)
        totals[item.region] += item.bytes
    totals['external_total'] = totals['coefficients'] + totals['external_other']
    totals['bram_total'] = totals['coefficient_working'] + totals['bram_other']
    limits = (('coefficients', COEFFICIENT_STORE_BYTES),
              ('coefficient_working', COEFFICIENT_WORKING_BYTES),
              ('external_total', EXTERNAL_RAM_BYTES), ('bram_total', BRAM_BYTES))
    return tuple(BudgetViolation(name, totals[name], limit)
                 for name, limit in limits if totals[name] > limit)


def oscillator_allocations():
    """Count the current default image recipes without building or mapping them.

    Counts include headers/directories/DC, not just coefficient payloads.
    These are required bytes even when no image has been generated yet.
    """
    from ..vm import mode_a_resource as mode_a, basis_resource as mode_b
    return (
        ResourceAllocation('Mode A default image', mode_a.HEADER.size +
            mode_a.TABLE_COUNT * (4 + mode_a.TABLE_BYTES), 'coefficients'),
        ResourceAllocation('Mode B default image', mode_b.resource_size(), 'coefficients'),
    )
