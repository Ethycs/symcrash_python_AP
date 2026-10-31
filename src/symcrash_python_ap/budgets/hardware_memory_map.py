"""Address checks for the paper's memory maps, without changing allocations.

The two BRAM tables describe alternative profiles and are assessed separately.
These are logical address maps, not a claim that EBR width/depth packing fits.
The documented sizes (including reserved space) are retained even when invalid.
"""
from dataclasses import dataclass

from .hardware_resources import BRAM_BYTES, EXTERNAL_RAM_BYTES, KIB, MIB, _size


@dataclass(frozen=True)
class MemorySpan:
    name: str
    start: int
    bytes: int

    def __post_init__(self):
        if type(self.name) is not str or not self.name.strip():
            raise ValueError('Memory span needs a name')
        _size(self.start, 'Span address')
        _size(self.bytes, 'Span size')
        if not self.bytes:
            raise ValueError('Memory span size must be positive')

    @property
    def end(self):
        """Exclusive end; Python arithmetic cannot wrap an overflowing address."""
        return self.start + self.bytes


@dataclass(frozen=True)
class MapViolation:
    kind: str
    names: tuple[str, ...]
    start: int
    end: int

    @property
    def bytes(self):
        return self.end - self.start


def check_memory_map(spans, capacity):
    """Report every out-of-range extent and pairwise address overlap.

    Intervals are half open, so adjacent allocations are legal. A span wholly
    outside the device and nested overlaps must also be reported. No relocation
    or size reduction is performed. Reserved spans consume address space.
    """
    _size(capacity, 'Memory capacity')
    if not capacity:
        raise ValueError('Memory capacity must be positive')
    items = tuple(spans)
    if any(type(item) is not MemorySpan for item in items):
        raise ValueError('Expected a MemorySpan')
    if len({item.name for item in items}) != len(items):
        raise ValueError('Duplicate memory span name')
    items = sorted(items, key=lambda item: (item.start, item.end, item.name))
    failures = []
    for index, left in enumerate(items):
        if left.end > capacity:
            failures.append(MapViolation('capacity', (left.name,),
                                         max(left.start, capacity), left.end))
        for right in items[index + 1:]:
            if right.start >= left.end:
                break
            failures.append(MapViolation('overlap', (left.name, right.name),
                                         right.start, min(left.end, right.end)))
    return tuple(failures)


@dataclass(frozen=True)
class DocumentedMemoryMap:
    name: str
    source: str
    capacity: int
    spans: tuple[MemorySpan, ...]


def documented_memory_maps():
    """Transcribe the normative address tables verbatim; do not repair them here."""
    return (
        DocumentedMemoryMap('bram_static', 'WSE-STD-BRAM-RAM §6.1', BRAM_BYTES, (
            MemorySpan('Voice Cache', 0x00000, 32 * KIB),
            MemorySpan('Coeff Working', 0x08000, 32 * KIB),
            MemorySpan('Reverb Taps', 0x10000, 16 * KIB),
            MemorySpan('FX Scratch', 0x14000, 8 * KIB),
            MemorySpan('DMA Staging', 0x16000, 8 * KIB),
            MemorySpan('Output Mix', 0x18000, 8 * KIB),
            MemorySpan('Reserved', 0x1A000, 24 * KIB),
        )),
        DocumentedMemoryMap('bram_tiered', 'WSE-STD-VOICE-TIERS §5.3', BRAM_BYTES, (
            MemorySpan('Active Voice Cache', 0x0000, 16 * KIB),
            MemorySpan('Prefetch Buffer', 0x4000, 16 * KIB),
            MemorySpan('Coefficient Working', 0x8000, 32 * KIB),
            MemorySpan('Reverb Tap Windows', 0x10000, 8 * KIB),
            MemorySpan('Ghost Mix Staging', 0x12000, 8 * KIB),
            MemorySpan('Mix Accumulators', 0x14000, 2 * KIB),
            MemorySpan('DMA Descriptors', 0x14800, 2 * KIB),
            MemorySpan('Reserved', 0x15000, 44 * KIB),
        )),
        DocumentedMemoryMap('external_tiered', 'WSE-STD-VOICE-TIERS §5.2', EXTERNAL_RAM_BYTES, (
            MemorySpan('Voice States', 0x00000000, 512 * KIB),
            MemorySpan('Ghost Buffers', 0x00080000, 12 * MIB),
            MemorySpan('Coefficient LUTs', 0x00C80000, 1 * MIB),
            MemorySpan('Basis Frames', 0x00D80000, 256 * KIB),
            MemorySpan('Reverb Delay Lines', 0x00E00000, 256 * KIB),
            MemorySpan('Sample ROM', 0x00E40000, 114 * MIB),
            MemorySpan('Reserved', 0x07F00000, 1 * MIB),
        )),
    )


def memory_map_report():
    from dataclasses import asdict
    reports = []
    for profile in documented_memory_maps():
        failures = check_memory_map(profile.spans, profile.capacity)
        reports.append(dict(name=profile.name, source=profile.source,
            capacity=profile.capacity, spans=[asdict(item) for item in profile.spans],
            violations=[dict(asdict(item), bytes=item.bytes) for item in failures],
            address_map_passed=not failures))
    return reports
