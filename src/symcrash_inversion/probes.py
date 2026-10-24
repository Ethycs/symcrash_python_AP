"""Probe manifests and the standard evaluation suite (spec §6.1, §6.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PROBE_SET_SCHEMA = "symcrash.probe-set.v0"
REQUIRED_PROBE_FIELDS = ("audio", "note", "velocity", "gate_s", "tail_s", "seed")
STANDARD_NOTES = (48, 60, 72)
STANDARD_VELOCITIES = (0.35, 0.70, 1.00)
STANDARD_GATE_S = 1.5
STANDARD_TAIL_S = 1.0
STANDARD_SEED = 0


class ProbeManifestError(ValueError):
    """Raised for a manifest that does not satisfy the probe-set schema."""

    def __init__(self, message: str, missing: Sequence[str] = (), probe_index: Optional[int] = None):
        super().__init__(message)
        self.missing = list(missing)
        self.probe_index = probe_index


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    audio: str
    note: int
    velocity: float
    gate_s: float
    tail_s: float
    seed: int
    extra: Dict[str, Any] = field(default_factory=dict)   # unknown metadata, preserved verbatim


@dataclass
class ProbeSet:
    schema: str
    instrument_id: str
    sample_rate_hz: int
    probes: List[ProbeSpec]
    identification_confidence: str = "high"              # "low" for a wrapped single WAV
    extra: Dict[str, Any] = field(default_factory=dict)   # unknown top-level metadata, preserved

    def to_dict(self) -> dict:
        d = {"schema": self.schema, "instrument_id": self.instrument_id,
             "sample_rate_hz": self.sample_rate_hz,
             "identification_confidence": self.identification_confidence,
             "probes": [{"id": p.id, "audio": p.audio, "note": p.note, "velocity": p.velocity,
                         "gate_s": p.gate_s, "tail_s": p.tail_s, "seed": p.seed, **p.extra}
                        for p in self.probes]}
        d.update(self.extra)
        return d


def probe_id(note: int, velocity: float) -> str:
    return f"n{int(note)}_v{int(round(float(velocity) * 100)):03d}"


def _probe_from_dict(raw: dict, index: int) -> ProbeSpec:
    missing = [k for k in REQUIRED_PROBE_FIELDS if k not in raw]
    if missing:
        raise ProbeManifestError(f"probe {index}: missing required field(s) {missing}", missing, index)
    known = set(REQUIRED_PROBE_FIELDS) | {"id"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return ProbeSpec(id=str(raw.get("id") or probe_id(raw["note"], raw["velocity"])),
                     audio=str(raw["audio"]), note=int(raw["note"]), velocity=float(raw["velocity"]),
                     gate_s=float(raw["gate_s"]), tail_s=float(raw["tail_s"]), seed=int(raw["seed"]),
                     extra=extra)


def probe_set_from_dict(data: dict) -> ProbeSet:
    if not isinstance(data, dict):
        raise ProbeManifestError("manifest root must be a mapping")
    if data.get("schema") != PROBE_SET_SCHEMA:
        raise ProbeManifestError(f"schema must be {PROBE_SET_SCHEMA!r}, got {data.get('schema')!r}", ["schema"])
    probes_raw = data.get("probes")
    if not isinstance(probes_raw, list) or not probes_raw:
        raise ProbeManifestError("manifest needs a non-empty 'probes' list", ["probes"])
    probes = [_probe_from_dict(p, i) for i, p in enumerate(probes_raw)]
    known = {"schema", "instrument_id", "sample_rate_hz", "probes", "identification_confidence"}
    extra = {k: v for k, v in data.items() if k not in known}
    return ProbeSet(schema=PROBE_SET_SCHEMA, instrument_id=str(data.get("instrument_id", "")),
                    sample_rate_hz=int(data.get("sample_rate_hz", 48000)), probes=probes,
                    identification_confidence=str(data.get("identification_confidence", "high")),
                    extra=extra)


def load_probe_set(path: Union[str, Path]) -> ProbeSet:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return probe_set_from_dict(data)


def standard_suite(instrument_id: str, sample_rate_hz: int = 48000) -> ProbeSet:
    """The nine-probe release gate: notes 48/60/72 x velocities 0.35/0.70/1.00, gate 1.5 s,
    tail 1.0 s, seed 0."""
    probes = [ProbeSpec(id=probe_id(n, v), audio=f"{probe_id(n, v)}.wav", note=n, velocity=v,
                        gate_s=STANDARD_GATE_S, tail_s=STANDARD_TAIL_S, seed=STANDARD_SEED)
              for n in STANDARD_NOTES for v in STANDARD_VELOCITIES]
    return ProbeSet(PROBE_SET_SCHEMA, instrument_id, sample_rate_hz, probes)


def wrap_single_wav(audio_path: Union[str, Path], note: int, velocity: float,
                    gate_s: float = STANDARD_GATE_S, tail_s: float = STANDARD_TAIL_S,
                    seed: int = STANDARD_SEED, instrument_id: Optional[str] = None,
                    sample_rate_hz: int = 48000) -> ProbeSet:
    """A legacy single WAV becomes a one-probe manifest marked identification_confidence: low."""
    p = Path(audio_path)
    spec = ProbeSpec(id=probe_id(note, velocity), audio=str(p), note=int(note), velocity=float(velocity),
                     gate_s=float(gate_s), tail_s=float(tail_s), seed=int(seed))
    return ProbeSet(PROBE_SET_SCHEMA, instrument_id or p.stem, sample_rate_hz, [spec],
                    identification_confidence="low")
