"""symcrash_inversion — M0 shared contracts of the Hybrid Preset Inversion System v0.

See docs/inversion/hybrid-preset-inversion-implementation.md. This package ships the
deterministic, ML-free contracts: operator symmetry groups and canonical forms
(canonical.py), probe manifests (probes.py), the side-effect-free exact render request
(render_service.py) and structured diagnostics (diagnostics.py).
"""
from .canonical import (  # noqa: F401
    apply_permutation,
    canonical_form,
    canonicalize,
    field_signature,
    graph_automorphisms,
    symmetry_group,
    verify_permutation,
)
from .diagnostics import Diagnostic, check_supported  # noqa: F401
from .probes import (  # noqa: F401
    PROBE_SET_SCHEMA,
    ProbeManifestError,
    ProbeSet,
    ProbeSpec,
    load_probe_set,
    standard_suite,
    wrap_single_wav,
)
from .render_service import ControlPoint, RenderRequest, RenderResult, render, render_fingerprint  # noqa: F401

__version__ = "0.1.0"
