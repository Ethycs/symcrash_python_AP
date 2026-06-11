"""
Preset Format Linter
====================

Validates preset YAML files against the instrument_ir_v0 format.
Supports both the sound-design format (graph:) and legacy layer format (layers:).

Rules:
  ERROR:
    top_level_key      - Must use instrument_ir_v0:, not preset:
    loads_via_parser    - Must load through load_instrument_from_dict()

  WARN:
    operator_camelcase - processing/ops_enable entries should be CamelCase
    section_order      - Sections should follow canonical order
    prefer_sound_design - Should use sound-design format (graph:), not layers:

  INFO:
    has_conformance    - Header comment should include conformance profile
    derived_processing - processing: derived from emphasis (no explicit list)

Usage:
    pixi run python tools/lint_presets.py presets/instruments/
    pixi run python tools/lint_presets.py presets/instruments/my_preset.yaml
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Install with: pip install pyyaml")
    sys.exit(1)


# Canonical section orders
SOUND_DESIGN_ORDER = ["meta", "audio", "graph", "controls", "character",
                      "envelopes", "modulators", "behavior", "compile"]
LEGACY_ORDER = ["meta", "control", "envelopes", "modulators", "io", "layers", "compile"]

# Known CamelCase operator names
CAMELCASE_OPS = {
    "Tilt", "DiffuseUp", "DiffuseDown", "IntraMix",
    "FineClamp", "NormalizeScale", "BandScale",
    "LowRankMorph", "SheafProject", "BandShear", "SwapBands",
    "RotateNodes", "FoldPressure", "GluePair", "Impulse", "GlobalPulse",
}

# Map from non-canonical to canonical names
OP_ALIASES = {
    "TILT": "Tilt",
    "DIFFUSE_UP": "DiffuseUp",
    "DIFFUSE_DOWN": "DiffuseDown",
    "INTRA_MIX": "IntraMix",
    "FINE_CLAMP": "FineClamp",
    "NORMALIZE_SCALE": "NormalizeScale",
    "BAND_SCALE": "BandScale",
}


class LintResult:
    def __init__(self, filepath):
        self.filepath = filepath
        self.errors = []
        self.warnings = []
        self.infos = []

    def error(self, rule, message):
        self.errors.append((rule, message))

    def warn(self, rule, message):
        self.warnings.append((rule, message))

    def info(self, rule, message):
        self.infos.append((rule, message))

    @property
    def ok(self):
        return not self.errors and not self.warnings

    def __str__(self):
        lines = []
        name = self.filepath.name
        for rule, msg in self.errors:
            lines.append(f"  ERROR [{rule}] {name}: {msg}")
        for rule, msg in self.warnings:
            lines.append(f"  WARN  [{rule}] {name}: {msg}")
        for rule, msg in self.infos:
            lines.append(f"  INFO  [{rule}] {name}: {msg}")
        return "\n".join(lines)


def _check_operator_camelcase(result, ops, context):
    """Check a list of operator names for CamelCase compliance."""
    for op in ops:
        if op in OP_ALIASES:
            result.warn("operator_camelcase",
                f"{context}: '{op}' should be '{OP_ALIASES[op]}'")
        elif op not in CAMELCASE_OPS and not op[0].isupper():
            result.warn("operator_camelcase",
                f"{context}: '{op}' is not in canonical CamelCase")


def lint_preset(filepath: Path) -> LintResult:
    """Lint a single preset YAML file."""
    result = LintResult(filepath)
    text = filepath.read_text(encoding="utf-8")

    # Load YAML
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        result.error("yaml_parse", f"YAML parse error: {e}")
        return result

    if not isinstance(data, dict):
        result.error("yaml_parse", "YAML root is not a dict")
        return result

    # Rule: top_level_key
    if "preset" in data and "instrument_ir_v0" not in data:
        result.error("top_level_key", "Uses deprecated 'preset:' format — must use 'instrument_ir_v0:'")
    elif "instrument_ir_v0" not in data:
        result.error("top_level_key", "Missing 'instrument_ir_v0:' top-level key")

    if "instrument_ir_v0" not in data:
        return result

    ir_data = data["instrument_ir_v0"]
    if not isinstance(ir_data, dict):
        result.error("yaml_parse", "instrument_ir_v0 is not a dict")
        return result

    # Detect format
    is_sound_design = "graph" in ir_data
    is_legacy = "layers" in ir_data

    if is_legacy and not is_sound_design:
        # Rule: prefer_sound_design
        result.warn("prefer_sound_design", "Uses legacy 'layers:' format — prefer 'graph:' sound-design format")

        # Legacy-specific rules
        _lint_legacy_format(result, ir_data, text)
    elif is_sound_design:
        # Sound-design format rules
        _lint_sound_design_format(result, ir_data)

    # Rule: section_order (format-aware)
    if is_sound_design:
        canonical = SOUND_DESIGN_ORDER
    else:
        canonical = LEGACY_ORDER
    existing_sections = [k for k in ir_data.keys() if k in canonical]
    canonical_filtered = [k for k in canonical if k in existing_sections]
    if existing_sections != canonical_filtered:
        result.warn("section_order",
            f"Sections out of canonical order: {existing_sections}")

    # Rule: has_conformance
    if "Conformance:" not in text:
        result.info("has_conformance", "Missing conformance profile comment in header")

    # Rule: loads_via_parser
    try:
        from symcrash_python_ap.instrument_ir import load_instrument_from_dict
        load_instrument_from_dict(data)
    except Exception as e:
        result.error("loads_via_parser", f"Failed to load: {e}")

    return result


def _lint_sound_design_format(result, ir_data):
    """Lint rules specific to the sound-design format (graph:)."""
    graph = ir_data.get("graph", {})

    # Check each graph node has type:
    for node_id, node_data in graph.items():
        if not isinstance(node_data, dict):
            continue
        if "type" not in node_data:
            result.error("graph_node_type", f"Graph node '{node_id}' missing 'type:' field")

        # Check Amp nodes have ADSR keys
        if node_data.get("type") == "Amp":
            for key in ["attack", "decay", "sustain", "release"]:
                if key not in node_data:
                    result.warn("amp_adsr", f"Amp node '{node_id}' missing '{key}:'")

    # Check behavior processing entries for CamelCase, and note derived operators
    behavior = ir_data.get("behavior", {})
    reserved = {"states", "transitions"}
    for state_name, state_def in behavior.items():
        if state_name in reserved or not isinstance(state_def, dict):
            continue
        has_processing = "processing" in state_def
        has_emphasis = "emphasis" in state_def
        if has_processing:
            ops = state_def.get("processing", [])
            _check_operator_camelcase(result, ops, f"behavior.{state_name}")
        elif has_emphasis:
            result.info("derived_processing",
                f"behavior.{state_name}: processing derived from emphasis")


def _lint_legacy_format(result, ir_data, text):
    """Lint rules specific to the legacy layer format (layers:)."""
    # Rule: l3_inside_layers
    lines = text.split("\n")
    ir_indent = None
    for line in lines:
        if line.strip().startswith("instrument_ir_v0:"):
            ir_indent = len(line) - len(line.lstrip())
            break

    if ir_indent is not None:
        section_indent = ir_indent + 2
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("L3_cmmm:"):
                leading = len(line) - len(line.lstrip())
                if leading == section_indent:
                    result.error("l3_inside_layers", "L3_cmmm: is at top level — must be nested inside layers:")
                    break

    # Rule: emit_not_regimes
    layers = ir_data.get("layers", {})
    l3 = layers.get("L3_cmmm", {}) if isinstance(layers, dict) else {}
    if isinstance(l3, dict) and "regimes" in l3:
        result.error("emit_not_regimes", "L3_cmmm uses 'regimes:' — must use 'emit_templates:'")

    if "L3_cmmm" in ir_data:
        top_l3 = ir_data["L3_cmmm"]
        if isinstance(top_l3, dict) and "regimes" in top_l3:
            result.error("emit_not_regimes", "Top-level L3_cmmm uses 'regimes:' — must use 'emit_templates:' inside layers:")

    # Rule: operator_camelcase in emit_templates
    if isinstance(l3, dict):
        for tmpl_name, tmpl in l3.get("emit_templates", {}).items():
            if not isinstance(tmpl, dict):
                continue
            _check_operator_camelcase(result, tmpl.get("ops_enable", []), f"Template {tmpl_name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/lint_presets.py <path> [--verbose]")
        print("  <path> can be a directory or a single .yaml file")
        sys.exit(1)

    target = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.yaml"))
    else:
        print(f"ERROR: {target} not found")
        sys.exit(1)

    total_errors = 0
    total_warnings = 0
    total_infos = 0

    for f in files:
        result = lint_preset(f)
        total_errors += len(result.errors)
        total_warnings += len(result.warnings)
        total_infos += len(result.infos)

        if result.errors or result.warnings or (verbose and result.infos):
            print(str(result))
        elif verbose:
            print(f"  OK    {f.name}")

    print(f"\n{len(files)} files checked: {total_errors} errors, {total_warnings} warnings, {total_infos} infos")

    if total_errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
