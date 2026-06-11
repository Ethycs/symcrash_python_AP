"""
DX7 SysEx Importer Tests (WS1b)
===============================

Round-trip tests for the packed 128-byte DX7 voice format decoder/encoder
in tools/import_dx7_sysex.py, plus conversion invariants (carrier headroom
normalization, level curve, algorithm numbering) and an end-to-end check
that imported YAML lints clean and loads through load_instrument_from_dict.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


def _load_tool(name):
    path = REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dx7 = _load_tool("import_dx7_sysex")
lint = _load_tool("lint_presets")


# =============================================================================
# Fixture voices
# =============================================================================

def random_voice(rng):
    """Generate a random voice exercising the full range of every field."""
    ops = []
    for _ in range(6):
        ops.append({
            "eg_rates": [rng.randint(0, 99) for _ in range(4)],
            "eg_levels": [rng.randint(0, 99) for _ in range(4)],
            "break_point": rng.randint(0, 99),
            "left_depth": rng.randint(0, 99),
            "right_depth": rng.randint(0, 99),
            "left_curve": rng.randint(0, 3),
            "right_curve": rng.randint(0, 3),
            "rate_scaling": rng.randint(0, 7),
            "detune": rng.randint(0, 14),
            "ams": rng.randint(0, 3),
            "kvs": rng.randint(0, 7),
            "output_level": rng.randint(0, 99),
            "mode": rng.randint(0, 1),
            "coarse": rng.randint(0, 31),
            "fine": rng.randint(0, 99),
        })
    name = "".join(chr(rng.randint(32, 126)) for _ in range(10))
    return {
        "operators": ops,
        "pitch_eg_rates": [rng.randint(0, 99) for _ in range(4)],
        "pitch_eg_levels": [rng.randint(0, 99) for _ in range(4)],
        "algorithm": rng.randint(0, 31),
        "feedback": rng.randint(0, 7),
        "osc_key_sync": rng.randint(0, 1),
        "lfo_speed": rng.randint(0, 99),
        "lfo_delay": rng.randint(0, 99),
        "lfo_pmd": rng.randint(0, 99),
        "lfo_amd": rng.randint(0, 99),
        "lfo_key_sync": rng.randint(0, 1),
        "lfo_wave": rng.randint(0, 5),
        "lfo_pms": rng.randint(0, 7),
        "transpose": rng.randint(0, 48),
        "name": name,
    }


# =============================================================================
# Packed format round-trip
# =============================================================================

class TestOperatorRoundtrip:
    def test_edge_bitfields(self):
        """All packed bit fields survive at their range extremes."""
        op = {
            "eg_rates": [0, 99, 0, 99], "eg_levels": [99, 0, 99, 0],
            "break_point": 99, "left_depth": 99, "right_depth": 0,
            "left_curve": 3, "right_curve": 3, "rate_scaling": 7,
            "detune": 14, "ams": 3, "kvs": 7, "output_level": 99,
            "mode": 1, "coarse": 31, "fine": 99,
        }
        packed = dx7.pack_operator(op)
        assert len(packed) == 17
        assert dx7.unpack_operator(packed) == op

    def test_zero_bitfields(self):
        op = {
            "eg_rates": [0, 0, 0, 0], "eg_levels": [0, 0, 0, 0],
            "break_point": 0, "left_depth": 0, "right_depth": 0,
            "left_curve": 0, "right_curve": 0, "rate_scaling": 0,
            "detune": 0, "ams": 0, "kvs": 0, "output_level": 0,
            "mode": 0, "coarse": 0, "fine": 0,
        }
        assert dx7.unpack_operator(dx7.pack_operator(op)) == op

    def test_shared_byte_independence(self):
        """detune/RS, KVS/AMS, coarse/mode pairs do not bleed into each other."""
        base = dx7.init_voice()["operators"][0]
        for field, values in (("detune", range(15)), ("rate_scaling", range(8)),
                              ("kvs", range(8)), ("ams", range(4)),
                              ("coarse", range(32)), ("mode", range(2))):
            for v in values:
                op = dict(base)
                op[field] = v
                out = dx7.unpack_operator(dx7.pack_operator(op))
                assert out[field] == v, f"{field}={v} did not round-trip"


class TestVoiceRoundtrip:
    def test_classic_voices(self):
        for voice in (dx7.epiano1_voice(), dx7.brass1_voice(),
                      dx7.init_voice()):
            packed = dx7.pack_voice(voice)
            assert len(packed) == 128
            out = dx7.unpack_voice(packed)
            # pack pads/truncates the name to 10 chars
            voice = dict(voice, name=voice["name"][:10].ljust(10))
            assert out == voice

    def test_random_voices(self):
        rng = random.Random(0x44583721)
        for _ in range(50):
            voice = random_voice(rng)
            out = dx7.unpack_voice(dx7.pack_voice(voice))
            assert out == voice

    def test_operator_wire_order(self):
        """SysEx stores OP6 first; unpack returns OP1..OP6."""
        voice = dx7.init_voice()
        for i, op in enumerate(voice["operators"]):
            op["output_level"] = 10 + i  # OP1=10 .. OP6=15
        packed = dx7.pack_voice(voice)
        assert packed[14] == 15          # first wire op record is OP6
        assert packed[5 * 17 + 14] == 10  # last wire op record is OP1
        out = dx7.unpack_voice(packed)
        assert [op["output_level"] for op in out["operators"]] == \
            [10, 11, 12, 13, 14, 15]


class TestBankRoundtrip:
    def test_build_and_parse(self):
        voices = [dx7.epiano1_voice(), dx7.brass1_voice()]
        bank = dx7.build_bank(voices)
        assert len(bank) == 4104
        assert bank[:6] == bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
        assert bank[-1] == 0xF7
        out, warnings = dx7.parse_bank(bank)
        assert warnings == []
        assert len(out) == 32
        assert out[0]["name"].strip() == "E.PIANO 1"
        assert out[1]["name"].strip() == "BRASS   1"
        assert out[2]["name"].strip() == "INIT VOICE"
        assert out[0]["algorithm"] == 4   # 0-based (displayed 5)

    def test_checksum_validity(self):
        bank = dx7.build_bank([dx7.epiano1_voice()])
        data = bank[6:-2]
        assert (sum(data) + bank[-2]) & 0x7F == 0

    def test_corrupt_checksum_warns(self):
        bank = bytearray(dx7.build_bank([dx7.epiano1_voice()]))
        bank[-2] = (bank[-2] + 1) & 0x7F
        out, warnings = dx7.parse_bank(bytes(bank))
        assert len(out) == 32
        assert any("checksum" in w for w in warnings)

    def test_headerless_raw_blob(self):
        bank = dx7.build_bank([dx7.brass1_voice()])
        raw = bank[6:6 + 4096]
        out, warnings = dx7.parse_bank(raw)
        assert len(out) == 32
        assert out[0]["name"].strip() == "BRASS   1"
        assert any("header" in w for w in warnings)

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            dx7.parse_bank(b"\x00\x01\x02 not a sysex bank")


# =============================================================================
# Parameter conversion invariants
# =============================================================================

class TestLevelCurve:
    def test_endpoints(self):
        assert dx7.level_to_linear(0) == 0.0
        assert dx7.level_to_linear(99) == pytest.approx(1.0)

    def test_monotonic(self):
        prev = -1.0
        for lvl in range(100):
            cur = dx7.level_to_linear(lvl)
            assert cur >= prev
            prev = cur


class TestRatio:
    def test_coarse_zero_is_half(self):
        op = dict(dx7.init_voice()["operators"][0], coarse=0, fine=0)
        assert dx7.op_frequency_ratio(op, []) == pytest.approx(0.5)

    def test_coarse_fine(self):
        op = dict(dx7.init_voice()["operators"][0], coarse=2, fine=50)
        assert dx7.op_frequency_ratio(op, []) == pytest.approx(3.0)

    def test_detune_center_is_neutral(self):
        op = dict(dx7.init_voice()["operators"][0], coarse=1, fine=0, detune=7)
        assert dx7.op_frequency_ratio(op, []) == pytest.approx(1.0)

    def test_detune_extremes_bracket_center(self):
        base = dict(dx7.init_voice()["operators"][0], coarse=1, fine=0)
        lo = dx7.op_frequency_ratio(dict(base, detune=0), [])
        hi = dx7.op_frequency_ratio(dict(base, detune=14), [])
        assert lo < 1.0 < hi

    def test_fixed_mode_noted(self):
        notes = []
        op = dict(dx7.init_voice()["operators"][0], mode=1, coarse=3, fine=0)
        ratio = dx7.op_frequency_ratio(op, notes)
        assert ratio == pytest.approx(1000.0 / 261.6256, rel=1e-3)
        assert any("fixed-frequency" in n for n in notes)


class TestEnvelope:
    def test_rate_to_time_monotonic(self):
        prev = float("inf")
        for rate in range(100):
            cur = dx7.eg_rate_to_full_time_ms(rate)
            assert cur <= prev
            prev = cur

    def test_percussive_eg(self):
        adsr, flags, peak = dx7.eg_to_adsr([99, 50, 50, 80], [99, 50, 0, 0])
        assert peak == pytest.approx(1.0)
        assert adsr["sustain"] == 0.0
        assert adsr["attack"] < 10.0
        assert adsr["decay"] > 10.0
        assert adsr["release"] > 0.0

    def test_non_monotonic_decay_flagged(self):
        _, flags, _ = dx7.eg_to_adsr([99, 50, 50, 80], [99, 30, 90, 0])
        assert any("non-monotonic" in f for f in flags)

    def test_nonzero_end_level_flagged(self):
        _, flags, _ = dx7.eg_to_adsr([99, 50, 50, 80], [99, 90, 80, 70])
        assert any("L4" in f for f in flags)


class TestConversion:
    def test_algorithm_numbering(self):
        """SysEx 0-based algorithm -> IR 1-based (WS1: same DX7 numbering)."""
        cv = dx7.convert_voice(1, dx7.epiano1_voice())
        assert cv.ir["algorithm"] == 5
        cv = dx7.convert_voice(2, dx7.brass1_voice())
        assert cv.ir["algorithm"] == 18

    def test_carrier_normalization(self):
        """Summed engine-carrier levels are auto-normalized to <= 0.5."""
        for voice in (dx7.epiano1_voice(), dx7.brass1_voice()):
            cv = dx7.convert_voice(1, voice)
            carrier_sum = sum(cv.ir["levels"][c - 1]
                              for c in cv.engine_carriers)
            assert carrier_sum <= dx7.CARRIER_SUM_MAX + 1e-6

    def test_carrier_normalization_proportional(self):
        """Normalization preserves carrier level ratios (alg 32: all carriers)."""
        voice = dx7.init_voice()
        voice["algorithm"] = 31   # displayed 32: additive, 6 carriers
        for i, lvl in enumerate([99, 99, 90, 80, 70, 60]):
            voice["operators"][i]["output_level"] = lvl
        cv = dx7.convert_voice(1, voice)
        levels = cv.ir["levels"]
        assert sum(levels) <= dx7.CARRIER_SUM_MAX + 1e-6
        assert cv.carrier_scale < 1.0
        # Op1/Op2 had identical raw levels -> identical normalized levels
        assert levels[0] == pytest.approx(levels[1], rel=1e-3)
        # Ordering preserved
        assert levels[0] > levels[2] > levels[3] > levels[4] > levels[5]

    def test_velocity_and_rate_scaling_mapped(self):
        cv = dx7.convert_voice(1, dx7.epiano1_voice())
        # Dominant carrier (op1): kvs=2 -> 2/7, rs=3 -> 3/7
        assert cv.amp["velocity_sens"] == pytest.approx(2 / 7, abs=1e-3)
        assert cv.amp["key_scale_rate"] == pytest.approx(3 / 7, abs=1e-3)

    def test_lossy_notes_present(self):
        cv = dx7.convert_voice(1, dx7.epiano1_voice())
        notes = "\n".join(cv.notes)
        assert "normalized" in notes          # carrier headroom
        assert "pitch LFO" in notes           # PMD/PMS mapping


# =============================================================================
# End-to-end: bank file -> YAML -> lint -> IR loader
# =============================================================================

class TestEndToEnd:
    @pytest.fixture
    def bank_path(self, tmp_path):
        return dx7.write_demo_bank(tmp_path / "demo.syx")

    def test_import_produces_lintable_loadable_yaml(self, bank_path, tmp_path):
        out_dir = tmp_path / "out"
        rc = dx7.main([str(bank_path), "--out-dir", str(out_dir)])
        assert rc == 0
        files = sorted(out_dir.glob("*.yaml"))
        assert len(files) == 2   # E.PIANO 1 + BRASS 1 (INIT VOICE pads skipped)

        from symcrash_python_ap.instrument_ir import load_instrument_from_dict
        for f in files:
            result = lint.lint_preset(f)
            assert result.errors == [], f"{f.name}: {result.errors}"
            assert result.warnings == [], f"{f.name}: {result.warnings}"
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            instrument = load_instrument_from_dict(data)
            assert instrument is not None

    def test_output_uses_crlf(self, bank_path, tmp_path):
        out_dir = tmp_path / "out"
        dx7.main([str(bank_path), "--out-dir", str(out_dir)])
        for f in out_dir.glob("*.yaml"):
            raw = f.read_bytes()
            assert b"\r\n" in raw
            assert b"\n" not in raw.replace(b"\r\n", b"")

    def test_voice_selection(self, bank_path, tmp_path):
        out_dir = tmp_path / "sel"
        rc = dx7.main([str(bank_path), "--out-dir", str(out_dir),
                       "--voices", "2"])
        assert rc == 0
        files = list(out_dir.glob("*.yaml"))
        assert len(files) == 1
        assert "brass" in files[0].name

    def test_report_mode(self, bank_path, tmp_path, capsys):
        out_dir = tmp_path / "rep"
        rc = dx7.main([str(bank_path), "--out-dir", str(out_dir), "--report"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "lossy-conversion" in captured
        assert "E.PIANO 1" in captured
        assert "Dexed" in captured            # WS1c limitation documented
        report_file = out_dir / "import_report.txt"
        assert report_file.exists()
        assert "carrier sum" in report_file.read_text(encoding="utf-8")

    def test_no_overwrite_without_force(self, bank_path, tmp_path, capsys):
        out_dir = tmp_path / "ow"
        dx7.main([str(bank_path), "--out-dir", str(out_dir)])
        first = {f.name: f.read_bytes() for f in out_dir.glob("*.yaml")}
        dx7.main([str(bank_path), "--out-dir", str(out_dir)])
        assert "SKIP" in capsys.readouterr().out
        for f in out_dir.glob("*.yaml"):
            assert f.read_bytes() == first[f.name]
