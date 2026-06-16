"""Rewrite test_wavelet_roundtrip.py thresholds to the measured hr=3
headroom contract.

Historical one-shot migration (companion to handoff packet E8 / the
wet-path headroom fix in ff65e81). Executed 2026-06-11; result committed
in c4e79e4. NOT idempotent - retained as the record of how the
roundtrip conformance thresholds were derived from measurement (the
2^h - 1 precision law + the full-scale HF overflow guard) rather than
guessed. Run from repo root (paths are CWD-relative)."""
from pathlib import Path

p = Path("tests/conformance/test_wavelet_roundtrip.py")
t = p.read_bytes().decode("utf-8")

edits = [
("""Key metrics:
- Max error: Maximum absolute sample error
- RMS error: Root-mean-square error
- Headroom bits: Effect of internal scaling on accuracy
""",
"""Key metrics:
- Max error: Maximum absolute sample error
- RMS error: Root-mean-square error
- Headroom bits: Effect of internal scaling on accuracy

Accuracy contract (measured 2026-06-11, effect_mode default headroom=3):
- Precision cost is 2^h - 1 LSB max error (amplitude-independent), so the
  default h=3 gives max_err <= 7 (~-73 dBFS), rms ~4.
- Overflow safety: h=3 is required for full-scale content at ALL
  frequencies (h=1 saturates internally >= 2 kHz: 19473 LSB @ 4 kHz;
  h=2 breaks at 8 kHz). Low-frequency, moderate-amplitude signals are
  safe at lower h with proportionally smaller error.
- DC passes exactly; unity mask is bit-identical to passthrough.
"""),
("""        # Error should be very small (< 1 LSB on average)
        assert result["max_err"] <= 2, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 1.0, f"RMS error too high: {result['rms_err']:.2f}\"""",
"""        # Contract at headroom=3: max 2^3 - 1 = 7 LSB
        assert result["max_err"] <= 8, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 5.0, f"RMS error too high: {result['rms_err']:.2f}\""""),
("""        samples = generate_offset_sine(freq=440.0, amplitude=32767, offset=5000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        # DC offset may introduce slightly higher error due to headroom
        assert result["max_err"] <= 5, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 2.0, f"RMS error too high: {result['rms_err']:.2f}\"""",
"""        # amplitude + offset must stay inside int16 (the old 32767+5000
        # input was out of range and measured its own clipping, not the
        # transform)
        samples = generate_offset_sine(freq=440.0, amplitude=27000, offset=5000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 8, f"Max error too high: {result['max_err']}"
        assert result["rms_err"] < 5.0, f"RMS error too high: {result['rms_err']:.2f}\""""),
("""        samples = generate_pure_sine(freq=440.0, amplitude=16384)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 2""",
"""        samples = generate_pure_sine(freq=440.0, amplitude=16384)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 8"""),
("""        # Small signals may have proportionally higher relative error
        # but absolute error should still be < 1 LSB
        assert result["max_err"] <= 1
        assert result["rms_err"] < 0.5""",
"""        # Precision cost is ABSOLUTE (2^h - 1 LSB from the headroom
        # shift), not relative, so small signals see the same ceiling
        assert result["max_err"] <= 8
        assert result["rms_err"] < 5.0"""),
("""        assert result["max_err"] <= 2, f"Max error: {result['max_err']}"
        assert result["rms_err"] < 1.0, f"RMS error: {result['rms_err']:.2f}\"""",
"""        assert result["max_err"] <= 8, f"Max error: {result['max_err']}"
        assert result["rms_err"] < 5.0, f"RMS error: {result['rms_err']:.2f}\""""),
("""        samples = generate_actual_carrier()
        processor = _get_jit_processor()

        errors = []
        for headroom in [0, 1, 2, 3, 4]:
            wet = processor.effect_mode(samples, None, headroom)
            wet_arr = np.array(wet, dtype=np.int32)
            samples_arr = np.array(samples, dtype=np.int32)
            diff = samples_arr - wet_arr
            max_err = np.max(np.abs(diff))
            rms_err = np.sqrt(np.mean(diff**2))
            errors.append((headroom, max_err, rms_err))

        # All headroom values should maintain good accuracy
        for headroom, max_err, rms_err in errors:
            assert max_err <= 4, (
                f"Headroom {headroom}: max_err={max_err} exceeds threshold"
            )
            assert rms_err < 2.0, (
                f"Headroom {headroom}: rms_err={rms_err:.2f} exceeds threshold"
            )""",
"""        # Use a safe amplitude (low freq, 16000 peak): at safe levels the
        # precision law is max_err <= 2^h - 1 from the headroom shift.
        # (The real carrier peaks at 29013, which overflows h=0/h=1 -
        # that is the overflow side of the contract, tested below.)
        samples = generate_pure_sine(freq=440.0, amplitude=16000)
        processor = _get_jit_processor()

        for headroom in [0, 1, 2, 3, 4]:
            wet = processor.effect_mode(samples, None, headroom)
            wet_arr = np.array(wet, dtype=np.int32)
            samples_arr = np.array(samples, dtype=np.int32)
            max_err = np.max(np.abs(samples_arr - wet_arr))
            assert max_err <= (1 << headroom), (
                f"Headroom {headroom}: max_err={max_err} breaks 2^h law"
            )

    def test_full_scale_high_frequency_overflow_guard(self):
        \"\"\"h=3 must survive full-scale content at all frequencies.

        Regression guard: effect_mode_jit shipped with headroom=1 for a
        while, which saturated internally >= 2 kHz (19473 LSB @ 4 kHz).
        \"\"\"
        processor = _get_jit_processor()
        for freq in [440.0, 2000.0, 4000.0, 8000.0]:
            samples = generate_pure_sine(freq=freq, amplitude=32767)
            wet = np.array(processor.effect_mode(samples, None, 3),
                           dtype=np.int32)
            max_err = np.max(np.abs(np.array(samples, dtype=np.int32) - wet))
            assert max_err <= 8, f"{freq} Hz @ h=3: max_err={max_err}\""""),
("""        # Compare headroom=0 vs headroom=2
        wet_0 = processor.effect_mode(samples, None, 0)
        wet_2 = processor.effect_mode(samples, None, 2)

        err_0 = np.max(np.abs(np.array(samples) - np.array(wet_0)))
        err_2 = np.max(np.abs(np.array(samples) - np.array(wet_2)))

        # Both should be similarly accurate
        assert err_0 <= 2
        assert err_2 <= 2""",
"""        # Compare headroom=0 vs headroom=2: each obeys the 2^h law
        wet_0 = processor.effect_mode(samples, None, 0)
        wet_2 = processor.effect_mode(samples, None, 2)

        err_0 = np.max(np.abs(np.array(samples) - np.array(wet_0)))
        err_2 = np.max(np.abs(np.array(samples) - np.array(wet_2)))

        assert err_0 <= 1
        assert err_2 <= 4"""),
("""    @pytest.mark.parametrize("freq", [100.0, 440.0, 1000.0, 4000.0])
    def test_roundtrip_at_frequency(self, freq):
        \"\"\"Round-trip should work well across frequencies.\"\"\"
        samples = generate_pure_sine(freq=freq, amplitude=20000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 2, (
            f"At {freq} Hz: max_err={result['max_err']}"
        )
        assert result["rms_err"] < 1.0, (
            f"At {freq} Hz: rms_err={result['rms_err']:.2f}"
        )""",
"""    @pytest.mark.parametrize("freq", [100.0, 440.0, 1000.0, 4000.0, 8000.0])
    def test_roundtrip_at_frequency(self, freq):
        \"\"\"Round-trip must hold across frequencies (default headroom).\"\"\"
        samples = generate_pure_sine(freq=freq, amplitude=20000)
        result = compute_roundtrip_error(samples, mask_q15=None)

        assert result["max_err"] <= 8, (
            f"At {freq} Hz: max_err={result['max_err']}"
        )
        assert result["rms_err"] < 5.0, (
            f"At {freq} Hz: rms_err={result['rms_err']:.2f}"
        )"""),
]
for old, new in edits:
    assert t.count(old) == 1, f"pattern not unique/found: {old[:60]!r}"
    t = t.replace(old, new)
p.write_bytes(t.encode("utf-8"))
print("rewrote", p)
