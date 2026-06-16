"""Verify all Q15/Q16 to float conversions round-trip correctly.

Covers the 6 FM6 presets migrated to float format.
"""

def find_min_dp(val, scale):
    """Find minimum decimal places for round-trip accuracy."""
    f = val / scale
    for dp in range(1, 10):
        fstr = f"{f:.{dp}f}"
        fval = float(fstr)
        if round(fval * scale) == val:
            return fstr
    return repr(f)


# Maps: preset_name -> (original_q16_ratios, original_q15_levels, original_q15_betas)
presets = {
    "fm_brass_v0": {
        "ratios": [65536, 65726, 65346, 65916, 65536, 65536],
        "levels": [32767, 32767, 32767, 28671, 10922, 8191],
        "betas": [("beta_start", 0), ("beta_target", 8192)],
    },
    "fm_bell_v0": {
        "ratios": [65536, 65536, 229376, 131072, 294912, 917504],
        "levels": [32767, 8192, 6553, 10922, 4915, 3276],
        "betas": [],
    },
    "fm_horn_v0": {
        "ratios": [65536, 65602, 65470, 65668, 32768, 65536],
        "levels": [32767, 22937, 22937, 16384, 16384, 9830],
        "betas": [("beta_start", 0), ("beta_target", 4096)],
    },
    "fm_trumpet_v0": {
        "ratios": [65536, 65536, 131072, 196608, 262144, 131072],
        "levels": [32767, 16384, 13107, 9830, 6553, 16384],
        "betas": [("beta_start", 0), ("beta_target", 8192)],
    },
    "brass_stab_v0": {
        "ratios": [65536, 65602, 65470, 65668, 65536, 131072],
        "levels": [32767, 24576, 24576, 20480, 20480, 16384],
        "betas": [],
    },
    "brass_stab_v1": {
        "ratios": [65536, 131072, 196608, 262144, 327680, 131072],
        "levels": [32767, 16384, 10923, 8192, 6554, 20480],
        "betas": [],
    },
}

all_ok = True
for name, data in presets.items():
    print(f"=== {name} ===")

    print("  Ratios (Q16):")
    for i, r in enumerate(data["ratios"]):
        fstr = find_min_dp(r, 65536)
        rt = round(float(fstr) * 65536)
        ok = "OK" if rt == r else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"    Op{i+1}: {r} -> {fstr} (rt={rt}) [{ok}]")

    print("  Levels (Q15):")
    for i, lv in enumerate(data["levels"]):
        fstr = find_min_dp(lv, 32767)
        rt = round(float(fstr) * 32767)
        ok = "OK" if rt == lv else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"    Op{i+1}: {lv} -> {fstr} (rt={rt}) [{ok}]")

    if data["betas"]:
        print("  Beta (Q15):")
        for bname, val in data["betas"]:
            fstr = find_min_dp(val, 32767)
            rt = round(float(fstr) * 32767)
            ok = "OK" if rt == val else "FAIL"
            if ok == "FAIL":
                all_ok = False
            print(f"    {bname}: {val} -> {fstr} (rt={rt}) [{ok}]")

    print()

if all_ok:
    print("ALL CONVERSIONS OK!")
else:
    print("SOME CONVERSIONS FAILED - CHECK ABOVE")
