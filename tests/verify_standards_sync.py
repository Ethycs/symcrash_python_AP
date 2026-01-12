"""
Verification script for standards consistency fixes.
Tests ValueType enum sync and codec round-trip.
"""

def test_enum_consistency():
    """Check ValueType enum values match between registries."""
    from symcrash_python_ap.control_registry import ValueType as CV
    from symcrash_python_ap.voice.voice_engine_registry import ValueType as VV

    print("=== ValueType Enum Consistency ===")
    all_match = True
    for name in ['Q15U', 'I16', 'U16', 'BOOL', 'ENUM8', 'U32', 'Q15S']:
        cv_val = getattr(CV, name)
        vv_val = getattr(VV, name)
        match = cv_val == vv_val
        status = "OK" if match else "MISMATCH"
        print(f"  {name}: CV={cv_val}, VV={vv_val} [{status}]")
        if not match:
            all_match = False

    return all_match


def test_codec_roundtrip():
    """Test U32 codec round-trip for FM6 ratios."""
    from symcrash_python_ap.library.capsule_codec import pack_voice_params, unpack_voice_params

    print("\n=== Codec Round-Trip Test ===")

    # Test data: FM6 ratio (U32) and level (Q15U)
    # pack_voice_params takes Dict[int, int] (param_id -> value)
    test_data = {
        82: 131072,  # FM6_OP0_RATIO = 2.0x in Q16.16 (U32)
        88: 16384,   # FM6_OP0_LEVEL = 0.5 (Q15U)
    }

    print(f"  Original: {test_data}")

    packed = pack_voice_params(test_data)
    print(f"  Packed bytes: {len(packed)} bytes")

    unpacked = unpack_voice_params(packed)
    print(f"  Unpacked: {unpacked}")

    # Compare values
    match = test_data == unpacked

    print(f"  Round-trip: {'OK' if match else 'FAILED'}")
    if not match:
        for k in test_data:
            if k not in unpacked:
                print(f"    Missing key: {k}")
            elif test_data[k] != unpacked[k]:
                print(f"    Mismatch at {k}: {test_data[k]} vs {unpacked[k]}")
    return match


def main():
    print("Standards Consistency Verification\n")

    enum_ok = test_enum_consistency()
    codec_ok = test_codec_roundtrip()

    print("\n=== Summary ===")
    print(f"  Enum consistency: {'PASS' if enum_ok else 'FAIL'}")
    print(f"  Codec round-trip: {'PASS' if codec_ok else 'FAIL'}")

    if enum_ok and codec_ok:
        print("\nAll verification tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
