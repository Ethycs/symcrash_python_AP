"""
Test Brass Stab Operator Application
====================================

Quick test to verify operators are being applied to brass_stab preset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from symcrash_python_ap.library.preset_data import load_preset
from symcrash_python_ap.operator_library import (
    build_operator_library, get_operator_id, OP_NAME_TO_ID
)

def test_brass_stab():
    print("=" * 60)
    print("BRASS STAB OPERATOR TEST")
    print("=" * 60)

    # Load preset
    ir = load_preset("brass_stab")
    print(f"\nPreset: {ir.meta.name}")

    # Check L3_cmmm
    stable = ir.layers.L3_cmmm.emit_templates.get("Stable")
    if stable:
        print(f"\nOperators enabled: {stable.ops_enable}")
        print(f"Families: {stable.families}")
    else:
        print("No Stable template found!")

    # Check beta in compile
    if hasattr(ir, 'compile') and ir.compile:
        policy = ir.compile.completion_policy
        if hasattr(policy, 'default_tree') and policy.default_tree:
            beta = getattr(policy.default_tree, 'beta', None)
            print(f"\nBeta from preset: {beta}")
            if beta and beta > 0:
                print("  --> WET PATH ENABLED")
            else:
                print("  --> WET PATH DISABLED (beta=0)")

    # Test operator name mapping
    print("\n" + "-" * 40)
    print("Operator Name to ID Mapping:")
    for op_name in stable.ops_enable if stable else []:
        op_id = get_operator_id(op_name)
        print(f"  {op_name} -> ID {op_id}")

    # Build operator library
    print("\n" + "-" * 40)
    print("Building operator library...")
    library = build_operator_library()
    print(f"Library has {len(library)} operators")

    # Check if each operator exists
    print("\nOperator verification:")
    for op_name in stable.ops_enable if stable else []:
        op_id = get_operator_id(op_name)
        if op_id in library:
            op = library[op_id]
            print(f"  {op_name} (ID {op_id}): {op.name} - {op.description}")
        else:
            print(f"  {op_name} (ID {op_id}): NOT FOUND!")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_brass_stab()
