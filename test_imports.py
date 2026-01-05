import sys
sys.path.insert(0, 'src')

print('Testing imports...')
try:
    from symcrash_python_ap.vm.coeff_tables import CoeffTable2D
    print('[OK] coeff_tables.py imports')
except Exception as e:
    print(f'[FAIL] coeff_tables.py error: {e}')
    import traceback
    traceback.print_exc()

try:
    from symcrash_python_ap.vm.coeff_osc import coeff_osc_generate_coeffs
    print('[OK] coeff_osc.py imports')
except Exception as e:
    print(f'[FAIL] coeff_osc.py error: {e}')
    import traceback
    traceback.print_exc()

try:
    from symcrash_python_ap.vm.lanes import lane_ccs_render, extract_64_leaves, inject_64_to_256
    print('[OK] lanes.py imports')
except Exception as e:
    print(f'[FAIL] lanes.py error: {e}')
    import traceback
    traceback.print_exc()

print('All imports successful!')
