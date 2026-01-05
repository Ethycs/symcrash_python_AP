import sys
sys.path.insert(0, 'src')
from symcrash_python_ap.vm.coeff_tables_fast import get_fast_coeff_table
import time

print('Testing fast table access speed...')
start = time.time()
table = get_fast_coeff_table()
print(f'Table created in {(time.time()-start)*1000:.2f} ms')

start = time.time()
c_sin = table.sin_tables[60, 0, :]
print(f'First access (generates): {(time.time()-start)*1000:.2f} ms')

start = time.time()
c_sin = table.sin_tables[60, 0, :]
print(f'Second access (cached): {(time.time()-start)*1000:.4f} ms')

print('Fast table works - no hanging!')
