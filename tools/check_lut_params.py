"""Check current LUT parameters and memory usage."""
import sys
sys.path.insert(0, 'src')
from symcrash_python_ap.vm.coeff_lut_streaming import DEFAULT_P, DEFAULT_K, NUM_LEAVES, BLOCK_SIZE

print("Current LUT parameters:")
print(f"  P (phase bins): {DEFAULT_P}")
print(f"  K (polyphase slots): {DEFAULT_K}")
print(f"  L (leaves): {NUM_LEAVES}")
print(f"  Block size: {BLOCK_SIZE}")
print()
print(f"LUT size per note: {NUM_LEAVES} x {DEFAULT_K} x {DEFAULT_P} x 2 bytes = {NUM_LEAVES * DEFAULT_K * DEFAULT_P * 2 / 1024:.1f} KB")
print(f"LUT size with P=2048: {NUM_LEAVES * DEFAULT_K * 2048 * 2 / 1024:.1f} KB")
print(f"LUT size with P=4096: {NUM_LEAVES * DEFAULT_K * 4096 * 2 / 1024:.1f} KB")
print()
print("Context: This is a Python prototype for embedded (RP2350 Pico 2)")
print("Runtime cost matters more than memory for the prototype.")
