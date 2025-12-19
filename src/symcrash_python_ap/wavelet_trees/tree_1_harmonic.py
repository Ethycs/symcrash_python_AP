"""
Tree 1 - Harmonic Anchor
========================

Wider lows, detailed highs.
Cleaner pitch reading, less over-resolved lows.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, enumerate_leaves


def build_tree_1_harmonic() -> TreeROM:
    """
    Tree 1 - Harmonic Anchor (wider lows, detailed highs)
    Cleaner pitch reading, less over-resolved lows.
    """
    rom = TreeROM(tree_id=1, name="harmonic")

    # Initialize all as not-split
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}

    # Step 1: Split to depth 3 everywhere
    for d in range(3):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True

    # Step 2: Low 2 regions (k=0,1 at d=3) split to depth 4 only
    for k in [0, 1]:
        rom.split_mask[3][k] = True
        # Children at d=4 are leaves (don't split)

    # Step 3: Remaining 6 regions (k=2..7 at d=3) split to depth 5
    for k in range(2, 8):
        rom.split_mask[3][k] = True
        # Their children at d=4
        for k4 in range(k * 2, k * 2 + 2):
            rom.split_mask[4][k4] = True
            # Children at d=5 are leaves by default

    # Step 4: Split 4 highest-freq depth-5 leaves to depth 6
    # Highest freq at d=5 are k=28,29,30,31
    for k5 in [28, 29, 30, 31]:
        rom.split_mask[5][k5] = True

    # Enumerate leaves
    leaves = enumerate_leaves(rom.split_mask)
    leaves.sort()

    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: N_top = 12
    rom.complex_mask = [i >= 20 for i in range(32)]

    return rom
