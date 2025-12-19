"""
Tree 2 - Texture Microscope
===========================

High-band resolution for texture/detail voices, noise-band, shatter.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, enumerate_leaves


def build_tree_2_texture() -> TreeROM:
    """
    Tree 2 - Texture Microscope (high-band resolution)
    For texture/detail voices, noise-band, shatter.
    """
    rom = TreeROM(tree_id=2, name="texture")

    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}

    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True

    # Step 2: Lower 4 depth-4 nodes are leaves, upper 12 split to depth 5
    for k in range(16):
        if k >= 4:
            rom.split_mask[4][k] = True

    # Step 3: Split 4 highest-freq depth-5 leaves to depth 6
    # After step 2, depth-5 nodes exist for k=8..31 (from parents k=4..15)
    # Highest freq: k=28,29,30,31 at d=5
    for k5 in [28, 29, 30, 31]:
        rom.split_mask[5][k5] = True

    leaves = enumerate_leaves(rom.split_mask)
    leaves.sort()

    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: N_top = 20 (aggressive)
    rom.complex_mask = [i >= 12 for i in range(32)]

    return rom
