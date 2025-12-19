"""
Tree 0 - Balanced Default
=========================

Uniform resolution with all leaves at depth 5.
Simple, predictable frequency distribution.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, enumerate_leaves


def build_tree_0_balanced() -> TreeROM:
    """
    Tree 0 - Balanced Default (uniform resolution)
    All leaves at depth 5. Simple, predictable.
    """
    rom = TreeROM(tree_id=0, name="balanced")

    # Initialize split mask
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {}
        for k in range(2 ** d):
            # Split all nodes at d < 5
            rom.split_mask[d][k] = (d < 5)

    # Enumerate leaves (all at depth 5)
    leaves = []
    for k in range(32):
        leaves.append(PacketNode(depth=5, index=k, is_leaf=True))

    # Sort by frequency
    leaves.sort()

    # Store
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: top half (N_top = 16)
    rom.complex_mask = [i >= 16 for i in range(32)]

    return rom
