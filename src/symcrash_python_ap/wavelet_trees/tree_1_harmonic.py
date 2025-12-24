"""
Tree 1 - Harmonic Anchor
========================

Wider lows, detailed highs.
Cleaner pitch reading, less over-resolved lows.
64 leaves for D=6 master tree.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, LEAF_COUNT, enumerate_leaves


def build_tree_1_harmonic() -> TreeROM:
    """
    Tree 1 - Harmonic Anchor (wider lows, detailed highs)
    For D=6 master tree, all 64 leaves at depth 6.
    The "harmonic" character is now handled by tuning gains, not tree structure.
    """
    rom = TreeROM(tree_id=1, name="harmonic")

    # Initialize split mask - split all nodes to depth 6
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {}
        for k in range(2 ** d):
            rom.split_mask[d][k] = (d < MAX_DEPTH)

    # Enumerate leaves (all at depth 6)
    leaves = []
    for k in range(LEAF_COUNT):
        leaves.append(PacketNode(depth=MAX_DEPTH, index=k, is_leaf=True))

    # Sort by frequency
    leaves.sort()

    # Store
    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: top half
    rom.complex_mask = [i >= (LEAF_COUNT // 2) for i in range(LEAF_COUNT)]

    return rom
