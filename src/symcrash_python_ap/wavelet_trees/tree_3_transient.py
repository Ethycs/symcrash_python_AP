"""
Tree 3 - Transient / Impact
===========================

Mid-band emphasis for attacks that bloom without brittle top fizz.
64 leaves for D=6 master tree.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, LEAF_COUNT, enumerate_leaves


def build_tree_3_transient() -> TreeROM:
    """
    Tree 3 - Transient / Impact (mid-band emphasis)
    For D=6 master tree, all 64 leaves at depth 6.
    The "transient" character is now handled by tuning gains, not tree structure.
    """
    rom = TreeROM(tree_id=3, name="transient")

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
