"""
Tree 3 - Transient / Impact
===========================

Mid-band emphasis for attacks that bloom without brittle top fizz.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, enumerate_leaves


def build_tree_3_transient() -> TreeROM:
    """
    Tree 3 - Transient / Impact (mid-band emphasis)
    Attacks bloom without brittle top fizz.
    """
    rom = TreeROM(tree_id=3, name="transient")

    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}

    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True

    # Step 2: Middle 8 depth-4 nodes (k=4..11) split to depth 5
    for k in range(4, 12):
        rom.split_mask[4][k] = True

    # Step 3: Split 8 central depth-5 leaves to depth 6
    # Depth-5 nodes from parents k=4..11 are k=8..23
    # Central 8: k=12..19
    for k5 in range(12, 20):
        rom.split_mask[5][k5] = True

    leaves = enumerate_leaves(rom.split_mask)
    leaves.sort()

    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: N_top = 12
    rom.complex_mask = [i >= 20 for i in range(32)]

    return rom
