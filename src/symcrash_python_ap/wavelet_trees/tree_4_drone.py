"""
Tree 4 - Field / Drone
======================

Low-band detail with calm top for slow violence and body motion.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, enumerate_leaves


def build_tree_4_drone() -> TreeROM:
    """
    Tree 4 - Field / Drone (low-band detail, calm top)
    Slow violence, body motion.
    """
    rom = TreeROM(tree_id=4, name="drone")

    # Initialize
    for d in range(MAX_DEPTH):
        rom.split_mask[d] = {k: False for k in range(2 ** d)}

    # Step 1: Split to depth 4 everywhere
    for d in range(4):
        for k in range(2 ** d):
            rom.split_mask[d][k] = True

    # Step 2: Low 8 depth-4 nodes (k=0..7) split to depth 5
    for k in range(8):
        rom.split_mask[4][k] = True

    # Step 3: Split 8 depth-5 leaves in low region to depth 6
    # Depth-5 nodes from parents k=0..7 are k=0..15
    # Split all 8 lowest: k=0..7
    for k5 in range(8):
        rom.split_mask[5][k5] = True

    leaves = enumerate_leaves(rom.split_mask)
    leaves.sort()

    rom.leaf_list = [(n.depth, n.index) for n in leaves]
    rom.leaf_depth = [n.depth for n in leaves]
    rom.leaf_k = [n.index for n in leaves]

    # Complex mask: N_top = 8 (calm top)
    rom.complex_mask = [i >= 24 for i in range(32)]

    return rom
