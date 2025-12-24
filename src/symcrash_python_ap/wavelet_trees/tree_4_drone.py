"""
Tree 4 - Field / Drone
======================

Low-band detail with calm top for slow violence and body motion.
64 leaves for D=6 master tree.
"""

from . import TreeROM, PacketNode, MAX_DEPTH, LEAF_COUNT, enumerate_leaves


def build_tree_4_drone() -> TreeROM:
    """
    Tree 4 - Field / Drone (low-band detail, calm top)
    For D=6 master tree, all 64 leaves at depth 6.
    The "drone" character is now handled by tuning gains, not tree structure.
    """
    rom = TreeROM(tree_id=4, name="drone")

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

    # Complex mask: only top 1/4 (calm top for drone)
    rom.complex_mask = [i >= (3 * LEAF_COUNT // 4) for i in range(LEAF_COUNT)]

    return rom
