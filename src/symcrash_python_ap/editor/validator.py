"""
Regime Graph Validator - V4 Validation
=======================================

Implements validate_regime_graph per problem.txt requirement 5.

Validates:
- No duplicate regime IDs
- Start regime exists
- All edge destinations exist
- All edge sources exist or are "any"
- All nodes reachable from start (DFS)
"""

from typing import List, Set
from ..instrument_ir import L2Regimes


def validate_regime_graph(l2: L2Regimes) -> List[str]:
    """
    Validate L2 regime graph structure.
    
    Returns a list of human-readable error strings.
    An empty list means the graph is valid.
    
    Per problem.txt requirement 5:
    - An L2 with no regime nodes is valid
    - Reports duplicate regime IDs
    - Reports start regime that is not a node
    - Reports edges with invalid source or destination
    - Reports unreachable nodes (when start regime is valid)
    
    Args:
        l2: L2Regimes layer to validate
        
    Returns:
        List of error strings (empty = valid)
    """
    errors = []
    
    # Empty graph is valid
    if not l2.regime_nodes:
        return errors
    
    # Collect regime IDs and check for duplicates
    regime_ids = set()
    for node in l2.regime_nodes:
        if node.id in regime_ids:
            errors.append(f"Duplicate regime ID: '{node.id}'")
        regime_ids.add(node.id)
    
    # Determine start regime
    if l2.regime_graph_spec is not None:
        start_regime = l2.regime_graph_spec.start_regime
    else:
        start_regime = l2.regime_nodes[0].id if l2.regime_nodes else None
    
    # Check if start regime exists
    start_regime_valid = start_regime in regime_ids
    if not start_regime_valid:
        errors.append(f"Start regime '{start_regime}' is not a node")
    
    # Determine active edge list
    if l2.regime_graph_spec is not None:
        edges = l2.regime_graph_spec.edges
    else:
        # Convert legacy transitions to edge format for validation
        edges = []
        for t in l2.transitions:
            # Use a simple object to hold src/dst
            class EdgeLike:
                def __init__(self, src, dst):
                    self.src_regime = src
                    self.dst_regime = dst
            edges.append(EdgeLike(t.from_regime, t.to))
    
    # Validate edges
    for edge in edges:
        # Destination must be a node
        if edge.dst_regime not in regime_ids:
            errors.append(
                f"Edge {edge.src_regime} -> {edge.dst_regime}: "
                f"destination '{edge.dst_regime}' is not a node"
            )
        
        # Source must be a node or "any"
        if edge.src_regime != "any" and edge.src_regime not in regime_ids:
            errors.append(
                f"Edge {edge.src_regime} -> {edge.dst_regime}: "
                f"source '{edge.src_regime}' is not a node"
            )
    
    # Check reachability (skip if start regime is invalid)
    if start_regime_valid:
        reachable = _find_reachable_nodes(start_regime, regime_ids, edges)
        unreachable = regime_ids - reachable
        
        for node_id in sorted(unreachable):
            errors.append(f"Node '{node_id}' is not reachable from start regime '{start_regime}'")
    
    return errors


def _find_reachable_nodes(start: str, all_nodes: Set[str], edges) -> Set[str]:
    """
    Find all nodes reachable from start via DFS.
    
    Edges with src="any" make their destination reachable from any node.
    
    Args:
        start: Starting regime ID
        all_nodes: Set of all regime IDs
        edges: List of edge-like objects with src_regime and dst_regime
        
    Returns:
        Set of reachable regime IDs
    """
    # Build adjacency list
    adjacency = {node: [] for node in all_nodes}
    any_targets = set()  # Destinations of "any" edges
    
    for edge in edges:
        if edge.src_regime == "any":
            any_targets.add(edge.dst_regime)
        elif edge.src_regime in adjacency:
            adjacency[edge.src_regime].append(edge.dst_regime)
    
    # DFS from start
    reachable = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        
        # Add neighbors
        for neighbor in adjacency.get(node, []):
            if neighbor not in reachable:
                stack.append(neighbor)
        
        # "any" edges make their targets reachable from anywhere
        for target in any_targets:
            if target not in reachable:
                stack.append(target)
    
    return reachable
