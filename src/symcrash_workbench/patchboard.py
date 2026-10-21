"""Structural OSC controls for the generated workbench patchboard.

Open Stage Control sessions are static JSON, while the loaded preset is not.
The surface therefore talks in stable slot names (``node1``, ``env1``,
``state1``); this module maps those slots onto the live InstrumentIR and
normalizes accepted connections back to the surface.

Only structures that the sound-design YAML can faithfully represent are
editable here:

* one downstream connection per L0 node (the YAML ``output`` field),
* many targets per named envelope, and
* many transitions per behavior state.

FM operator topology is visualized by the same patchbay metaphor, but is
read-only: selecting an algorithm is the canonical way to change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from symcrash_python_ap.envelope_targets import EnvelopeTarget
from symcrash_python_ap.instrument_ir import (
    L0Edge,
    RegimeTransition,
    create_default_l2_regimes,
)
from symcrash_python_ap.vm.fm_algorithms import get_algorithm

MAX_GRAPH_NODES = 8
MAX_ENVELOPES = 8
MAX_REGIMES = 8

GRAPH_ADDRESS = "/editor/graph"
MOD_ADDRESS = "/editor/mod"
BEHAVIOR_ADDRESS = "/editor/behavior"
MATERIALIZE_ADDRESS = "/editor/behavior/materialize"
RESYNC_ADDRESS = "/editor/resync"

MOD_TARGETS = tuple(member.name for member in EnvelopeTarget)


@dataclass
class StructuralDirty:
    graph: bool = False
    envelopes: bool = False
    behavior: bool = False
    materialized_behavior: bool = False

    @property
    def any(self) -> bool:
        return self.graph or self.envelopes or self.behavior

    def clear(self):
        self.graph = False
        self.envelopes = False
        self.behavior = False
        self.materialized_behavior = False


def _slot(token, prefix: str, limit: int) -> Optional[int]:
    token = str(token).strip().lower()
    if not token.startswith(prefix):
        return None
    try:
        index = int(token[len(prefix):]) - 1
    except ValueError:
        return None
    return index if 0 <= index < limit else None


def _state_name(node) -> str:
    """Recover the simple-YAML state key represented by a regime node."""
    template = str(node.emit_template or "")
    if template.startswith("T_") and len(template) > 2:
        return template[2:].lower()
    bits = str(node.id).split("_", 1)
    return (bits[1] if len(bits) > 1 else bits[0]).lower()


def _graph_is_acyclic(node_ids: Iterable[str], edges) -> bool:
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in edges:
        if edge.src in adjacency and edge.dst in adjacency:
            adjacency[edge.src].append(edge.dst)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return False
        if node_id in visited:
            return True
        visiting.add(node_id)
        for dst in adjacency[node_id]:
            if not visit(dst):
                return False
        visiting.remove(node_id)
        visited.add(node_id)
        return True

    return all(visit(node_id) for node_id in adjacency)


class PatchboardController:
    """Apply and persist the workbench's structural OSC messages."""

    def __init__(self, session):
        self.session = session
        self.dirty = StructuralDirty()
        self.status = "Ready"

    @property
    def ir(self):
        return self.session.ir

    def reset(self, status: str = "Ready"):
        self.dirty.clear()
        self.status = status

    def handles(self, address: str) -> bool:
        return address in {
            GRAPH_ADDRESS,
            MOD_ADDRESS,
            BEHAVIOR_ADDRESS,
            MATERIALIZE_ADDRESS,
            RESYNC_ADDRESS,
        }

    def handle(self, address: str, args: Sequence[object]) -> bool:
        if address == GRAPH_ADDRESS:
            return self._set_graph_route(args)
        if address == MOD_ADDRESS:
            return self._set_envelope_targets(args)
        if address == BEHAVIOR_ADDRESS:
            return self._set_behavior_routes(args)
        if address == MATERIALIZE_ADDRESS:
            if args and not self._pressed(args[0]):
                return True
            return self._materialize_behavior()
        if address == RESYNC_ADDRESS:
            self.status = "Surface resynchronized"
            return True
        return False

    @staticmethod
    def _pressed(value) -> bool:
        try:
            return value is None or float(value) >= 0.5
        except (TypeError, ValueError):
            return False

    # ---------------------------------------------------------- L0 graph

    def _set_graph_route(self, args: Sequence[object]) -> bool:
        if not args:
            self.status = "Graph edit rejected: missing source slot"
            return False
        nodes = self.ir.layers.L0_graph.nodes[:MAX_GRAPH_NODES]
        src_index = _slot(args[0], "node", MAX_GRAPH_NODES)
        if src_index is None or src_index >= len(nodes):
            self.status = f"Graph edit rejected: inactive source {args[0]!r}"
            return False

        # L0 sound-design YAML has one `output` per node. AUDIO represents
        # the terminal/no-edge state. If the patchbay briefly reports AUDIO
        # plus a node while a cord is being dragged, prefer the node.
        destinations = []
        for token in args[1:]:
            index = _slot(token, "node", MAX_GRAPH_NODES)
            if index is not None and index < len(nodes):
                destinations.append(index)
            elif str(token).lower() != "audio":
                self.status = f"Graph edit rejected: inactive output {token!r}"
                return False
        dst_index = destinations[-1] if destinations else None
        src = nodes[src_index]
        if dst_index == src_index:
            self.status = "Graph edit rejected: a node cannot feed itself"
            return False

        graph = self.ir.layers.L0_graph
        candidate = [edge for edge in graph.edges if edge.src != src.id]
        if dst_index is not None:
            candidate.append(L0Edge(src=src.id, dst=nodes[dst_index].id))
        if not _graph_is_acyclic((node.id for node in nodes), candidate):
            self.status = "Graph edit rejected: connection would create a cycle"
            return False

        graph.edges = candidate
        self.dirty.graph = True
        dst_label = "AUDIO" if dst_index is None else nodes[dst_index].id
        self.status = f"Routed {src.id} -> {dst_label} (unsaved)"
        return True

    # ---------------------------------------------------- envelope matrix

    def _set_envelope_targets(self, args: Sequence[object]) -> bool:
        if not args:
            self.status = "Mod-map edit rejected: missing envelope slot"
            return False
        envs = list(self.ir.envelopes or [])[:MAX_ENVELOPES]
        index = _slot(args[0], "env", MAX_ENVELOPES)
        if index is None or index >= len(envs):
            self.status = f"Mod-map edit rejected: inactive source {args[0]!r}"
            return False
        targets = []
        for token in args[1:]:
            name = str(token).upper()
            if name not in MOD_TARGETS:
                self.status = f"Mod-map edit rejected: unknown target {token!r}"
                return False
            if name not in targets:
                targets.append(name)
        envs[index].targets = targets
        self.dirty.envelopes = True
        rendered = ", ".join(targets) if targets else "nothing"
        self.status = f"Routed {envs[index].id} -> {rendered} (unsaved)"
        return True

    # ------------------------------------------------------- behavior map

    def _set_behavior_routes(self, args: Sequence[object]) -> bool:
        if not args:
            self.status = "Behavior edit rejected: missing state slot"
            return False
        l2 = self.ir.layers.L2_regimes
        states = l2.regime_nodes[:MAX_REGIMES]
        source_token = str(args[0]).lower()
        if source_token == "any":
            source_id = "any"
        else:
            index = _slot(source_token, "state", MAX_REGIMES)
            if index is None or index >= len(states):
                self.status = f"Behavior edit rejected: inactive source {args[0]!r}"
                return False
            source_id = states[index].id

        destination_ids = []
        for token in args[1:]:
            index = _slot(token, "state", MAX_REGIMES)
            if index is None or index >= len(states):
                self.status = f"Behavior edit rejected: inactive output {token!r}"
                return False
            dst = states[index].id
            if dst not in destination_ids:
                destination_ids.append(dst)

        old = list(l2.transitions)
        matching = {
            transition.to: transition
            for transition in old
            if transition.from_regime == source_id
        }
        replacement = []
        for destination_id in destination_ids:
            transition = matching.get(destination_id)
            if transition is None:
                transition = RegimeTransition(
                    from_regime=source_id,
                    to=destination_id,
                    when=self._default_guard(source_id, destination_id),
                )
            replacement.append(transition)

        first = next((i for i, transition in enumerate(old)
                      if transition.from_regime == source_id), len(old))
        retained = [transition for transition in old
                    if transition.from_regime != source_id]
        l2.transitions = retained[:first] + replacement + retained[first:]
        self.dirty.behavior = True
        self.status = (f"Updated {_state_name_for_id(states, source_id)} "
                       f"transitions (unsaved)")
        return True

    @staticmethod
    def _default_guard(source_id: str, destination_id: str) -> str:
        if source_id == "any":
            return "Gate falls"
        if "IDLE" in source_id.upper() and "ATTACK" in destination_id.upper():
            return "Gate rises"
        return "dwell_done"

    def _materialize_behavior(self) -> bool:
        l2 = self.ir.layers.L2_regimes
        if l2.regime_nodes:
            self.status = "Behavior is already explicitly authored"
            return True
        defaults = create_default_l2_regimes()
        l2.regime_nodes = defaults.regime_nodes
        l2.transitions = defaults.transitions
        self.dirty.behavior = True
        self.dirty.materialized_behavior = True
        self.status = "Materialized the default behavior graph (unsaved)"
        return True

    # ------------------------------------------------------------- YAML

    def apply_yaml(self, root) -> list[str]:
        """Apply dirty structural snapshots to a ruamel round-trip root."""
        applied = []
        if self.dirty.graph:
            self._apply_graph_yaml(root)
            applied.append(GRAPH_ADDRESS)
        if self.dirty.envelopes:
            self._apply_envelopes_yaml(root)
            applied.append(MOD_ADDRESS)
        if self.dirty.behavior:
            self._apply_behavior_yaml(root)
            applied.append(BEHAVIOR_ADDRESS)
        return applied

    def _apply_graph_yaml(self, root):
        graph_yaml = root.get("graph")
        if graph_yaml is None:
            raise ValueError("structural graph editing requires sound-design graph YAML")
        outgoing = {edge.src: edge.dst for edge in self.ir.layers.L0_graph.edges}
        for node in self.ir.layers.L0_graph.nodes:
            if node.id not in graph_yaml:
                continue
            node_yaml = graph_yaml[node.id]
            if node.id in outgoing:
                node_yaml["output"] = outgoing[node.id]
            else:
                node_yaml.pop("output", None)

    def _apply_envelopes_yaml(self, root):
        envelopes_yaml = root.get("envelopes")
        if envelopes_yaml is None:
            raise ValueError("mod-map editing requires sound-design envelopes YAML")
        for envelope in self.ir.envelopes or []:
            if envelope.id not in envelopes_yaml:
                raise ValueError(f"envelope {envelope.id!r} is absent from YAML")
            envelopes_yaml[envelope.id]["targets"] = list(envelope.targets)

    def _apply_behavior_yaml(self, root):
        from ruamel.yaml.comments import CommentedMap, CommentedSeq

        behavior = root.get("behavior")
        if behavior is None:
            behavior = CommentedMap()
            root["behavior"] = behavior
        states = self.ir.layers.L2_regimes.regime_nodes[:MAX_REGIMES]
        id_to_name = {node.id: _state_name(node) for node in states}

        if self.dirty.materialized_behavior:
            states_yaml = behavior.get("states")
            if states_yaml is None:
                states_yaml = CommentedMap()
                behavior["states"] = states_yaml
            for node in states:
                state = states_yaml.get(_state_name(node))
                if state is None:
                    state = CommentedMap()
                    states_yaml[_state_name(node)] = state
                state["kind"] = node.kind
                if node.dwell_ms is not None:
                    state["dwell"] = node.dwell_ms

        # Reuse existing mapping objects for unchanged edges so comments and
        # authored guards/hysteresis survive topology edits.
        existing = behavior.get("transitions") or []
        pool = {}
        for item in existing:
            key = (str(item.get("from", "")), str(item.get("to", "")))
            pool.setdefault(key, []).append(item)
        rendered = CommentedSeq()
        for transition in self.ir.layers.L2_regimes.transitions:
            source = ("any" if transition.from_regime == "any" else
                      id_to_name.get(transition.from_regime,
                                     transition.from_regime))
            destination = id_to_name.get(transition.to, transition.to)
            bucket = pool.get((source, destination), [])
            item = bucket.pop(0) if bucket else CommentedMap()
            item["from"] = source
            item["to"] = destination
            item["when"] = transition.when
            if transition.hysteresis is not None:
                item["hysteresis"] = CommentedMap({
                    "enter": transition.hysteresis.enter,
                    "exit": transition.hysteresis.exit,
                })
            else:
                item.pop("hysteresis", None)
            rendered.append(item)
        behavior["transitions"] = rendered

    # ----------------------------------------------------------- feedback

    def push_state(self, send: Callable[[str, object], None]):
        self._push_graph(send)
        self._push_fm(send)
        self._push_mod_map(send)
        self._push_behavior(send)
        send("/editor/status", self.status)
        send("/editor/dirty", 1.0 if self.dirty.any else 0.0)

    def _push_graph(self, send):
        nodes = self.ir.layers.L0_graph.nodes[:MAX_GRAPH_NODES]
        slot_for = {node.id: f"node{i + 1}" for i, node in enumerate(nodes)}
        outgoing = {edge.src: edge.dst for edge in self.ir.layers.L0_graph.edges}
        for i in range(MAX_GRAPH_NODES):
            token = f"node{i + 1}"
            if i < len(nodes):
                node = nodes[i]
                send(f"/editor/graph/{token}",
                     f"{i + 1}. {node.id}  [{node.type}]")
                dst = slot_for.get(outgoing.get(node.id))
                send(GRAPH_ADDRESS, [token, dst or "audio"])
            else:
                send(f"/editor/graph/{token}", f"{i + 1}. -- empty --")
                send(GRAPH_ADDRESS, [token])

    def _push_fm(self, send):
        fm_node = next((node for node in self.ir.layers.L0_graph.nodes
                        if node.type == "FM6"), None)
        if fm_node is None or fm_node.params.algorithm is None:
            send("/editor/fm/name", "No FM6 node in this preset")
            for i in range(6):
                send("/editor/fm/topology", [f"op{i + 1}"])
            return
        try:
            algorithm = get_algorithm(int(fm_node.params.algorithm))
        except KeyError:
            send("/editor/fm/name",
                 f"Unknown algorithm {fm_node.params.algorithm}")
            return
        send("/editor/fm/name",
             f"Algorithm {algorithm.alg_id}: {algorithm.name}")
        feedback = int(fm_node.params.feedback or 0)
        for src in range(6):
            outputs = [f"op{dst + 1}" for dst in range(6)
                       if algorithm.modulation_matrix[src, dst]]
            if algorithm.is_carrier(src):
                outputs.append("audio")
            if feedback > 0 and algorithm.feedback_op == src:
                outputs.append(f"op{src + 1}")
            send("/editor/fm/topology", [f"op{src + 1}", *outputs])

    def _push_mod_map(self, send):
        envs = list(self.ir.envelopes or [])[:MAX_ENVELOPES]
        for i in range(MAX_ENVELOPES):
            token = f"env{i + 1}"
            if i < len(envs):
                envelope = envs[i]
                send(f"/editor/mod/{token}",
                     f"{i + 1}. {envelope.id}")
                outputs = [target for target in envelope.targets
                           if target in MOD_TARGETS]
                send(MOD_ADDRESS, [token, *outputs])
            else:
                send(f"/editor/mod/{token}", f"{i + 1}. -- empty --")
                send(MOD_ADDRESS, [token])

    def _push_behavior(self, send):
        states = self.ir.layers.L2_regimes.regime_nodes[:MAX_REGIMES]
        slot_for = {node.id: f"state{i + 1}"
                    for i, node in enumerate(states)}
        outgoing = {}
        for transition in self.ir.layers.L2_regimes.transitions:
            outgoing.setdefault(transition.from_regime, []).append(transition.to)
        for i in range(MAX_REGIMES):
            token = f"state{i + 1}"
            if i < len(states):
                node = states[i]
                dwell = ("" if node.dwell_ms is None else
                         f" · {node.dwell_ms} ms")
                send(f"/editor/behavior/{token}",
                     f"{i + 1}. {_state_name(node)} [{node.kind}]{dwell}")
                outputs = [slot_for[dst] for dst in outgoing.get(node.id, [])
                           if dst in slot_for]
                send(BEHAVIOR_ADDRESS, [token, *outputs])
            else:
                send(f"/editor/behavior/{token}", f"{i + 1}. -- empty --")
                send(BEHAVIOR_ADDRESS, [token])
        any_outputs = [slot_for[dst] for dst in outgoing.get("any", [])
                       if dst in slot_for]
        send(BEHAVIOR_ADDRESS, ["any", *any_outputs])


def _state_name_for_id(states, regime_id: str) -> str:
    if regime_id == "any":
        return "ANY"
    node = next((candidate for candidate in states
                 if candidate.id == regime_id), None)
    return _state_name(node).upper() if node is not None else regime_id


__all__ = [
    "BEHAVIOR_ADDRESS",
    "GRAPH_ADDRESS",
    "MATERIALIZE_ADDRESS",
    "MAX_ENVELOPES",
    "MAX_GRAPH_NODES",
    "MAX_REGIMES",
    "MOD_ADDRESS",
    "MOD_TARGETS",
    "PatchboardController",
]
