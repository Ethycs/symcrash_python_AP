"""Execution controls shared by native and frozen-packet VM callers.

The global scheduler's abstract compute/memory units are deliberately separate
from the per-voice degradation ladder's coefficient ticks. Reports describe the
canonical operator cost model, not measured CPU time or memory traffic.
"""
from dataclasses import dataclass, replace
from typing import Optional

from ..budgets.degradation_ladder import apply_degradation
from ..budgets.budget_predictor import predict_ops_ticks
from ..budgets.regime_budgets import RegimeBudget
from ..budgets.tail_modes import TailMode, get_tail_ops
from ..operator_library.emission_groups import is_op_allowed


CMMM_EVENT_OPS = (13, 14, 15, 16, 21, 22, 20, 23)
TAIL_OPS = frozenset((2, 11, 12, 19, 23))


def cmmm_emission_mask(mid_due: bool, top_due: bool) -> int:
    return sum(1 << lane for lane, op in enumerate(CMMM_EVENT_OPS)
               if is_op_allowed(op, mid_due, top_due))


@dataclass(frozen=True)
class ExecutionPolicy:
    serviced: bool = True
    mid_due: bool = True
    top_due: bool = True
    budget: Optional[RegimeBudget] = None


@dataclass
class ExecutionReport:
    serviced: bool = False
    backbone_ops: tuple = ()
    event_ops: tuple = ()
    tail_ops: tuple = ()
    node_mask: int = 0
    predicted_ticks: int = 0
    degradation_steps: int = 0
    emergency: bool = False
    budget_fits: bool = True
    consumer_events: tuple = ()  # Accepted post-synthesis / ensemble commands.

    def reset(self):
        self.__dict__.update(ExecutionReport().__dict__)


def _wire_op(event, version):
    if event.lane == 0:
        return (13, 8, 14)[event.variant]
    if event.lane == 4 and version == 1:
        return 14 if event.variant == 2 else 13
    return None  # Persistent controls are never dropped as musical events.


def _event_operations(cfg):
    events = [('legacy', i, e.op_id) for i, e in enumerate(cfg.event_instrs)]
    if cfg.cmmm_events is not None:
        events.extend(('cmmm', lane, op) for lane, op in enumerate(CMMM_EVENT_OPS)
                      if cfg.cmmm_events.active_mask & (1 << lane))
    events.extend(('wire', i, _wire_op(e, cfg.wire_version))
                  for i, e in enumerate(cfg.wire_events)
                  if _wire_op(e, cfg.wire_version) is not None)
    return events


def _record_execution(cfg, backbone, events, report, budget=None):
    if report is None:
        return
    report.serviced = True
    report.backbone_ops = tuple(e.op_id for e in backbone)
    report.event_ops = tuple(op for _, _, op in events)
    tail = set(cfg.tail_ops) | {e.op_id for e in cfg.wire_tail_ops}
    report.tail_ops = tuple(op for op in (2, 11, 12, 23, 19) if op in tail)
    report.node_mask = cfg.mask_u32
    report.predicted_ticks, nonlinear = predict_ops_ticks(
        list(report.backbone_ops + report.event_ops + report.tail_ops), cfg.K)
    report.emergency = 19 in tail
    if budget is not None:
        report.budget_fits = (
            report.predicted_ticks <= budget.max_ticks_per_block
            and cfg.K <= budget.max_active_nodes_K
            and (budget.allow_nonlinear or not nonlinear))


def prepare_execution(cfg, backbone):
    """Fit the actual resolved chain, after dynamic selection and wire toggles.

    Only the decoded block plan changes. Authored frames and persistent masks,
    toggles, and tail latches remain available when the budget recovers.
    """
    policy = cfg.execution_policy
    report = cfg.execution_report
    if policy is None:
        if report is not None:
            _record_execution(cfg, backbone, _event_operations(cfg), report)
        return backbone

    allowed = lambda op: is_op_allowed(op, policy.mid_due, policy.top_due)
    tail_requests = list(cfg.wire_tail_ops)
    creative = []
    for entry in backbone:
        if entry.op_id in TAIL_OPS:
            tail_requests.append(entry)
        elif allowed(entry.op_id):
            creative.append(entry)

    # Tail projections execute once, even if an authored list repeats them.
    tail_requests = list({entry.op_id: entry for entry in tail_requests}.values())

    # GluePair is optional and budgeted; projections belong to the safety tail.
    creative.extend(entry for entry in tail_requests if entry.op_id == 12)
    requested = {entry.op_id for entry in tail_requests if entry.op_id != 12}
    tail_mode = TailMode(cfg.tail_mode)
    if 23 in requested and tail_mode != TailMode.FULL:
        tail_mode = TailMode.FULL
    elif 11 in requested and tail_mode in (TailMode.MIN, TailMode.EMERGENCY):
        tail_mode = TailMode.MID
    extra_emergency = 19 in requested or cfg.tail_mode == TailMode.EMERGENCY

    # Note excitation is the minimal continuity injection (Scheduler §7).
    events = [(kind, index, op) for kind, index, op in _event_operations(cfg)
              if allowed(op) or (kind == 'wire' and cfg.wire_events[index].lane == 4)]

    if policy.budget is not None:
        # An explicitly requested emergency projection cannot disappear when
        # a FULL/MID tail is reduced. Reserve its cost outside the ladder.
        extra_ticks = (predict_ops_ticks([19], 0)[0]
                       if extra_emergency and tail_mode != TailMode.EMERGENCY else 0)
        budget = replace(policy.budget, max_ticks_per_block=max(
            0, policy.budget.max_ticks_per_block - extra_ticks))
        result = apply_degradation([e.op_id for e in creative],
                                   [e[2] for e in events], cfg.mask_u32,
                                   tail_mode, budget)
        creative = [e for e in creative if e.op_id in result.backbone_ops]
        events = [e for e in events if e[2] in result.event_ops]
        cfg.mask_u32 = result.node_mask
        cfg.K = cfg.mask_u32.bit_count()
        tail_mode = result.tail_mode
        if report is not None:
            report.degradation_steps = result.degradation_steps

    cfg.tail_mode = tail_mode
    cfg.tail_ops = get_tail_ops(tail_mode)
    # All requested projections are represented by the fitted tail mode except
    # explicit emergency. This prevents wire FORCE_ON from evading degradation.
    cfg.wire_tail_ops = [e for e in creative if e.op_id == 12]
    if extra_emergency:
        from .decode import BackboneEntry
        cfg.wire_tail_ops.append(BackboneEntry(19))
    creative = [e for e in creative if e.op_id != 12]

    keep = {(kind, index) for kind, index, _ in events}
    cfg.event_instrs = [e for i, e in enumerate(cfg.event_instrs) if ('legacy', i) in keep]
    if cfg.cmmm_events is not None:
        mask = sum(1 << lane for kind, lane, _ in events if kind == 'cmmm')
        cfg.cmmm_events = replace(cfg.cmmm_events, active_mask=mask)
    cfg.wire_events = tuple(e for i, e in enumerate(cfg.wire_events)
                            if _wire_op(e, cfg.wire_version) is None or ('wire', i) in keep)

    _record_execution(cfg, creative, events, report, policy.budget)
    return creative
