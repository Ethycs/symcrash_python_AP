"""Latched ControlRegistry v0 bindings shared by both voice engine modes.

Only supplied controls override the preset. Registry family slots are not VM
family slots: COLOR/LOCAL/CATASTROPHE map explicitly to VM slots 1/3/7.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..control_registry import ControlID, ControlRegistry
from ..primitives import Q15_ONE


_REGISTRY = ControlRegistry()
_FAMILY_CONTROLS = {ControlID.TIMBRE: 1, ControlID.TEXTURE: 3, ControlID.CHAOS: 7}


@dataclass
class VoiceControls:
    values: Dict[int, int] = field(default_factory=dict)
    detune_override: Optional[int] = None
    eta_override: Optional[int] = None

    def update(self, items) -> None:
        for item in items:
            if not _REGISTRY.has_control(item.control_id):
                continue  # Reserved IDs have no binding.
            entry = _REGISTRY.get_entry(item.control_id)
            if item.value_type != entry.value_type:
                raise ValueError(f"Wrong value type for control {entry.name}")
            self.values[item.control_id] = _REGISTRY.clamp(item.control_id, int(item.value))

    @property
    def pitch_offset_cents(self) -> int:
        detune = self.detune_override
        if detune is None:
            detune = self.values.get(ControlID.DETUNE, 0)
        return detune + self.values.get(ControlID.PITCH_BEND, 0)

    def apply_envelope(self, scheduler) -> None:
        if not self.values:
            return  # All unspecified block controls leave the scheduler intact.
        scheduler.set_block_controls(
            attack_blocks=self.values.get(ControlID.ATK_BLOCKS),
            decay_blocks=self.values.get(ControlID.DCY_BLOCKS),
            sustain_level_q15=self.values.get(ControlID.SUS_Q15),
            release_blocks=self.values.get(ControlID.REL_BLOCKS),
        )

    def apply_frame(self, frame) -> None:
        if not self.values and self.eta_override is None:
            return

        if ControlID.DYNAMICS in self.values:
            frame.gain_q15 = max(0, min(Q15_ONE,
                (frame.gain_q15 * self.values[ControlID.DYNAMICS]) >> 15))
        if ControlID.MORPH in self.values:
            frame.tree_blend_q15 = self.values[ControlID.MORPH]
        if self.eta_override is not None:
            frame.beta_target_q15 = self.eta_override
        scales = {slot: self.values[cid] for cid, slot in _FAMILY_CONTROLS.items()
                  if cid in self.values}
        if scales:
            from ..vm.decode import scale_operator_families
            frame.families = scale_operator_families(frame.ops_enable, frame.families, scales)
            if getattr(frame, 'operator_strengths_q15', None) is not None:
                from ..vm.decode import _OP_FAMILY_SLOT, _canonical_op_name
                frame.operator_strengths_q15 = {name:
                    value * scales.get(_OP_FAMILY_SLOT.get(_canonical_op_name(name)), Q15_ONE) // Q15_ONE
                    for name, value in frame.operator_strengths_q15.items()}
