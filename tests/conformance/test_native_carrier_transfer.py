"""Carrier field ownership and fallback detachment through the public API."""
from dataclasses import dataclass, field
from types import SimpleNamespace

from symcrash_python_ap.library.native_carrier import NativeCarrierConfig
from symcrash_python_ap.library.classic_modulation import ClassicModulation


def test_carrier_transfer_preserves_voice_owned_fields_and_complete_payload():
    carrier = NativeCarrierConfig(waveform_type=2, coeff_table_id=4,
        partial_phase_spread_q15=12345, pwm_enabled=True,
        oeis_sequence_id=7, oeis_rolloff_db=-6.5,
        noise_color=2, noise_level_q15=147, noise_attack_blocks=7,
        noise_decay_blocks=123, noise_seed_u32=0x12345678,
        fm_depth_q15=6789, fm_source=2, drone_mode=True, legato_mode=True,
        fm_topo_a_id=0, fm_topo_b_id=7, fm_topo_blend_q15=777,
        classic_modulation=ClassicModulation(amp_depth_q15=100,
            attack_amount_q15=700, attack_samples=4800, attack_harmonics=(3, 7)),
        unison_voices=8, unison_spread_cents=99, unison_width_q15=2345,
        detune_cents=101, filter_cutoff_q15=25000, filter_reso_q15=1234, filter_mode=2)
    owned = dict(version=71, classic_modulation='voice clock owner',
        unison_voices=3, unison_spread_cents=8, unison_width_q15=9000,
        detune_cents=-17, filter_cutoff_q15=4100, filter_reso_q15=29000, filter_mode=1,
        note=48, cents=-13, pwm_width_q15=321, gain_q15=8900,
        partial_amps_q15=[7, 11])
    frame = SimpleNamespace(**owned)
    carrier.apply(frame)
    expected = dict(owned, waveform_type=2, coeff_table_id=4,
        partial_phase_spread_q15=12345, pwm_enabled=True,
        oeis_sequence_id=7, oeis_rolloff_db=-6.5,
        noise_color=2, noise_level_q15=147, noise_attack_blocks=7,
        noise_decay_blocks=123, noise_seed_u32=0x12345678,
        fm_depth_q15=6789, fm_source=2, drone_mode=True, legato_mode=True,
        fm_topo_a_id=0, fm_topo_b_id=7, fm_topo_blend_q15=777)
    assert vars(frame) == expected
    assert frame.partial_amps_q15 is owned['partial_amps_q15']
    assert NativeCarrierConfig.from_dict(carrier.to_dict()) == carrier


def test_extended_carrier_keeps_detached_nested_serialization():
    @dataclass(frozen=True)
    class Payload:
        values: list = field(default_factory=lambda: [1, 2])

    @dataclass(frozen=True)
    class ExtendedCarrier(NativeCarrierConfig):
        extra: Payload = field(default_factory=Payload)

    carrier = ExtendedCarrier()
    frame = SimpleNamespace()
    carrier.apply(frame)
    assert frame.extra == {'values': [1, 2]}
    frame.extra['values'].append(3)
    assert carrier.extra.values == [1, 2]


def test_nonatomic_value_inserted_after_construction_keeps_reference_detachment():
    carrier = NativeCarrierConfig()
    # Invalid as an authored carrier, but the transfer method historically
    # copies such values. Validation elsewhere must not acquire shared state.
    object.__setattr__(carrier, 'noise_level_q15', [7, 11])
    frame = SimpleNamespace()
    carrier.apply(frame)
    assert frame.noise_level_q15 == [7, 11]
    frame.noise_level_q15[0] = 99
    assert carrier.noise_level_q15 == [7, 11]
