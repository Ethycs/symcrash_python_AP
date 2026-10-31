"""Prepare shared oscillator data at startup/allocation, before audio service."""


def prepare_oscillator_resources():
    from .basis_resource import prepare_basis_resources
    from .mode_a_resource import prepare_mode_a_resources
    from .coefficient_tail_jit import prepare
    prepare()
    return prepare_basis_resources(), prepare_mode_a_resources()
