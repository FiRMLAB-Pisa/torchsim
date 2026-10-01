"""The parameter table has to describe the buffers the kernels actually take.

Its whole purpose is that a count appears once rather than in the Python
dispatch, the CPU extension and the Triton kernels separately. That only holds
while the table and the things it describes agree, which is what these check --
a new parameter added to the table but not to the dataclass, or to the kernels'
pointer list, fails here rather than by reading past the end of a buffer.
"""

import dataclasses

import torch

from torchsim.sequence import (
    EpgEngine,
    TissueProperties,
    _accelerators,
    fse_description,
)
from torchsim.sequence._accelerators import _pack_events
from torchsim.sequence._parameters import (
    EVENT_PARAMETERS,
    FLOAT_INPUTS,
    PACKED_COUNT,
    PROPERTY_NAMES,
    SEED_INPUT,
    TISSUE_COUNT,
    TISSUE_NAMES,
    TISSUE_PARAMETERS,
    TRANSMIT_INPUTS,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from torchsim.sequence._simulation import _prepare_tissue


def test_the_table_names_the_tissue_dataclass_fields():
    """``_prepare_tissue`` reads the table, so a typo would silently drop one."""
    fields = tuple(field.name for field in dataclasses.fields(TissueProperties))

    assert PROPERTY_NAMES == fields
    # The ABI is the leading run of them: the kernels take that many buffers
    # and index them by position, so a property the signal carries rather than
    # the states has to sit past the last one they read.
    assert PROPERTY_NAMES[: len(TISSUE_NAMES)] == TISSUE_NAMES


def test_every_tissue_parameter_is_prepared():
    """A short tuple here would leave a kernel reading an unset pointer."""
    prepared, _, _ = _prepare_tissue(
        TissueProperties(t1_ms=torch.tensor([800.0]), t2_ms=torch.tensor([45.0])),
        "cpu",
    )

    assert len(prepared) == TISSUE_COUNT


def test_every_event_parameter_is_packed():
    """The packed events carry a field per event parameter the table names."""
    packed = _pack_events(
        fse_description(
            torch.deg2rad(torch.full((4,), 140.0)),
            echo_spacing_s=5e-3,
            phases_rad=torch.pi / 2,
        ),
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )

    for parameter in EVENT_PARAMETERS:
        assert hasattr(packed, parameter.name), parameter.name


def test_the_seed_follows_the_packed_buffers():
    """A differentiable adjoint takes the packed inputs and then its seed."""
    assert SEED_INPUT == PACKED_COUNT
    assert PACKED_COUNT == TISSUE_COUNT + len(EVENT_PARAMETERS)


def test_only_duration_flip_and_phase_carry_an_event_gradient():
    """Gradient tuples are ordered by this, so the split has to be right.

    The structural buffers carry no gradient because they are integers. The
    saturation and the frequency a pulse is played at are floats that carry
    none either: what a pulse deposits in the bound pool is differentiable
    through the flip angle the kernels square, not through the shape's own
    power, and the offset is where the pulse sits rather than something a
    sequence solves for.
    """
    undifferentiated = {
        parameter.name for parameter in EVENT_PARAMETERS if not parameter.differentiable
    }

    assert undifferentiated == {
        "kind",
        "action",
        "output_index",
        "shim_index",
        "saturation",
        "rf_frequency_hz",
    }
    assert all(parameter.differentiable for parameter in TISSUE_PARAMETERS)


def test_the_extension_hardcodes_where_the_transmit_pair_sits():
    """``_epg_cpu.cpp`` names ``B1_INDEX`` and ``B1_PHASE_INDEX`` as literals
    to lay out its gradient plane, while the registry derives them. Inserting
    a tissue property ahead of the pair moves one and not the other, so this
    fails here rather than silently in a shimmed adjoint.
    """
    assert TRANSMIT_INPUTS == (3, 4)
    assert TISSUE_COUNT == 17


def test_a_shim_only_widens_the_transmit_pair():
    """Every other property belongs to the voxel, whatever the array does."""
    assert tissue_gradient_rows(1) == (1,) * TISSUE_COUNT
    assert tissue_gradient_rows(4) == (
        1,
        1,
        1,
        4,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert tissue_gradient_bases(1) == tuple(range(TISSUE_COUNT))
    assert tissue_gradient_bases(4) == (
        0,
        1,
        2,
        3,
        7,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
    )
    assert tissue_gradient_height(1) == TISSUE_COUNT
    assert tissue_gradient_height(4) == TISSUE_COUNT + 6


def test_the_gradient_order_matches_what_the_adjoint_returns():
    """Every tissue property, then the float event buffers, skipping ``kind``."""
    tissue = tuple(range(TISSUE_COUNT))
    events = tuple(
        TISSUE_COUNT + offset
        for offset, parameter in enumerate(EVENT_PARAMETERS)
        if parameter.differentiable
    )

    assert FLOAT_INPUTS == (*tissue, *events)


def test_autograd_asks_for_exactly_the_differentiable_inputs():
    """``needs_input_grad`` is indexed by the table, so its width must match."""
    widths = []
    original = _accelerators._wanted

    def record(needs_input_grad):
        widths.append(len(needs_input_grad))
        return original(needs_input_grad)

    _accelerators._wanted = record
    try:
        t2 = torch.tensor([45.0, 120.0], requires_grad=True)
        signal = (
            EpgEngine()
            .simulate(
                fse_description(
                    torch.deg2rad(torch.full((4,), 140.0)),
                    echo_spacing_s=5e-3,
                    phases_rad=torch.pi / 2,
                ),
                TissueProperties(t1_ms=torch.tensor([800.0, 1400.0]), t2_ms=t2),
                nstates=8,
            )
            .signal
        )
        torch.autograd.grad(signal.abs().square().sum(), t2)
    finally:
        _accelerators._wanted = original

    # The twelve trailing arguments -- state count, output count, thread count,
    # the sequence geometry, the transition table, the per-voxel rotations and
    # the row each event reads of them, the semisolid pool's lineshape, how the
    # chemically exchanging pools are carried, which terms the tissue asks for,
    # and the pool tables with the row each event reads of them -- follow the
    # packed buffers.
    assert widths and all(width == PACKED_COUNT + 12 for width in widths)
