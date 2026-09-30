"""Free water beside up to four chemically exchanging pools and a semisolid one.

BART's Bloch-McConnell model carries the free water, up to four pools that
exchange with it and not with each other, and a semisolid pool. Past one
exchanging pool the kernels read a tissue as the operators its intervals apply,
tabulated per voxel and per interval length by ``torchsim.sequence._pools``, so
what is held here is the tables and the kernels that walk them: against the
state machine written out in torch, against the package's own N-pool
operators, against the closed-form kernels at one pool, and against the
fast-exchange limit.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch
from torch.autograd import forward_ad

from torchsim.sequence import EpgEngine, TissueProperties, _pools, fse_description
from torchsim.sequence._accelerators import (
    _EXCITATION,
    _INVERSION,
    _POST_SHIFT,
    _PRE_SHIFT,
    _RECORD,
    _SHIFT_AFTER,
    _SPOIL_AFTER,
    NO_GEOMETRY,
    _NativeEpg,
)
from torchsim.sequence._lineshape import lineshape_table
from torchsim.sequence._parameters import POOL_NAMES, Geometry
from torchsim.sequence._transition import DynamicPairs
from utils.packed_reference import simulate_packed_pools

STATES = 6
OUTPUTS = 4
VOXELS = 3
GEOMETRY = Geometry(flow_scale=40.0, washout_scale=2.0)

# Fraction, exchange rate in Hz, T1 and T2 in ms and chemical shift in Hz of
# pools B to E, one column per voxel: fat, and three CEST-like pools.
POOLS = (
    ((0.1, 0.12, 0.08), (40.0, 50.0, 30.0), (300.0, 350.0, 280.0), (25.0, 30.0, 20.0), (-420.0, -400.0, -440.0)),
    ((0.05, 0.04, 0.06), (200.0, 150.0, 250.0), (900.0, 950.0, 850.0), (40.0, 45.0, 35.0), (100.0, 120.0, 80.0)),
    ((0.03, 0.02, 0.04), (1000.0, 800.0, 1200.0), (700.0, 650.0, 750.0), (30.0, 28.0, 32.0), (300.0, 280.0, 320.0)),
    ((0.02, 0.03, 0.01), (500.0, 600.0, 400.0), (800.0, 850.0, 750.0), (20.0, 22.0, 18.0), (-150.0, -140.0, -160.0)),
)  # fmt: skip

# Every live term at once: an inversion, excitations shifted before, after and
# at the readout, a spoiled echo, and relaxation intervals of five lengths.
_DURATION = (
    0.0,
    0.2,
    0.0,
    0.0,
    3e-3,
    0.0,
    2e-3,
    0.0,
    0.0,
    3e-3,
    0.0,
    4e-3,
    0.0,
    0.0,
    0.2,
)
_KIND = (1, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 0, 1, 2, 0)
_FLIP = (0.0, 0.0, 1.3, 0.0, 0.0, 2.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_PHASE = (0.0, 0.0, 0.4, 0.4, 0.0, 0.3, 0.0, 0.3, 1.1, 0.0, 1.1, 0.0, 0.2, 0.2, 0.0)
_ACTION = (
    _INVERSION, 0, _EXCITATION | _SHIFT_AFTER, _RECORD | _POST_SHIFT, 0, _PRE_SHIFT,
    0, _RECORD | _SPOIL_AFTER, 0, 0, _RECORD | _SHIFT_AFTER, 0, 0, _RECORD, 0,
)  # fmt: skip
_OUTPUT = (-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1)
_SHIM = (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
_PAIR_ROW = (0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 1, 0, 0)

ROTATIONS = ("instant", "profile", "dynamic", "shimmed")
TISSUES = ((2, False), (3, True), (4, True))


@functools.cache
def _instantaneous_table():
    """A pulse with no gradient across it: one rotation, every position.

    Built once, outside any forward-mode level: the table is integrated with
    forward mode of its own, which does not nest.
    """
    import numpy as np

    from torchsim.sequence._description import RfDefinition, RfShape
    from torchsim.sequence._transition import transition_table

    flat = RfDefinition(
        id=0,
        bandwidth_hz=0.0,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,),
        band_bandwidth_hz=0.0,
        total_b1sq_power=1.0,
        magnitude=RfShape(num_uncompressed=8, samples=np.ones(8, dtype=np.float32)),
    )
    return transition_table(flat, torch.zeros(1), bins=1024, rf_raster_time_s=1e-6)


def _case(exchanging: int, semisolid: bool, rotation: str) -> dict:
    """The leaves of one train, keyed by name, with the pools as one tensor."""
    shims = 2 if rotation == "shimmed" else 1
    b1 = torch.tensor([1.0, 0.9, 1.1])
    b1_phase = torch.tensor([0.0, 0.05, -0.1])
    leaves = dict(
        t1=torch.tensor([900.0, 1200.0, 700.0]),
        t2=torch.tensor([70.0, 90.0, 50.0]),
        m0=torch.tensor([1.0, 0.8, 1.2]),
        b1=torch.cat([b1 * (1.0 - 0.2 * row) for row in range(shims)]),
        b1_phase=torch.cat([b1_phase + 0.3 * row for row in range(shims)]),
        b0=torch.tensor([3.0, -7.0, 11.0]),
        efficiency=torch.tensor([0.95, 1.0, 0.9]),
        diffusion=torch.tensor([3.0, 0.0, 6.0]),
        velocity=torch.tensor([0.01, -0.02, 0.005]),
        bound_fraction=torch.tensor([0.1, 0.12, 0.08] if semisolid else [0.0] * 3),
        bound_exchange=torch.tensor([30.0, 25.0, 40.0] if semisolid else [0.0] * 3),
        t1_bound=torch.tensor([1000.0, 900.0, 1100.0]),
        pools=torch.tensor(POOLS[:exchanging]),
        duration=torch.tensor(_DURATION),
        flip=torch.tensor(_FLIP),
        phase=torch.tensor(_PHASE),
    )
    if rotation == "dynamic":
        generator = torch.Generator().manual_seed(1)
        a = torch.randn(4, VOXELS, dtype=torch.complex128, generator=generator)
        b = torch.randn(4, VOXELS, dtype=torch.complex128, generator=generator)
        norm = (a.abs().square() + b.abs().square()).sqrt()
        a, b = a / norm, b / norm
        leaves["pairs"] = torch.stack((a.real, a.imag, b.real, b.imag), -1).float()
    return leaves


def _tissue(leaves: dict) -> tuple[torch.Tensor, ...]:
    """The prepared buffers in packing order, pool B's from the pool tensor."""
    names = (
        "t1", "t2", "m0", "b1", "b1_phase", "b0", "efficiency", "diffusion",
        "velocity", "bound_fraction", "bound_exchange", "t1_bound",
    )  # fmt: skip
    return (*(leaves[name] for name in names), *leaves["pools"][0])


def _extras(pools: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Pools C to E in ``POOL_NAMES`` order, empty past the ones carried."""
    fields = ("_fraction", "_exchange_hz", "t1_", "t2_", "_shift_hz")
    extras = []
    for name in POOL_NAMES:
        tokens = name.split("_")
        index = "bcde".index(tokens[tokens.index("pool") + 1])
        field = next(
            position
            for position, marker in enumerate(fields)
            if (
                name.startswith(marker)
                if marker.endswith("_")
                else name.endswith(marker)
            )
        )
        extras.append(
            pools[index, field] if index < pools.shape[0] else torch.zeros(VOXELS)
        )
    return tuple(extras)


def _events(leaves: dict, rotation: str) -> tuple[torch.Tensor, ...]:
    count = len(_DURATION)
    return (
        leaves["duration"],
        torch.tensor(_KIND, dtype=torch.int32),
        leaves["flip"],
        leaves["phase"],
        torch.tensor(_ACTION, dtype=torch.uint8),
        torch.tensor(_OUTPUT, dtype=torch.int32),
        torch.tensor(
            _SHIM if rotation == "shimmed" else (0,) * count, dtype=torch.int32
        ),
        torch.full((count,), 2e-6),
        torch.zeros(count),
    )


def _kernel(leaves: dict, semisolid: bool, rotation: str, *, tabulate_from: int = 1):
    """The fused kernel, reading the tables from ``tabulate_from`` pools on."""
    tissue = _tissue(leaves)
    events = _events(leaves, rotation)
    exchanging = leaves["pools"].shape[0]
    if exchanging >= tabulate_from:
        values, index, layout = _pools.tabulate(
            tissue,
            _extras(leaves["pools"]) if exchanging > 1 else None,
            events[0],
            exchanging=exchanging,
            semisolid=semisolid,
        )
    else:
        values = index = None
        layout = True
    pairs = leaves.get("pairs")
    return _NativeEpg.apply(
        *tissue,
        *events,
        STATES,
        OUTPUTS,
        1,
        GEOMETRY,
        _instantaneous_table() if rotation == "profile" else None,
        pairs,
        None if pairs is None else torch.tensor(_PAIR_ROW, dtype=torch.int32),
        lineshape_table() if semisolid else None,
        layout,
        None,
        values,
        index,
    )


def _oracle(leaves: dict, semisolid: bool, rotation: str):
    pairs = leaves.get("pairs")
    return simulate_packed_pools(
        _tissue(leaves),
        leaves["pools"],
        _events(leaves, rotation),
        state_count=STATES,
        output_count=OUTPUTS,
        geometry=GEOMETRY,
        profile=_instantaneous_table() if rotation == "profile" else None,
        dynamic=None
        if pairs is None
        else DynamicPairs.from_packed(
            pairs, torch.tensor(_PAIR_ROW, dtype=torch.int32)
        ),
        lineshape=lineshape_table() if semisolid else None,
    )


def _relative(measured: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (measured - expected).detach().abs().max() / expected.detach().abs().max()
    )


def _tolerance(rotation: str) -> float:
    # A tabulated rotation is interpolated between bins, and the kernel's
    # derivative is the slope of that interpolation where the oracle's is
    # autograd's through it.
    return 1e-3 if rotation == "profile" else 3e-5


def _gradients(leaves: dict, semisolid: bool, rotation: str, run) -> dict:
    """The gradient of a fixed real functional of the train, with its graph."""
    signal = run(leaves, semisolid, rotation)
    weight = torch.randn(
        signal.shape, dtype=torch.complex64, generator=torch.Generator().manual_seed(2)
    )
    loss = (weight.conj() * signal).real.sum()
    values = torch.autograd.grad(
        loss, tuple(leaves.values()), create_graph=True, allow_unused=True
    )
    return dict(zip(leaves, values, strict=True))


def _live(leaves: dict) -> dict:
    return {name: value.clone().requires_grad_(True) for name, value in leaves.items()}


# --- against the state machine written out in torch ---


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize(("exchanging", "semisolid"), TISSUES)
def test_the_tabulated_forward_matches_the_oracle(exchanging, semisolid, rotation):
    """Every event kind, shift and spoil, across every pool, reads the table."""
    leaves = _case(exchanging, semisolid, rotation)

    measured = _kernel(leaves, semisolid, rotation)
    expected = _oracle(leaves, semisolid, rotation)

    assert float(expected.abs().max()) > 1e-2
    assert _relative(measured, expected) < 1e-5


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize(("exchanging", "semisolid"), TISSUES)
def test_the_tabulated_adjoint_matches_the_oracle(exchanging, semisolid, rotation):
    """A gradient reaches every property through the kernel's own adjoint.

    The pool properties are differentiated by autograd through the tables and
    everything else by the kernel, so the two halves meet at the table's
    cotangent -- which is what a wrong entry would break.
    """
    leaves = _live(_case(exchanging, semisolid, rotation))

    measured = _gradients(leaves, semisolid, rotation, _kernel)
    expected = _gradients(leaves, semisolid, rotation, _oracle)

    for name, value in expected.items():
        if value is None or not float(value.detach().abs().max()):
            assert measured[name] is None or not float(
                measured[name].detach().abs().max()
            )
            continue
        assert _relative(measured[name], value) < _tolerance(rotation), name


@pytest.mark.parametrize("rotation", ("instant", "dynamic"))
@pytest.mark.parametrize(("exchanging", "semisolid"), ((2, False), (4, True)))
def test_the_tabulated_second_order_matches_the_oracle(exchanging, semisolid, rotation):
    """The gradient's own derivative along a direction in every leaf at once,
    which reaches the table's slopes along the interval lengths.
    """
    leaves = _live(_case(exchanging, semisolid, rotation))
    generator = torch.Generator().manual_seed(3)
    direction = {
        name: torch.randn(value.shape, generator=generator)
        * (value.detach().abs().mean() + 1e-3)
        for name, value in leaves.items()
    }

    def curvature(run):
        gradient = _gradients(leaves, semisolid, rotation, run)
        along = sum(
            (value * direction[name]).sum()
            for name, value in gradient.items()
            if value is not None
        )
        values = torch.autograd.grad(along, tuple(leaves.values()), allow_unused=True)
        return dict(zip(leaves, values, strict=True))

    measured = curvature(_kernel)
    expected = curvature(_oracle)

    for name, value in expected.items():
        if value is None or not float(value.detach().abs().max()):
            continue
        assert _relative(measured[name], value) < _tolerance(rotation), name


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize(("exchanging", "semisolid"), ((2, False), (4, True)))
def test_the_tabulated_forward_mode_matches_the_oracle(exchanging, semisolid, rotation):
    """A tangent in every leaf at once, the interval lengths included."""
    leaves = _case(exchanging, semisolid, rotation)
    generator = torch.Generator().manual_seed(4)
    scale = dict(duration=1e-4, flip=1e-2, phase=1e-2, pairs=1e-2)
    tangents = {
        name: torch.randn(value.shape, generator=generator)
        * scale.get(name, float(value.abs().mean()) * 1e-2 + 1e-4)
        for name, value in leaves.items()
    }

    _instantaneous_table()

    def pushed(run):
        with forward_ad.dual_level():
            dual = {
                name: forward_ad.make_dual(value, tangents[name])
                for name, value in leaves.items()
            }
            return forward_ad.unpack_dual(run(dual, semisolid, rotation)).tangent

    assert _relative(pushed(_kernel), pushed(_oracle)) < _tolerance(rotation)


# --- the tables against the closed forms ---


@pytest.mark.parametrize("semisolid", (False, True))
def test_one_tabulated_pool_matches_the_closed_form_kernel(semisolid):
    """At one exchanging pool the kernels have a closed form of their own, so a
    table of that tissue has to walk to the same train and the same gradient.
    """
    leaves = _live(_case(1, semisolid, "instant"))

    def tabulated(values, semisolid, rotation):
        return _kernel(values, semisolid, rotation, tabulate_from=1)

    def closed(values, semisolid, rotation):
        return _kernel(values, semisolid, rotation, tabulate_from=2)

    assert (
        _relative(
            tabulated(leaves, semisolid, "instant"),
            closed(leaves, semisolid, "instant"),
        )
        < 2e-6
    )
    measured = _gradients(leaves, semisolid, "instant", tabulated)
    expected = _gradients(leaves, semisolid, "instant", closed)
    for name, value in expected.items():
        if value is not None and float(value.detach().abs().max()):
            assert _relative(measured[name], value) < 2e-5, name


# --- against the package's own N-pool operators ---


def _pulse_events(delay: float, readouts=(0.0,), *, inversion: bool):
    """An optional inversion, a delay, a hard ninety degrees, and readouts at
    the given times after it.
    """
    intervals = [
        after - before
        for before, after in zip((0.0, *readouts), readouts, strict=False)
    ]
    count = len(readouts)
    return (
        torch.tensor([0.0, delay, 0.0, *intervals]),
        torch.tensor([1, 0, 1, *([2] * count)], dtype=torch.int32),
        torch.tensor([0.0, 0.0, 0.5 * math.pi, *([0.0] * count)]),
        torch.tensor([0.0, 0.0, 0.5 * math.pi, *([0.0] * count)]),
        torch.tensor(
            [_INVERSION if inversion else 0, 0, _EXCITATION, *([_RECORD] * count)],
            dtype=torch.uint8,
        ),
        torch.tensor([-1, -1, -1, *range(count)], dtype=torch.int32),
        torch.zeros(3 + count, dtype=torch.int32),
        torch.zeros(3 + count),
        torch.zeros(3 + count),
    )


def _one_voxel(pools, semisolid=None, *, t1_ms=1000.0, t2_ms=80.0):
    """Leaves of one voxel of free water beside ``pools``, rows of fraction,
    exchange rate, T1, T2 and shift, and an optional semisolid pool given as
    fraction, exchange rate and T1.
    """
    fraction, rate, t1_bound = semisolid or (0.0, 0.0, 1000.0)
    leaves = _case(len(pools), False, "instant")
    for name in ("m0", "b1", "efficiency"):
        leaves[name] = torch.ones(1)
    for name in ("b1_phase", "b0", "diffusion", "velocity"):
        leaves[name] = torch.zeros(1)
    leaves.update(
        t1=torch.tensor([t1_ms]),
        t2=torch.tensor([t2_ms]),
        bound_fraction=torch.tensor([fraction]),
        bound_exchange=torch.tensor([rate]),
        t1_bound=torch.tensor([t1_bound]),
        pools=torch.tensor(pools, dtype=torch.float32)[..., None],
    )
    return leaves


def _played(leaves, events, *, semisolid: bool) -> torch.Tensor:
    """What the fused kernel records for a voxel over ``events``."""
    tissue = _tissue(leaves)
    pools = leaves["pools"]
    padded = torch.zeros((4, 5, 1))
    padded[: pools.shape[0]] = pools
    values, index, layout = _pools.tabulate(
        tissue,
        _extras(padded),
        events[0],
        exchanging=pools.shape[0],
        semisolid=semisolid,
    )
    recorded = int((events[4] & _RECORD).ne(0).sum())
    return _NativeEpg.apply(
        *tissue,
        *events,
        STATES,
        recorded,
        1,
        NO_GEOMETRY,
        None,
        None,
        None,
        lineshape_table() if semisolid else None,
        layout,
        None,
        values,
        index,
    ).flatten()


def _exchange_matrix(weights, rates) -> torch.Tensor:
    """The free water, first, exchanging with every other pool alone, each at
    its own rate scaled by the share it leaves towards.
    """
    count = len(weights)
    matrix = torch.zeros((count, count), dtype=torch.float64)
    for at in range(1, count):
        matrix[0, 0] -= rates[at] * weights[at]
        matrix[at, 0] = rates[at] * weights[at]
        matrix[0, at] = rates[at] * weights[0]
        matrix[at, at] = -rates[at] * weights[0]
    return matrix


# Pools B to E of one voxel: fraction, exchange rate in Hz, T1 and T2 in ms,
# chemical shift in Hz.
FOUR = (
    (0.1, 40.0, 300.0, 25.0, -420.0),
    (0.05, 200.0, 900.0, 40.0, 100.0),
    (0.03, 1000.0, 700.0, 30.0, 300.0),
    (0.02, 500.0, 800.0, 20.0, -150.0),
)
SEMISOLID = (0.1, 30.0, 1100.0)


def _gain() -> complex:
    """What a hard ninety degrees reads off unit longitudinal magnetization."""
    empty = _one_voxel([(0.0, 0.0, 1000.0, 100.0, 0.0)])
    return complex(
        _played(empty, _pulse_events(0.0, inversion=False), semisolid=False)[0]
    )


def test_the_longitudinal_step_reproduces_the_package_operator():
    """Six longitudinal pools against ``epg``'s generic exchange operator.

    ``longitudinal_relaxation_exchange_op`` exponentiates a matrix it is
    handed and reaches the recovery by a linear solve, where the table carries
    ``E eq`` itself. An inversion turns the free water and the exchanging pools
    over and leaves the semisolid one alone, so all six start away from
    equilibrium, and a hard ninety degrees writes every transverse pool's ``Z``
    into the plane for the coil to sum.
    """
    from utils.epg import longitudinal_relaxation_exchange_op

    leaves = _one_voxel(FOUR, SEMISOLID)
    fractions = [row[0] for row in FOUR]
    weights = [1.0 - sum(fractions) - SEMISOLID[0], *fractions, SEMISOLID[0]]
    matrix = _exchange_matrix(weights, [0.0, *(row[1] for row in FOUR), SEMISOLID[1]])
    relaxation = [1.0, *(1000.0 / row[2] for row in FOUR), 1000.0 / SEMISOLID[2]]
    start = torch.tensor(
        [*(-w for w in weights[:-1]), weights[-1]], dtype=torch.complex128
    )
    gain = _gain()

    for delay in (0.05, 0.3, 1.0):
        measured = complex(
            _played(leaves, _pulse_events(delay, inversion=True), semisolid=True)[0]
        )
        operator, recovery = longitudinal_relaxation_exchange_op(
            torch.tensor(weights, dtype=torch.complex128),
            matrix.to(torch.complex128),
            torch.tensor(relaxation, dtype=torch.complex128),
            torch.tensor(delay, dtype=torch.float64),
        )
        settled = operator @ start + recovery
        expected = complex(settled[:-1].sum()) * gain
        assert abs(measured - expected) / abs(gain) < 2e-5, delay


def test_the_transverse_step_reproduces_the_package_operator():
    """Five transverse pools, each at its own shift, against ``epg``'s
    operator.

    The semisolid pool carries no transverse magnetization, so it is absent
    from the transverse exchange; what it takes is its share of the voxel,
    which is why the free water's weight here is what every other pool leaves.
    """
    from utils.epg import transverse_relaxation_exchange_op

    leaves = _one_voxel(FOUR, SEMISOLID)
    times = (1e-3, 4e-3, 12e-3)
    measured = _played(
        leaves, _pulse_events(0.0, times, inversion=False), semisolid=True
    )
    fractions = [row[0] for row in FOUR]
    weights = [1.0 - sum(fractions) - SEMISOLID[0], *fractions]
    matrix = _exchange_matrix(weights, [0.0, *(row[1] for row in FOUR)])
    relaxation = torch.tensor([1000.0 / 80.0, *(1000.0 / row[3] for row in FOUR)])
    shifts = torch.tensor([0.0, *(row[4] for row in FOUR)], dtype=torch.float64)
    gain = _gain()

    for index, time in enumerate(times):
        operator = transverse_relaxation_exchange_op(
            matrix.to(torch.complex128),
            relaxation.to(torch.float64),
            torch.tensor(time, dtype=torch.float64),
            shifts,
        )
        expected = (
            complex((operator @ torch.tensor(weights, dtype=torch.complex128)).sum())
            * gain
        )
        assert abs(complex(measured[index]) - expected) / abs(gain) < 2e-5, time


# --- the fast-exchange limit ---


def test_fast_exchange_recovers_at_the_population_average_rate():
    """Pools exchanging far faster than any of them relaxes behave as one pool
    whose relaxation rate is the population-weighted mean of theirs
    (Zimmerman and Brittin, J Phys Chem 61:1328, 1957), which is a closed form
    no table or kernel here computes.
    """
    pools = (
        (0.2, 1e6, 300.0, 40.0, 0.0),
        (0.1, 1e6, 1500.0, 40.0, 0.0),
        (0.15, 1e6, 600.0, 40.0, 0.0),
    )
    leaves = _one_voxel(pools, t1_ms=1000.0, t2_ms=40.0)
    free = 1.0 - sum(row[0] for row in pools)
    rate = free * 1.0 + sum(row[0] * 1000.0 / row[2] for row in pools)
    gain = _gain()

    for delay in (0.1, 0.4, 1.2):
        measured = complex(
            _played(leaves, _pulse_events(delay, inversion=True), semisolid=False)[0]
        )
        expected = (1.0 - 2.0 * math.exp(-rate * delay)) * gain
        assert abs(measured - expected) / abs(gain) < 1e-4, delay


def test_fast_exchange_precesses_at_the_population_average_shift():
    """And the transverse magnetization of such pools precesses at their
    population-weighted mean shift and decays at their mean rate.
    """
    pools = (
        (0.2, 1e6, 1000.0, 30.0, 20.0),
        (0.1, 1e6, 1000.0, 60.0, -15.0),
    )
    leaves = _one_voxel(pools, t2_ms=50.0)
    times = (2e-3, 8e-3, 20e-3)
    measured = _played(
        leaves, _pulse_events(0.0, times, inversion=False), semisolid=False
    )
    free = 1.0 - sum(row[0] for row in pools)
    decay = free * 1000.0 / 50.0 + sum(row[0] * 1000.0 / row[3] for row in pools)
    shift = sum(row[0] * row[4] for row in pools)
    gain = _gain()

    for index, time in enumerate(times):
        expected = (
            gain
            * complex(math.exp(-decay * time))
            * complex(
                math.cos(2.0 * math.pi * shift * time),
                -math.sin(2.0 * math.pi * shift * time),
            )
        )
        assert abs(complex(measured[index]) - expected) / abs(gain) < 1e-4, time


# --- through the public API ---


FSE_ECHOES = 8
BASE = dict(
    t1_ms=1000.0,
    t2_ms=80.0,
    bound_fraction=0.1,
    bound_exchange_hz=30.0,
    t1_bound_ms=1000.0,
    pool_b_fraction=0.1,
    pool_b_exchange_hz=40.0,
    t1_pool_b_ms=300.0,
    t2_pool_b_ms=25.0,
    pool_b_shift_hz=-420.0,
)
EXTRA = dict(
    c=dict(fraction=0.05, exchange_hz=200.0, t1=900.0, t2=40.0, shift_hz=100.0),
    d=dict(fraction=0.03, exchange_hz=1000.0, t1=700.0, t2=30.0, shift_hz=300.0),
    e=dict(fraction=0.02, exchange_hz=500.0, t1=800.0, t2=20.0, shift_hz=-150.0),
)


def _pool(letter: str, **overrides) -> dict:
    values = {**EXTRA[letter], **overrides}
    return {
        f"pool_{letter}_fraction": values["fraction"],
        f"pool_{letter}_exchange_hz": values["exchange_hz"],
        f"t1_pool_{letter}_ms": values["t1"],
        f"t2_pool_{letter}_ms": values["t2"],
        f"pool_{letter}_shift_hz": values["shift_hz"],
    }


def _signal(
    properties: dict, *, live: tuple[str, ...] = ()
) -> tuple[torch.Tensor, dict]:
    """An FSE train through the engine, with ``live`` properties differentiated."""
    tensors = {
        name: torch.tensor([float(value)], requires_grad=name in live)
        for name, value in properties.items()
    }
    description = fse_description(
        torch.deg2rad(torch.full((FSE_ECHOES,), 140.0)),
        echo_spacing_s=5e-3,
        phases_rad=torch.pi / 2,
    )
    signal = (
        EpgEngine()
        .simulate(description, TissueProperties(**tensors), nstates=STATES)
        .signal
    )
    return signal, tensors


def test_every_exchanging_pool_reaches_the_answer():
    """Each of pools C to E moves the train when it is added to the others."""
    every = {**BASE, **_pool("c"), **_pool("d"), **_pool("e")}
    reference = _signal(every)[0]

    for letter in "cde":
        emptied = {**every, **_pool(letter, fraction=0.0)}
        change = _relative(_signal(emptied)[0], reference)
        assert change > 1e-3, letter


def test_an_empty_pool_leaves_the_answer_without_it():
    """A pool of zero fraction is decoupled from the free water, so carrying it
    is the tissue without it -- to the table's round-off, since the operators
    it carries are formed over one more pool.
    """
    without = _signal({**BASE, **_pool("c")})[0]
    carried = _signal(
        {**BASE, **_pool("c"), **_pool("d", fraction=0.0)}, live=("pool_d_fraction",)
    )[0]

    assert _relative(carried, without) < 2e-6


def test_fractions_past_the_whole_voxel_are_refused():
    """The free water is what every other pool leaves, so fractions summing past
    one would start it at a negative magnetization.
    """
    past = {
        **BASE,
        **_pool("c", fraction=0.3),
        **_pool("d", fraction=0.3),
        **_pool("e", fraction=0.3),
    }

    with pytest.raises(ValueError, match="cannot sum past one"):
        _signal(past)


@pytest.mark.parametrize("letter", ("c", "d", "e"))
def test_a_gradient_reaches_every_property_of_every_pool(letter):
    """Through the tables and back to the tissue the caller wrote, checked
    against central differences of the engine's own forward pass.
    """
    every = {**BASE, **_pool("c"), **_pool("d"), **_pool("e")}
    names = tuple(_pool(letter))
    signal, tensors = _signal(every, live=names)
    weight = torch.linspace(0.5, 1.5, signal.numel()).reshape(signal.shape)
    (weight * signal.abs()).sum().backward()

    for name in names:
        step = 1e-2 * max(abs(every[name]), 1.0)
        ahead = (weight * _signal({**every, name: every[name] + step})[0].abs()).sum()
        behind = (weight * _signal({**every, name: every[name] - step})[0].abs()).sum()
        difference = float(ahead - behind) / (2.0 * step)
        gradient = float(tensors[name].grad)
        assert abs(gradient - difference) <= 2e-2 * abs(difference) + 1e-6, name
