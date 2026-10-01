"""The Triton kernels for tabulated pools, held to the C++ ones in the interpreter.

``TRITON_INTERPRET=1`` runs a kernel in Python over host tensors, so the
device kernels meet the host ones on the same buffers: the tables, the
trajectory the adjoint records and walks back, the per-problem cotangent slots
and every rotation mode. The C++ kernels are held to the state machine written
out in torch by ``tests/sequence/test_many_pools.py``; these hold the Triton
kernels to them, to float32 round-off.

Run as ``python -m utils.interpreted_pools <pass> <pools> <rotation>
[variant ...]`` with ``TRITON_INTERPRET=1`` set before the process starts.
``<pools>`` is the exchanging pool count followed by ``s`` for a semisolid pool
or ``f`` for none. ``trains`` packs two trains of different lengths and
flips, ``chunked`` records one problem at a time, and ``undeclared`` turns
off every optional term the tissue does not declare.
"""

from __future__ import annotations

import sys

import torch

from sequence.test_many_pools import (
    _PAIR_ROW,
    GEOMETRY,
    OUTPUTS,
    STATES,
    _case,
    _events,
    _extras,
    _instantaneous_table,
    _tissue,
)
from torchsim.sequence import _accelerators, _pools, _pools_triton
from torchsim.sequence._lineshape import lineshape_table
from torchsim.sequence._parameters import NO_GEOMETRY
from torchsim.sequence._transition import DynamicPairs

# What one kernel is held to against the other, relative to the largest entry
# of each result. Both accumulate in float32, in different orders.
TOLERANCE = 2e-5


def _launch(pass_name: str, pools: str, rotation: str, variants: set[str]) -> float:
    exchanging, semisolid = int(pools[0]), pools[1] == "s"
    leaves = _case(exchanging, semisolid, rotation)
    tissue = tuple(value.float().contiguous() for value in _tissue(leaves))
    events = _events(leaves, rotation)
    trains = 2 if "trains" in variants else 1
    if trains > 1:
        stretch = torch.tensor([[1.0], [1.3]])
        events = (
            (events[0] * stretch).contiguous(),
            events[1],
            (events[2] * stretch).contiguous(),
            (events[3] + 0.1 * stretch).contiguous(),
            *events[4:],
        )
    if "chunked" in variants:
        # Less than one problem's trajectory, so every problem is a wave.
        _pools_triton._TRAJECTORY_BUDGET_BYTES = 1
    features = frozenset() if "undeclared" in variants else None
    geometry = NO_GEOMETRY if "undeclared" in variants else GEOMETRY

    following = pass_name in ("jvp", "vjp_jvp")
    # An adjoint differentiating the durations reads the tables' slopes.
    slopes = pass_name != "forward"
    values, index, layout = _pools.tabulate(
        tissue,
        _extras(leaves["pools"]) if exchanging > 1 else None,
        events[0],
        exchanging=exchanging,
        semisolid=semisolid,
        slopes=slopes,
    )
    generator = torch.Generator().manual_seed(5)
    tables = _pools.PoolTables(
        layout=layout,
        values=values,
        index=index,
        direction=torch.randn(values.shape, generator=generator) * 0.01
        if following
        else None,
    )
    dynamic = None
    dynamic_direction = None
    if "pairs" in leaves:
        dynamic = DynamicPairs.from_packed(
            leaves["pairs"], torch.tensor(_PAIR_ROW, dtype=torch.int32)
        )
        if following:
            dynamic_direction = (
                torch.randn(leaves["pairs"].shape, generator=generator) * 0.1
            )
    profile = _accelerators._tables(
        _instantaneous_table() if rotation == "profile" else None, events, dynamic
    )
    lineshape = lineshape_table() if semisolid else None
    tissue_tangents = tuple(
        torch.randn(value.shape, generator=generator)
        * (0.01 * value.abs().mean() + 1e-3)
        for value in tissue
    )
    event_tangents = tuple(
        torch.randn(events[at].shape, generator=generator) * scale
        for at, scale in ((0, 1e-4), (2, 0.05), (3, 0.05))
    )
    seed = torch.randn(
        (3, OUTPUTS) if trains == 1 else (trains, 3, OUTPUTS),
        dtype=torch.complex64,
        generator=generator,
    )
    shared = dict(
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        features=features,
        pools=tables,
    )

    if pass_name == "forward":
        host = (
            _accelerators._run_packed(
                tissue, events, STATES, OUTPUTS, 1, exchanging=layout, **shared
            ),
        )
        device = (
            _pools_triton.simulate(
                tissue, events, state_count=STATES, output_count=OUTPUTS, **shared
            ),
        )
    elif pass_name == "jvp":
        host = (
            _accelerators._run_packed_jvp(
                tissue,
                events,
                tissue_tangents,
                event_tangents,
                STATES,
                OUTPUTS,
                1,
                exchanging=layout,
                dynamic_direction=dynamic_direction,
                **shared,
            ),
        )
        device = (
            _pools_triton.simulate_jvp(
                tissue,
                events,
                tissue_tangents,
                event_tangents,
                state_count=STATES,
                output_count=OUTPUTS,
                dynamic_direction=dynamic_direction,
                **shared,
            ),
        )
    elif pass_name == "vjp":
        host = _accelerators._run_packed_vjp(
            tissue, events, seed, STATES, OUTPUTS, 1, exchanging=layout, **shared
        )
        device = _pools_triton.simulate_vjp(
            tissue, events, seed, state_count=STATES, output_count=OUTPUTS, **shared
        )
    else:
        tangents = (*tissue_tangents, *event_tangents)
        host = sum(
            _accelerators._run_packed_vjp_jvp(
                tissue,
                events,
                tangents,
                seed,
                STATES,
                OUTPUTS,
                1,
                exchanging=layout,
                dynamic_direction=dynamic_direction,
                **shared,
            ),
            (),
        )
        device = sum(
            _pools_triton.simulate_vjp_jvp(
                tissue,
                events,
                tangents,
                seed,
                state_count=STATES,
                output_count=OUTPUTS,
                dynamic_direction=dynamic_direction,
                **shared,
            ),
            (),
        )

    assert len(host) == len(device), f"{len(host)} results against {len(device)}"
    worst = 0.0
    for position, (expected, measured) in enumerate(zip(host, device, strict=True)):
        expected = expected.reshape(-1)
        measured = measured.reshape(-1)
        assert expected.numel() == measured.numel(), (
            f"result {position}: {expected.numel()} entries against {measured.numel()}"
        )
        if not expected.numel():
            continue
        scale = max(float(expected.abs().max()), 1e-6)
        worst = max(worst, float((expected - measured).abs().max()) / scale)
    # A comparison of two zeros is no comparison at all.
    assert any(float(value.abs().max()) > 0.0 for value in host if value.numel())
    return worst


if __name__ == "__main__":
    pass_name, pools, rotation, *rest = sys.argv[1:]
    worst = _launch(pass_name, pools, rotation, set(rest))
    print(f"{pass_name} {pools} {rotation} {' '.join(rest)}: {worst:.2e}")
    assert worst <= TOLERANCE, f"{worst:.2e} past {TOLERANCE:.0e}"
    print("checked")
