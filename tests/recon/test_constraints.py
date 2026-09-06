"""Saying what a parameter may be, in a reconstruction and in a fit alike."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from torchsim.model import Simulator
from torchsim.recon import (
    GaussNewton,
    ModelOperator,
    Schedule,
    TrustRegion,
    direct,
    iterative,
)
from torchsim.simulators import MultiEchoSimulator

#: ``iterative()`` with nothing named falls back to deepinv's ``least_squares``,
#: which TorchSim does not depend on.
needs_deepinv = pytest.mark.skipif(
    importlib.util.find_spec("deepinv") is None,
    reason="deepinv supplies the fallback inner solve",
)

TE_MS = torch.linspace(1.2, 12.0, 10)
#: Where the main fat peak sits at 3 T, in hertz.
FAT_HZ = -434.0


class FatWater(Simulator):
    """A gradient echo whose water and fat fractions must sum to one.

    Written on the fat fraction alone: water is ``1 - f`` inside the model, so
    the sum is one at every iterate by construction rather than by correction.
    That is how an equality constraint is imposed -- by removing the degree of
    freedom, not by restoring it afterwards.
    """

    properties = ("fat_fraction", "R2s")

    def __init__(self) -> None:
        self.seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def evaluate(self, properties: Mapping[str, Any], *, TE: Any) -> torch.Tensor:
        echo = torch.as_tensor(TE) * 1e-3
        fat = properties["fat_fraction"][..., None]
        water = 1.0 - fat
        self.seen.append((water.squeeze(-1).clone(), fat.squeeze(-1).clone()))
        rate = properties["R2s"][..., None]
        phase = torch.exp(2j * torch.pi * FAT_HZ * echo)
        return (water + fat * phase) * torch.exp(-rate * echo)


# %% box bounds


def test_a_bound_holds_at_every_iterate_of_a_fit() -> None:
    """Not on the answer -- on every point the model is evaluated at.

    Asserting only on what comes back would pass for a solver that wandered
    outside and was pulled back, which is what leaves a bound sitting exactly
    on the answer.
    """
    low, high = 20.0, 60.0
    acquisition = MultiEchoSimulator(TE=TE_MS)
    operator = ModelOperator(acquisition, "T2", bounds={"T2": (low, high)})
    seen: list[torch.Tensor] = []
    watched = _watching(operator, seen)
    # A truth well outside the bound, so the solve pushes against it throughout.
    measured = torch.as_tensor(acquisition.simulate(T2=torch.tensor([200.0, 5.0])))

    GaussNewton(TrustRegion(), solve=direct, max_iterations=30).minimize(
        watched, measured.to(torch.complex64), operator.initial((2,), T2=40.0)
    )

    assert seen, "the model was never evaluated"
    for values in seen:
        assert float(values.min()) >= low
        assert float(values.max()) <= high


@needs_deepinv
def test_a_bound_holds_at_every_iterate_under_an_encoding() -> None:
    """Where it matters more: one voxel out of range spoils every sample.

    A fit that wanders leaves that voxel wrong. A reconstruction evaluates the
    model at every voxel to predict every k-space sample, so a single
    unphysical voxel corrupts the whole residual.
    """
    low, high = 20.0, 120.0
    acquisition = MultiEchoSimulator(TE=TE_MS)
    operator = ModelOperator(acquisition, "T2", bounds={"T2": (low, high)})
    seen: list[torch.Tensor] = []
    watched = _watching(operator, seen)
    truth = torch.full((1, 8, 8), 300.0)
    images = torch.as_tensor(acquisition.simulate(T2=truth)).to(torch.complex64)
    encoding = _Fourier()
    kspace = encoding.A(images.movedim(-1, 1))

    GaussNewton(
        Schedule(initial=1e-3, minimum=1e-6), solve=iterative(), max_iterations=10
    ).minimize(
        watched,
        kspace,
        operator.initial((1, 8, 8), T2=50.0),
        encoding=encoding,
    )

    assert seen
    for values in seen:
        assert float(values.min()) >= low
        assert float(values.max()) <= high


# %% equality constraints


def test_an_equality_constraint_is_the_model_written_on_what_is_free() -> None:
    """One channel for the pair, because there is one degree of freedom.

    The operator never learns about the constraint and never enforces it. The
    model has no way to violate it, which is the point.
    """
    model = FatWater()
    acquisition = model.bind(TE=TE_MS)
    operator = ModelOperator(
        acquisition,
        "fat_fraction",
        "R2s",
        bounds={"fat_fraction": (0.0, 1.0), "R2s": (0.0, 200.0)},
    )

    assert operator.channels == 4  # f, R2s, and the two of the amplitude
    truth = torch.tensor([0.1, 0.35, 0.7])
    measured = torch.as_tensor(
        acquisition.simulate(fat_fraction=truth, R2s=torch.full((3,), 40.0))
    )

    found = GaussNewton(TrustRegion(), solve=direct, max_iterations=60).minimize(
        operator, measured, operator.initial((3,), fat_fraction=0.5, R2s=50.0)
    )

    fraction = operator.split(found.x)["fat_fraction"]
    torch.testing.assert_close(fraction, truth, atol=2e-3, rtol=1e-3)
    # Every point the model was evaluated at sat on the constraint surface.
    for water, fat in model.seen:
        assert bool((water == 1.0 - fat).all())
        assert float(fat.min()) >= 0.0 and float(fat.max()) <= 1.0


def test_the_constraint_is_what_makes_the_pair_identifiable() -> None:
    """Freeing both fractions leaves the amplitude able to absorb their sum.

    Water, fat and a complex scale are three amplitudes for two measurements'
    worth of information, so the recast is not a convenience -- without it the
    problem the solver is handed is a different, degenerate one.
    """
    model = FatWater()
    operator = ModelOperator(
        model.bind(TE=TE_MS),
        "fat_fraction",
        "R2s",
        bounds={"fat_fraction": (0.0, 1.0), "R2s": (0.0, 200.0)},
    )
    x = operator.initial((4,), fat_fraction=0.3, R2s=40.0)

    blocks = operator.jacobian(x)

    rows = torch.cat((blocks.real, blocks.imag), dim=-1)
    curvature = rows @ rows.mT
    # Four unknowns, and the model is written so all four are separable.
    assert int(torch.linalg.matrix_rank(curvature).min()) == operator.channels


# %% helpers


class _Fourier:
    """A fully sampled Cartesian encoding, adjoint exact."""

    def A(self, images: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(images, norm="ortho")

    def A_adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(kspace, norm="ortho")


def _watching(operator: ModelOperator, seen: list) -> ModelOperator:
    """The same operator, recording the property it is evaluated at."""

    class _Watched(type(operator)):
        def _named(self, x):
            given, rho = super()._named(x)
            seen.append(next(iter(given.values())).detach().clone())
            return given, rho

    watched = _Watched.__new__(_Watched)
    watched.__dict__.update(operator.__dict__)
    return watched
