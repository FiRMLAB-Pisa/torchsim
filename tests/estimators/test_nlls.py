"""Levenberg-Marquardt over a whole volume at once."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

import torchsim.recon._operator as _operator
from torchsim.estimators import NonlinearLeastSquares
from torchsim.simulators import (
    FSESimulator,
    InversionRecoverySimulator,
    MultiEchoSimulator,
)

TE_MS = torch.tensor([10.0, 20.0, 30.0, 50.0, 80.0, 120.0])
TI_MS = torch.tensor([50.0, 200.0, 600.0, 1200.0, 2500.0])


def multiecho_mapping(**settings):
    """A T2/M0 fit from a multi-echo decay, trained and ready."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    mapping = NonlinearLeastSquares(acquisition, **settings)
    mapping.fit(T2=(5.0, 300.0), M0=(0.2, 2.0), seed=0, samples=256)
    return acquisition, mapping


def test_a_noiseless_fit_recovers_what_made_the_signal() -> None:
    """Two parameters, four voxels, no noise: the answer is the truth."""
    acquisition, mapping = multiecho_mapping(
        bounds={"T2": (1.0, 1000.0), "M0": (0.0, 5.0)}
    )
    T2 = torch.tensor([25.0, 45.0, 80.0, 150.0])
    M0 = torch.tensor([1.0, 0.7, 1.3, 0.9])

    maps = mapping(acquisition.simulate(T2=T2, M0=M0))

    for name, truth in (("T2", T2), ("M0", M0)):
        error = ((maps[name].flatten() - truth) / truth).abs().max()
        assert float(error) < 1e-4, f"{name}: {float(error):.2e}"


def test_it_finds_what_scipy_finds() -> None:
    """The same minimum a trusted solver reaches, voxel by voxel.

    ``scipy.optimize.least_squares`` solves one voxel at a time with its own
    trust region; agreeing with it says the batched stepping has not changed
    where the solver lands, only how many land at once.
    """
    acquisition, mapping = multiecho_mapping(
        bounds={"T2": (1.0, 1000.0), "M0": (0.0, 5.0)}
    )
    T2 = torch.tensor([25.0, 45.0, 80.0, 150.0])
    M0 = torch.tensor([1.0, 0.7, 1.3, 0.9])
    measured = acquisition.simulate(T2=T2, M0=M0)

    maps = mapping(measured)

    echoes = TE_MS.numpy().astype(np.float64)
    for index in range(len(T2)):
        observed = measured[index].numpy().astype(np.float64)
        alone = least_squares(
            lambda p, observed=observed: p[1] * np.exp(-echoes / p[0]) - observed,
            x0=[50.0, 1.0],
            bounds=([1.0, 0.0], [1000.0, 5.0]),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        assert abs(float(maps["T2"][index]) - alone.x[0]) < 1e-3
        assert abs(float(maps["M0"][index]) - alone.x[1]) < 1e-5


def test_no_iterate_ever_leaves_the_bounds(monkeypatch) -> None:
    """A bound is kept by construction, not by clipping the result.

    Asserting only on what comes back would pass for a solver that wandered
    outside and was pulled back, which is what makes a bound sit exactly on
    the answer. Every parameter the model is evaluated at is recorded instead.
    """
    low, high = 20.0, 60.0
    acquisition = MultiEchoSimulator(TE=TE_MS)
    mapping = NonlinearLeastSquares(
        acquisition, bounds={"T2": (low, high), "M0": (0.0, 5.0)}
    ).fit(T2=(low, high), M0=(0.2, 2.0), seed=0, samples=256)
    seen: list[torch.Tensor] = []
    natural = _operator.to_natural
    monkeypatch.setattr(
        _operator,
        "to_natural",
        lambda free, bounds, names: (
            seen.append(out := natural(free, bounds, names)) or out
        ),
    )

    # A truth well outside the bound, so the fit pushes against it throughout.
    mapping(acquisition.simulate(T2=torch.tensor([200.0, 5.0]), M0=torch.ones(2)))

    assert seen, "the model was never evaluated"
    for values in seen:
        column = values[:, 0]
        assert float(column.min()) >= low
        assert float(column.max()) <= high


def test_a_starting_point_on_a_bound_is_refused() -> None:
    """A value sitting exactly on a bound has no unconstrained image.

    The transformed variable is infinite there, so the curvature vanishes and
    the first step is unbounded -- the fit would walk to the far bound and
    stay. Saying so where the value is written beats converging on nonsense.
    """
    acquisition = MultiEchoSimulator(TE=TE_MS)
    with pytest.raises(ValueError, match="strictly inside"):
        NonlinearLeastSquares(
            acquisition, bounds={"T2": (10.0, 200.0)}, initial={"T2": 10.0}
        ).fit(T2=(5.0, 300.0), seed=0, samples=64)


def test_a_starting_point_just_inside_a_bound_converges() -> None:
    """Close to a bound is fine; on it is not."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    mapping = NonlinearLeastSquares(
        acquisition, bounds={"T2": (10.0, 200.0)}, initial={"T2": 12.0}
    ).fit(T2=(5.0, 300.0), seed=0, samples=64)
    truth = torch.tensor([45.0, 90.0])

    got = mapping(acquisition.simulate(T2=truth))["T2"]

    assert float((got.flatten() - truth).abs().max()) < 1e-2


def test_an_unbounded_fit_is_the_same_fit() -> None:
    """Bounds are a way of writing the problem, not a different problem."""
    acquisition, bounded = multiecho_mapping(
        bounds={"T2": (1.0, 1000.0), "M0": (0.0, 5.0)}
    )
    _, plain = multiecho_mapping()
    measured = acquisition.simulate(
        T2=torch.tensor([25.0, 80.0]), M0=torch.tensor([1.0, 1.3])
    )

    with_bounds = bounded(measured)
    without = plain(measured)

    for name in ("T2", "M0"):
        torch.testing.assert_close(
            with_bounds[name], without[name], atol=1e-3, rtol=1e-4
        )


def test_a_complex_contrast_is_two_real_residuals() -> None:
    """What is minimized is the squared modulus, so both parts count."""
    flip = torch.full((16,), 150.0)
    acquisition = FSESimulator(ESP=8.0, TR=2000.0, flip=flip)
    mapping = NonlinearLeastSquares(
        acquisition, bounds={"T1": (1.0, 5000.0), "T2": (1.0, 500.0)}
    ).fit(T1=(300.0, 2500.0), T2=(20.0, 200.0), seed=0, samples=256)
    T1 = torch.tensor([500.0, 900.0, 1400.0, 2000.0])
    T2 = torch.tensor([40.0, 60.0, 90.0, 150.0])

    measured = acquisition.simulate(T1=T1, T2=T2)
    maps = mapping(measured)

    assert torch.is_complex(measured)
    for name, truth in (("T1", T1), ("T2", T2)):
        error = ((maps[name].flatten() - truth) / truth).abs().max()
        assert float(error) < 1e-3, f"{name}: {float(error):.2e}"


def test_a_property_measured_separately_reaches_the_model() -> None:
    """A known map varies per voxel and is not fitted, but is not ignored."""
    acquisition = InversionRecoverySimulator(TI=TI_MS)
    mapping = NonlinearLeastSquares(acquisition, bounds={"T1": (1.0, 6000.0)}).fit(
        T1=(200.0, 3000.0),
        known={"inv_efficiency": (0.7, 1.0)},
        seed=0,
        samples=256,
    )
    T1 = torch.tensor([600.0, 1100.0, 1900.0])
    efficiency = torch.tensor([0.75, 0.9, 1.0])

    measured = acquisition.simulate(T1=T1, inv_efficiency=efficiency)
    got = mapping(measured, known={"inv_efficiency": efficiency})["T1"]

    assert float(((got.flatten() - T1) / T1).abs().max()) < 1e-3


def test_a_subspace_is_applied_to_the_prediction_too() -> None:
    """The residual is taken where the measurement lives.

    A mapping with a rank projects the measurements, so a model that answered
    in full contrasts would be subtracting two different things.
    """
    acquisition = MultiEchoSimulator(TE=TE_MS)
    mapping = NonlinearLeastSquares(
        acquisition, bounds={"T2": (1.0, 1000.0), "M0": (0.0, 5.0)}
    ).fit(T2=(5.0, 300.0), M0=(0.2, 2.0), rank=3, seed=0, samples=256)
    T2 = torch.tensor([25.0, 80.0])
    M0 = torch.tensor([1.0, 1.3])

    maps = mapping(acquisition.simulate(T2=T2, M0=M0))

    assert mapping.subspace.rank == 3
    for name, truth in (("T2", T2), ("M0", M0)):
        error = ((maps[name].flatten() - truth) / truth).abs().max()
        assert float(error) < 1e-2, f"{name}: {float(error):.2e}"


def test_converged_voxels_stop_being_solved() -> None:
    """Late iterations cost what is left, not what was started with."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    method = NonlinearLeastSquares(acquisition, bounds={"T2": (1.0, 1000.0)})
    method.fit(T2=(5.0, 300.0), seed=0, samples=128)
    mapping = method

    mapping(acquisition.simulate(T2=torch.full((64,), 50.0)))

    assert method.unconverged == 0
    assert 0 < method.iterations <= method.loop.max_iterations


def test_the_map_is_shaped_like_the_volume() -> None:
    """A volume in, one map per unknown out."""
    acquisition, mapping = multiecho_mapping(bounds={"T2": (1.0, 1000.0)})
    volume = acquisition.simulate(T2=torch.full((24,), 60.0)).reshape(4, 6, -1)

    maps = mapping(volume)

    assert set(maps) == {"T2", "M0"}
    assert maps["T2"].shape == (4, 6)


def test_a_flat_voxel_does_not_stop_the_others() -> None:
    """One voxel whose normal equations are singular is not a failed volume."""
    acquisition, mapping = multiecho_mapping(bounds={"T2": (1.0, 1000.0)})
    measured = acquisition.simulate(T2=torch.tensor([50.0, 90.0]))
    measured = torch.cat((torch.zeros(1, len(TE_MS)), measured), dim=0)

    maps = mapping(measured)

    assert torch.isfinite(maps["T2"]).all()
    assert abs(float(maps["T2"][1]) - 50.0) < 1e-2


# %% what the caller has to write themselves


def test_an_equality_constraint_is_written_into_the_model() -> None:
    """Two fractions summing to one, imposed by there being one unknown.

    Writing water as ``1 - f`` removes the degree of freedom rather than
    restoring it after each step, so the constraint holds at every iterate
    and not merely at the answer.
    """
    from torchsim.model import Simulator

    class FatWater(Simulator):
        """Two species at their own frequencies, in known proportion."""

        properties = ("fat_fraction", "M0")

        def evaluate(self, properties, *, TE):
            fraction = properties["fat_fraction"].reshape(-1, 1)
            density = properties.get("M0", torch.ones(1)).reshape(-1, 1)
            turn = 2j * torch.pi * -420.0 * TE * 1e-3
            # Water is what fat is not: the sum is one by construction.
            return density * ((1.0 - fraction) + fraction * torch.exp(turn))

    echoes = torch.tensor([1.2, 2.4, 3.6, 4.8, 6.0, 7.2])
    acquisition = FatWater(TE=echoes)
    mapping = NonlinearLeastSquares(
        acquisition, bounds={"fat_fraction": (0.0, 1.0), "M0": (0.0, 3.0)}
    ).fit(fat_fraction=(0.0, 1.0), M0=(0.5, 1.5), seed=0, samples=256)
    truth = torch.tensor([0.05, 0.3, 0.7])

    got = mapping(acquisition.simulate(fat_fraction=truth, M0=torch.ones(3)))

    fraction = got["fat_fraction"].flatten()
    assert float((fraction - truth).abs().max()) < 1e-3
    # The water fraction is never stated, so it cannot disagree.
    assert bool(((fraction >= 0.0) & (fraction <= 1.0)).all())


# %% what it refuses


def test_a_bound_on_something_not_being_estimated_is_a_mistake() -> None:
    """A misspelt name would otherwise be silently ignored."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    with pytest.raises(ValueError, match="T3"):
        NonlinearLeastSquares(acquisition, bounds={"T3": (1.0, 2.0)}).fit(
            T2=(5.0, 300.0), seed=0, samples=8
        )
    with pytest.raises(ValueError, match="T3"):
        NonlinearLeastSquares(acquisition, initial={"T3": 1.0}).fit(
            T2=(5.0, 300.0), seed=0, samples=8
        )


def test_a_bound_that_does_not_increase_is_a_mistake() -> None:
    """An empty interval has no inside to fit in."""
    acquisition = MultiEchoSimulator(TE=TE_MS)
    with pytest.raises(ValueError, match="not increasing"):
        NonlinearLeastSquares(acquisition, bounds={"T2": (100.0, 10.0)}).fit(
            T2=(5.0, 300.0), seed=0, samples=8
        )


def test_fitting_without_a_model_says_so() -> None:
    """A method that fits the model needs the model."""
    method = NonlinearLeastSquares()

    assert not method.fitted
    with pytest.raises(RuntimeError, match="no model"):
        method.fit(signals=torch.zeros(4, 6), parameters=torch.zeros(4, 1))
    with pytest.raises(RuntimeError, match="no model"):
        method(torch.zeros(4, 6))


def test_the_solve_is_a_gauss_newton_and_nothing_else_is_here() -> None:
    """The estimator adapts a loop to a protocol; it holds no algorithm.

    Every knob a Levenberg-Marquardt has -- how many steps, how the damping
    moves, what stops a voxel -- lives on the loop, so there is one place to
    set it and one place for it to be wrong.
    """
    from torchsim.recon import GaussNewton, TrustRegion, direct

    default = NonlinearLeastSquares()

    assert isinstance(default.loop, GaussNewton)
    assert isinstance(default.loop.damping, TrustRegion)
    assert default.loop.solve is direct

    given = GaussNewton(TrustRegion(tau=1e-3), solve=direct, max_iterations=60)
    assert NonlinearLeastSquares(loop=given).loop is given


@pytest.mark.parametrize(
    "settings,complaint",
    [
        (dict(max_iterations=0), "max_iterations"),
    ],
)
def test_loop_settings_that_make_no_sense(settings, complaint) -> None:
    """Caught where they are written, not where they misbehave."""
    from torchsim.recon import GaussNewton

    with pytest.raises(ValueError, match=complaint):
        GaussNewton(**settings)
