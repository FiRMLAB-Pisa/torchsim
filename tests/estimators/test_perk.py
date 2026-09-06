"""Tests for the PERK estimator."""

from __future__ import annotations

import math

import pytest
import torch

from torchsim.estimators import PERK, DictionaryMatcher
from torchsim.model import Simulator
from torchsim.simulators import MultiEchoSimulator


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_perk_learns_smooth_nonlinear_inverse(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(4)
    train_parameter = 0.1 + 2.8 * torch.rand(
        2048, 1, generator=generator, device=device
    )
    train_signal = torch.cat(
        (torch.sin(train_parameter), torch.cos(train_parameter)), dim=-1
    )
    test_parameter = torch.linspace(0.15, 2.85, 128, device=device)[:, None]
    test_signal = torch.cat(
        (torch.sin(test_parameter), torch.cos(test_parameter)), dim=-1
    )
    estimator = PERK(
        n_features=256,
        regularization=1e-5,
        chunk_size=257,
        feature_seed=8,
    ).to(device)

    estimator.fit(signals=train_signal, parameters=train_parameter)
    actual = estimator(test_signal)

    assert torch.mean((actual - test_parameter) ** 2) < 2e-4


def test_perk_accepts_complex_signals_and_known_parameters() -> None:
    parameter = torch.linspace(0.1, 1.0, 512)[:, None]
    known = torch.linspace(0.8, 1.2, 512)[:, None]
    signal = known * torch.exp(1j * parameter)
    estimator = PERK(n_features=128, regularization=1e-5, feature_seed=2)

    estimator.fit(signals=signal, parameters=parameter, known=known)
    actual = estimator(signal[:16], known[:16])

    assert actual.shape == (16, 1)
    torch.testing.assert_close(actual, parameter[:16], atol=2e-2, rtol=2e-2)


def test_perk_estimation_is_differentiable() -> None:
    parameter = torch.linspace(0.1, 1.0, 256)[:, None]
    signal = torch.cat((parameter, parameter.square()), dim=-1)
    estimator = PERK(n_features=64, regularization=1e-4, feature_seed=1).fit(
        signals=signal, parameters=parameter
    )
    measured = signal[:8].clone().requires_grad_()

    estimator(measured).sum().backward()

    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()


class _Counted(Simulator):
    """A model whose signal is its property and its square, and that counts."""

    properties = ("x",)

    def __init__(self, seen: list[int]) -> None:
        super().__init__()
        self._seen = seen

    def evaluate(self, properties, **sequence):
        values = properties["x"].reshape(-1, 1)
        self._seen.append(values.shape[0])
        return torch.cat((values, values.square()), dim=-1)


def test_a_streaming_fit_chunks_generation() -> None:
    """Memory follows the chunk, and the dictionary is never retained."""
    parameter = torch.linspace(0.1, 1.0, 65)
    seen: list[int] = []

    estimator = PERK(
        _Counted(seen), n_features=32, feature_seed=1, stream=True, uncertainty=False
    ).fit(x=parameter, chunk=16)

    assert estimator.fitted
    # One pass to read the kernel width off the inputs, one to fit.
    assert seen == [16, 16, 16, 16, 1] * 2


def test_a_given_length_scale_costs_one_pass() -> None:
    """The kernel width is the only reason to look at the data twice.

    Nothing can be accumulated until the random features exist, and they
    cannot be drawn until the width is known. Say what it is and the training
    set is walked once -- which for a source that simulates is the difference
    between simulating it once and simulating it twice.
    """
    parameter = torch.linspace(0.1, 1.0, 65)
    seen: list[int] = []

    estimator = PERK(
        _Counted(seen),
        n_features=32,
        feature_seed=1,
        length_scale=0.5,
        stream=True,
        uncertainty=False,
    ).fit(x=parameter, chunk=16)

    assert estimator.fitted
    assert seen == [16, 16, 16, 16, 1]


def test_learning_what_the_answer_is_wrong_by_costs_a_second_pass() -> None:
    """Because the residual is not known until the first solve is done.

    That is the whole price of the error bar, and it is paid at training: what
    :meth:`map` then does to report one is a matrix multiply, not a rerun.
    """
    parameter = torch.linspace(0.1, 1.0, 65)
    seen: list[int] = []

    PERK(
        _Counted(seen), n_features=32, feature_seed=1, length_scale=0.5, stream=True
    ).fit(x=parameter, chunk=16)

    assert seen == [16, 16, 16, 16, 1] * 2


def test_the_merged_pass_gives_the_covariance_it_would_have_centred() -> None:
    """Means and products come out of one pass, not a pass each.

    A covariance can be accumulated centred, which needs the mean first, or as
    a raw second moment the mean is subtracted from afterwards. The second
    reads the data once. This asserts the two agree where it matters -- in the
    weights that come out.
    """
    generator = torch.Generator().manual_seed(0)
    signals = torch.randn(2000, 12, generator=generator)
    parameters = torch.stack((signals.square().sum(-1), signals.abs().mean(-1)), dim=-1)
    estimator = PERK(n_features=64, feature_seed=3).fit(
        signals=signals, parameters=parameters
    )

    features = _reference_features(estimator, signals).to(torch.float64)
    targets = parameters.to(torch.float64)
    centred = features - features.mean(0)
    covariance = centred.mT @ centred / (signals.shape[0] - 1)
    covariance.diagonal().add_(estimator.regularization)
    cross = (targets - targets.mean(0)).mT @ centred / (signals.shape[0] - 1)
    expected = torch.linalg.solve(covariance, cross.mT).mT

    assert torch.allclose(estimator.weight, expected.to(torch.float32), rtol=1e-4)


def _reference_features(estimator, signals):
    """The feature map, written out rather than called."""
    scale = math.sqrt(2.0 / estimator.frequency.shape[0])
    return scale * torch.cos(signals @ estimator.frequency.mT + estimator.phase)


def test_streaming_reaches_what_holding_the_dictionary_reaches() -> None:
    """The flag says where the training set lives, not what is learnt from it."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))
    unknown = {"T2": (10.0, 300.0), "M0": (0.5, 1.5)}
    measured = acquisition.simulate(
        T2=torch.tensor([40.0, 90.0]), M0=torch.tensor([1.0, 1.2])
    )

    held = PERK(acquisition, n_features=256, feature_seed=3).fit(
        unknown, seed=0, samples=2048
    )
    streamed = PERK(acquisition, n_features=256, feature_seed=3, stream=True).fit(
        unknown, seed=0, samples=2048
    )

    for name in unknown:
        assert torch.allclose(held.map(measured)[name], streamed.map(measured)[name])


def test_a_streaming_fit_cannot_be_asked_for_a_basis() -> None:
    """A basis is read off the whole dictionary, which streaming never holds."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))

    with pytest.raises(ValueError, match="streaming fit never holds"):
        PERK(acquisition, n_features=32, stream=True).fit(
            T2=(10.0, 300.0), seed=0, rank=4, samples=64
        )


def test_the_basis_is_on_the_estimator_whichever_method_fitted_it() -> None:
    """A subspace reconstruction is handed the basis and gives back the
    coefficients, so both ends have to agree on which basis that is."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))
    unknown = {"T2": (10.0, 300.0), "M0": (0.5, 1.5)}
    truth = {"T2": torch.tensor([40.0, 90.0]), "M0": torch.tensor([1.0, 1.2])}
    measured = acquisition.simulate(**truth)

    regression = PERK(acquisition, n_features=256, feature_seed=3).fit(
        unknown, seed=0, rank=4, samples=2048
    )
    matcher = DictionaryMatcher(acquisition).fit(unknown, seed=0, rank=4, samples=2048)

    assert regression.subspace.rank == matcher.subspace.rank == 4
    assert torch.allclose(regression.subspace.basis, matcher.subspace.basis, atol=1e-5)

    # What a subspace reconstruction returns is already in the basis, so it is
    # read back without being projected a second time.
    coefficients = regression.subspace.project(measured)
    maps = regression.from_coefficients(coefficients)

    for name, value in truth.items():
        assert torch.allclose(maps[name], value, rtol=0.1)


def test_a_basis_fitted_elsewhere_can_be_worked_in() -> None:
    """The reconstruction and the estimator reading its coefficients have to
    agree on the basis, and the way to be sure is to hand over the same one."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))
    unknown = dict(T2=(10.0, 300.0), M0=(0.5, 1.5))
    truth = {"T2": torch.tensor([40.0, 90.0]), "M0": torch.tensor([1.0, 1.2])}
    measured = acquisition.simulate(**truth)

    fitted_here = DictionaryMatcher(acquisition).fit(
        **unknown, seed=0, rank=4, samples=2048
    )
    borrowed = PERK(acquisition, n_features=256, feature_seed=3).fit(
        **unknown, seed=0, subspace=fitted_here.subspace, samples=2048
    )

    assert borrowed.subspace is fitted_here.subspace
    maps = borrowed.map(measured)
    for name, value in truth.items():
        assert torch.allclose(maps[name], value, rtol=0.1)


def test_a_borrowed_basis_streams() -> None:
    """Streaming cannot fit a basis, but it can be given one: each chunk is
    projected as it is simulated, so no dictionary is held either way."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))
    unknown = dict(T2=(10.0, 300.0), M0=(0.5, 1.5))
    measured = acquisition.simulate(
        T2=torch.tensor([40.0, 90.0]), M0=torch.tensor([1.0, 1.2])
    )
    basis = (
        DictionaryMatcher(acquisition)
        .fit(**unknown, seed=0, rank=4, samples=2048)
        .subspace
    )

    held = PERK(acquisition, n_features=256, feature_seed=3).fit(
        **unknown, seed=0, subspace=basis, samples=2048
    )
    streamed = PERK(acquisition, n_features=256, feature_seed=3, stream=True).fit(
        **unknown, seed=0, subspace=basis, samples=2048
    )

    for name in unknown:
        assert torch.allclose(
            held.map(measured)[name], streamed.map(measured)[name], atol=1e-3
        )


def test_a_rank_and_a_basis_are_alternatives() -> None:
    """One asks for a basis to be fitted, the other supplies one."""
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 12))
    basis = (
        DictionaryMatcher(acquisition)
        .fit(T2=(10.0, 300.0), seed=0, rank=3, samples=256)
        .subspace
    )

    with pytest.raises(ValueError, match="not both"):
        PERK(acquisition, n_features=32).fit(
            T2=(10.0, 300.0), seed=0, rank=3, subspace=basis, samples=64
        )
