"""The Jacobian shape contract must not drift.

One directional derivative per differentiated property covers every voxel, so
what a caller sees is a parameter axis that a single name collapses and a
sequence of names keeps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsim
from torchsim.model import Simulator

FLIP = np.ones(7) * 120.0
ECHOES = 7


def _fse(t1, t2, diff):
    return torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=t1, T2=t2, diff=diff)


VECTOR_T1 = torch.tensor([800.0, 1000.0, 1200.0])
VECTOR_T2 = torch.tensor([50.0, 70.0, 90.0])


@pytest.mark.parametrize(
    ("t1", "t2", "diff", "signal_shape", "jacobian_shape"),
    [
        (1000.0, 80.0, "T2", (ECHOES,), (ECHOES,)),
        (VECTOR_T1, VECTOR_T2, "T2", (3, ECHOES), (3, ECHOES)),
        (1000.0, 80.0, ("T1", "T2"), (ECHOES,), (2, ECHOES)),
        (VECTOR_T1, VECTOR_T2, ("T1", "T2"), (3, ECHOES), (3, 2, ECHOES)),
        (VECTOR_T1, VECTOR_T2, ("T1", "T2", "B1"), (3, ECHOES), (3, 3, ECHOES)),
    ],
)
def test_jacobian_shapes(t1, t2, diff, signal_shape, jacobian_shape) -> None:
    """A bare string ``diff`` collapses the parameter axis; a tuple keeps it."""
    signal, jacobian = _fse(t1, t2, diff)
    assert signal.shape == signal_shape
    assert jacobian.shape == jacobian_shape


def test_jacobian_matches_finite_differences() -> None:
    """Forward mode is exact where central differences are only close."""
    step = 1e-2
    _, jacobian = _fse(VECTOR_T1, VECTOR_T2, "T2")
    forward = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2 + step)
    backward = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2 - step)
    numerical = (forward - backward) / (2.0 * step)
    assert torch.allclose(jacobian, numerical, atol=1e-4)


def test_signal_matches_undifferentiated_call() -> None:
    """The primal returned alongside the Jacobian is the ordinary signal."""
    signal, _ = _fse(VECTOR_T1, VECTOR_T2, "T2")
    plain = torchsim.fse_sim(flip=FLIP, ESP=5.0, T1=VECTOR_T1, T2=VECTOR_T2)
    assert torch.equal(signal, plain)


def test_the_answer_comes_back_in_the_array_library_it_was_asked_in() -> None:
    """NumPy in, NumPy out -- and the same numbers either way.

    The tissue decides, since the signal is per voxel. A caller who works in
    NumPy never sees a torch tensor, and never converts one.
    """
    numpy_t1 = VECTOR_T1.numpy()
    numpy_t2 = VECTOR_T2.numpy()
    signal, jacobian = torchsim.fse_sim(
        flip=FLIP, ESP=5.0, T1=numpy_t1, T2=numpy_t2, diff="T2"
    )
    assert isinstance(signal, np.ndarray)
    assert isinstance(jacobian, np.ndarray)

    same, _ = _fse(VECTOR_T1, VECTOR_T2, "T2")
    assert np.array_equal(signal, same.numpy())


class _DecayModel(Simulator):
    """The smallest model there is: one property, one closed form."""

    properties = ("decay",)

    def evaluate(self, properties, *, times):
        return torch.exp(-times / properties["decay"][..., None])


def test_a_model_of_ones_own_gets_the_same_contract() -> None:
    """Nothing in the contract is particular to the shipped models."""
    model = _DecayModel()
    signal, jacobian = model.jacobian(
        "decay",
        decay=torch.tensor([50.0, 100.0, 200.0]),
        times=torch.linspace(0.0, 100.0, ECHOES),
    )

    assert signal.shape == (3, ECHOES)
    assert jacobian.shape == (3, ECHOES)
    times = torch.linspace(0.0, 100.0, ECHOES)
    decay = torch.tensor([50.0, 100.0, 200.0])[:, None]
    expected = times / decay.square() * torch.exp(-times / decay)
    assert torch.allclose(jacobian, expected, atol=1e-4)
