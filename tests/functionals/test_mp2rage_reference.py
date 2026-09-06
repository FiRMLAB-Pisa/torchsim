"""The MP2RAGE closed form, against the signal equation it implements.

The oracle below is Marques' ``MPRAGEfunc`` for two inversion-recovery blocks
of a spoiled train, written out term by term. It is the equation the T1 maps
of every MP2RAGE study are read through, so the closed form is held to it over
the whole T1 range a lookup table spans rather than at a single point.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.model import Simulator
from torchsim.simulators import MP2RAGESimulator

# The protocol of Marques et al., Neuroimage 49(2):1271, Table 1 at 7 T.
TI_MS = (800.0, 2700.0)
FLIP_DEG = (4.0, 5.0)
TRSPGR_MS = 6.7
TRMP2RAGE_MS = 6000.0
EFFICIENCY = 0.96


def reference(
    TI_ms, flip_deg, TRspgr_ms, TRmp2rage_ms, before, after, T1_s, efficiency
):
    """Return the two sampled magnetizations, one column per block.

    Parameters
    ----------
    TI_ms:
        The two inversion times, to the sampled shot of each block.
    flip_deg:
        Excitation flip angle of each block.
    TRspgr_ms, TRmp2rage_ms:
        Repetition time of one readout, and of the whole inversion block.
    before, after:
        Readouts either side of the sampled one.
    T1_s:
        Longitudinal relaxation times, in seconds.
    efficiency:
        Inversion efficiency, one for a perfect inversion.

    Returns
    -------
    numpy.ndarray
        ``(len(T1_s), 2)``.
    """
    TI = np.asarray(TI_ms, dtype=np.float64) * 1e-3
    flip = np.deg2rad(np.asarray(flip_deg, dtype=np.float64))
    readout = TRspgr_ms * 1e-3
    block = TRmp2rage_ms * 1e-3
    T1 = np.asarray(T1_s, dtype=np.float64)
    shots = before + after

    E1 = np.exp(-readout / T1)
    TD = np.array(
        [
            TI[0] - before * readout,
            TI[1] - TI[0] - (after + before) * readout,
            block - TI[1] - after * readout,
        ]
    )
    ETD = np.exp(-TD[:, None] / T1[None, :])
    cosalfaE1 = np.cos(flip)[:, None] * E1
    sinalfa = np.sin(flip)[:, None]

    settled = 1.0 / (
        1.0 + efficiency * np.prod(cosalfaE1, axis=0) ** shots * np.prod(ETD, axis=0)
    )
    numerator = 1 - ETD[0]
    for block_index in (0, 1):
        c = cosalfaE1[block_index]
        numerator = numerator * c**shots + (1 - E1) * (1 - c**shots) / (1 - c)
        numerator = numerator * ETD[block_index + 1] + (1 - ETD[block_index + 1])
    settled = settled * numerator

    c0, c1 = cosalfaE1
    temp = (-efficiency * settled * ETD[0] + (1 - ETD[0])) * c0**before + (1 - E1) * (
        1 - c0**before
    ) / (1 - c0)
    first = sinalfa[0] * temp

    temp = temp * c0**after + (1 - E1) * (1 - c0**after) / (1 - c0)
    temp = (temp * ETD[1] + (1 - ETD[1])) * c1**before + (1 - E1) * (1 - c1**before) / (
        1 - c1
    )
    second = sinalfa[1] * temp

    return np.stack((first, second), axis=-1)


def unified(signal):
    """Return the MP2RAGE unified image, the ratio the T1 map is read from."""
    first, second = signal[..., 0], signal[..., 1]
    return (first * second) / (first**2 + second**2)


@pytest.mark.parametrize(
    "before,after",
    [
        (64, 64),  # k-space centre mid-train
        (26, 102),  # slice partial Fourier, centre early in the train
        (100, 28),  # centre late in the train
    ],
)
def test_the_closed_form_is_the_signal_equation(before: int, after: int) -> None:
    """Both blocks, over the T1 range a lookup table spans."""
    T1_s = np.arange(0.05, 5.0001, 0.05)
    want = reference(
        TI_MS, FLIP_DEG, TRSPGR_MS, TRMP2RAGE_MS, before, after, T1_s, EFFICIENCY
    )

    simulator = MP2RAGESimulator(
        TI=TI_MS,
        flip=FLIP_DEG,
        TRspgr=TRSPGR_MS,
        TRmp2rage=TRMP2RAGE_MS,
        nshots=(before, after),
    )
    got = simulator.simulate(
        T1=tuple((T1_s * 1e3).tolist()), inv_efficiency=EFFICIENCY
    ).numpy()

    assert got.shape == want.shape
    for index in (0, 1):
        error = np.abs(got[:, index] - want[:, index]).max()
        scale = np.abs(want[:, index]).max()
        assert error / scale < 1e-5, f"block {index}: {error / scale:.2e}"


def test_the_unified_image_is_the_reference_curve() -> None:
    """What the T1 map is actually read through.

    The unified image spans [-0.5, 0.5], and a T1 map interpolates along it, so
    an absolute error here is what limits the map however well the individual
    blocks agree.
    """
    T1_s = np.arange(0.05, 5.0001, 0.05)
    want = unified(
        reference(TI_MS, FLIP_DEG, TRSPGR_MS, TRMP2RAGE_MS, 64, 64, T1_s, EFFICIENCY)
    )

    simulator = MP2RAGESimulator(
        TI=TI_MS,
        flip=FLIP_DEG,
        TRspgr=TRSPGR_MS,
        TRmp2rage=TRMP2RAGE_MS,
        nshots=128,
    )
    got = unified(
        simulator.simulate(
            T1=tuple((T1_s * 1e3).tolist()), inv_efficiency=EFFICIENCY
        ).numpy()
    )

    assert np.abs(got - want).max() < 1e-5


def test_a_shared_flip_angle_is_used_for_both_blocks() -> None:
    """One angle stands for two."""
    T1_s = np.array([0.8, 1.2, 2.5])
    want = reference(
        TI_MS, (5.0, 5.0), TRSPGR_MS, TRMP2RAGE_MS, 64, 64, T1_s, EFFICIENCY
    )

    got = (
        MP2RAGESimulator(
            TI=TI_MS,
            flip=5.0,
            TRspgr=TRSPGR_MS,
            TRmp2rage=TRMP2RAGE_MS,
            nshots=128,
        )
        .simulate(T1=tuple((T1_s * 1e3).tolist()), inv_efficiency=EFFICIENCY)
        .numpy()
    )

    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5


def test_a_perfect_inversion_is_carried() -> None:
    """The efficiency reaches the steady state and both readouts."""
    T1_s = np.array([0.8, 1.2, 2.5])
    want = reference(TI_MS, FLIP_DEG, TRSPGR_MS, TRMP2RAGE_MS, 64, 64, T1_s, 1.0)

    got = (
        MP2RAGESimulator(
            TI=TI_MS,
            flip=FLIP_DEG,
            TRspgr=TRSPGR_MS,
            TRmp2rage=TRMP2RAGE_MS,
            nshots=128,
        )
        .simulate(T1=tuple((T1_s * 1e3).tolist()), inv_efficiency=1.0)
        .numpy()
    )

    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5


def test_the_density_scales_both_blocks() -> None:
    """M0 is a scaling, so it cancels out of the unified image."""
    T1_s = np.array([0.8, 1.2, 2.5])
    simulator = MP2RAGESimulator(
        TI=TI_MS,
        flip=FLIP_DEG,
        TRspgr=TRSPGR_MS,
        TRmp2rage=TRMP2RAGE_MS,
        nshots=128,
    )
    T1_ms = tuple((T1_s * 1e3).tolist())

    plain = simulator.simulate(T1=T1_ms, inv_efficiency=EFFICIENCY)
    scaled = simulator.simulate(T1=T1_ms, M0=(2.0, 2.0, 2.0), inv_efficiency=EFFICIENCY)

    torch.testing.assert_close(scaled, 2.0 * plain)
    torch.testing.assert_close(unified(scaled), unified(plain))


# %% the same sequence, played event by event


PROTOCOL = dict(
    TI=TI_MS,
    flip=FLIP_DEG,
    TRspgr=TRSPGR_MS,
    TRmp2rage=TRMP2RAGE_MS,
)
T1_MS = (400.0, 800.0, 1200.0, 2000.0, 3000.0)
# Enough inversion blocks for the train to forget equilibrium and settle into
# the steady state the closed form solves for directly.
REPETITIONS = 8


def played(description, **properties):
    """Return the settled playing's two samples, as signed magnetizations.

    ``from_description`` is the route a stream from a scanner takes, and
    ``record="echo"`` keeps the readouts flagged as reaching the k-space
    origin. The kernels return transverse magnetization, whose phase carries
    the sign the closed form writes directly.
    """
    replayed = Simulator.from_description(
        description, MP2RAGESimulator.model, record="echo", states=1
    )
    signal = replayed.simulate(**properties, repetitions=REPETITIONS)
    return (1j * signal.reshape(-1, 2)).real


@pytest.mark.parametrize("nshots", [(64, 64), (26, 102), (100, 28)])
def test_a_description_played_back_is_the_closed_form(nshots) -> None:
    """The two are one sequence, so they must answer alike.

    The closed form is what a lookup table is built from and the train is what
    a scanner plays. A disagreement between them would be a T1 bias that
    nothing downstream could see, so it is asserted at three positions of the
    k-space centre rather than only at the middle of the train.
    """
    protocol = dict(PROTOCOL, nshots=nshots)
    simulator = MP2RAGESimulator(**protocol)

    closed = simulator.simulate(T1=T1_MS, inv_efficiency=EFFICIENCY)
    got = played(simulator.describe(**protocol), T1=T1_MS, inv_efficiency=EFFICIENCY)

    error = (got - closed).abs().max() / closed.abs().max()
    assert float(error) < 1e-4, f"{float(error):.2e}"


def test_only_the_k_space_centre_of_each_block_is_flagged() -> None:
    """Which readout reaches the k-space origin is what the description says.

    An MP2RAGE contrast is a subset of its readouts, and ``is_echo`` is what
    separates that case from an echo-resolved train.
    """
    before, after = 26, 102
    protocol = dict(PROTOCOL, nshots=(before, after))

    readouts = MP2RAGESimulator(**protocol).describe(**protocol).adc_events
    flagged = [index for index, event in enumerate(readouts) if event.is_echo]

    assert len(readouts) == 2 * (before + after)
    assert flagged == [before, before + after + before]


@pytest.mark.parametrize(
    "overrides,complaint",
    [
        (dict(TI=(50.0, 2700.0)), r"TI\[0\]"),
        (dict(TI=(800.0, 900.0)), r"TI\[1\]"),
        (dict(TI=(800.0, 5900.0)), "TRmp2rage"),
    ],
)
def test_a_train_that_does_not_fit_says_which_wait_is_negative(
    overrides, complaint
) -> None:
    """Three free-recovery waits, three ways to overrun, and each names itself."""
    protocol = dict(PROTOCOL, nshots=(64, 64), **overrides)

    with pytest.raises(ValueError, match=complaint):
        MP2RAGESimulator(**protocol).describe(**protocol)
