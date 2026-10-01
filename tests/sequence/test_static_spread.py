"""What a spread of the field across a voxel does to a sample.

A voxel is not one frequency. Where the field varies across it, the spins
inside dephase against each other, and a Lorentzian spread of half-width
``1 / T2'`` damps a sample by ``exp(-|tau| / T2')`` in the time ``tau`` it has
gone unrefocused -- growing either side of a spin echo and never recovering
after a gradient echo.

These hold that against things outside TorchSim: the characteristic function of
the Cauchy distribution, summed here over a quadrature rather than taken on
trust, and the closed form ``T2*`` a gradient echo decays at. The ensemble is
also played through the states themselves, so the claim that a spread and a
population of frequencies are the same thing is checked and not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsim.sequence import (
    EpgEngine,
    EventAction,
    TissueProperties,
    _builders,
)
from torchsim.sequence._description import (
    AdcRole,
    SequenceDescription,
    ideal_rf_definition,
)
from torchsim.sequence._operators import (
    Dephase,
    Excitation,
    Readout,
    Refocusing,
    compose,
)

T1_MS, T2_MS = 1000.0, 80.0
T2_PRIME_MS = 25.0
LENGTH = 12
SPACING_S = 8e-3

# Where around each spin echo the train is sampled, in seconds. The echo itself
# and a symmetric pair either side of it, which is where the static spread has
# gone and come back.
OFFSETS_S = (-2.5e-3, 0.0, 3.5e-3)

# How finely the Cauchy is quadratured. Its tails are heavy, so the mean over
# uniform quantiles converges slowly; this is enough for three digits.
ISOCHROMATS = 20001


def _sampled_spin_echo(echoes: int = 4) -> SequenceDescription:
    """A refocused train sampled off the echo centres as well as on them."""
    modules: list = [(0.0, Excitation(torch.pi / 2, torch.pi / 2))]
    for index in range(echoes):
        echo_s = (index + 1) * SPACING_S
        modules.append((echo_s - 0.5 * SPACING_S, Refocusing(torch.pi, 0.0)))
        modules.extend(
            (
                echo_s + offset_s,
                Readout(0.0, role=AdcRole.ECHO_CENTER, is_echo=offset_s == 0.0),
            )
            for offset_s in OFFSETS_S
        )
    events, _ = compose(*modules)
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * echoes * SPACING_S,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )


def _described(modules: list, duration_s: float) -> SequenceDescription:
    events, _ = compose(*modules)
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=1e6 * duration_s,
        events=events,
        rf_definitions={0: ideal_rf_definition()},
    )


def _crushed_between_pulses(echoes: int = 4) -> SequenceDescription:
    """A spin-echo train whose crushers are played midway into each gap."""
    modules: list = [(0.0, Excitation(torch.pi / 2, torch.pi / 2))]
    for index in range(echoes):
        pulse_s = (index + 0.5) * SPACING_S
        modules += [
            (pulse_s - 0.25 * SPACING_S, Dephase()),
            (pulse_s, Refocusing(torch.pi * 0.8, 0.0, crushed=False)),
            (pulse_s + 0.25 * SPACING_S, Dephase()),
            (pulse_s + 0.5 * SPACING_S, Readout(0.0, is_echo=True)),
        ]
    return _described(modules, echoes * SPACING_S)


def _crushed_at_pulses(echoes: int = 4) -> SequenceDescription:
    modules: list = [(0.0, Excitation(torch.pi / 2, torch.pi / 2))]
    for index in range(echoes):
        pulse_s = (index + 0.5) * SPACING_S
        modules += [
            (pulse_s, Refocusing(torch.pi * 0.8, 0.0)),
            (pulse_s + 0.5 * SPACING_S, Readout(0.0, is_echo=True)),
        ]
    return _described(modules, echoes * SPACING_S)


def _stimulated_echo(mixing_s: float = 30e-3) -> SequenceDescription:
    """The spin echo of two pulses, spoiled, then the stimulated echo of a third."""
    return _described(
        [
            (0.0, Excitation(torch.pi / 3, 0.0)),
            (0.5 * SPACING_S, Refocusing(torch.pi / 2, 0.0)),
            (SPACING_S, Readout(0.0, action=EventAction.SPOIL_AFTER)),
            (0.5 * SPACING_S + mixing_s, Excitation(torch.pi / 2, 0.0)),
            (0.75 * SPACING_S + mixing_s, Dephase()),
            (SPACING_S + mixing_s, Readout(0.0, is_echo=True)),
        ],
        SPACING_S + mixing_s,
    )


def _spoiled(echo_time_s: float = 4e-3) -> SequenceDescription:
    return _builders.spgr_description(
        torch.full((LENGTH,), np.deg2rad(20.0)),
        torch.full((LENGTH,), 12e-3),
        echo_time_s,
    )


DESCRIPTIONS = {
    "spoiled, TE 4 ms": _spoiled(4e-3),
    "spoiled, TE 7 ms": _spoiled(7e-3),
    "refocused, sampled off the echoes": _sampled_spin_echo(),
}


def _signal(description, *, repetitions=1, **tissue) -> np.ndarray:
    result = EpgEngine().simulate(
        description,
        TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS, **tissue),
        nstates=48,
        repetitions=repetitions,
    )
    return result.signal.reshape(-1).numpy()


def _unrefocused_s(description) -> np.ndarray:
    """How long each sample has gone unrefocused, from the run's own labels."""
    from torchsim.sequence._accelerators import _pack_events

    packed = _pack_events(
        description,
        repetitions=1,
        record="all",
        device=torch.device("cpu"),
        rf_raster_time_s=1e-6,
    )
    return packed.unrefocused_us.reshape(-1).numpy() * 1e-6


def _cauchy_hz(t2_prime_ms: float) -> np.ndarray:
    """A Lorentzian spread of half-width ``1 / T2'``, at uniform quantiles.

    The half-width is quoted as an angular rate, so in Hz it is
    ``1 / (2 pi T2')`` -- which is what makes the transform of the whole
    distribution ``exp(-|tau| / T2')`` and not some multiple of it.
    """
    quantiles = (np.arange(ISOCHROMATS) + 0.5) / ISOCHROMATS
    half_width_hz = 1.0 / (2.0 * np.pi * t2_prime_ms * 1e-3)
    return half_width_hz * np.tan(np.pi * (quantiles - 0.5))


@pytest.mark.parametrize("name", list(DESCRIPTIONS))
def test_the_damping_is_what_a_population_of_frequencies_does(name):
    """The spread against the ensemble average it stands for, summed here."""
    description = DESCRIPTIONS[name]
    on_resonance = _signal(description)
    tau_s = _unrefocused_s(description)

    # Each isochromat turns through its own frequency for as long as the sample
    # has gone unrefocused; their mean is what one voxel of them records.
    turns = np.exp(-2j * np.pi * _cauchy_hz(T2_PRIME_MS)[:, None] * tau_s[None, :])
    predicted = on_resonance * turns.mean(axis=0)

    spread = _signal(description, t2_prime_ms=T2_PRIME_MS)
    assert np.abs(spread - predicted).max() < 2e-3 * np.abs(on_resonance).max()


@pytest.mark.parametrize("name", list(DESCRIPTIONS))
def test_the_same_population_carried_through_the_states_agrees(name):
    """The ensemble played rather than summed, one voxel per isochromat.

    Nothing here reads the unrefocused time: each voxel carries its own field
    through every operator, and what comes back is averaged. That the two
    routes meet is the whole content of applying the spread to a sample.
    """
    description = DESCRIPTIONS[name]
    frequencies = torch.as_tensor(_cauchy_hz(T2_PRIME_MS), dtype=torch.float32)
    played = _signal(description, b0_hz=frequencies).reshape(ISOCHROMATS, -1)
    spread = _signal(description, t2_prime_ms=T2_PRIME_MS)
    scale = np.abs(spread).max()
    assert np.abs(played.mean(axis=0) - spread).max() < 3e-3 * scale


def test_a_gradient_echo_decays_at_t2_star():
    """``1 / T2* = 1 / T2 + 1 / T2'``, which is the definition of the term."""
    t2_star_ms = 1.0 / (1.0 / T2_MS + 1.0 / T2_PRIME_MS)
    description = _spoiled(5e-3)
    spread = _signal(description, t2_prime_ms=T2_PRIME_MS)
    starred = (
        EpgEngine()
        .simulate(description, TissueProperties(t1_ms=T1_MS, t2_ms=t2_star_ms))
        .signal.reshape(-1)
        .numpy()
    )
    assert np.abs(spread - starred).max() < 1e-5 * np.abs(starred).max()


def test_a_spin_echo_is_left_exactly_as_it_stands():
    """Which is what refocusing is for, and what separates T2 from T2*."""
    description = _builders.fse_description(
        torch.full((LENGTH,), np.deg2rad(180.0)),
        SPACING_S,
        phases_rad=0.0,
        excitation_phase_rad=np.pi / 2,
    )
    plain = _signal(description)
    spread = _signal(description, t2_prime_ms=T2_PRIME_MS)
    assert np.abs(spread - plain).max() < 1e-6 * np.abs(plain).max()


def test_the_spread_comes_back_either_side_of_an_echo():
    """It grows with the time since the last pulse and winds back to nothing."""
    description = _sampled_spin_echo()
    plain = _signal(description)
    spread = _signal(description, t2_prime_ms=T2_PRIME_MS)
    ratio = np.abs(spread) / np.abs(plain)

    offsets_s = np.tile(np.abs(OFFSETS_S), len(ratio) // len(OFFSETS_S))
    expected = np.exp(-offsets_s / (T2_PRIME_MS * 1e-3))
    assert np.abs(ratio - expected).max() < 1e-5


def test_a_crusher_between_pulses_winds_as_one_beside_the_pulse():
    """Where in the gap a crusher plays changes nothing a sample can see."""
    beside = _crushed_at_pulses()
    between = _crushed_between_pulses()
    for tissue in ({"b0_hz": 31.0}, {"t2_prime_ms": T2_PRIME_MS}):
        expected = _signal(beside, **tissue)
        np.testing.assert_allclose(
            _signal(between, **tissue), expected, atol=1e-6 * np.abs(expected).max()
        )


def test_a_stimulated_echo_is_left_exactly_as_it_stands():
    """Its coherence spent as long dephasing before the store as after it."""
    description = _stimulated_echo()
    plain = _signal(description)
    for tissue in ({"b0_hz": 31.0}, {"t2_prime_ms": T2_PRIME_MS}):
        np.testing.assert_allclose(
            _signal(description, **tissue), plain, atol=1e-6 * np.abs(plain).max()
        )


def test_a_sequence_that_does_not_wind_at_one_rate_refuses_the_spread():
    """There is no term in the states to fall back on, so it is not dropped."""
    description = _builders.mrf_description(
        torch.full((LENGTH,), np.deg2rad(40.0)),
        torch.linspace(11e-3, 15e-3, LENGTH),
    )
    # The same sequence without a spread is simulated, so what is refused is
    # the spread and not the sequence.
    assert _signal(description).shape == (LENGTH,)
    with pytest.raises(ValueError, match="unrefocused"):
        _signal(description, t2_prime_ms=T2_PRIME_MS)


def test_a_spread_that_is_no_spread_is_not_refused():
    """The refusal is about what the tissue has, not what it named.

    A map handed a whole array of them says a spread is in play before a
    buffer is touched, which is what keeps the common case off the device;
    what an array of infinities holds is still no spread, and refusing it
    would turn a sequence away over nothing.
    """
    description = _builders.mrf_description(
        torch.full((LENGTH,), np.deg2rad(40.0)),
        torch.linspace(11e-3, 15e-3, LENGTH),
    )
    idle = torch.full((2,), float("inf"))
    signal = _signal(description, t2_prime_ms=idle).reshape(2, -1)
    torch.testing.assert_close(
        torch.as_tensor(signal),
        torch.as_tensor(_signal(description).reshape(1, -1)).expand(2, -1).clone(),
    )


def test_the_spread_is_differentiated():
    """Against the derivative of the factor it applies, which is a closed form."""
    description = _spoiled(6e-3)
    t2_prime = torch.tensor([T2_PRIME_MS], requires_grad=True)
    signal = (
        EpgEngine()
        .simulate(
            description,
            TissueProperties(t1_ms=T1_MS, t2_ms=T2_MS, t2_prime_ms=t2_prime),
        )
        .signal
    )
    (gradient,) = torch.autograd.grad(signal.abs().sum(), t2_prime)

    # d/dT2' of |S0| exp(-tau / T2') is |S| tau / T2'**2, summed over samples.
    tau_ms = 1e3 * _unrefocused_s(description)
    magnitude = signal.detach().abs().reshape(-1).numpy()
    expected = float((magnitude * np.abs(tau_ms) / T2_PRIME_MS**2).sum())
    assert gradient.item() == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("route", ["auto", 50])
def test_a_settled_state_carries_the_spread(route):
    """The damping is a constant per sample, so the recursion stays affine.

    Both routes to a settled state read it off recorded signals, and a signal
    the spread has scaled is scaled the same way wherever it is read -- so the
    map they recover is the one the states obey and its fixed point is reached
    with the spread on it.
    """
    description = _builders.spgr_description(
        torch.full((16,), np.deg2rad(15.0)), torch.full((16,), 12e-3), 5e-3
    )
    settled = _signal(description, t2_prime_ms=T2_PRIME_MS, repetitions=route)
    reached = _signal(description, t2_prime_ms=T2_PRIME_MS, repetitions=200)
    assert np.abs(settled - reached).max() < 1e-5 * np.abs(reached).max()


def test_a_model_declares_it_like_any_other_property():
    from torchsim.model._state_machine import SpinPhysics
    from torchsim.sequence._parameters import features_of

    physics = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "T2p": "t2_prime_ms"}
    )
    declared = {"T1": T1_MS, "T2": T2_MS}
    tissue = physics.tissue({**declared, "T2p": T2_PRIME_MS})
    assert tissue.t2_prime_ms == T2_PRIME_MS
    assert "T2_PRIME" in features_of(tissue)
    # Left out, it is at its identity and the signal carries no spread at all.
    assert "T2_PRIME" not in features_of(physics.tissue(declared))
