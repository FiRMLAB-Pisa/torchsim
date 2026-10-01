"""Pulses played sample by sample, against the Bloch-McConnell equations.

The reference integrates the equations exactly over each sample of a piecewise
constant waveform, with the matrix exponential of their generator, so what the
hard-pulse approximation leaves out -- relaxation, precession and exchange
while a sample plays -- is part of what is compared. The magnetization rotates
right-handedly about ``(w1 cos phase, w1 sin phase, -2 pi (B0 + shift))``, the
sample is the sum of ``Mx + i My`` over the pools, and a pool exchanges with
the free water alone, at ``k f_pool`` from the water and ``k f_water`` back.
The pulse shapes are written out here from their definitions rather than
taken from the package.
"""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from torchsim.sequence import SampledPulse
from torchsim.simulators import CESTSimulator, FLASHSimulator, TrueFISPSimulator

GAMMA_BAR = 42.57747892e6
DWELL = 1e-5

# Fraction, exchange rate in Hz, T1 and T2 in ms and chemical shift in Hz of
# two pools beside the free water.
POOLS = ((0.05, 600.0, 1000.0, 20.0, -95.0), (0.02, 1500.0, 1200.0, 10.0, 250.0))
TISSUE = {"T1": 900.0, "T2": 70.0, "M0": 1.3, "B0": 40.0, "B1": 0.9}


def _pool_properties(pools):
    properties = {}
    for letter, (fraction, exchange, t1, t2, shift) in zip(
        "BCDE"[: len(pools)], pools, strict=True
    ):
        properties |= {
            f"pool{letter}_fraction": fraction,
            f"pool{letter}_exchange": exchange,
            f"pool{letter}_T1": t1,
            f"pool{letter}_T2": t2,
            f"pool{letter}_shift": shift,
        }
    return properties


class _BlochMcConnell:
    """The free water and its pools, from equilibrium, as ``[M_water, M_B, ..., 1]``."""

    def __init__(self, T1, T2, M0=1.0, B0=0.0, B1=1.0, pools=()):
        water = 1.0 - sum(pool[0] for pool in pools)
        self.pools = [(water, 0.0, T1, T2, 0.0), *pools]
        self.water, self.m0, self.b0, self.b1 = water, M0, B0, B1
        self.m = np.zeros(3 * len(self.pools) + 1)
        self.m[-1] = 1.0
        for index, pool in enumerate(self.pools):
            self.m[3 * index + 2] = pool[0] * M0

    def generator(self, w1=0.0, phase=0.0, offset_hz=0.0):
        size = 3 * len(self.pools) + 1
        generator = np.zeros((size, size))
        wx = self.b1 * w1 * math.cos(phase)
        wy = self.b1 * w1 * math.sin(phase)
        for index, (fraction, exchange, t1, t2, shift) in enumerate(self.pools):
            wz = -2 * math.pi * (self.b0 + shift - offset_hz)
            at = slice(3 * index, 3 * index + 3)
            generator[at, at] = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
            generator[at, at] -= np.diag([1e3 / t2, 1e3 / t2, 1e3 / t1])
            generator[3 * index + 2, -1] = 1e3 / t1 * fraction * self.m0
            for axis in range(3 * (index > 0)):
                w, p = axis, 3 * index + axis
                generator[w, w] -= exchange * fraction
                generator[w, p] += exchange * self.water
                generator[p, p] -= exchange * self.water
                generator[p, w] += exchange * fraction
        return generator

    def play(self, duration_s, w1=0.0, phase=0.0, offset_hz=0.0):
        if duration_s > 0:
            generator = self.generator(w1, phase, offset_hz)
            self.m = expm(generator * duration_s) @ self.m

    def pulse(self, waveform, phase=0.0):
        for sample in np.asarray(waveform):
            self.play(DWELL, abs(sample), phase + np.angle(sample))

    def invert(self):
        self.m[1:-1:3] *= -1
        self.m[2:-1:3] *= -1

    def spoil(self):
        self.m[0:-1:3] = 0.0
        self.m[1:-1:3] = 0.0

    def signal(self):
        return self.m[0:-1:3].sum() + 1j * self.m[1:-1:3].sum()

    def longitudinal(self):
        return self.m[2:-1:3].sum()


def _sinc(flip_deg, duration_s, bandwidth_time=4.0):
    """A Hamming-windowed sinc whose samples add to the flip angle."""
    t = (np.arange(round(duration_s / DWELL)) + 0.5) * DWELL - duration_s / 2
    x = t * bandwidth_time / duration_s
    shape = (0.54 + 0.46 * np.cos(2 * np.pi * x / bandwidth_time)) * np.sinc(x)
    return np.deg2rad(flip_deg) * shape / (shape.sum() * DWELL)


def _secant(duration_s=10e-3):
    """BART's adiabatic inversion: B1 of 14 uT, beta 800 1/s, mu 4.9."""
    t = (np.arange(round(duration_s / DWELL)) + 0.5) * DWELL - duration_s / 2
    envelope = 1 / np.cosh(800.0 * t)
    return 2 * np.pi * GAMMA_BAR * 14e-6 * envelope * envelope ** (4.9j)


def _inverted(bloch, inversion, phase_deg, spoiler_s):
    if inversion == "ideal":
        bloch.invert()
    elif inversion == "adiabatic":
        bloch.pulse(_secant(), np.deg2rad(phase_deg))
    if inversion is not None and spoiler_s > 0:
        bloch.play(spoiler_s)
        bloch.spoil()


def _close(signal, expected, tolerance):
    signal = np.asarray(signal)
    error = np.abs(signal - expected).max() / np.abs(expected).max()
    assert error < tolerance, error


# %% The pulse, against closed forms


def test_a_rectangular_pulse_nutates_about_the_effective_field():
    ppm = np.array([0.0, 0.03, 0.06, -0.1, 0.2])
    b1_ut, duration_ms = 0.5, 10.0
    sequence = CESTSimulator(
        offsets=ppm, B1sat=b1_ut, pulse_duration=duration_ms, recovery=0.0
    )
    z = sequence.simulate(T1=1e9, T2=1e9)

    w1 = 2 * np.pi * GAMMA_BAR * 1e-6 * b1_ut
    dw = 2 * np.pi * GAMMA_BAR * 1e-6 * 3.0 * ppm
    effective = np.hypot(w1, dw)
    half = effective * 1e-3 * duration_ms / 2
    np.testing.assert_allclose(
        z, 1 - 2 * (w1 / effective) ** 2 * np.sin(half) ** 2, atol=2e-5
    )


def test_continuous_saturation_reaches_the_torrey_steady_state():
    ppm = np.array([0.0, 0.1, -0.3, 1.0])
    t1, t2, b1_ut = 60.0, 25.0, 0.3
    sequence = CESTSimulator(
        offsets=ppm, B1sat=b1_ut, pulse_duration=800.0, recovery=0.0, dwell=0.05
    )
    z = sequence.simulate(T1=t1, T2=t2)

    w1 = 2 * np.pi * GAMMA_BAR * 1e-6 * b1_ut
    dw_t2 = 2 * np.pi * GAMMA_BAR * 1e-6 * 3.0 * ppm * 1e-3 * t2
    expected = (1 + dw_t2**2) / (1 + dw_t2**2 + w1**2 * 1e-6 * t1 * t2)
    np.testing.assert_allclose(z, expected, atol=1e-4)


def test_the_saturation_amplitude_derivative_is_the_nutation_derivative():
    b1_ut = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    sequence = CESTSimulator(
        offsets=[0.0], B1sat=b1_ut, pulse_duration=10.0, recovery=0.0
    )
    z = sequence.simulate(T1=1e9, T2=1e9)
    (derivative,) = torch.autograd.grad(z.sum(), b1_ut)

    per_ut = 2 * np.pi * GAMMA_BAR * 1e-6 * 10e-3
    assert z.shape == (1,)
    np.testing.assert_allclose(z.detach(), np.cos(per_ut * 0.5), atol=2e-5)
    np.testing.assert_allclose(derivative, -per_ut * np.sin(per_ut * 0.5), rtol=1e-4)


def test_the_z_spectrum_reads_mz_whatever_the_transmit_field_scales_the_read_by():
    b1 = torch.tensor([0.7, 1.0, 1.3])
    phase = torch.tensor([0.0, 0.8, -2.0])
    sequence = CESTSimulator(offsets=[0.0, 0.05], B1sat=0.5, pulse_duration=10.0)
    z = sequence.simulate(T1=1e9, T2=1e9, B1=b1, B1phase=phase)

    w1 = 2 * np.pi * GAMMA_BAR * 1e-6 * 0.5 * b1.numpy()
    dw = 2 * np.pi * GAMMA_BAR * 1e-6 * 3.0 * np.array([0.0, 0.05])[:, None]
    effective = np.hypot(w1, dw)
    expected = 1 - 2 * (w1 / effective) ** 2 * np.sin(effective * 5e-3) ** 2
    assert z.shape == (3, 2)
    np.testing.assert_allclose(z, expected.T, atol=2e-5)


def test_a_train_off_resonance_on_a_voxel_at_that_offset_plays_as_one_on_resonance():
    offset_ppm = 0.4
    offset_hz = GAMMA_BAR * 1e-6 * 3.0 * offset_ppm
    pools = _pool_properties(POOLS)
    split = CESTSimulator(
        offsets=[offset_ppm], B1sat=1.0, npulses=3, pulse_duration=10.0
    )
    whole = CESTSimulator(offsets=[0.0], B1sat=1.0, npulses=1, pulse_duration=30.0)

    np.testing.assert_allclose(
        split.simulate(T1=900.0, T2=70.0, B0=offset_hz, **pools),
        whole.simulate(T1=900.0, T2=70.0, B0=0.0, **pools),
        atol=1e-5,
    )


# %% The sequences, against the Bloch-McConnell equations


@pytest.mark.parametrize("pools", [(), POOLS], ids=["water", "three-pools"])
def test_the_z_spectrum_matches_the_bloch_mcconnell_equations(pools):
    ppm = np.array([-3.0, -0.75, 0.0, 2.0])
    sequence = CESTSimulator(
        offsets=ppm, B1sat=1.5, npulses=3, pulse_duration=20.0,
        interpulse_delay=5.0, recovery=2.0,
    )  # fmt: skip
    z = sequence.simulate(**TISSUE, **_pool_properties(pools))

    expected = []
    for offset_hz in GAMMA_BAR * 1e-6 * 3.0 * ppm:
        bloch = _BlochMcConnell(**TISSUE, pools=pools)
        w1 = 2 * np.pi * GAMMA_BAR * 1.5e-6
        for pulse in range(3):
            # In a frame that turns with the pulse while it plays and stands
            # still between pulses, each pulse is one exponential.
            bloch.play(20e-3, w1, offset_hz=offset_hz)
            if pulse < 2:
                bloch.play(5e-3)
        bloch.spoil()
        bloch.play(2e-3)
        expected.append(bloch.longitudinal())
    _close(z, np.array(expected), 2e-4)


@pytest.mark.parametrize("pools", [(), POOLS], ids=["water", "three-pools"])
@pytest.mark.parametrize("inversion", [None, "ideal", "adiabatic"])
def test_flash_matches_the_bloch_mcconnell_equations(inversion, pools):
    flip, tr, te, shots, spoiler = 25.0, 5e-3, 2e-3, 10, 1e-3
    phase = math.degrees(4.9 * math.log(14e-6))
    sequence = FLASHSimulator(
        flip=flip, TR=1e3 * tr, TE=1e3 * te, nshots=shots, inversion=inversion,
        inversion_phase=phase, spoiler=1e3 * spoiler,
    )  # fmt: skip
    signal = sequence.simulate(**TISSUE, **_pool_properties(pools))

    bloch = _BlochMcConnell(**TISSUE, pools=pools)
    _inverted(bloch, inversion, phase, spoiler)
    expected = []
    for _ in range(shots):
        bloch.pulse(_sinc(flip, 1e-3))
        bloch.play(te - 0.5e-3)
        expected.append(bloch.signal())
        bloch.spoil()
        bloch.play(tr - 0.5e-3 - te)
    _close(signal, np.array(expected), 2e-4)


@pytest.mark.parametrize("pools", [(), POOLS], ids=["water", "three-pools"])
@pytest.mark.parametrize(
    "preparation", [dict(preparation=2.5), dict(preparation=1.0, TE=2.0)]
)
@pytest.mark.parametrize("inversion", [None, "adiabatic"])
def test_truefisp_matches_the_bloch_mcconnell_equations(inversion, preparation, pools):
    flip, tr, shots = 50.0, 5e-3, 10
    te = 1e-3 * preparation.get("TE", 2.5)
    phase = math.degrees(4.9 * math.log(14e-6))
    sequence = TrueFISPSimulator(
        flip=flip, TR=1e3 * tr, nshots=shots, inversion=inversion,
        inversion_phase=phase, **preparation,
    )  # fmt: skip
    signal = sequence.simulate(**TISSUE, **_pool_properties(pools))

    bloch = _BlochMcConnell(**TISSUE, pools=pools)
    _inverted(bloch, inversion, phase, 0.0)
    bloch.pulse(_sinc(flip / 2, 1e-3), np.pi)
    bloch.play(1e-3 * preparation["preparation"] - 1e-3)
    expected = []
    for shot in range(shots):
        bloch.pulse(_sinc(flip, 1e-3), np.pi * (shot % 2))
        bloch.play(te - 0.5e-3)
        expected.append(bloch.signal() * (-1) ** shot)
        bloch.play(tr - 0.5e-3 - te)
    _close(signal, np.array(expected), 2e-4)


def test_an_instantaneous_preparation_is_an_alpha_half_rotation_at_the_train():
    flip = 40.0
    sequence = TrueFISPSimulator(flip=flip, TR=5.0, nshots=6, preparation=0.0)
    signal = sequence.simulate(T1=900.0, T2=70.0)

    bloch = _BlochMcConnell(900.0, 70.0)
    bloch.play(1e-9, np.deg2rad(flip / 2) * 1e9, np.pi)
    expected = []
    for shot in range(6):
        bloch.pulse(_sinc(flip, 1e-3), np.pi * (shot % 2))
        bloch.play(2.5e-3 - 0.5e-3)
        expected.append(bloch.signal() * (-1) ** shot)
        bloch.play(2.5e-3 - 0.5e-3)
    _close(signal, np.array(expected), 2e-4)


# %% What the pulse and the sequences refuse


def test_a_single_shot_of_a_single_voxel_is_one_sample():
    signal = FLASHSimulator(flip=10.0, TR=5.0, TE=2.0, nshots=1).simulate(
        T1=900.0, T2=70.0
    )
    assert signal.shape == (1,)


@pytest.mark.parametrize(
    ("waveform", "dwell", "extra"),
    [
        (torch.ones(0), 1e-5, {}),
        (torch.ones(2, 3), 1e-5, {}),
        (torch.ones(3), 0.0, {}),
        (torch.ones(3), 1e-5, {"offset_hz": [1.0, 2.0], "definition_id": 1}),
    ],
    ids=["empty", "two-dimensional", "no-dwell", "definition-at-two-offsets"],
)
def test_a_sampled_pulse_refuses_what_it_cannot_play(waveform, dwell, extra):
    with pytest.raises(ValueError):
        SampledPulse(waveform, dwell, **extra)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(pulse_duration=1.005),
        dict(TE=0.2),
        dict(TE=4.8),
        dict(flip=[10.0, 20.0]),
        dict(inversion="hard"),
    ],
    ids=[
        "partial-sample",
        "echo-in-pulse",
        "echo-past-next-pulse",
        "flip-count",
        "inversion",
    ],
)
def test_flash_refuses_a_train_it_cannot_play(arguments):
    arguments = {"flip": 10.0, "TR": 5.0, "TE": 2.0, "nshots": 3, **arguments}
    with pytest.raises(ValueError):
        FLASHSimulator(**arguments).simulate(T1=900.0, T2=70.0)


def test_truefisp_refuses_a_preparation_shorter_than_its_pulse():
    with pytest.raises(ValueError):
        TrueFISPSimulator(flip=40.0, TR=5.0, nshots=3, preparation=0.5).simulate(
            T1=900.0, T2=70.0
        )
