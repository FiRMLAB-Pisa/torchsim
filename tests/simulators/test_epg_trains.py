"""Trains of instantaneous pulses, against a summation over isochromats.

The reference spreads isochromats uniformly over one cycle of a crusher, so a
crusher turns the ``j``-th of ``N`` through ``2 pi j / N`` and the mean over
them keeps every configuration order below ``N`` exactly. A pulse rotates each
isochromat right-handedly about ``(cos phase, sin phase, 0)`` through ``B1``
times its flip angle, free precession turns it right-handedly about ``z``
through ``-2 pi B0 t``, and the sample is the mean of ``Mx + i My``,
demodulated by the receiver phase. The phase schedules are written out here
from the simulators' definitions.
"""

import math

import numpy as np
import pytest

from torchsim.simulators import (
    FLASHSimulator,
    HyperechoSimulator,
    StimulatedEchoSimulator,
    TrueFISPSimulator,
    fmSSFPSimulator,
)

TISSUE = {"T1": 900.0, "T2": 70.0, "M0": 1.3, "B0": 23.0, "B1": 0.9}
UNRELAXED = {"T1": 1e12, "T2": 1e12, "M0": 1.0, "B1": 1.0}


class _Isochromats:
    def __init__(self, count, T1, T2, M0=1.0, B0=0.0, B1=1.0):
        self.xy = np.zeros(count, complex)
        self.z = np.full(count, float(M0))
        self.t1_s, self.t2_s, self.m0 = 1e-3 * T1, 1e-3 * T2, M0
        self.b0, self.b1 = B0, B1
        self.winding = np.exp(2j * np.pi * np.arange(count) / count)

    def pulse(self, flip_rad, phase_rad=0.0):
        angle = self.b1 * flip_rad
        ux, uy = math.cos(phase_rad), math.sin(phase_rad)
        x, y, z = self.xy.real, self.xy.imag, self.z
        c, s = math.cos(angle), math.sin(angle)
        along = (ux * x + uy * y) * (1 - c)
        self.xy = (x * c + uy * z * s + ux * along) + 1j * (
            y * c - ux * z * s + uy * along
        )
        self.z = z * c + (ux * y - uy * x) * s

    def wait(self, duration_s):
        self.xy = self.xy * np.exp(
            -duration_s / self.t2_s - 2j * np.pi * self.b0 * duration_s
        )
        recovery = np.exp(-duration_s / self.t1_s)
        self.z = self.z * recovery + self.m0 * (1 - recovery)

    def crush(self):
        self.xy = self.xy * self.winding

    def spoil(self):
        self.xy[:] = 0.0

    def signal(self, phase_rad=0.0):
        return self.xy.mean() * np.exp(-1j * phase_rad)


def _close(signal, expected, tolerance=1e-5):
    signal, expected = np.asarray(signal), np.asarray(expected)
    error = np.abs(signal - expected).max() / np.abs(expected).max()
    assert error < tolerance, error


def test_fmssfp_sweeps_the_pulse_phase_through_one_passband_per_sweep():
    flip, tr_s, shots, sweeps = 30.0, 5e-3, 12, 2
    voxel = _Isochromats(1, **TISSUE)
    expected = []
    phase = 0.0
    for shot in range((sweeps + 1) * shots):
        phase += math.pi + 2 * math.pi * (shot % shots) / shots
        voxel.pulse(math.radians(flip), phase)
        voxel.wait(tr_s / 2)
        if shot >= sweeps * shots:
            expected.append(voxel.signal(phase))
        voxel.wait(tr_s / 2)
    sequence = fmSSFPSimulator(flip=flip, TR=1e3 * tr_s, nshots=shots, sweeps=sweeps)
    _close(sequence.simulate(**TISSUE), expected)


def _hyperecho_train(flips_deg, phases_deg):
    flips = [*np.radians(flips_deg), math.pi, *np.radians(flips_deg[::-1])]
    phases = [*np.radians(phases_deg), 0.0, *(math.pi - np.radians(phases_deg[::-1]))]
    return flips, phases


@pytest.mark.parametrize("tissue", [TISSUE, {**UNRELAXED, "B0": 37.0}])
def test_the_hyperecho_train_matches_crushed_isochromats(tissue):
    generator = np.random.default_rng(1)
    flips_deg = generator.uniform(40.0, 170.0, 5)
    phases_deg = generator.uniform(-180.0, 180.0, 5)
    spacing_s = 8e-3
    voxel = _Isochromats(64, **tissue)
    voxel.pulse(math.pi / 2, math.pi / 2)
    expected = []
    for flip, phase in zip(*_hyperecho_train(flips_deg, phases_deg), strict=True):
        voxel.wait(spacing_s / 2)
        voxel.crush()
        voxel.pulse(flip, phase)
        voxel.crush()
        voxel.wait(spacing_s / 2)
        expected.append(voxel.signal())
    sequence = HyperechoSimulator(
        flip=flips_deg, phases=phases_deg, ESP=1e3 * spacing_s, TR=1e15
    )
    _close(sequence.simulate(**tissue), expected)


def test_the_hyperecho_recovers_the_whole_excited_magnetization():
    """Whatever the train before the central pulse, and off resonance too."""
    generator = np.random.default_rng(7)
    sequence = HyperechoSimulator(
        flip=generator.uniform(20.0, 160.0, 6),
        phases=generator.uniform(-180.0, 180.0, 6),
        ESP=6.0,
        TR=1e15,
    )
    signal = np.asarray(sequence.simulate(**UNRELAXED, B0=-41.0))
    assert abs(abs(signal[-1]) - 1.0) < 1e-5
    assert np.abs(signal[:-1]).max() < 0.99


@pytest.mark.parametrize("off_resonance_hz", [0.0, 23.0, -61.0])
def test_the_spin_and_stimulated_echoes_are_their_closed_forms_at_any_off_resonance(
    off_resonance_hz,
):
    echo_s, mixing_s = 20e-3, 70e-3
    flips_deg = np.array([60.0, 100.0, 75.0])
    tissue = {**TISSUE, "B0": off_resonance_hz}
    sequence = StimulatedEchoSimulator(
        flip=flips_deg, TE=1e3 * echo_s, TM=1e3 * mixing_s
    )
    a1, a2, a3 = tissue["B1"] * np.radians(flips_deg)
    transverse = tissue["M0"] * math.exp(-echo_s / (1e-3 * tissue["T2"]))
    stored = math.exp(-mixing_s / (1e-3 * tissue["T1"]))
    expected = [
        1j * transverse * math.sin(a1) * math.sin(a2 / 2) ** 2,
        1j * transverse * math.sin(a1) * math.sin(a2) * math.sin(a3) * stored / 2,
    ]
    _close(sequence.simulate(**tissue), expected)


def test_the_stimulated_echo_matches_crushed_isochromats():
    echo_s, mixing_s = 20e-3, 70e-3
    flips = np.radians([60.0, 100.0, 75.0])
    voxel = _Isochromats(16, **TISSUE)
    voxel.pulse(flips[0])
    voxel.wait(echo_s / 2)
    voxel.crush()
    voxel.pulse(flips[1])
    voxel.crush()
    voxel.wait(echo_s / 2)
    spin_echo = voxel.signal()
    voxel.spoil()
    voxel.wait(mixing_s - echo_s / 2)
    voxel.pulse(flips[2])
    voxel.wait(echo_s / 4)
    voxel.crush()
    voxel.wait(echo_s / 4)
    sequence = StimulatedEchoSimulator(
        flip=np.degrees(flips), TE=1e3 * echo_s, TM=1e3 * mixing_s
    )
    _close(sequence.simulate(**TISSUE), [spin_echo, voxel.signal()])


def test_rf_spoiled_flash_matches_crushed_isochromats():
    shots, tr_s, te_s, flip, increment = 40, 6e-3, 2e-3, 20.0, 117.0
    voxel = _Isochromats(128, **TISSUE)
    expected = []
    for shot in range(shots):
        phase = math.radians(increment) * shot * (shot + 1) / 2
        voxel.pulse(math.radians(flip), phase)
        voxel.wait(te_s)
        expected.append(voxel.signal(phase))
        voxel.crush()
        voxel.wait(tr_s - te_s)
    sequence = FLASHSimulator(
        flip=flip,
        TR=1e3 * tr_s,
        TE=1e3 * te_s,
        nshots=shots,
        pulse_duration=0.0,
        rf_spoiling=increment,
    )
    _close(sequence.simulate(**TISSUE), expected)


def test_instantaneous_flash_is_a_train_of_hard_pulses_spoiled_ideally():
    shots, tr_s, te_s, flip = 20, 6e-3, 2e-3, 20.0
    voxel = _Isochromats(1, **TISSUE)
    expected = []
    for _ in range(shots):
        voxel.pulse(math.radians(flip))
        voxel.wait(te_s)
        expected.append(voxel.signal())
        voxel.spoil()
        voxel.wait(tr_s - te_s)
    sequence = FLASHSimulator(
        flip=flip, TR=1e3 * tr_s, TE=1e3 * te_s, nshots=shots, pulse_duration=0.0
    )
    _close(sequence.simulate(**TISSUE), expected)


def test_instantaneous_truefisp_is_a_train_of_hard_pulses_after_an_alpha_half():
    shots, tr_s, flip = 20, 5e-3, 40.0
    voxel = _Isochromats(1, **TISSUE)
    voxel.pulse(math.radians(flip) / 2, math.pi)
    voxel.wait(tr_s / 2)
    expected = []
    for shot in range(shots):
        phase = math.pi * (shot % 2)
        voxel.pulse(math.radians(flip), phase)
        voxel.wait(tr_s / 2)
        expected.append(voxel.signal(phase))
        voxel.wait(tr_s / 2)
    sequence = TrueFISPSimulator(
        flip=flip, TR=1e3 * tr_s, nshots=shots, pulse_duration=0.0
    )
    _close(sequence.simulate(**TISSUE), expected)


@pytest.mark.parametrize(
    "arguments",
    [
        {"flip": 30.0, "TR": 5.0, "nshots": 10, "TE": 6.0},
        {"flip": 30.0, "TR": 5.0, "nshots": 10, "sweeps": -1},
        {"flip": [30.0, 40.0], "TR": 5.0, "nshots": 10},
    ],
)
def test_fmssfp_refuses_a_train_it_cannot_play(arguments):
    with pytest.raises(ValueError):
        fmSSFPSimulator(**arguments).simulate(T1=900.0, T2=70.0)


@pytest.mark.parametrize(
    "arguments",
    [
        {"flip": 90.0, "TE": 20.0, "TM": 5.0},
        {"flip": [90.0, 90.0], "TE": 20.0, "TM": 70.0},
    ],
)
def test_the_stimulated_echo_refuses_what_it_cannot_play(arguments):
    with pytest.raises(ValueError):
        StimulatedEchoSimulator(**arguments).simulate(T1=900.0, T2=70.0)
