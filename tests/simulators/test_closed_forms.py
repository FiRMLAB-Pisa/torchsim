"""The closed-form simulators, each against its signal written out here.

Every expected value is built in float64 NumPy from the published expression,
or from the recursion the expression solves, so a sign, a unit or a factor
lost in the simulator shows as a mismatch rather than as agreement with
itself.
"""

import numpy as np
import pytest
import torch

from torchsim.simulators import (
    ASLSimulator,
    DiffusionSimulator,
    HSFPSimulator,
    IRbSSFPSimulator,
    IRMultiGradientEchoSimulator,
    LookLockerSimulator,
    LorentzianSimulator,
    MOLLISimulator,
    MultiGradientEchoSimulator,
    SpinEchoSimulator,
)

GAMMA_BAR = 42.57747892e6
HAMILTON = [
    (-3.80, 0.086),
    (-3.40, 0.537),
    (-2.60, 0.165),
    (-1.94, 0.046),
    (-0.39, 0.052),
    (+0.60, 0.114),
]


def _close(signal, expected, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(np.asarray(signal), expected, rtol=rtol, atol=atol)


# %% Look-Locker family


def _spoiled_train(mz, flip_rad, tr_ms, t1_ms, m0, count):
    """Longitudinal magnetization ahead of each pulse, pulse by pulse."""
    e1 = np.exp(-tr_ms / t1_ms)
    out = []
    for _ in range(count):
        out.append(mz)
        mz = mz * np.cos(flip_rad) * e1 + m0 * (1 - e1)
    return np.array(out), mz


@pytest.mark.parametrize("b1", [1.0, 0.8])
def test_look_locker_matches_the_pulse_by_pulse_recursion(b1):
    tr, flip, t1, m0, count = 5.0, 7.0, 900.0, 2.0, 300
    sequence = LookLockerSimulator(flip=flip, TR=tr, TI=tr * np.arange(count))
    signal = sequence.simulate(T1=t1, M0=m0, B1=b1, inv_efficiency=0.95)

    angle = np.deg2rad(flip) * b1
    mz, _ = _spoiled_train(-0.95 * m0, angle, tr, t1, m0, count)
    _close(signal, mz * np.sin(angle), rtol=1e-4, atol=1e-5)


def test_look_locker_short_tr_steady_state_is_r1_over_r1_star():
    tr, flip, t1 = 4.0, 8.0, 1000.0
    sequence = LookLockerSimulator(flip=flip, TR=tr, TI=1e6, short_TR=True)
    angle = np.deg2rad(flip)
    r1 = 1 / t1
    r1s = r1 - np.log(np.cos(angle)) / tr
    _close(sequence.simulate(T1=t1), r1 / r1s * np.sin(angle))


def test_look_locker_from_the_steady_state_starts_at_minus_the_steady_state():
    tr, flip, t1 = 4.0, 8.0, 1000.0
    sequence = LookLockerSimulator(
        flip=flip, TR=tr, TI=(0.0, 1e6), from_steady_state=True
    )
    first, last = sequence.simulate(T1=t1).numpy()
    _close(first, -last)


def test_molli_blocks_recover_freely_between_them():
    tr, flip, t1, m0 = 3.0, 30.0, 700.0, 1.0
    readouts, recovery = (20, 15, 20), 600.0
    sequence = MOLLISimulator(flip=flip, TR=tr, readouts=readouts, recovery=recovery)
    signal = sequence.simulate(T1=t1, M0=m0)

    angle = np.deg2rad(flip)
    start, expected = -m0, []
    for count in readouts:
        block, _ = _spoiled_train(start, angle, tr, t1, m0, count)
        expected.append(block)
        start = m0 + (block[-1] - m0) * np.exp(-recovery / t1)
    _close(signal, np.concatenate(expected) * np.sin(angle), rtol=1e-4, atol=1e-5)


def test_ir_bssfp_is_schmitts_recovery_towards_the_bssfp_steady_state():
    flip, t1, t2, m0 = 50.0, 1100.0, 80.0, 1.5
    ti = 4.0 * np.arange(0, 2000, 7)
    signal = IRbSSFPSimulator(flip=flip, TI=ti).simulate(T1=t1, T2=t2, M0=m0)

    a = np.deg2rad(flip)
    steady = m0 * np.sin(a) / (1 + np.cos(a) + (1 - np.cos(a)) * t1 / t2)
    start = m0 * np.sin(a / 2)
    r1s = np.cos(a / 2) ** 2 / t1 + np.sin(a / 2) ** 2 / t2
    _close(signal, steady - (steady + start) * np.exp(-ti * r1s), rtol=1e-4, atol=1e-6)


# %% spin echo and the hybrid state


def test_spin_echo_carries_the_recovery_left_by_the_refocusing_pulse():
    te, tr, t1, t2 = np.array([10.0, 40.0, 90.0]), 800.0, 1200.0, 70.0
    signal = SpinEchoSimulator(TE=te, TR=tr).simulate(T1=t1, T2=t2)
    recovered = 1 - 2 * np.exp(-(tr - te / 2) / t1) + np.exp(-tr / t1)
    _close(signal, recovered * np.exp(-te / t2))


def test_inversion_multiplies_the_spin_echo_by_its_recovery():
    ti, te, tr, t1, t2 = np.array([50.0, 400.0, 2000.0]), 20.0, 5000.0, 900.0, 60.0
    signal = SpinEchoSimulator(TE=te, TR=tr, TI=ti).simulate(T1=t1, T2=t2)
    recovered = 1 - 2 * np.exp(-(tr - te / 2) / t1) + np.exp(-tr / t1)
    _close(signal, (1 - 2 * np.exp(-ti / t1)) * recovered * np.exp(-te / t2))


@pytest.mark.parametrize("beta", [-1.0, 1.0])
def test_hsfp_is_the_state_a_repeated_train_settles_into(beta):
    tr, t1, t2 = 4.5, 800.0, 60.0
    flip = np.deg2rad(70.0 * np.sin(np.linspace(0.05, np.pi - 0.05, 300)) ** 2)
    signal = HSFPSimulator(flip=np.rad2deg(flip), TR=tr, beta=beta).simulate(
        T1=t1, T2=t2
    )

    # Along the spin-locked axis the length relaxes at sin^2/T2 + cos^2/T1 and
    # is driven by cos/T1. Play the train until it repeats itself.
    rates = np.sin(flip) ** 2 / t2 + np.cos(flip) ** 2 / t1
    r = 1.0
    for _ in range(200):
        trace = []
        for rate, angle in zip(rates, flip, strict=True):
            trace.append(r)
            r = np.exp(-rate * tr) * r + tr * np.cos(angle) / t1
        r = beta * r
    _close(signal, np.array(trace), rtol=1e-4, atol=1e-6)


# %% gradient echoes


def _fat(te_ms, field):
    return sum(
        a * np.exp(2j * np.pi * GAMMA_BAR * field * p * 1e-6 * te_ms * 1e-3)
        for p, a in HAMILTON
    )


def test_water_and_fat_dephase_with_the_hamilton_spectrum():
    te = 1.1 * np.arange(1, 9)
    ff, phase, t2s, fat_t2s, b0 = 0.3, 20.0, 25.0, 15.0, 30.0
    signal = MultiGradientEchoSimulator(TE=te, field_strength=1.5).simulate(
        fat_fraction=ff, fat_phase=phase, T2star=t2s, fat_T2star=fat_t2s, B0=b0, M0=2.0
    )
    water = (1 - ff) * np.exp(-te / t2s)
    fat = ff * np.exp(1j * np.deg2rad(phase)) * _fat(te, 1.5) * np.exp(-te / fat_t2s)
    expected = 2.0 * (water + fat) * np.exp(-2j * np.pi * b0 * te * 1e-3)
    _close(signal, expected)


def test_water_alone_without_t2star_carries_only_the_off_resonance_phase():
    te = np.array([2.0, 4.0, 6.0])
    signal = MultiGradientEchoSimulator(TE=te).simulate(B0=-40.0)
    _close(signal, np.exp(2j * np.pi * 40.0 * te * 1e-3))


def test_fat_modulation_is_the_spectrum_amplitude_sum_at_the_echo_top():
    signal = MultiGradientEchoSimulator(TE=0.0).simulate(fat_fraction=1.0)
    _close(signal, sum(a for _, a in HAMILTON))


def test_ir_gradient_echo_recovers_water_and_fat_along_their_own_t1():
    ti, te = np.meshgrid([30.0, 300.0, 1500.0], [1.2, 2.4], indexing="ij")
    ti, te = ti.ravel(), te.ravel()
    t1, fat_t1, ff, t2s, b0 = 1100.0, 350.0, 0.25, 30.0, 12.0
    signal = IRMultiGradientEchoSimulator(TI=ti, TE=te).simulate(
        T1=t1, fat_T1=fat_t1, fat_fraction=ff, T2star=t2s, B0=b0, inv_efficiency=0.9
    )
    water = (1 - ff) * (1 - 1.9 * np.exp(-ti / t1))
    fat = ff * (1 - 1.9 * np.exp(-ti / fat_t1)) * _fat(te, 3.0)
    expected = (water + fat) * np.exp(-te / t2s) * np.exp(-2j * np.pi * b0 * te * 1e-3)
    _close(signal, expected)


# %% diffusion, Z-spectra, perfusion


def test_isotropic_diffusion_is_stejskal_tanner():
    b = np.array([0.0, 500.0, 1000.0, 3000.0])
    signal = DiffusionSimulator(b=b).simulate(D=0.7, M0=3.0)
    _close(signal, 3.0 * np.exp(-b * 0.7e-3))


def test_tensor_diffusion_weights_the_gradient_direction():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(12, 3))
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    b = np.full(12, 1000.0)
    tensor = np.array([[1.7, 0.1, 0.05], [0.1, 0.4, -0.02], [0.05, -0.02, 0.3]])
    named = dict(
        Dxx=tensor[0, 0], Dyy=tensor[1, 1], Dzz=tensor[2, 2],
        Dxy=tensor[0, 1], Dxz=tensor[0, 2], Dyz=tensor[1, 2],
    )  # fmt: skip
    signal = DiffusionSimulator(b=b, directions=g).simulate(**named)
    expected = np.exp(-b * 1e-3 * np.einsum("ni,ij,nj->n", g, tensor, g))
    _close(signal, expected)


def test_lorentzian_pools_subtract_their_lines_from_equilibrium():
    offsets = np.linspace(-6, 6, 25)
    pools = [(0.8, 1.2, 0.0), (0.05, 1.0, 3.5), (0.1, 30.0, -2.5)]
    named = {}
    for index, (a, w, s) in enumerate(pools, start=1):
        named |= {
            f"pool{index}_amplitude": a,
            f"pool{index}_width": w,
            f"pool{index}_shift": s,
        }
    signal = LorentzianSimulator(3, offsets=offsets).simulate(M0=1.5, **named)
    lines = sum(
        a * (w / 2) ** 2 / ((w / 2) ** 2 + (offsets - s) ** 2) for a, w, s in pools
    )
    _close(signal, 1.5 * (1 - lines))


def _buxton(t, cbf, att, t1, pulsed, tau, alpha, t1b=1650.0, lam=0.9, m0=1.0):
    """Buxton et al. 1998, eqs. 4 and 5 for pulsed, 7 for continuous, in seconds."""
    t, att, tau, t1, t1b = t * 1e-3, att * 1e-3, tau * 1e-3, t1 * 1e-3, t1b * 1e-3
    f = cbf / 6000.0
    t1app = 1 / (1 / t1 + f / lam)
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        if pulsed:
            k = 1 / t1b - 1 / t1app
            if att < ti < att + tau:
                q = (
                    np.exp(k * ti)
                    * (np.exp(-k * att) - np.exp(-k * ti))
                    / (k * (ti - att))
                )
                out[i] = 2 * m0 / lam * alpha * f * (ti - att) * np.exp(-ti / t1b) * q
            elif ti >= att + tau:
                q = (
                    np.exp(k * ti)
                    * (np.exp(-k * att) - np.exp(-k * (tau + att)))
                    / (k * tau)
                )
                out[i] = 2 * m0 / lam * alpha * f * tau * np.exp(-ti / t1b) * q
        else:
            scale = 2 * m0 / lam * alpha * f * t1app * np.exp(-att / t1b)
            if att <= ti <= att + tau:
                out[i] = scale * (1 - np.exp(-(ti - att) / t1app))
            elif ti > att + tau:
                out[i] = (
                    scale
                    * (1 - np.exp(-tau / t1app))
                    * np.exp(-(ti - tau - att) / t1app)
                )
    return out


@pytest.mark.parametrize(
    ("labelling", "tau", "alpha"),
    [("continuous", 1800.0, 0.85), ("pulsed", 800.0, 0.98)],
)
def test_asl_follows_the_general_kinetic_model(labelling, tau, alpha):
    t = np.linspace(0.0, 5000.0, 101)
    signal = ASLSimulator(t=t, tau=tau, labelling=labelling).simulate(
        CBF=60.0, ATT=1200.0, T1=1400.0
    )
    expected = _buxton(t, 60.0, 1200.0, 1400.0, labelling == "pulsed", tau, alpha)
    _close(signal, expected, rtol=1e-4, atol=1e-8)


# %% what every model owes a fit


@pytest.mark.parametrize(
    ("sequence", "properties", "diff"),
    [
        (LookLockerSimulator(flip=6.0, TR=4.0, TI=4.0 * np.arange(50)), {"T1": 900.0, "B1": 0.9}, ("T1", "B1")),
        (MOLLISimulator(flip=30.0, TR=3.0, readouts=(5, 5), recovery=500.0), {"T1": 900.0}, ("T1",)),
        (IRbSSFPSimulator(flip=45.0, TI=4.0 * np.arange(50)), {"T1": 900.0, "T2": 60.0}, ("T1", "T2")),
        (SpinEchoSimulator(TE=(10.0, 20.0), TR=900.0), {"T1": 900.0, "T2": 60.0}, ("T1", "T2")),
        (HSFPSimulator(flip=np.linspace(10, 60, 40), TR=4.0), {"T1": 900.0, "T2": 60.0}, ("T1", "T2")),
        (MultiGradientEchoSimulator(TE=(1.0, 2.0, 3.0)), {"fat_fraction": 0.2, "T2star": 30.0, "B0": 5.0}, ("fat_fraction", "T2star", "B0")),
        (IRMultiGradientEchoSimulator(TI=(20.0, 900.0), TE=(1.0, 2.0)), {"T1": 900.0, "fat_fraction": 0.2}, ("T1", "fat_fraction")),
        (DiffusionSimulator(b=(0.0, 1000.0)), {"D": 0.8}, ("D",)),
        (LorentzianSimulator(1, offsets=(-1.0, 0.0, 1.0)), {"pool1_amplitude": 0.9, "pool1_width": 1.0, "pool1_shift": 0.1}, ("pool1_width", "pool1_shift")),
        (ASLSimulator(t=(1000.0, 2500.0, 4000.0), tau=1800.0), {"CBF": 60.0, "ATT": 1200.0, "T1": 1400.0}, ("CBF", "ATT")),
    ],
)  # fmt: skip
def test_forward_mode_derivative_matches_a_central_difference(
    sequence, properties, diff
):
    maps = {
        name: torch.tensor([value], dtype=torch.float64)
        for name, value in properties.items()
    }
    _, jacobian = sequence.jacobian(diff, **maps)
    for column, name in enumerate(diff):
        step = 1e-6 * max(1.0, abs(properties[name]))
        up = {**maps, name: maps[name] + step}
        down = {**maps, name: maps[name] - step}
        numeric = (sequence.simulate(**up) - sequence.simulate(**down)) / (2 * step)
        np.testing.assert_allclose(
            jacobian[..., column, :].numpy(), numeric.numpy(), rtol=1e-4, atol=1e-7
        )
