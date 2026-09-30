"""Pulse shapes sampled for :func:`~torchsim.sequence.SampledPulse`, and the preparations they build."""

from __future__ import annotations

from typing import Any

import torch

from ..sequence import Delay, Excitation, Inversion, SampledPulse, Spoil
from .gradient_echo import GAMMA_BAR_HZ_PER_T

#: The hyperbolic secant inversion BART's Bloch simulation plays: its peak B1
#: in tesla, its modulation rate in 1/s, and the dimensionless ``mu`` that sets
#: its frequency sweep to ``mu * beta`` rad/s either side of the carrier.
SECANT_PEAK_T = 14e-6
SECANT_BETA_PER_S = 800.0
SECANT_MU = 4.9


def sample_count(duration_s: float, dwell_s: float) -> int:
    """How many samples of ``dwell_s`` a pulse of ``duration_s`` is played as."""
    count = round(float(duration_s) / float(dwell_s))
    if count < 1 or abs(count * float(dwell_s) - float(duration_s)) > 1e-9:
        raise ValueError(
            f"a pulse of {1e3 * float(duration_s):g} ms is not a whole number "
            f"of samples of {1e3 * float(dwell_s):g} ms"
        )
    return count


def windowed_sinc(
    flip_rad: Any,
    duration_s: float,
    dwell_s: float,
    *,
    bandwidth_time: float = 4.0,
    window: float = 0.46,
) -> torch.Tensor:
    """Return a windowed sinc, in rad/s, whose samples turn through ``flip_rad``.

    ``bandwidth_time`` zero crossings span the pulse, half either side of its
    centre, under the window ``(1 - window) + window cos(pi x / (bandwidth_time
    / 2))``: 0.46 is Hamming's and 0.5 Hanning's. The samples are scaled so
    that their areas add to the flip angle.
    """
    count = sample_count(duration_s, dwell_s)
    middle = (torch.arange(count, dtype=torch.float64) + 0.5) * dwell_s
    crossings = (middle - duration_s / 2) * (bandwidth_time / duration_s)
    half = bandwidth_time / 2
    shape = ((1 - window) + window * torch.cos(torch.pi * crossings / half)) * (
        torch.sinc(crossings)
    )
    return torch.as_tensor(flip_rad) * shape / (shape.sum() * dwell_s)


def hyperbolic_secant(duration_s: float, dwell_s: float) -> torch.Tensor:
    """Return BART's adiabatic inversion pulse, complex, in rad/s.

    The amplitude is ``gamma B1 sech(beta t)`` and the phase
    ``mu log(sech(beta t))`` about the centre of the pulse [1]_, which sweeps
    the frequency through ``mu beta tanh(beta t)``.

    References
    ----------
    .. [1] Silver, M. S., Joseph, R. I., Hoult, D. I., "Highly selective
       pi/2 and pi pulse generation", Journal of Magnetic Resonance 59.2
       (1984), pp. 347-351. https://doi.org/10.1016/0022-2364(84)90181-0
    """
    count = sample_count(duration_s, dwell_s)
    middle = (torch.arange(count, dtype=torch.float64) + 0.5) * dwell_s
    envelope = 1.0 / torch.cosh(SECANT_BETA_PER_S * (middle - duration_s / 2))
    peak = 2 * torch.pi * GAMMA_BAR_HZ_PER_T * SECANT_PEAK_T
    return peak * envelope * torch.exp(1j * SECANT_MU * torch.log(envelope))


def inversion_parts(
    inversion: str | None,
    *,
    duration_s: float,
    phase_rad: Any,
    spoiler_s: float,
    dwell_s: float,
) -> list:
    """The operators an inversion prepares a train with, or none.

    ``"ideal"`` turns the longitudinal magnetization over at once, by the
    tissue's ``inv_efficiency``; ``"adiabatic"`` plays the hyperbolic secant
    sample by sample. The spoiler dephases whatever the inversion left in the
    transverse plane while the magnetization recovers for its duration.
    """
    if inversion is None:
        return []
    if inversion == "ideal":
        parts: list = [Inversion()]
    elif inversion == "adiabatic":
        parts = [
            SampledPulse(
                hyperbolic_secant(duration_s, dwell_s), dwell_s, phase_rad=phase_rad
            )
        ]
    else:
        raise ValueError(
            f"inversion is 'ideal', 'adiabatic' or None, not {inversion!r}"
        )
    if spoiler_s > 0:
        parts += [Delay(spoiler_s), Spoil()]
    return parts


def half_alpha_parts(
    flip_rad: Any,
    *,
    spacing_s: float,
    duration_s: float,
    dwell_s: float,
    bandwidth_time: float,
) -> list:
    """The alpha/2 pulse a balanced train is prepared with, ``spacing_s`` before it.

    Played at the opposite phase to the train's first pulse, it tips the
    magnetization halfway to where the train holds it. A spacing of zero is
    an instantaneous rotation at the train's start.
    """
    if spacing_s == 0:
        return [Excitation(flip_rad / 2, torch.pi)]
    if spacing_s < duration_s:
        raise ValueError(
            "the alpha/2 preparation is spaced from the train by at least its own pulse"
        )
    waveform = windowed_sinc(
        flip_rad / 2, duration_s, dwell_s, bandwidth_time=bandwidth_time
    )
    parts: list = [SampledPulse(waveform, dwell_s, phase_rad=torch.pi)]
    if spacing_s > duration_s:
        parts.append(Delay(spacing_s - duration_s))
    return parts
