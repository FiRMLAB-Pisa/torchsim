"""A spoiled gradient-echo train played with pulses of finite duration."""

from __future__ import annotations

__all__ = ["FLASHSimulator"]

from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import EventAction, Readout, SampledPulse, module
from ..sequence._array import as_torch
from ._pulses import inversion_parts, windowed_sinc


class FLASHSimulator(Simulator):
    """A spoiled gradient-echo train from equilibrium, optionally inverted first.

    Every excitation is a windowed sinc played sample by sample, so the
    magnetization relaxes, precesses off resonance and exchanges between pools
    while the pulse plays, and the transverse magnetization is spoiled after
    each sample. With ``inversion`` the train is the inversion-recovery FLASH
    of Look-Locker T1 mapping [1]_. This is the sequence BART's ``sim`` plays
    as ``FLASH`` and ``IR-FLASH``.

    References
    ----------
    .. [1] Deichmann, R., Haase, A., "Quantification of T1 values by SNAPSHOT-
       FLASH NMR imaging", Journal of Magnetic Resonance 96.3 (1992),
       pp. 608-612. https://doi.org/10.1016/0022-2364(92)90347-A

    Examples
    --------
    .. exec::

        from torchsim.simulators import FLASHSimulator

        sequence = FLASHSimulator(
            flip=8.0, TR=5.0, TE=2.0, nshots=100, inversion="adiabatic"
        )
        signal = sequence.simulate(T1=1000.0, T2=100.0)
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "B0": "b0_hz",
        },
    )
    # Nothing winds the states on: the pulses are played in the rotating
    # frame and the spoiler discards what is left transverse.
    states = 1

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        TR: float,
        TE: float,
        nshots: int,
        pulse_duration: float = 1.0,
        bandwidth_time: float = 4.0,
        inversion: str | None = None,
        inversion_duration: float = 10.0,
        inversion_phase: float = 0.0,
        spoiler: float = 0.0,
        dwell: float = 0.01,
    ) -> list:
        """Return the train, one sample per shot.

        Parameters
        ----------
        flip : float or array-like
            Flip angle in degrees, scalar or one per shot.
        TR : float
            Repetition time in milliseconds, pulse centre to pulse centre.
        TE : float
            Echo time in milliseconds, from the centre of the pulse.
        nshots : int
            Excitations in the train.
        pulse_duration : float, optional
            Duration of each excitation in milliseconds.
        bandwidth_time : float, optional
            Zero crossings of the Hamming-windowed sinc across the pulse.
        inversion : {None, "ideal", "adiabatic"}, optional
            What prepares the train: nothing, an instantaneous inversion
            scaled by ``inv_efficiency``, or a hyperbolic secant played sample
            by sample.
        inversion_duration : float, optional
            Duration of the adiabatic inversion in milliseconds.
        inversion_phase : float, optional
            Phase of the adiabatic inversion in degrees, relative to the
            train's first pulse, which decides where the transverse
            magnetization the inversion leaves points.
        spoiler : float, optional
            Time in milliseconds between the inversion and the train, at the
            end of which the transverse magnetization is spoiled.
        dwell : float, optional
            How long each sample of a pulse is held, in milliseconds.

        Raises
        ------
        ValueError
            If the sample falls inside the pulse or after the next one, if
            ``flip`` is neither scalar nor one per shot, or if a pulse is not
            a whole number of samples.
        """
        pulse_s, dwell_s = 1e-3 * pulse_duration, 1e-3 * dwell
        repetition_s, echo_s = 1e-3 * TR, 1e-3 * TE
        if echo_s < pulse_s / 2 or pulse_s / 2 + echo_s > repetition_s:
            raise ValueError(
                "TE is measured from the centre of the pulse, so it is at least "
                "half the pulse and ends before the next one begins"
            )
        angles = _per_shot(flip, nshots)
        shape = windowed_sinc(1.0, pulse_s, dwell_s, bandwidth_time=bandwidth_time)
        parts = inversion_parts(
            inversion,
            duration_s=1e-3 * inversion_duration,
            phase_rad=torch.deg2rad(torch.as_tensor(inversion_phase)),
            spoiler_s=1e-3 * spoiler,
            dwell_s=dwell_s,
        )
        for angle in angles.unbind(0):
            parts.append(
                module(
                    SampledPulse(angle * shape, dwell_s),
                    (
                        pulse_s / 2 + echo_s,
                        Readout(action=EventAction.SPOIL_AFTER),
                    ),
                    duration_s=repetition_s,
                )
            )
        return parts


def _per_shot(flip: Any, nshots: int) -> torch.Tensor:
    """Flip angles in radians, one per shot."""
    angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)).to(torch.float64))
    if angles.numel() == 1:
        return angles.expand(int(nshots))
    if angles.numel() != int(nshots):
        raise ValueError("flip must be scalar or contain one value per shot")
    return angles
