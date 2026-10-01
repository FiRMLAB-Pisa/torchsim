"""Frequency-modulated balanced SSFP."""

from __future__ import annotations

__all__ = ["fmSSFPSimulator"]

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import Readout, module
from ._pulses import excitation
from .flash import _per_shot


class fmSSFPSimulator(Simulator):
    """A balanced SSFP train whose RF frequency is swept through one passband per sweep.

    The n-th pulse of a sweep, counted from zero, is played ``pi + 2 pi n /
    nshots`` radians ahead of the pulse before it. A phase increment acts on
    the magnetization as an off-resonance of ``increment / (2 pi TR)``, so
    across one sweep of ``nshots`` repetitions every voxel is carried once
    through the period ``1 / TR`` of the balanced SSFP off-resonance profile
    [1]_. Each sample is demodulated by the phase of its own pulse, and
    ``sweeps`` sweeps are played before the one recorded, which is how the
    radial acquisition of [2]_ is prepared. The pulses are instantaneous
    rotations unless ``pulse_duration`` is given, and are then the windowed
    sincs :class:`TrueFISPSimulator` plays. This is the sequence BART's ``epg``
    plays as fmSSFP.

    References
    ----------
    .. [1] Foxall, D. L., "Frequency-modulated steady-state free precession
       imaging", Magnetic Resonance in Medicine 48.3 (2002), pp. 502-508.
       https://doi.org/10.1002/mrm.10225

    .. [2] Roeloffs, V., Rosenzweig, S., Holme, H. C. M., Uecker, M., Frahm,
       J., "Frequency-modulated SSFP with radial sampling and subspace
       reconstruction: A time-efficient alternative to phase-cycled bSSFP",
       Magnetic Resonance in Medicine 81.3 (2019), pp. 1566-1579.
       https://doi.org/10.1002/mrm.27505

    Examples
    --------
    .. exec::

        from torchsim.simulators import fmSSFPSimulator

        sequence = fmSSFPSimulator(flip=30.0, TR=5.0, nshots=100)
        signal = sequence.simulate(T1=1000.0, T2=100.0, B0=20.0)
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
    # A balanced train winds nothing on, so order zero is the whole state.
    states = 1

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        TR: float,
        nshots: int,
        sweeps: int = 1,
        TE: float | None = None,
        pulse_duration: float = 0.0,
        bandwidth_time: float = 4.0,
        dwell: float = 0.01,
    ) -> list:
        """Return the sweeps, one sample per shot of the last.

        Parameters
        ----------
        flip : float or array-like
            Flip angle in degrees, scalar or one per shot of a sweep.
        TR : float
            Repetition time in milliseconds, pulse centre to pulse centre.
        nshots : int
            Excitations in one sweep.
        sweeps : int, optional
            Sweeps played before the one recorded.
        TE : float, optional
            Echo time in milliseconds, from the centre of the pulse; half the
            repetition time when not given, where a balanced train refocuses.
        pulse_duration : float, optional
            Duration of each pulse in milliseconds. Zero plays each as an
            instantaneous rotation.
        bandwidth_time : float, optional
            Zero crossings of the Hamming-windowed sinc across a pulse.
        dwell : float, optional
            How long each sample of a pulse is held, in milliseconds.

        Raises
        ------
        ValueError
            If the sample falls inside the pulse or after the next one, if
            ``sweeps`` is negative, if ``flip`` is neither scalar nor one per
            shot, or if a pulse is not a whole number of samples.
        """
        pulse_s, dwell_s = 1e-3 * pulse_duration, 1e-3 * dwell
        repetition_s = 1e-3 * TR
        echo_s = repetition_s / 2 if TE is None else 1e-3 * TE
        if echo_s < pulse_s / 2 or pulse_s / 2 + echo_s > repetition_s:
            raise ValueError(
                "TE is measured from the centre of the pulse, so it is at least "
                "half the pulse and ends before the next one begins"
            )
        if int(sweeps) < 0:
            raise ValueError(f"sweeps must not be negative, not {sweeps!r}")
        angles = _per_shot(flip, nshots)
        pulse = excitation(pulse_s, dwell_s, bandwidth_time=bandwidth_time)

        played = (int(sweeps) + 1) * int(nshots)
        within = torch.arange(played, dtype=torch.float64) % int(nshots)
        increments = torch.pi + 2 * torch.pi * within / int(nshots)
        turns = torch.remainder(torch.cumsum(increments, 0), 2 * torch.pi).tolist()
        recorded = played - int(nshots)

        parts = []
        for shot, turn in enumerate(turns):
            angle = angles[shot % int(nshots)]
            if shot < recorded:
                parts.append(module(pulse(angle, turn), duration_s=repetition_s))
            else:
                parts.append(
                    module(
                        pulse(angle, turn),
                        (pulse_s / 2 + echo_s, Readout(turn)),
                        duration_s=repetition_s,
                    )
                )
        return parts
