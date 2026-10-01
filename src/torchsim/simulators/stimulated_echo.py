"""The spin echo and the stimulated echo of three pulses."""

from __future__ import annotations

__all__ = ["StimulatedEchoSimulator"]

import numpy.typing as npt
import torch

from ..model import REFOCUSED, Simulator, SpinPhysics
from ..sequence import Dephase, EventAction, Readout
from ..sequence._array import as_torch


class StimulatedEchoSimulator(Simulator):
    """Three pulses, sampled at the spin echo of the first two and at the stimulated echo.

    The second pulse, ``TE / 2`` after the first, refocuses part of the
    transverse magnetization into a spin echo at ``TE`` and stores part of it
    along z, where it relaxes with T1 alone; the third pulse, ``TM`` after the
    second, returns that part to the transverse plane, where it rephases into
    the stimulated echo ``TE / 2`` later [1]_. With instantaneous pulses about
    x, the two samples are ``i M0 sin(a1) sin^2(a2 / 2) exp(-TE / T2)`` and
    ``i M0 sin(a1) sin(a2) sin(a3) exp(-TE / T2) exp(-TM / T1) / 2``, whatever
    the off-resonance. The transverse magnetization is spoiled after the spin
    echo, so the second sample is the stimulated echo alone. With one flip
    angle, and ``TE`` twice and ``TM`` three times its own echo time, this is
    the sequence BART's ``epg`` plays as Spinecho.

    References
    ----------
    .. [1] Hahn, E. L., "Spin echoes", Physical Review 80.4 (1950),
       pp. 580-594. https://doi.org/10.1103/PhysRev.80.580

    Examples
    --------
    .. exec::

        from torchsim.simulators import StimulatedEchoSimulator

        sequence = StimulatedEchoSimulator(flip=90.0, TE=20.0, TM=100.0)
        spin_echo, stimulated_echo = sequence.simulate(T1=1000.0, T2=80.0)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "B0": "b0_hz",
        },
        operators=REFOCUSED,
    )
    # One order either side of the second pulse and one after the third.
    states = 4

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        TE: float,
        TM: float,
    ) -> list:
        """Return the three pulses and the two samples.

        Parameters
        ----------
        flip : float or array-like
            Flip angle in degrees, scalar or one per pulse.
        TE : float
            Echo time in milliseconds: twice the spacing of the first two
            pulses, and of the third pulse and the stimulated echo.
        TM : float
            Mixing time in milliseconds, from the second pulse to the third.

        Raises
        ------
        ValueError
            If the third pulse would play before the spin echo, or if ``flip``
            is neither scalar nor one per pulse.
        """
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)).to(torch.float64))
        if angles.numel() == 1:
            angles = angles.expand(3)
        if angles.numel() != 3:
            raise ValueError("flip must be scalar or contain one value per pulse")
        if TM < TE / 2:
            raise ValueError(
                "the mixing time is at least half the echo time, so the third "
                "pulse follows the spin echo"
            )
        echo_s, mixing_s = 1e-3 * TE, 1e-3 * TM
        return [
            (0.0, self.operators.excitation(angles[0])),
            (echo_s / 2, self.operators.refocusing(angles[1])),
            (echo_s, Readout(action=EventAction.SPOIL_AFTER)),
            (echo_s / 2 + mixing_s, self.operators.excitation(angles[2])),
            (3 * echo_s / 4 + mixing_s, Dephase()),
            (echo_s + mixing_s, Readout()),
        ]
