"""Hybrid-state free precession, in closed form."""

from __future__ import annotations

__all__ = ["HSFPSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


class HSFPSimulator(Simulator):
    """A balanced train whose flip angle varies slowly, in the hybrid state.

    With the flip angle varied slowly enough, the magnetization stays on the
    spin-locked axis the train defines and only its length evolves, relaxing
    at ``sin^2(a) / T2 + cos^2(a) / T1`` and driven by ``cos(a) / T1`` [1]_.
    The train is played repeatedly and the recorded pass starts from where the
    previous one ended, times ``beta``: ``-1`` for an inversion between
    passes, which is what makes the answer sensitive to T1 and T2 at once, and
    ``1`` for none.

    The signal is the magnetization along the spin-locked axis at each pulse,
    which for a balanced train is the transverse magnetization at the echo.

    References
    ----------
    .. [1] Asslaender, J., Novikov, D. S., Lattanzi, R., Sodickson, D. K.,
       Cloos, M. A., "Hybrid-state free precession in nuclear magnetic
       resonance", Communications Physics 2 (2019), 73.
       https://doi.org/10.1038/s42005-019-0174-0

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import HSFPSimulator

        train = 60.0 * torch.sin(torch.linspace(0.0, torch.pi, 400)) ** 2
        sequence = HSFPSimulator(flip=train, TR=4.5)
        signal = sequence.simulate(T1=1000.0, T2=(50.0, 100.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
            "M0": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: npt.ArrayLike,
        TR: float | npt.ArrayLike,
        beta: float = -1.0,
    ) -> torch.Tensor:
        """Return the magnetization along the spin-locked axis at each pulse.

        Parameters
        ----------
        properties:
            ``T1`` and ``T2`` in milliseconds, and ``M0`` as a scaling.
        flip:
            The flip-angle train in degrees, one entry per pulse.
        TR:
            Repetition time in milliseconds.
        beta:
            What connects the end of one pass to the start of the next.
        """
        held = across_contrasts(properties, flip)
        dtype = torch.promote_types(torch.as_tensor(held["T1"]).dtype, flip.dtype)
        angle = (torch.pi / 180.0) * flip.abs().to(torch.float64)
        repetition_s = torch.as_tensor(TR).to(torch.float64) * 1e-3
        r1 = 1e3 / torch.as_tensor(held["T1"]).to(torch.float64)
        r2 = 1e3 / torch.as_tensor(held["T2"]).to(torch.float64)

        # Everything below is a running sum along the train. Sums reach
        # exp(+R TR n), which single precision cannot hold over a long train.
        rates = torch.sin(angle) ** 2 * r2 + torch.cos(angle) ** 2 * r1
        through = torch.cumsum(rates * repetition_s, dim=-1)
        driven = (
            torch.cumsum(torch.cos(angle) * torch.exp(through), dim=-1) * repetition_s
        )
        before = torch.nn.functional.pad(through, (1, 0))[..., :-1]
        driven_before = torch.nn.functional.pad(driven, (1, 0))[..., :-1]

        decayed = torch.exp(-through[..., -1:])
        start = beta * r1 * decayed / (1 - beta * decayed) * driven[..., -1:]
        length = torch.exp(-before) * (start + r1 * driven_before)
        return (held.get("M0", 1.0) * length).to(dtype)
