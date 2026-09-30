"""Inversion recovery read by a continuous train of pulses, in closed form."""

from __future__ import annotations

__all__ = ["IRbSSFPSimulator", "LookLockerSimulator", "MOLLISimulator"]

from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts


def _apparent(
    held: Mapping[str, Any],
    flip: torch.Tensor,
    TR: torch.Tensor,  # noqa: N803
    short_TR: bool,  # noqa: N803
) -> tuple[Any, Any, Any]:
    """Return the turned angle, the apparent rate R1* in 1/s and Mss / M0.

    The steady state under a spoiled train is Deichmann and Haase's [1]_
    ``(1 - E1) / (1 - E1 cos a)``; its short-TR limit is ``R1 / R1*``.

    References
    ----------
    .. [1] Deichmann, R., Haase, A., "Quantification of T1 values by
       SNAPSHOT-FLASH NMR imaging", Journal of Magnetic Resonance 96.3 (1992),
       pp. 608-612. https://doi.org/10.1016/0022-2364(92)90347-A
    """
    angle = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
    repetition_s = TR * 1e-3
    rate = 1e3 / held["T1"]
    apparent = rate - torch.log(torch.cos(angle)) / repetition_s
    if short_TR:
        steady = rate / apparent
    else:
        steady = (1 - torch.exp(-rate * repetition_s)) / (
            1 - torch.exp(-apparent * repetition_s)
        )
    return angle, apparent, steady


class LookLockerSimulator(Simulator):
    """Recovery from an inversion under a spoiled train of small flip angles.

    Each pulse tips a little of the longitudinal magnetization away, so the
    recovery runs towards a steady state below equilibrium at the apparent
    rate ``R1* = R1 - ln(cos a) / TR`` [1]_, and each readout records the
    longitudinal magnetization just ahead of its pulse, times ``sin a``.
    Reading the recovery at a set of times after the inversion gives T1, the
    flip angle's efficiency ``B1`` and the proton density together.

    References
    ----------
    .. [1] Look, D. C., Locker, D. R., "Time saving in measurement of NMR and
       EPR relaxation times", Review of Scientific Instruments 41.2 (1970),
       pp. 250-251. https://doi.org/10.1063/1.1684482

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import LookLockerSimulator

        sequence = LookLockerSimulator(flip=6.0, TR=4.0, TI=4.0 * torch.arange(500))
        signal = sequence.simulate(T1=(800.0, 1400.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        TI: float | npt.ArrayLike,
        short_TR: bool = False,  # noqa: N803
        from_steady_state: bool = False,
    ) -> torch.Tensor:
        """Return the transverse magnetization each readout records.

        Parameters
        ----------
        properties:
            ``T1`` in milliseconds, ``B1`` as a fraction of the nominal flip
            angle, ``inv_efficiency`` as a fraction of a perfect inversion,
            and ``M0`` as a scaling.
        flip:
            Flip angle of the readout pulses in degrees.
        TR:
            Spacing of the readout pulses in milliseconds.
        TI:
            Time of each readout after the inversion, in milliseconds.
        short_TR:
            Take the steady state as ``M0 R1 / R1*``, its limit for
            ``TR << T1``.
        from_steady_state:
            Invert the steady state rather than the equilibrium
            magnetization.
        """
        held = across_contrasts(properties, flip, TR, TI)
        density = held.get("M0", 1.0)
        angle, apparent, steady = _apparent(held, flip, TR, short_TR)
        steady = density * steady
        start = -held.get("inv_efficiency", 1.0) * (
            steady if from_steady_state else density
        )
        longitudinal = steady - (steady - start) * torch.exp(-apparent * TI * 1e-3)
        return longitudinal * torch.sin(angle)


class MOLLISimulator(Simulator):
    """Look-Locker readout blocks separated by free recovery, after one inversion.

    Modified Look-Locker inversion recovery [1]_ reads the recovery in
    blocks, one per heartbeat. The first block starts from the inversion;
    between blocks the pulses stop, and the longitudinal magnetization the
    last readout of a block recorded recovers towards ``M0`` for
    ``recovery`` milliseconds before the next block starts from it.

    References
    ----------
    .. [1] Messroghli, D. R., Radjenovic, A., Kozerke, S., et al., "Modified
       Look-Locker inversion recovery (MOLLI) for high-resolution T1 mapping of
       the heart", Magnetic Resonance in Medicine 52.1 (2004), pp. 141-146.
       https://doi.org/10.1002/mrm.20110

    Examples
    --------
    .. exec::

        from torchsim.simulators import MOLLISimulator

        sequence = MOLLISimulator(flip=35.0, TR=2.5, readouts=(40, 40, 40), recovery=800.0)
        signal = sequence.simulate(T1=(800.0, 1400.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **self.played(**sequence))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TR: float | npt.ArrayLike,
        readouts: Sequence[int],
        recovery: float | npt.ArrayLike,
        short_TR: bool = False,  # noqa: N803
    ) -> torch.Tensor:
        """Return the transverse magnetization each readout records.

        Parameters
        ----------
        properties:
            ``T1`` in milliseconds, ``B1`` as a fraction of the nominal flip
            angle, ``inv_efficiency`` as a fraction of a perfect inversion,
            and ``M0`` as a scaling.
        flip:
            Flip angle of the readout pulses in degrees.
        TR:
            Spacing of the readout pulses in milliseconds.
        readouts:
            How many readouts each block holds.
        recovery:
            Free recovery between blocks, in milliseconds.
        short_TR:
            Take the steady state as ``M0 R1 / R1*``, its limit for
            ``TR << T1``.
        """
        counts = [int(count) for count in readouts]
        positions = torch.cat([torch.arange(count) for count in counts])
        values = arrays({"flip": flip, "TR": TR, "recovery": recovery})
        flip, TR, recovery = values["flip"], values["TR"], values["recovery"]  # noqa: N806
        held = across_contrasts(properties, positions)
        density = held.get("M0", 1.0)
        angle, apparent, steady = _apparent(held, flip, TR, short_TR)
        steady = density * steady
        relaxed = torch.exp(-1e3 / held["T1"] * recovery * 1e-3)
        elapsed = (
            torch.arange(max(counts)).to(torch.as_tensor(apparent).dtype) * TR * 1e-3
        )

        start = -held.get("inv_efficiency", 1.0) * density
        blocks = []
        for count in counts:
            block = steady - (steady - start) * torch.exp(-apparent * elapsed[:count])
            blocks.append(block)
            start = density + (block[..., -1:] - density) * relaxed
        return torch.cat(blocks, dim=-1) * torch.sin(angle)


class IRbSSFPSimulator(Simulator):
    """Recovery from an inversion under a balanced SSFP train.

    After an alpha/2 preparation the balanced train drives the magnetization
    along one exponential towards the bSSFP steady state, at a rate that
    mixes R1 and R2 by the flip angle [1]_, so the recovery curve yields T1,
    T2 and the proton density together. The signal is read at a series of
    times after the inversion; ``inv_efficiency`` scales the magnetization
    the recovery starts from.

    References
    ----------
    .. [1] Schmitt, P., Griswold, M. A., Jakob, P. M., et al., "Inversion
       recovery TrueFISP: quantification of T1, T2, and spin density",
       Magnetic Resonance in Medicine 51.4 (2004), pp. 661-667.
       https://doi.org/10.1002/mrm.20058

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import IRbSSFPSimulator

        sequence = IRbSSFPSimulator(flip=45.0, TI=4.5 * torch.arange(1000))
        signal = sequence.simulate(T1=1000.0, T2=(50.0, 100.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2": None,
            "M0": None,
            "B1": None,
            "inv_efficiency": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        flip: float | npt.ArrayLike,
        TI: float | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the transverse magnetization at each time after the inversion.

        Parameters
        ----------
        properties:
            ``T1`` and ``T2`` in milliseconds, ``B1`` as a fraction of the
            nominal flip angle, ``inv_efficiency`` as a fraction of a perfect
            inversion, and ``M0`` as a scaling.
        flip:
            Flip angle of the balanced train in degrees.
        TI:
            Time of each readout after the inversion, in milliseconds: the
            repetition time times the readout's index.
        """
        held = across_contrasts(properties, flip, TI)
        angle = held.get("B1", 1.0) * (torch.pi / 180.0) * flip
        density = held.get("M0", 1.0)
        ratio = held["T1"] / held["T2"]
        half = angle / 2
        start = held.get("inv_efficiency", 1.0) * density * torch.sin(half)
        apparent = 1e3 * (
            torch.cos(half) ** 2 / held["T1"] + torch.sin(half) ** 2 / held["T2"]
        )
        steady = (
            density * torch.sin(angle) / ((ratio + 1) - torch.cos(angle) * (ratio - 1))
        )
        return steady - (steady + start) * torch.exp(-apparent * TI * 1e-3)
