"""MP2RAGE, in closed form and as the train it stands for."""

from __future__ import annotations

__all__ = ["MP2RAGESimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import SPOILED, Simulator, SpinPhysics
from ..sequence import AdcRole
from ..sequence._array import arrays, as_torch, matched


class MP2RAGESimulator(Simulator):
    """Two gradient-echo blocks read at two inversion times, in closed form.

    The train is spoiled and sampled at the k-space centre of each block, so
    only the longitudinal magnetization carries between shots and the steady
    state it settles into has a closed form. Combining the two blocks into one
    ratio gives an image free of the receive bias field, which is what the
    sequence is for [1]_.

    References
    ----------
    .. [1] Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W., Van de
       Moortele, P.-F., Gruetter, R., "MP2RAGE, a self bias-field corrected
       sequence for improved segmentation and T1-mapping at high field",
       NeuroImage 49.2 (2010), pp. 1271-1281.
       https://doi.org/10.1016/j.neuroimage.2009.10.002

    Examples
    --------
    .. exec::

        from torchsim.simulators import MP2RAGESimulator

        sequence = MP2RAGESimulator(
            TI=(500.0, 1500.0),
            flip=5.0,
            TRspgr=5.0,
            TRmp2rage=3000.0,
            nshots=128,
        )
        signal = sequence.simulate(T1=(200.0, 1000.0), inv_efficiency=0.95)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "M0": "m0",
            "inv_efficiency": "inversion_efficiency",
        },
        # Both blocks spoil after each readout, so nothing transverse survives
        # an interval and no T2 is asked for.
        fixed={"t2_ms": 100.0},
        operators=SPOILED,
    )
    states = 1
    record = "acquired"

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def layout(
        self,
        *,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float | npt.ArrayLike,
        TRmp2rage: float | npt.ArrayLike,
        nshots: int | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
    ) -> list:
        """Return the whole train: one inversion, then two blocks of readouts.

        The closed form of :meth:`evaluate` is what a lookup table is built
        from, because it costs one expression per T1. This is the same
        sequence written out event by event, which is what a description
        arriving from a scanner is compared against and what carries a train
        the closed form has no parameter for.

        Parameters
        ----------
        TI : array-like
            The two inversion times in milliseconds, each measured to the
            sampled shot of its block.
        flip : float or array-like
            Excitation flip angle in degrees, one per block or one shared.
        TRspgr : float or array-like
            Repetition time in milliseconds of one readout.
        TRmp2rage : float or array-like
            Repetition time in milliseconds of the whole inversion block.
        nshots : int or array-like
            Readouts per block, either the total -- halved for each block --
            or ``(before, after)`` the sampled shot.
        phases : float or array-like, optional
            Excitation phases in degrees, one per block or one shared.

        Raises
        ------
        ValueError
            If either inversion time falls before its block's first
            excitation, or if the second block does not fit inside the
            repetition time.
        """
        angle = torch.deg2rad(_shared_or_two(as_torch(flip)))
        turn = torch.deg2rad(_shared_or_two(matched(phases, angle)))
        before, after = _shots_either_side(nshots)
        before, after = int(before), int(after)
        shots = before + after
        readout_s = as_torch(TRspgr).flatten()[0] * 1e-3
        inversion_s = as_torch(TI).flatten() * 1e-3
        block_s = as_torch(TRmp2rage).flatten()[0] * 1e-3

        # The same three free-recovery waits the closed form takes: to the
        # first block's sampled shot, between the blocks, and to the end.
        waits = (
            inversion_s[0] - before * readout_s,
            inversion_s[1] - inversion_s[0] - (after + before) * readout_s,
            block_s - inversion_s[1] - after * readout_s,
        )
        for wait, complaint in zip(waits, _COMPLAINTS, strict=False):
            if bool(wait < 0):
                raise ValueError(complaint)

        parts = [self.operators.inversion(duration_s=waits[0])]
        for block in (0, 1):
            if block:
                parts.append(self.operators.delay(waits[1]))
            for index in range(shots):
                sampled = index == before
                parts.append(self.operators.excitation(angle[block], turn[block]))
                parts.append(
                    self.operators.readout(
                        turn[block],
                        role=AdcRole.SINGLE if sampled else AdcRole.NON_ACQUIRED,
                        is_echo=sampled,
                        duration_s=readout_s,
                    )
                )
        parts.append(self.operators.delay(waits[2]))
        return parts

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TI: npt.ArrayLike,
        flip: float | npt.ArrayLike,
        TRspgr: float | npt.ArrayLike,
        TRmp2rage: float | npt.ArrayLike,
        nshots: int | npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the two sampled magnetizations, along a trailing axis.

        Parameters
        ----------
        properties:
            ``T1`` in milliseconds, ``M0`` as a scaling, and the inversion
            efficiency.
        TI:
            The two inversion times in milliseconds, measured to the sampled
            shot of each block.
        flip:
            Excitation flip angle in degrees, one per block or one shared.
        TRspgr:
            Repetition time in milliseconds of one readout.
        TRmp2rage:
            Repetition time in milliseconds of the whole inversion block.
        nshots:
            Readouts per block, either the total -- halved for each block --
            or ``(before, after)`` the sampled shot.
        """
        radians = torch.pi / 180.0
        angle = _shared_or_two(radians * flip)
        before, after = _shots_either_side(nshots)
        inversion_s = TI.flatten() * 1e-3
        readout_s = TRspgr * 1e-3
        block_s = TRmp2rage * 1e-3

        efficiency = properties.get("inv_efficiency", 1.0)
        rate = 1e3 / properties["T1"]

        # The three waits the magnetization recovers through freely: before the
        # first block, between the two, and after the second.
        waits = (
            inversion_s[0] - before * readout_s,
            inversion_s[1] - inversion_s[0] - (after + before) * readout_s,
            block_s - inversion_s[1] - after * readout_s,
        )
        held = [torch.exp(-rate * wait) for wait in waits]
        shot = torch.exp(-rate * readout_s)
        turn = torch.cos(angle)
        shots = before + after

        # The steady state one whole block leaves behind, which is what the
        # inversion of the next block acts on.
        settled = _through_shots(1 - held[0], turn[0], shot, shots)
        settled = settled * held[1] + (1 - held[1])
        settled = _through_shots(settled, turn[1], shot, shots)
        settled = settled * held[2] + (1 - held[2])
        settled = settled / (
            1
            + efficiency
            * held[0]
            * held[1]
            * held[2]
            * (turn[0] * shot * turn[1] * shot) ** shots
        )

        # The first block reads what the inversion left, driven down over the
        # readouts before its centre.
        driven = _through_shots(
            -efficiency * settled * held[0] + (1 - held[0]), turn[0], shot, before
        )
        first = torch.sin(angle[0]) * driven

        # The second reads what the rest of the first block, the wait between
        # them and its own leading readouts leave.
        driven = _through_shots(driven, turn[0], shot, after)
        driven = _through_shots(driven * held[1] + (1 - held[1]), turn[1], shot, before)
        second = torch.sin(angle[1]) * driven

        # Both blocks carry the voxel shape and the signal is (..., voxel,
        # block), so one trailing axis lines the density up with them.
        density = properties.get("M0", 1.0)
        if torch.is_tensor(density):
            density = density[..., None]
        return density * torch.stack((first, second), dim=-1)


# %% private module subroutines


def _through_shots(
    held: Any, turn: torch.Tensor, shot: torch.Tensor, count: Any
) -> torch.Tensor:
    """Return what ``count`` spoiled readouts leave of what they find.

    Each readout tips the longitudinal magnetization down by ``turn`` and lets
    it recover by ``shot``, which drives whatever it started from towards a
    steady state of its own.
    """
    survives = (turn * shot) ** count
    return held * survives + (1 - shot) * (1 - survives) / (1 - turn * shot)


def _shared_or_two(value: torch.Tensor) -> torch.Tensor:
    """Return one value per block, sharing a single one between the two."""
    flat = value.flatten()
    return flat.repeat(2) if flat.numel() == 1 else flat


def _shots_either_side(nshots: int | npt.ArrayLike) -> tuple[Any, Any]:
    """Split the readouts into those before and after the sampled one."""
    counts = as_torch(nshots).flatten()
    if counts.numel() == 1:
        half = counts[0] // 2
        return half, half
    return counts[0], counts[1]


#: What each of the three free-recovery waits means when it comes out negative.
_COMPLAINTS = (
    "TI[0] must not precede the first MP2RAGE excitation",
    "TI[1] must leave room for both blocks' readouts between the inversion times",
    "TRmp2rage must leave room for the second block's readouts",
)
