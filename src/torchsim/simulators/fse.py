"""Fast spin echo."""

from __future__ import annotations

__all__ = ["FSESimulator", "HyperechoSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import REFOCUSED, Simulator, SpinPhysics
from ..sequence._array import as_torch, matched


class FSESimulator(Simulator):
    """A refocused echo train, sampled at every echo.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import FSESimulator

        sequence = FSESimulator(flip=180.0 * torch.ones(128), ESP=2.0, TR=5000.0)
        signal = sequence.simulate(T1=1000.0, T2=80.0)

    """

    model = SpinPhysics(
        properties={"T1": "t1_ms", "T2": "t2_ms", "M0": None, "B1": "b1"},
        operators=REFOCUSED,
    )
    states = 10

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> list:
        """Return the train, placed at the echo times it is timed from.

        Parameters
        ----------
        flip : float or array-like
            Refocusing flip angles in degrees, one per echo.
        ESP : float or array-like
            Echo spacing in milliseconds.
        phases : float or array-like, optional
            Refocusing phases in degrees.
        exc_flip, exc_phase : float, optional
            The excitation, in degrees.
        TR : float or array-like, optional
            Repetition time in milliseconds, which sets how far the
            longitudinal magnetization recovers before the next train.
        """
        angles = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, angles))
        return self._train(angles, turns, turns, ESP, exc_flip, exc_phase)

    def _train(
        self,
        angles: torch.Tensor,
        turns: torch.Tensor,
        demodulation: torch.Tensor,
        ESP: Any,
        exc_flip: float,
        exc_phase: float,
    ) -> list:
        """The excitation, then each refocusing pulse and the echo it forms."""
        # Not tensorized: a scalar spacing keeps the precision the caller
        # gave it, and the echo times are what the whole train is placed by.
        spacing_s = ESP * 1e-3

        parts = [
            (
                0.0,
                self.operators.excitation(
                    torch.pi / 180.0 * exc_flip, torch.pi / 180.0 * exc_phase
                ),
            )
        ]
        for index in range(angles.shape[-1]):
            echo_s = (index + 1) * spacing_s
            parts.append(
                (
                    echo_s - 0.5 * spacing_s,
                    self.operators.refocusing(angles[..., index], turns[..., index]),
                )
            )
            parts.append((echo_s, self.operators.readout(demodulation[..., index])))
        return parts

    def _echoes(self, played: Mapping[str, Any]) -> int:
        """How many echoes the train records."""
        return torch.atleast_1d(as_torch(played["flip"])).shape[-1]

    def repetition_s(self, played_s: Any, **protocol: Any) -> Any:
        """Return the TR, which the train waits out before the next one."""
        del played_s
        return protocol.get("TR", 1e6) * 1e-3

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Simulate one train, then let it recover for what is left of the TR."""
        played = self.played(**sequence)
        signal = super().evaluate(properties, **sequence)
        echoes = self._echoes(played)

        def beside(value: Any) -> Any:
            """Put a value where the signal came back from.

            A run may be asked for a device the tissue was not written on, and
            what is worked out here multiplies the signal rather than reaching
            a kernel.
            """
            return value.to(signal.device) if torch.is_tensor(value) else value

        left_ms = beside(played.get("TR", 1e6)) - beside(played["ESP"]) * echoes
        recovered = torch.exp(-1e3 / beside(properties["T1"]) * left_ms * 1e-3)
        # Both carry the tissue shape and the signal is (..., tissue, echo), so
        # one trailing axis lines them up and any leading train axis broadcasts.
        recovered = recovered[..., None]
        density = beside(properties.get("M0", 1.0))
        if torch.is_tensor(density):
            density = density[..., None]
        return density * signal * (1 - recovered) / (1 - recovered * signal)


class HyperechoSimulator(FSESimulator):
    """A refocused train mirrored about a central refocusing pulse.

    The pulses after the central one replay those before it in reverse order,
    each with the opposite flip angle about the opposite phase -- ``-alpha``
    about ``-phi``, which is ``alpha`` at ``180 - phi`` degrees -- so every
    coherence pathway the first half split is brought back by the last echo,
    the hyperecho [1]_. But for relaxation, that echo is the full spin echo of
    the excitation, whatever the flip angles before it. Every echo is
    demodulated at zero phase, the axis of the central pulse. With one flip
    angle at zero phase throughout, this is the sequence BART's ``epg`` plays
    as Hyperecho.

    References
    ----------
    .. [1] Hennig, J., Scheffler, K., "Hyperechoes", Magnetic Resonance in
       Medicine 46.1 (2001), pp. 6-12. https://doi.org/10.1002/mrm.1153

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import HyperechoSimulator

        sequence = HyperechoSimulator(flip=60.0 * torch.ones(5), ESP=10.0)
        signal = sequence.simulate(T1=1000.0, T2=80.0)
        print(signal.shape)

    """

    # The hyperecho is brought back from the highest order the first half
    # reached, so the orders are sized from the whole winding.
    states = None

    def layout(
        self,
        *,
        flip: float | npt.ArrayLike,
        ESP: float | npt.ArrayLike,
        phases: float | npt.ArrayLike = 0.0,
        refocusing: float = 180.0,
        exc_flip: float = 90.0,
        exc_phase: float = 90.0,
        TR: float | npt.ArrayLike = 1e6,
    ) -> list:
        """Return the train, placed at the echo times it is timed from.

        Parameters
        ----------
        flip : float or array-like
            Flip angles in degrees of the refocusing pulses before the central
            one, one per echo; the train holds twice as many and one more.
        ESP : float or array-like
            Echo spacing in milliseconds.
        phases : float or array-like, optional
            Phases in degrees of the refocusing pulses before the central one.
        refocusing : float, optional
            Flip angle in degrees of the central pulse, played at zero phase.
        exc_flip, exc_phase : float, optional
            The excitation, in degrees.
        TR : float or array-like, optional
            Repetition time in milliseconds, which sets how far the
            longitudinal magnetization recovers before the next train.
        """
        before = torch.deg2rad(torch.atleast_1d(as_torch(flip)))
        turns = torch.deg2rad(matched(phases, before))
        central = torch.full_like(before[..., :1], torch.pi / 180.0 * refocusing)
        angles = torch.cat([before, central, before.flip(-1)], dim=-1)
        turns = torch.cat(
            [turns, torch.zeros_like(central), torch.pi - turns.flip(-1)], dim=-1
        )
        return self._train(
            angles, turns, torch.zeros_like(turns), ESP, exc_flip, exc_phase
        )

    def _echoes(self, played: Mapping[str, Any]) -> int:
        return 2 * super()._echoes(played) + 1
