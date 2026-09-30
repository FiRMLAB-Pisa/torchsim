"""The Z-spectrum as a sum of Lorentzian lines, in closed form."""

from __future__ import annotations

__all__ = ["LorentzianSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator
from ..sequence._array import arrays
from ._contrast import across_contrasts


class LorentzianSimulator(Simulator):
    """A Z-spectrum as equilibrium minus one Lorentzian line per pool.

    Each pool saturates the water signal along a Lorentzian line centred on
    its own offset [1]_, and the lines add:

    ``M0 (1 - sum_i A_i (W_i / 2)^2 / ((W_i / 2)^2 + (offset - shift_i)^2))``

    Pool ``i``, counted from one, exposes ``pool{i}_amplitude``,
    ``pool{i}_width`` (the full width at half maximum) and ``pool{i}_shift``.

    Parameters
    ----------
    pools:
        How many Lorentzian lines.
    **values
        Properties or sequence arguments to fix, as for any simulator.

    References
    ----------
    .. [1] Zaiss, M., Schmitt, B., Bachert, P., "Quantitative separation of
       CEST effect from magnetization transfer and spillover effects by
       Lorentzian-line-fit analysis of z-spectra", Journal of Magnetic
       Resonance 211.2 (2011), pp. 149-155.
       https://doi.org/10.1016/j.jmr.2011.05.001

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import LorentzianSimulator

        sequence = LorentzianSimulator(2, offsets=torch.linspace(-5.0, 5.0, 41))
        signal = sequence.simulate(
            pool1_amplitude=0.9, pool1_width=1.5, pool1_shift=0.0,
            pool2_amplitude=0.1, pool2_width=2.0, pool2_shift=3.5,
        )
        print(signal.shape)

    """

    properties = ("M0",)
    pools: int = 1

    def __init__(self, pools: int = 1, **values: Any) -> None:
        self.pools = int(pools)
        if self.pools < 1:
            raise ValueError(f"a Z-spectrum needs at least one pool, got {pools}")
        self.properties = (
            "M0",
            *(
                f"pool{index}_{what}"
                for index in range(1, self.pools + 1)
                for what in ("amplitude", "width", "shift")
            ),
        )
        super().__init__(**values)

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        offsets: npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the saturated water signal at each offset.

        Parameters
        ----------
        properties:
            ``pool{i}_amplitude`` as a fraction of ``M0``, ``pool{i}_width``
            and ``pool{i}_shift`` in the unit of ``offsets``, and ``M0`` as a
            scaling.
        offsets:
            Saturation frequency offsets from water, conventionally in ppm.
        """
        held = across_contrasts(properties, offsets)
        remaining = 1.0
        for index in range(1, self.pools + 1):
            amplitude = held[f"pool{index}_amplitude"]
            half = held[f"pool{index}_width"] / 2
            distance = offsets - held[f"pool{index}_shift"]
            remaining = remaining - amplitude * half**2 / (half**2 + distance**2)
        return held.get("M0", 1.0) * remaining
