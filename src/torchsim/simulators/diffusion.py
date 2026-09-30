"""Diffusion-weighted signal attenuation, in closed form."""

from __future__ import annotations

__all__ = ["DiffusionSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts

#: The six unique entries of a diffusion tensor, in the order the b-matrix
#: weights them.
TENSOR = ("Dxx", "Dyy", "Dzz", "Dxy", "Dxz", "Dyz")


class DiffusionSimulator(Simulator):
    """Attenuation by Gaussian diffusion, at a series of b-values.

    The signal is ``M0 exp(-b D)`` for an isotropic ``D`` [1]_, or
    ``M0 exp(-b g^T D g)`` for a diffusion tensor along unit gradient
    directions ``g``, which is what giving ``directions`` asks for: the six
    entries of :data:`TENSOR` are then the properties, and ``D`` is not.

    References
    ----------
    .. [1] Stejskal, E. O., Tanner, J. E., "Spin diffusion measurements: spin
       echoes in the presence of a time-dependent field gradient", The Journal
       of Chemical Physics 42.1 (1965), pp. 288-292.
       https://doi.org/10.1063/1.1695690

    Examples
    --------
    .. exec::

        from torchsim.simulators import DiffusionSimulator

        sequence = DiffusionSimulator(b=(0.0, 500.0, 1000.0, 2000.0))
        signal = sequence.simulate(D=(0.8, 3.0))
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "D": None,
            "M0": None,
            **dict.fromkeys(TENSOR),
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        b: float | npt.ArrayLike,
        directions: npt.ArrayLike | None = None,
    ) -> torch.Tensor:
        """Return the signal at each b-value.

        Parameters
        ----------
        properties:
            ``D`` or the entries of :data:`TENSOR` in square micrometres per
            millisecond, and ``M0`` as a scaling.
        b:
            b-values in seconds per square millimetre.
        directions:
            Unit gradient directions, ``(len(b), 3)``, for a tensor.

        Raises
        ------
        ValueError
            If ``directions`` is not one row of three per b-value.
        """
        held = across_contrasts(properties, b)
        # 1 s/mm^2 times 1 um^2/ms is 1e-3.
        weight = b * 1e-3
        if directions is None:
            exponent = weight * held["D"]
        else:
            g = torch.as_tensor(directions).to(weight.dtype)
            if g.shape != (*weight.shape, 3):
                raise ValueError(
                    f"directions must be {(*weight.shape, 3)}, one per b-value, "
                    f"got {tuple(g.shape)}"
                )
            x, y, z = g.unbind(-1)
            columns = (x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z)
            exponent = sum(
                weight * column * held.get(name, 0.0)
                for name, column in zip(TENSOR, columns, strict=True)
            )
        return held.get("M0", 1.0) * torch.exp(-exponent)
