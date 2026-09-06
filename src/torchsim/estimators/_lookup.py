"""Inverting a one-parameter model by interpolating its own curve."""

from __future__ import annotations

__all__ = ["LookupTable"]

from collections.abc import Callable
from typing import Any

import torch

from ._mapping import Estimator


class LookupTable(Estimator):
    """Read one unknown back from a monotonic signal curve.

    Where a model has a single unknown, a dictionary match degenerates: the
    atoms lie on a curve rather than filling a space, and the nearest one is
    found by looking along it. Interpolating between the two nearest atoms
    then costs nothing and removes the grid spacing from the answer entirely,
    which is what a matched estimate is otherwise limited by.

    This is how an MP2RAGE T1 map is made [1]_. The two inversion-recovery
    blocks are combined into one ratio per voxel, that ratio is a monotonic
    function of T1 over the range a brain spans, and the map is the inverse of
    that function.

    Parameters
    ----------
    acquisition : Simulator, optional
        The sequence being inverted: a simulator that ships with TorchSim, one
        written by subclassing :class:`~torchsim.model.Simulator`, or any other
        :class:`~torchsim.model.Simulator`. Every tissue property that is
        neither unknown nor measured separately is fixed on it beforehand, with
        the constructor or :meth:`~torchsim.model.Simulator.bind`. Leave it
        out to fit from signals handed to :meth:`fit` directly.
    combine : callable, optional
        Reduce a model's contrasts to the one number the curve is in, called
        as ``combine(signals)`` on ``(..., contrasts)`` and returning
        ``(...)``. Which combination makes a curve monotonic is a property of
        the sequence, not of the table, so it is stated here rather than
        assumed -- for MP2RAGE it is the unified image. A model with a single
        contrast needs none.
    monotonic : bool, optional
        Restrict the table to the longest monotonic run of the curve. A
        signal curve that turns back on itself has no inverse where it turns,
        and the turning points lie outside the range of interest.

    Notes
    -----
    A lookup is a binary search and one linear blend per voxel, so this
    estimator runs where its signals already are. Streaming a volume to a card
    for it would spend more time on the transfer than on the search.

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

        import torch
        from torchsim.estimators import LookupTable
        from torchsim.simulators import MP2RAGESimulator

        protocol = dict(
            TI=(800.0, 2700.0), flip=(4.0, 5.0),
            TRspgr=6.7, TRmp2rage=6000.0, nshots=128,
        )
        T1 = torch.arange(50.0, 5000.0, 50.0)
        blocks = MP2RAGESimulator(**protocol).simulate(
            T1=T1, inv_efficiency=0.96
        )
        unified = (blocks[:, 0] * blocks[:, 1]) / blocks.square().sum(-1)

        table = LookupTable().fit(signals=unified[:, None], parameters=T1[:, None])
        print(table.points, "points spanning", [round(v, 3) for v in table.span])
        print(table(torch.tensor([[0.1], [-0.2]])).flatten().tolist())

    """

    def __init__(
        self,
        acquisition: Any = None,
        *,
        combine: Callable[[torch.Tensor], torch.Tensor] | None = None,
        monotonic: bool = True,
    ) -> None:
        super().__init__(acquisition)
        self.combine = combine
        self.monotonic = bool(monotonic)
        self.register_buffer("intensity", torch.empty(0))
        self.register_buffer("parameter", torch.empty(0))

    @property
    def fitted(self) -> bool:
        """Whether the table holds a curve."""
        return self.intensity.numel() != 0

    @property
    def points(self) -> int:
        """How many points the table keeps."""
        return int(self.intensity.numel())

    @property
    def span(self) -> tuple[float, float]:
        """The signal range the table covers, low to high.

        A measurement outside it is read as the nearer endpoint, so a wide
        span is what stops a noisy background from pinning the map to a value
        the model never produced.
        """
        if not self.fitted:
            raise RuntimeError("the table has no curve to span")
        return float(self.intensity[0]), float(self.intensity[-1])

    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> LookupTable:
        """Build the table from a model's own curve.

        Parameters
        ----------
        signals : torch.Tensor
            ``(samples, contrasts)``. Several contrasts are reduced to one by
            the ``combine`` given to the constructor; without one, a single
            column is expected.
        parameters : torch.Tensor
            ``(samples, 1)`` or ``(samples,)`` -- the single unknown.
        known : torch.Tensor, optional
            Not supported. A table is one curve, and a property measured per
            voxel would need a curve for every voxel.
        noise_std : float or torch.Tensor, optional
            Accepted and unused. The table is the clean model a measurement is
            read against.

        Returns
        -------
        LookupTable
            This table, holding the curve.

        Raises
        ------
        ValueError
            If ``known`` is given, if either input carries more than one
            column, or if the curve is nowhere monotonic.
        """
        del noise_std
        if known is not None:
            raise ValueError(
                "a lookup table cannot take a separately measured property; "
                "estimate it as an unknown instead"
            )
        curve = _one_column(self._reduced(signals), "signals")
        values = _one_column(parameters, "parameters")
        if curve.numel() != values.numel():
            raise ValueError("signals and parameters must have equal length")
        if curve.numel() < 2:
            raise ValueError("a table needs at least two points")

        # The curve is a function of the parameter, so it is read in that
        # order whatever order the samples arrived in.
        order = torch.argsort(values)
        curve, values = curve[order], values[order]
        if self.monotonic:
            curve, values = _monotonic_run(curve, values)

        # Interpolation reads along increasing signal, which is the opposite
        # direction whenever the curve falls with the parameter.
        if bool(curve[0] > curve[-1]):
            curve, values = curve.flip(0), values.flip(0)
        self.intensity = curve.contiguous()
        self.parameter = values.contiguous()
        return self

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the parameter each signal stands for.

        Parameters
        ----------
        signals : torch.Tensor
            ``(..., contrasts)``, reduced by ``combine`` as in :meth:`fit`.
            Values outside the table's span are read as the nearer endpoint.
        known : torch.Tensor, optional
            Not supported.

        Returns
        -------
        torch.Tensor
            ``(..., 1)``, one estimate per signal.

        Raises
        ------
        ValueError
            If ``known`` is given or the signals carry more than one column.
        RuntimeError
            If the table has not been fitted.
        """
        if known is not None:
            raise ValueError(
                "a lookup table cannot take a separately measured property"
            )
        if not self.fitted:
            raise RuntimeError("the table has no curve to read")
        signals = self._reduced(torch.as_tensor(signals))
        shape = signals.shape[:-1] if signals.shape[-1:] == (1,) else signals.shape
        flat = signals.reshape(-1)
        if signals.ndim and signals.shape[-1] != 1 and signals.numel() != flat.numel():
            raise ValueError(
                "signals carry several contrasts; give the table a "
                "'combine' that reduces them to one"
            )

        intensity = self.intensity.to(flat.device)
        parameter = self.parameter.to(flat.device)
        found = _interpolate(flat.to(intensity.dtype), intensity, parameter)
        return found.reshape(*shape, 1)

    def _reduced(self, signals: Any) -> torch.Tensor:
        """Reduce the contrasts to the one number the curve is in."""
        signals = torch.as_tensor(signals)
        if self.combine is None:
            return signals
        return torch.as_tensor(self.combine(signals))


# %% private module subroutines


def _one_column(values: Any, name: str) -> torch.Tensor:
    """Return ``values`` as a flat vector, refusing a second column."""
    tensor = torch.as_tensor(values)
    if tensor.ndim > 2 or (tensor.ndim == 2 and tensor.shape[-1] != 1):
        raise ValueError(
            f"{name} must carry a single column for a lookup table, got "
            f"shape {tuple(tensor.shape)}"
        )
    tensor = tensor.reshape(-1)
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real; combine the contrasts first")
    return tensor.to(torch.float32) if not tensor.is_floating_point() else tensor


def _monotonic_run(
    curve: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the longest stretch over which the curve does not turn back.

    A signal curve is invertible only where it is monotonic. The stretch
    between its extremes is the one an inversion-recovery contrast is read
    over, and the turning points sit outside the parameter range of interest.
    """
    step = torch.diff(curve)
    if not bool((step != 0).any()):
        raise ValueError("the signal curve is constant and cannot be inverted")

    rising = step > 0
    best_start = best_stop = 0
    start = 0
    for index in range(1, len(step) + 1):
        turned = index == len(step) or bool(rising[index] != rising[start])
        if turned:
            if index - start > best_stop - best_start:
                best_start, best_stop = start, index
            start = index
    return curve[best_start : best_stop + 1], values[best_start : best_stop + 1]


def _interpolate(
    query: torch.Tensor, grid: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation along an increasing ``grid``, clamped at its ends."""
    upper = torch.searchsorted(grid, query.contiguous()).clamp(1, len(grid) - 1)
    lower = upper - 1
    left, right = grid[lower], grid[upper]
    # Equal neighbours would divide by zero; the blend is then either endpoint.
    weight = (query - left) / (right - left).clamp_min(torch.finfo(grid.dtype).tiny)
    return torch.lerp(values[lower], values[upper], weight.clamp(0.0, 1.0))
