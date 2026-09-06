"""Fitting a model to every voxel at once, by damped Gauss-Newton."""

from __future__ import annotations

__all__ = ["NonlinearLeastSquares"]

from collections.abc import Mapping
from typing import Any

import torch

from .._bounds import bound_of
from ..optim._design import crlb
from ..recon import GaussNewton, ModelOperator, TrustRegion, direct
from ._mapping import Estimator


class NonlinearLeastSquares(Estimator):
    """Levenberg-Marquardt, stepping every voxel together.

    Where a dictionary spans a grid whose size is the product of the parameter
    ranges, a nonlinear fit walks downhill from a starting guess and pays
    nothing for a third parameter beyond a third column of the Jacobian. What
    it gives up is the guarantee: it finds a local minimum of the residual,
    and which one depends on where it started.

    This is an :class:`~torchsim.Estimator` face on
    :class:`~torchsim.recon.GaussNewton`, and holds no algorithm of its own.
    Fitting images voxel by voxel and reconstructing maps from k-space are the
    same loop over the same :class:`~torchsim.recon.ModelOperator` with
    nothing encoding the voxels together, so what is here is only the
    adaptation: where the fit starts, and what the training set is for.

    The loop it runs by default carries a per-voxel trust region, so every
    voxel takes its step in the same pass, carries its own damping, accepts or
    rejects on its own, and drops out when it has converged -- and the
    remaining ones close up, so late iterations cost only what is left.

    Parameters
    ----------
    acquisition : Simulator, optional
        The sequence being inverted: a simulator that ships with TorchSim, one
        written by subclassing :class:`~torchsim.model.Simulator`, or any other
        :class:`~torchsim.model.Simulator`. Every tissue property that is
        neither unknown nor measured separately is fixed on it beforehand, with
        the constructor or :meth:`~torchsim.model.Simulator.bind`. Leave it
        out to fit from signals handed to :meth:`fit` directly.
    bounds : mapping, optional
        ``{name: (low, high)}``, either end ``None`` for unbounded. A bound is
        kept by fitting a transformed variable rather than by clipping a
        result, so no iterate ever leaves the interval and the bound cannot be
        sitting exactly on the answer. It also puts every parameter on the
        same scale whatever its units, which is what the damping term assumes.
    initial : mapping, optional
        ``{name: value}`` to start from, which must be strictly inside that
        property's bound. Without one, :meth:`fit` takes the median of the
        parameters the training set drew.
    loop : GaussNewton, optional
        The solve to run, and where every knob it has lives -- how many steps,
        how the damping moves, which tolerance stops a voxel. The default is
        Levenberg-Marquardt: a :class:`~torchsim.recon.TrustRegion` over
        :func:`~torchsim.recon.direct`, twenty steps.

    Notes
    -----
    **Equality constraints are written into the model, not declared here.**
    A constraint that fixes one parameter in terms of the others removes a
    degree of freedom, so the way to impose it is to not have that freedom:
    write the model in terms of the parameters that remain. For a fat-water
    fit where the two fractions must sum to one, make the fat fraction ``f``
    the only unknown and write water as ``1 - f`` inside the model. The
    constraint then holds identically at every iterate, rather than being
    restored after each one.

    The solve is iterative and runs without building a graph, so an estimate
    carries no gradient with respect to the measurement.

    Examples
    --------
    .. code-block:: python

        fit = NonlinearLeastSquares(
            FSESimulator(ESP=5.0, TR=1800.0, flip=train),
            bounds={"T2": (1.0, 500.0)},
        ).fit(T1=(200.0, 3000.0), T2=(10.0, 300.0))
        maps = fit.map(volume)

    A solve that needs more room, or a different one entirely:

    .. code-block:: python

        NonlinearLeastSquares(
            bounds={"T2": (1.0, 500.0)},
            loop=GaussNewton(TrustRegion(tau=1e-3), solve=direct,
                             max_iterations=60),
        )
    """

    def __init__(
        self,
        acquisition: Any = None,
        *,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        initial: Mapping[str, float] | None = None,
        loop: GaussNewton | None = None,
    ) -> None:
        super().__init__(acquisition)
        self.bounds = dict(bounds or {})
        self.initial = dict(initial or {})
        self.loop = (
            loop
            if loop is not None
            else GaussNewton(TrustRegion(), solve=direct, max_iterations=20)
        )
        self._start: torch.Tensor | None = None
        #: Steps the last solve took, and how many voxels ran out of them.
        self.iterations = 0
        self.unconverged = 0

    @property
    def fitted(self) -> bool:
        """Whether a model and a starting point are both in place."""
        return self.acquisition is not None and self._start is not None

    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> NonlinearLeastSquares:
        """Take a starting point from the parameters the training set drew.

        The training signals are not needed -- the model is what this fits
        against -- but the parameters say what range the answer is in, and
        their median is a better first guess than the middle of a bound.

        Parameters
        ----------
        signals : torch.Tensor
            Ignored. Accepted so that a fit and a learned method are called
            the same way.
        parameters : torch.Tensor
            ``(samples, parameters)``, in the order given to :meth:`bind`.
        known : torch.Tensor, optional
            Ignored, for the same reason.
        noise_std : float or torch.Tensor, optional
            Accepted and unused. A least-squares fit weights every contrast
            alike, which is what uniform noise implies.

        Returns
        -------
        NonlinearLeastSquares
            This estimator, ready to be called.

        Raises
        ------
        ValueError
            If a starting value is not strictly inside its bound.
        RuntimeError
            If no model has been bound.
        """
        del signals, known, noise_std
        stray = {name for given in (self.bounds, self.initial) for name in given} - set(
            self.unknown
        )
        if stray:
            raise ValueError(
                f"bounds or initial name {sorted(stray)}, which "
                f"{'is' if len(stray) == 1 else 'are'} not being estimated"
            )
        if self.acquisition is None:
            raise RuntimeError(
                "no model to fit; give this estimator the acquisition it is "
                "inverting when it is made"
            )
        drawn = torch.as_tensor(parameters).reshape(-1, len(self.unknown))
        median = drawn.to(torch.float32).median(dim=0).values
        start = []
        for index, name in enumerate(self.unknown):
            stated = name in self.initial
            value = float(self.initial[name]) if stated else float(median[index])
            low, high = bound_of(self.bounds, name)
            # The transformed variable is infinite at a bound, so a fit that
            # started there would have no direction to move in.
            if (low is not None and value <= low) or (
                high is not None and value >= high
            ):
                whose = (
                    "the starting value"
                    if stated
                    else "the median of the training range"
                )
                raise ValueError(
                    f"{name}: {whose}, {value:g}, is not strictly inside its "
                    f"bound ({low}, {high})"
                )
            start.append(torch.as_tensor(value, dtype=torch.float32))
        self._start = torch.stack(start)
        return self

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the parameters that best explain each signal.

        Parameters
        ----------
        signals : torch.Tensor
            ``(..., contrasts)``.
        known : torch.Tensor, optional
            ``(..., knowns)``, the properties measured separately.

        Returns
        -------
        torch.Tensor
            ``(..., parameters)``, in the order given to :meth:`bind`.

        Raises
        ------
        RuntimeError
            If no model has been bound or no starting point chosen.
        """
        if not self.fitted:
            raise RuntimeError(
                "the estimator has no model and starting point to fit from"
            )
        signals = torch.as_tensor(signals)
        shape = signals.shape[:-1]
        measured = signals.reshape(-1, signals.shape[-1])
        given = (
            torch.as_tensor(known).reshape(-1, len(self.known))
            if known is not None
            else None
        )
        found = self._solve(measured, given)
        return found.reshape(*shape, len(self.unknown))

    def _uncertainty_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor,
    ) -> torch.Tensor:
        """The standard error of the fit, from its own sensitivity.

        A least-squares solution moves with the noise by as much as the model
        is insensitive to the parameters there, which is the inverse Fisher
        matrix at the solution -- the same quantity :func:`~torchsim.crlb`
        bounds an unbiased estimate by, read at the answer rather than at a
        truth nobody has. It is a linearization about the solution, so it is
        the standard error a fit reports and holds where the residual is small
        enough that the model is straight across it.
        """
        scale = torch.as_tensor(
            self.noise_std, dtype=torch.float32, device=values.device
        )
        if not torch.any(scale != 0):
            return torch.zeros_like(values)
        acquisition = self.acquisition
        if known is not None:
            acquisition = acquisition.bind(
                **{name: known[:, index] for index, name in enumerate(self.known)}
            )
        _, sensitivity = acquisition.jacobian(
            list(self.unknown),
            **{name: values[..., i] for i, name in enumerate(self.unknown)},
        )
        if self.subspace is not None:
            sensitivity = self.subspace.project(sensitivity)
        # Per-voxel noise divides the sensitivity rather than the variance,
        # which is the same thing and takes a map as readily as a number.
        sensitivity = sensitivity / scale.reshape(
            *scale.shape, *((1,) * (sensitivity.ndim - scale.ndim))
        )
        return (
            crlb(sensitivity, noise_variance=1.0, singular="infinite")
            .clamp_min(0.0)
            .sqrt()
        )

    def _solve(
        self, measured: torch.Tensor, known: torch.Tensor | None
    ) -> torch.Tensor:
        """Levenberg-Marquardt over every voxel, compacting as they finish.

        The loop is :class:`~torchsim.recon.GaussNewton` under a per-voxel
        trust region, and the model, its bounds and its derivative are a
        :class:`~torchsim.recon.ModelOperator` -- the same two pieces a
        model-based reconstruction is built from, with nothing encoding the
        voxels together.
        """
        operator = self._operator(measured.shape[0], known)
        found = self.loop.minimize(operator, measured, self._at(operator, measured))
        self.iterations = found.iterations
        self.unconverged = found.unconverged
        maps = operator.split(found.x)
        return torch.stack([maps[name] for name in self.unknown], dim=-1)

    def _operator(self, voxels: int, known: torch.Tensor | None) -> ModelOperator:
        """The model to fit, with anything measured separately held on it."""
        acquisition = self.acquisition
        if known is not None:
            acquisition = acquisition.bind(
                **{name: known[:, index] for index, name in enumerate(self.known)}
            )
        return ModelOperator(
            acquisition,
            *self.unknown,
            bounds=self.bounds,
            amplitude=False,
            subspace=self.subspace,
        )

    def _at(self, operator: ModelOperator, measured: torch.Tensor) -> torch.Tensor:
        """Every voxel started from the same point, as variables to solve for."""
        start = operator.initial(
            measured.shape[:1],
            **{
                name: float(self._start[index])
                for index, name in enumerate(self.unknown)
            },
        )
        return start.to(measured.device)
