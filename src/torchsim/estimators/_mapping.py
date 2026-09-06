"""Stating a parameter-mapping problem, and the machinery every method shares.

An estimator is made from the acquisition it inverts and whatever settings the
method itself has. What is unknown, over what range, and how noisy belongs to
the training step rather than to the estimator, so one estimator can be fitted
over a different sampling without being rebuilt::

    fitter = PERK(acquisition, n_features=1000)
    fitter.fit(T1=(200.0, 3000.0), T2=(10.0, 300.0), noise_std=0.01)
    maps = fitter.map(volume)

The properties are named as keywords. A mapping given positionally is the same
thing, and is the way to name one that collides with a keyword ``fit`` itself
takes.

The training set comes from the same simulator the design side optimizes, so
what an estimator is trained on cannot drift from what the scanner will
produce. An estimator that fits the model rather than a sampling of it reads
:attr:`~Estimator.acquisition` directly and ignores the signals it is handed.

Handing arrays in instead is the other way to fit any of these: a dictionary
that came from somewhere else is ``fit(signals=..., parameters=...)``, and then
:meth:`~Estimator.map` returns the parameter columns as a tensor rather than
named maps. What decides which is whether the estimator was told the names.
"""

from __future__ import annotations

__all__ = ["Estimator"]

from collections.abc import Mapping
from typing import Any

import torch

from .._subspace import Subspace
from ..model import Simulator

#: How many samples are simulated at once while a training set is drawn.
_CHUNK = 4096
#: Training draws, where nothing states a grid of its own.
_SAMPLES = 10_000


class Estimator(torch.nn.Module):
    """What every mapping method is, and the machinery all of them share.

    :class:`~torchsim.DictionaryMatcher`, :class:`~torchsim.LookupTable`,
    :class:`~torchsim.NonlinearLeastSquares` and :class:`~torchsim.PERK` are
    all one of these. What they inherit is the whole problem statement --
    :meth:`fit` naming the unknowns and drawing the training set from
    :attr:`acquisition`, :meth:`map` returning one named map per unknown --
    so a method differs from the next only in what it does with plain tensors.

    Writing another one is those two tensor steps. Subclass this, take the
    method's own settings in the constructor, and implement
    :meth:`_fit_arrays` and :meth:`_estimate_arrays`, which see signals and
    parameters as matrices and know nothing about names, subspaces or
    simulators. Everything above them then works unchanged.

    Parameters
    ----------
    acquisition : Simulator, optional
        The sequence being inverted, with any tissue property that is neither
        unknown nor measured already fixed on it. Left out where the estimator
        is given signals directly.

    Attributes
    ----------
    acquisition : Simulator or None
        The sequence being inverted.
    noise_std : float or torch.Tensor
        The noise :meth:`fit` was told the measurement has.
    rank : int or None
        The rank :meth:`fit` was asked to compress to, if any.
    seed : int or None
        The seed the training draw used, if one was given.
    """

    def __init__(self, acquisition: Simulator | None = None) -> None:
        super().__init__()
        self.acquisition = acquisition
        self.noise_std: float | torch.Tensor = 0.0
        self.rank: int | None = None
        self.seed: int | None = None
        self._unknown: dict[str, Any] = {}
        self._known: dict[str, Any] = {}
        self._subspace: Subspace | None = None
        self._fitted = False

    @property
    def unknown(self) -> tuple[str, ...]:
        """The properties being estimated, in the order the maps come back."""
        return tuple(self._unknown)

    @property
    def known(self) -> tuple[str, ...]:
        """The properties measured separately, in the order they are given."""
        return tuple(self._known)

    @property
    def subspace(self) -> Subspace | None:
        """The temporal basis in use, or ``None`` if the estimator works in full."""
        return self._subspace

    @property
    def trained(self) -> bool:
        """Whether the estimator has what it needs to map.

        A dictionary handed straight to the constructor counts, which is why
        this consults the subclass rather than only recording that :meth:`fit`
        was called.
        """
        return self._fitted or bool(getattr(self, "fitted", False))

    # -- fitting ---------------------------------------------------------

    def fit(
        self,
        unknown: Mapping[str, Any] | None = None,
        *,
        known: Mapping[str, Any] | None = None,
        noise_std: float | torch.Tensor = 0.0,
        rank: int | None = None,
        subspace: Subspace | None = None,
        seed: int | None = None,
        samples: int | None = None,
        chunk: int = _CHUNK,
        signals: torch.Tensor | None = None,
        parameters: torch.Tensor | None = None,
        **ranges: Any,
    ) -> Any:
        """Fit the estimator over a sampling of the tissue it will meet.

        Parameters
        ----------
        unknown : mapping, optional
            The properties being estimated, as ``{name: (low, high)}`` to draw
            uniformly over, or an array of values to use as given. The order
            is the order of the maps that come back. Naming them as keywords
            instead is the same thing and is usually shorter; a mapping is the
            way to name one that collides with a keyword here.
        known : mapping or torch.Tensor, optional
            Properties measured separately, as ``{name: range}`` or
            ``{name: values}``. They are drawn over here and reach the
            simulator, so the training signals carry their effect, and the
            measured maps are given again to :meth:`map`. A tensor alongside
            ``signals``, where the columns are given outright.
        noise_std : float or torch.Tensor, optional
            Standard deviation of the noise added to the training signals, in
            the units the signal is in -- a fully relaxed voxel is 1. This is
            what teaches a method how far to trust a measurement, so it should
            be the noise the scan actually has.
        rank : int, optional
            Fit a temporal :class:`~torchsim.Subspace` of this rank to the
            training signals and work in it. Every contrast dropped is
            arithmetic neither training nor mapping has to do;
            :attr:`subspace` reports what the compression kept.
        subspace : Subspace, optional
            Work in a basis fitted elsewhere rather than fitting one here --
            the one another estimator carries, or one
            :func:`~torchsim.simulate_subspace` produced. This is what lets a
            reconstruction and the estimator that reads its coefficients agree
            on the basis by construction. Not with ``rank``, which asks for a
            basis to be fitted.
        seed : int, optional
            Seed for the training draw.
        samples : int, optional
            How many tissues to draw. Stating a property as an array of values
            fixes this, so it may be left out; where everything is a range to
            draw from, it defaults to ten thousand.
        chunk : int, optional
            How many to simulate at a time.
        signals : torch.Tensor, optional
            ``(samples, contrasts)`` given outright, instead of simulated. The
            acquisition is then not consulted and :meth:`map` returns a tensor
            rather than named maps, unless names were given as well.
        parameters : torch.Tensor, optional
            ``(samples, unknowns)``, alongside ``signals``.
        ranges
            The unknown properties, named as keywords.

        Returns
        -------
        This estimator, fitted.

        Raises
        ------
        RuntimeError
            If nothing is given to fit from: no signals, and no acquisition to
            simulate them.
        ValueError
            If a name is both unknown and known, if a range is not a pair, if
            ``rank`` is not positive, or if both ``rank`` and ``subspace`` are
            given.
        """
        properties = {**(unknown or {}), **ranges}
        measured = known if isinstance(known, Mapping) else None
        overlap = set(properties) & set(measured or ())
        if overlap:
            raise ValueError(f"{sorted(overlap)} cannot be both unknown and known")
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank is not None and subspace is not None:
            raise ValueError(
                "give a rank to fit a basis here, or a subspace to work in one "
                "fitted elsewhere -- not both"
            )
        self._subspace = subspace
        if properties:
            self._unknown = dict(properties)
        if measured is not None:
            self._known = dict(measured)
        self.noise_std = noise_std
        self.rank = rank
        self.seed = seed

        if signals is None:
            if self.acquisition is None:
                raise RuntimeError(
                    "nothing to fit from: give this estimator the acquisition "
                    "it is inverting, or hand fit() the signals directly"
                )
            if not self._unknown:
                raise ValueError("name at least one property to estimate")
            if samples is None:
                samples = self._stated_samples() or _SAMPLES
            self._fit_simulated(samples, chunk=chunk, rank=rank, noise_std=noise_std)
        else:
            given = known if isinstance(known, torch.Tensor) else None
            if self._subspace is not None:
                signals = self._subspace.project(torch.as_tensor(signals))
            self._fit_arrays(signals, parameters, given, noise_std=noise_std)
        self._fitted = True
        return self

    def _fit_simulated(
        self,
        samples: int,
        *,
        chunk: int,
        rank: int | None,
        noise_std: float | torch.Tensor,
    ) -> None:
        """Simulate a training set, fit a basis over it if asked, then fit to it.

        The signals live only for the length of this call. They are what a
        basis is estimated from and what the method is fitted to, and nothing
        holds them afterwards.
        """
        signals, parameters, known = self.training_set(samples, chunk=chunk)
        if rank is not None:
            self._subspace = Subspace.fit(signals, rank)
        if self._subspace is not None:
            signals = self._subspace.project(signals)
        self._fit_arrays(signals, parameters, known, noise_std=noise_std)

    # -- the training set ------------------------------------------------

    def _stated_samples(self) -> int | None:
        """How many samples the given value arrays already fix, if any.

        Grids stated outright -- which is how a lookup table is built -- carry
        their own length, and repeating it as ``samples`` would be one more
        place for it to disagree. ``None`` where everything is a range to draw
        from, which fixes nothing.
        """
        lengths = {
            torch.as_tensor(spec).reshape(-1).numel()
            for spec in (*self._unknown.values(), *self._known.values())
            if not (isinstance(spec, (tuple, list)) and len(spec) == 2)
        }
        lengths.discard(1)
        if len(lengths) > 1:
            raise ValueError(
                f"properties given as values disagree on length: {sorted(lengths)}"
            )
        if not lengths:
            return None
        return int(lengths.pop())

    def training_set(
        self, samples: int, *, chunk: int = _CHUNK
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulate a training set: signals, unknowns, and knowns.

        Memory is ``samples`` by contrasts, because the signals are held once
        rather than resimulated for each pass a method makes over them. The
        properties are the ones :meth:`fit` was given, so this is what a caller
        wanting the training signals themselves calls after fitting.

        Parameters
        ----------
        samples : int
            How many tissues to draw.
        chunk : int, optional
            How many to simulate at a time.

        Returns
        -------
        tuple
            ``(signals, parameters, known)``, the last ``None`` when nothing
            is measured separately.

        Raises
        ------
        RuntimeError
            If the estimator has no acquisition, or was never told what is
            unknown.
        ValueError
            If ``samples`` or ``chunk`` is not positive, or if an array given
            for a property has a different length.
        """
        if self.acquisition is None:
            raise RuntimeError(
                "this estimator has no acquisition to simulate from; give one "
                "when it is made, or hand fit() the signals directly"
            )
        if not self._unknown:
            raise RuntimeError(
                "this estimator has not been told what is unknown; name the "
                "properties when fit() is called"
            )
        if samples < 1:
            raise ValueError(f"samples must be positive, got {samples}")
        if chunk < 1:
            raise ValueError(f"chunk must be positive, got {chunk}")
        drawn, parameters, known = self._draw_parameters(samples)
        pieces = []
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            given = {name: value[start:stop] for name, value in drawn.items()}
            pieces.append(torch.as_tensor(self.acquisition.simulate(**given)))
        signals = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return signals, parameters, known

    def _draw_parameters(
        self, samples: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
        """The tissue a training set covers, before anything is simulated."""
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        drawn = {
            name: _draw(name, spec, samples, generator)
            for name, spec in (*self._unknown.items(), *self._known.items())
        }
        parameters = torch.stack([drawn[name] for name in self._unknown], dim=-1)
        known = (
            torch.stack([drawn[name] for name in self._known], dim=-1)
            if self._known
            else None
        )
        return drawn, parameters, known

    # -- mapping ---------------------------------------------------------

    def map(
        self,
        volume: Any,
        known: Any = None,
        *,
        uncertainty: bool = False,
    ) -> Any:
        """Estimate the tissue a measurement came from.

        Parameters
        ----------
        volume : array-like
            ``(..., contrasts)``. Every leading axis is the voxel axis and
            comes back on the maps unchanged.
        known : mapping or torch.Tensor, optional
            The measured maps, under the names the estimator was given. Each
            is broadcast to the voxel shape.
        uncertainty : bool, optional
            Also return the standard deviation the noise leaves on each map.
            See :meth:`uncertainty_of` for what that number is and what it is
            not. Not every method can state one.

        Returns
        -------
        dict or torch.Tensor
            ``{name: map}`` where the estimator was told what it is solving
            for, each shaped like ``volume`` without its contrast axis. Where
            it was fitted from bare arrays, the parameter columns as a tensor.
            With ``uncertainty=True``, a pair of those -- the maps, and the
            standard deviations beside them.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        NotImplementedError
            If ``uncertainty`` is asked of a method that does not state one.
        ValueError
            If a measured map is missing, or has the wrong voxel count.
        """
        if not self._unknown:
            values = self._estimate_arrays(volume, known)
            if not uncertainty:
                return values
            signals = torch.as_tensor(volume)
            return values, self._spread(
                signals.reshape(-1, signals.shape[-1]), known, values
            )
        return self._named(volume, known, project=True, uncertainty=uncertainty)

    def forward(
        self,
        volume: Any,
        known: Any = None,
        *,
        uncertainty: bool = False,
    ) -> Any:
        """Alias of :meth:`map`, so a fitted estimator is callable."""
        return self.map(volume, known, uncertainty=uncertainty)

    def uncertainty_of(self, volume: Any, known: Any = None) -> Any:
        """The standard deviation the measurement noise leaves on each map.

        The noise :meth:`fit` was told about, propagated through the estimate
        this method makes -- so it says how much the answer would move if the
        scan were repeated, and it is a property of the method as much as of
        the sequence.

        It is not the error. An estimator that is biased is biased the same way
        in every realization, so repeating the measurement never reveals that
        part and this number does not contain it. Read against
        :func:`~torchsim.crlb`, which is the lowest standard deviation an
        unbiased estimate could have, it says how much of the distance to the
        truth the acquisition is responsible for.

        Parameters
        ----------
        volume : array-like
            ``(..., contrasts)``, as :meth:`map` takes.
        known : mapping or torch.Tensor, optional
            The measured maps, as :meth:`map` takes.

        Returns
        -------
        dict or torch.Tensor
            One standard deviation per unknown, shaped like the maps.

        Raises
        ------
        NotImplementedError
            If this method does not state one.
        """
        _, spread = self.map(volume, known, uncertainty=True)
        return spread

    def from_coefficients(
        self, coefficients: Any, known: Mapping[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        """Map coefficients that are already in this estimator's basis.

        A subspace reconstruction solves for the coefficients directly and
        never forms the contrast images, so what it returns is already
        projected. :meth:`map` would project it a second time; this does not.
        The basis it is in is :attr:`subspace`, which is also what the
        reconstruction was given.

        Parameters
        ----------
        coefficients : array-like
            ``(..., rank)``. Every leading axis is a voxel axis.
        known : mapping, optional
            The measured maps, as for :meth:`map`.

        Returns
        -------
        dict
            ``{name: map}``, shaped like ``coefficients`` without its last
            axis.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted, or was fitted without a
            rank, in which case there is no basis for these to be in.
        ValueError
            If the last axis is not the rank of that basis.
        """
        if self._subspace is None:
            raise RuntimeError(
                "this estimator has no subspace, so there are no coefficients "
                "to read; state a rank when it is fitted"
            )
        values = torch.as_tensor(coefficients)
        if values.shape[-1] != self._subspace.rank:
            raise ValueError(
                f"the basis has rank {self._subspace.rank}, "
                f"got {values.shape[-1]} coefficients"
            )
        return self._named(values, known, project=False)

    def _named(
        self,
        volume: Any,
        known: Mapping[str, Any] | None,
        *,
        project: bool,
        uncertainty: bool = False,
    ) -> Any:
        """Fill in the maps, from contrasts or from coefficients."""
        if not self.trained:
            raise RuntimeError("an estimator must be fitted before it maps")
        signals = torch.as_tensor(volume)
        # Where the maps belong is where the volume is, not where the method
        # happens to have been fitted.
        home = signals.device
        shape = signals.shape[:-1]
        voxels = int(torch.tensor(shape).prod()) if shape else 1
        signals = signals.reshape(voxels, signals.shape[-1])
        columns = _known_matrix(known, self._known, voxels)
        seen = signals
        if project and self._subspace is not None:
            seen = self._subspace.project(signals)
        values = self._estimate_arrays(seen, columns)

        def laid_out(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                name: matrix[..., column].reshape(shape).to(home)
                for column, name in enumerate(self._unknown)
            }

        maps = laid_out(values) | {
            name: value.reshape(shape).to(home)
            for name, value in self._extra_maps(signals, values).items()
        }
        if not uncertainty:
            return maps
        return maps, laid_out(self._spread(seen, columns, values, measured=signals))

    def _spread(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The standard deviations, with the noise the fit was told about."""
        spread = self._uncertainty_arrays(
            signals, known, values, measured=signals if measured is None else measured
        )
        if spread is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not state an uncertainty. A method "
                f"whose answer is a grid point moves in steps rather than "
                f"smoothly, so propagating the noise through it gives zero "
                f"rather than an error bar; PERK and NonlinearLeastSquares do "
                f"state one, and torchsim.crlb bounds what any unbiased method "
                f"could reach on this sequence."
            )
        return spread

    # -- what a subclass supplies ----------------------------------------

    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None,
        *,
        noise_std: float | torch.Tensor,
    ) -> None:
        """Fit to plain tensors."""
        raise NotImplementedError

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate ``(..., unknowns)`` from plain tensors."""
        raise NotImplementedError

    def _extra_maps(
        self, measured: torch.Tensor, values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Maps a method answers for besides the properties it was asked for."""
        return {}

    def _uncertainty_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor,
    ) -> torch.Tensor | None:
        """One standard deviation per unknown, or ``None`` where there is none.

        ``signals`` are as :meth:`_estimate_arrays` saw them -- projected into
        the basis where there is one -- and ``values`` is what it answered, so
        a method that linearizes about its own estimate has it to hand.
        ``measured`` is the measurement before any basis was applied, which is
        the domain the noise is independent in and carries whether the data is
        complex at all.
        """
        return None


# %% private module subroutines


def _draw(
    name: str, spec: Any, samples: int, generator: torch.Generator
) -> torch.Tensor:
    """One property's training values: a range to draw over, or values given."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        low, high = (float(value) for value in spec)
        if not high > low:
            raise ValueError(f"{name}: the range {spec} is not increasing")
        return low + (high - low) * torch.rand(samples, generator=generator)
    values = torch.as_tensor(spec, dtype=torch.float32).reshape(-1)
    if values.numel() == 1:
        return values.expand(samples).clone()
    if values.numel() != samples:
        raise ValueError(
            f"{name}: {values.numel()} values for {samples} training samples"
        )
    return values


def _known_matrix(
    given: Mapping[str, Any] | None,
    names: Mapping[str, Any],
    voxels: int,
) -> torch.Tensor | None:
    """The measured maps as ``(voxels, known)``, in the estimator's own order."""
    if not names:
        return None
    if given is None:
        raise ValueError(f"this estimator needs {sorted(names)} measured")
    if isinstance(given, torch.Tensor):
        return given
    columns = []
    for name in names:
        if name not in given:
            raise ValueError(f"{name} was not given")
        value = torch.as_tensor(given[name], dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.expand(voxels)
        elif value.numel() != voxels:
            raise ValueError(f"{name}: {value.numel()} values for {voxels} voxels")
        columns.append(value)
    return torch.stack(columns, dim=-1)
