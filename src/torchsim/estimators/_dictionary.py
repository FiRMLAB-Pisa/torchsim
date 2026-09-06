"""Memory-bounded matching against simulated signal dictionaries."""

from __future__ import annotations

__all__ = ["MatchResult", "DictionaryMatcher"]

from dataclasses import dataclass
from typing import Any

import torch

from .._calibrate import crossover
from .._execution import per_voxel
from ._grouped import Grouping, correlate, match_in_groups
from ._mapping import Estimator


@dataclass(frozen=True)
class MatchResult:
    """Best dictionary atoms and their complex least-squares scales.

    Attributes
    ----------
    parameters : torch.Tensor, optional
        ``(..., candidates, parameters)`` -- the parameter values of the atoms
        that matched. ``None`` where the dictionary carries no parameter
        values, which is what a matcher fitted to signals alone has.
    indices : torch.Tensor
        ``(..., candidates)`` -- which atom each candidate is, into the
        dictionary as it was fitted.
    scores : torch.Tensor
        ``(..., candidates)`` -- how well each candidate matched, as the
        normalized inner product the search maximized.
    scales : torch.Tensor
        ``(..., candidates)`` -- the complex amplitude the measurement is of
        the atom, which is the least-squares scale and carries the proton
        density and the receive phase.
    densities : torch.Tensor
        ``(..., candidates)`` -- the size of that amplitude, which is the
        proton density. Equal to ``scales.abs()``, and read off the score and
        the atom's own length rather than by touching the atom: a match scores
        ``|<y_hat, a_hat>|``, so multiplying by the measurement's length and
        dividing by the atom's is the least-squares scale already.
    """

    parameters: torch.Tensor | None
    indices: torch.Tensor
    scores: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor


class DictionaryMatcher(Estimator):
    """Match signals using normalized complex inner products.

    Parameters
    ----------
    acquisition : Simulator, optional
        The sequence being inverted: a simulator that ships with TorchSim, one
        written by subclassing :class:`~torchsim.model.Simulator`, or any other
        :class:`~torchsim.model.Simulator`. Every tissue property that is
        neither unknown nor measured separately is fixed on it beforehand, with
        the constructor or :meth:`~torchsim.model.Simulator.bind`. Leave it
        out to fit from signals handed to :meth:`fit` directly.
    dictionary : torch.Tensor, optional
        Simulated atoms shaped ``(n_atoms, n_contrasts)``. Leave it out and
        give an ``acquisition`` instead, and the atoms are simulated from it.
    parameters : torch.Tensor, optional
        Parameter values shaped ``(n_atoms, n_parameters)``. If provided, a
        call returns parameter estimates; otherwise it returns atom indices.
    query_chunk_size : int, optional
        Maximum measured signals compared at once.
    dictionary_chunk_size : int, optional
        Maximum dictionary atoms compared at once.
    top_k : int, optional
        Number of candidates retained by :meth:`match`. The first is the
        conventional dictionary-matching estimate.
    groups : int, optional
        Cluster the dictionary into this many groups and match against their
        representative signals first, ruling out whole groups before any atom
        in them is scored. Read :attr:`grouping` afterwards: its ``condition``
        says whether the representatives are still distinct enough to prune
        with, and its ``compression`` says how much shorter an inner product
        inside a group is.
    prune : float, optional
        How far below its best group score a voxel still considers a group,
        as a fraction of that score. Larger keeps more groups and costs more.
        The default is the value Cauley et al. [1]_ tuned on 280 groups of
        700 atoms; on a smaller dictionary or a coarser parameter grid it can rule
        out the group actually holding the match, which shows up as a handful
        of voxels landing several grid steps away. Widening it is the fix.

    Notes
    -----
    Compression comes first and is global: one temporal basis for the whole
    dictionary, which the signals are in too. Grouping then clusters within
    that basis, so the two savings multiply -- the basis shortens every inner
    product, the grouping cuts how many are taken. State a ``rank`` and the
    dictionary it matches against is compressed to ``(atoms, rank)``.

    The expensive operation is a matrix product. Torch therefore dispatches
    directly to the installed CPU BLAS or cuBLAS implementation; a separate
    C++ or Triton matrix-multiplication kernel would duplicate a faster
    vendor implementation. Chunking bounds the temporary score matrix.

    References
    ----------
    .. [1] Cauley, S. F., Setsompop, K., Ma, D., et al., "Fast group matching
       for MR fingerprinting reconstruction", Magnetic Resonance in Medicine
       74.2 (2015), pp. 523-528. https://doi.org/10.1002/mrm.25439
    """

    def __init__(
        self,
        acquisition: Any = None,
        *,
        dictionary: torch.Tensor | None = None,
        parameters: torch.Tensor | None = None,
        query_chunk_size: int = 4096,
        dictionary_chunk_size: int = 16384,
        top_k: int = 1,
        groups: int | None = None,
        prune: float = 5e-3,
    ) -> None:
        super().__init__(acquisition)
        if query_chunk_size < 1 or dictionary_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if groups is not None and groups < 1:
            raise ValueError(f"groups must be positive, got {groups}")
        if not 0.0 <= prune < 1.0:
            raise ValueError(f"prune must be in [0, 1), got {prune}")
        self.query_chunk_size = int(query_chunk_size)
        self.dictionary_chunk_size = int(dictionary_chunk_size)
        self.top_k = int(top_k)
        self.groups = None if groups is None else int(groups)
        self.prune = float(prune)
        self._grouping: Grouping | None = None
        self.register_buffer("dictionary", torch.empty(0))
        self.register_buffer("normalized_dictionary", torch.empty(0))
        self.register_buffer("dictionary_power", torch.empty(0))
        self.register_buffer("parameter_values", torch.empty(0))
        # Copies of the dictionary, one per device a match has reached.
        self._replicas: dict[str, DictionaryMatcher] = {}
        if dictionary is not None:
            self._adopt(dictionary, parameters)

    @property
    def fitted(self) -> bool:
        """Whether the matcher holds a dictionary."""
        return self.dictionary.numel() != 0

    @property
    def grouping(self) -> Grouping | None:
        """How the dictionary was clustered, or ``None`` if it was not."""
        return self._grouping

    def _apply(self, *args: Any, **kwargs: Any) -> DictionaryMatcher:
        """Keep the grouping beside the dictionary when the module moves.

        The clusters are ordinary attributes rather than buffers, because a
        grouping is ragged: each group keeps a basis of its own length.
        """
        moved = super()._apply(*args, **kwargs)
        if moved._grouping is not None:
            moved._grouping = moved._grouping.to(moved.dictionary.device)
        return moved

    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> DictionaryMatcher:
        """Adopt simulated signals as the dictionary to match against.

        Parameters
        ----------
        signals : torch.Tensor
            ``(samples, contrasts)`` -- the atoms.
        parameters : torch.Tensor
            ``(samples, parameters)`` -- what each atom stands for.
        known : torch.Tensor, optional
            Not supported. A dictionary spans one grid of parameters, and a
            property measured per voxel would need a different sub-dictionary
            for every voxel. Estimate it instead, or use a method that takes
            it as a feature.
        noise_std : float or torch.Tensor, optional
            Accepted and unused. A matched estimate comes from a normalized
            inner product, which noise on the atoms would only degrade -- the
            dictionary is the clean model the measurement is compared to.

        Returns
        -------
        DictionaryMatcher
            This matcher, holding the dictionary.

        Raises
        ------
        ValueError
            If ``known`` is given.
        """
        del noise_std
        if known is not None:
            raise ValueError(
                "a dictionary match cannot take a separately measured "
                "property; estimate it as an unknown instead"
            )
        self._adopt(signals, parameters)
        return self

    def _estimate_arrays(
        self, signals: torch.Tensor, known: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return best parameter values, or indices if none were supplied."""
        if known is not None:
            raise ValueError(
                "a dictionary match cannot take a separately measured property"
            )
        result = self.match(signals)
        if result.parameters is None:
            return result.indices[..., 0]
        if not self._unknown:
            # Fitted from bare arrays, so the columns are the caller's and an
            # unnamed one appended to them would be a surprise.
            return result.parameters[..., 0, :]
        return torch.cat(
            (result.parameters[..., 0, :], result.densities[..., :1]), dim=-1
        )

    def _extra_maps(
        self, measured: torch.Tensor, values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The proton density, which the match worked out on its way."""
        if values.shape[-1] <= len(self._unknown):
            return {}
        return {"M0": values[..., len(self._unknown)]}

    def _placed(self, signals: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        """Match under the execution policy, or ``None`` if none applies.

        Every voxel is matched against the same dictionary, so a volume too
        large for a card is streamed through it and two cards halve it. The
        dictionary crosses once per device; the volume is what moves.
        """
        atoms, contrasts = self.dictionary.shape
        outcome = per_voxel(
            [signals],
            bytes_per_voxel=contrasts * 8 + self.dictionary_chunk_size * 4,
            work=int(signals.shape[0]) * atoms * contrasts,
            crossover=lambda device: crossover(
                (atoms, contrasts, self.top_k),
                device,
                self._probe(contrasts),
                atoms * contrasts,
            ),
            body=lambda chunk, device: self._beside(device)._match_here(chunk[0]),
        )
        return outcome

    def _probe(self, contrasts: int) -> Any:
        """A closure the calibrator can time, running the real match."""

        def build(device: torch.device, voxels: int) -> Any:
            generator = torch.Generator(device=device).manual_seed(0)
            signals = torch.randn(
                voxels,
                contrasts,
                dtype=torch.float32,
                generator=generator,
                device=device,
            ).to(self.dictionary.dtype)
            replica = self._beside(device)
            return lambda: replica._match_here(signals)

        return build

    def _beside(self, device: torch.device) -> DictionaryMatcher:
        """This matcher with its dictionary on ``device``."""
        key = str(device)
        replica = self._replicas.get(key)
        if replica is None:
            if self.dictionary.device == device:
                replica = self
            else:
                replica = DictionaryMatcher(
                    query_chunk_size=self.query_chunk_size,
                    dictionary_chunk_size=self.dictionary_chunk_size,
                    top_k=self.top_k,
                    groups=self.groups,
                    prune=self.prune,
                )
                # Clustering is a property of the dictionary, not of where it
                # sits, so the replica moves it rather than repeating it.
                if self._grouping is not None:
                    replica._grouping = self._grouping.to(device)
                replica.dictionary = self.dictionary.to(device)
                replica.normalized_dictionary = self.normalized_dictionary.to(device)
                replica.dictionary_power = self.dictionary_power.to(device)
                replica.parameter_values = self.parameter_values.to(device)
            self._replicas[key] = replica
        return replica

    def _adopt(self, dictionary: torch.Tensor, parameters: torch.Tensor | None) -> None:
        """Keep these atoms, and what each of them stands for."""
        dictionary = torch.as_tensor(dictionary)
        if dictionary.ndim != 2 or dictionary.shape[0] < 1:
            raise ValueError("dictionary must have shape (atoms, contrasts)")
        if self.top_k > dictionary.shape[0]:
            raise ValueError("top_k must be between one and the atom count")
        if not torch.is_floating_point(dictionary) and not torch.is_complex(dictionary):
            dictionary = dictionary.to(torch.float32)
        norm = torch.linalg.vector_norm(dictionary, dim=-1).clamp_min(
            torch.finfo(dictionary.real.dtype).eps
        )
        self.dictionary = dictionary
        self.normalized_dictionary = dictionary / norm[:, None]
        self.dictionary_power = norm.square()
        self.parameter_values = _prepare_parameters(parameters, dictionary)
        self._grouping = (
            None
            if self.groups is None
            else Grouping.fit(dictionary, min(self.groups, dictionary.shape[0]))
        )
        self._replicas = {}

    @torch.no_grad()
    def match(self, signals: torch.Tensor) -> MatchResult:
        """Return the top matching atoms, scores, scales, and parameters."""
        if not self.fitted:
            raise RuntimeError("the matcher has no dictionary to match against")
        signals = torch.as_tensor(signals)
        if signals.shape[-1] != self.dictionary.shape[-1]:
            raise ValueError("signal and dictionary contrast counts differ")
        sample_shape = signals.shape[:-1]
        # Promoted, never narrowed: a complex measurement of a real-valued
        # model keeps both its parts, and correlate() reads them.
        signals = signals.reshape(-1, signals.shape[-1]).to(
            torch.promote_types(signals.dtype, self.dictionary.dtype)
        )
        placed = self._placed(signals)
        found = (
            placed
            if placed is not None
            else self._match_here(signals.to(self.dictionary.device))
        )
        return self._shaped(found, sample_shape)

    def _match_here(
        self, signals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match ``(voxels, contrasts)`` against the dictionary beside it.

        The indices, scores, scales and matched parameters come back flat
        along the voxel axis, which is the shape a chunk of a larger volume
        has to be in to be joined to the others.
        """
        signal_norm = torch.linalg.vector_norm(signals, dim=-1).clamp_min(
            torch.finfo(signals.real.dtype).eps
        )
        normalized_signals = signals / signal_norm[:, None]
        if self._grouping is not None:
            indices, scores = match_in_groups(
                normalized_signals, self._grouping, self.top_k, self.prune
            )
            return (
                indices,
                scores,
                self._density(signal_norm, scores, indices),
                *self._scaled(signals, indices),
            )

        score_chunks = []
        index_chunks = []
        for query in normalized_signals.split(self.query_chunk_size):
            best_scores = torch.empty(
                (query.shape[0], 0), dtype=query.real.dtype, device=query.device
            )
            best_indices = torch.empty(
                (query.shape[0], 0), dtype=torch.int64, device=query.device
            )
            for start in range(0, self.dictionary.shape[0], self.dictionary_chunk_size):
                stop = min(start + self.dictionary_chunk_size, self.dictionary.shape[0])
                scores = correlate(query, self.normalized_dictionary[start:stop])
                local_count = min(self.top_k, scores.shape[-1])
                local_scores, local_indices = torch.topk(scores, local_count, dim=-1)
                local_indices += start
                candidates = torch.cat((best_scores, local_scores), dim=-1)
                candidate_indices = torch.cat((best_indices, local_indices), dim=-1)
                keep = min(self.top_k, candidates.shape[-1])
                best_scores, selection = torch.topk(candidates, keep, dim=-1)
                best_indices = torch.gather(candidate_indices, -1, selection)
            score_chunks.append(best_scores)
            index_chunks.append(best_indices)

        scores = torch.cat(score_chunks, dim=0)
        indices = torch.cat(index_chunks, dim=0)
        return (
            indices,
            scores,
            self._density(signal_norm, scores, indices),
            *self._scaled(signals, indices),
        )

    def _density(
        self,
        signal_norm: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """The proton density, from the score and the atoms' own lengths.

        The search normalized both sides, so the score is a cosine and the
        sizes it divided out are the two norms. One of them is the
        measurement's and the other is stored per atom, which is a number each
        rather than a signal each -- so putting the scale back costs no atom.
        """
        lengths = self.dictionary_power[indices].sqrt()
        return signal_norm[:, None] * scores / lengths

    def _scaled(
        self, signals: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The complex least-squares scale of each matched atom, and what it means.

        The score says which atom, in a normalization that threw the size of
        the signal away; this puts it back, which is the proton density.
        """
        atoms = self.dictionary[indices]
        scales = (
            torch.sum(atoms.conj() * signals[:, None, :], dim=-1)
            / self.dictionary_power[indices]
        )
        matched = (
            torch.empty(
                (*indices.shape, 0),
                dtype=self.parameter_values.dtype,
                device=indices.device,
            )
            if self.parameter_values.numel() == 0
            else self.parameter_values[indices]
        )
        return scales, matched

    def _shaped(
        self,
        found: tuple[torch.Tensor, ...],
        sample_shape: torch.Size,
    ) -> MatchResult:
        """One flat match, given the voxel shape it came from."""
        indices, scores, densities, scales, matched = found
        output_shape = (*sample_shape, self.top_k)
        return MatchResult(
            parameters=None
            if matched.shape[-1] == 0
            else matched.reshape(*output_shape, -1),
            indices=indices.reshape(output_shape),
            scores=scores.reshape(output_shape),
            scales=scales.reshape(output_shape),
            densities=densities.reshape(output_shape),
        )


# %% private module subroutines


def _prepare_parameters(
    parameters: torch.Tensor | None,
    dictionary: torch.Tensor,
) -> torch.Tensor:
    if parameters is None:
        return torch.empty(0, dtype=torch.float32, device=dictionary.device)
    output = torch.as_tensor(
        parameters, dtype=dictionary.real.dtype, device=dictionary.device
    )
    if output.ndim == 1:
        output = output[:, None]
    if output.ndim != 2 or output.shape[0] != dictionary.shape[0]:
        raise ValueError("parameters must have shape (atoms, parameters)")
    return output
