"""Scalable parameter estimation via Gaussian random Fourier features."""

from __future__ import annotations

__all__ = ["PERK"]

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal

import torch

from .._calibrate import crossover
from .._execution import one_device, per_voxel
from ._mapping import Estimator


class PERK(Estimator):
    """Parameter estimation via regression with kernels.

    This is the random-Fourier-feature form of PERK [1]_. Training accumulates
    the
    required covariances in chunks, so memory depends on the feature order
    rather than the number of simulated training samples. Estimation uses only
    ``cos`` and matrix multiplication and therefore dispatches to optimized
    Torch CPU/CUDA kernels without a separate native implementation.

    Parameters
    ----------
    acquisition : Simulator, optional
        The sequence being inverted: a simulator that ships with TorchSim, one
        written by subclassing :class:`~torchsim.model.Simulator`, or any other
        :class:`~torchsim.model.Simulator`. Every tissue property that is
        neither unknown nor measured separately is fixed on it beforehand, with
        the constructor or :meth:`~torchsim.model.Simulator.bind`. Leave it
        out to fit from signals handed to :meth:`fit` directly.
    n_features : int, optional
        Number of random Fourier features. The default is ``1000``.
    length_scale : float or tensor, optional
        Gaussian-kernel length scale for every signal/known-parameter feature.
        ``None`` estimates it from the training standard deviation.
    regularization : float, optional
        Tikhonov parameter added to the feature covariance. The default is
        ``1e-5``.
    chunk_size : int, optional
        Maximum samples transformed at once. The default is ``65536``.
    feature_seed : int, optional
        Seed used for the fixed random feature map. Separate from the seed
        :meth:`fit` draws the training set with.
    complex_mode : {"cartesian", "magnitude"}, optional
        Representation used for complex signals. Magnitude data are often
        preferable when image phase is a nuisance parameter.
    normalize : bool, optional
        Normalize every signal vector before feature projection. This removes
        an unknown global scale when it is not itself a target.
    stream : bool, optional
        Simulate the training set a chunk at a time and accumulate the
        covariances as it goes, so memory follows the feature order rather
        than the number of training samples. The dictionary is never held, and
        so cannot have a basis estimated from it: a ``rank`` given to
        :meth:`fit` needs the default. A basis fitted elsewhere, passed to
        :meth:`fit` as ``subspace=``, streams perfectly well -- each chunk is
        projected as it is simulated.

    Notes
    -----
    Cartesian complex signals are represented as interleaved real/imaginary
    features. The learned estimator remains differentiable with respect to its
    input, which permits its use inside a larger reconstruction network.

    References
    ----------
    .. [1] Nataraj, G., Nielsen, J.-F., Scott, C., Fessler, J. A.,
       "Dictionary-free MRI PERK: parameter estimation via regression with
       kernels", IEEE Transactions on Medical Imaging 37.9 (2018),
       pp. 2103-2114. https://doi.org/10.1109/TMI.2018.2817547
    """

    def __init__(
        self,
        acquisition: Any = None,
        *,
        n_features: int = 1000,
        length_scale: float | torch.Tensor | None = None,
        regularization: float = 1e-5,
        chunk_size: int = 65536,
        feature_seed: int | None = None,
        complex_mode: Literal["cartesian", "magnitude"] = "cartesian",
        normalize: bool = False,
        stream: bool = False,
        uncertainty: bool = True,
    ) -> None:
        super().__init__(acquisition)
        self.feature_seed = feature_seed
        self.stream = bool(stream)
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if complex_mode not in {"cartesian", "magnitude"}:
            raise ValueError("complex_mode must be 'cartesian' or 'magnitude'")
        self.n_features = int(n_features)
        self.regularization = float(regularization)
        self.chunk_size = int(chunk_size)
        self.complex_mode = complex_mode
        self.normalize = bool(normalize)
        self.uncertainty = bool(uncertainty)
        self._requested_length_scale = length_scale
        self.register_buffer("frequency", torch.empty(0))
        # The same frequencies laid out for the host kernel's inner loop.
        self.register_buffer("frequency_t", torch.empty(0))
        self.register_buffer("phase", torch.empty(0))
        self.register_buffer("feature_mean", torch.empty(0))
        self.register_buffer("parameter_mean", torch.empty(0))
        self.register_buffer("weight", torch.empty(0))
        self.register_buffer("spread_mean", torch.empty(0))
        self.register_buffer("spread_weight", torch.empty(0))
        self.register_buffer("length_scale", torch.empty(0))
        # Copies of the fitted tensors, one per device a mapping has reached.
        self._replicas: dict[str, tuple[torch.Tensor, ...]] = {}

    @property
    def fitted(self) -> bool:
        """Whether training statistics have been fitted."""
        return self.weight.numel() != 0

    @torch.no_grad()
    def _fit_arrays(
        self,
        signals: torch.Tensor,
        parameters: torch.Tensor,
        known: torch.Tensor | None = None,
        *,
        noise_std: float | torch.Tensor = 0.0,
    ) -> PERK:
        """Fit the estimator from simulated signals and parameter targets.

        Parameters
        ----------
        signals : torch.Tensor
            Training signals shaped ``(samples, ..., contrasts)``. All leading
            sample dimensions are flattened.
        parameters : torch.Tensor
            Unknown target parameters shaped ``(samples, ..., n_parameters)``.
        known : torch.Tensor, optional
            Known parameters appended to the regression features.
        noise_std : float or torch.Tensor, optional
            Standard deviation of Gaussian training noise. Independent noise
            is added to real and imaginary channels for complex signals.

        Returns
        -------
        PERK
            The fitted estimator.
        """
        signals = torch.as_tensor(signals)
        parameters = _parameter_matrix(parameters, signals)
        sample_count = parameters.shape[0]
        if signals.numel() % sample_count:
            raise ValueError("signals and parameters have different sample counts")
        signals = signals.reshape(sample_count, -1)
        known = _known_matrix(known, signals, sample_count)
        random_seed = _random_seed(self.feature_seed)

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, sample_count, self.chunk_size):
                stop = min(start + self.chunk_size, sample_count)
                generator = _generator(signals.device, random_seed + start)
                signal_chunk = _add_noise(
                    signals[start:stop], noise_std, generator=generator
                )
                known_chunk = None if known is None else known[start:stop]
                yield (
                    _feature_matrix(
                        signal_chunk,
                        known_chunk,
                        stop - start,
                        complex_mode=self.complex_mode,
                        normalize=self.normalize,
                    ),
                    self._targets(parameters[start:stop], signal_chunk),
                )

        return self._fit_batches(batches, sample_count)

    @torch.no_grad()
    def _fit_simulated(
        self,
        samples: int,
        *,
        chunk: int,
        rank: int | None,
        noise_std: float | torch.Tensor = 0.0,
    ) -> None:
        """Accumulate the covariances chunk by chunk, holding no dictionary.

        Where the estimator was not asked to stream this is the shared path:
        the whole training set is simulated, a basis is fitted over it if one
        was asked for, and it is discarded once the covariances are in.
        """
        if not self.stream:
            super()._fit_simulated(samples, chunk=chunk, rank=rank, noise_std=noise_std)
            return
        if rank is not None:
            raise ValueError(
                "a basis is estimated from the whole dictionary, which a "
                "streaming fit never holds; leave stream out to fit one here, "
                "or pass subspace= to work in one fitted elsewhere"
            )
        chunk_size = chunk or self.chunk_size
        if chunk_size < 1:
            raise ValueError(f"chunk must be positive, got {chunk_size}")
        drawn, parameters, known_matrix = self._draw_parameters(samples)
        random_seed = _random_seed(self.feature_seed)

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, samples, chunk_size):
                stop = min(start + chunk_size, samples)
                known_chunk = None if known_matrix is None else known_matrix[start:stop]
                given = {name: value[start:stop] for name, value in drawn.items()}
                signals = torch.as_tensor(
                    self.acquisition.simulate(**given), device=parameters.device
                )
                if self.subspace is not None:
                    signals = self.subspace.project(signals)
                generator = _generator(parameters.device, random_seed + start)
                signals = _add_noise(signals, noise_std, generator=generator)
                yield (
                    _feature_matrix(
                        signals.reshape(stop - start, -1),
                        known_chunk,
                        stop - start,
                        complex_mode=self.complex_mode,
                        normalize=self.normalize,
                    ),
                    self._targets(parameters[start:stop], signals),
                )

        self._fit_batches(batches, samples)

    def _estimate_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate parameters from measured signals."""
        if not self.fitted:
            raise RuntimeError("PERK must be fitted before estimation")
        signals = torch.as_tensor(signals, device=self.frequency.device)
        sample_shape = signals.shape[:-1]
        sample_count = math.prod(sample_shape) if sample_shape else 1
        inputs = _feature_matrix(
            signals,
            known,
            sample_count,
            complex_mode=self.complex_mode,
            normalize=self.normalize,
        )
        placed = self._placed(inputs)
        if placed is not None:
            return placed[0].reshape(*sample_shape, -1)
        if self._fused(inputs):
            values = _FusedRegression.apply(
                inputs,
                self.frequency,
                self.frequency_t,
                self.phase,
                self.feature_mean,
                self.weight,
                self.parameter_mean,
            )
            return values.reshape(*sample_shape, -1)
        outputs = []
        for chunk in inputs.split(self.chunk_size):
            features = _rff(chunk, self.frequency, self.phase)
            outputs.append(
                self.parameter_mean + (features - self.feature_mean) @ self.weight.mT
            )
        return torch.cat(outputs, dim=0).reshape(*sample_shape, -1)

    def _uncertainty_arrays(
        self,
        signals: torch.Tensor,
        known: torch.Tensor | None,
        values: torch.Tensor,
        *,
        measured: torch.Tensor,
    ) -> torch.Tensor:
        """The spread of the answer, from the regression's own second moment.

        A kernel ridge regression predicts the mean of the target given the
        features. Run against the target's *square* it predicts the second
        moment, and the difference of the two is the variance of the target
        given the features -- which is the spread of this estimate, over the
        prior it was trained on and the noise it was trained at.

        So there is nothing to repeat: one more matrix multiply per voxel, at
        the same features the answer was read from.
        """
        if self.spread_weight.numel() == 0:
            raise NotImplementedError(
                "this PERK was fitted with uncertainty=False, so what it is "
                "wrong by was never learned; fit again with the default"
            )
        # For an error that is roughly Gaussian, the mean absolute value is
        # sqrt(2 / pi) of the standard deviation, which is what is reported.
        spread = self._departure(signals, known).clamp_min(0.0) * math.sqrt(
            0.5 * math.pi
        )
        return spread[..., : len(self._unknown)] if self._unknown else spread

    def _departure(
        self, signals: torch.Tensor, known: torch.Tensor | None
    ) -> torch.Tensor:
        """The predicted absolute error of the answer, voxel by voxel."""
        signals = torch.as_tensor(signals, device=self.frequency.device)
        shape = signals.shape[:-1]
        count = math.prod(shape) if shape else 1
        inputs = _feature_matrix(
            signals.reshape(count, signals.shape[-1]),
            known,
            count,
            complex_mode=self.complex_mode,
            normalize=self.normalize,
        )
        outputs = [
            self.spread_mean
            + (_rff(chunk, self.frequency, self.phase) - self.feature_mean)
            @ self.spread_weight.mT
            for chunk in inputs.split(self.chunk_size)
        ]
        return torch.cat(outputs, dim=0).reshape(*shape, -1)

    def _extra_maps(
        self, measured: torch.Tensor, values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The proton density, which the regression answered for along the way.

        The features are blind to amplitude, so what was learned is one over
        the length of the fingerprint the relaxation times imply. Multiplying
        by the length the measurement actually has is the density, and no
        forward simulation is involved.
        """
        if not self.normalize or values.shape[-1] <= len(self._unknown):
            return {}
        measured = torch.as_tensor(measured)
        lengths = torch.linalg.vector_norm(
            measured.reshape(int(values.shape[0]), -1), dim=-1
        )
        return {"M0": lengths.to(values) * values[..., len(self._unknown)]}

    def _placed(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        """Run under the execution policy, or ``None`` if none applies.

        Voxels are independent, so a volume too large for a card is streamed
        through it and a machine with two cards uses both. What the policy
        cannot carry is a gradient: a streamed chunk is copied through a
        reused pinned buffer, which is not something autograd can be walked
        back through. A call that wants a derivative gets the ordinary path.
        """
        if torch.is_grad_enabled() and inputs.requires_grad:
            return None
        contrasts = int(inputs.shape[-1])
        parameters = int(self.weight.shape[0])
        with torch.no_grad():
            return per_voxel(
                [inputs],
                bytes_per_voxel=(contrasts + parameters) * 4,
                work=int(inputs.shape[0]) * contrasts * self.n_features,
                crossover=lambda device: crossover(
                    (contrasts, self.n_features, parameters),
                    device,
                    self._probe(contrasts, parameters),
                    contrasts * self.n_features,
                ),
                body=lambda chunk, device: (
                    self._regress(chunk[0], self._fitted_on(device)),
                ),
            )

    def _probe(self, contrasts: int, parameters: int) -> Any:
        """A closure the calibrator can time, running the real regression."""

        def build(device: torch.device, voxels: int) -> Any:
            generator = torch.Generator(device=device).manual_seed(0)
            signals = torch.randn(voxels, contrasts, generator=generator, device=device)
            held = self._fitted_on(device)
            return lambda: self._regress(signals, held)

        return build

    def _fitted_on(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The fitted tensors on ``device``.

        They are the same for every chunk, so they cross once per device
        rather than once per chunk.
        """
        key = str(device)
        held = self._replicas.get(key)
        if held is None:
            held = tuple(
                tensor.to(device)
                for tensor in (
                    self.frequency,
                    self.frequency_t,
                    self.phase,
                    self.feature_mean,
                    self.weight,
                    self.parameter_mean,
                )
            )
            self._replicas[key] = held
        return held

    def _regress(
        self, inputs: torch.Tensor, held: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """One chunk, on whichever device its tensors are on."""
        frequency, transposed, phase, feature_mean, weight, parameter_mean = held
        backend = _kernels(inputs.device)
        if backend is not None:
            if inputs.device.type == "cuda":
                return backend.regress(
                    inputs,
                    frequency,
                    phase,
                    feature_mean,
                    weight,
                    parameter_mean,
                )
            return backend.regress(
                inputs,
                frequency,
                transposed,
                phase,
                feature_mean,
                weight,
                parameter_mean,
            )
        outputs = []
        for piece in inputs.split(self.chunk_size):
            features = _rff(piece, frequency, phase)
            outputs.append(parameter_mean + (features - feature_mean) @ weight.mT)
        return torch.cat(outputs, dim=0)

    def _fused(self, inputs: torch.Tensor) -> bool:
        """Whether the fused kernel can answer this call.

        It differentiates with respect to the signals, which is what a PERK
        inside a reconstruction network needs. A gradient wanted for one of the
        fitted tensors instead is rare enough to be worth the composed path
        rather than a second adjoint.
        """
        if _kernels(inputs.device) is None:
            return False
        return not any(
            tensor.requires_grad
            for tensor in (
                self.frequency,
                self.phase,
                self.feature_mean,
                self.weight,
                self.parameter_mean,
            )
        )

    def _targets(self, parameters: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
        """The regression's targets: the parameters, and the scale if it is lost.

        Normalized features carry no amplitude, so the density a measurement is
        of its own fingerprint is regressed alongside the relaxation times
        rather than recovered afterwards by simulating one. A fit from bare
        arrays answers in the caller's own columns and gets no extra one.
        """
        if not self.normalize or not self._unknown:
            return parameters
        return torch.cat(
            (parameters, _reciprocal_norm(signals, parameters.shape[0]).to(parameters)),
            dim=-1,
        )

    def _fit_spread(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
        *,
        covariance: torch.Tensor,
        feature_mean: torch.Tensor,
        parameter_mean: torch.Tensor,
        weight: torch.Tensor,
        frequency: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Regress the squared error of the fitted estimator on its features.

        What comes back predicts, for a measurement, the mean squared distance
        between the answer and the truth -- the whole of it, the part the noise
        moves and the part the prior pulls, since a regression trained on a
        prior is wrong in the same direction every time and repeating a scan
        would never show it.
        """
        residual_sum = torch.zeros_like(parameter_mean)
        residual_cross = torch.zeros_like(weight, dtype=torch.float64)
        for inputs, targets in batches():
            inputs = inputs.to(covariance.device, torch.float32)
            targets = targets.to(covariance.device, torch.float64)
            features = _rff(
                inputs, frequency.to(inputs.device), phase.to(inputs.device)
            )
            features = features.to(torch.float64)
            predicted = parameter_mean + (features - feature_mean) @ weight.mT
            # The absolute residual, not its square: a squared residual has a
            # tail heavy enough that a regression over it is mostly noise,
            # while |r| is well behaved and carries the same scale.
            residual = (targets - predicted).abs()
            residual_sum += residual.sum(dim=0)
            residual_cross += residual.mT @ features
        spread_mean = residual_sum / sample_count
        cross = (
            residual_cross - sample_count * torch.outer(spread_mean, feature_mean)
        ) / (sample_count - 1)
        return spread_mean, torch.linalg.solve(covariance, cross.mT).mT

    def _fit_batches(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
    ) -> PERK:
        """Fit from a source that can be walked more than once.

        The kernel width is read from the spread of the training inputs, and
        the random features cannot be drawn until it is known -- so estimating
        it costs one pass over the source before the fitting one. Give
        ``length_scale`` and there is only the fitting pass.
        """
        if sample_count < 2:
            raise ValueError("PERK requires at least two training samples")

        length_scale = None
        if self._requested_length_scale is None:
            length_scale = self._estimated_length_scale(batches, sample_count)

        home: torch.device | None = None
        where: torch.device | None = None
        frequency: torch.Tensor | None = None
        phase: torch.Tensor | None = None
        feature_sum = None
        second_moment = None
        cross_moment = None
        parameter_sum = None
        observed = 0
        for inputs, targets in batches():
            if where is None:
                home = inputs.device
                where = self._fitting_device(inputs, sample_count) or home
            inputs = inputs.to(where, torch.float32)
            targets = targets.to(where, torch.float32)
            if frequency is None:
                length_scale = (
                    self._given_length_scale(inputs.shape[-1], inputs.device)
                    if length_scale is None
                    else length_scale.to(inputs.device)
                )
                frequency, phase = self._random_features(length_scale)
                feature_sum = torch.zeros(
                    self.n_features, dtype=torch.float64, device=inputs.device
                )
                second_moment = torch.zeros(
                    self.n_features,
                    self.n_features,
                    dtype=torch.float64,
                    device=inputs.device,
                )
                cross_moment = torch.zeros(
                    targets.shape[-1],
                    self.n_features,
                    dtype=torch.float64,
                    device=inputs.device,
                )
                parameter_sum = torch.zeros(
                    targets.shape[-1], dtype=torch.float64, device=inputs.device
                )
            features = _rff(inputs, frequency, phase).to(torch.float64)
            targets = targets.to(torch.float64)
            feature_sum += features.sum(dim=0)
            parameter_sum += targets.sum(dim=0)
            second_moment += features.mT @ features
            cross_moment += targets.mT @ features
            observed += inputs.shape[0]
        if observed != sample_count or frequency is None:
            raise ValueError("batch source returned an inconsistent sample count")

        # Centred covariances from uncentred moments, so one pass carries both
        # the means and the products they would otherwise have to precede.
        feature_mean = feature_sum / sample_count
        parameter_mean = parameter_sum / sample_count
        covariance = (
            second_moment - sample_count * torch.outer(feature_mean, feature_mean)
        ) / (sample_count - 1)
        cross_covariance = (
            cross_moment - sample_count * torch.outer(parameter_mean, feature_mean)
        ) / (sample_count - 1)
        covariance.diagonal().add_(self.regularization)
        weight = torch.linalg.solve(covariance, cross_covariance.mT).mT

        # What the answer is wrong by, regressed the same way. The residual is
        # known only once the first solve is done, so this is a second walk of
        # the same source -- and it is the residual rather than the target's
        # second moment because a squared residual cannot be negative, while
        # the difference of two independently regularized regressions can be,
        # and is exactly where a variance is wanted.
        spread_mean, spread_weight = (
            self._fit_spread(
                batches,
                sample_count,
                covariance=covariance,
                feature_mean=feature_mean,
                parameter_mean=parameter_mean,
                weight=weight,
                frequency=frequency,
                phase=phase,
            )
            if self.uncertainty
            else (torch.zeros(0), torch.zeros(0))
        )

        # Where the fit ran was a speed decision; where the estimator lives
        # is the caller's, so it comes back beside the data it was given.
        self._replicas = {}
        self.frequency = frequency.to(home)
        self.frequency_t = self.frequency.mT.contiguous()
        self.phase = phase.to(home)
        self.feature_mean = feature_mean.to(torch.float32).to(home)
        self.parameter_mean = parameter_mean.to(torch.float32).to(home)
        self.spread_mean = spread_mean.to(torch.float32).to(home)
        self.weight = weight.to(torch.float32).to(home)
        self.spread_weight = spread_weight.to(torch.float32).to(home)
        self.length_scale = length_scale.to(home)
        return self

    def _fitting_device(
        self, inputs: torch.Tensor, sample_count: int
    ) -> torch.device | None:
        """Where to accumulate this fit, under the policy in force.

        Fitting is a reduction: the covariance of a thousand features is eight
        megabytes however many samples it was built from, so it stays in one
        place and the training set is fed to it a chunk at a time. What the
        policy decides is which place.
        """
        contrasts = int(inputs.shape[-1])
        return one_device(
            work=sample_count * contrasts * self.n_features,
            voxels=sample_count,
            bytes_per_voxel=(contrasts + self.n_features) * 4,
            crossover=lambda device: crossover(
                (contrasts, self.n_features, "fit"),
                device,
                self._fit_probe(contrasts),
                contrasts * self.n_features,
            ),
        )

    def _fit_probe(self, contrasts: int) -> Any:
        """A closure the calibrator can time, accumulating one covariance."""

        def build(device: torch.device, voxels: int) -> Any:
            generator = torch.Generator(device=device).manual_seed(0)
            inputs = torch.randn(voxels, contrasts, generator=generator, device=device)
            frequency = torch.randn(
                self.n_features, contrasts, generator=generator, device=device
            )
            phase = torch.zeros(self.n_features, device=device)

            def once() -> torch.Tensor:
                features = _rff(inputs, frequency, phase).to(torch.float64)
                return features.mT @ features

            return once

        return build

    def _random_features(
        self, length_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fixed frequency and phase of the random Fourier feature map.

        Drawn on the host whatever device the fit runs on. Torch's generators
        do not agree between devices at the same seed, and where a fit ran is
        supposed to be a decision about speed rather than about which
        estimator comes out of it.
        """
        generator = _generator(torch.device("cpu"), _random_seed(self.feature_seed))
        frequency = (
            torch.randn(
                self.n_features,
                length_scale.numel(),
                dtype=torch.float32,
                generator=generator,
            ).to(length_scale.device)
            / length_scale[None, :]
        )
        phase = (
            2.0
            * math.pi
            * torch.rand(self.n_features, dtype=torch.float32, generator=generator).to(
                length_scale.device
            )
        )
        return frequency, phase

    def _estimated_length_scale(
        self,
        batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
        sample_count: int,
    ) -> torch.Tensor:
        """Read the kernel width off the spread of the training inputs."""
        input_sum = None
        input_square_sum = None
        observed = 0
        for inputs, _targets in batches():
            inputs = inputs.to(torch.float32)
            if input_sum is None:
                input_sum = torch.zeros(
                    inputs.shape[-1], dtype=torch.float64, device=inputs.device
                )
                input_square_sum = torch.zeros_like(input_sum)
            input_sum += inputs.sum(dim=0, dtype=torch.float64)
            input_square_sum += inputs.square().sum(dim=0, dtype=torch.float64)
            observed += inputs.shape[0]
        if observed != sample_count or input_sum is None:
            raise ValueError("batch source returned an inconsistent sample count")
        variance = (input_square_sum - input_sum.square() / sample_count) / (
            sample_count - 1
        )
        floor = torch.finfo(torch.float32).eps
        scale = variance.clamp_min(0.0).sqrt().to(torch.float32)
        # Per-coordinate standard deviations make a high-dimensional Gaussian
        # kernel vanish between almost every pair of samples. This factor
        # implements the usual O(sqrt(d)) median-distance scaling while
        # retaining feature-wise physical units.
        scale *= math.sqrt(input_sum.numel())
        return scale.clamp_min(floor)

    def _given_length_scale(self, width: int, device: torch.device) -> torch.Tensor:
        """The kernel width the caller asked for, checked against the inputs."""
        scale = torch.as_tensor(
            self._requested_length_scale, dtype=torch.float32, device=device
        ).flatten()
        if scale.numel() == 1:
            scale = scale.expand(width)
        if scale.numel() != width or torch.any(scale <= 0):
            raise ValueError("length_scale must be positive and match input features")
        return scale


# %% private module subroutines


def _loaded(name: str) -> Any:
    """One kernel backend, or ``None`` where it cannot be imported."""
    try:
        return __import__(f"torchsim.estimators.{name}", fromlist=[name])
    except ImportError:
        return None


_TRITON = _loaded("_perk_triton")
_NATIVE = _loaded("_perk_native")


def _kernels(device: torch.device) -> Any:
    """The fused backend for this device, or ``None``."""
    return _TRITON if device.type == "cuda" else _NATIVE


class _FusedRegression(torch.autograd.Function):
    """The feature map and its regression, without the features in between."""

    @staticmethod
    def forward(
        ctx: Any,
        signals: torch.Tensor,
        frequency: torch.Tensor,
        transposed: torch.Tensor,
        phase: torch.Tensor,
        feature_mean: torch.Tensor,
        weight: torch.Tensor,
        parameter_mean: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(signals, frequency, transposed, phase, weight)
        if signals.device.type == "cuda":
            return _TRITON.regress(
                signals, frequency, phase, feature_mean, weight, parameter_mean
            )
        return _NATIVE.regress(
            signals,
            frequency,
            transposed,
            phase,
            feature_mean,
            weight,
            parameter_mean,
        )

    @staticmethod
    def backward(ctx: Any, cotangent: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        signals, frequency, transposed, phase, weight = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None,) * 7
        gradient = (
            _TRITON.regress_vjp(cotangent, signals, frequency, phase, weight)
            if signals.device.type == "cuda"
            else _NATIVE.regress_vjp(
                cotangent, signals, frequency, transposed, phase, weight
            )
        )
        return gradient, None, None, None, None, None, None


def _feature_matrix(
    signals: torch.Tensor,
    known: torch.Tensor | None,
    sample_count: int,
    *,
    complex_mode: str,
    normalize: bool,
) -> torch.Tensor:
    signals = signals.reshape(sample_count, -1)
    if normalize:
        norm = torch.linalg.vector_norm(signals, dim=-1).clamp_min(
            torch.finfo(signals.real.dtype).eps
        )
        signals = signals / norm[:, None]
    if torch.is_complex(signals) and complex_mode == "magnitude":
        signals = signals.abs()
    elif torch.is_complex(signals):
        signals = torch.view_as_real(signals).reshape(sample_count, -1)
    signals = signals.to(torch.float32)
    if known is None:
        return signals
    known = torch.as_tensor(known, device=signals.device).reshape(sample_count, -1)
    if torch.is_complex(known):
        known = torch.view_as_real(known).reshape(sample_count, -1)
    return torch.cat((signals, known.to(torch.float32)), dim=-1)


def _reciprocal_norm(signals: torch.Tensor, sample_count: int) -> torch.Tensor:
    """One over the length of each training signal, as a target column.

    Normalizing the features throws the scale away, which is what makes the
    regression blind to the proton density. It is not lost, though: the length
    of a fingerprint simulated at unit density is a function of the relaxation
    times, so the regression can learn its reciprocal alongside them and the
    measurement's own length is the scale it was missing.
    """
    lengths = torch.linalg.vector_norm(signals.reshape(sample_count, -1), dim=-1)
    return lengths.clamp_min(torch.finfo(torch.float32).eps).reciprocal()[:, None]


def _rff(
    inputs: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    scale = math.sqrt(2.0 / frequency.shape[0])
    return scale * torch.cos(inputs @ frequency.mT + phase)


def _sample_columns(
    values: Mapping[str, Any], expected: int | None = None
) -> list[torch.Tensor]:
    """One column per named property, all agreeing on the sample count."""
    columns = [
        torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        for value in values.values()
    ]
    counts = {int(column.numel()) for column in columns}
    if expected is not None:
        counts.add(expected)
    if len(counts) > 1:
        raise ValueError(
            f"the properties disagree on the training sample count: {sorted(counts)}"
        )
    return columns


def _parameter_matrix(parameters: Any, reference: torch.Tensor) -> torch.Tensor:
    parameters = torch.as_tensor(
        parameters, device=reference.device, dtype=reference.real.dtype
    )
    if parameters.ndim == 1:
        parameters = parameters[:, None]
    return parameters.reshape(-1, parameters.shape[-1]).to(torch.float32)


def _known_matrix(
    known: torch.Tensor | None,
    reference: torch.Tensor,
    sample_count: int,
) -> torch.Tensor | None:
    if known is None:
        return None
    output = torch.as_tensor(known, device=reference.device)
    if output.numel() % sample_count:
        raise ValueError("known parameters have a different sample count")
    return output.reshape(sample_count, -1)


def _random_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(torch.empty((), dtype=torch.int64).random_().item())


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (2**63 - 1))
    return generator


def _add_noise(
    signals: torch.Tensor,
    noise_std: Any,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    scale = torch.as_tensor(noise_std, dtype=signals.real.dtype, device=signals.device)
    if not torch.any(scale != 0):
        return signals
    shape = signals.shape
    if torch.is_complex(signals):
        noise = torch.complex(
            torch.randn(
                shape,
                dtype=signals.real.dtype,
                device=signals.device,
                generator=generator,
            ),
            torch.randn(
                shape,
                dtype=signals.real.dtype,
                device=signals.device,
                generator=generator,
            ),
        )
    else:
        noise = torch.randn(
            shape,
            dtype=signals.dtype,
            device=signals.device,
            generator=generator,
        )
    return signals + scale * noise
