"""The signal model as an operator a reconstruction can compose with.

A physics-based reconstruction writes the forward operator as a chain

    F = P . F . C . M

-- sampling, Fourier encoding, coil sensitivities, and the signal model --
and solves for the parameter maps ``M`` is evaluated at. Everything to the
left of ``M`` is encoding, is the same for every quantitative sequence ever
written, and is not ours: mri-nufft supplies it, with a deepinv bridge, and
this module composes with it rather than reimplementing it.

``M`` is the part that changes with the sequence, and it is the part a user
should have to write. :class:`ModelOperator` turns any
simulator into it: parameter maps in, one image per
contrast out, with derivatives, bounds and the complex amplitude a
reconstruction needs, and nothing about Fourier encoding anywhere in it.

**The derivatives never build a Jacobian.** The model is voxel-diagonal, so
its Jacobian is a stack of small blocks, but a Gauss-Newton step never needs
the blocks -- it needs products with them. One forward-mode pass gives
``J d`` for the whole volume whatever the parameter count, and one reverse
pass gives ``J^H v``. :meth:`ModelOperator.jacobian` materializes the blocks
for the one caller that wants them, the voxel-wise fit, where they are solved
outright.

**The amplitude is complex and the model is not.** Proton density and receive
phase multiply whatever the sequence does, so they are carried here as one
complex map rather than written into every model. Its two Jacobian columns are
the model output itself, times one and times i, so it costs no pass at all --
and a model recast on its free parameters alone, which is how an equality
constraint is imposed, needs exactly this scale and now gets it for nothing.
"""

from __future__ import annotations

__all__ = ["ModelOperator"]

from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from typing import Any

import torch

from .._bounds import bound_of, to_free, to_natural, widen
from .._calibrate import crossover
from .._execution import per_voxel

#: What the amplitude occupies, when it is carried: real part then imaginary.
_AMPLITUDE = ("amplitude.real", "amplitude.imag")


class ModelOperator(torch.nn.Module):
    """A signal model over parameter maps, with its derivatives.

    Maps are stacked on the **last** axis, as everywhere else in TorchSim:
    ``(..., channels)`` in, ``(..., contrasts)`` out, every leading axis a
    voxel axis. :meth:`physics` is where that flips to the channel-first
    convention deepinv and mri-nufft use, and it is the only place it flips.

    Parameters
    ----------
    acquisition : Simulator
        The sequence being inverted, with every property that is not being
        solved for already fixed on it. A property bound as a map -- a
        measured B1, a known T1 -- is one value per voxel and rides along.
    unknown : str, optional
        The property names being solved for, in the order their channels
        appear. At least one.
    bounds : mapping, optional
        ``{name: (low, high)}``, either end ``None`` for unbounded. A bound is
        kept by solving for a transformed variable, so no iterate is ever
        outside it -- which matters more here than in a fit, because the model
        is evaluated at every voxel to predict every k-space sample and one
        voxel out of range corrupts the whole residual. A two-sided bound also
        puts the parameter on a scale of order one whatever its units, which
        is the preconditioning a mixed parameter set needs.
    scale : mapping, optional
        ``{name: value}``, the size of a step in a parameter left unbounded.
        Bounded parameters are already scaled by their interval.
    amplitude : bool, optional
        Whether to carry a complex amplitude multiplying the model output.
        On, which is what a reconstruction wants; off for a fit whose model
        already exposes its own proton density.
    subspace : Subspace, optional
        A :class:`~torchsim.Subspace` the prediction is projected through, for
        a reconstruction that solves in the temporal basis rather than in the
        contrasts.

    Examples
    --------
    .. code-block:: python

        operator = ModelOperator(
            MultiEchoSimulator(TE=echo_times, T1=1000.0),
            "T2",
            bounds={"T2": (10.0, 300.0)},
        )
        maps = operator.initial((128, 128), T2=80.0)
        images = operator.A(maps)          # (128, 128, echoes), complex

    Notes
    -----
    The operator holds nothing on a device of its own. It follows the maps it
    is called with, taking the sequence and any bound tissue along, so there
    is nothing here to move and ``to()`` moves nothing.

    **Equality constraints are written into the model, not declared here.**
    A constraint that fixes one parameter in terms of the others removes a
    degree of freedom, so the way to impose it is to not have that freedom:
    write the model on the parameters that remain. For a fat-water separation
    whose two fractions must sum to one, make the fat fraction ``f`` the only
    unknown and write water as ``1 - f`` inside the model. The constraint then
    holds identically at every Gauss-Newton iterate, to the bit, rather than
    being restored after each one.

    Raises
    ------
    ValueError
        If nothing is unknown, if a bound or a scale names something that is
        not unknown, or if a bound is not increasing.
    """

    def __init__(
        self,
        acquisition: Any,
        *unknown: str,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        scale: Mapping[str, float] | None = None,
        amplitude: bool = True,
        subspace: Any = None,
    ) -> None:
        super().__init__()
        if not unknown:
            raise ValueError("name at least one property to solve for")
        self.acquisition = acquisition
        self.unknown = tuple(unknown)
        self.bounds = dict(bounds or {})
        self.amplitude = bool(amplitude)
        self.subspace = subspace
        for label, given in (("bounds", self.bounds), ("scale", scale or {})):
            stray = set(given) - set(self.unknown)
            if stray:
                raise ValueError(
                    f"{label} names {sorted(stray)}, which "
                    f"{'is' if len(stray) == 1 else 'are'} not being solved for"
                )
        for name in self.unknown:
            bound_of(self.bounds, name)
        self.scale = tuple(float((scale or {}).get(name, 1.0)) for name in self.unknown)
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("every scale must be positive")
        self._elsewhere: dict[str, Any] = {}

    # -- what the maps are --------------------------------------------------

    @property
    def channels(self) -> int:
        """How many map channels ``x`` carries."""
        return len(self.unknown) + 2 * self.amplitude

    @property
    def names(self) -> tuple[str, ...]:
        """What each channel of ``x`` is, in order."""
        return self.unknown + (_AMPLITUDE if self.amplitude else ())

    def split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The maps ``x`` stands for, in their own units.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``, the variables being solved for.

        Returns
        -------
        dict
            One entry per unknown, inside its bound, plus ``"amplitude"``
            complex where one is carried.
        """
        given, rho = self._named(x)
        return given if rho is None else {**given, "amplitude": rho}

    def initial(self, shape: Sequence[int] = (), **values: Any) -> torch.Tensor:
        """Maps to start from, as the variables actually solved for.

        Parameters
        ----------
        shape : sequence of int, optional
            The voxel shape. Empty gives one voxel.
        values : float or array-like, optional
            ``{name: value}`` in the property's own units. An unknown left out
            starts at the middle of its bound, and a one-sided or absent bound
            has no middle, so it must be given. The amplitude starts at one
            unless ``amplitude=`` says otherwise.

        Returns
        -------
        torch.Tensor
            ``(*shape, channels)``.

        Raises
        ------
        ValueError
            If a property with no two-sided bound is left out, or if a value
            sits on a bound -- a value on a bound has no unconstrained image.
        """
        stray = set(values) - set(self.unknown) - {"amplitude"}
        if stray:
            raise ValueError(f"{sorted(stray)} is not a channel of this operator")
        columns = []
        for name in self.unknown:
            low, high = bound_of(self.bounds, name)
            if name in values:
                start = float(values[name])
            elif low is not None and high is not None:
                start = 0.5 * (low + high)
            else:
                raise ValueError(
                    f"{name} has no two-sided bound, so give it a starting value"
                )
            if (low is not None and start <= low) or (
                high is not None and start >= high
            ):
                raise ValueError(
                    f"{name}: {start} is not strictly inside its bound ({low}, {high})"
                )
            columns.append(torch.full(tuple(shape), start))
        natural = torch.stack(columns, dim=-1)
        scale = torch.tensor(self.scale)
        free = to_free(natural, self.bounds, self.unknown) / scale
        if not self.amplitude:
            return free
        start = torch.as_tensor(values.get("amplitude", 1.0 + 0.0j))
        start = torch.broadcast_to(start, tuple(shape)).to(torch.complex64)
        return torch.cat((free, start.real[..., None], start.imag[..., None]), dim=-1)

    # -- the operator -------------------------------------------------------

    def select(self, keep: torch.Tensor) -> ModelOperator:
        """This operator over a subset of the voxels.

        A property bound as a map -- a measured B1, a known efficiency -- has
        one value per voxel, so a loop that retires the voxels it has finished
        has to take the maps with it. Anything that is not one value per voxel
        is left alone.

        Parameters
        ----------
        keep : torch.Tensor
            An index or a boolean mask over the voxel axis.

        Returns
        -------
        ModelOperator
            A copy over those voxels.
        """
        declared = set(self.acquisition.exposes)
        narrowed = {
            name: value[keep]
            for name, value in self.acquisition.bound.items()
            if name in declared and torch.is_tensor(value) and value.ndim
        }
        if not narrowed:
            return self
        chosen = shallow_copy(self)
        chosen.acquisition = self.acquisition.bind(**narrowed)
        chosen._elsewhere = {}
        return chosen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The contrasts these maps record.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``.

        Returns
        -------
        torch.Tensor
            ``(..., contrasts)``, complex.
        """
        return self._voxelwise("A", (x,), lambda chunk: (self._predict(chunk[0]),))

    #: The name a reconstruction knows the forward operator by.
    A = forward

    def A_jvp(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """The directional derivative ``J d``, in one forward-mode pass.

        One pass whatever the channel count -- the step itself is the tangent,
        so nothing is built column by column.

        Parameters
        ----------
        x, d : torch.Tensor
            ``(..., channels)``, the point and the direction.

        Returns
        -------
        torch.Tensor
            ``(..., contrasts)``, complex.
        """
        return self._voxelwise(
            "jvp",
            (x, d),
            lambda chunk: (torch.func.jvp(self._predict, (chunk[0],), (chunk[1],))[1],),
        )

    def A_vjp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The adjoint product ``J^H v``, in one reverse-mode pass.

        Real, because the maps are: what comes back is the real part of the
        conjugate product, which is the gradient direction of the squared
        modulus and the adjoint of :meth:`A_jvp` under the real inner
        product.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``, the point.
        v : torch.Tensor
            ``(..., contrasts)``, the cotangent.

        Returns
        -------
        torch.Tensor
            ``(..., channels)``, real.
        """
        return self._voxelwise(
            "vjp",
            (x, v),
            lambda chunk: (torch.func.vjp(self._predict, chunk[0])[1](chunk[1])[0],),
        )

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """The derivative block of every voxel, built.

        Wanted where the blocks are solved outright, which is the voxel-wise
        fit; a Gauss-Newton step under an encoding operator wants
        :meth:`A_jvp` and :meth:`A_vjp` instead and never sees a block.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., channels)``.

        Returns
        -------
        torch.Tensor
            ``(..., channels, contrasts)``, complex.
        """
        return self._voxelwise(
            "jacobian", (x,), lambda chunk: (self._blocks(chunk[0]),)
        )

    def physics(self, **kwargs: Any) -> Any:
        """This operator as a :class:`deepinv.physics.Physics`.

        The wrapper is where the map axis moves to the front, because that is
        the convention deepinv and mri-nufft share: ``(batch, channels, *xyz)``
        in, ``(batch, contrasts, *xyz)`` out. Composing it with an encoding
        operator -- ``encoding * operator.physics()`` -- gives a
        ``ComposedPhysics`` every deepinv optimizer and prior applies to.

        Parameters
        ----------
        kwargs : dict, optional
            Passed to :class:`deepinv.physics.Physics`, so a noise model can
            be given.

        Returns
        -------
        deepinv.physics.Physics

        Raises
        ------
        ImportError
            If deepinv is not installed. TorchSim does not depend on it.

        Notes
        -----
        A ``ComposedPhysics`` takes its Jacobian products by automatic
        differentiation through the whole chain, so composing this way gives
        up the analytic derivative below.
        :class:`~torchsim.recon.GaussNewton` chains the two operators' own
        products instead, and keeps it.
        """
        try:
            from deepinv.physics import Physics
        except ImportError as error:  # pragma: no cover - depends on the env
            raise ImportError(
                "physics() needs deepinv; TorchSim does not depend on it. "
                "pip install deepinv"
            ) from error

        operator = self

        class _ModelPhysics(Physics):
            def A(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
                return _to_channels(operator.A(_to_trailing(x)))

            def A_vjp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return _to_channels(operator.A_vjp(_to_trailing(x), _to_trailing(v)))

        return _ModelPhysics(**kwargs)

    # -- what a caller does not write ---------------------------------------

    def _parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The free variables and the scale that turns them into properties."""
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"this operator has {self.channels} channels, got {x.shape[-1]}"
            )
        free = x[..., : len(self.unknown)]
        scale = torch.tensor(self.scale, dtype=free.dtype, device=free.device)
        return free, scale

    def _on(self, device: torch.device) -> Any:
        """The acquisition, wherever the maps being evaluated are.

        A chunk that has crossed to a card has left the sequence behind: the
        echo times and flip train live on the simulator, not on the maps, and
        an exponential of one against the other has to find both in the same
        place. Each device's copy is made once and kept.
        """
        key = str(device)
        if key not in self._elsewhere:
            self._elsewhere[key] = self.acquisition.to(device)
        return self._elsewhere[key]

    def _named(self, x: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor | None]:
        """The properties to simulate at, and the amplitude to scale by."""
        free, scale = self._parts(x)
        natural = to_natural(free * scale, self.bounds, self.unknown)
        given = {name: natural[..., index] for index, name in enumerate(self.unknown)}
        rho = torch.complex(x[..., -2], x[..., -1]) if self.amplitude else None
        return given, rho

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """``(voxels, channels)`` to ``(voxels, contrasts)``."""
        given, rho = self._named(x)
        signal = torch.as_tensor(self._on(x.device).simulate(**given))
        if self.subspace is not None:
            signal = self.subspace.project(signal)
        if rho is None:
            return signal
        return (
            signal.to(torch.promote_types(signal.dtype, rho.dtype)) * (rho[..., None])
        )

    def _blocks(self, x: torch.Tensor) -> torch.Tensor:
        """``(voxels, channels, contrasts)``, built column by column.

        The model's own columns cost one forward-mode pass each. The
        amplitude's two cost nothing: the model output *is* its derivative,
        times one and times i.
        """
        given, rho = self._named(x)
        free, scale = self._parts(x)
        signal, derivative = self._on(x.device).jacobian(self.unknown, **given)
        signal = torch.as_tensor(signal)
        derivative = torch.as_tensor(derivative)
        if self.subspace is not None:
            signal = self.subspace.project(signal)
            derivative = self.subspace.project(derivative)
        chain = widen(self.bounds, self.unknown, free * scale) * scale
        columns = derivative * chain[..., None]
        if rho is None:
            return columns
        dtype = torch.promote_types(signal.dtype, rho.dtype)
        columns = columns.to(dtype) * rho[..., None, None]
        signal = signal.to(dtype)
        return torch.cat(
            (columns, signal[..., None, :], (1j * signal)[..., None, :]), dim=-2
        )

    def _voxelwise(
        self, kind: str, inputs: Sequence[torch.Tensor], body: Any
    ) -> torch.Tensor:
        """Run a per-voxel body wherever the execution policy says to.

        Every axis but the last is a voxel axis, so the work is flattened to
        one, handed to :func:`~torchsim._execution.per_voxel`, and folded
        back. Where no policy is in force it runs as written.

        So does a call whose inputs carry a derivative, and that covers two
        cases at once: a streamed chunk is copied through a reused pinned
        buffer, which autograd cannot be walked back through, and a tensor
        wrapped by an outer ``torch.func`` transform -- which is how a caller
        differentiates through a composed operator -- reports the same way.
        """
        shape = inputs[0].shape[:-1]
        flat = [value.reshape(-1, value.shape[-1]) for value in inputs]
        answer = None
        differentiating = torch.is_grad_enabled() and any(
            value.requires_grad for value in flat
        )
        if not differentiating:
            contrasts = self._contrasts(flat[0])
            with torch.no_grad():
                answer = per_voxel(
                    flat,
                    bytes_per_voxel=(self.channels + 2 * contrasts) * 4,
                    work=int(flat[0].shape[0]) * contrasts,
                    crossover=lambda device: crossover(
                        (kind, contrasts, self.channels),
                        device,
                        self._probe(kind, body),
                        contrasts,
                    ),
                    body=lambda chunk, device: body(
                        [value.to(device) for value in chunk]
                    ),
                )
        if answer is None:
            answer = body(flat)
        return answer[0].reshape(*shape, *answer[0].shape[1:])

    def _contrasts(self, flat: torch.Tensor) -> int:
        """How many samples one voxel records, measured once and kept."""
        cached = getattr(self, "_contrast_count", None)
        if cached is None:
            with torch.no_grad():
                cached = int(self._predict(flat[:1]).shape[-1])
            self._contrast_count = cached
        return cached

    def _probe(self, kind: str, body: Any) -> Any:
        """A closure the calibrator can time, running the real work."""

        def build(device: torch.device, voxels: int) -> Any:
            # Zero is inside every bound this operator can carry, and the cost
            # of a pass does not depend on where in the interval it is taken.
            here = torch.zeros(voxels, self.channels, device=device)
            arguments = [here]
            if kind == "jvp":
                arguments.append(torch.ones_like(here))
            elif kind == "vjp":
                arguments.append(
                    torch.ones(
                        voxels,
                        self._contrasts(here),
                        device=device,
                        dtype=torch.complex64,
                    )
                )
            return lambda: body(arguments)

        return build


# %% private module subroutines


def _to_trailing(x: torch.Tensor) -> torch.Tensor:
    """Channel-first, as deepinv passes it, to channel-last as we read it."""
    return x.movedim(1, -1)


def _to_channels(x: torch.Tensor) -> torch.Tensor:
    """The way back."""
    return x.movedim(-1, 1)
