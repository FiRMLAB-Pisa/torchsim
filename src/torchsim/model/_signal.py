"""What a signal model owes, and what it is spared.

A model says three things: which properties it exposes, how it turns them into
a signal, and nothing else. Which kernel runs, how the work is cut across
memory and devices, and how derivatives are taken belong below.

Three rules earn their place here.

**A property stays as the caller gave it.** A model reaching the fused state
machine gets the terms it needs and no more, and the kernels decide that from
what was passed rather than by reducing over a buffer -- so a default widened
to the voxel count before the simulation sees it reads as a live map, and the
run pays for off-resonance, diffusion or flow it does not have. Nothing is
broadcast here except a property being differentiated, which has to be wide
enough for one directional derivative to cover every voxel.

**The caller's array library is the caller's.** Properties and sequence
arguments arrive in whatever they were written in -- NumPy, CuPy, torch -- and
are read through DLPack, over the same memory rather than a copy. The signal
goes back the same way. Everything between is torch, which is what the kernels
and the autograd graph are written against; see :mod:`torchsim.sequence._array`
for what that costs, and for the one thing it cannot carry back.

**Derivatives are taken in the mode that suits what is differentiated.** A
Bloch simulation records far more samples than it takes parameters, so a
derivative with respect to tissue is cheapest forward: :meth:`jacobian` issues
one directional derivative per property and gets every voxel from each. A
derivative of a scalar cost with respect to a sequence -- the acquisition
optimization problem -- is the other way round, and is plain autograd on the
returned signal. That path is deliberately not wrapped: the engine reads which
inputs carry a gradient and picks its kernel from that, and a layer here would
only hide it.
"""

from __future__ import annotations

__all__ = ["_SignalModel"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from types import MappingProxyType
from typing import Any

import torch

from ..sequence._array import as_torch, brought, is_array, like
from ..sequence._parameters import PUBLIC_PROPERTIES

_NOTHING: Mapping[str, Any] = MappingProxyType({})


class _SignalModel(ABC):
    """What every model owes, and the two derivative modes over it.

    The interface :class:`~torchsim.model.Simulator` presents, and everything
    downstream consumes: declared properties, and a signal. What a model has
    to supply is :attr:`properties` and :meth:`evaluate`.

    **The constructor takes the keywords** :meth:`simulate` **takes, and fixes
    them; a call overrides.** :meth:`bind` fixes more on a copy, so what a
    caller is left to give per call is whatever is actually varying.

    Attributes
    ----------
    properties : mapping or sequence of str
        What the model exposes, as names. A mapping is read for its keys, so a
        subclass that also needs to say where each property goes -- as
        :class:`~torchsim.model.SpinPhysics` does, naming the tissue
        field each fills -- declares both at once.
    bound : mapping
        The arguments fixed on this model, under the names :meth:`simulate`
        takes them by. A call may name any of them again, and the call wins.
    """

    properties: Mapping[str, str | None] | Sequence[str] = ()
    bound: Mapping[str, Any] = _NOTHING

    def __init__(self, **values: Any) -> None:
        """Fix the arguments this model is asked about.

        Parameters
        ----------
        values : float or array-like, optional
            Properties or sequence arguments, under the names
            :meth:`simulate` takes them by.
        """
        self.bound = self._fix(values)

    @abstractmethod
    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the signal these properties and this sequence record.

        ``properties`` holds the declared values the caller passed, under the
        names :attr:`properties` gives them, and holds nothing for those the
        caller left out. Public units are the caller's -- milliseconds and
        degrees -- and converting them belongs here, with the sequence.
        """

    # -- what is fixed on the model -----------------------------------------

    @property
    def exposes(self) -> tuple[str, ...]:
        """The property names this model declares, in its own order."""
        return tuple(self.properties)

    @property
    def accepts(self) -> tuple[str, ...]:
        """Every property name a call may use: what is declared, and the rest.

        A model declares the vocabulary its protocol is written in, and a voxel
        has more fields than any one model names. Both are accepted, so asking
        for off-resonance or a second pool is giving one a value rather than
        rebuilding the model around it.
        """
        return tuple(dict.fromkeys((*self.exposes, *PUBLIC_PROPERTIES)))

    def bind(self, **values: Any) -> Any:
        """This model with more arguments fixed on it, or some of them changed.

        What a fit does with a property measured separately: the same sequence,
        with one more map held on it.

        Parameters
        ----------
        values : float or array-like, optional
            Properties or sequence arguments, under the names
            :meth:`simulate` takes them by.

        Returns
        -------
        Simulator
            A copy. This one is left as it was.
        """
        held = shallow_copy(self)
        held.bound = {**self.bound, **self._fix(values)}
        return held

    def to(self, device: torch.device | str) -> Any:
        """This model, with everything fixed on it on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Where to put it.

        Returns
        -------
        Simulator
            A copy. This one is left where it was.
        """
        moved = shallow_copy(self)
        moved.bound = _moved(self.bound, torch.device(device))
        return moved

    # -- the two derivative modes -------------------------------------------

    def simulate(self, **values: Any) -> torch.Tensor:
        """Return the recorded signal.

        Property and sequence arguments are given together and told apart by
        :attr:`properties`, and join whatever :attr:`bound` already holds. A
        cost built on this and differentiated with
        :meth:`torch.Tensor.backward` reaches the engine's adjoint, which is
        the mode that suits a sequence parameter.
        """
        backend = self._backend(values)
        held, sequence = self._split({**self.bound, **values})
        signal = self._shaped(self.evaluate(held, **sequence), self._batch(held))
        return like(signal, backend)

    def jacobian(
        self, diff: str | Sequence[str], **values: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the signal and its derivative with respect to ``diff``.

        Forward mode, one directional derivative per differentiated property.
        Voxels are independent, so each pass yields every voxel's derivative at
        once and the cost is one pass per property rather than per voxel.

        Parameters
        ----------
        diff : str or sequence of str
            One property name, or several. A single name collapses the
            parameter axis; a sequence keeps it, stacked before the samples.
        values : float or array-like, optional
            The property and sequence arguments, as for :meth:`simulate`.

        Returns
        -------
        tuple
            The signal, and its Jacobian.

        Raises
        ------
        ValueError
            If ``diff`` names something the model does not expose, or
            something that carries no derivative.
        """
        names = (diff,) if isinstance(diff, str) else tuple(diff)
        backend = self._backend(values)
        held, sequence = self._split({**self.bound, **values})
        for name in names:
            if name not in held:
                raise ValueError(
                    f"{name!r} is not among the properties this call gave "
                    f"{type(self).__name__}: {sorted(held)}. Forward mode "
                    f"differentiates a property the tissue carries; for a "
                    f"sequence argument, build a cost and call backward()."
                )
            if not torch.is_floating_point(held[name]):
                raise ValueError(f"{name!r} does not carry a derivative")
        batch = self._batch(held)

        # Only the differentiated properties are widened, and only because one
        # directional derivative has to cover every voxel. Widening the rest
        # would report them as maps and put their terms back in the kernel.
        # Broadcasting also leaves zero-stride views, which forward mode cannot
        # attach a tangent to, hence the copy.
        primals = tuple(
            held[name].expand(batch).contiguous() if batch else held[name]
            for name in names
        )
        rest = {name: value for name, value in held.items() if name not in names}

        def along(*inputs: torch.Tensor) -> torch.Tensor:
            declared = {**rest, **dict(zip(names, inputs, strict=True))}
            return self._shaped(self.evaluate(declared, **sequence), batch)

        columns = []
        signal = None
        for name in names:
            tangents = tuple(
                torch.ones_like(value) if other == name else torch.zeros_like(value)
                for other, value in zip(names, primals, strict=True)
            )
            signal, column = torch.func.jvp(along, primals, tangents)
            columns.append(column)
        jacobian = columns[0] if isinstance(diff, str) else torch.stack(columns, dim=-2)
        return like(signal, backend), like(jacobian, backend)

    # -- what a subclass does not have to write -----------------------------

    def _backend(self, values: Mapping[str, Any]) -> Any:
        """Return the array namespace the caller's own arrays belong to.

        The first argument carrying a buffer decides; ``None`` -- a torch
        tensor, or nothing but plain numbers -- asks for no conversion.
        """
        return brought(values.values())

    def _fix(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Read values to hold on the model, as a call would read them.

        A declared property is read as torch here rather than at the call: one
        left as NumPy would decide the answer's array library, and the gradient
        a design loop needs does not survive the trip out of torch and back. A
        sequence argument keeps whatever it was given as, since a plain number
        carries more precision than a default-dtype tensor and a sequence's
        event times are accumulated from exactly these.
        """
        declared = set(self.exposes)
        return {
            name: as_torch(value) if name in declared or is_array(value) else value
            for name, value in values.items()
        }

    def _split(self, values: Mapping[str, Any]) -> tuple[dict, dict]:
        """Tell the property arguments from the sequence ones."""
        declared = self.accepts
        held = {name: as_torch(values[name]) for name in declared if name in values}
        sequence = {
            name: value for name, value in values.items() if name not in declared
        }
        return held, sequence

    def _batch(self, held: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
        """Return the voxel shape the declared maps agree on.

        Empty when every property is a scalar, which is what makes a
        single-voxel call give one signal rather than a batch of one.
        """
        if not held:
            return ()
        return tuple(torch.broadcast_shapes(*(value.shape for value in held.values())))

    def _shaped(self, signal: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
        """Give the voxel axis the shape the caller's properties had."""
        voxels = 1
        for size in batch:
            voxels *= size
        if signal.ndim and signal.shape[0] == voxels:
            return signal.reshape(*batch, *signal.shape[1:])
        return signal


# %% private module subroutines


def _moved(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """The mapping with every tensor in it on ``device``, the rest untouched."""
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in values.items()
    }
