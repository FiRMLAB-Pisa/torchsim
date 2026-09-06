"""Taking whatever array a caller brought, and giving it back.

A caller works in NumPy, or CuPy, or torch, and should not have to convert
anything to use a simulator. Everything inside is torch -- that is what the
kernels and the autograd graph are written against -- so the conversion
happens once on the way in and once on the way out.

It costs nothing to do. Every array library that matters exports its buffer
through DLPack, and :func:`as_torch` reads it: a NumPy array becomes a torch
tensor over the same host memory, a CuPy array a tensor over the same device
memory, and no element is copied either way. :func:`like` sends the answer
back through the same door.

**What does not survive the round trip is the autograd graph.** A gradient
needs the tensor it was taken with respect to, so a cost differentiated with
:meth:`torch.Tensor.backward` has to be built on torch inputs. Forward-mode
derivatives are unaffected -- :meth:`~torchsim.model.Simulator.jacobian`
takes them internally and hands back plain arrays in whatever the caller
brought.
"""

from __future__ import annotations

__all__ = [
    "arrays",
    "as_torch",
    "backend_of",
    "brought",
    "is_array",
    "like",
    "matched",
    "read",
]

import sys
from collections.abc import Mapping
from typing import Any

import torch

# What ``__dlpack_device__`` calls a host buffer, as against a CUDA one.
_CPU_DEVICE = 1


def as_torch(value: Any) -> Any:
    """Return ``value`` as torch, over the same memory where it can be.

    A torch tensor is returned unchanged. Anything exporting DLPack -- NumPy,
    CuPy, anything else that speaks the protocol -- is wrapped without a copy.
    A Python number or a nested sequence is built into a tensor, since there
    is no buffer to share.

    Parameters
    ----------
    value:
        Whatever the caller passed.

    Returns
    -------
    The value as a torch tensor, or unchanged if it was one.
    """
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "__dlpack__"):
        try:
            return torch.from_dlpack(value)
        except (RuntimeError, TypeError, BufferError):
            # A read-only or otherwise unshareable buffer still has to arrive.
            return torch.as_tensor(_copied(value))
    return torch.as_tensor(value)


def backend_of(value: Any) -> Any:
    """Return the array namespace ``value`` belongs to, or ``None``.

    ``None`` for a torch tensor and for anything with no buffer of its own --
    a Python float, a list -- since neither asks for an answer in a particular
    library.
    """
    if isinstance(value, torch.Tensor) or not hasattr(value, "__dlpack__"):
        return None
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is not None:
        return namespace()
    # CuPy and others still predate the array-API attribute; their module is
    # the namespace, and it is the one that exports ``from_dlpack``.
    return sys.modules.get(type(value).__module__.split(".")[0])


def is_array(value: Any) -> bool:
    """Whether ``value`` carries a buffer rather than being one number."""
    return isinstance(value, torch.Tensor) or hasattr(value, "__dlpack__")


def brought(values: Any) -> Any:
    """Return the array namespace the first array among ``values`` belongs to.

    The *first* array decides, torch included -- a torch tensor answers
    ``None``, meaning no conversion. Scanning past it would let a later NumPy
    argument, a flip-angle schedule say, decide what a torch caller gets back.
    """
    for value in values:
        if is_array(value):
            return backend_of(value)
    return None


def like(result: torch.Tensor, backend: Any) -> Any:
    """Return ``result`` in ``backend``, or unchanged when there is none.

    Parameters
    ----------
    result:
        What the simulation produced.
    backend:
        An array namespace from :func:`backend_of`.

    Returns
    -------
    The result as the caller's own array type.
    """
    if backend is None or not isinstance(result, torch.Tensor):
        return result
    converted = backend.from_dlpack(_beside(result, backend))
    return converted


def read(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``values`` with every array read as torch.

    Only things with a buffer are touched. A Python number is left alone: it
    carries more precision than a default-dtype tensor would, and a sequence's
    event times are accumulated from exactly these, so converting one moves
    the timestamps of the whole train.
    """
    return {
        name: as_torch(value) if is_array(value) else value
        for name, value in values.items()
    }


def arrays(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``values`` with every number read as torch, scalars included.

    For a closed form, where the arithmetic is torch's and there are no event
    times to accumulate, so nothing is lost by giving a lone number a tensor
    of its own.
    """
    return {
        name: value if isinstance(value, (str, bool, type(None))) else as_torch(value)
        for name, value in values.items()
    }


def matched(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as one entry per entry of ``reference``.

    A sequence gives its per-event parameters either one apiece or as a single
    value the whole train shares. Both arrive here and leave the same shape,
    so a layout is written once rather than twice.

    Raises
    ------
    ValueError
        If ``value`` has entries of its own and they do not match.
    """
    held = as_torch(value).to(reference.dtype)
    if held.numel() == 1:
        return held.reshape(()).expand(reference.shape)
    if held.shape != reference.shape:
        raise ValueError(
            f"expected one value per event or a single one, got {tuple(held.shape)} "
            f"against {tuple(reference.shape)}"
        )
    return held


# %% private module subroutines


def _copied(value: Any) -> Any:
    """Return a writable copy of a buffer torch declined to share."""
    namespace = backend_of(value)
    return value if namespace is None else namespace.asarray(value, copy=True)


def _beside(result: torch.Tensor, backend: Any) -> torch.Tensor:
    """Put ``result`` on the kind of device ``backend`` reads from.

    DLPack does not cross between host and device, so a run that went to a
    card has to come back before a NumPy caller can be handed it, and the
    reverse for a CuPy one.
    """
    probe = getattr(backend, "zeros", None)
    if probe is None:
        return result.detach()
    kind = probe(1).__dlpack_device__()[0]
    if kind == _CPU_DEVICE:
        return result.detach().cpu()
    if result.device.type == "cpu":
        return result.detach().cuda()
    return result.detach()
