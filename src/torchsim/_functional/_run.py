"""Running one model from a flat argument list."""

from __future__ import annotations

__all__ = ["evaluated"]

from collections.abc import Sequence
from typing import Any

import torch

from ..model import Simulator


def evaluated(
    model: Simulator,
    diff: str | Sequence[str] | None,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the signal, and its Jacobian where one was asked for.

    Property and sequence arguments are given together; the simulator tells
    them apart, reads them in whatever array library they were written in, and
    gives the answer back in the same one.

    Parameters
    ----------
    model:
        The simulator to run.
    diff:
        What to differentiate with respect to, or ``None`` for the signal
        alone.
    device:
        Where to run, or ``None`` to follow the inputs.
    values:
        The simulator's property and sequence arguments.

    Returns
    -------
    torch.Tensor or tuple
        The signal, or the signal and its Jacobian.
    """
    given = {name: value for name, value in values.items() if value is not None}
    if device is not None:
        given["device"] = device
    if diff is None:
        return model.simulate(**given)
    return model.jacobian(diff, **given)
