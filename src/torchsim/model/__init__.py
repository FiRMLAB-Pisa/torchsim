"""Authoring signal models.

A model is written in two pieces. :class:`SpinPhysics` says what a voxel holds
and what each kind of event does to it; :class:`Simulator` says what order the
events are played in, or -- where a closed form exists -- says the signal
outright. :class:`Simulator` is what everything downstream consumes, and is
all a closed form needs.
"""

from __future__ import annotations

__all__ = [
    "BALANCED",
    "REFOCUSED",
    "SPOILED",
    "UNBALANCED",
    "EventOperators",
    "Simulator",
    "SpinPhysics",
]

from ._state_machine import (
    BALANCED,
    REFOCUSED,
    SPOILED,
    UNBALANCED,
    EventOperators,
    Simulator,
    SpinPhysics,
)
