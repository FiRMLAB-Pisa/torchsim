"""Composable modules a sequence description is assembled from.

An operator is one module -- a pulse, a readout, a delay, a preparation --
that knows what it plays and how long it holds the timeline. :func:`compose`
lays modules end to end and turns their spans into the absolute timestamps a
description carries, which is the accumulator a builder would otherwise keep
for itself. A sequence whose timing is computed rather than accumulated calls
:meth:`Operator.emit` at the time it wants instead.

An operator names a thing rather than an action -- ``Readout``, ``Dephase``,
``SSFPFidReadout`` -- and is capitalized to say so, even where it is a function
rather than a class. ``bSSFPReadout`` keeps the lowercase ``b`` the sequence is
written with everywhere else.

Operators speak only the vocabulary the fused kernels already implement: the
three event types, the four dephasing actions, and the RF and ADC roles. A
module that needs nothing beyond that -- a T2 preparation, a
magnetization-transfer preparation, a shaped or per-channel pulse -- is
written here and reaches the kernels with no change to them.

What an operator cannot say is *how much* a gradient dephases. A description
carries one crusher moment for the whole sequence and dephasing is quantized
to whole configuration orders, so a bipolar pair, a b-value of its own, or a
crusher of twice its neighbour's area have no representation here.
"""

from __future__ import annotations

__all__ = [
    "Delay",
    "Dephase",
    "Excitation",
    "FSEReadout",
    "Inversion",
    "Operator",
    "Readout",
    "Refocusing",
    "SPGRReadout",
    "SSFPEchoReadout",
    "SSFPFidReadout",
    "SampledPulse",
    "Saturation",
    "Spoil",
    "bSSFPReadout",
    "compose",
    "module",
    "operator",
    "operator_names",
    "register_operator",
]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ._description import (
    AdcRole,
    EventAction,
    EventType,
    RfUse,
    SequenceEvent,
)

# Seconds to the microseconds a description timestamps in.
_TO_US = 1e6


@dataclass(frozen=True)
class Operator:
    """One module of a sequence: what it plays, and how long it lasts.

    Parameters
    ----------
    emit : callable
        Given the absolute time the module starts, in seconds, the events it
        plays.
    duration_s : float or torch.Tensor, optional
        What the module holds the timeline for. Zero for a module that plays
        instantaneously and leaves the next one to start where it did.
    """

    emit: Callable[[Any], tuple[SequenceEvent, ...]]
    duration_s: Any = 0.0

    def __matmul__(self, other: Operator) -> Operator:
        """``a @ b``: the two played back to back, as one operator.

        The same thing :func:`module` does for a list, spelled for the common
        case of joining two -- a sample and the gradient after it, a pulse and
        its crusher -- so a handler reads as the composition it is.
        """
        return module(self, other, duration_s=self.duration_s + other.duration_s)


def compose(
    *parts: Operator | tuple[Any, Operator], start_s: Any = 0.0
) -> tuple[tuple[SequenceEvent, ...], Any]:
    """Lay operators out and return their events and the span they cover.

    A bare operator starts where the one before it ended. One given as
    ``(offset_s, operator)`` starts that far after ``start_s`` instead, which
    is how a sequence that times itself from an echo rather than by
    accumulating says so.

    Parameters
    ----------
    parts : Operator or tuple, optional
        The operators, in the order they play.
    start_s : float or torch.Tensor, optional
        When the first one starts.

    Returns
    -------
    tuple
        The events at absolute times, and where the last one ended.
    """
    events: list[SequenceEvent] = []
    cursor = start_s
    for part in parts:
        offset, played = part if isinstance(part, tuple) else (None, part)
        when = cursor if offset is None else start_s + offset
        events.extend(played.emit(when))
        cursor = when + played.duration_s
    return tuple(events), cursor


def module(*parts: Operator | tuple[Any, Operator], duration_s: Any) -> Operator:
    """Package several operators as one that can be placed and repeated.

    Offsets inside are relative to wherever the module itself is placed, which
    is what lets a shot -- a pulse, a sample some way into the repetition, and
    the rest of the repetition after it -- be one thing a caller lays down
    once.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        events, _ = compose(*parts, start_s=start_s)
        return events

    return Operator(emit, duration_s)


def Excitation(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """Return a pulse that tips magnetization into the transverse plane."""

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.EXCITATION,
                flip_rad,
                phase_rad,
                frequency_hz,
                shim_id,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Refocusing(
    flip_rad: Any,
    phase_rad: Any = 0.0,
    *,
    crushed: bool = True,
    duration_s: Any = 0.0,
    definition_id: int = 0,
    frequency_hz: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Return a pulse the sequence sits between crushers, unless told otherwise.

    ``crushed`` is what makes the pair of unbalanced gradients around the
    pulse part of the module rather than something the caller remembers to
    add.
    """
    action = (
        EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER
        if crushed
        else EventAction.NONE
    )

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.REFOCUSING,
                flip_rad,
                phase_rad,
                frequency_hz,
                shim_id,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Inversion(
    *,
    duration_s: Any = 0.0,
    flip_rad: Any = torch.pi,
    phase_rad: Any = 0.0,
    definition_id: int = 0,
) -> Operator:
    """Return an ideal inversion, scaled by the tissue's inversion efficiency.

    The kernels answer this with ``Mz -> -efficiency * Mz`` on every order and
    no rotation at all, so the pulse deposits nothing in a bound pool. A
    preparation whose saturation matters is an :func:`excitation` or a
    :func:`saturation` carrying a real waveform instead.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.INVERSION,
                flip_rad,
                phase_rad,
            ),
        )

    return Operator(emit, duration_s)


def Saturation(
    definition_id: int,
    flip_rad: Any,
    *,
    offset_hz: Any = 0.0,
    phase_rad: Any = 0.0,
    duration_s: Any = 0.0,
    shim_id: int = 0,
) -> Operator:
    """Return an off-resonance pulse that deposits power in a semisolid pool.

    The deposit follows the waveform ``definition_id`` names and the offset it
    is played at, read against the tissue's lineshape, so this is an
    :class:`RfUse.EXCITATION` carrying a real shape rather than an ideal
    rotation.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.rf(
                _TO_US * start_s,
                definition_id,
                RfUse.EXCITATION,
                flip_rad,
                phase_rad,
                offset_hz,
                shim_id,
            ),
        )

    return Operator(emit, duration_s)


def SampledPulse(
    waveform_rad_per_s: Any,
    dwell_s: float,
    *,
    phase_rad: Any = 0.0,
    offset_hz: Any = 0.0,
    definition_id: int = 0,
    shim_id: int = 0,
) -> Operator:
    """Return a pulse played as one hard pulse per sample, with free precession between.

    Each sample of the waveform turns the magnetization through its own area
    at the middle of its dwell, and between two samples the magnetization
    precesses at each pool's own frequency, relaxes and exchanges. This is the
    hard-pulse approximation [1]_, second order in the dwell with the turns
    centred; it is what a pulse needs whenever relaxation or exchange during
    it matters, a saturation pulse above all, which an instantaneous rotation
    leaves out.

    A real waveform turns through its signed area, so the negative lobes of a
    sinc turn the other way; a complex one turns about the axis its phase
    names. Played ``offset_hz`` off the centre frequency, the axis advances by
    ``-2 pi offset_hz`` radians per second from the start of the pulse, which
    puts a voxel whose ``B0`` is the offset on resonance. A ``phase_rad`` or
    ``offset_hz`` with one value per train plays one train per value.

    Every sample names ``definition_id``, which decides what it deposits in a
    semisolid pool: the ideal definition deposits nothing, and a rectangular
    one a dwell long deposits what the sample does. A semisolid pool reads its
    lineshape at the offset, so a definition other than the ideal one takes a
    single offset for every train.

    Parameters
    ----------
    waveform_rad_per_s : array-like or torch.Tensor
        The nutation rate, gamma times B1, one sample per dwell, in rad/s.
    dwell_s : float
        How long each sample is held, in seconds.
    phase_rad : float or torch.Tensor, optional
        The phase the whole pulse is played at.
    offset_hz : float or torch.Tensor, optional
        How far off the centre frequency the pulse is played.
    definition_id, shim_id : int, optional
        The definition and transmit shim every sample names.

    Returns
    -------
    Operator
        The pulse, holding the timeline for every sample's dwell.

    Raises
    ------
    ValueError
        If the waveform is empty or the dwell is not positive, or if a
        definition other than the ideal one is played at more than one
        offset.

    References
    ----------
    .. [1] Pauly, J., Le Roux, P., Nishimura, D., Macovski, A., "Parameter
       relations for the Shinnar-Le Roux selective excitation pulse design
       algorithm", IEEE Transactions on Medical Imaging 10.1 (1991),
       pp. 53-65. https://doi.org/10.1109/42.75611
    """
    samples = torch.as_tensor(waveform_rad_per_s)
    if samples.dim() != 1 or samples.numel() == 0:
        raise ValueError("a sampled pulse is a non-empty one-dimensional waveform")
    if not float(dwell_s) > 0.0:
        raise ValueError(f"dwell_s must be positive, not {dwell_s!r}")
    offset = torch.as_tensor(offset_hz, dtype=torch.float64)
    if definition_id != 0 and offset.numel() != 1:
        raise ValueError(
            "a semisolid pool reads its lineshape at one offset, so a sampled "
            "pulse naming a definition is played at one offset for every train"
        )
    frequency_hz = float(offset.reshape(-1)[0]) if offset.numel() == 1 else 0.0
    if samples.is_complex():
        areas = samples.abs() * dwell_s
        turns = samples.angle()
    else:
        areas = samples * dwell_s
        turns = torch.zeros_like(samples, dtype=torch.float64)
    middle_s = (torch.arange(samples.numel(), dtype=torch.float64) + 0.5) * dwell_s
    phases = (
        torch.as_tensor(phase_rad)[..., None]
        + turns
        - 2.0 * torch.pi * offset[..., None] * middle_s
    )
    # A value per train leads the axis a train count is read from, and a
    # plain number stays one so a train of a thousand samples packs as fast
    # as a train of a thousand ideal pulses.
    flips = list(areas.unbind(0)) if areas.requires_grad else areas.tolist()
    if phases.requires_grad:
        per_sample = list(phases.movedim(-1, 0).unbind(0))
    elif phases.dim() == 1:
        per_sample = phases.tolist()
    else:
        per_sample = list(phases.movedim(-1, 0).unbind(0))
    offsets_s = middle_s.tolist()

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return tuple(
            SequenceEvent.rf(
                _TO_US * (start_s + at_s),
                definition_id,
                RfUse.EXCITATION,
                flip,
                phase,
                frequency_hz,
                shim_id,
            )
            for at_s, flip, phase in zip(offsets_s, flips, per_sample, strict=True)
        )

    return Operator(emit, samples.numel() * dwell_s)


def Readout(
    phase_rad: Any = 0.0,
    *,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
    duration_s: Any = 0.0,
    action: EventAction = EventAction.NONE,
) -> Operator:
    """Return one sample of the transverse magnetization, demodulated by a phase.

    ``action`` is what the sequence plays after the sample -- ``SPOIL_AFTER``
    for a spoiled gradient echo, ``SHIFT_AFTER`` for an unbalanced one, and
    nothing for a train that rewinds.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        return (
            SequenceEvent.adc(
                _TO_US * start_s,
                role,
                phase_rad,
                is_echo=is_echo,
                action=action,
            ),
        )

    return Operator(emit, duration_s)


def Delay(duration_s: Any, *, action: EventAction = EventAction.NONE) -> Operator:
    """Return time passing, and whatever the sequence plays across it.

    :meth:`SequenceEvent.wait` takes no action, so a delay that spoils or
    winds the states is built here rather than by the caller reaching for the
    raw constructor.
    """

    def emit(start_s: Any) -> tuple[SequenceEvent, ...]:
        if action is EventAction.NONE:
            return (SequenceEvent.wait(_TO_US * start_s),)
        return (SequenceEvent(EventType.WAIT, _TO_US * start_s, (), action),)

    return Operator(emit, duration_s)


def Dephase() -> Operator:
    """Return an unbalanced gradient, taking no time to play.

    Every configuration order winds on by one: this is the crusher a
    refocusing pulse sits between and the unbalanced gradient an SSFP-FID
    readout is followed by, which are the same winding played in two places.
    How far it dephases is the description's ``crusher_dephasing_rad``, one
    figure for the whole sequence.
    """
    return Operator(
        lambda start_s: (
            SequenceEvent(
                EventType.WAIT, _TO_US * start_s, (), EventAction.CRUSH_AFTER
            ),
        )
    )


def Spoil() -> Operator:
    """Return ideal transverse spoiling, taking no time to play.

    Every transverse order is discarded rather than wound on, which is what a
    spoiled gradient echo assumes its spoiler and its RF phase cycling achieve
    together.
    """
    return Operator(
        lambda start_s: (
            SequenceEvent(
                EventType.WAIT, _TO_US * start_s, (), EventAction.SPOIL_AFTER
            ),
        )
    )


def bSSFPReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Return a sample the repetition rewinds after.

    The states are left where they were, which is what makes the sequence
    balanced. ``duration_s`` is what is left of the repetition after the
    sample.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SSFPFidReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Return a sample followed by one unbalanced gradient.

    ``duration_s`` is what is left of the repetition after the sample.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Dephase(),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SSFPEchoReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Return one unbalanced gradient, then the sample the echo forms at.

    The repetition's other sample. An unbalanced train winds every order on
    once per repetition, so the order the next pulse would refocus sits at
    zero once the gradient has played -- which is where this reads it, and
    what makes the sample an echo of the *previous* excitation rather than a
    free induction decay after this one. It is the strongly T2-weighted half
    of a reversed-FISP pair, and it takes the same ideal winding
    :func:`SSFPFidReadout` does, in the other order.

    ``duration_s`` is what is left of the repetition after the sample.
    """
    return module(
        Dephase(),
        Readout(phase_rad, role=role, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


def SPGRReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    role: AdcRole = AdcRole.SINGLE,
    is_echo: bool = True,
) -> Operator:
    """Return a sample followed by ideal transverse spoiling.

    ``duration_s`` is what is left of the repetition after the sample.
    """
    return module(
        Readout(phase_rad, role=role, is_echo=is_echo),
        Spoil(),
        Delay(duration_s),
        duration_s=duration_s,
    )


def FSEReadout(
    phase_rad: Any = 0.0,
    *,
    duration_s: Any = 0.0,
    is_echo: bool = True,
) -> Operator:
    """Return the sample at a spin echo, marked as the echo centre.

    A refocused train dephases around its pulses rather than after its
    samples, so nothing is played here -- what distinguishes this readout is
    that the sample is the centre of an echo, which is what a reconstruction
    reads it as. ``duration_s`` is what is left of the echo spacing after it.
    """
    return module(
        Readout(phase_rad, role=AdcRole.ECHO_CENTER, is_echo=is_echo),
        Delay(duration_s),
        duration_s=duration_s,
    )


_REGISTRY: dict[str, Callable[..., Operator]] = {
    "excitation": Excitation,
    "refocusing": Refocusing,
    "inversion": Inversion,
    "saturation": Saturation,
    "sampled-pulse": SampledPulse,
    "readout": Readout,
    "delay": Delay,
    "dephase": Dephase,
    "spoil": Spoil,
    "bssfp-readout": bSSFPReadout,
    "ssfp-fid-readout": SSFPFidReadout,
    "ssfp-echo-readout": SSFPEchoReadout,
    "spgr-readout": SPGRReadout,
    "fse-readout": FSEReadout,
}


def register_operator(name: str, factory: Callable[..., Operator]) -> None:
    """Make an operator reachable by name.

    A stream that arrives already labelled -- from an MRD file, or from any
    generator that names what it plays -- dispatches through this rather than
    through the caller's own mapping.

    Raises
    ------
    ValueError
        If the name is already registered.
    """
    normalized = name.lower().replace("_", "-")
    if normalized in _REGISTRY:
        raise ValueError(f"operator {name!r} is already registered")
    _REGISTRY[normalized] = factory


def operator(name: str) -> Callable[..., Operator]:
    """Look up an operator factory by name.

    Raises
    ------
    ValueError
        If no operator of that name is registered.
    """
    normalized = name.lower().replace("_", "-")
    try:
        return _REGISTRY[normalized]
    except KeyError as error:
        raise ValueError(f"unknown operator {name!r}") from error


def operator_names() -> tuple[str, ...]:
    """Every registered operator name, in registration order."""
    return tuple(_REGISTRY)
