"""Inference-only dispatch to fused EPG state-machine kernels."""

from __future__ import annotations

__all__: list[str] = []

import os
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch

from .._execution import (
    Lane,
    _Choice,
    _Offload,
    choose,
    chunk_voxels,
    execution,  # noqa: F401  -- re-exported: this is where callers look for it
    offload,  # noqa: F401
    offloading,
    stream_chunks,
    without_policy,
)
from . import _pools
from ._calibration import crossover, detection
from ._description import (
    EventType,
    RfDefinition,
    RfMode,
    RfUse,
    SequenceDescription,
    SequenceEvent,
)
from ._parameters import (
    FLOAT_INPUTS as _FLOAT_INPUTS,
)
from ._parameters import (
    NO_GEOMETRY,
    TISSUE_NAMES,
    Geometry,
    feature_mask,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import (
    OUTSIDE_THE_SUBSPACE as _OUTSIDE_THE_SUBSPACE,
)
from ._parameters import (
    PACKED_COUNT as _PACKED_COUNT,
)
from ._parameters import (
    SEED_INPUT as _SEED_INPUT,
)
from ._parameters import (
    TISSUE_COUNT as _TISSUE_COUNT,
)
from ._pools import PoolLayout, PoolTables, pools_of
from ._transition import (
    DynamicPairs,
    ExactSliceProfile,
    SliceTables,
    TransitionTable,
    dynamic_pair,
    transition_table,
)
from ._transmit import shim_rows

# A forward-mode call appends one tangent per differentiable input to the
# packed buffers; the scalar arguments follow those.
_TANGENT_END = _PACKED_COUNT + len(_FLOAT_INPUTS)

_PRE_SHIFT = 1
_POST_SHIFT = 2
_INVERSION = 4
_SPOIL_AFTER = 8
_SHIFT_AFTER = 16
_RECORD = 32
_EXCITATION = 64
_REFOCUSING = 128


@dataclass(frozen=True)
class _PackedEvents:
    duration: torch.Tensor
    kind: torch.Tensor
    flip: torch.Tensor
    phase: torch.Tensor
    action: torch.Tensor
    output_index: torch.Tensor
    shim_index: torch.Tensor
    saturation: torch.Tensor
    rf_frequency_hz: torch.Tensor
    # Which stacked table each event's pulse reads. Kept beside the event
    # buffers rather than among them: only a sequence with a table has one, and
    # it travels with the table rather than with the events.
    profile_index: torch.Tensor
    time_us: torch.Tensor
    # Signed time each sample's coherence has dephased through, in
    # microseconds. Not a kernel buffer: it is what the analytic off-resonance
    # and static-spread terms are applied with, after the states are done.
    unrefocused_us: torch.Tensor
    # Whether off-resonance may be applied to the samples rather than carried
    # by the states; see ``_windings_are_uniform``. A tensor rather than a bool
    # so that it travels with the other buffers a binding compares.
    analytic_dephasing: torch.Tensor
    event_index: torch.Tensor
    repetition: torch.Tensor
    echo: torch.Tensor

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """The event buffers the kernels take, in the order the registry sets."""
        return (
            self.duration,
            self.kind,
            self.flip,
            self.phase,
            self.action,
            self.output_index,
            self.shim_index,
            self.saturation,
            self.rf_frequency_hz,
        )

    @property
    def output_count(self) -> int:
        return int(self.time_us.numel())

    @property
    def train_count(self) -> int:
        return self.flip.shape[0] if self.is_batched else 1

    @property
    def is_batched(self) -> bool:
        """Whether the description carried a train axis, even a lone one."""
        return self.flip.dim() == 2


def _across_the_table(
    tissue: tuple[torch.Tensor, ...], locations: int
) -> tuple[torch.Tensor, ...]:
    """Copy each voxel once per slice position, leaving the transmit alone.

    A table holds the rotation itself at each position, so a position is not a
    scaling of anything -- the kernels pick the row from the voxel's index and
    the transmit field keeps the meaning it has everywhere else.
    """
    if locations == 1:
        return tissue
    return tuple(
        value[:, None].expand(-1, locations).reshape(-1).contiguous()
        for value in tissue
    )


def _dynamic_rotations(
    description: SequenceDescription,
    packed: _PackedEvents,
    sensitivities: torch.Tensor,
    *,
    repetitions: int,
    positions: torch.Tensor | None,
    rf_raster_time_s: float,
) -> tuple[Any, int]:
    """The rotation each pulse performs, voxel by voxel, and how many positions.

    One row per distinct pulse: a pair depends on the whole per-channel vector
    and on the flip driving it, so two events share a row only when they agree
    on both. Events that drive none read row zero, which nothing reads for them.

    A definition carrying a single waveform is played by every channel alike,
    which is what makes a hard or shaped pulse expressible in a train that also
    carries a dynamic one -- the kernels take one rotation mode for the whole
    sequence. The channels share it rather than each repeating it, so that a
    voxel of unit sensitivity everywhere takes the nominal flip from either
    kind of pulse and the sensitivities keep one reading across the train.
    """
    device = sensitivities.device
    channels = int(sensitivities.shape[-1])
    shared = torch.full(
        (channels,), 1.0 / channels, dtype=torch.complex128, device=device
    )
    stream = [event for _ in range(repetitions) for event in description.events]
    count = int(packed.kind.numel())
    if len(stream) != count:
        raise ValueError(
            f"the description walks {len(stream)} events and the packed stream {count}"
        )
    trains = packed.train_count
    flips = packed.flip.reshape(trains, count)

    # Two events share a row only when they are the same pulse driven to the
    # same flip. A flip carrying a gradient is never shared: the row's cotangent
    # would reach whichever event built it and none of the others it stands for.
    sharing = not flips.requires_grad
    index = torch.zeros(trains * count, dtype=torch.int32)
    halves: list[tuple[torch.Tensor, torch.Tensor]] = []
    rows: dict[Any, int] = {}
    for train in range(trains):
        for position, event in enumerate(stream):
            if event.type is not EventType.RF or event.rf_use is RfUse.INVERSION:
                continue
            definition = description.rf_definitions[event.rf_definition_id]
            flip = flips[train, position]
            key = (
                (event.rf_definition_id, float(flip.detach()))
                if sharing
                else (train, position)
            )
            row = rows.get(key)
            if row is None:
                row = len(halves)
                rows[key] = row
                halves.append(
                    dynamic_pair(
                        definition,
                        sensitivities,
                        weights=None if definition.channel_count > 1 else shared,
                        flip=flip,
                        positions=positions,
                        rf_raster_time_s=rf_raster_time_s,
                    )
                )
            index[train * count + position] = row
    if not halves:
        return None, 1 if positions is None else int(positions.numel())

    pairs = DynamicPairs(
        a=torch.stack([half[0] for half in halves]),
        b=torch.stack([half[1] for half in halves]),
        index=index.to(device),
    )
    return pairs, 1 if positions is None else int(positions.numel())


def pack_description(
    description: SequenceDescription,
    *,
    repetitions: int,
    record: str,
    device: torch.device,
    rf_raster_time_s: float = 1e-6,
    record_every: bool = False,
    slice_profile: Any = None,
) -> _PackedEvents:
    """Pack a description's events into the buffers a run would hand a kernel.

    The same call :func:`simulate_native` makes, exposed for a caller holding
    one description's structure fixed across many sets of values.
    """
    shapes = (
        _pulse_shapes(description) if _wants_a_table(description, slice_profile) else {}
    )
    return _pack_for(
        description,
        shapes,
        repetitions=repetitions,
        record=record,
        device=device,
        rf_raster_time_s=rf_raster_time_s,
        record_every=record_every,
    )


def _pack_for(
    description: SequenceDescription,
    shapes: dict[int, RfDefinition],
    *,
    repetitions: int,
    record: str,
    device: torch.device,
    rf_raster_time_s: float,
    record_every: bool = False,
) -> _PackedEvents:
    return _pack_events(
        description,
        repetitions=repetitions,
        record=record,
        device=device,
        rf_raster_time_s=rf_raster_time_s,
        record_every=record_every,
        shim_rows=shim_rows(description),
        table_rows={key: row for row, key in enumerate(shapes)},
    )


def _wants_a_table(description: SequenceDescription, slice_profile: Any) -> bool:
    """Whether the pulses are to be integrated into a rotation over the slice.

    A definition carrying a waveform worth integrating asks for one by itself,
    which is what makes the mode a property of the pulse rather than of the
    call. A caller naming positions asks for one whatever the waveform is,
    since a hard pulse tabulated at a bin is the hard pulse.
    """
    if slice_profile is not None:
        return True
    # Asked once per definition rather than once per event -- the mode is a
    # property of the pulse, and a train of a thousand refocusings names one --
    # and memoized on the (immutable) description, since a caller holding one
    # description asks again on every run.
    cached = getattr(description, "_profiled_cache", None)
    if cached is None:
        driven = {
            event.rf_definition_id
            for event in description.events
            if event.type is EventType.RF and event.rf_use is not RfUse.INVERSION
        }
        cached = any(
            description.rf_definitions[key].rf_mode() is RfMode.PROFILED
            for key in driven
        )
        object.__setattr__(description, "_profiled_cache", cached)
    return cached


def _pulse_shapes(
    description: SequenceDescription,
) -> dict[int, RfDefinition]:
    """The RF definitions a sequence's tables are built from, in event order.

    A table is indexed by slice position and effective flip angle, so one
    covers every pulse sharing a shape however they differ in flip. Pulses of
    different shapes get one each, and the event carries which it reads.

    Ordered by first use, so the row a definition takes does not depend on how
    the description happened to number it.

    Raises
    ------
        ValueError: if the sequence plays no pulse to integrate.
    """
    used: dict[int, RfDefinition] = {}
    for event in description.events:
        if event.type is not EventType.RF or event.rf_use is RfUse.INVERSION:
            continue
        key = event.rf_definition_id
        if key not in used:
            used[key] = description.rf_definitions[key]
    if not used:
        raise ValueError(
            "a slice profile is the rotation a pulse performs across the "
            "slice, and this sequence plays no pulse to take it from"
        )
    return used


def largest_pulse_offset(description: SequenceDescription) -> float:
    """How far off centre the sequence drives a pulse, in Hz.

    A bound pool reads its lineshape at the pulse's frequency less the voxel's
    own off-resonance, and this is the half of that read the sequence fixes.
    Inversions count: they saturate the bound pool like any other pulse.
    """
    return max(
        (
            abs(float(event.rf_frequency_hz))
            for event in description.events
            if event.type is EventType.RF
        ),
        default=0.0,
    )


# The apparent diffusion coefficient is given in um**2/ms; the b-factor wants
# it in m**2/s.
_DIFFUSION_UNIT = 1e-9


def dephasing_per_m(description: SequenceDescription) -> float:
    """The winding the sequence's unbalanced gradient puts across a metre.

    Zero when no such gradient is declared, which is what leaves every
    order-weighted term out at once.
    """
    if not description.crusher_dephasing_rad:
        return 0.0
    if description.voxel_size_m is None:
        raise ValueError(
            "a crusher winds its dephasing across a voxel, so a sequence that "
            "declares crusher_dephasing_rad must declare voxel_size_m too"
        )
    return description.crusher_dephasing_rad / description.voxel_size_m


def damping_scale(description: SequenceDescription) -> float:
    """What an apparent diffusion coefficient is multiplied by to give a rate."""
    dephasing = dephasing_per_m(description)
    return dephasing * dephasing * _DIFFUSION_UNIT


def geometry_of(description: SequenceDescription) -> Geometry:
    """The two scales the kernels read a velocity through."""
    voxel = description.voxel_size_m
    return Geometry(
        flow_scale=dephasing_per_m(description),
        washout_scale=0.0 if voxel is None else 1.0 / voxel,
    )


def _order_weighted_rates(
    tissue: tuple[torch.Tensor, ...], description: SequenceDescription
) -> tuple[torch.Tensor, ...]:
    """Fold the gradient geometry into the diffusion coefficient.

    What the state machine needs of a coefficient and the geometry is one rate
    that an interval multiplies: ``k0**2 * D`` for the b-factor a state
    accumulates. It is a plain multiply, so a gradient with respect to the
    coefficient flows back through it, and a sequence that declares no crusher
    leaves the buffer exactly zero.
    """
    scaled = {"diffusion_um2_per_ms": damping_scale(description)}
    return tuple(
        value * scaled[name] if name in scaled else value
        for name, value in zip(TISSUE_NAMES, tissue, strict=True)
    )


def _applies(value: torch.Tensor) -> bool:
    """Whether this per-voxel term has anything to do to the signal.

    A term left at its identity everywhere is skipped rather than multiplied
    through, which spares an idle pass over the whole signal -- but one being
    differentiated is applied whatever it holds, since a derivative at an
    identity is a number like any other.
    """
    return bool(value.requires_grad) or bool(value.any())


def _refuse_a_spread(r2_prime_hz: torch.Tensor) -> None:
    """Refuse a field spread the sequence gives no unrefocused time to read.

    The states carry no such term -- a configuration order stands for winding
    and not for elapsed time -- so where the two do not grow together there is
    nothing to fall back on, and a spread quietly dropped would read as a
    tissue that has none.

    Raises
    ------
        ValueError: if the tissue declares a spread and the sequence does not
            wind at one steady rate.
    """
    if not _applies(r2_prime_hz):
        return
    raise ValueError(
        "t2_prime_ms damps each sample by how long it has gone unrefocused, "
        "which this sequence does not say: its dephasing order stands for the "
        "winding the gradients put on the states and not for the time they "
        "took, so the two have to grow together. A train that keeps its "
        "coherences unwound across a pulse, or whose repetitions last "
        "unlike, does not"
    )


def _analytic_dephasing(
    packed: _PackedEvents,
    tissue: tuple[torch.Tensor, ...],
    r2_prime_hz: torch.Tensor | None,
    output_shape: torch.Size,
) -> torch.Tensor | None:
    """What the field does to each sample, or ``None`` to leave it to the states.

    A bulk off-resonance turns a sample through a phase and a spread across the
    voxel damps it, both by how long the sample has gone unrefocused: the turn
    by the signed time, since a refocusing pulse winds it back, and the damping
    by its magnitude, since dephasing either side of an echo costs the same.
    They are one complex factor and are applied as one.

    ``None`` where the sequence does not admit the analytic form -- see
    :func:`_windings_are_uniform` -- and where neither term has anything to do,
    which spares an idle multiply over the whole signal. The turn is negative
    because the states are held in the frame the pulses define, which the
    off-resonance runs behind.
    """
    if not bool(packed.analytic_dephasing):
        return None
    off_resonance = tissue[5]
    turning = _applies(off_resonance)
    spreading = r2_prime_hz is not None and _applies(r2_prime_hz)
    if not turning and not spreading:
        return None
    unrefocused_s = packed.unrefocused_us.to(off_resonance.dtype) * 1e-6
    damping = (
        torch.exp(-r2_prime_hz.reshape(*output_shape, 1) * unrefocused_s.abs())
        if spreading
        else None
    )
    if not turning:
        # A real factor rather than a complex one of zero argument, which is
        # half the multiply and the same answer.
        return damping
    angle = (-2.0 * torch.pi) * off_resonance.reshape(*output_shape, 1) * unrefocused_s
    return torch.polar(torch.ones_like(angle) if damping is None else damping, angle)


def _half_turns(
    packed: _PackedEvents,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """How far each pulse and sample sits from the axis the stream shares.

    Returns the multiple of a half turn each event is out by, the sign that
    multiple puts on a flip, and the sign it puts on a recorded sample --
    ``None`` where nothing is out by one, or where the phases lie on different
    axes and no sign can bring them together.

    Turning through ``-alpha`` about an axis is turning through ``alpha`` about
    the opposite one, and demodulating a sample a half turn round negates it.
    Both hold at every phase, so subtracting the half turns is an identity in
    value and in derivative alike, and what it leaves is a stream whose pulses
    and samples share one phase -- which is what the reduced kernels ask for.
    """
    kind, phase, action = packed.kind, packed.phase, packed.action
    turning = (kind == 1) & ((action & _INVERSION) == 0)
    recording = (kind == 2) & ((action & _RECORD) != 0)
    present = torch.stack((turning.any(), recording.any())).tolist()
    if not present[0]:
        return None
    # Anchored on a sample where the stream takes any, so a train whose samples
    # already agree needs no sign on its signal.
    anchor = recording if present[1] else turning
    settled = phase.detach()
    delta = settled - settled[..., anchor][..., :1]
    on_the_axis = turning | recording
    zero = torch.zeros_like(delta)
    out_by = torch.where(on_the_axis, torch.round(delta / torch.pi), zero)
    residue = torch.where(on_the_axis, delta - out_by * torch.pi, zero)
    sign = torch.cos(out_by * torch.pi)
    summary = torch.stack(
        (residue.abs().max(), (sign < 0).any().to(delta.dtype))
    ).tolist()
    if summary[0] > _PHASE_TOLERANCE or not summary[1]:
        return None
    samples = None
    if present[1]:
        samples = sign[..., recording]
        if samples.dim() > 1:
            # The sign multiplies a signal whose train axis leads its voxels,
            # so one that differs train by train has nowhere to go.
            if not bool((samples == samples[:1]).all()):
                return None
            samples = samples[0]
    return out_by, torch.where(turning, sign, torch.ones_like(sign)), samples


def _one_axis(packed: _PackedEvents) -> tuple[_PackedEvents, torch.Tensor | None]:
    """The stream with its pulses and samples brought onto one axis.

    Returns it beside the sign its samples carry away, which the caller applies
    to the signal. The stream is returned as it stands where its pulses already
    share a phase, and where they share no axis at all.
    """
    corrections = _remembered(
        "half turns",
        (packed.kind, packed.phase, packed.action),
        lambda: _half_turns(packed),
    )
    if corrections is None:
        return packed, None
    out_by, flips, samples = corrections
    return (
        replace(
            packed,
            flip=(packed.flip * flips).contiguous(),
            phase=(packed.phase - out_by * torch.pi).contiguous(),
        ),
        samples,
    )


def simulate_native(
    description: SequenceDescription,
    prepared_tissue: tuple[torch.Tensor, ...],
    output_shape: torch.Size,
    *,
    repetitions: int,
    record: str,
    nstates: int,
    slice_profile: ExactSliceProfile | None,
    rf_raster_time_s: float,
    lineshape: Any = None,
    exchanging: bool = False,
    features: frozenset[str] | None = None,
    r2_prime_hz: torch.Tensor | None = None,
    transmit: torch.Tensor | None = None,
    packed: _PackedEvents | None = None,
    record_every: bool = False,
    pools: int = 0,
    extra_pools: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run a fused CPU/CUDA state machine with explicit AD rules.

    ``features`` names the terms the tissue gives anything to do, and the
    kernels carry those and no others. ``None`` leaves every term in.

    ``r2_prime_hz`` is the field spread across each voxel, as a rate. It
    reaches the samples rather than the states, so it is not among the prepared
    tissue the kernels take.

    ``transmit`` is the complex per-channel sensitivity, ``(voxels,
    channels)``, given when the sequence drives a pulse whose channels carry
    their own waveforms. Its rotation is then integrated per voxel rather than
    reached through a flip and a phase.

    ``packed`` is the event stream already in buffers. The description is still
    read for everything structural -- its RF definitions, its gradient
    geometry, how far it winds -- so the two must describe the same sequence;
    what is skipped is walking the events again to rebuild what only their
    values changed.

    ``pools`` counts the chemically exchanging pools the tissue carries from
    pool B on, and ``extra_pools`` holds the prepared buffers of pools C to E.
    From :data:`~torchsim.sequence._pools.TABULATE_FROM` pools on, the kernels
    read their operators from a table rather than forming them.
    """
    device = prepared_tissue[0].device
    if device.type not in {"cpu", "cuda"} or not _backend_available(device):
        return None

    asked = slice_profile if slice_profile is not None else ExactSliceProfile(points=1)
    shapes = (
        _pulse_shapes(description) if _wants_a_table(description, slice_profile) else {}
    )
    if packed is not None and (
        packed.kind.device != device
        or int(packed.kind.numel()) != repetitions * len(description.events)
    ):
        # A packing built for another device or another number of repetitions
        # is not this run's; rebuilding is slower and right.
        packed = None
    if packed is None:
        packed = _pack_for(
            description,
            shapes,
            repetitions=repetitions,
            record=record,
            device=device,
            rf_raster_time_s=rf_raster_time_s,
            record_every=record_every,
        )
    tissue = tuple(
        value.to(dtype=torch.float32).contiguous() for value in prepared_tissue
    )
    if r2_prime_hz is not None and not bool(packed.analytic_dephasing):
        _refuse_a_spread(r2_prime_hz)
    dephasing = _analytic_dephasing(packed, tissue, r2_prime_hz, output_shape)
    if dephasing is not None:
        # Held out of the states rather than zeroed in place: the kernels then
        # carry no off-resonance term and produce no derivative along one, and
        # the factor applied below carries both. What reaches them is a
        # constant, so a caller differentiating off-resonance is answered by it.
        tissue = (*tissue[:5], torch.zeros_like(tissue[5]), *tissue[6:])
        features = None if features is None else features - {"B0"}
    tissue = _order_weighted_rates(tissue, description)
    table = None
    dynamic = None
    samples = None
    if not shapes and transmit is None:
        # A shaped or per-voxel pulse is turned through by a table rather than
        # by a flip, so there is no flip to carry a sign; those go as they are.
        packed, samples = _one_axis(packed)
    if transmit is not None:
        dynamic, locations = _dynamic_rotations(
            description,
            packed,
            transmit,
            repetitions=repetitions,
            positions=None if slice_profile is None else slice_profile.positions(),
            rf_raster_time_s=rf_raster_time_s,
        )
        tissue = _across_the_table(tissue, locations)
    elif shapes:
        table = SliceTables(
            tables=tuple(
                transition_table(
                    definition,
                    asked.positions(),
                    bins=asked.bins,
                    theta_max=asked.theta_max,
                    rf_raster_time_s=rf_raster_time_s,
                )
                for definition in shapes.values()
            ),
            index=packed.profile_index,
        )
        locations = table.points
        tissue = _across_the_table(tissue, locations)
    else:
        locations = 1
    pool_values = None
    pool_index = None
    if exchanging and pools >= _pools.TABULATE_FROM:
        if extra_pools is not None and locations > 1:
            extra_pools = _across_the_table(extra_pools, locations)
        pool_values, pool_index, exchanging = _pools.tabulate(
            tissue,
            extra_pools,
            packed.duration,
            exchanging=pools,
            semisolid=lineshape is not None,
        )
    threads = int(os.environ.get("TORCHSIM_NUM_THREADS", str(torch.get_num_threads())))
    signal = _NativeEpg.apply(
        *tissue,
        packed.duration,
        packed.kind,
        packed.flip,
        packed.phase,
        packed.action,
        packed.output_index,
        packed.shim_index,
        packed.saturation,
        packed.rf_frequency_hz,
        nstates,
        packed.output_count,
        threads,
        geometry_of(description),
        table,
        None if dynamic is None else dynamic.packed(),
        None if dynamic is None else dynamic.index,
        lineshape,
        exchanging,
        features,
        pool_values,
        pool_index,
    )
    leading = (packed.train_count,) if packed.is_batched else ()
    if locations > 1:
        signal = signal.reshape(*leading, -1, locations, packed.output_count).mean(
            dim=-2
        )
    signal = signal.reshape(*leading, *output_shape, packed.output_count)
    if dephasing is not None:
        signal = signal * dephasing
    if samples is not None:
        signal = signal * samples
    return (
        signal,
        packed.time_us,
        packed.event_index,
        packed.repetition,
        packed.echo,
    )


# %% private module subroutines


@dataclass(frozen=True)
class _Placed:
    """One RF definition's pulses, and where in the stream they sit.

    ``positions`` indexes the event axis; ``angles`` and ``integral_phases``
    carry one entry per position, and a trailing train axis where the pulses
    were given a value per train.
    """

    positions: torch.Tensor
    angles: torch.Tensor
    integral_phases: torch.Tensor


def _placed_width(placed: list[_Placed], trains: int | None) -> int | None:
    """The train count the pulses imply, checked against what the rest implies."""
    for group in placed:
        for value in (group.angles, group.integral_phases):
            if value.dim() < 2:
                continue
            if trains is not None and trains != value.shape[1]:
                raise ValueError(
                    f"inconsistent train counts across events: {trains} "
                    f"and {value.shape[1]}"
                )
            trains = value.shape[1]
    return trains


def _placed_into(
    base: torch.Tensor,
    placed: list[_Placed],
    trains: int | None,
    accumulate: bool,
) -> torch.Tensor:
    """A float event buffer written a definition at a time rather than an event.

    Every pulse of one definition lands in one scatter, so a train of a thousand
    identical pulses costs one operation here instead of a thousand: a slice per
    pulse to place its flip, an addition per pulse to fold in the phase its
    envelope carries, and a stack of a thousand scalars to finish. Resolving a
    binding packs the same description several times over, under two nested
    forward-mode interpreters, which is what makes the difference worth having.

    ``accumulate`` distinguishes the two: the flip is what the pulse turns
    through and replaces what the buffer held, while the phase adds to what the
    event already carried.
    """
    # Scatter along the event axis, which is the trailing one in a batched
    # buffer, so the buffer is turned over for the scatter and back after it.
    buffer = base if trains is None else base.transpose(0, 1)
    for group in placed:
        values = group.integral_phases if accumulate else group.angles
        values = _placed_values(values, trains)
        buffer = (
            buffer.index_add(0, group.positions, values)
            if accumulate
            else buffer.index_put((group.positions,), values)
        )
    result = buffer if trains is None else buffer.transpose(0, 1)
    return result.to(torch.float32).contiguous()


def _placed_values(value: torch.Tensor, trains: int | None) -> torch.Tensor:
    """One definition's per-pulse values, widened to the buffer's train axis."""
    if trains is None or value.dim() >= 2:
        return value
    return value.unsqueeze(1).expand(value.shape[0], trains)


# Timestamps are held in float32 as often as not, so a stretch measured late in
# a long train is known to a few parts in ten million of where the train has got
# to rather than of its own length. Stretches are held to that, which on a
# five-second train is a few microseconds -- below an RF raster tick, and a turn
# of a few thousandths at any off-resonance worth simulating.
_TIMESTAMP_RESOLUTION = 16.0 * float(torch.finfo(torch.float32).eps)


def _windings_are_uniform(stretches: list[tuple[int, float]], span_us: float) -> bool:
    """Whether the dephasing order advances at one steady rate through the run.

    A bulk off-resonance turns the transverse states through a phase that grows
    with time, and the gradients wind them through one that grows with area. A
    state's configuration order stands for the second, so it stands for the
    first as well -- and the phase can be read off the sample rather than
    carried -- exactly where the two grow together: where every stretch between
    two pulses winds through the same orders in the same time.

    ``stretches`` are the ones whose transverse states reach the next pulse, as
    ``(orders, microseconds)``. A stretch that spoils is not among them, since
    nothing survives it to carry a phase across. A stretch that winds nothing is,
    and rules the analytic route out: a balanced sequence keeps its coherences
    unwound across the pulse, and their order then says nothing about how long
    they have been dephasing. So does a train whose repetition time varies,
    whose stretches wind alike but last unlike.
    """
    if not stretches:
        # Nothing survives a pulse, so nothing carries a phase across one.
        return True
    orders, duration = stretches[0]
    if orders < 1 or not duration > 0.0:
        return False
    # A refocused train's first stretch is half of the ones that follow -- the
    # excitation to the first crusher against crusher to crusher -- so each is
    # held to the time its own winding earns at the first one's rate, not to
    # the first one's time.
    tolerance = _TIMESTAMP_RESOLUTION * max(span_us, duration)
    return all(
        count >= 1 and abs(length - count * duration / orders) <= tolerance
        for count, length in stretches
    )


def _lead(value: Any) -> float:
    """One number standing for a duration that may differ per train.

    A batch whose trains are timed differently has a different structure per
    train, and the analytic terms are decided for the description as a whole.
    Such a duration comes back as a NaN, which no comparison accepts, so the
    decision falls to the states.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        first = float(value.reshape(-1)[0])
        if value.numel() > 1 and bool((value != value.reshape(-1)[0]).any()):
            return float("nan")
        return first
    return float(value)


def sample_times_us(interval_us: torch.Tensor, recorded: torch.Tensor) -> torch.Tensor:
    """When each recorded sample is taken, from the intervals that precede it.

    A stream's clock is the running total of the gaps between its events, so a
    sample's timestamp is the sum of every interval up to and including its
    own. That is what lets a rebinding produce the times from the intervals it
    already maps, where a repetition time reaching every later timestamp is not
    something one source element accounts for; see
    :mod:`torchsim.model._binding`, which is where this is read.

    The sum is taken at double width. The intervals are float32 and a long
    train accumulates thousands of them, which is a rounding the samples' own
    width has no room to absorb.

    Parameters
    ----------
    interval_us:
        How long each event of the packed stream sits after the one before it,
        in microseconds.
    recorded:
        Which of those events an ADC records, in stream order.

    Returns
    -------
    torch.Tensor
        One timestamp per recorded sample, in microseconds.
    """
    return torch.cumsum(interval_us.double(), -1)[..., recorded].to(torch.float32)


def _pack_events(
    description: SequenceDescription,
    *,
    repetitions: int,
    record: str,
    device: torch.device,
    rf_raster_time_s: float,
    record_every: bool = False,
    shim_rows: dict[int, int] | None = None,
    table_rows: dict[int, int] | None = None,
) -> _PackedEvents:
    """The buffers a run hands the kernels.

    ``repetitions`` plays the stream that many times and records the last of
    them, which is how a sequence is carried into the state a scanner plays it
    in. The magnetization crosses every playing as it stands; only the
    recording is suppressed, so a settling playing costs its arithmetic and
    none of the signal.

    ``record_every`` keeps them all instead, which is what reading the settled
    signal off a handful of playings needs and no caller asks for directly.
    """
    # Values stay as plain Python numbers unless the description carries
    # tensors (an optimizer differentiating through flip angles), so the common
    # case builds each buffer with a single allocation instead of one scalar
    # tensor per event.
    shim_rows = shim_rows or {}
    table_rows = table_rows or {}
    # When each event happens, kept as the description gives it. The intervals
    # the kernels take are one difference over the whole stream at the end, and
    # the recorded times are a slice of it -- so an event whose timing is a
    # tensor costs nothing here, and a walk under forward mode, of which
    # resolving a binding does several, issues no dispatch per event at all.
    clock: list[Any] = []
    kinds: list[int] = []
    flips: list[Any] = []
    phases: list[Any] = []
    actions: list[int] = []
    output_indices: list[int] = []
    shim_indices: list[int] = []
    profile_indices: list[int] = []
    saturations: list[float] = []
    frequencies: list[float] = []
    since_pulse: list[Any] = []
    shifts_since_pulse: list[int] = []
    event_indices: list[int] = []
    repetition_indices: list[int] = []
    echo_flags: list[bool] = []
    # When the pulse that last turned the transverse states was played. Where
    # the dephasing order advances at one steady rate -- see
    # ``_windings_are_uniform`` -- every state carries the off-resonance phase
    # of its order at each pulse, so the coherence a sample reads at order zero
    # has dephased through the time since the last pulse less the time the
    # shifts since that pulse stand for, whichever pathway it took. A bulk
    # off-resonance turns the sample through that signed time and a static
    # field spread damps by it, which is what lets both be applied to the
    # echoes rather than carried by the states; see ``analytic_dephasing``.
    turned_at: Any = 0.0
    # What the analytic terms need to know about the structure: how fast the
    # dephasing order advances in each stretch between pulses, and whether the
    # transverse states survive that stretch at all. ``stretch_orders`` counts
    # the shifts since the last pulse, and ``order_us`` is the time one of them
    # stands for, read off the first stretch that winds and survives.
    stretches: list[tuple[int, float]] = []
    stretch_orders = 0
    stretch_spoiled = False
    order_us: Any = None
    # Nothing transverse exists before the first pulse that lays some down, so
    # what a preparation spends its time doing says nothing about the winding.
    turning_yet = False
    output_index = 0
    # Where each pulse's flip angle goes, gathered by the definition that says
    # what its envelope integrates to.
    turned: dict[int, list[tuple[int, Any]]] = {}

    for repetition in range(repetitions):
        recording = record_every or repetition == repetitions - 1
        # A single playing never advances by the repetition time, so it does
        # not matter what shape that time has -- which is what lets a protocol
        # give each train in a batch its own.
        repetition_offset = (
            description.tr_duration_us * repetition if repetition else 0.0
        )
        for event_index, event in enumerate(description.events):
            absolute = (
                repetition_offset + event.timestamp_us
                if repetition
                else event.timestamp_us
            )
            clock.append(absolute)
            kinds.append(int(event.type))
            action = int(event.action)
            flip: Any = 0.0
            phase: Any = 0.0
            event_output_index = -1
            shim_indices.append(
                shim_rows.get(event.rf_shim_id, 0) if event.type is EventType.RF else 0
            )
            profile_indices.append(
                table_rows.get(event.rf_definition_id, 0)
                if event.type is EventType.RF
                else 0
            )
            saturation = 0.0
            frequency = 0.0
            if (action & _PRE_SHIFT) != 0:
                stretch_orders += 1
            if event.type is EventType.RF:
                phase = event.rf_phase_rad
                frequency = event.rf_frequency_hz
                saturation = description.rf_definitions[
                    event.rf_definition_id
                ].saturation(rf_raster_time_s=rf_raster_time_s)
                if event.rf_use is RfUse.INVERSION:
                    # An inversion turns no transverse magnetization, so the
                    # stretch it sits in runs on through it, shifts and all.
                    action |= _INVERSION
                else:
                    since_turning = absolute - turned_at
                    if turning_yet and not stretch_spoiled:
                        span_us = float(_lead(since_turning))
                        if span_us > 0.0:
                            stretches.append((stretch_orders, span_us))
                            if order_us is None and stretch_orders > 0:
                                order_us = since_turning / stretch_orders
                    turning_yet = True
                    stretch_orders = 0
                    stretch_spoiled = False
                    # The pulse's role, carried explicitly rather than inferred
                    # from the crusher bits a given policy happens to set.
                    refocusing = event.rf_use is RfUse.REFOCUSING
                    action |= _REFOCUSING if refocusing else _EXCITATION
                    turned_at = absolute
                    # What the pulse turns through is settled after the walk,
                    # one call per definition rather than one per event: the
                    # arithmetic is the same either way, and a train of a
                    # thousand pulses is a thousand dispatches saved on every
                    # packing -- of which resolving a binding does several.
                    turned.setdefault(event.rf_definition_id, []).append(
                        (len(flips), event.rf_amplitude_hz)
                    )
            elif event.type is EventType.ADC:
                phase = event.adc_phase_rad
                if recording and _record_event(event, record):
                    action |= _RECORD
                    event_output_index = output_index
                    since_pulse.append(absolute - turned_at)
                    shifts_since_pulse.append(stretch_orders)
                    event_indices.append(event_index)
                    repetition_indices.append(repetition if record_every else 0)
                    echo_flags.append(event.is_echo)
                    output_index += 1
            if (action & _POST_SHIFT) != 0:
                stretch_orders += 1
            if (action & _SPOIL_AFTER) != 0:
                stretch_spoiled = True
            elif (action & _SHIFT_AFTER) != 0:
                stretch_orders += 1
            output_indices.append(event_output_index)
            flips.append(flip)
            phases.append(phase)
            actions.append(action)
            saturations.append(saturation)
            frequencies.append(frequency)

    placed: list[_Placed] | None = []
    for definition_id, occurrences in turned.items():
        definition = description.rf_definitions[definition_id]
        gathered = _gathered_amplitudes(
            [amplitude for _, amplitude in occurrences], device
        )
        if gathered is None or placed is None:
            # A group whose pulses disagree about their shape -- some a number,
            # some a value per train -- is rare enough to be left to the plain
            # path, where each occurrence stands on its own.
            placed = None
            continue
        angles, integral_phases = definition.flip_angle(
            gathered, rf_raster_time_s=rf_raster_time_s
        )
        placed.append(
            _Placed(
                torch.as_tensor(
                    [position for position, _ in occurrences],
                    dtype=torch.int64,
                    device=device,
                ),
                angles,
                integral_phases,
            )
        )
    if placed is None:
        for definition_id, occurrences in turned.items():
            definition = description.rf_definitions[definition_id]
            for position, amplitude in occurrences:
                angle, integral_phase = definition.flip_angle(
                    amplitude, rf_raster_time_s=rf_raster_time_s
                )
                flips[position] = angle
                phases[position] = phases[position] + integral_phase

    trains = _batch_width([*flips, *phases])
    if placed is not None:
        trains = _placed_width(placed, trains)
    flip_buffer = _stack_values(flips, device, trains)
    phase_buffer = _stack_values(phases, device, trains)
    if placed:
        flip_buffer = _placed_into(flip_buffer, placed, trains, accumulate=False)
        phase_buffer = _placed_into(phase_buffer, placed, trains, accumulate=True)
    # The clock the stream ran on, and the intervals and sample times read off
    # it: one difference and one slice over the whole stream, where the walk
    # would have taken a subtraction per event.
    clock_us = _stack_clock(clock, device)
    intervals_s = (torch.diff(clock_us, prepend=clock_us.new_zeros(1)) * 1e-6).to(
        torch.float32
    )
    output_buffer = torch.as_tensor(output_indices, dtype=torch.int32, device=device)
    recorded = output_buffer >= 0
    return _PackedEvents(
        duration=(
            intervals_s if trains is None else intervals_s.expand(trains, -1)
        ).contiguous(),
        kind=torch.as_tensor(kinds, dtype=torch.int32, device=device).contiguous(),
        flip=flip_buffer,
        phase=phase_buffer,
        action=torch.as_tensor(actions, dtype=torch.uint8, device=device).contiguous(),
        output_index=output_buffer.contiguous(),
        shim_index=torch.as_tensor(
            shim_indices, dtype=torch.int32, device=device
        ).contiguous(),
        saturation=torch.as_tensor(
            saturations, dtype=torch.float32, device=device
        ).contiguous(),
        rf_frequency_hz=torch.as_tensor(
            frequencies, dtype=torch.float32, device=device
        ).contiguous(),
        profile_index=torch.as_tensor(
            profile_indices, dtype=torch.int32, device=device
        ).contiguous(),
        time_us=clock_us[recorded].to(torch.float32).contiguous(),
        unrefocused_us=_stack_values(
            [
                since if order_us is None or not shifts else since - shifts * order_us
                for since, shifts in zip(since_pulse, shifts_since_pulse, strict=True)
            ],
            device,
        ),
        analytic_dephasing=torch.tensor(
            _windings_are_uniform(stretches, float(_lead(clock[-1] if clock else 0.0))),
            dtype=torch.bool,
            device=device,
        ),
        event_index=torch.as_tensor(event_indices, dtype=torch.int64, device=device),
        repetition=torch.as_tensor(
            repetition_indices, dtype=torch.int64, device=device
        ),
        echo=torch.as_tensor(echo_flags, dtype=torch.bool, device=device),
    )


def _gathered_amplitudes(amplitudes: list[Any], device: torch.device) -> Any:
    """The pulse amplitudes of one definition as one tensor, or ``None``.

    ``None`` says they cannot be stacked as they are -- which is the signal to
    take them one at a time rather than to convert them, since converting is
    the per-event work this exists to avoid.

    A number among tensors is widened to their shape rather than refused: a
    refocused train whose excitation is a fixed angle and whose refocusing
    pulses are a schedule is one definition holding exactly that mixture, and
    it is the common case rather than the awkward one.
    """
    tensors = [
        amplitude for amplitude in amplitudes if isinstance(amplitude, torch.Tensor)
    ]
    if not tensors:
        return torch.tensor(amplitudes, dtype=torch.float32, device=device)
    first = tensors[0]
    if not all(
        amplitude.shape == first.shape
        and amplitude.dtype is first.dtype
        and amplitude.device == first.device
        for amplitude in tensors
    ):
        return None
    if len(tensors) == len(amplitudes):
        return torch.stack(amplitudes).to(device=device, dtype=torch.float32)
    if not all(
        isinstance(amplitude, (int, float)) or isinstance(amplitude, torch.Tensor)
        for amplitude in amplitudes
    ):
        return None
    widened = [
        amplitude
        if isinstance(amplitude, torch.Tensor)
        else torch.full_like(first, float(amplitude))
        for amplitude in amplitudes
    ]
    return torch.stack(widened).to(device=device, dtype=torch.float32)


def _record_event(event: SequenceEvent, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "acquired":
        return event.adc_role.value != 0
    return event.is_echo


def _scalar(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        # Packing a train of a thousand events issues these one per event, and
        # under the forward-mode passes that resolve a binding it issues them
        # again per seed, so a conversion that would not change anything is
        # worth not making.
        if value.ndim == 0 and value.dtype is torch.float32 and value.device == device:
            return value
        return value.to(device=device, dtype=torch.float32).reshape(())
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(())


def _stack_clock(values: list[Any], device: torch.device) -> torch.Tensor:
    """The stream's timestamps, at the width the description timed them in.

    A description that times itself in Python arithmetic keeps double width
    here rather than being narrowed first: a train played many times over runs
    a clock into the millions of microseconds, where float32 cannot resolve the
    interval between two neighbouring events at all, and the intervals are read
    off this by differencing it.
    """
    if not values:
        return torch.empty(0, dtype=torch.float64, device=device)
    if any(isinstance(value, torch.Tensor) for value in values):
        return torch.stack([_scalar(value, device) for value in values])
    return torch.tensor(values, dtype=torch.float64, device=device)


def _stack_values(
    values: list[Any], device: torch.device, trains: int | None = None
) -> torch.Tensor:
    """Pack per-event values, keeping any autograd graph intact.

    ``trains`` is the width the caller requires: ``None`` packs ``(n_events,)``,
    an integer packs ``(n_trains, n_events)`` and broadcasts scalar values
    across the train axis. The three float event buffers must be packed to the
    same width, since the kernels stride all of them by ``event_count``.
    """
    if not values:
        return torch.empty(0, dtype=torch.float32, device=device)
    if trains is None:
        if any(isinstance(value, torch.Tensor) for value in values):
            stacked = torch.stack([_scalar(value, device) for value in values])
            return stacked.to(torch.float32).contiguous()
        return torch.tensor(values, dtype=torch.float32, device=device).contiguous()
    columns = [_broadcast_to_trains(value, trains, device) for value in values]
    return torch.stack(columns, dim=1).to(torch.float32).contiguous()


def _batch_width(values: list[Any]) -> int | None:
    """Train count implied by ``values``, or ``None`` when all are scalar.

    A train axis is recognized by rank, not by size, so a lone train stays
    batched: ``(1, n_events)`` buffers and an output that keeps the train axis.
    Keying off the size instead would make a one-train batch silently change
    shape.
    """
    trains: int | None = None
    for value in values:
        if not isinstance(value, torch.Tensor) or value.dim() == 0:
            continue
        if value.dim() != 1:
            raise ValueError(
                f"batched event values must be one-dimensional, got {tuple(value.shape)}"
            )
        if trains is not None and trains != value.shape[0]:
            raise ValueError(
                f"inconsistent train counts across events: {trains} and {value.shape[0]}"
            )
        trains = value.shape[0]
    return trains


def _broadcast_to_trains(value: Any, trains: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(trains)
        return tensor.reshape(trains)
    return torch.full((trains,), float(value), dtype=torch.float32, device=device)


def _stack(
    values: list[torch.Tensor], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.stack(values).to(dtype=dtype)


def _spread(
    gradients: tuple[torch.Tensor, ...], needed: tuple[bool, ...]
) -> tuple[torch.Tensor | None, ...]:
    """Place the differentiable gradients back among all the packed inputs."""
    lookup = dict(zip(_FLOAT_INPUTS, gradients, strict=True))
    return tuple(
        lookup[index] if index in lookup and needed[index] else None
        for index in range(_PACKED_COUNT)
    )


def _wanted(needs_input_grad: tuple[bool, ...]) -> tuple[bool, ...]:
    """Which differentiable inputs the caller will read, in gradient order."""
    return tuple(bool(needs_input_grad[index]) for index in _FLOAT_INPUTS)


def _extras(results: Sequence[Any], dynamic: bool, tabulated: bool) -> tuple[Any, Any]:
    """The pair's and the pools' entries past the differentiable inputs'.

    Each is present only for a run that carries it, the pair's first.
    """
    rest = list(results[len(_FLOAT_INPUTS) :])
    pair = rest.pop(0) if dynamic and rest else None
    pools = rest.pop(0) if tabulated and rest else None
    return pair, pools


def _needs(ctx: Any, index: int) -> bool:
    """Whether an input past the ones every call passes wants a gradient."""
    return len(ctx.needs_input_grad) > index and bool(ctx.needs_input_grad[index])


def _directed(pools: PoolTables | None, direction: Any) -> PoolTables | None:
    """The tables, carrying a direction along them for a pass that follows one."""
    if pools is None:
        return None
    return replace(pools, direction=_followed(direction, pools.values))


class _LastDerivative(torch.autograd.Function):
    """An identity whose own derivative is refused.

    The kernels reach two orders. A third would be asked of results that came
    out of them as plain tensors, while the inputs those results were computed
    from are still in the graph -- so autograd can route around the missing
    term and return an answer that is quietly short of it. Passing the results
    through here puts a node in the way that says so instead.
    """

    @staticmethod
    def forward(count: int, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(value.view_as(value) for value in values[:count])

    @staticmethod
    def setup_context(
        _ctx: Any, _inputs: tuple[Any, ...], _output: tuple[torch.Tensor, ...]
    ) -> None:
        return None

    @staticmethod
    def backward(_ctx: Any, *_cotangents: torch.Tensor) -> tuple[Any, ...]:
        raise RuntimeError(
            "the fused EPG kernels give first and second derivatives; a third "
            "would need a pass that is not written"
        )


def _last(
    results: tuple[torch.Tensor | None, ...], sources: Sequence[torch.Tensor]
) -> tuple[torch.Tensor | None, ...]:
    """Refuse a further derivative of ``results``, which the kernels cannot give.

    Only when a graph is being built, which is the sole way a third derivative
    can be reached.
    """
    if not torch.is_grad_enabled():
        return results
    graphed = tuple(value for value in sources if value.requires_grad)
    positions = [index for index, value in enumerate(results) if value is not None]
    if not graphed or not positions:
        return results
    guarded = _LastDerivative.apply(
        len(positions), *(results[index] for index in positions), *graphed
    )
    replaced = list(results)
    for index, value in zip(positions, guarded, strict=True):
        replaced[index] = value
    return tuple(replaced)


class _NativeEpg(torch.autograd.Function):
    @staticmethod
    def forward(
        t1: torch.Tensor,
        t2: torch.Tensor,
        m0: torch.Tensor,
        b1: torch.Tensor,
        b1_phase: torch.Tensor,
        b0: torch.Tensor,
        inversion_efficiency: torch.Tensor,
        diffusion: torch.Tensor,
        velocity: torch.Tensor,
        bound_fraction: torch.Tensor,
        bound_exchange: torch.Tensor,
        t1_bound: torch.Tensor,
        pool_b_fraction: torch.Tensor,
        pool_b_exchange: torch.Tensor,
        t1_pool_b: torch.Tensor,
        t2_pool_b: torch.Tensor,
        pool_b_shift: torch.Tensor,
        duration: torch.Tensor,
        kind: torch.Tensor,
        flip: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor,
        output_index: torch.Tensor,
        shim_index: torch.Tensor,
        saturation: torch.Tensor,
        rf_frequency_hz: torch.Tensor,
        state_count: int,
        output_count: int,
        threads: int,
        geometry: Geometry,
        profile: Any,
        pair_values: torch.Tensor | None,
        pair_index: torch.Tensor | None,
        lineshape: Any,
        exchanging: bool | PoolLayout,
        features: frozenset[str] | None,
        pool_values: torch.Tensor | None = None,
        pool_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tissue = (
            t1,
            t2,
            m0,
            b1,
            b1_phase,
            b0,
            inversion_efficiency,
            diffusion,
            velocity,
            bound_fraction,
            bound_exchange,
            t1_bound,
            pool_b_fraction,
            pool_b_exchange,
            t1_pool_b,
            t2_pool_b,
            pool_b_shift,
        )
        events = (
            duration,
            kind,
            flip,
            phase,
            action,
            output_index,
            shim_index,
            saturation,
            rf_frequency_hz,
        )
        return _run_packed(
            tissue,
            events,
            state_count,
            output_count,
            threads,
            geometry=geometry,
            profile=profile,
            dynamic=_pairs_of(pair_values, pair_index),
            lineshape=lineshape,
            exchanging=exchanging,
            features=features,
            pools=pools_of(pool_values, pool_index, exchanging),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        tensors = inputs[:_PACKED_COUNT]
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)
        ctx.state_count = inputs[_PACKED_COUNT]
        ctx.output_count = inputs[_PACKED_COUNT + 1]
        ctx.threads = inputs[_PACKED_COUNT + 2]
        ctx.geometry = inputs[_PACKED_COUNT + 3]
        ctx.profile = inputs[_PACKED_COUNT + 4]
        ctx.pair_values = inputs[_PACKED_COUNT + 5]
        ctx.pair_index = inputs[_PACKED_COUNT + 6]
        ctx.lineshape = inputs[_PACKED_COUNT + 7]
        ctx.exchanging = inputs[_PACKED_COUNT + 8]
        ctx.features = inputs[_PACKED_COUNT + 9]
        ctx.pool_values = (
            inputs[_PACKED_COUNT + 10] if len(inputs) > _PACKED_COUNT + 10 else None
        )
        ctx.pool_index = (
            inputs[_PACKED_COUNT + 11] if len(inputs) > _PACKED_COUNT + 11 else None
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: torch.Tensor | None) -> torch.Tensor:
        saved = ctx.saved_tensors
        float_tangents = tuple(
            torch.zeros_like(saved[index])
            if tangents[index] is None
            else tangents[index].to(saved[index]).contiguous()
            for index in _FLOAT_INPUTS
        )
        return _NativeEpgJvp.apply(
            *saved,
            *float_tangents,
            ctx.state_count,
            ctx.output_count,
            ctx.threads,
            ctx.geometry,
            ctx.profile,
            ctx.pair_values,
            ctx.pair_index,
            _followed(tangents[_PACKED_COUNT + 5], ctx.pair_values),
            ctx.lineshape,
            ctx.exchanging,
            ctx.features,
            ctx.pool_values,
            ctx.pool_index,
            _followed(
                tangents[_PACKED_COUNT + 10]
                if len(tangents) > _PACKED_COUNT + 10
                else None,
                ctx.pool_values,
            ),
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        wanted = _wanted(ctx.needs_input_grad)
        fused = _NativeEpgVjp.apply(
            *saved,
            grad_output,
            ctx.state_count,
            ctx.output_count,
            ctx.threads,
            wanted,
            ctx.geometry,
            ctx.profile,
            ctx.pair_values,
            ctx.pair_index,
            ctx.lineshape,
            ctx.exchanging,
            ctx.features,
            ctx.pool_values,
            ctx.pool_index,
        )
        # The rotation and the pool operators are worked out before the
        # kernels run, so what they return for either is a cotangent on the
        # table itself; autograd carries it the rest of the way to what the
        # table was formed from.
        pair_grad, pool_grad = _extras(
            fused, ctx.pair_values is not None, ctx.pool_values is not None
        )
        return (
            *_spread(fused[: len(_FLOAT_INPUTS)], ctx.needs_input_grad),
            None,
            None,
            None,
            None,
            None,
            pair_grad if ctx.needs_input_grad[_PACKED_COUNT + 5] else None,
            None,
            None,
            None,
            None,
            pool_grad if _needs(ctx, _PACKED_COUNT + 10) else None,
            None,
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *inputs: Any,
    ) -> tuple[torch.Tensor, int]:
        return _loop_vmap(_NativeEpg, info.batch_size, in_dims, inputs)


class _NativeEpgVjp(torch.autograd.Function):
    """The adjoint ``J(x)^T w``, itself differentiable once more.

    Its own backward is two more kernel calls. Contracting the gradients
    with a direction ``v`` gives ``<v, J(x)^T w> = <J(x) v, w>``, a form that is
    linear in ``w`` and whose derivatives are therefore the two passes already
    written: the forward-over-reverse contraction for ``d/dx``, and a plain
    directional derivative ``J(x) v`` for ``d/dw``.
    """

    @staticmethod
    def forward(*inputs: Any) -> tuple[torch.Tensor, ...]:
        primal = inputs[:_PACKED_COUNT]
        return _run_packed_vjp(
            primal[:_TISSUE_COUNT],
            primal[_TISSUE_COUNT:_PACKED_COUNT],
            inputs[_SEED_INPUT],
            state_count=inputs[_SEED_INPUT + 1],
            output_count=inputs[_SEED_INPUT + 2],
            threads=inputs[_SEED_INPUT + 3],
            wanted=inputs[_SEED_INPUT + 4],
            geometry=inputs[_SEED_INPUT + 5],
            profile=inputs[_SEED_INPUT + 6],
            dynamic=_pairs_of(inputs[_SEED_INPUT + 7], inputs[_SEED_INPUT + 8]),
            lineshape=inputs[_SEED_INPUT + 9],
            exchanging=inputs[_SEED_INPUT + 10],
            features=inputs[_SEED_INPUT + 11],
            pools=pools_of(
                inputs[_SEED_INPUT + 12],
                inputs[_SEED_INPUT + 13],
                inputs[_SEED_INPUT + 10],
            ),
        )

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], _output: tuple[torch.Tensor, ...]
    ) -> None:
        ctx.save_for_backward(*inputs[: _SEED_INPUT + 1])
        ctx.state_count = inputs[_SEED_INPUT + 1]
        ctx.output_count = inputs[_SEED_INPUT + 2]
        ctx.threads = inputs[_SEED_INPUT + 3]
        ctx.geometry = inputs[_SEED_INPUT + 5]
        ctx.profile = inputs[_SEED_INPUT + 6]
        ctx.pair_values = inputs[_SEED_INPUT + 7]
        ctx.pair_index = inputs[_SEED_INPUT + 8]
        ctx.lineshape = inputs[_SEED_INPUT + 9]
        ctx.exchanging = inputs[_SEED_INPUT + 10]
        ctx.features = inputs[_SEED_INPUT + 11]
        ctx.pool_values = inputs[_SEED_INPUT + 12]
        ctx.pool_index = inputs[_SEED_INPUT + 13]

    @staticmethod
    def backward(ctx: Any, *cotangents: torch.Tensor | None) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        primal, seed = saved[:_PACKED_COUNT], saved[_SEED_INPUT]
        dynamic = _pairs_of(ctx.pair_values, ctx.pair_index)
        pools = pools_of(ctx.pool_values, ctx.pool_index, ctx.exchanging)
        # The pair's and the pools' own cotangents follow the others, and each
        # is a direction along its table exactly as the others are along the
        # buffers they belong to.
        pair_direction, pool_direction = _extras(
            cotangents, dynamic is not None, pools is not None
        )
        cotangents = cotangents[: len(_FLOAT_INPUTS)]
        pair_direction = _followed(pair_direction, ctx.pair_values)
        if pools is not None:
            pools = replace(pools, direction=_followed(pool_direction, ctx.pool_values))
        directions = tuple(
            torch.zeros_like(primal[index])
            if cotangent is None
            else cotangent.to(primal[index]).contiguous()
            for index, cotangent in zip(_FLOAT_INPUTS, cotangents, strict=True)
        )
        wanted = _wanted(ctx.needs_input_grad)
        curvature, _ = _run_packed_vjp_jvp(
            primal[:_TISSUE_COUNT],
            primal[_TISSUE_COUNT:_PACKED_COUNT],
            directions,
            seed,
            state_count=ctx.state_count,
            output_count=ctx.output_count,
            threads=ctx.threads,
            wanted=wanted,
            geometry=ctx.geometry,
            profile=ctx.profile,
            dynamic=dynamic,
            dynamic_direction=pair_direction,
            lineshape=ctx.lineshape,
            exchanging=ctx.exchanging,
            features=ctx.features,
            pools=pools,
        )
        seed_grad = None
        if ctx.needs_input_grad[_SEED_INPUT]:
            seed_grad = _run_packed_jvp(
                primal[:_TISSUE_COUNT],
                primal[_TISSUE_COUNT:_PACKED_COUNT],
                directions[:_TISSUE_COUNT],
                directions[_TISSUE_COUNT:],
                ctx.state_count,
                ctx.output_count,
                ctx.threads,
                geometry=ctx.geometry,
                profile=ctx.profile,
                dynamic=dynamic,
                dynamic_direction=pair_direction,
                lineshape=ctx.lineshape,
                exchanging=ctx.exchanging,
                features=ctx.features,
                pools=pools,
            )
        pair_curvature, pool_curvature = _extras(
            curvature, dynamic is not None, pools is not None
        )
        guarded = _last(
            (
                *_spread(curvature[: len(_FLOAT_INPUTS)], ctx.needs_input_grad),
                seed_grad,
            ),
            saved,
        )
        return (
            *guarded,
            None,
            None,
            None,
            None,
            None,
            None,
            pair_curvature if ctx.needs_input_grad[_SEED_INPUT + 7] else None,
            None,
            None,
            None,
            None,
            pool_curvature if _needs(ctx, _SEED_INPUT + 12) else None,
            None,
        )


class _NativeEpgJvp(torch.autograd.Function):
    @staticmethod
    def forward(*inputs: Any) -> torch.Tensor:
        saved = inputs[:_PACKED_COUNT]
        tangents = inputs[_PACKED_COUNT:_TANGENT_END]
        tissue_tangents = tangents[:_TISSUE_COUNT]
        event_tangents = tangents[_TISSUE_COUNT:]
        return _run_packed_jvp(
            saved[:_TISSUE_COUNT],
            saved[_TISSUE_COUNT:_PACKED_COUNT],
            tissue_tangents,
            event_tangents,
            inputs[_TANGENT_END],
            inputs[_TANGENT_END + 1],
            inputs[_TANGENT_END + 2],
            geometry=inputs[_TANGENT_END + 3],
            profile=inputs[_TANGENT_END + 4],
            dynamic=_pairs_of(inputs[_TANGENT_END + 5], inputs[_TANGENT_END + 6]),
            dynamic_direction=inputs[_TANGENT_END + 7],
            lineshape=inputs[_TANGENT_END + 8],
            exchanging=inputs[_TANGENT_END + 9],
            features=inputs[_TANGENT_END + 10],
            pools=_directed(
                pools_of(
                    inputs[_TANGENT_END + 11],
                    inputs[_TANGENT_END + 12],
                    inputs[_TANGENT_END + 9],
                ),
                inputs[_TANGENT_END + 13],
            ),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        ctx.save_for_backward(*inputs[:_TANGENT_END])
        ctx.state_count = inputs[_TANGENT_END]
        ctx.output_count = inputs[_TANGENT_END + 1]
        ctx.geometry = inputs[_TANGENT_END + 3]
        ctx.profile = inputs[_TANGENT_END + 4]
        ctx.pair_values = inputs[_TANGENT_END + 5]
        ctx.pair_index = inputs[_TANGENT_END + 6]
        ctx.pair_direction = inputs[_TANGENT_END + 7]
        ctx.lineshape = inputs[_TANGENT_END + 8]
        ctx.exchanging = inputs[_TANGENT_END + 9]
        ctx.features = inputs[_TANGENT_END + 10]
        ctx.pool_values = inputs[_TANGENT_END + 11]
        ctx.pool_index = inputs[_TANGENT_END + 12]
        ctx.pool_direction = inputs[_TANGENT_END + 13]
        ctx.threads = inputs[_TANGENT_END + 2]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        saved = ctx.saved_tensors
        primal = saved[:_PACKED_COUNT]
        tangent = saved[_PACKED_COUNT:_TANGENT_END]
        # Each is wanted if either the primal input or its forward direction
        # is being differentiated.
        wanted = tuple(
            bool(ctx.needs_input_grad[index])
            or bool(ctx.needs_input_grad[_PACKED_COUNT + position])
            for position, index in enumerate(_FLOAT_INPUTS)
        )
        primal_grads, tangent_grads = _run_packed_vjp_jvp(
            primal[:_TISSUE_COUNT],
            primal[_TISSUE_COUNT:_PACKED_COUNT],
            tangent,
            grad_output,
            state_count=ctx.state_count,
            output_count=ctx.output_count,
            threads=getattr(ctx, "threads", 0),
            wanted=wanted,
            geometry=ctx.geometry,
            profile=ctx.profile,
            dynamic=_pairs_of(ctx.pair_values, ctx.pair_index),
            dynamic_direction=ctx.pair_direction,
            lineshape=ctx.lineshape,
            exchanging=ctx.exchanging,
            features=ctx.features,
            pools=_directed(
                pools_of(ctx.pool_values, ctx.pool_index, ctx.exchanging),
                ctx.pool_direction,
            ),
        )
        dynamic, tabulated = ctx.pair_values is not None, ctx.pool_values is not None
        pair_curvature, pool_curvature = _extras(primal_grads, dynamic, tabulated)
        pair_slope, pool_slope = _extras(tangent_grads, dynamic, tabulated)
        directions = tuple(
            gradient if ctx.needs_input_grad[_PACKED_COUNT + offset] else None
            for offset, gradient in enumerate(tangent_grads[: len(_FLOAT_INPUTS)])
        )
        guarded = _last(
            (
                *_spread(primal_grads[: len(_FLOAT_INPUTS)], ctx.needs_input_grad),
                *directions,
            ),
            saved,
        )
        return (
            *guarded,
            None,
            None,
            None,
            None,
            None,
            pair_curvature if ctx.needs_input_grad[_TANGENT_END + 5] else None,
            None,
            pair_slope if ctx.needs_input_grad[_TANGENT_END + 7] else None,
            None,
            None,
            None,
            pool_curvature if _needs(ctx, _TANGENT_END + 11) else None,
            None,
            pool_slope if _needs(ctx, _TANGENT_END + 13) else None,
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        *inputs: Any,
    ) -> tuple[torch.Tensor, int]:
        return _loop_vmap(_NativeEpgJvp, info.batch_size, in_dims, inputs)


def _backend_available(device: torch.device) -> bool:
    try:
        if device.type == "cpu":
            from torchsim import _epg_cpu  # noqa: F401
        else:
            from . import _epg_triton  # noqa: F401
    except ImportError:
        return False
    return True


def _loop_vmap(
    function: type[torch.autograd.Function],
    batch_size: int,
    in_dims: tuple[int | None, ...],
    inputs: tuple[Any, ...],
) -> tuple[torch.Tensor, int]:
    outputs = []
    for batch in range(batch_size):
        sliced = tuple(
            value if dimension is None else value.select(dimension, batch)
            for value, dimension in zip(inputs, in_dims, strict=True)
        )
        outputs.append(function.apply(*sliced))
    return torch.stack(outputs), 0


# How far two RF phases may drift and still count as the same one.
_PHASE_TOLERANCE = 1e-5

_WHOLE_TURN = 2.0 * torch.pi


_REMEMBERED: OrderedDict[
    tuple[int, ...],
    tuple[tuple[weakref.ref[torch.Tensor], ...], tuple[int, ...], Any],
] = OrderedDict()

# A binding resolves one sequence and rebinds its values, so a handful of event
# streams is what a design loop or a dictionary sweep alternates between, and an
# entry costs a tuple of weak references.
_REMEMBERED_LIMIT = 32


def _remembered(
    tag: str, inputs: tuple[torch.Tensor, ...], read: Callable[[], Any]
) -> Any:
    """``read()``, computed once per set of buffers it would read.

    ``tag`` names what is being remembered, so that two questions asked of the
    same buffers do not answer each other.

    The answer is a property of the buffers, so it changes only when one of them
    is replaced or written to. Keying on identity and version says exactly that,
    and both are free where ``read`` is a device round trip: a call that hits
    touches no memory the kernel does not touch anyway.

    The references are weak, so a remembered answer never keeps a chunk of a
    streamed run alive, and a dead reference is a miss rather than a match
    against whatever later took the same address.

    An in-place write that bypasses a tensor's version counter is not seen,
    which is the assumption autograd already makes of the same buffers.
    """
    try:
        key = (tag, *map(id, inputs))
        versions = tuple(value._version for value in inputs)
        references = tuple(weakref.ref(value) for value in inputs)
    except (AttributeError, TypeError):
        return read()
    found = _REMEMBERED.get(key)
    if found is not None:
        seen_references, seen_versions, answer = found
        if seen_versions == versions and all(
            reference() is value
            for reference, value in zip(seen_references, inputs, strict=True)
        ):
            _REMEMBERED.move_to_end(key)
            return answer
    answer = read()
    _REMEMBERED[key] = (references, versions, answer)
    _REMEMBERED.move_to_end(key)
    while len(_REMEMBERED) > _REMEMBERED_LIMIT:
        _REMEMBERED.popitem(last=False)
    return answer


def forget_event_summaries() -> None:
    """Drop every remembered event-stream summary, so the next call reads again."""
    _REMEMBERED.clear()


def real_subspace_axis(
    events: tuple[torch.Tensor, ...],
    tissue: tuple[torch.Tensor, ...],
    profile: Any = None,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> int | None:
    """The real axis the states are confined to, or ``None``.

    Returns 1 when every pulse in the sequence turns the same way about one
    axis -- all the RF phases equal -- and there is neither off-resonance,
    transmit phase, nor spin velocity. The states then start real, and every operator the kernels carry
    keeps them real: the rotation about a single axis has real coefficients on
    the axis the signal lies on, the shift's conjugate coupling becomes a sign,
    relaxation and spoiling scale, and an ideal inversion scales the
    longitudinal states alone.

    A refocused train whose excitation shares the phase of its refocusing
    pulses is the CPMG arrangement; a spoiled or unbalanced train whose pulses
    all share one phase is the same condition with no refocusing pulse in it,
    which is what a fingerprinting schedule is.

    Pulses a half turn apart are refused although their states do stay real.
    They turn about one axis in opposite senses, and what the reduced kernels
    carry is a flip angle with no sign to say which sense -- so an excitation a
    half turn from its refocusing pulses, or a train alternating its phase,
    goes to the full kernels.

    Flow dephasing turns each dephasing order through a phase of its own, which
    is a rotation out of the axis rather than a scaling along it. Washout stays
    on the axis -- it scales both relaxation factors and nothing more -- but the
    real kernels do not carry it, so any velocity at all is refused here
    whatever geometry drives it.

    A transmit array is not on that list: a row per shim scales the flip its
    own pulse turns through, which is a scaling along the axis. Its phase is,
    and a shimmed array is held to the same zero as a single map.

    Two arrangements make the recorded echoes look real without confining the
    states, and both are refused. Off-resonance is refocused at the echo
    centers but not between them. An excitation a quarter turn from the
    refocusing pulses gives a signal that is real at every sample while its
    states fill the plane -- a real projection of a complex trajectory, not a
    subspace a kernel can be built on, and the phase spread is what rules it
    out.

    ``features`` names the terms the caller's tissue gives anything to do, as
    :func:`torchsim.sequence._parameters.features_of` reads them off the values
    passed in. A term absent from that set is at its identity everywhere, which
    settles three of the four questions here without reading a buffer at all.
    """
    if profile is not None or dynamic is not None:
        # A shaped pulse turns about an axis with a component along z, which
        # makes its Cayley-Klein ``a`` complex -- and a complex ``a`` is
        # exactly what carries a state out of the real subspace. A per-voxel
        # pair is that same rotation reached another way, and the real kernels
        # take no pair argument at all: a verdict of 1 would not slow the
        # answer down, it would drop the pulse.
        return None
    if not _tissue_stays_on_the_axis(tissue, features):
        return None
    has_pulse, spread = _pulse_axis(events)
    if not has_pulse or spread > _PHASE_TOLERANCE:
        return None
    return 1


def _tissue_stays_on_the_axis(
    tissue: tuple[torch.Tensor, ...],
    features: frozenset[str] | None,
) -> bool:
    """Whether off-resonance, transmit phase and velocity are all absent.

    A feature the caller did not declare is one whose parameter was left at its
    identity, so the answer is already known and no buffer is touched. What is
    declared is still reduced over, since a property given as a full tensor is
    declared whatever it holds, and a map of zeros is a map a run should not
    lose the fast path to. The three reductions are stacked into one transfer:
    read back separately they would be three synchronizations on a device.
    """
    wanted = {"B0", "B1_PHASE", "FLOW"}
    if features is not None and not (wanted & features):
        return True
    b1_phase, b0 = tissue[4], tissue[5]
    velocity = tissue[TISSUE_NAMES.index("velocity_m_per_s")]
    answers = torch.stack(
        ((b0 == 0).all(), (b1_phase == 0).all(), (velocity == 0).all())
    ).tolist()
    return all(answers)


def _pulse_axis(events: tuple[torch.Tensor, ...]) -> tuple[bool, float]:
    """Whether the stream turns any spins, and how far its phases spread.

    The spread covers the pulses and the samples together: the reduced kernels
    report the transverse state in the frame the pulses set and demodulate by
    nothing, so a sample demodulated by some other phase is one they cannot
    produce.

    An ideal inversion is left out of that spread: it scales the longitudinal
    states and turns nothing, so whatever sits in its phase buffer says nothing
    about the axis the transverse states lie on.

    The answer belongs to the event stream, which a binding resolves once and
    reuses for every call afterwards, so it is remembered against the buffers
    it was read from rather than recomputed per call.
    """
    return _remembered(
        "pulse axis",
        (events[1], events[3], events[4]),
        lambda: _read_pulse_axis(events),
    )


def _read_pulse_axis(events: tuple[torch.Tensor, ...]) -> tuple[bool, float]:
    """Reduce the phases to one flag and one number in one round trip."""
    kind, phase, action = events[1], events[3], events[4]
    turning = (kind == 1) & ((action & _INVERSION) == 0)
    on_the_axis = turning | ((kind == 2) & ((action & _RECORD) != 0))
    # A whole turn, not half of one. Pulses a half turn apart turn about one
    # axis in opposite senses, and the reduced kernels carry a flip without a
    # sign to say which -- so a packing brings them into line before this is
    # asked, negating the flips it turns round; see ``_one_axis``. What reaches
    # here disagreeing by a half turn is a stream no packing settled.
    wrapped = torch.remainder(phase, _WHOLE_TURN)
    infinity = torch.full_like(wrapped, float("inf"))

    def extent(values: torch.Tensor) -> torch.Tensor:
        lowest = torch.where(on_the_axis, values, infinity).min()
        highest = torch.where(on_the_axis, values, -infinity).max()
        return highest - lowest

    # One set that straddles zero -- phases a hair under a whole turn beside
    # phases a hair over it -- has the full spread on this scale and none on a
    # scale turned half of the way round, so the smaller of the two is the
    # spread that means what it says.
    summary = torch.stack(
        (
            turning.any().to(wrapped.dtype),
            torch.minimum(
                extent(wrapped),
                extent(torch.remainder(wrapped + torch.pi, _WHOLE_TURN)),
            ),
        )
    ).tolist()
    return bool(summary[0]), summary[1]


def _auto_real_axis(
    kind: str,
    events: tuple[torch.Tensor, ...],
    tissue: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...] = (),
    profile: Any = None,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> int | None:
    """The subspace verdict, when deciding it costs less than it saves.

    Deciding scans the phase buffer, so it is cheap next to a kernel over many
    atoms and states but not next to a small one; ``detection`` is where that
    line falls.

    ``tangents`` are the forward-mode directions, if any. A seed along
    off-resonance, transmit phase or RF phase leaves the subspace however real
    the primal is, and the real kernels do not produce those derivatives, so
    their presence rules the fast path out.
    """
    work = int(tissue[0].numel()) * _train_count(events) * int(events[1].numel())
    if work < detection(kind, tissue[0].device):
        return None
    if tangents:
        seeded = torch.stack([(direction != 0).any() for direction in tangents]).any()
        if bool(seeded):
            return None
    return real_subspace_axis(events, tissue, profile, dynamic, features)


def _auto_real_axis_adjoint(
    events: tuple[torch.Tensor, ...],
    tissue: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    wanted: tuple[bool, ...] | None,
    profile: Any = None,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> int | None:
    """The subspace verdict for an adjoint, given what the caller will read.

    A real adjoint returns zero for the three gradients that leave the
    subspace, and those are not zero in truth, so it is only available to a
    caller that has said which gradients it wants and left those out. ``None``
    for ``wanted`` means all of them, which rules the fast path out.

    ``tangents`` may be empty, which is a pass that follows no direction at
    all and so seeds nothing outside the subspace.
    """
    if wanted is None or any(wanted[index] for index in _OUTSIDE_THE_SUBSPACE):
        return None
    return _auto_real_axis(
        "adjoint",
        events,
        tissue,
        tuple(tangents[index] for index in _OUTSIDE_THE_SUBSPACE) if tangents else (),
        profile=profile,
        dynamic=dynamic,
        features=features,
    )


def _pointers(values: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    """Raw addresses for the kernels, after checking they may be indexed flatly.

    The kernels stride these pointers themselves, so a non-contiguous argument
    -- a stride-0 broadcast view of a scalar property, most easily -- would be
    read past its real extent. A device pointer is worse still: the address is
    valid to take and meaningless to dereference on the host. Checking here
    costs far less than the kernel and turns silent corruption into an
    exception.
    """
    for value in values:
        if value.device.type != "cpu":
            raise ValueError(
                f"the CPU kernels need CPU tensors, got one on {value.device}"
            )
        if not value.is_contiguous():
            raise ValueError(
                f"kernel buffers must be contiguous, got a {tuple(value.shape)} "
                f"tensor with strides {value.stride()}"
            )
    return tuple(value.data_ptr() for value in values)


def _profiled_pointers(
    values: tuple[torch.Tensor, ...],
    table: torch.Tensor | None,
    rows: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """Buffer addresses with the transition tables' and their index's on the end.

    A sequence with no table still passes both slots, as nulls the kernels
    check to pick the flip-and-phase operator instead.
    """
    if table is None:
        return (*_pointers(values), 0, 0)
    return _pointers((*values, table, rows))


def _dynamic_pointers(
    pairs: torch.Tensor | None,
    rows: torch.Tensor | None,
    direction: torch.Tensor | None,
    gradient: torch.Tensor | None = None,
    curvature: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """The per-voxel rotations, their per-event index and a direction along them.

    A sequence that drives no pulse of varying channel weights passes nulls,
    which is what leaves the kernels on the flip-and-phase operator or on the
    tabulated one. The direction is null for a pass that follows none.
    """
    if pairs is None:
        return (0, 0, 0, 0, 0)
    held = _pointers((pairs, rows))
    tail = tuple(
        0 if value is None else _pointers((value,))[0]
        for value in (direction, gradient, curvature)
    )
    return (*held, *tail)


def _followed(direction: Any, values: torch.Tensor | None) -> Any:
    """A direction along the pair, as the kernels take one.

    Forward mode leaves a tangent out where the input it belongs to is not
    being followed; the kernels read a buffer rather than a null there, so a
    pair that is present is followed by zeros rather than by nothing.
    """
    if values is None:
        return None
    return torch.zeros_like(values) if direction is None else direction


def _pairs_of(values: torch.Tensor | None, index: torch.Tensor | None) -> Any:
    """The per-voxel rotations a packed buffer and its index describe.

    The pair crosses the autograd boundary as a plain tensor so that the
    cotangent the kernels return for it has somewhere to go; this is where it
    becomes a pair again.
    """
    if values is None:
        return None
    from ._transition import DynamicPairs

    return DynamicPairs.from_packed(values, index)


def _carries_the_pair(dynamic: Any, route: str) -> None:
    """Refuse a route that cannot take the per-voxel rotations with it.

    Every other table a pulse reads is the same for every voxel, so a route
    that cuts the volume up or moves a piece of it at a time can broadcast
    one. The pair is per voxel, so it has to be cut the same way.

    Raises
    ------
        NotImplementedError: if a dynamic pulse reaches such a route.
    """
    if dynamic is not None:
        raise NotImplementedError(
            f"a pulse whose channel weights vary carries a rotation per "
            f"voxel, which the {route} route does not cut yet"
        )


def _carries_the_pools(pools: PoolTables | None, route: str) -> None:
    """Refuse a route that cannot take the pool tables with it.

    The tables are per voxel, so a route that cuts the volume up or moves a
    piece of it at a time has to cut them the same way.

    Raises
    ------
        NotImplementedError: if a tabulated tissue reaches such a route.
    """
    if pools is not None:
        raise NotImplementedError(
            f"a tissue with more than one exchanging pool carries its "
            f"operators per voxel, which the {route} route does not cut yet"
        )


def _moved_dynamic(dynamic: Any, device: torch.device) -> Any:
    """The per-voxel rotations on another device, whichever shape they came in.

    A pair set travels as a :class:`DynamicPairs`; a direction along one
    travels as the packed buffer alone.
    """
    if dynamic is None:
        return None
    if isinstance(dynamic, torch.Tensor):
        return dynamic.to(device)
    from ._transition import DynamicPairs

    return DynamicPairs(
        a=dynamic.a.to(device),
        b=dynamic.b.to(device),
        index=dynamic.index.to(device),
    )


def _bound_pointers(
    values: tuple[torch.Tensor, ...],
    table: torch.Tensor | None,
    rows: torch.Tensor | None,
    absorption: torch.Tensor | None,
    pairs: torch.Tensor | None = None,
    pair_rows: torch.Tensor | None = None,
    pair_direction: torch.Tensor | None = None,
    pair_gradient: torch.Tensor | None = None,
    pair_curvature: torch.Tensor | None = None,
    pools: PoolTables | None = None,
    pool_gradient: torch.Tensor | None = None,
    pool_curvature: torch.Tensor | None = None,
) -> tuple[int, ...]:
    """The buffers, the pool tables, the profile, the rotations, the lineshape.

    A sequence with no bound pool passes a null on the end, which is what
    selects the single-pool kernel.
    """
    profiled = _profiled_pointers(values, table, rows)
    dynamic = _dynamic_pointers(
        pairs, pair_rows, pair_direction, pair_gradient, pair_curvature
    )
    tabulated = _pool_pointers(pools, pool_gradient, pool_curvature)
    lineshape = (0,) if absorption is None else _pointers((absorption,))
    return (*profiled[:-2], *tabulated, *profiled[-2:], *dynamic, *lineshape)


def _pool_pointers(
    pools: PoolTables | None,
    gradient: torch.Tensor | None,
    curvature: torch.Tensor | None,
) -> tuple[int, ...]:
    """The pool table, its per-event index, a direction and two cotangents.

    Nulls for a run that is not tabulated.
    """
    if pools is None:
        return (0, 0, 0, 0, 0)
    held = _pointers((pools.values, pools.index))
    tail = tuple(
        0 if value is None else _pointers((value,))[0]
        for value in (pools.direction, gradient, curvature)
    )
    return (*held, *tail)


def _pool_cotangent(
    pools: PoolTables | None, events: tuple[torch.Tensor, ...]
) -> torch.Tensor | None:
    """Room for a table's cotangent, a table per train so each has one writer."""
    if pools is None:
        return None
    return torch.zeros(
        (_train_count(events), *pools.values.shape),
        dtype=pools.values.dtype,
        device=pools.values.device,
    )


def _extra_cotangents(
    pair: torch.Tensor | None, pools: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    """The pair's and the table's cotangents, for the runs that carry them."""
    return (
        *(() if pair is None else (pair,)),
        *(() if pools is None else (pools.sum(0),)),
    )


def _pool_counts(pools: PoolTables | None) -> tuple[int, int, int]:
    """The exchanging pools, the rows and the blocks a table holds."""
    if pools is None:
        return (0, 0, 0)
    layout = pools.layout
    return (layout.exchanging, layout.rows, layout.blocks)


# Which pools a kernel is to carry, as the extension's own enum reads it.
_POOL_ONE, _POOL_SEMISOLID, _POOL_EXCHANGING, _POOL_THREE = 0, 1, 2, 3
_POOL_TABULATED, _POOL_TABULATED_SATURATED = 4, 5


def _pool_kind(lineshape: Any, exchanging: bool | PoolLayout) -> int:
    """Which pools the kernels are to carry beside the free water."""
    if isinstance(exchanging, PoolLayout):
        return _POOL_TABULATED_SATURATED if exchanging.semisolid else _POOL_TABULATED
    if lineshape is not None and exchanging:
        return _POOL_THREE
    if exchanging:
        return _POOL_EXCHANGING
    return _POOL_SEMISOLID if lineshape is not None else _POOL_ONE


def _tables(profile: Any, events: tuple[torch.Tensor, ...], dynamic: Any = None) -> Any:
    """A bare table read as the one every pulse drives.

    Raises
    ------
        ValueError: if a table is given beside a per-voxel pair. The two are
            alternatives -- a rotation integrated at the voxel's own position
            already is a profile -- and the kernels read the pair, so a table
            handed over with one says nothing at all.
    """
    if profile is not None and dynamic is not None:
        raise ValueError(
            "a pulse reaches its rotation through a table over the slice or "
            "through a pair per voxel, not through both"
        )
    if isinstance(profile, TransitionTable):
        return SliceTables.alone(profile, int(events[1].numel()), events[1].device)
    return profile


def _within_the_table(profile: Any, flip: torch.Tensor) -> None:
    """Refuse a pulse the table does not reach.

    A flip past the grid saturates at the last knot, which is the right
    numerical behaviour -- a cubic run off its end leaves the unit circle --
    but a silently saturated pulse is a wrong simulation that still returns
    numbers. The nominal flip is checked because that is where the mistake
    usually is: a sequence and a table built for different rasters give flips
    that differ by the ratio and nothing says so.

    Transmit scaling on top of the nominal flip is not checked, since reading
    its largest value means a reduction over every voxel.

    Raises
    ------
        ValueError: if any pulse asks for more than the table holds.
    """
    largest = float(flip.abs().max()) if flip.numel() else 0.0
    if largest > profile.theta_max:
        raise ValueError(
            f"the sequence drives a pulse to {largest:.3f} rad, past the "
            f"{profile.theta_max:.3f} rad the transition table covers"
        )


def _unprofiled(profile: Any, what: str, active: Any) -> None:
    """Refuse a path a transition table has not reached yet.

    Raises
    ------
        NotImplementedError: if a table is present and the path is taken.
    """
    if profile is not None and active is not None:
        raise NotImplementedError(f"{what} does not carry a transition table yet")


def _shim_count(tissue: tuple[torch.Tensor, ...]) -> int:
    """How many shim rows the transmit buffers carry, one per voxel each.

    Read off the buffers rather than passed alongside them, so the stride the
    kernels use cannot disagree with the memory they are given. A transmit
    field given as one value for the whole tissue carries no row at all, and is
    the one shim it describes.
    """
    atoms = tissue[0].numel()
    return 1 if atoms == 0 else max(1, tissue[3].numel() // atoms)


def _train_count(events: tuple[torch.Tensor, ...]) -> int:
    """Number of echo trains packed into the float event buffers.

    ``duration``, ``flip`` and ``phase`` are either ``(n_events,)`` for a single
    train or ``(n_trains, n_events)``; the structural buffers never carry a
    train axis because every train shares the same event sequence.

    Raises
    ------
        ValueError: if the three float buffers disagree. The kernels stride all
            of them by ``event_count``, so a mismatch reads out of bounds.
    """
    widths = {
        value.shape[0] if value.dim() == 2 else 1
        for value in (events[0], events[2], events[3])
    }
    if len(widths) != 1:
        raise ValueError(
            f"duration/flip/phase disagree on train count: {sorted(widths)}"
        )
    return widths.pop()


# ---------------------------------------------------------------------------
# Spreading echo trains over several CUDA devices.
#
# Trains are the axis to split on: they are independent, the event buffers
# already carry them as their leading dimension, and the signal comes back with
# them leading too. Tissue is shared by every train, so it is replicated.
# ---------------------------------------------------------------------------

_DEVICES: tuple[torch.device, ...] = ()


@contextmanager
def distribute(devices: Sequence[torch.device | str]) -> Iterator[None]:
    """Spread echo trains across ``devices`` for the duration of the block.

    Only CUDA work shards, and only when there is more than one train to split;
    anything else runs where it already is. Naming one device is allowed and
    simply runs everything there.

    Raises
    ------
        ValueError: if ``devices`` is empty or names a device that is not CUDA.
    """
    global _DEVICES
    resolved = tuple(torch.device(value) for value in devices)
    if not resolved:
        raise ValueError("distribute needs at least one device")
    for device in resolved:
        if device.type != "cuda":
            raise ValueError(f"distribute is for CUDA devices, got {device}")
    previous = _DEVICES
    _DEVICES = resolved
    try:
        yield
    finally:
        _DEVICES = previous


def _shard_bounds(train_count: int) -> list[tuple[int, int, torch.device]]:
    """Contiguous train spans, one per device, remainders going to the first.

    Empty when there is nothing to gain: one device, or fewer trains than the
    call already runs in one piece.
    """
    if len(_DEVICES) < 2 or train_count < 2:
        return []
    step, extra = divmod(train_count, len(_DEVICES))
    bounds = []
    start = 0
    for index, device in enumerate(_DEVICES):
        width = step + (1 if index < extra else 0)
        if width:
            bounds.append((start, start + width, device))
        start += width
    return bounds


def _to_device(
    values: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Replicate buffers that every train shares."""
    return tuple(value.to(device) for value in values)


def _shard_events(
    events: tuple[torch.Tensor, ...], begin: int, end: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """One shard's events: the per-train buffers cut, the shared ones copied."""
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
    ) = events
    return (
        duration[begin:end].contiguous().to(device),
        kind.to(device),
        flip[begin:end].contiguous().to(device),
        phase[begin:end].contiguous().to(device),
        action.to(device),
        output_index.to(device),
        shim_index.to(device),
        saturation.to(device),
        rf_frequency.to(device),
    )


def _shard_event_tangents(
    tangents: tuple[torch.Tensor, ...], begin: int, end: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """``(duration, flip, phase)`` seeds, cut the same way as the events."""
    return tuple(value[begin:end].contiguous().to(device) for value in tangents)


def _gather_shards(
    parts: list[tuple[int, torch.Tensor]],
    home: torch.device,
    output_count: int,
) -> torch.Tensor:
    """Stack the shards' signals back into one train-leading tensor.

    A shard holding a single train gets the shape a single-train call returns,
    which has no train axis at all, so each piece is reshaped from the span it
    was asked for rather than from what came back.
    """
    return torch.cat(
        [signal.to(home).reshape(width, -1, output_count) for width, signal in parts]
    )


def _combine_shards(
    parts: tuple[tuple[torch.Tensor, ...], ...], home: torch.device
) -> tuple[torch.Tensor, ...]:
    """Reassemble one side of the second-order result from its shards.

    A tissue gradient collects a contribution from every train, so the shards
    hold partial sums that have to be added. An event gradient belongs to one
    train, so the shards hold disjoint pieces that line up end to end.
    """
    combined = []
    for index, pieces in enumerate(zip(*parts, strict=True)):
        moved = [piece.to(home) for piece in pieces]
        if index < _TISSUE_COUNT:
            combined.append(torch.stack(moved).sum(dim=0))
        else:
            combined.append(torch.cat(moved))
    return tuple(combined)


# What one voxel costs on the device, by pass. These are what both the chunk
# size and the whole-problem footprint are derived from, so they stay in step.
def _bytes_per_voxel(
    kind: str,
    train_count: int,
    event_count: int,
    output_count: int,
    state_count: int,
    real_axis: int | None,
    shims: int = 1,
) -> int:
    """Device bytes one voxel needs for one pass.

    ``kind`` is ``forward``, ``jvp``, ``adjoint`` or ``first-order adjoint``.
    """
    planes = 2 if real_axis == 1 else 4
    scratch = train_count * planes * state_count * 4
    # Two float planes of signal and the complex copy staged for the transfer.
    signal = train_count * (2 * 4 * output_count + 8 * output_count)
    # One float per tissue property, except the transmit pair, which a voxel
    # carries once per shim.
    tissue = tissue_gradient_height(shims) * 4
    if kind == "adjoint":
        # The recorded trajectory dominates by orders of magnitude: it holds
        # three state planes for every event, not just the recorded ones. The
        # cotangent adds two more float planes on top of the signal buffers,
        # and the two gradient planes another tissue buffer each.
        return (
            2 * tissue
            + signal
            + train_count * 2 * 4 * output_count
            + train_count * event_count * 3 * state_count * planes * 4
            + scratch
            + 2 * tissue
        )
    if kind == "first-order adjoint":
        # The same shape without the dual: one accumulator per gradient rather
        # than two, and half the trajectory planes, which is what makes the
        # chunks wider for one budget.
        return (
            tissue
            + signal
            + train_count * 2 * 4 * output_count
            + train_count * event_count * 3 * state_count * (planes // 2) * 4
            + scratch
        )
    inputs = 2 * tissue if kind == "jvp" else tissue
    return inputs + signal + scratch


def _chunk_voxels(plan: _Offload, kind: str, *shape: Any) -> int:
    """Largest voxel chunk whose buffers fit the budget, across every lane."""
    return chunk_voxels(plan, _bytes_per_voxel(kind, *shape))


def _choose(
    kind: str,
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_count: int,
    state_count: int,
    real_axis: int | None,
) -> _Choice | None:
    """Decide where one simulation runs, given the policy and the free memory.

    The policy is general; what an EPG pass costs is not. The work estimate and
    the per-voxel footprint are worked out here, and the decision itself is
    :func:`torchsim._execution.choose`.
    """
    train_count = _train_count(events)
    voxels = int(tissue[0].numel())
    event_count = int(events[1].numel())
    return choose(
        work=voxels * train_count * event_count,
        voxels=voxels,
        bytes_per_voxel=_bytes_per_voxel(
            kind,
            train_count,
            event_count,
            output_count,
            state_count,
            real_axis,
            _shim_count(tissue),
        ),
        crossover=crossover(kind),
    )


class _Lane(Lane):
    """The device buffers one lane reuses for every chunk of a simulation."""

    def __init__(
        self,
        device: torch.device,
        chunk: int,
        train_count: int,
        output_count: int,
        state_count: int,
        real_axis: int | None,
        voxel_inputs: int = _TISSUE_COUNT,
        shims: int = 1,
    ) -> None:
        super().__init__(device)
        self.train_count = train_count
        self.output_count = output_count
        # Tissue, and for a forward-mode pass its per-voxel seeds behind it.
        # Each is one row of voxels, except the transmit pair, which is one row
        # per shim; ``load`` packs a chunk to the same shape at its own width.
        self.rows = tissue_gradient_rows(shims) * (voxel_inputs // _TISSUE_COUNT)
        self.inputs = [
            torch.empty(rows * chunk, dtype=torch.float32, device=device)
            for rows in self.rows
        ]
        # Flat, because the kernels stride the signal by the chunk they are
        # given: a shorter final chunk needs a buffer packed for that width,
        # which a slice of a wider one is not.
        size = train_count * chunk * output_count
        self.real = torch.empty(size, dtype=torch.float32, device=device)
        self.imag = torch.empty_like(self.real)
        self.signal = torch.empty(size, dtype=torch.complex64, device=device)
        # A forward-over-reverse pass needs a trajectory and gradient
        # accumulators as well; the adjoint path hangs them here.
        self.adjoint: Any = None

    def views(self, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This lane's buffers, packed for a chunk of ``width`` voxels."""
        shape = (
            (self.train_count, width, self.output_count)
            if self.train_count > 1
            else (width, self.output_count)
        )
        size = self.train_count * width * self.output_count
        return (
            self.real[:size].view(shape),
            self.imag[:size].view(shape),
            self.signal[:size].view(shape),
        )

    def send(self, piece: torch.Tensor) -> torch.Tensor:
        """Put one chunk of a host signal on this lane's signal buffer."""
        size = piece.numel()
        return self.stage(piece, self.signal[:size].view(piece.shape))

    def load(
        self, values: Sequence[torch.Tensor], begin: int, end: int
    ) -> tuple[torch.Tensor, ...]:
        """Stage per-voxel inputs for one chunk and return the device views.

        A chunk is a slice of the voxel axis, which for the transmit pair is
        the last axis rather than the whole buffer. Each row is packed to the
        chunk's own width, because that is the stride the kernels give it, and
        goes across on its own: a row is contiguous at both ends, where the
        rows together would be a strided read that cannot transfer
        asynchronously.
        """
        width = end - begin
        staged = []
        for slot, rows, value in zip(self.inputs, self.rows, values, strict=True):
            if value.numel() == 1:
                # A property whose term the run leaves out is kept as the one
                # value it was given. There is nothing to cut along the voxel
                # axis, and the kernel takes the pointer without indexing it.
                piece = slot[:1]
                piece.copy_(value.reshape(1), non_blocking=True)
                staged.append(piece)
                continue
            piece = slot[: rows * width]
            source = value.view(rows, -1)
            for row in range(rows):
                piece[row * width : (row + 1) * width].copy_(
                    source[row, begin:end], non_blocking=True
                )
            staged.append(piece)
        return tuple(staged)


def _offload_plan(
    plan: _Offload,
    kind: str,
    events: tuple[torch.Tensor, ...],
    voxels: int,
    output_count: int,
    state_count: int,
    real_axis: int | None,
    voxel_inputs: int = _TISSUE_COUNT,
    shims: int = 1,
    locations: int = 1,
) -> tuple[int, dict[torch.device, tuple[torch.Tensor, ...]], list[_Lane]]:
    """Chunk width, the events replicated per device, and the lanes.

    A kernel reading a transition table takes a voxel's slice position from its
    index within the chunk, so ``locations`` rounds the chunk to whole voxels
    and a chunk boundary never falls inside one.
    """
    train_count = _train_count(events)
    shape = (
        train_count,
        int(events[1].numel()),
        output_count,
        state_count,
        real_axis,
        shims,
    )
    chunk = min(voxels, _chunk_voxels(plan, kind, *shape))
    if locations > 1:
        chunk = max(locations, (chunk // locations) * locations)
    per_device = {device: _to_device(events, device) for device in plan.devices}
    lanes = [
        _Lane(
            device,
            chunk,
            train_count,
            output_count,
            state_count,
            real_axis,
            voxel_inputs,
            shims,
        )
        for device in plan.devices
        for _ in range(plan.lanes)
    ]
    return chunk, per_device, lanes


def _host_signal(train_count: int, voxels: int, output_count: int) -> torch.Tensor:
    shape = (
        (train_count, voxels, output_count)
        if train_count > 1
        else (voxels, output_count)
    )
    return torch.empty(shape, dtype=torch.complex64, pin_memory=True)


def _write_signal(
    signal: torch.Tensor, staged: torch.Tensor, begin: int, end: int, batched: bool
) -> None:
    destination = signal[:, begin:end] if batched else signal[begin:end]
    destination.copy_(staged, non_blocking=True)


def _run_offloaded(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    plan: _Offload,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    geometry: Geometry,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    features: frozenset[str] | None = None,
) -> torch.Tensor:
    """Stream a host-resident volume through the devices, chunk by chunk."""
    from ._epg_triton import simulate_into

    train_count = _train_count(events)
    voxels = tissue[0].numel()
    chunk, per_device, lanes = _offload_plan(
        plan,
        "forward",
        events,
        voxels,
        output_count,
        state_count,
        real_axis,
        shims=_shim_count(tissue),
        locations=1 if profile is None else profile.points,
    )
    signal = _host_signal(train_count, voxels, output_count)

    def body(lane: _Lane, begin: int, end: int) -> None:
        width = end - begin
        staged_tissue = lane.load(tissue, begin, end)
        real, imag, staged = lane.views(width)
        simulate_into(
            staged_tissue,
            per_device[lane.device],
            real,
            imag,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            atom_count=width,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            features=features,
        )
        torch.complex(real, imag, out=staged)
        _write_signal(signal, staged, begin, end, train_count > 1)

    stream_chunks(plan, voxels, chunk, lanes, body)
    return signal


def _run_offloaded_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, ...],
    plan: _Offload,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    geometry: Geometry,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    features: frozenset[str] | None = None,
) -> torch.Tensor:
    """Stream a forward-mode pass; the seeds ride along with the tissue.

    Tissue seeds are per voxel and are chunked with it. Event seeds are shared
    by every voxel, so they are replicated once per device instead.
    """
    from ._epg_triton import simulate_jvp_into

    train_count = _train_count(events)
    voxels = tissue[0].numel()
    chunk, per_device, lanes = _offload_plan(
        plan,
        "jvp",
        events,
        voxels,
        output_count,
        state_count,
        real_axis,
        voxel_inputs=2 * _TISSUE_COUNT,
        shims=_shim_count(tissue),
        locations=1 if profile is None else profile.points,
    )
    seeds = {device: _to_device(event_tangents, device) for device in plan.devices}
    signal = _host_signal(train_count, voxels, output_count)
    staged_inputs = (*tissue, *tissue_tangents)

    def body(lane: _Lane, begin: int, end: int) -> None:
        width = end - begin
        loaded = lane.load(staged_inputs, begin, end)
        real, imag, staged = lane.views(width)
        simulate_jvp_into(
            loaded[:_TISSUE_COUNT],
            per_device[lane.device],
            loaded[_TISSUE_COUNT:],
            seeds[lane.device],
            real,
            imag,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            atom_count=width,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            features=features,
        )
        torch.complex(real, imag, out=staged)
        _write_signal(signal, staged, begin, end, train_count > 1)

    stream_chunks(plan, voxels, chunk, lanes, body)
    return signal


def _run_offloaded_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    plan: _Offload,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    geometry: Geometry,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Stream a first-order adjoint through the devices, chunk by chunk.

    The gradients part the same way the forward-over-reverse pass parts them: a
    tissue gradient belongs to one voxel and goes straight back to pinned host
    memory, an event gradient collects from every voxel and each lane keeps its
    own until the end.

    Carrying no forward direction, a chunk holds half the trajectory the other
    pass holds -- so for one budget the chunks are wider, which is the second
    saving on top of the kernel being faster.
    """
    from ._epg_triton import (
        GradientBuffers,
        simulate_real_vjp_into,
        simulate_vjp_into,
    )

    train_count = _train_count(events)
    voxels = tissue[0].numel()
    event_count = int(events[1].numel())
    shape = (train_count, event_count, output_count, state_count, real_axis, 1)
    chunk = min(voxels, _chunk_voxels(plan, "first-order adjoint", *shape))
    grad_output = grad_output.resolve_conj()
    batched = train_count > 1

    rows = tissue_gradient_rows(1)
    tissue_grads = [
        torch.empty(count * voxels, dtype=torch.float32, pin_memory=True)
        for count in rows
    ]
    per_device = {device: _to_device(events, device) for device in plan.devices}
    lanes = [
        _Lane(device, chunk, train_count, output_count, state_count, real_axis)
        for device in plan.devices
        for _ in range(plan.lanes)
    ]
    for lane in lanes:
        lane.adjoint = GradientBuffers(
            per_device[lane.device],
            chunk,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
        )

    def body(lane: _Lane, begin: int, end: int) -> None:
        width = end - begin
        loaded = lane.load(tissue, begin, end)
        piece = grad_output[:, begin:end] if batched else grad_output[begin:end]
        cotangent = lane.send(piece)
        arguments = dict(
            state_count=state_count,
            output_count=output_count,
            atom_count=width,
            features=features,
        )
        if real_axis == 1:
            voxel_grads = simulate_real_vjp_into(
                loaded,
                per_device[lane.device],
                cotangent,
                lane.adjoint,
                **arguments,
            )
        else:
            voxel_grads = simulate_vjp_into(
                loaded,
                per_device[lane.device],
                cotangent,
                lane.adjoint,
                geometry=geometry,
                **arguments,
            )
        for index, count in enumerate(rows):
            home = tissue_grads[index].view(count, voxels)
            for row in range(count):
                home[row, begin:end].copy_(
                    voxel_grads[index][row * width : (row + 1) * width],
                    non_blocking=True,
                )

    stream_chunks(plan, voxels, chunk, lanes, body)
    event_grads = [
        torch.zeros_like(value) for value in (events[0], events[2], events[3])
    ]
    for lane in lanes:
        for index, value in enumerate(lane.adjoint.event_gradients()):
            event_grads[index] += value.cpu()
    return (*tissue_grads, *event_grads)


def _run_offloaded_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    plan: _Offload,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    geometry: Geometry,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    features: frozenset[str] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Stream a forward-over-reverse pass through the devices, chunk by chunk.

    The two kinds of gradient part ways here. A tissue gradient belongs to one
    voxel, so each chunk writes its slice straight back to pinned host memory.
    An event gradient collects a contribution from every voxel, so each lane
    accumulates its own on the device and they are added once at the end.

    Nothing in the loop reads a device result, which is what lets the host run
    ahead and keep the lanes fed.
    """
    from ._epg_triton import AdjointBuffers, simulate_vjp_jvp_into

    train_count = _train_count(events)
    voxels = tissue[0].numel()
    event_count = int(events[1].numel())
    shims = _shim_count(tissue)
    shape = (train_count, event_count, output_count, state_count, real_axis, shims)
    chunk = min(voxels, _chunk_voxels(plan, "adjoint", *shape))
    if profile is not None and profile.points > 1:
        # A chunk boundary must not fall inside a voxel's slice copies: the
        # kernel takes the table row from the index within the chunk.
        locations = profile.points
        chunk = max(locations, (chunk // locations) * locations)
    grad_output = grad_output.resolve_conj()
    batched = train_count > 1

    # Pinned, so a chunk's rows go back asynchronously and the loop runs on.
    # Each is shaped like the buffer it belongs to, so the transmit pair has a
    # row per shim and a chunk lands in a slice of every row.
    rows = tissue_gradient_rows(shims)
    tissue_grads = [
        [
            torch.empty(count * voxels, dtype=torch.float32, pin_memory=True)
            for count in rows
        ]
        for _ in range(2)
    ]

    per_device = {device: _to_device(events, device) for device in plan.devices}
    seeds = {
        device: _to_device(tangents[_TISSUE_COUNT:], device) for device in plan.devices
    }
    lanes = [
        _Lane(
            device,
            chunk,
            train_count,
            output_count,
            state_count,
            real_axis,
            voxel_inputs=2 * _TISSUE_COUNT,
            shims=shims,
        )
        for device in plan.devices
        for _ in range(plan.lanes)
    ]
    for lane in lanes:
        lane.adjoint = AdjointBuffers(
            per_device[lane.device],
            chunk,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            shims=shims,
            pools=_pool_kind(lineshape, exchanging),
        )
    staged_inputs = (*tissue, *tangents[:_TISSUE_COUNT])

    def body(lane: _Lane, begin: int, end: int) -> None:
        width = end - begin
        loaded = lane.load(staged_inputs, begin, end)
        piece = grad_output[:, begin:end] if batched else grad_output[begin:end]
        cotangent = lane.send(piece)
        voxel_grads = simulate_vjp_jvp_into(
            loaded[:_TISSUE_COUNT],
            per_device[lane.device],
            (*loaded[_TISSUE_COUNT:], *seeds[lane.device]),
            cotangent,
            lane.adjoint,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            atom_count=width,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            features=features,
        )
        for plane, side in enumerate(voxel_grads):
            for index, count in enumerate(rows):
                # A row at a time, so both ends of the copy are contiguous and
                # the transfer can run while the loop goes on; see ``load``.
                home = tissue_grads[plane][index].view(count, voxels)
                for row in range(count):
                    home[row, begin:end].copy_(
                        side[index][row * width : (row + 1) * width],
                        non_blocking=True,
                    )

    stream_chunks(plan, voxels, chunk, lanes, body)
    event_grads = [
        [torch.zeros_like(value) for value in (events[0], events[2], events[3])]
        for _ in range(2)
    ]
    for lane in lanes:
        for plane, side in enumerate(lane.adjoint.event_gradients()):
            for index in range(3):
                event_grads[plane][index] += side[index].cpu()
    return tuple((*tissue_grads[plane], *event_grads[plane]) for plane in range(2))


def _elsewhere(choice: _Choice, values: tuple[torch.Tensor, ...]) -> bool:
    """Whether this call has to move its inputs to honour the choice."""
    return values[0].device != _home(choice)


def _home(choice: _Choice) -> torch.device:
    return torch.device("cpu") if choice.where == "cpu" else choice.devices[0]


def _leaves_the_host(
    kind: str,
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_count: int,
    state_count: int,
    real_axis: int | None,
) -> bool:
    """Whether a policy sends a host-resident call to a device.

    Asked by a pass that reaches the devices through another one rather than
    through routes of its own, so that both read the same verdict.
    """
    if tissue[0].device.type != "cpu":
        return False
    if offloading() is not None:
        return True
    choice = _choose(kind, tissue, events, output_count, state_count, real_axis)
    if choice is None:
        return False
    return choice.where == "stream" or _elsewhere(choice, tissue)


@contextmanager
def _using(devices: tuple[torch.device, ...]) -> Iterator[None]:
    """Shard over these devices for one call, without a nested policy."""
    global _DEVICES
    previous = _DEVICES
    _DEVICES = devices
    try:
        with without_policy():
            yield
    finally:
        _DEVICES = previous


def _run_packed(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    state_count: int,
    output_count: int,
    threads: int,
    real_axis: int | None = None,
    *,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool | PoolLayout = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
    pools: PoolTables | None = None,
) -> torch.Tensor:
    """Run the forward state machine.

    ``real_axis`` selects the real-subspace kernel. Leave it ``None`` and the
    verdict is worked out here; pass a value to overrule that.

    ``profile`` is the transition table a shaped pulse turns through, if the
    sequence has one. Voxels are spread over its slice positions voxel-major,
    so the table's row count is also how many copies of each voxel the tissue
    holds.

    ``features`` names the terms the tissue gives anything to do, and the
    kernels carry those and no others. ``None`` leaves every term in.
    """
    profile = _tables(profile, events, dynamic)
    if real_axis is None:
        real_axis = _auto_real_axis(
            "forward",
            events,
            tissue,
            profile=profile,
            dynamic=dynamic,
            features=features,
        )
    # Neither second pool is inside a real subspace the reduced kernels stand
    # for: the semisolid one adds a longitudinal component, and the exchanging
    # one adds a transverse pair whose operator is complex. The verdict is
    # overruled rather than refused, the full kernel giving the same answer
    # more slowly.
    if lineshape is not None or exchanging:
        real_axis = None
    if profile is not None:
        _within_the_table(profile, events[2])
    forced = offloading()
    if forced is not None and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded(
            tissue,
            events,
            forced,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    choice = _choose("forward", tissue, events, output_count, state_count, real_axis)
    streaming = choice is not None and choice.where == "stream"
    if streaming and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded(
            tissue,
            events,
            choice.offload,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    if choice is not None and choice.where != "stream" and _elsewhere(choice, tissue):
        _carries_the_pools(pools, "moved")
        home = _home(choice)
        with _using(choice.devices):
            moved = _run_packed(
                _to_device(tissue, home),
                _to_device(events, home),
                state_count,
                output_count,
                threads,
                real_axis,
                geometry=geometry,
                profile=profile,
                lineshape=lineshape,
                exchanging=exchanging,
                dynamic=_moved_dynamic(dynamic, home),
                features=features,
            )
        return moved.to(tissue[0].device)
    if tissue[0].device.type == "cuda":
        from ._epg_triton import simulate

        shards = _shard_bounds(_train_count(events))
        if shards:
            _carries_the_pair(dynamic, "sharded")
            _carries_the_pools(pools, "sharded")
            # Every launch is asynchronous, so issuing them all before touching
            # a result lets the devices run at the same time.
            parts = [
                (
                    end - begin,
                    simulate(
                        _to_device(tissue, device),
                        _shard_events(events, begin, end, device),
                        state_count=state_count,
                        output_count=output_count,
                        real_axis=real_axis,
                        geometry=geometry,
                        profile=profile,
                        lineshape=lineshape,
                        exchanging=exchanging,
                        features=features,
                    ),
                )
                for begin, end, device in shards
            ]
            return _gather_shards(parts, tissue[0].device, output_count)
        return simulate(
            tissue,
            events,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            dynamic=dynamic,
            features=features,
            pools=pools,
        )
    from torchsim import _epg_cpu

    trains = _train_count(events)
    shape = (
        (tissue[0].numel(), output_count)
        if trains == 1
        else (trains, tissue[0].numel(), output_count)
    )
    output_real = torch.empty(shape, dtype=torch.float32)
    output_imag = torch.empty_like(output_real)
    pointers = (*tissue, *events, output_real, output_imag)
    table = None if profile is None else profile.packed()
    table_rows = None if profile is None else profile.rows()
    absorption = None if lineshape is None else lineshape.packed()
    pairs = None if dynamic is None else dynamic.packed()
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), events[1].numel())
    )
    _epg_cpu.simulate(
        _bound_pointers(
            pointers, table, table_rows, absorption, pairs, pair_rows, pools=pools
        ),
        tissue[0].numel(),
        trains,
        events[1].numel(),
        state_count,
        output_count,
        threads,
        real_axis if real_axis is not None else -1,
        geometry.flow_scale,
        geometry.washout_scale,
        _shim_count(tissue),
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        1.0 if profile is None else profile.step,
        0 if lineshape is None else lineshape.bins,
        1.0 if lineshape is None else lineshape.step,
        _pool_kind(lineshape, exchanging),
        feature_mask(features, geometry),
        *_pool_counts(pools),
    )
    return torch.complex(output_real, output_imag)


def _run_packed_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    state_count: int,
    output_count: int,
    threads: int,
    wanted: tuple[bool, ...] | None = None,
    *,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool | PoolLayout = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
    pools: PoolTables | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return gradients w.r.t. the tissue and three float event buffers.

    The result follows the differentiable-input order -- every tissue
    property, then event duration, flip and phase -- each shaped like the
    buffer it belongs to, so the transmit pair carries a row per shim.

    ``wanted`` says which of those the caller will read, which is what lets
    the real-subspace kernels -- four of them short -- be chosen. All of them,
    if it is not given.

    ``features`` names the terms the tissue gives anything to do, and the
    kernels carry those and no others. ``None`` leaves every term in.
    """
    profile = _tables(profile, events, dynamic)
    real_axis = _auto_real_axis_adjoint(
        events, tissue, (), wanted, profile, dynamic, features
    )
    # Neither second pool is inside a real subspace the reduced kernels stand
    # for; see the forward path for why.
    if lineshape is not None or exchanging:
        real_axis = None
    if profile is not None:
        _within_the_table(profile, events[2])
    # A first-order adjoint needs no dual: on a device it is its own kernel,
    # recording half the trajectory and holding half the state the
    # forward-over-reverse pass does. The real-subspace one carries a shim row
    # but no table, no pair and no pool; the complex one carries all of them.
    # A whole volume takes it here, a shard takes it a level down, and the
    # streamed route has chunked launchers of its own.
    if tissue[0].device.type == "cuda":
        shards = _shard_bounds(_train_count(events))
        if shards:
            _carries_the_pair(dynamic, "sharded")
            _carries_the_pools(pools, "sharded")
            # Each shard is a smaller adjoint of the same kind, and a device
            # holding one is a device with nothing left to split, so the piece
            # can take the first-order kernel the whole would not have.
            home = tissue[0].device
            parts = []
            for begin, end, device in shards:
                with distribute([device]):
                    parts.append(
                        _run_packed_vjp(
                            _to_device(tissue, device),
                            _shard_events(events, begin, end, device),
                            grad_output[begin:end].contiguous().to(device),
                            state_count=state_count,
                            output_count=output_count,
                            threads=threads,
                            wanted=wanted,
                            geometry=geometry,
                            profile=profile,
                            lineshape=lineshape,
                            exchanging=exchanging,
                            dynamic=dynamic,
                            features=features,
                        )
                    )
            return _combine_shards(tuple(parts), home)
    if (
        tissue[0].device.type == "cuda"
        # The real kernel carries a shim row but no table, no pair and no
        # pool; the complex one carries all of them.
        and (
            real_axis != 1
            or (
                profile is None
                and dynamic is None
                and lineshape is None
                and not exchanging
            )
        )
        and offloading() is None
        and not _leaves_the_host(
            "adjoint", tissue, events, output_count, state_count, real_axis
        )
    ):
        from ._epg_triton import simulate_real_vjp, simulate_vjp

        if real_axis == 1:
            return simulate_real_vjp(
                tissue,
                events,
                grad_output,
                state_count=state_count,
                output_count=output_count,
                features=features,
            )
        return simulate_vjp(
            tissue,
            events,
            grad_output,
            state_count=state_count,
            output_count=output_count,
            geometry=geometry,
            profile=profile,
            dynamic=dynamic,
            lineshape=lineshape,
            exchanging=exchanging,
            features=features,
            pools=pools,
        )
    streamable = (
        profile is None
        and lineshape is None
        and not exchanging
        and dynamic is None
        and pools is None
        and _shim_count(tissue) == 1
    )
    if streamable and tissue[0].device.type == "cpu":
        plan = offloading()
        if plan is None:
            choice = _choose(
                "adjoint", tissue, events, output_count, state_count, real_axis
            )
            if choice is not None and choice.where == "stream":
                plan = choice.offload
        if plan is not None:
            # A chunk of a first-order adjoint is a first-order adjoint, so the
            # piece takes the kernel the whole volume could not be given.
            return _run_offloaded_vjp(
                tissue,
                events,
                grad_output,
                plan,
                state_count,
                output_count,
                real_axis,
                geometry,
                features,
            )
    if tissue[0].device.type != "cpu" or _leaves_the_host(
        "adjoint", tissue, events, output_count, state_count, real_axis
    ):
        # An adjoint does not depend on any forward direction, so the
        # forward-over-reverse kernel given no direction to follow returns it
        # on its own -- and with it come the routes to the devices, which
        # otherwise this pass alone would not take.
        still = tuple(
            torch.zeros_like(value)
            for value in (*tissue, events[0], events[2], events[3])
        )
        _, adjoint = _run_packed_vjp_jvp(
            tissue,
            events,
            still,
            grad_output,
            state_count=state_count,
            output_count=output_count,
            threads=threads,
            real_axis=real_axis,
            wanted=wanted,
            geometry=geometry,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            dynamic=dynamic,
            features=features,
            pools=pools,
        )
        return adjoint
    from torchsim import _epg_cpu

    grad_output = grad_output.resolve_conj().contiguous()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()
    atom_grads = tuple(torch.zeros_like(value) for value in tissue)
    duration_grad = torch.zeros_like(events[0])
    flip_grad = torch.zeros_like(events[2])
    phase_grad = torch.zeros_like(events[3])
    pointers = (
        *tissue,
        *events,
        grad_real,
        grad_imag,
        *atom_grads,
        flip_grad,
        phase_grad,
        duration_grad,
    )
    table = None if profile is None else profile.packed()
    table_rows = None if profile is None else profile.rows()
    absorption = None if lineshape is None else lineshape.packed()
    # Bound to names rather than built in the call: the pointers are addresses,
    # so a buffer only the argument list holds is freed before the kernel runs.
    pairs = None if dynamic is None else dynamic.packed()
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), events[1].numel())
    )
    pair_grad = None if dynamic is None else torch.zeros_like(pairs)
    pool_grad = _pool_cotangent(pools, events)
    _epg_cpu.simulate_vjp(
        _bound_pointers(
            pointers,
            table,
            table_rows,
            absorption,
            pairs,
            pair_rows,
            None,
            pair_grad,
            pools=pools,
            pool_gradient=pool_grad,
        ),
        tissue[0].numel(),
        _train_count(events),
        events[1].numel(),
        state_count,
        output_count,
        threads,
        real_axis if real_axis is not None else -1,
        geometry.flow_scale,
        geometry.washout_scale,
        _shim_count(tissue),
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        1.0 if profile is None else profile.step,
        0 if lineshape is None else lineshape.bins,
        1.0 if lineshape is None else lineshape.step,
        _pool_kind(lineshape, exchanging),
        feature_mask(features, geometry),
        *_pool_counts(pools),
    )
    return (
        *atom_grads,
        duration_grad,
        flip_grad,
        phase_grad,
        *_extra_cotangents(pair_grad, pool_grad),
    )


def _run_packed_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    state_count: int,
    output_count: int,
    threads: int,
    real_axis: int | None = None,
    wanted: tuple[bool, ...] | None = None,
    *,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool | PoolLayout = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: PoolTables | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse pass through the JVP state machine.

    ``tangents`` follows the differentiable-input order -- every tissue
    property, then event duration, flip and phase. Returns
    ``(primal_gradients, tangent_gradients)`` in that same order.

    ``wanted`` says which of those the caller will read, which is what
    decides whether the real-subspace adjoint -- four of them short -- may be
    chosen when ``real_axis`` is left open. All of them, if it is not given.

    ``features`` names the terms the tissue gives anything to do, and the
    kernels carry those and no others. ``None`` leaves every term in.
    """
    profile = _tables(profile, events, dynamic)
    if real_axis is None:
        real_axis = _auto_real_axis_adjoint(
            events, tissue, tangents, wanted, profile, dynamic, features
        )
    # Neither second pool is inside a real subspace the reduced kernels stand
    # for; see the forward path for why.
    if lineshape is not None or exchanging:
        real_axis = None
    if profile is not None:
        _within_the_table(profile, events[2])
    forced = offloading()
    if forced is not None and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded_vjp_jvp(
            tissue,
            events,
            tangents,
            grad_output,
            forced,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    choice = _choose("adjoint", tissue, events, output_count, state_count, real_axis)
    streaming = choice is not None and choice.where == "stream"
    if streaming and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded_vjp_jvp(
            tissue,
            events,
            tangents,
            grad_output,
            choice.offload,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    if choice is not None and choice.where != "stream" and _elsewhere(choice, tissue):
        _carries_the_pools(pools, "moved")
        home = _home(choice)
        with _using(choice.devices):
            moved = _run_packed_vjp_jvp(
                _to_device(tissue, home),
                _to_device(events, home),
                _to_device(tangents, home),
                grad_output.to(home),
                state_count,
                output_count,
                threads,
                real_axis,
                geometry=geometry,
                profile=profile,
                lineshape=lineshape,
                exchanging=exchanging,
                dynamic=_moved_dynamic(dynamic, home),
                dynamic_direction=_moved_dynamic(dynamic_direction, home),
                features=features,
            )
        origin = tissue[0].device
        return tuple(tuple(value.to(origin) for value in side) for side in moved)
    if tissue[0].device.type == "cuda":
        from ._epg_triton import simulate_vjp_jvp

        shards = _shard_bounds(_train_count(events))
        if shards:
            _carries_the_pair(dynamic, "sharded")
            _carries_the_pools(pools, "sharded")
            home = tissue[0].device
            parts = [
                simulate_vjp_jvp(
                    _to_device(tissue, device),
                    _shard_events(events, begin, end, device),
                    (
                        *_to_device(tangents[:_TISSUE_COUNT], device),
                        *_shard_event_tangents(
                            tangents[_TISSUE_COUNT:], begin, end, device
                        ),
                    ),
                    grad_output[begin:end].contiguous().to(device),
                    state_count=state_count,
                    output_count=output_count,
                    real_axis=real_axis,
                    geometry=geometry,
                    profile=profile,
                    lineshape=lineshape,
                    exchanging=exchanging,
                    features=features,
                )
                for begin, end, device in shards
            ]
            sides = zip(*parts, strict=True)
            return tuple(_combine_shards(side, home) for side in sides)
        return simulate_vjp_jvp(
            tissue,
            events,
            tangents,
            grad_output,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            dynamic=dynamic,
            dynamic_direction=dynamic_direction,
            features=features,
            pools=pools,
        )
    from torchsim import _epg_cpu

    grad_output = grad_output.resolve_conj().contiguous()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()
    value_grads = tuple(torch.zeros_like(value) for value in tangents)
    tangent_grads = tuple(torch.zeros_like(value) for value in tangents)
    pointers = (
        *tissue,
        *events,
        *tangents,
        grad_real,
        grad_imag,
        *value_grads,
        *tangent_grads,
    )
    table = None if profile is None else profile.packed()
    table_rows = None if profile is None else profile.rows()
    absorption = None if lineshape is None else lineshape.packed()
    # Bound to names rather than built in the call: the pointers are addresses,
    # so a buffer only the argument list holds is freed before the kernel runs.
    pairs = None if dynamic is None else dynamic.packed()
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), events[1].numel())
    )
    # The value plane of the dual cotangent is the adjoint and the tangent
    # plane its derivative, which is the same split the tissue gradients take.
    pair_grad = None if dynamic is None else torch.zeros_like(pairs)
    pair_curve = None if dynamic is None else torch.zeros_like(pairs)
    pool_grad = _pool_cotangent(pools, events)
    pool_curve = _pool_cotangent(pools, events)
    _epg_cpu.simulate_vjp_jvp(
        _bound_pointers(
            pointers,
            table,
            table_rows,
            absorption,
            pairs,
            pair_rows,
            dynamic_direction,
            pair_grad,
            pair_curve,
            pools=pools,
            pool_gradient=pool_grad,
            pool_curvature=pool_curve,
        ),
        tissue[0].numel(),
        _train_count(events),
        events[1].numel(),
        state_count,
        output_count,
        threads,
        real_axis if real_axis is not None else -1,
        geometry.flow_scale,
        geometry.washout_scale,
        _shim_count(tissue),
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        1.0 if profile is None else profile.step,
        0 if lineshape is None else lineshape.bins,
        1.0 if lineshape is None else lineshape.step,
        _pool_kind(lineshape, exchanging),
        feature_mask(features, geometry),
        *_pool_counts(pools),
    )
    # value part -> d/d(tangent inputs); tangent part -> d/d(primal inputs)
    return (
        (*tangent_grads, *_extra_cotangents(pair_curve, pool_curve)),
        (*value_grads, *_extra_cotangents(pair_grad, pool_grad)),
    )


def _run_packed_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, ...],
    state_count: int,
    output_count: int,
    threads: int,
    real_axis: int | None = None,
    *,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool | PoolLayout = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: PoolTables | None = None,
) -> torch.Tensor:
    profile = _tables(profile, events, dynamic)
    if profile is not None:
        _within_the_table(profile, events[2])
    if real_axis is None:
        # b1_phase, b0 and RF phase are the directions that leave the subspace,
        # and the real kernels do not produce derivatives along them.
        real_axis = _auto_real_axis(
            "jvp",
            events,
            tissue,
            (tissue_tangents[4], tissue_tangents[5], event_tangents[2]),
            profile=profile,
            dynamic=dynamic,
            features=features,
        )
    # Neither second pool is inside a real subspace the reduced kernels stand
    # for; see the forward path for why.
    if lineshape is not None or exchanging:
        real_axis = None
    forced = offloading()
    if forced is not None and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            forced,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    choice = _choose("jvp", tissue, events, output_count, state_count, real_axis)
    streaming = choice is not None and choice.where == "stream"
    if streaming and tissue[0].device.type == "cpu":
        _carries_the_pair(dynamic, "streamed")
        _carries_the_pools(pools, "streamed")
        return _run_offloaded_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            choice.offload,
            state_count,
            output_count,
            real_axis,
            geometry,
            profile,
            lineshape,
            exchanging,
            features,
        )
    if choice is not None and choice.where != "stream" and _elsewhere(choice, tissue):
        _carries_the_pools(pools, "moved")
        home = _home(choice)
        with _using(choice.devices):
            moved = _run_packed_jvp(
                _to_device(tissue, home),
                _to_device(events, home),
                _to_device(tissue_tangents, home),
                _to_device(event_tangents, home),
                state_count,
                output_count,
                threads,
                real_axis,
                geometry=geometry,
                profile=profile,
                lineshape=lineshape,
                exchanging=exchanging,
                dynamic=_moved_dynamic(dynamic, home),
                dynamic_direction=_moved_dynamic(dynamic_direction, home),
                features=features,
            )
        return moved.to(tissue[0].device)
    if tissue[0].device.type == "cuda":
        from ._epg_triton import simulate_jvp

        shards = _shard_bounds(_train_count(events))
        if shards:
            _carries_the_pair(dynamic, "sharded")
            _carries_the_pools(pools, "sharded")
            parts = [
                (
                    end - begin,
                    simulate_jvp(
                        _to_device(tissue, device),
                        _shard_events(events, begin, end, device),
                        _to_device(tissue_tangents, device),
                        _shard_event_tangents(event_tangents, begin, end, device),
                        state_count=state_count,
                        output_count=output_count,
                        real_axis=real_axis,
                        geometry=geometry,
                        profile=profile,
                        lineshape=lineshape,
                        exchanging=exchanging,
                        features=features,
                    ),
                )
                for begin, end, device in shards
            ]
            return _gather_shards(parts, tissue[0].device, output_count)
        return simulate_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
            state_count=state_count,
            output_count=output_count,
            real_axis=real_axis,
            geometry=geometry,
            profile=profile,
            lineshape=lineshape,
            exchanging=exchanging,
            dynamic=dynamic,
            dynamic_direction=dynamic_direction,
            features=features,
            pools=pools,
        )
    from torchsim import _epg_cpu

    trains = _train_count(events)
    shape = (
        (tissue[0].numel(), output_count)
        if trains == 1
        else (trains, tissue[0].numel(), output_count)
    )
    output_real = torch.empty(shape, dtype=torch.float32)
    output_imag = torch.empty_like(output_real)
    pointers = (
        *tissue,
        *events,
        *tissue_tangents,
        *event_tangents,
        output_real,
        output_imag,
    )
    table = None if profile is None else profile.packed()
    table_rows = None if profile is None else profile.rows()
    absorption = None if lineshape is None else lineshape.packed()
    # Bound to names rather than built in the call: the pointers are addresses,
    # so a buffer only the argument list holds is freed before the kernel runs.
    pairs = None if dynamic is None else dynamic.packed()
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), events[1].numel())
    )
    _epg_cpu.simulate_jvp(
        _bound_pointers(
            pointers,
            table,
            table_rows,
            absorption,
            pairs,
            pair_rows,
            dynamic_direction,
            pools=pools,
        ),
        tissue[0].numel(),
        trains,
        events[1].numel(),
        state_count,
        output_count,
        threads,
        real_axis if real_axis is not None else -1,
        geometry.flow_scale,
        geometry.washout_scale,
        _shim_count(tissue),
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        1.0 if profile is None else profile.step,
        0 if lineshape is None else lineshape.bins,
        1.0 if lineshape is None else lineshape.step,
        _pool_kind(lineshape, exchanging),
        feature_mask(features, geometry),
        *_pool_counts(pools),
    )
    return torch.complex(output_real, output_imag)
