"""The rotation a shaped pulse performs, worked out ahead of time.

The state machine treats a pulse as instantaneous. A real one has a duration,
and under a slice-select gradient the spins it acts on are off-resonance by an
amount that grows with distance from the slice centre, so what it does to a
spin is a rotation about an axis that is neither transverse nor the same
everywhere. A flip angle and a phase cannot name that rotation; the
Cayley-Klein pair ``(a, b)`` can, and that pair is what the kernels turn
the states through.

Working the pair out during the simulation would mean integrating the pulse
once per voxel per event. It is tabulated here instead, and the table is small
because of what the rotation actually depends on:

* **Flip angle and transmit scaling enter only through their product.** During
  the pulse the effective field is ``(w1(t) * s, 0, gamma * G * z)`` and only
  the transverse part carries ``s``, so a nominal flip of 180 degrees at
  ``B1 = 0.7`` is the same rotation as 126 degrees at ``B1 = 1``. One axis, not
  two.
* **The RF phase factors out.** Turning the whole pulse by ``phi`` turns the
  rotation axis with it, which is ``b -> b * exp(-1j * phi)``. So the table is
  built for zero phase and the phase applied to the result, exactly as the
  instantaneous operator applies it.

That leaves a table over slice position and effective flip angle. Sampled with
its own slope and read back by cubic Hermite interpolation, 64 bins carry it to
about 1e-8, which is far below what float32 states resolve.
"""

from __future__ import annotations

__all__ = [
    "ExactSliceProfile",
    "SliceTables",
    "TransitionTable",
    "compose_spinor",
    "dynamic_pair",
    "exact_slice_profile",
    "transition_table",
]

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Any

import torch

from ._description import RfDefinition


@dataclass(frozen=True)
class TransitionTable:
    """A pulse's rotation, sampled over slice position and effective flip.

    ``a`` and ``b`` hold the Cayley-Klein pair at every knot, shaped
    ``(points, bins)``; ``slope_a`` and ``slope_b`` hold its derivative in the
    flip angle there. The flip axis runs from zero to ``theta_max`` radians in
    ``bins`` evenly spaced knots.
    """

    a: torch.Tensor
    b: torch.Tensor
    slope_a: torch.Tensor
    slope_b: torch.Tensor
    theta_max: float

    @property
    def points(self) -> int:
        """How many slice positions the table carries."""
        return int(self.a.shape[0])

    @property
    def bins(self) -> int:
        """How many flip-angle knots each position carries."""
        return int(self.a.shape[1])

    @property
    def step(self) -> float:
        """Flip angle between neighbouring knots, in radians."""
        return self.theta_max / (self.bins - 1)

    def packed(self, device: torch.device | str | None = None) -> torch.Tensor:
        """The table laid out as the kernels index it.

        ``(points, bins, 8)``: the pair then its slope, real before imaginary,
        so the two knots a Hermite read needs are sixteen contiguous floats.
        """
        return (
            torch.stack(
                (
                    self.a.real,
                    self.a.imag,
                    self.b.real,
                    self.b.imag,
                    self.slope_a.real,
                    self.slope_a.imag,
                    self.slope_b.real,
                    self.slope_b.imag,
                ),
                dim=-1,
            )
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

    def at(
        self, position: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair at a flip angle between the knots, by cubic Hermite.

        ``position`` indexes the slice axis and ``theta`` is the effective flip
        in radians, clamped to the grid. Cubic rather than linear because the
        second-order pass differentiates this twice, and because storing the
        slope makes the cubic cost the same two loads.

        Returns the pair; the caller applies the RF phase to ``b``.
        """
        step = self.theta_max / (self.bins - 1)
        scaled = (theta / step).clamp(0.0, self.bins - 1.0)
        lower = scaled.floor().clamp(max=self.bins - 2.0)
        u = scaled - lower
        index = lower.to(torch.int64)

        # Hermite basis on the unit interval.
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        def blend(values: torch.Tensor, slopes: torch.Tensor) -> torch.Tensor:
            near = values[position, index]
            far = values[position, index + 1]
            near_slope = slopes[position, index]
            far_slope = slopes[position, index + 1]
            return (
                h00 * near
                + h10 * step * near_slope
                + h01 * far
                + h11 * step * far_slope
            )

        return blend(self.a, self.slope_a), blend(self.b, self.slope_b)


def compose_spinor(
    drive: Any, turn_z: torch.Tensor | Iterable[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Cayley-Klein pair a shaped pulse leaves, sample by sample.

    One SU(2) element per raster sample, composed in the order the scanner
    plays them. Each is the rotation about the effective field a spin sees
    while that sample lasts: the RF in the transverse plane, the gradient
    along z.

    Parameters
    ----------
    drive
        The transverse field per sample, radians turned per sample. A tensor
        with the sample axis first, or any iterable of per-sample tensors --
        which is how a per-voxel field is composed without ever holding one
        for every sample at once.
    turn_z
        The longitudinal turn per sample, radians, one per spin. A tensor is
        the turn every sample makes -- a gradient the scanner holds -- and an
        iterable of per-sample tensors is one it moves, in the same form
        ``drive`` takes.

    Returns
    -------
    a, b : torch.Tensor
        Complex128, shaped like the turn. Each sample turns the magnetisation
        right-handedly about ``(drive.real, drive.imag, turn_z)`` by that
        vector's length, so a spin starting along ``+z`` ends with
        ``Mxy = -2 conj(a b)`` and ``Mz = |a|^2 - |b|^2``. A real drive of
        area ``t`` on resonance gives ``(cos(t/2), -i sin(t/2))``.

    Raises
    ------
    ValueError
        If a moving gradient runs out before the pulse does, or the other way
        about.
    """
    held = isinstance(turn_z, torch.Tensor)
    if held:
        turns: Iterator[torch.Tensor] = repeat(turn_z)
        spins = turn_z
    else:
        turns = iter(turn_z)
        spins = next(turns)
        turns = chain((spins,), turns)
    a = torch.ones_like(spins, dtype=torch.complex128)
    b = torch.zeros_like(a)
    steps = zip(drive, turns, strict=False) if held else zip(drive, turns, strict=True)
    for sample, turn in steps:
        turn_x = sample.real.expand_as(spins)
        turn_y = sample.imag.expand_as(spins)
        turn = turn.expand_as(spins)
        # Both coefficients are smooth functions of the squared angle, and are
        # taken that way near the origin: a spin at the slice centre under no
        # flip sits at exactly zero, where the square root is continuous but
        # its derivative is not, and what reads this carries derivatives.
        square = turn_x**2 + turn_y**2 + turn**2
        turning = square > 1e-18
        angle = torch.sqrt(torch.where(turning, square, torch.ones_like(square)))
        half = 0.5 * angle
        cosine = torch.where(
            turning, torch.cos(half), 1.0 - square / 8.0 + square**2 / 384.0
        )
        # sin(angle / 2) / angle, which is 1/2 at the origin.
        scale = torch.where(
            turning,
            torch.sin(half) / angle,
            0.5 - square / 48.0 + square**2 / 3840.0,
        )
        step_a = cosine - 1j * turn * scale
        step_b = -1j * (turn_x - 1j * turn_y) * scale
        a, b = step_a * a - step_b * b.conj(), step_b * a.conj() + step_a * b
    return a, b


@dataclass(frozen=True)
class DynamicPairs:
    """The rotation each pulse performs, voxel by voxel.

    ``a`` and ``b`` hold the Cayley-Klein pair shaped ``(rows, voxels)``, and
    ``index`` says which row an event reads -- packed alongside the events, in
    their order, exactly as the shim row and the profile row are. Events that
    drive no pulse carry any row; nothing reads it.

    A row is one pulse rather than one shape. A static array's rotation depends
    on the array through a single complex scalar, which is what lets one table
    over effective flip cover every pulse sharing a shape; this one depends on
    the whole per-channel vector as it varies, so pulses differing in anything
    at all share nothing.

    The pair is built at zero RF phase, so the event's own phase turns the axis
    afterwards -- ``b -> b exp(-i phi)`` -- which is what lets phase cycling
    reuse a row rather than re-integrate it.
    """

    a: torch.Tensor
    b: torch.Tensor
    index: torch.Tensor

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape:
            raise ValueError(
                f"the pair's halves are shaped {tuple(self.a.shape)} and "
                f"{tuple(self.b.shape)}"
            )
        if self.a.ndim != 2:
            raise ValueError(
                f"a dynamic pair is one row per pulse of one entry per voxel, "
                f"so it is two-dimensional, not {self.a.ndim}"
            )
        if self.index.ndim not in (1, 2):
            raise ValueError(
                f"the index names a row per event, for one train or for each, "
                f"so it is one- or two-dimensional, not {self.index.ndim}"
            )
        if self.index.numel() and (
            int(self.index.min()) < 0 or int(self.index.max()) >= self.rows
        ):
            raise ValueError(
                f"the index reaches row {int(self.index.max())} of {self.rows}"
            )

    @property
    def rows(self) -> int:
        """How many pulses the set carries a rotation for."""
        return int(self.a.shape[0])

    @property
    def voxels(self) -> int:
        """How many voxels each row covers."""
        return int(self.a.shape[1])

    def at(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair one pulse performs, one entry per voxel."""
        return self.a[row], self.b[row]

    def rows_per_event(self, train_count: int, event_count: int) -> torch.Tensor:
        """The row each event of each train reads, as the kernels index it.

        The kernels read this at ``train * event_count + event``, so it has to
        span every train -- a shorter buffer is read past its end rather than
        wrapped. An index given for one train is taken to mean that every train
        reads the same rows, which is what a set of pulses shared across a
        batch is; one given per train is flattened in place.

        Raises
        ------
            ValueError: if the index fits neither shape.
        """
        flat = self.index.reshape(-1).to(torch.int32)
        if flat.numel() == train_count * event_count:
            return flat.contiguous()
        if flat.numel() == event_count:
            return flat.repeat(train_count).contiguous()
        raise ValueError(
            f"the index names {flat.numel()} rows, but this launch reads one "
            f"per event of every train -- {event_count} or "
            f"{train_count * event_count}"
        )

    @classmethod
    def from_packed(cls, values: torch.Tensor, index: torch.Tensor) -> DynamicPairs:
        """The set a packed buffer describes, ``(rows, voxels, 4)``.

        The inverse of :meth:`packed`, which is what lets the pair cross a
        kernel boundary as one differentiable tensor and come back as a pair.
        """
        return cls(
            a=torch.complex(values[..., 0], values[..., 1]),
            b=torch.complex(values[..., 2], values[..., 3]),
            index=index,
        )

    def packed(self, device: torch.device | str | None = None) -> torch.Tensor:
        """The pair laid out as the kernels index it.

        ``(rows, voxels, 4)``: the pair, real before imaginary, so what a
        single read needs is four contiguous floats.
        """
        return (
            torch.stack((self.a.real, self.a.imag, self.b.real, self.b.imag), dim=-1)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )


def _sample_steps(
    definition: RfDefinition,
    rf_raster_time_s: float,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """How long each sample is held, and what a spin off centre accrues in it.

    Returns ``(dwell, steps)`` in seconds: ``dwell`` is what the pulse's own
    sample times say each sample lasts, and ``steps`` is that under the
    gradient played across it, so a gradient the scanner switches off
    contributes no position-dependent turn however long the sample.
    """
    dwell = torch.as_tensor(
        definition.sample_durations(rf_raster_time_s=rf_raster_time_s),
        dtype=torch.float64,
        device=device,
    )
    waveform = definition.gradient_waveform()
    if waveform is None:
        return dwell, dwell
    moving = torch.as_tensor(waveform, dtype=torch.float64, device=device)
    return dwell, dwell * moving


def _longitudinal(
    detune: torch.Tensor,
    rate: torch.Tensor | None,
    steps: torch.Tensor,
) -> torch.Tensor | Any:
    """The longitudinal turn per sample: off-resonance, and where the spin sits.

    ``rate`` is what a spin at its position turns per second of pulse. Where
    every sample accrues alike the two add once and the result is handed over
    as one tensor; otherwise it is an iterable, which is the second form
    :func:`compose_spinor` takes.
    """
    if rate is None:
        return detune
    if bool(torch.all(steps == steps[0])):
        return detune + rate * steps[0]
    return (detune + rate * step for step in steps)


def dynamic_pair(
    definition: RfDefinition,
    sensitivities: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    flip: torch.Tensor | float = 1.0,
    off_resonance_hz: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    rf_raster_time_s: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rotation a dynamically shimmed pulse performs, voxel by voxel.

    A static array reduces to one flip angle and one phase because its channel
    weights are constant while the pulse plays, so the spatial factor comes
    outside the pulse integral. Weights that vary during the pulse -- kT-points,
    spokes -- do not factor that way: the field a voxel sees changes direction
    as well as size, and what the pulse leaves is a rotation about an axis the
    voxel has to itself. That rotation is this pair.

    There is no table to build. A static pulse's rotation depends on the array
    only through one complex scalar, which is why one grid covers every voxel;
    here it depends on the whole per-channel vector, so the integral is taken
    per voxel and the field is formed one sample at a time rather than held for
    every sample at once.

    Parameters
    ----------
    definition
        The pulse. Its complex envelope is played sample by sample at
        ``rf_raster_time_s`` and normalized by its own integral, so a voxel the
        array puts unit field on turns through exactly ``flip``. A per-channel
        envelope is the drive itself; a single one is the waveform every
        channel plays, and ``weights`` says how hard each drives it.
    sensitivities
        Complex per-channel transmit sensitivities, ``(voxels, channels)``.
    weights
        Complex channel weights, ``(channels, samples)`` while the pulse plays,
        or ``(channels,)`` for a weight they hold. Unity when not given.
    flip
        The nominal flip the event drives, in radians. Scales every channel
        alike and so still factors out of the array.
    off_resonance_hz
        What each voxel is off resonance by while the pulse plays. Zero when
        not given.
    positions
        Where across the slice to integrate, in units of the slice thickness,
        so that the passband is ``[-0.5, 0.5]``. The slice-select gradient is
        taken to hold for the whole pulse, which makes position times the
        definition's bandwidth the offset in Hz it adds. One location when not
        given.

    Returns
    -------
        ``(a, b)`` in ``complex64``, one entry per voxel and location with
        **locations fastest** -- ``voxel * locations + location``, which is how
        the kernels index an atom across a slice.

    Raises
    ------
        ValueError: if the pulse has no samples, if it integrates to nothing,
            or if the envelope, the weights and the sensitivities disagree on
            the channel count.
    """
    device = sensitivities.device
    envelope = torch.as_tensor(
        definition.complex_envelope(), dtype=torch.complex128, device=device
    )
    if envelope.numel() == 0:
        raise ValueError(f"RF definition {definition.id} has an empty envelope")
    if envelope.ndim == 1:
        envelope = envelope[None]
    dwell, steps = _sample_steps(definition, rf_raster_time_s, device)
    envelope = envelope * dwell
    area = envelope.sum()
    if area.abs() <= 1e-6 * envelope.abs().sum():
        raise ValueError(
            f"RF definition {definition.id} integrates to nothing, so it has no "
            f"flip angle to drive"
        )
    shape = envelope / area
    samples = int(shape.shape[1])

    sensitivities = torch.as_tensor(sensitivities).to(torch.complex128)
    if weights is None:
        weights = torch.ones(shape.shape[0], 1, dtype=torch.complex128, device=device)
    else:
        weights = torch.as_tensor(weights).to(torch.complex128)
        if weights.ndim == 1:
            weights = weights[:, None]
    if shape.shape[0] == 1 and weights.shape[0] != 1:
        # One waveform every channel plays: the weights are what makes the
        # channels differ, which is the static array this generalizes.
        shape = shape.expand(weights.shape[0], -1)
    channels = int(shape.shape[0])
    if weights.shape[0] != channels:
        raise ValueError(
            f"the pulse drives {channels} channels and the weights {weights.shape[0]}"
        )
    if sensitivities.shape[-1] != channels:
        raise ValueError(
            f"the weights drive {channels} channels and the "
            f"sensitivities map {sensitivities.shape[-1]}"
        )
    if weights.shape[1] not in (1, samples):
        raise ValueError(
            f"the weights carry {weights.shape[1]} samples and the pulse {samples}"
        )

    voxels = int(sensitivities.shape[0])
    if positions is None:
        locations = 1
        across = None
    else:
        across = torch.as_tensor(positions, dtype=torch.float64, device=device).reshape(
            -1
        )
        locations = int(across.numel())
        across = 2.0 * torch.pi * float(definition.bandwidth_hz) * across

    detune = torch.zeros(voxels * locations, dtype=torch.float64, device=device)
    if off_resonance_hz is not None:
        detune = detune + (
            2.0
            * torch.pi
            * torch.as_tensor(off_resonance_hz).to(torch.float64)
            * rf_raster_time_s
        ).expand(voxels).repeat_interleave(locations)
    placed = None if across is None else across.repeat(voxels)
    if locations > 1:
        # A voxel's transmit does not depend on where in the slice it sits.
        sensitivities = sensitivities.repeat_interleave(locations, dim=0)

    driven = torch.as_tensor(flip, dtype=torch.float64, device=device)

    def field() -> Any:
        held = weights.shape[1] == 1
        for sample in range(samples):
            column = shape[:, sample] * weights[:, 0 if held else sample]
            yield driven * (sensitivities @ column)

    a, b = compose_spinor(field(), _longitudinal(detune, placed, steps))
    if placed is not None:
        # Referenced to the middle of the pulse, as a table is and for the same
        # reason: the event this stands in for is instantaneous, so the turn
        # the slice-select gradient leaves belongs either side of it.
        a = a * torch.exp(0.5j * (placed * steps.sum()).to(torch.complex128))
    return a.to(torch.complex64), b.to(torch.complex64)


def transition_table(
    definition: RfDefinition,
    positions: torch.Tensor,
    *,
    bins: int = 64,
    theta_max: float = 2.0 * torch.pi,
    rf_raster_time_s: float = 1e-6,
    device: torch.device | str | None = None,
) -> TransitionTable:
    """Integrate a pulse into the rotation it performs, over a grid.

    Parameters
    ----------
    definition
        The pulse. Its complex envelope is played sample by sample at
        ``rf_raster_time_s``, and its ``bandwidth_hz`` sets how fast
        off-resonance grows away from the slice centre.
    positions
        Where across the slice to sample, in units of the slice thickness, so
        that the passband is ``[-0.5, 0.5]``.
    bins
        Knots along the flip axis. 64 carries a sinc to about 1e-8.
    theta_max
        Largest effective flip the table covers, in radians. A pulse driven
        past this reads the last knot.

    Returns
    -------
        The table, with slopes taken by forward-mode differentiation of the
        same integration rather than by differencing it.

    Raises
    ------
        ValueError: if the pulse has no samples, or ``bins`` is below two.
    """
    if bins < 2:
        raise ValueError(f"a Hermite table needs at least two knots, got {bins}")

    # Integrated in double and stored in single: the product runs once per
    # raster sample, so rounding accumulates over thousands of steps to reach a
    # table everything downstream is measured against.
    envelope = torch.as_tensor(
        definition.complex_envelope(), dtype=torch.complex128, device=device
    )
    if envelope.numel() == 0:
        raise ValueError(f"RF definition {definition.id} has an empty envelope")
    # Divided by its own integral, which is how RfDefinition.flip_angle reads a
    # pulse: the magnitude of the integral sets the flip and its angle sets the
    # axis. Normalizing by the complex sum puts both right -- an on-resonance
    # spin turns through exactly theta, about the axis at zero -- so the grid
    # axis is the same flip angle the packed events carry.
    dwell, steps = _sample_steps(definition, rf_raster_time_s, device)
    envelope = envelope * dwell
    area = envelope.sum()
    if area.abs() <= 1e-6 * envelope.abs().sum():
        raise ValueError(
            f"RF definition {definition.id} integrates to nothing, so it has no "
            f"flip angle to tabulate against"
        )
    weight = envelope / area

    positions = torch.as_tensor(positions, dtype=torch.float64, device=device)
    theta = torch.linspace(0.0, theta_max, bins, dtype=torch.float64, device=device)

    # What a spin at each position turns per second of pulse. The pulse selects
    # one slice thickness across its bandwidth, so position measured in
    # thicknesses times bandwidth is the offset in Hz.
    rate = 2.0 * torch.pi * float(definition.bandwidth_hz) * positions

    # The turn that gradient leaves a spin holding over the whole pulse.
    carried = (rate * steps.sum()).to(torch.complex128)

    def integrate(flip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair the whole pulse leaves, at every (position, flip)."""
        wide = (positions.numel(), flip.numel())
        turn_z = _longitudinal(
            torch.zeros((), dtype=torch.float64, device=device),
            rate[:, None].expand(wide),
            steps,
        )
        if not isinstance(turn_z, torch.Tensor):
            turn_z = (turn.expand(wide) for turn in turn_z)
        a, b = compose_spinor(weight[:, None, None] * flip, turn_z)
        # Referenced to the middle of the pulse, because that is where the
        # instantaneous event the table stands in for sits. What a spin accrues
        # under the slice-select gradient either side of that instant belongs
        # to the gradient -- and is what the rephaser rewinds -- so a rotation
        # referenced to the end of the pulse carries a phase that grows across
        # the slice, and averaging over the slice then cancels the signal
        # rather than shaping it.
        return a * torch.exp(0.5j * carried)[:, None].expand(wide), b

    values, slopes = torch.func.jvp(integrate, (theta,), (torch.ones_like(theta),))
    return TransitionTable(
        a=values[0].to(torch.complex64).contiguous(),
        b=values[1].to(torch.complex64).contiguous(),
        slope_a=slopes[0].to(torch.complex64).contiguous(),
        slope_b=slopes[1].to(torch.complex64).contiguous(),
        theta_max=float(theta_max),
    )


@dataclass(frozen=True)
class ExactSliceProfile:
    """Where across the slice a sequence's pulse is to be integrated.

    Passed as ``slice_profile=``. The table is built from the sequence's own RF
    definition when the simulation runs, because that is where the pulse and
    its raster are known -- a caller names positions, never a response.

    ``points`` are the slice positions sampled, in units of the slice
    thickness, so the passband is ``[-0.5, 0.5]``. An integer asks for that
    many evenly spaced across ``extent`` thicknesses; a tensor names them.

    Attributes
    ----------
    points : int or torch.Tensor
        Slice positions, or how many to space evenly across ``extent``.
    extent : float
        How many slice thicknesses the sampled positions span.
    bins : int
        Knots along the flip-angle axis. 64 carries a sinc to about 1e-8.
    theta_max : float
        Largest effective flip the table covers, in radians. A sequence that
        drives a pulse past this is refused rather than saturated.
    """

    points: int | torch.Tensor = 21
    extent: float = 2.0
    bins: int = 64
    theta_max: float = 2.0 * torch.pi

    def positions(self) -> torch.Tensor:
        """Where across the slice the table is sampled."""
        if isinstance(self.points, int):
            if self.points < 1:
                raise ValueError(
                    f"a slice needs at least one position, got {self.points}"
                )
            if self.points == 1:
                return torch.zeros(1, dtype=torch.float64)
            half = 0.5 * self.extent
            return torch.linspace(-half, half, self.points, dtype=torch.float64)
        return torch.as_tensor(self.points, dtype=torch.float64).reshape(-1)


def exact_slice_profile(
    points: int | torch.Tensor = 21,
    *,
    extent: float = 2.0,
    bins: int = 64,
    theta_max: float = 2.0 * torch.pi,
) -> ExactSliceProfile:
    """Ask for a pulse to be integrated across the slice.

    A slice profile is a Bloch response, so it is not proportional to the pulse
    driving it and cannot be carried as a scaling on the flip angle. This names
    the positions at which the rotation the pulse actually performs is worked
    out, which is right at any transmit scaling.

    A definition carrying a waveform worth integrating is tabulated whether or
    not this is passed; what this adds is where across the slice to sample it,
    and it puts a pulse that would otherwise be played hard on the same route.

    Parameters
    ----------
    points : int or torch.Tensor, optional
        Slice positions, or how many to space evenly across ``extent``.
    extent : float, optional
        How many slice thicknesses the sampled positions span.
    bins : int, optional
        Knots along the flip-angle axis. 64 carries a sinc to about 1e-8.
    theta_max : float, optional
        Largest effective flip the table covers, in radians. A sequence that
        drives a pulse past this is refused rather than saturated.

    Returns
    -------
    ExactSliceProfile
        The request, resolved against the sequence's RF definition at
        simulation time.
    """
    return ExactSliceProfile(
        points=points, extent=extent, bins=bins, theta_max=theta_max
    )


def across_the_slice(asked: Any) -> ExactSliceProfile | None:
    """Where across the slice to work a shaped pulse out, from what was asked.

    How many positions to sample is the common way to ask and is spelled as
    one; :func:`exact_slice_profile` is there for the rest -- which positions,
    how wide, how finely to tabulate -- and passes through.

    Nothing else is taken. A response is not proportional to the pulse driving
    it, so an array of flip-angle scalings is not a slice profile, and it is
    close enough to a list of positions that guessing between them would
    sometimes be wrong in silence.

    Raises
    ------
        TypeError: if ``asked`` is neither a count nor a profile.
    """
    if asked is None or isinstance(asked, ExactSliceProfile):
        return asked
    if isinstance(asked, bool) or not isinstance(asked, int):
        raise TypeError(
            f"across_slice says where across the slice to integrate the "
            f"sequence's own pulse: how many positions, or "
            f"exact_slice_profile() to say which. Got {type(asked).__name__}. "
            f"An array of flip-angle scalings treats a Bloch response as "
            f"proportional to the pulse driving it, which it is not; give the "
            f"RF definition its waveform instead"
        )
    return exact_slice_profile(asked)


@dataclass(frozen=True)
class SliceTables:
    """The tables a sequence reads, and which pulse reads which.

    A table is indexed by slice position and effective flip angle, so one
    covers every pulse sharing a shape however they differ in flip. Pulses of
    different shapes need one each, and ``index`` says which an event reads --
    packed alongside the events, in their order, exactly as the shim row is.

    Every table spans the same positions and the same flip grid, because they
    are built together from one request.
    """

    tables: tuple[TransitionTable, ...]
    index: torch.Tensor

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValueError("a slice table set holds at least one table")
        first = self.tables[0]
        for table in self.tables[1:]:
            if (
                table.points != first.points
                or table.bins != first.bins
                or table.theta_max != first.theta_max
            ):
                raise ValueError(
                    "slice tables are stacked into one buffer, so they must "
                    "span the same positions and the same flip grid"
                )

    @classmethod
    def alone(
        cls, table: TransitionTable, events: int, device: Any = None
    ) -> SliceTables:
        """One table every pulse reads, which is the single-shape case."""
        return cls(
            tables=(table,),
            index=torch.zeros(events, dtype=torch.int32, device=device),
        )

    @property
    def shapes(self) -> int:
        """How many distinct pulse shapes the set holds."""
        return len(self.tables)

    @property
    def points(self) -> int:
        """How many slice positions each table carries."""
        return self.tables[0].points

    @property
    def bins(self) -> int:
        """How many flip-angle knots each position carries."""
        return self.tables[0].bins

    @property
    def step(self) -> float:
        """Flip angle between neighbouring knots, in radians."""
        return self.tables[0].step

    @property
    def theta_max(self) -> float:
        """Largest effective flip the tables cover, in radians."""
        return self.tables[0].theta_max

    def packed(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Every table stacked, laid out as the kernels index them.

        ``(shapes * points, bins, 8)``, so a pulse's row is its shape times the
        position count plus the voxel's position.
        """
        return torch.cat(
            [table.packed(device) for table in self.tables], dim=0
        ).contiguous()

    def rows(self, device: torch.device | str | None = None) -> torch.Tensor:
        """The per-event table index, on the device the kernels run on."""
        return self.index.to(device=device, dtype=torch.int32).contiguous()
