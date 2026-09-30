"""Triton kernels for a tissue whose pools are tabulated per interval.

The device side of ``simulate_pooled`` and ``simulate_pooled_adjoint`` in
``_epg_cpu.cpp``, reading the tables ``torchsim.sequence._pools`` builds. One
program carries one (train, voxel) problem, and its states are tiles of pools
by dephasing orders, so an interval's relaxation and exchange is a product of
the tabulated operator with the tile and a pulse turns every exchanging pool at
once.

Each body is written once over dual numbers. A complex dual is the tuple
``(real, imag, tangent real, tangent imag)`` and a real one ``(value,
tangent)``; with ``following`` off the helpers leave the tangents at zero and
compute none of them, so the pass that follows no direction pays for none.
"""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import torch
import triton
import triton.language as tl

from ._accelerators import _shim_count, _train_count
from ._epg_triton import (
    _TRAJECTORY_BUDGET_BYTES,
    _atom_stride,
    _dynamic_pair_at,
    _dynamic_pair_dual_at,
    _lineshape_at_curve,
    _lineshape_at_slope,
    _output_shape,
    _profile_pair,
    _profile_pair_slope,
    _profiled_pair_dual,
    _rotate_spinor,
    _rotate_spinor_dual,
    _shift,
    _shift_adjoint,
    _spinor_adjoint,
    _spinor_adjoint_dual,
    _table_row,
)
from ._parameters import (
    FLOAT_NAMES,
    NO_GEOMETRY,
    TISSUE_NAMES,
    Geometry,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import feature_flags as _feature_flags

# The seven per-voxel properties a tabulated tissue still reads, in packing
# order. Relaxation, exchange and the pools' shares are inside the tables.
_VOXEL = (
    "m0",
    "b1",
    "b1_phase_rad",
    "b0_hz",
    "inversion_efficiency",
    "diffusion_um2_per_ms",
    "velocity_m_per_s",
)
_VOXEL_INDEX = tuple(TISSUE_NAMES.index(name) for name in _VOXEL)


# ---------------------------------------------------------------------------
# Dual arithmetic.
# ---------------------------------------------------------------------------


@triton.jit
def _rmul(x, y, following: tl.constexpr):
    """Two real duals multiplied."""
    if following:
        return (x[0] * y[0], x[1] * y[0] + x[0] * y[1])
    else:
        return (x[0] * y[0], 0.0)


@triton.jit
def _cmul(x, y, following: tl.constexpr):
    """Two complex duals multiplied."""
    real = x[0] * y[0] - x[1] * y[1]
    imag = x[0] * y[1] + x[1] * y[0]
    if following:
        return (
            real,
            imag,
            x[2] * y[0] - x[3] * y[1] + x[0] * y[2] - x[1] * y[3],
            x[2] * y[1] + x[3] * y[0] + x[0] * y[3] + x[1] * y[2],
        )
    else:
        return (real, imag, 0.0, 0.0)


@triton.jit
def _cscale(r, x, following: tl.constexpr):
    """A real dual times a complex one."""
    if following:
        return (
            r[0] * x[0],
            r[0] * x[1],
            r[1] * x[0] + r[0] * x[2],
            r[1] * x[1] + r[0] * x[3],
        )
    else:
        return (r[0] * x[0], r[0] * x[1], 0.0, 0.0)


@triton.jit
def _conj(x):
    return (x[0], -x[1], x[2], -x[3])


@triton.jit
def _cadd(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])


@triton.jit
def _polar(angle, following: tl.constexpr):
    """``exp(i angle)`` for a real dual angle."""
    cosine = tl.cos(angle[0])
    sine = tl.sin(angle[0])
    if following:
        return (cosine, sine, -sine * angle[1], cosine * angle[1])
    else:
        return (cosine, sine, 0.0, 0.0)


@triton.jit
def _re_dot(x, y, following: tl.constexpr):
    """``Re(conj(x) y)`` as a real dual."""
    value = x[0] * y[0] + x[1] * y[1]
    if following:
        return (value, x[2] * y[0] + x[3] * y[1] + x[0] * y[2] + x[1] * y[3])
    else:
        return (value, 0.0)


@triton.jit
def _total(value, mask):
    """The sum of a tile over the entries ``mask`` keeps."""
    return tl.sum(tl.sum(tl.where(mask, value, 0.0), axis=1), axis=0)


@triton.jit
def _rtotal(x, mask, following: tl.constexpr):
    """A real dual tile summed to a real dual scalar."""
    if following:
        return (_total(x[0], mask), _total(x[1], mask))
    else:
        return (_total(x[0], mask), 0.0)


# ---------------------------------------------------------------------------
# Operators on pool tiles.
# ---------------------------------------------------------------------------


@triton.jit
def _times(operator, planes, transposed: tl.constexpr):
    """``operator @ planes`` over the pools, or its transpose's."""
    if transposed:
        return tl.sum(operator[:, :, None] * planes[:, None, :], axis=0)
    else:
        return tl.sum(operator[:, :, None] * planes[None, :, :], axis=1)


@triton.jit
def _outer(left, right):
    """``sum_k left[i, k] right[j, k]``: the operator a pair of tiles makes."""
    return tl.sum(left[:, None, :] * right[None, :, :], axis=2)


@triton.jit
def _apply(operator, planes, conjugate: tl.constexpr, transposed: tl.constexpr,
           following: tl.constexpr):  # fmt: skip
    """A complex dual operator applied to complex dual pool tiles."""
    er = operator[0]
    ei = operator[1]
    if conjugate:
        ei = -ei
    real = _times(er, planes[0], transposed) - _times(ei, planes[1], transposed)
    imag = _times(er, planes[1], transposed) + _times(ei, planes[0], transposed)
    if following:
        etr = operator[2]
        eti = operator[3]
        if conjugate:
            eti = -eti
        tangent_real = (
            _times(etr, planes[0], transposed)
            - _times(eti, planes[1], transposed)
            + _times(er, planes[2], transposed)
            - _times(ei, planes[3], transposed)
        )
        tangent_imag = (
            _times(etr, planes[1], transposed)
            + _times(eti, planes[0], transposed)
            + _times(er, planes[3], transposed)
            + _times(ei, planes[2], transposed)
        )
        return (real, imag, tangent_real, tangent_imag)
    else:
        return (real, imag, 0.0, 0.0)


@triton.jit
def _apply_real(operator, planes, transposed: tl.constexpr, following: tl.constexpr):
    """A real dual operator applied to complex dual pool tiles."""
    real = _times(operator[0], planes[0], transposed)
    imag = _times(operator[0], planes[1], transposed)
    if following:
        return (
            real,
            imag,
            _times(operator[1], planes[0], transposed)
            + _times(operator[0], planes[2], transposed),
            _times(operator[1], planes[1], transposed)
            + _times(operator[0], planes[3], transposed),
        )
    else:
        return (real, imag, 0.0, 0.0)


@triton.jit
def _couter(left, right, following: tl.constexpr):
    """``sum_k left[i, k] right[j, k]`` of two complex dual tiles."""
    real = _outer(left[0], right[0]) - _outer(left[1], right[1])
    imag = _outer(left[0], right[1]) + _outer(left[1], right[0])
    if following:
        return (
            real,
            imag,
            _outer(left[2], right[0])
            - _outer(left[3], right[1])
            + _outer(left[0], right[2])
            - _outer(left[1], right[3]),
            _outer(left[2], right[1])
            + _outer(left[3], right[0])
            + _outer(left[0], right[3])
            + _outer(left[1], right[2]),
        )
    else:
        return (real, imag, 0.0, 0.0)


@triton.jit
def _read(values, directions, at, live: tl.constexpr, identity,
          following: tl.constexpr):  # fmt: skip
    """A buffer entry and the direction along it, or its identity when absent."""
    if live:
        value = tl.load(values + at)
        if following:
            return (value, tl.load(directions + at))
        else:
            return (value, 0.0)
    else:
        return (identity, 0.0)


@triton.jit
def _entries(slot, directions, offset, mask, along, sloped_at,
             directed: tl.constexpr, sloped: tl.constexpr,
             following: tl.constexpr):  # fmt: skip
    """Table entries at ``offset``, moving with the table's direction and slope.

    An event reads the row of its own interval length; a pass following a
    direction in that length moves every entry along the row's slope, which
    sits ``sloped_at`` further into the table.
    """
    value = tl.load(slot + offset, mask=mask, other=0.0)
    if following:
        tangent = value * 0.0
        if directed:
            tangent += tl.load(directions + offset, mask=mask, other=0.0)
        if sloped:
            tangent += tl.load(slot + sloped_at + offset, mask=mask, other=0.0) * along
        return (value, tangent)
    else:
        return (value, 0.0)


@triton.jit
def _operators(slot, directions, row_offset, along, slope_offset, pool, column,
               n: tl.constexpr, m: tl.constexpr, directed: tl.constexpr,
               sloped: tl.constexpr, following: tl.constexpr):  # fmt: skip
    """One interval's longitudinal operator, its recovery and transverse operator."""
    longitudinal = _entries(
        slot,
        directions,
        row_offset + pool * n + column,
        (pool < n) & (column < n),
        along,
        slope_offset,
        directed,
        sloped,
        following,
    )
    restored = _entries(
        slot,
        directions,
        row_offset + n * n + pool,
        pool < n,
        along,
        slope_offset,
        directed,
        sloped,
        following,
    )
    across = row_offset + n * n + n + 2 * (pool * m + column)
    carried = (pool < m) & (column < m)
    real = _entries(
        slot, directions, across, carried, along, slope_offset, directed, sloped,
        following,
    )  # fmt: skip
    imag = _entries(
        slot, directions, across + 1, carried, along, slope_offset, directed, sloped,
        following,
    )  # fmt: skip
    return longitudinal, restored, (real[0], imag[0], real[1], imag[1])


@triton.jit
def _factors(dt, damping_rate, b0, flow_rate, washout_rate, order,
             off_axis: tl.constexpr, moving: tl.constexpr,
             diffusing: tl.constexpr, following: tl.constexpr):  # fmt: skip
    """What an interval does to every pool alike, per dephasing order.

    Returns the fraction washout leaves, the transverse and longitudinal
    factors before washout, and the per-order damping weights.
    """
    squared = order * order
    transverse_weight = squared + order + 0.3333333333333333
    damp_z = (order * 0.0 + 1.0, 0.0)
    damp_t = (order * 0.0 + 1.0, 0.0)
    if diffusing:
        b_factor = _rmul(damping_rate, dt, following)
        z = tl.exp(-squared * b_factor[0])
        t = tl.exp(-transverse_weight * b_factor[0])
        if following:
            damp_z = (z, z * (-squared * b_factor[1]))
            damp_t = (t, t * (-transverse_weight * b_factor[1]))
        else:
            damp_z = (z, 0.0)
            damp_t = (t, 0.0)
    wout = (1.0, 0.0)
    if moving:
        fraction = washout_rate[0] * dt[0]
        left = 1.0 - tl.minimum(fraction, 1.0)
        if following:
            wout = (
                left,
                tl.where(
                    fraction < 1.0, -(washout_rate[1] * dt[0] + washout_rate[0] * dt[1]),
                    0.0,
                ),
            )  # fmt: skip
        else:
            wout = (left, 0.0)
    unit_t = (damp_t[0], damp_t[0] * 0.0, damp_t[1], 0.0)
    unit_z = (damp_z[0], damp_z[0] * 0.0, damp_z[1], 0.0)
    if off_axis or moving:
        angle = _rmul((-6.283185307179586, 0.0), _rmul(b0, dt, following), following)
        turn = (0.0, 0.0)
        if moving:
            turn = _rmul(flow_rate, dt, following)
        half = -(order + 0.5)
        theta = (angle[0] + half * turn[0], angle[1] + half * turn[1])
        unit_t = _cscale(damp_t, _polar(theta, following), following)
        if moving:
            unit_z = _cscale(
                damp_z, _polar((-order * turn[0], -order * turn[1]), following),
                following,
            )  # fmt: skip
    if following:
        # Every tangent a tile, so the operator products can take them.
        unit_t = (
            unit_t[0],
            unit_t[1],
            unit_t[2] + order * 0.0,
            unit_t[3] + order * 0.0,
        )
        unit_z = (
            unit_z[0],
            unit_z[1],
            unit_z[2] + order * 0.0,
            unit_z[3] + order * 0.0,
        )
    return wout, unit_t, unit_z, squared, transverse_weight


@triton.jit
def _relax(plus, minus, longitudinal, transverse_op, longitudinal_op, restored,
           equilibrium, wout, carried, spin, state, following: tl.constexpr):  # fmt: skip
    """One interval's relaxation and exchange over every order.

    Returns the three states it leaves and the operator products before the
    per-order factors, which the adjoint reuses.
    """
    mixed_plus = _apply(transverse_op, plus, False, False, following)
    mixed_minus = _apply(transverse_op, minus, True, False, following)
    mixed_z = _apply_real(longitudinal_op, longitudinal, False, following)
    out_plus = _cmul(carried, mixed_plus, following)
    out_minus = _cmul(_conj(carried), mixed_minus, following)
    out_z = _cmul(spin, mixed_z, following)
    # Inflowing spins arrive at equilibrium, so washout scales what the pools
    # held and not what they recover towards.
    origin = state == 0
    grown = equilibrium[0] - wout[0] * restored[0]
    if following:
        grown_tangent = equilibrium[1] - (wout[1] * restored[0] + wout[0] * restored[1])
        out_z = (
            out_z[0] + tl.where(origin, grown, 0.0),
            out_z[1],
            out_z[2] + tl.where(origin, grown_tangent, 0.0),
            out_z[3],
        )
    else:
        out_z = (out_z[0] + tl.where(origin, grown, 0.0), out_z[1], 0.0, 0.0)
    return out_plus, out_minus, out_z, mixed_plus, mixed_minus, mixed_z


@triton.jit
def _hard_pair(alpha, phi, following: tl.constexpr):
    """A hard pulse as its Cayley-Klein pair, with the pair's slope in the flip.

    ``a = cos(alpha / 2)`` and ``b = -i sin(alpha / 2) exp(-i phi)``, the
    rotation ``_rotate_flip_phase`` performs.
    """
    half = 0.5 * alpha[0]
    cosine = tl.cos(half)
    sine = tl.sin(half)
    nothing = cosine * 0.0
    turn = _polar((-phi[0], -phi[1]), following)
    if following:
        a = (cosine, nothing, -0.5 * sine * alpha[1], nothing)
        b = (nothing, -sine, nothing, -0.5 * cosine * alpha[1])
        slope_a = (-0.5 * sine, nothing, -0.25 * cosine * alpha[1], nothing)
        slope_b = (nothing, -0.5 * cosine, nothing, 0.25 * sine * alpha[1])
    else:
        a = (cosine, nothing, 0.0, 0.0)
        b = (nothing, -sine, 0.0, 0.0)
        slope_a = (-0.5 * sine, nothing, 0.0, 0.0)
        slope_b = (nothing, -0.5 * cosine, 0.0, 0.0)
    return a, _cmul(b, turn, following), slope_a, _cmul(slope_b, turn, following), turn


@triton.jit
def _absorption(lineshape, rf_frequency, saturation, event, alpha, b0,
                lineshape_bins, lineshape_step, following: tl.constexpr):  # fmt: skip
    """The semisolid pool's saturation by a pulse, and the lineshape it read.

    Returns ``exp(saturation * alpha^2 * G(offset))``, ``G`` and its slope in
    the offset, each a real dual.
    """
    offset = tl.load(rf_frequency + event) - b0[0]
    deposited = tl.load(saturation + event)
    if following:
        shape, slope, curve = _lineshape_at_curve(
            lineshape, offset, lineshape_bins, lineshape_step
        )
        # The lineshape is read at the pulse's offset from the voxel, so a
        # step in the voxel's own off-resonance moves the read the other way.
        shape_dual = (shape, slope * -b0[1])
        slope_dual = (slope, curve * -b0[1])
    else:
        shape, slope = _lineshape_at_slope(
            lineshape, offset, lineshape_bins, lineshape_step
        )
        shape_dual = (shape, 0.0)
        slope_dual = (slope, 0.0)
    exponent = _rmul(
        (deposited, 0.0),
        _rmul(_rmul(alpha, alpha, following), shape_dual, following),
        following,
    )
    absorbed = tl.exp(exponent[0])
    return (absorbed, absorbed * exponent[1]), shape_dual, slope_dual, deposited


# ---------------------------------------------------------------------------
# The forward state machine, and with ``following`` its derivative along a
# direction. With ``keep`` it records the state entering every event for the
# adjoint instead of writing the signal.
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "state_count",
        "rows",
        "locations",
        "profile_bins",
        "lineshape_bins",
    ]
)
def _pooled_kernel(
    m0,
    b1,
    b1_phase,
    b0,
    efficiency,
    diffusion,
    velocity,
    dm0,
    db1,
    db1_phase,
    db0,
    defficiency,
    ddiffusion,
    dvelocity,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    dduration,
    dflip,
    dphase,
    table,
    dtable,
    pool_index,
    profile,
    profile_index,
    lineshape,
    pairs,
    pair_index,
    dpairs,
    output_real,
    output_imag,
    trajectory,
    base,
    atom_count,
    event_count,
    output_count,
    state_count,
    rows,
    flow_scale,
    washout_scale,
    profile_step,
    lineshape_step,
    locations,
    profile_bins,
    lineshape_bins,
    n: tl.constexpr,
    m: tl.constexpr,
    blocks: tl.constexpr,
    planes: tl.constexpr,
    atom_stride: tl.constexpr,
    shimmed: tl.constexpr,
    profiled: tl.constexpr,
    dynamic: tl.constexpr,
    directed_pairs: tl.constexpr,
    directed_table: tl.constexpr,
    following: tl.constexpr,
    keep: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
):
    problem = tl.program_id(0).to(tl.int64) + base
    atom = problem % atom_count
    train = problem // atom_count
    event_base = train * event_count
    voxel_at = atom * atom_stride
    location = atom % locations
    live = problem >= 0

    pool = tl.arange(0, P)[:, None]
    column = tl.arange(0, P)[None, :]
    state = tl.arange(0, S)[None, :]
    state_mask = state < state_count
    order = state.to(tl.float32)
    exchanging = pool < m
    semisolid = pool == n - 1
    row_width = n * n + n + 2 * m * m
    width = n + rows * row_width * blocks
    slot = table + atom * width
    directions = dtable + atom * width

    density_of = _read(m0, dm0, voxel_at, density, 1.0, following)
    voxel_b1 = _read(b1, db1, voxel_at, transmit, 1.0, following)
    voxel_b1_phase = _read(b1_phase, db1_phase, voxel_at, off_axis, 0.0, following)
    voxel_b0 = _read(b0, db0, voxel_at, off_axis, 0.0, following)
    inversion = _read(efficiency, defficiency, voxel_at, inverting, 1.0, following)
    damping_rate = _read(diffusion, ddiffusion, voxel_at, diffusing, 0.0, following)
    moved = _read(velocity, dvelocity, voxel_at, moving, 0.0, following)
    flow_rate = (flow_scale * moved[0], flow_scale * moved[1])
    washout_rate = (0.0, 0.0)
    if moving:
        heading = tl.where(moved[0] > 0.0, 1.0, 0.0) - tl.where(
            moved[0] < 0.0, 1.0, 0.0
        )
        washout_rate = (
            washout_scale * tl.abs(moved[0]),
            washout_scale * heading * moved[1],
        )

    equilibrium = _entries(
        slot, directions, pool, pool < n, 0.0, 0, directed_table, False, following
    )
    zero = tl.zeros((P, S), tl.float32)
    fpr = zero
    fpi = zero
    fmr = zero
    fmi = zero
    zr = tl.where(state == 0, equilibrium[0], 0.0)
    zi = zero
    dfpr = zero
    dfpi = zero
    dfmr = zero
    dfmi = zero
    dzr = zero
    dzi = zero
    if following:
        dzr = tl.where(state == 0, equilibrium[1], 0.0)
    tile = pool * S + state

    for event in range(0, event_count):
        if keep:
            at = trajectory + ((problem - base) * event_count + event) * (
                planes * P * S
            )
            tl.store(at + 0 * P * S + tile, fpr)
            tl.store(at + 1 * P * S + tile, fpi)
            tl.store(at + 2 * P * S + tile, fmr)
            tl.store(at + 3 * P * S + tile, fmi)
            tl.store(at + 4 * P * S + tile, zr)
            tl.store(at + 5 * P * S + tile, zi)
            if following:
                tl.store(at + 6 * P * S + tile, dfpr)
                tl.store(at + 7 * P * S + tile, dfpi)
                tl.store(at + 8 * P * S + tile, dfmr)
                tl.store(at + 9 * P * S + tile, dfmi)
                tl.store(at + 10 * P * S + tile, dzr)
                tl.store(at + 11 * P * S + tile, dzi)

        dt = _read(
            duration + event_base, dduration + event_base, event, True, 0.0, following
        )
        wout, unit_t, unit_z, _squared, _weight = _factors(
            dt, damping_rate, voxel_b0, flow_rate, washout_rate, order,
            off_axis, moving, diffusing, following,
        )  # fmt: skip
        carried = _cscale(wout, unit_t, following)
        spin = _cscale(wout, unit_z, following)
        row = tl.load(pool_index + event_base + event).to(tl.int64)
        longitudinal_op, restored, transverse_op = _operators(
            slot, directions, n + row * row_width, dt[1], rows * row_width, pool,
            column, n, m, directed_table, blocks > 1, following,
        )  # fmt: skip
        plus, minus, longitudinal, _mp, _mm, _mz = _relax(
            (fpr, fpi, dfpr, dfpi),
            (fmr, fmi, dfmr, dfmi),
            (zr, zi, dzr, dzi),
            transverse_op,
            longitudinal_op,
            restored,
            equilibrium,
            wout,
            carried,
            spin,
            state,
            following,
        )
        fpr = plus[0]
        fpi = plus[1]
        fmr = minus[0]
        fmi = minus[1]
        zr = longitudinal[0]
        zi = longitudinal[1]
        if following:
            dfpr = plus[2]
            dfpi = plus[3]
            dfmr = minus[2]
            dfmi = minus[3]
            dzr = longitudinal[2]
            dzi = longitudinal[3]

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event).to(tl.int32)
        if (event_action & 1) != 0:
            fpr, fpi, fmr, fmi = _shift(
                fpr, fpi, fmr, fmi, state, state_mask, state_count
            )
            if following:
                dfpr, dfpi, dfmr, dfmi = _shift(
                    dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count
                )
        if event_kind == 1:
            if (event_action & 4) != 0:
                # Every exchanging pool is free water and inverts like it; a
                # semisolid one is saturated by the pulse's own term.
                if following:
                    dzr = tl.where(
                        exchanging, -(inversion[1] * zr + inversion[0] * dzr), dzr
                    )
                    dzi = tl.where(
                        exchanging, -(inversion[1] * zi + inversion[0] * dzi), dzi
                    )
                zr = tl.where(exchanging, -inversion[0] * zr, zr)
                zi = tl.where(exchanging, -inversion[0] * zi, zi)
            else:
                pulse_b1 = voxel_b1
                pulse_b1_phase = voxel_b1_phase
                if shimmed:
                    transmit_at = (
                        tl.load(shim_index + event).to(tl.int64) * atom_count + atom
                    )
                    pulse_b1 = _read(b1, db1, transmit_at, transmit, 1.0, following)
                    pulse_b1_phase = _read(
                        b1_phase, db1_phase, transmit_at, True, 0.0, following
                    )
                nominal = _read(
                    flip + event_base, dflip + event_base, event, True, 0.0, following
                )
                played = _read(
                    phase + event_base, dphase + event_base, event, True, 0.0, following
                )
                alpha = _rmul(nominal, pulse_b1, following)
                phi = (played[0] + pulse_b1_phase[0], played[1] + pulse_b1_phase[1])
                if n > m:
                    absorbed, _shape, _slope, _deposited = _absorption(
                        lineshape, rf_frequency, saturation, event, alpha, voxel_b0,
                        lineshape_bins, lineshape_step, following,
                    )  # fmt: skip
                    if following:
                        dzr = tl.where(
                            semisolid, absorbed[1] * zr + absorbed[0] * dzr, dzr
                        )
                        dzi = tl.where(
                            semisolid, absorbed[1] * zi + absorbed[0] * dzi, dzi
                        )
                    zr = tl.where(semisolid, absorbed[0] * zr, zr)
                    zi = tl.where(semisolid, absorbed[0] * zi, zi)
                if dynamic:
                    if following:
                        a, spun = _dynamic_pair_dual_at(
                            pairs, dpairs, pair_index, event_base, event, atom,
                            atom_count, live, phi[0], phi[1], directed_pairs,
                        )  # fmt: skip
                    else:
                        held = _dynamic_pair_at(
                            pairs, pair_index, event_base, event, atom, atom_count, live
                        )
                        a = (held[0], held[1], 0.0, 0.0)
                        spun = _cmul(
                            (held[2], held[3], 0.0, 0.0),
                            _polar((-phi[0], 0.0), following),
                            following,
                        )
                elif profiled:
                    at_row = _table_row(profile_index, event, location, locations)
                    turn = _polar((-phi[0], -phi[1]), following)
                    if following:
                        read = _profile_pair_slope(
                            profile, at_row, alpha[0], profile_bins, profile_step
                        )
                        a = (read[0], read[2], read[1] * alpha[1], read[3] * alpha[1])
                        b = (read[4], read[6], read[5] * alpha[1], read[7] * alpha[1])
                    else:
                        read = _profile_pair(
                            profile, at_row, alpha[0], profile_bins, profile_step
                        )
                        a = (read[0], read[1], 0.0, 0.0)
                        b = (read[2], read[3], 0.0, 0.0)
                    spun = _cmul(b, turn, following)
                else:
                    a, spun, _sa, _sb, _turn = _hard_pair(alpha, phi, following)
                if following:
                    turned = _rotate_spinor_dual(
                        a[0], a[1], spun[0], spun[1], a[2], a[3], spun[2], spun[3],
                        fpr, fpi, fmr, fmi, zr, zi, dfpr, dfpi, dfmr, dfmi, dzr, dzi,
                    )  # fmt: skip
                    dfpr = tl.where(exchanging, turned[6], dfpr)
                    dfpi = tl.where(exchanging, turned[7], dfpi)
                    dfmr = tl.where(exchanging, turned[8], dfmr)
                    dfmi = tl.where(exchanging, turned[9], dfmi)
                    dzr = tl.where(exchanging, turned[10], dzr)
                    dzi = tl.where(exchanging, turned[11], dzi)
                else:
                    turned = _rotate_spinor(
                        a[0], a[1], spun[0], spun[1], fpr, fpi, fmr, fmi, zr, zi
                    )
                fpr = tl.where(exchanging, turned[0], fpr)
                fpi = tl.where(exchanging, turned[1], fpi)
                fmr = tl.where(exchanging, turned[2], fmr)
                fmi = tl.where(exchanging, turned[3], fmi)
                zr = tl.where(exchanging, turned[4], zr)
                zi = tl.where(exchanging, turned[5], zi)
        if not keep:
            if (event_kind == 2) & ((event_action & 32) != 0):
                origin = state == 0
                recorded = (
                    _total(fpr, origin),
                    _total(fpi, origin),
                    _total(dfpr, origin),
                    _total(dfpi, origin),
                )
                read_phase = _read(
                    phase + event_base, dphase + event_base, event, True, 0.0, following
                )
                signal = _cscale(
                    density_of,
                    _cmul(
                        recorded,
                        _polar((-read_phase[0], -read_phase[1]), following),
                        following,
                    ),
                    following,
                )
                out = tl.load(output_index + event)
                written = problem * output_count + out
                if following:
                    tl.store(output_real + written, signal[2], mask=out >= 0)
                    tl.store(output_imag + written, signal[3], mask=out >= 0)
                else:
                    tl.store(output_real + written, signal[0], mask=out >= 0)
                    tl.store(output_imag + written, signal[1], mask=out >= 0)
        if (event_action & 2) != 0:
            fpr, fpi, fmr, fmi = _shift(
                fpr, fpi, fmr, fmi, state, state_mask, state_count
            )
            if following:
                dfpr, dfpi, dfmr, dfmi = _shift(
                    dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count
                )
        if (event_action & 8) != 0:
            fpr = zero
            fpi = zero
            fmr = zero
            fmi = zero
            if following:
                dfpr = zero
                dfpi = zero
                dfmr = zero
                dfmi = zero
        elif (event_action & 16) != 0:
            fpr, fpi, fmr, fmi = _shift(
                fpr, fpi, fmr, fmi, state, state_mask, state_count
            )
            if following:
                dfpr, dfpi, dfmr, dfmi = _shift(
                    dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count
                )


# ---------------------------------------------------------------------------
# The adjoint, walking back from the states ``_pooled_kernel`` recorded. With
# ``following`` it is the forward-over-reverse pass: the value planes carry the
# first-order cotangents and the tangent planes their derivative.
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "state_count",
        "rows",
        "locations",
        "profile_bins",
        "lineshape_bins",
    ]
)
def _pooled_adjoint_kernel(
    m0,
    b1,
    b1_phase,
    b0,
    efficiency,
    diffusion,
    velocity,
    dm0,
    db1,
    db1_phase,
    db0,
    defficiency,
    ddiffusion,
    dvelocity,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    dduration,
    dflip,
    dphase,
    table,
    dtable,
    pool_index,
    profile,
    profile_index,
    lineshape,
    pairs,
    pair_index,
    dpairs,
    grad_real,
    grad_imag,
    grad_tissue,
    dgrad_tissue,
    grad_duration,
    dgrad_duration,
    grad_flip,
    dgrad_flip,
    grad_phase,
    dgrad_phase,
    grad_table,
    dgrad_table,
    grad_pairs,
    dgrad_pairs,
    trajectory,
    base,
    atom_count,
    event_count,
    output_count,
    state_count,
    rows,
    flow_scale,
    washout_scale,
    profile_step,
    lineshape_step,
    locations,
    profile_bins,
    lineshape_bins,
    m0_row,
    b1_row,
    b1_phase_row,
    b0_row,
    efficiency_row,
    diffusion_row,
    velocity_row,
    n: tl.constexpr,
    m: tl.constexpr,
    blocks: tl.constexpr,
    planes: tl.constexpr,
    atom_stride: tl.constexpr,
    shimmed: tl.constexpr,
    profiled: tl.constexpr,
    dynamic: tl.constexpr,
    directed_pairs: tl.constexpr,
    directed_table: tl.constexpr,
    following: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
):
    problem = tl.program_id(0).to(tl.int64) + base
    atom = problem % atom_count
    train = problem // atom_count
    event_base = train * event_count
    voxel_at = atom * atom_stride
    location = atom % locations
    live = problem >= 0

    pool = tl.arange(0, P)[:, None]
    column = tl.arange(0, P)[None, :]
    state = tl.arange(0, S)[None, :]
    state_mask = state < state_count
    order = state.to(tl.float32)
    origin = state == 0
    exchanging = pool < m
    semisolid = pool == n - 1
    held_rows = pool < n
    square = (pool < n) & (column < n)
    across_mask = (pool < m) & (column < m)
    live_exchanging = exchanging & state_mask
    row_width = n * n + n + 2 * m * m
    width = n + rows * row_width * blocks
    slot = table + atom * width
    directions = dtable + atom * width
    slot_grad = grad_table + problem * width
    slot_curve = dgrad_table + problem * width

    density_of = _read(m0, dm0, voxel_at, density, 1.0, following)
    voxel_b1 = _read(b1, db1, voxel_at, transmit, 1.0, following)
    voxel_b1_phase = _read(b1_phase, db1_phase, voxel_at, off_axis, 0.0, following)
    voxel_b0 = _read(b0, db0, voxel_at, off_axis, 0.0, following)
    inversion = _read(efficiency, defficiency, voxel_at, inverting, 1.0, following)
    damping_rate = _read(diffusion, ddiffusion, voxel_at, diffusing, 0.0, following)
    moved = _read(velocity, dvelocity, voxel_at, moving, 0.0, following)
    flow_rate = (flow_scale * moved[0], flow_scale * moved[1])
    washout_rate = (0.0, 0.0)
    heading = 0.0
    if moving:
        heading = tl.where(moved[0] > 0.0, 1.0, 0.0) - tl.where(
            moved[0] < 0.0, 1.0, 0.0
        )
        washout_rate = (
            washout_scale * tl.abs(moved[0]),
            washout_scale * heading * moved[1],
        )
    equilibrium = _entries(
        slot, directions, pool, held_rows, 0.0, 0, directed_table, False, following
    )

    zero = tl.zeros((P, S), tl.float32)
    pbr = zero
    pbi = zero
    mbr = zero
    mbi = zero
    zbr = zero
    zbi = zero
    dpbr = zero
    dpbi = zero
    dmbr = zero
    dmbi = zero
    dzbr = zero
    dzbi = zero
    grad_eq = equilibrium[0] * 0.0
    curve_eq = equilibrium[0] * 0.0
    grad_m0 = 0.0
    grad_b1 = 0.0
    grad_b1_phase = 0.0
    grad_b0 = 0.0
    grad_efficiency = 0.0
    grad_damping = 0.0
    grad_flow = 0.0
    grad_washout = 0.0
    # Only what every interval adds to is carried in the derivative plane;
    # what a pulse or a readout adds is stored as the branch reaches it. A
    # carried scalar updated inside one of those branches crashes the layout
    # pass of Triton 3.8 in this kernel.
    curve_b0 = 0.0
    curve_damping = 0.0
    curve_flow = 0.0
    curve_washout = 0.0
    # Transmit gradients are summed per shim: the running pair is flushed to
    # its row whenever the walk back reaches a pulse on a different one.
    held = 0
    tile = pool * S + state

    for step in range(0, event_count):
        event = event_count - 1 - step
        at = trajectory + ((problem - base) * event_count + event) * (planes * P * S)
        if following:
            plus_in = (
                tl.load(at + 0 * P * S + tile),
                tl.load(at + 1 * P * S + tile),
                tl.load(at + 6 * P * S + tile),
                tl.load(at + 7 * P * S + tile),
            )
            minus_in = (
                tl.load(at + 2 * P * S + tile),
                tl.load(at + 3 * P * S + tile),
                tl.load(at + 8 * P * S + tile),
                tl.load(at + 9 * P * S + tile),
            )
            z_in = (
                tl.load(at + 4 * P * S + tile),
                tl.load(at + 5 * P * S + tile),
                tl.load(at + 10 * P * S + tile),
                tl.load(at + 11 * P * S + tile),
            )
        else:
            plus_in = (tl.load(at + tile), tl.load(at + P * S + tile), 0.0, 0.0)
            minus_in = (
                tl.load(at + 2 * P * S + tile),
                tl.load(at + 3 * P * S + tile),
                0.0,
                0.0,
            )
            z_in = (
                tl.load(at + 4 * P * S + tile),
                tl.load(at + 5 * P * S + tile),
                0.0,
                0.0,
            )

        dt = _read(
            duration + event_base, dduration + event_base, event, True, 0.0, following
        )
        wout, unit_t, unit_z, squared, weight = _factors(
            dt, damping_rate, voxel_b0, flow_rate, washout_rate, order,
            off_axis, moving, diffusing, following,
        )  # fmt: skip
        carried = _cscale(wout, unit_t, following)
        spin = _cscale(wout, unit_z, following)
        row = tl.load(pool_index + event_base + event).to(tl.int64)
        row_offset = n + row * row_width
        longitudinal_op, restored, transverse_op = _operators(
            slot, directions, row_offset, dt[1], rows * row_width, pool, column,
            n, m, directed_table, blocks > 1, following,
        )  # fmt: skip

        # Replay the interval to recover the states the event acted on.
        relaxed_plus, relaxed_minus, relaxed_z, mixed_plus, mixed_minus, mixed_z = (
            _relax(
                plus_in,
                minus_in,
                z_in,
                transverse_op,
                longitudinal_op,
                restored,
                equilibrium,
                wout,
                carried,
                spin,
                state,
                following,
            )
        )
        spr = relaxed_plus[0]
        spi = relaxed_plus[1]
        smr = relaxed_minus[0]
        smi = relaxed_minus[1]
        dspr = zero
        dspi = zero
        dsmr = zero
        dsmi = zero
        if following:
            dspr = relaxed_plus[2]
            dspi = relaxed_plus[3]
            dsmr = relaxed_minus[2]
            dsmi = relaxed_minus[3]
        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event).to(tl.int32)
        if (event_action & 1) != 0:
            spr, spi, smr, smi = _shift(
                spr, spi, smr, smi, state, state_mask, state_count
            )
            if following:
                dspr, dspi, dsmr, dsmi = _shift(
                    dspr, dspi, dsmr, dsmi, state, state_mask, state_count
                )

        # The trailing shift or spoil.
        if (event_action & 8) != 0:
            pbr = zero
            pbi = zero
            mbr = zero
            mbi = zero
            if following:
                dpbr = zero
                dpbi = zero
                dmbr = zero
                dmbi = zero
        elif (event_action & 16) != 0:
            pbr, pbi, mbr, mbi = _shift_adjoint(
                pbr, pbi, mbr, mbi, state, state_mask, state_count
            )
            if following:
                dpbr, dpbi, dmbr, dmbi = _shift_adjoint(
                    dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count
                )
        if (event_action & 2) != 0:
            pbr, pbi, mbr, mbi = _shift_adjoint(
                pbr, pbi, mbr, mbi, state, state_mask, state_count
            )
            if following:
                dpbr, dpbi, dmbr, dmbi = _shift_adjoint(
                    dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count
                )

        # The readout. It carries no pulse, so what it recorded is the state
        # the pre-shift left.
        out = tl.load(output_index + event)
        if (event_kind == 2) & ((event_action & 32) != 0) & (out >= 0):
            index = problem * output_count + out
            seed = (tl.load(grad_real + index), tl.load(grad_imag + index), 0.0, 0.0)
            read_phase = _read(
                phase + event_base, dphase + event_base, event, True, 0.0, following
            )
            demodulation = _polar((-read_phase[0], -read_phase[1]), following)
            recorded = (
                _total(spr, origin),
                _total(spi, origin),
                _total(dspr, origin),
                _total(dspi, origin),
            )
            density_grad = _re_dot(
                seed, _cmul(recorded, demodulation, following), following
            )
            turned = (
                demodulation[1],
                -demodulation[0],
                demodulation[3],
                -demodulation[2],
            )
            phase_grad = _re_dot(
                seed,
                _cscale(density_of, _cmul(recorded, turned, following), following),
                following,
            )
            grad_m0 += density_grad[0]
            tl.atomic_add(grad_phase + event_base + event, phase_grad[0])
            weighted = _cmul(
                _conj(_cscale(density_of, demodulation, following)), seed, following
            )
            put = origin & exchanging
            pbr += tl.where(put, weighted[0], 0.0)
            pbi += tl.where(put, weighted[1], 0.0)
            if following:
                tl.atomic_add(
                    dgrad_tissue + m0_row * atom_count + atom, density_grad[1]
                )
                tl.atomic_add(dgrad_phase + event_base + event, phase_grad[1])
                dpbr += tl.where(put, weighted[2], 0.0)
                dpbi += tl.where(put, weighted[3], 0.0)

        # The pulse.
        if event_kind == 1:
            if (event_action & 4) != 0:
                bar = (zbr, zbi, dzbr, dzbi)
                taken = _rtotal(
                    _re_dot(bar, relaxed_z, following), live_exchanging, following
                )
                grad_efficiency -= taken[0]
                if following:
                    tl.atomic_add(
                        dgrad_tissue + efficiency_row * atom_count + atom, -taken[1]
                    )
                    dzbr = tl.where(
                        exchanging, -(inversion[1] * zbr + inversion[0] * dzbr), dzbr
                    )
                    dzbi = tl.where(
                        exchanging, -(inversion[1] * zbi + inversion[0] * dzbi), dzbi
                    )
                zbr = tl.where(exchanging, -inversion[0] * zbr, zbr)
                zbi = tl.where(exchanging, -inversion[0] * zbi, zbi)
            else:
                pulse_b1 = voxel_b1
                pulse_b1_phase = voxel_b1_phase
                transmit_row = 0
                if shimmed:
                    shim = tl.load(shim_index + event).to(tl.int32)
                    changed = shim != held
                    b1_at = (b1_row + held).to(tl.int64) * atom_count + atom
                    b1_phase_at = (b1_phase_row + held).to(tl.int64) * atom_count + atom
                    tl.atomic_add(grad_tissue + b1_at, grad_b1, mask=changed)
                    tl.atomic_add(
                        grad_tissue + b1_phase_at, grad_b1_phase, mask=changed
                    )
                    grad_b1 = tl.where(changed, 0.0, grad_b1)
                    grad_b1_phase = tl.where(changed, 0.0, grad_b1_phase)
                    held = shim
                    transmit_row = shim
                    transmit_at = shim.to(tl.int64) * atom_count + atom
                    pulse_b1 = _read(b1, db1, transmit_at, transmit, 1.0, following)
                    pulse_b1_phase = _read(
                        b1_phase, db1_phase, transmit_at, True, 0.0, following
                    )
                nominal = _read(
                    flip + event_base, dflip + event_base, event, True, 0.0, following
                )
                played = _read(
                    phase + event_base, dphase + event_base, event, True, 0.0, following
                )
                alpha = _rmul(nominal, pulse_b1, following)
                phi = (played[0] + pulse_b1_phase[0], played[1] + pulse_b1_phase[1])
                turn = _polar((-phi[0], -phi[1]), following)
                if dynamic:
                    a, spun = _dynamic_pair_dual_at(
                        pairs, dpairs, pair_index, event_base, event, atom, atom_count,
                        live, phi[0], phi[1], directed_pairs,
                    )  # fmt: skip
                    slope_a = a
                    slope_b = spun
                elif profiled:
                    a, spun, slope_a, slope_b = _profiled_pair_dual(
                        profile,
                        _table_row(profile_index, event, location, locations),
                        alpha[0],
                        alpha[1],
                        phi[0],
                        phi[1],
                        profile_bins,
                        profile_step,
                    )
                else:
                    a, spun, slope_a, slope_b, _turn = _hard_pair(alpha, phi, following)
                if following:
                    pair_a, pair_b, back_p, back_m, back_z = _spinor_adjoint_dual(
                        a,
                        spun,
                        (spr, spi, dspr, dspi),
                        (smr, smi, dsmr, dsmi),
                        relaxed_z,
                        (pbr, pbi, dpbr, dpbi),
                        (mbr, mbi, dmbr, dmbi),
                        (zbr, zbi, dzbr, dzbi),
                    )
                    grad_a = (
                        _total(pair_a[0], live_exchanging),
                        _total(pair_a[1], live_exchanging),
                        _total(pair_a[2], live_exchanging),
                        _total(pair_a[3], live_exchanging),
                    )
                    grad_b = (
                        _total(pair_b[0], live_exchanging),
                        _total(pair_b[1], live_exchanging),
                        _total(pair_b[2], live_exchanging),
                        _total(pair_b[3], live_exchanging),
                    )
                    dpbr = tl.where(exchanging, back_p[2], dpbr)
                    dpbi = tl.where(exchanging, back_p[3], dpbi)
                    dmbr = tl.where(exchanging, back_m[2], dmbr)
                    dmbi = tl.where(exchanging, back_m[3], dmbi)
                    dzbr = tl.where(exchanging, back_z[2], dzbr)
                    dzbi = tl.where(exchanging, back_z[3], dzbi)
                    pbr = tl.where(exchanging, back_p[0], pbr)
                    pbi = tl.where(exchanging, back_p[1], pbi)
                    mbr = tl.where(exchanging, back_m[0], mbr)
                    mbi = tl.where(exchanging, back_m[1], mbi)
                    zbr = tl.where(exchanging, back_z[0], zbr)
                    zbi = tl.where(exchanging, back_z[1], zbi)
                else:
                    back = _spinor_adjoint(
                        a[0], a[1], spun[0], spun[1], spr, spi, smr, smi,
                        relaxed_z[0], relaxed_z[1], pbr, pbi, mbr, mbi, zbr, zbi,
                    )  # fmt: skip
                    grad_a = (
                        _total(back[0], live_exchanging),
                        _total(back[1], live_exchanging),
                        0.0,
                        0.0,
                    )
                    grad_b = (
                        _total(back[2], live_exchanging),
                        _total(back[3], live_exchanging),
                        0.0,
                        0.0,
                    )
                    pbr = tl.where(exchanging, back[4], pbr)
                    pbi = tl.where(exchanging, back[5], pbi)
                    mbr = tl.where(exchanging, back[6], mbr)
                    mbi = tl.where(exchanging, back[7], mbi)
                    zbr = tl.where(exchanging, back[8], zbr)
                    zbi = tl.where(exchanging, back[9], zbi)
                # The RF phase turns the axis once the pair is out, so it
                # reaches ``b`` alone -- under every mode.
                grad_phi = _re_dot(
                    grad_b, (spun[1], -spun[0], spun[3], -spun[2]), following
                )
                grad_alpha = (0.0, 0.0)
                if dynamic:
                    # The flip is inside the pair rather than read against it,
                    # so the cotangent goes out on the pair; ``b`` was turned
                    # by the phase after the pair came out, so it turns back.
                    unturned = _cmul(grad_b, _conj(turn), following)
                    entry = (
                        tl.load(pair_index + event_base + event).to(tl.int64)
                        * atom_count
                        + atom
                    ) * 4
                    tl.atomic_add(grad_pairs + entry + 0, grad_a[0])
                    tl.atomic_add(grad_pairs + entry + 1, grad_a[1])
                    tl.atomic_add(grad_pairs + entry + 2, unturned[0])
                    tl.atomic_add(grad_pairs + entry + 3, unturned[1])
                    if following:
                        tl.atomic_add(dgrad_pairs + entry + 0, grad_a[2])
                        tl.atomic_add(dgrad_pairs + entry + 1, grad_a[3])
                        tl.atomic_add(dgrad_pairs + entry + 2, unturned[2])
                        tl.atomic_add(dgrad_pairs + entry + 3, unturned[3])
                else:
                    along_a = _re_dot(grad_a, slope_a, following)
                    along_b = _re_dot(grad_b, slope_b, following)
                    grad_alpha = (along_a[0] + along_b[0], along_a[1] + along_b[1])
                if n > m:
                    # The pulse scales every order of the semisolid pool by one
                    # real number, so its cotangent is one sum over the states.
                    absorbed, shape, slope, deposited = _absorption(
                        lineshape, rf_frequency, saturation, event, alpha, voxel_b0,
                        lineshape_bins, lineshape_step, following,
                    )  # fmt: skip
                    taken = _rtotal(
                        _re_dot((zbr, zbi, dzbr, dzbi), relaxed_z, following),
                        semisolid & state_mask,
                        following,
                    )
                    if following:
                        dzbr = tl.where(
                            semisolid, absorbed[1] * zbr + absorbed[0] * dzbr, dzbr
                        )
                        dzbi = tl.where(
                            semisolid, absorbed[1] * zbi + absorbed[0] * dzbi, dzbi
                        )
                    zbr = tl.where(semisolid, absorbed[0] * zbr, zbr)
                    zbi = tl.where(semisolid, absorbed[0] * zbi, zbi)
                    exponent = _rmul(taken, absorbed, following)
                    swing = _rmul(
                        (2.0 * deposited, 0.0),
                        _rmul(_rmul(exponent, alpha, following), shape, following),
                        following,
                    )
                    grad_alpha = (grad_alpha[0] + swing[0], grad_alpha[1] + swing[1])
                    shifted = _rmul(
                        (deposited, 0.0),
                        _rmul(
                            _rmul(_rmul(exponent, alpha, following), alpha, following),
                            slope,
                            following,
                        ),
                        following,
                    )
                    grad_b0 -= shifted[0]
                    if following:
                        tl.atomic_add(
                            dgrad_tissue + b0_row * atom_count + atom, -shifted[1]
                        )
                flip_grad = _rmul(grad_alpha, pulse_b1, following)
                b1_grad = _rmul(grad_alpha, nominal, following)
                tl.atomic_add(grad_flip + event_base + event, flip_grad[0])
                tl.atomic_add(grad_phase + event_base + event, grad_phi[0])
                grad_b1 += b1_grad[0]
                grad_b1_phase += grad_phi[0]
                if following:
                    tl.atomic_add(dgrad_flip + event_base + event, flip_grad[1])
                    tl.atomic_add(dgrad_phase + event_base + event, grad_phi[1])
                    tl.atomic_add(
                        dgrad_tissue
                        + (b1_row + transmit_row).to(tl.int64) * atom_count
                        + atom,
                        b1_grad[1],
                    )
                    tl.atomic_add(
                        dgrad_tissue
                        + (b1_phase_row + transmit_row).to(tl.int64) * atom_count
                        + atom,
                        grad_phi[1],
                    )

        if (event_action & 1) != 0:
            pbr, pbi, mbr, mbi = _shift_adjoint(
                pbr, pbi, mbr, mbi, state, state_mask, state_count
            )
            if following:
                dpbr, dpbi, dmbr, dmbi = _shift_adjoint(
                    dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count
                )

        # The interval. Order zero also carries the recovery, which is the
        # equilibrium less what washout leaves of the operator applied to it.
        plus_bar = (pbr, pbi, dpbr, dpbi)
        minus_bar = (mbr, mbi, dmbr, dmbi)
        z_bar = (zbr, zbi, dzbr, dzbi)
        if not following:
            plus_bar = (pbr, pbi, 0.0, 0.0)
            minus_bar = (mbr, mbi, 0.0, 0.0)
            z_bar = (zbr, zbi, 0.0, 0.0)
        seed = (
            tl.sum(tl.where(origin, zbr, 0.0), axis=1)[:, None],
            tl.sum(tl.where(origin, dzbr, 0.0), axis=1)[:, None],
        )
        grad_eq += seed[0]
        restored_grad = (-(wout[0] * seed[0]), -(wout[1] * seed[0] + wout[0] * seed[1]))
        grad_wout = (
            -_total(seed[0] * restored[0], held_rows),
            -_total(seed[1] * restored[0] + seed[0] * restored[1], held_rows),
        )
        if following:
            curve_eq += seed[1]

        out_plus = _cmul(
            _conj(plus_bar), _cmul(carried, mixed_plus, following), following
        )
        out_minus = _cmul(
            _conj(minus_bar), _cmul(_conj(carried), mixed_minus, following), following
        )
        out_z = _cmul(_conj(z_bar), _cmul(spin, mixed_z, following), following)
        # The damping is homogeneous of degree one in every state it acts on,
        # so its gradient times the damping itself is the cotangent taken
        # against the states the interval leaves; the turns are the same
        # derivatives with an imaginary weight.
        transverse_scaled = tl.sum(out_plus[0] + out_minus[0], axis=0)[None, :]
        transverse_angle = tl.sum(out_minus[1] - out_plus[1], axis=0)[None, :]
        longitudinal_scaled = tl.sum(out_z[0], axis=0)[None, :]
        longitudinal_angle = tl.sum(-out_z[1], axis=0)[None, :]
        wout_plus = _re_dot(plus_bar, _cmul(unit_t, mixed_plus, following), following)
        wout_minus = _re_dot(
            minus_bar, _cmul(_conj(unit_t), mixed_minus, following), following
        )
        wout_z = _re_dot(z_bar, _cmul(unit_z, mixed_z, following), following)
        grad_wout = (
            grad_wout[0] + _total(wout_plus[0] + wout_minus[0] + wout_z[0], state_mask),
            grad_wout[1],
        )
        half = order + 0.5
        grad_angle = (_total(transverse_angle, state_mask), 0.0)
        grad_b_factor = (
            -_total(
                weight * transverse_scaled + squared * longitudinal_scaled, state_mask
            ),
            0.0,
        )
        grad_turn = (
            -_total(half * transverse_angle + order * longitudinal_angle, state_mask),
            0.0,
        )
        if following:
            grad_wout = (
                grad_wout[0],
                grad_wout[1]
                + _total(wout_plus[1] + wout_minus[1] + wout_z[1], state_mask),
            )
            transverse_scaled_t = tl.sum(out_plus[2] + out_minus[2], axis=0)[None, :]
            transverse_angle_t = tl.sum(out_minus[3] - out_plus[3], axis=0)[None, :]
            longitudinal_scaled_t = tl.sum(out_z[2], axis=0)[None, :]
            longitudinal_angle_t = tl.sum(-out_z[3], axis=0)[None, :]
            grad_angle = (grad_angle[0], _total(transverse_angle_t, state_mask))
            grad_b_factor = (
                grad_b_factor[0],
                -_total(
                    weight * transverse_scaled_t + squared * longitudinal_scaled_t,
                    state_mask,
                ),
            )
            grad_turn = (
                grad_turn[0],
                -_total(
                    half * transverse_angle_t + order * longitudinal_angle_t, state_mask
                ),
            )

        # The operators' own entries. ``F-`` follows the conjugate of the
        # transverse operator, so its cotangent lands on the entry itself.
        released = _conj(carried)
        transverse_grad = _cadd(
            _couter(_cmul(plus_bar, released, following), _conj(plus_in), following),
            _couter(_cmul(_conj(minus_bar), released, following), minus_in, following),
        )
        weighed = _cmul(_conj(z_bar), spin, following)
        longitudinal_grad = (
            _outer(weighed[0], z_in[0]) - _outer(weighed[1], z_in[1]),
            0.0,
        )
        if following:
            longitudinal_grad = (
                longitudinal_grad[0],
                _outer(weighed[2], z_in[0])
                - _outer(weighed[3], z_in[1])
                + _outer(weighed[0], z_in[2])
                - _outer(weighed[1], z_in[3]),
            )

        # The cotangents back through the interval.
        back_plus = _cmul(
            released, _apply(transverse_op, plus_bar, True, True, following), following
        )
        back_minus = _cmul(
            carried, _apply(transverse_op, minus_bar, False, True, following), following
        )
        back_z = _apply_real(
            longitudinal_op, _cmul(_conj(spin), z_bar, following), True, following
        )
        pbr = back_plus[0]
        pbi = back_plus[1]
        mbr = back_minus[0]
        mbi = back_minus[1]
        zbr = back_z[0]
        zbi = back_z[1]
        if following:
            dpbr = back_plus[2]
            dpbi = back_plus[3]
            dmbr = back_minus[2]
            dmbi = back_minus[3]
            dzbr = back_z[2]
            dzbi = back_z[3]

        # The row this event read is shared with every event of its length;
        # its cotangent is summed into it, and reaches the event's own length
        # through the row's slope.
        longitudinal_at = row_offset + pool * n + column
        restored_at = row_offset + n * n + pool
        across = row_offset + n * n + n + 2 * (pool * m + column)
        tl.atomic_add(slot_grad + longitudinal_at, longitudinal_grad[0], mask=square)
        tl.atomic_add(slot_grad + restored_at, restored_grad[0], mask=held_rows)
        tl.atomic_add(slot_grad + across, transverse_grad[0], mask=across_mask)
        tl.atomic_add(slot_grad + across + 1, transverse_grad[1], mask=across_mask)
        if following:
            tl.atomic_add(
                slot_curve + longitudinal_at, longitudinal_grad[1], mask=square
            )
            tl.atomic_add(slot_curve + restored_at, restored_grad[1], mask=held_rows)
            tl.atomic_add(slot_curve + across, transverse_grad[2], mask=across_mask)
            tl.atomic_add(slot_curve + across + 1, transverse_grad[3], mask=across_mask)
        table_duration = (0.0, 0.0)
        if blocks > 1:
            further = rows * row_width
            slope_z = _entries(
                slot, directions, further + longitudinal_at, square, dt[1], further,
                directed_table, True, following,
            )  # fmt: skip
            slope_restored = _entries(
                slot, directions, further + restored_at, held_rows, dt[1], further,
                directed_table, True, following,
            )  # fmt: skip
            slope_real = _entries(
                slot, directions, further + across, across_mask, dt[1], further,
                directed_table, True, following,
            )  # fmt: skip
            slope_imag = _entries(
                slot, directions, further + across + 1, across_mask, dt[1], further,
                directed_table, True, following,
            )  # fmt: skip
            table_duration = (
                _total(longitudinal_grad[0] * slope_z[0], square)
                + _total(restored_grad[0] * slope_restored[0], held_rows)
                + _total(
                    transverse_grad[0] * slope_real[0]
                    + transverse_grad[1] * slope_imag[0],
                    across_mask,
                ),
                0.0,
            )
            if following:
                table_duration = (
                    table_duration[0],
                    _total(
                        longitudinal_grad[1] * slope_z[0]
                        + longitudinal_grad[0] * slope_z[1],
                        square,
                    )
                    + _total(
                        restored_grad[1] * slope_restored[0]
                        + restored_grad[0] * slope_restored[1],
                        held_rows,
                    )
                    + _total(
                        transverse_grad[2] * slope_real[0]
                        + transverse_grad[0] * slope_real[1]
                        + transverse_grad[3] * slope_imag[0]
                        + transverse_grad[1] * slope_imag[1],
                        across_mask,
                    ),
                )
                # At the row's own length the slope reaches the output only
                # through a direction in that length, so only the tangent
                # plane takes this.
                tl.atomic_add(
                    slot_curve + further + longitudinal_at,
                    longitudinal_grad[0] * dt[1],
                    mask=square,
                )
                tl.atomic_add(
                    slot_curve + further + restored_at,
                    restored_grad[0] * dt[1],
                    mask=held_rows,
                )
                tl.atomic_add(
                    slot_curve + further + across,
                    transverse_grad[0] * dt[1],
                    mask=across_mask,
                )
                tl.atomic_add(
                    slot_curve + further + across + 1,
                    transverse_grad[1] * dt[1],
                    mask=across_mask,
                )

        # Washout scales every factor the interval applies and the recovery it
        # leaves; past the clamp nothing depends on the rate.
        fraction_grad = (0.0, 0.0)
        if moving:
            inside = washout_rate[0] * dt[0] < 1.0
            fraction_grad = (
                tl.where(inside, -grad_wout[0], 0.0),
                tl.where(inside, -grad_wout[1], 0.0),
            )
        angle_rate = _rmul((-6.283185307179586, 0.0), dt, following)
        b0_grad = _rmul(grad_angle, angle_rate, following)
        damping_grad = _rmul(grad_b_factor, dt, following)
        flow_grad = _rmul(grad_turn, dt, following)
        washout_grad = _rmul(fraction_grad, dt, following)
        duration_grad = _rmul(
            grad_angle, _rmul((-6.283185307179586, 0.0), voxel_b0, following), following
        )
        through_damping = _rmul(grad_b_factor, damping_rate, following)
        through_flow = _rmul(grad_turn, flow_rate, following)
        through_washout = _rmul(fraction_grad, washout_rate, following)
        grad_b0 += b0_grad[0]
        grad_damping += damping_grad[0]
        grad_flow += flow_grad[0]
        grad_washout += washout_grad[0]
        tl.atomic_add(
            grad_duration + event_base + event,
            duration_grad[0]
            + through_damping[0]
            + through_flow[0]
            + through_washout[0]
            + table_duration[0],
        )
        if following:
            curve_b0 += b0_grad[1]
            curve_damping += damping_grad[1]
            curve_flow += flow_grad[1]
            curve_washout += washout_grad[1]
            tl.atomic_add(
                dgrad_duration + event_base + event,
                duration_grad[1]
                + through_damping[1]
                + through_flow[1]
                + through_washout[1]
                + table_duration[1],
            )

    # The equilibrium is also where every pool starts, which the walk back
    # reaches last.
    grad_eq += tl.sum(tl.where(origin, zbr, 0.0), axis=1)[:, None]
    tl.atomic_add(slot_grad + pool, grad_eq, mask=held_rows)
    tl.atomic_add(grad_tissue + m0_row * atom_count + atom, grad_m0)
    tl.atomic_add(
        grad_tissue + (b1_row + held).to(tl.int64) * atom_count + atom, grad_b1
    )
    tl.atomic_add(
        grad_tissue + (b1_phase_row + held).to(tl.int64) * atom_count + atom,
        grad_b1_phase,
    )
    tl.atomic_add(grad_tissue + b0_row * atom_count + atom, grad_b0)
    tl.atomic_add(grad_tissue + efficiency_row * atom_count + atom, grad_efficiency)
    tl.atomic_add(grad_tissue + diffusion_row * atom_count + atom, grad_damping)
    # One buffer drives two rates, so the velocity gradient is the sum of what
    # each geometry carries back.
    tl.atomic_add(
        grad_tissue + velocity_row * atom_count + atom,
        flow_scale * grad_flow + heading * washout_scale * grad_washout,
    )
    if following:
        curve_eq += tl.sum(tl.where(origin, dzbr, 0.0), axis=1)[:, None]
        tl.atomic_add(slot_curve + pool, curve_eq, mask=held_rows)
        tl.atomic_add(dgrad_tissue + b0_row * atom_count + atom, curve_b0)
        tl.atomic_add(dgrad_tissue + diffusion_row * atom_count + atom, curve_damping)
        tl.atomic_add(
            dgrad_tissue + velocity_row * atom_count + atom,
            flow_scale * curve_flow + heading * washout_scale * curve_washout,
        )


# ---------------------------------------------------------------------------
# Launchers.
# ---------------------------------------------------------------------------


def _tiles(layout: Any, state_count: int) -> dict[str, int]:
    """The tile shape and warps a launch compiles for.

    The widest intermediate is an operator times a tile of states, pools by
    pools by orders, and the warps are sized to hold it.
    """
    pools = triton.next_power_of_2(layout.longitudinal)
    states = triton.next_power_of_2(state_count)
    return {
        "n": layout.longitudinal,
        "m": layout.transverse,
        "blocks": layout.blocks,
        "P": pools,
        "S": states,
        "num_warps": max(1, min(8, pools * pools * states // 256)),
    }


def _inputs(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    pools: Any,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    tissue_tangents: tuple[torch.Tensor, ...] | None,
    event_tangents: tuple[torch.Tensor, ...] | None,
    dynamic_direction: Any,
) -> tuple[torch.Tensor, ...]:
    """The buffers both kernels open with, in their order.

    A buffer whose branch is compiled out is still an argument, so one the
    launch already holds stands in for it.
    """
    device = tissue[0].device
    voxel = tuple(tissue[index] for index in _VOXEL_INDEX)
    moving = (
        voxel
        if tissue_tangents is None
        else tuple(tissue_tangents[index] for index in _VOXEL_INDEX)
    )
    duration, kind, flip, phase = events[:4]
    stepping = (duration, flip, phase) if event_tangents is None else event_tangents
    pairs = duration if dynamic is None else dynamic.packed(device)
    return (
        *voxel,
        *moving,
        *events[:9],
        *stepping,
        pools.values,
        pools.values if pools.direction is None else pools.direction,
        pools.index,
        duration if profile is None else profile.packed(device),
        kind if profile is None else profile.rows(device),
        duration if lineshape is None else lineshape.packed(device),
        pairs,
        kind
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), kind.numel()).to(device),
        pairs if dynamic_direction is None else dynamic_direction.to(device),
    )


def _scalars(
    base: int,
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_count: int,
    state_count: int,
    pools: Any,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
) -> tuple[Any, ...]:
    """The scalar arguments both kernels take after their buffers."""
    return (
        base,
        tissue[0].numel(),
        events[1].numel(),
        output_count,
        state_count,
        pools.layout.rows,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        0 if lineshape is None else lineshape.bins,
    )


def _switches(
    tissue: tuple[torch.Tensor, ...],
    moving: tuple[torch.Tensor, ...] | None,
    pools: Any,
    profile: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    geometry: Geometry,
    following: bool,
) -> dict[str, Any]:
    """The constexpr switches both kernels take."""
    return {
        "atom_stride": _atom_stride(tissue)
        if moving is None
        else _atom_stride(tissue, moving),
        "shimmed": _shim_count(tissue) > 1,
        "profiled": profile is not None and profile.bins > 0,
        "dynamic": dynamic is not None,
        "directed_pairs": following and dynamic_direction is not None,
        "directed_table": following and pools.direction is not None,
        "following": following,
        **_feature_flags(features, geometry),
    }


def _forward(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...] | None,
    event_tangents: tuple[torch.Tensor, ...] | None,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    pools: Any,
) -> torch.Tensor:
    following = tissue_tangents is not None
    atoms = tissue[0].numel()
    total = _train_count(events) * atoms
    output_real = torch.zeros(
        _output_shape(_train_count(events), atoms, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.zeros_like(output_real)
    if total:
        _pooled_kernel[(total,)](
            *_inputs(
                tissue,
                events,
                pools,
                profile,
                lineshape,
                dynamic,
                tissue_tangents,
                event_tangents,
                dynamic_direction,
            ),
            output_real,
            output_imag,
            output_real,
            *_scalars(
                0,
                tissue,
                events,
                output_count,
                state_count,
                pools,
                geometry,
                profile,
                lineshape,
            ),  # fmt: skip
            planes=12 if following else 6,
            keep=False,
            **_switches(
                tissue,
                tissue_tangents,
                pools,
                profile,
                dynamic,
                dynamic_direction,
                features,
                geometry,
                following,
            ),
            **_tiles(pools.layout, state_count),
        )
    return torch.complex(output_real, output_imag)


def simulate(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> torch.Tensor:
    """The signals of a tissue whose pools are tabulated, on CUDA."""
    return _forward(
        tissue,
        events,
        None,
        None,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=None,
        features=features,
        pools=pools,
    )


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> torch.Tensor:
    """The signals' derivative along a direction, for a tabulated tissue.

    ``pools.direction`` is the direction along the tables, if they move.
    """
    return _forward(
        tissue,
        events,
        tissue_tangents,
        tuple(event_tangents),
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
        pools=pools,
    )


def _adjoint(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...] | None,
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    pools: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Both planes of the adjoint: the cotangents, then their derivative.

    Each plane holds the tissue's gradient rows, then duration, flip and
    phase, then the pair's and the tables' cotangents where there are any.
    The derivative plane is zero unless ``tangents`` gives a direction.
    """
    following = tangents is not None
    device = tissue[0].device
    atoms = tissue[0].numel()
    trains = _train_count(events)
    event_count = events[1].numel()
    total = trains * atoms
    shims = _shim_count(tissue)
    tissue_tangents = None if tangents is None else tangents[: len(TISSUE_NAMES)]
    event_tangents = (
        None
        if tangents is None
        else tuple(tangents[FLOAT_NAMES.index(name)] for name in _EVENT)
    )
    inputs = _inputs(
        tissue,
        events,
        pools,
        profile,
        lineshape,
        dynamic,
        tissue_tangents,
        event_tangents,
        dynamic_direction,
    )
    switches = _switches(
        tissue,
        tissue_tangents,
        pools,
        profile,
        dynamic,
        dynamic_direction,
        features,
        geometry,
        following,
    )
    tiles = _tiles(pools.layout, state_count)
    planes = 12 if following else 6
    held = planes * tiles["P"] * tiles["S"] * max(1, event_count)
    wave = max(1, min(total, _TRAJECTORY_BUDGET_BYTES // (4 * held)))
    trajectory = torch.empty(wave * held, dtype=torch.float32, device=device)

    grad_output = grad_output.resolve_conj()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()
    duration, _kind, flip, phase = events[:4]
    pairs = inputs[_PAIRS]

    def plane() -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                tissue_gradient_height(shims) * atoms,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros_like(duration),
            torch.zeros_like(flip),
            torch.zeros_like(phase),
            torch.zeros(
                (trains, *pools.values.shape), dtype=torch.float32, device=device
            ),
            torch.zeros_like(pairs),
        )

    value, tangent = plane(), plane()
    rows = tissue_gradient_bases(shims)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        scalars = _scalars(
            base, tissue, events, output_count, state_count, pools, geometry,
            profile, lineshape,
        )  # fmt: skip
        _pooled_kernel[(span,)](
            *inputs,
            grad_real,
            grad_imag,
            trajectory,
            *scalars,
            planes=planes,
            keep=True,
            **switches,
            **tiles,
        )
        _pooled_adjoint_kernel[(span,)](
            *inputs,
            grad_real,
            grad_imag,
            *(entry for pair in zip(value, tangent, strict=True) for entry in pair),
            trajectory,
            *scalars,
            *(rows[TISSUE_NAMES.index(name)] for name in _VOXEL),
            planes=planes,
            **switches,
            **tiles,
        )

    def gathered(side: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        voxel = tuple(
            side[0][start * atoms : (start + count) * atoms]
            for start, count in zip(rows, tissue_gradient_rows(shims), strict=True)
        )
        return (
            *voxel,
            side[1],
            side[2],
            side[3],
            *(() if dynamic is None else (side[5],)),
            side[4].sum(0),
        )

    return gathered(value), gathered(tangent)


# Where the pair sits among the buffers ``_inputs`` returns.
_PAIRS = 2 * len(_VOXEL) + 9 + 3 + 3 + 3

_EVENT = ("duration", "flip", "phase")


def simulate_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    dynamic: Any = None,
    lineshape: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint of a tabulated tissue, on CUDA.

    Returns the tissue's gradient rows, then duration, flip and phase, then the
    pair's cotangent where there is a pair and the tables' last.
    """
    value, _ = _adjoint(
        tissue,
        events,
        None,
        grad_output,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=None,
        features=features,
        pools=pools,
    )
    return value


def simulate_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through a tabulated tissue, on CUDA.

    Returns the gradients with respect to the primal inputs -- the derivative
    of the adjoint along ``tangents`` -- and then with respect to the tangent
    inputs, which is the adjoint itself.
    """
    value, tangent = _adjoint(
        tissue,
        events,
        tangents,
        grad_output,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
        pools=pools,
    )
    return tangent, value
