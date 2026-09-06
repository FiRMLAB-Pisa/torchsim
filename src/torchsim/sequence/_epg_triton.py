"""Fused Triton kernel for inference-only EPG state machines."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._accelerators import _shim_count, _train_count
from ._parameters import BOUND_POOL_INPUTS as _BOUND_POOL_INPUTS
from ._parameters import EXCHANGE_POOL_INPUTS as _EXCHANGE_POOL_INPUTS
from ._parameters import FLOAT_NAMES as _FLOAT_NAMES
from ._parameters import (
    NARROW_SPREAD,
    NO_GEOMETRY,
    Geometry,
    narrow_three_pool,
    three_pool_spread_rate,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import TISSUE_COUNT as _TISSUE_PARAMETERS
from ._parameters import TRANSMIT_INPUTS as _TRANSMIT_INPUTS
from ._parameters import (
    feature_flags as _feature_flags,
)

# Triton reads globals only through its own constexpr wrapper.
_TISSUE_COUNT = tl.constexpr(_TISSUE_PARAMETERS)

# How many tissue parameters the free pool alone accounts for. Both second
# pools' sit past them, and are written by the kernels that carry them; a
# single-pool run leaves their planes at the zero they were cleared to.
_FREE_POOL_COUNT = tl.constexpr(
    _TISSUE_PARAMETERS - len(_BOUND_POOL_INPUTS) - len(_EXCHANGE_POOL_INPUTS)
)
_BOUND_ROW = tl.constexpr(_BOUND_POOL_INPUTS[0])
_POOL_B_ROW = tl.constexpr(_EXCHANGE_POOL_INPUTS[0])

# The gradient plane holds a row of voxels per tissue parameter, except that
# the transmit pair holds one per shim. Both sit ahead of everything that
# widens, so a plane's row is its parameter index shifted by the rows the pair
# added ahead of it.
_B1_ROW = tl.constexpr(_TRANSMIT_INPUTS[0])
_B1_PHASE_ROW = tl.constexpr(_TRANSMIT_INPUTS[1])

# Where the two event directions the real-subspace adjoint follows sit among
# the differentiable inputs. Named rather than counted, so a tissue parameter
# added ahead of them moves them instead of silently renaming a neighbour.
_DURATION_SEED = _FLOAT_NAMES.index("duration")
_FLIP_SEED = _FLOAT_NAMES.index("flip")


@triton.jit
def _up(values, state):
    """``values`` moved one configuration order up: ``result[k] = values[k - 1]``.

    A shift along the state axis of a tile the program already holds. Order
    zero is left to the caller, which fills it from the sequence's own boundary
    condition rather than from a neighbour.
    """
    index = tl.broadcast_to(tl.maximum(state - 1, 0), values.shape)
    return tl.gather(values, index, 1)


@triton.jit
def _down(values, state):
    """``values`` moved one order down: ``result[k] = values[k + 1]``.

    The top order has no neighbour to read, so it reads itself and the caller
    masks it away.
    """
    index = tl.broadcast_to(tl.minimum(state + 1, values.shape[1] - 1), values.shape)
    return tl.gather(values, index, 1)


@triton.jit
def _first(values, state):
    """Order zero of ``values``, spread across every order."""
    index = tl.broadcast_to(state * 0, values.shape)
    return tl.gather(values, index, 1)


@triton.jit
def _event_value(values, event_base, event, active_atom, single_train: tl.constexpr):
    """One event's entry of a buffer carrying a row per train.

    ``duration``, ``flip`` and ``phase`` are indexed by the train and the event
    and never by the atom, so where there is one train the address is the same
    for every lane of the program and the value can be read once. Triton cannot
    see that through ``event_base``, and a tile-shaped load emits one
    instruction per element the lane holds -- four reads of one number.
    """
    if single_train:
        # Spread over the program's problems, which a jitted helper has to do
        # for itself: both arms of the branch have to hand back the one shape.
        return tl.load(values + event) + tl.zeros_like(event_base.to(tl.float32))
    return tl.load(values + event_base + event, mask=active_atom, other=0.0)


@triton.jit
def _shift(
    fplus_real,
    fplus_imag,
    fminus_real,
    fminus_imag,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    keep_up = (state > 0) & state_mask
    keep_down = (state + 1 < state_count) & state_mask
    plus_real = tl.where(keep_up, _up(fplus_real, state), 0.0)
    plus_imag = tl.where(keep_up, _up(fplus_imag, state), 0.0)
    minus_real = tl.where(keep_down, _down(fminus_real, state), 0.0)
    minus_imag = tl.where(keep_down, _down(fminus_imag, state), 0.0)
    plus_real = tl.where(state == 0, minus_real, plus_real)
    plus_imag = tl.where(state == 0, -minus_imag, plus_imag)
    return plus_real, plus_imag, minus_real, minus_imag


@triton.jit
def _shift_real(
    plus,
    minus,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    shifted_plus = tl.where((state > 0) & state_mask, _up(plus, state), 0.0)
    shifted_minus = tl.where(
        (state + 1 < state_count) & state_mask, _down(minus, state), 0.0
    )
    return tl.where(state == 0, -shifted_minus, shifted_plus), shifted_minus


# ---------------------------------------------------------------------------
# Dual complex arithmetic.
#
# A dual complex number is four planes: the real and imaginary parts of the
# value, then of the tangent. Triton has no structs, so every quantity travels
# as four separate registers and these helpers keep the bookkeeping in one
# place rather than spread through the kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _damping(rate, dt, order):
    """Longitudinal and transverse diffusion damping for one interval.

    ``rate`` already carries the sequence's gradient geometry, so an interval's
    b-factor is that rate times its duration. Order zero has no longitudinal
    weight, which is what keeps the recovery term undamped.
    """
    b_factor = rate * dt
    squared = order * order
    return (
        tl.exp(-b_factor * squared),
        tl.exp(-b_factor * (squared + order + 0.3333333333333333)),
    )


@triton.jit
def _flow(rate, dt, order):
    """Phase each dephasing order turns through over one interval.

    ``rate`` already carries the sequence's gradient geometry, so it is the
    winding per unit order per second: a longitudinal state at order l turns
    through ``l * rate * dt``. The transverse states sit half an order further
    along the gradient, which is where the extra half turn comes from. Order
    zero is left alone while longitudinal, so the recovery term is unaffected.
    """
    turn = rate * dt
    return -order * turn, -(order + 0.5) * turn


@triton.jit
def _washout(rate, dt):
    """The fraction of a voxel's spins that stay put over one interval.

    Inflowing spins are taken to be fully relaxed and unexcited, which makes
    washout an affine map of the shape longitudinal recovery already has:

        wout * (Z * e1 + (1 - e1)) + win  ==  Z * (e1 * wout) + (1 - e1 * wout)

    so scaling both relaxation factors by it carries the whole term. Clamped at
    one, past which the interval has replaced the voxel outright.
    """
    return 1.0 - tl.minimum(rate * dt, 1.0)


@triton.jit
def _two_pool_step(r1_free, r1_bound, exchange, bound, dt, attenuation):
    """The two-pool longitudinal operator over one interval, and its recovery.

    ``expm((K - diag(R1)) t)`` in the exact 2x2 closed form. Its discriminant
    is a square plus a product of two non-negative rates, so the root is real
    and the branch a general exponential would need does not exist here.
    ``sinh(d)/d`` is taken by series near the origin, where the root has no
    derivative of its own.

    The equilibrium each pool relaxes toward is its own fraction, so the
    recovery is ``(I - E1) (1 - f, f)`` and needs no solve. Returned as
    ``(e11, e12, e21, e22, recovery_free, recovery_bound)``.
    """
    free = 1.0 - bound
    kab = exchange * bound
    kba = exchange * free
    l11 = (-kab - r1_free) * dt
    l12 = kba * dt
    l21 = kab * dt
    l22 = (-kba - r1_bound) * dt

    half_trace = 0.5 * (l11 + l22)
    half_gap = 0.5 * (l11 - l22)
    square = half_gap * half_gap + l12 * l21
    # tau +/- d are the eigenvalues, both non-positive for a decaying system,
    # so their exponentials are bounded by one. Formed that way rather than as
    # e^tau cosh(d), which over a long interval is an underflow times an
    # overflow.
    root = tl.sqrt(tl.maximum(square, 0.0))
    upper = tl.exp(half_trace + root)
    lower = tl.exp(half_trace - root)
    cosine = 0.5 * (upper + lower)
    turning = square > 1e-12
    guarded = tl.where(turning, root, 1.0)
    scale = tl.where(
        turning,
        0.5 * (upper - lower) / guarded,
        tl.exp(half_trace) * (1.0 + square / 6.0 + square * square / 120.0),
    )
    e11 = attenuation * (cosine + scale * half_gap)
    e12 = attenuation * scale * l12
    e21 = attenuation * scale * l21
    e22 = attenuation * (cosine - scale * half_gap)
    return (
        e11,
        e12,
        e21,
        e22,
        free - (e11 * free + e12 * bound),
        bound - (e21 * free + e22 * bound),
    )


# The three-pool longitudinal step is the one operator the state machine forms
# in double. A 2x2's closed form loses accuracy like the interval; a 3x3's
# loses it like the square of it, because the answer's entries are order one
# while the terms that build them are order |L dt|^2. That is intrinsic to
# writing the answer as a polynomial in the generator, so it is met with
# precision rather than with rearrangement. It is formed once per interval, not
# once per dephasing order, so the cost stays out of the state loop.
_SPREAD_CUT = tl.constexpr(1.0)
_SINCH_CUT = tl.constexpr(1e-4)
# Where two roots meet the sorted roots have a vertical tangent the operator
# itself does not, so the arc cosine is held a hair off its endpoints.
_ARG_LIMIT = tl.constexpr(1.0 - 1e-16)
_TURN_THIRD = tl.constexpr(2.09439510239319549231)


@triton.jit
def _exp_difference(lower, upper, exp_lower, exp_upper):
    """``[a, b] exp``, from exponentials the caller has already taken.

    Near the coalescence ``sinh(d)/d`` is even in the gap, so the series is a
    polynomial in its square; the exponential of the midpoint is reached from
    the lower one by a series too, because over a gap this small it is one.
    """
    half = 0.5 * (upper - lower)
    near = tl.abs(half) < _SINCH_CUT
    square = half * half
    # exp(mid) * sinh(half)/half, with both factors expanded about zero.
    series = exp_lower * (1.0 + half + 0.5 * square) * (1.0 + square / 6.0)
    gap = tl.where(near, 1.0, upper - lower)
    return tl.where(near, series, (exp_upper - exp_lower) / gap)


@triton.jit
def _three_pool_recovery(
    e00, e01, e02, e10, e11, e12, e20, e21, e22, free, pool_b, pool_c
):
    """What each pool recovers over the interval, beside the operator itself.

    Returns the nine entries and the three recoveries, narrowed to float32
    once they are an operator.
    """
    grow_free = free - (e00 * free + e01 * pool_b + e02 * pool_c)
    grow_pool_b = pool_b - (e10 * free + e11 * pool_b + e12 * pool_c)
    grow_bound = pool_c - (e20 * free + e21 * pool_b + e22 * pool_c)
    return (
        e00.to(tl.float32),
        e01.to(tl.float32),
        e02.to(tl.float32),
        e10.to(tl.float32),
        e11.to(tl.float32),
        e12.to(tl.float32),
        e20.to(tl.float32),
        e21.to(tl.float32),
        e22.to(tl.float32),
        grow_free.to(tl.float32),
        grow_pool_b.to(tl.float32),
        grow_bound.to(tl.float32),
    )


@triton.jit
def _three_pool_step(
    r1_free,
    r1_pool_b,
    r1_bound,
    exchange_b,
    exchange_c,
    fraction_b,
    fraction_c,
    dt,
    attenuation,
    narrow: tl.constexpr = False,
):
    """``expm((K - diag(R1)) t)`` for free water beside both second pools.

    Free water is pool a, the chemically exchanging pool b and the semisolid
    pool c; each second pool exchanges with the free water and not with the
    other. Returns the nine entries and the three recoveries, narrowed to
    float32 once they are an operator.

    Two branches, by how far apart the eigenvalues are. Where they are close
    the exponential's own series is reduced modulo the characteristic
    polynomial, which forms no root at all. Where they are far apart the
    interpolating polynomial is taken in Newton form at the three roots, each
    of which is non-positive, so a long interval cannot overflow.

    ``narrow`` says the caller has bounded the spread below
    :data:`torchsim.sequence._parameters.NARROW_SPREAD` for every voxel and
    every interval it will pass, so
    only the series can be reached. The roots then cost nothing, and the series
    holds the answer to float32 without being carried in double --
    :func:`torchsim.sequence._parameters.narrow_three_pool` is what decides it.
    """
    work: tl.constexpr = tl.float32 if narrow else tl.float64
    terms: tl.constexpr = 24 if narrow else 16
    step = dt.to(work)
    free = (1.0 - fraction_b - fraction_c).to(work)
    pool_b = fraction_b.to(work)
    pool_c = fraction_c.to(work)
    kab = exchange_b.to(work) * pool_b
    kba = exchange_b.to(work) * free
    kac = exchange_c.to(work) * pool_c
    kca = exchange_c.to(work) * free
    a00 = (-kab - kac - r1_free.to(work)) * step
    a01 = kba * step
    a02 = kca * step
    a10 = kab * step
    a11 = (-kba - r1_pool_b.to(work)) * step
    a20 = kac * step
    a22 = (-kca - r1_bound.to(work)) * step

    third = (a00 + a11 + a22) / 3.0
    s00 = a00 - third
    s11 = a11 - third
    s22 = a22 - third
    # The two second pools do not exchange, so the generator keeps a pair of
    # structural zeros the products below are written around.
    minors = s00 * s11 - a01 * a10 + s00 * s22 - a02 * a20 + s11 * s22
    determinant = s00 * s11 * s22 - a01 * (a10 * s22) + a02 * (-s11 * a20)

    # --- close together: the series reduced modulo x^3 + minors x - det ---
    flat = 1.0 + 0.0 * third
    linear = 0.0 * third
    square = 0.0 * third
    sum_flat = flat
    sum_linear = linear
    sum_square = square
    factorial = 1.0
    for order in tl.static_range(1, terms):
        next_flat = square * determinant
        next_linear = flat - square * minors
        next_square = linear
        flat = next_flat
        linear = next_linear
        square = next_square
        factorial = factorial * order
        weight = 1.0 / factorial
        sum_flat = sum_flat + weight * flat
        sum_linear = sum_linear + weight * linear
        sum_square = sum_square + weight * square
    q00 = s00 * s00 + a01 * a10 + a02 * a20
    q01 = s00 * a01 + a01 * s11
    q02 = s00 * a02 + a02 * s22
    q10 = a10 * s00 + s11 * a10
    q11 = a10 * a01 + s11 * s11
    q12 = a10 * a02
    q20 = a20 * s00 + s22 * a20
    q21 = a20 * a01
    q22 = a20 * a02 + s22 * s22
    lift = tl.exp(third)
    c00 = lift * (sum_flat + sum_linear * s00 + sum_square * q00)
    c01 = lift * (sum_linear * a01 + sum_square * q01)
    c02 = lift * (sum_linear * a02 + sum_square * q02)
    c10 = lift * (sum_linear * a10 + sum_square * q10)
    c11 = lift * (sum_flat + sum_linear * s11 + sum_square * q11)
    c12 = lift * (sum_square * q12)
    c20 = lift * (sum_linear * a20 + sum_square * q20)
    c21 = lift * (sum_square * q21)
    c22 = lift * (sum_flat + sum_linear * s22 + sum_square * q22)

    # --- far apart: the Newton form at the three roots ---
    damp = attenuation.to(work)
    if narrow:
        e00 = damp * c00
        e01 = damp * c01
        e02 = damp * c02
        e10 = damp * c10
        e11 = damp * c11
        e12 = damp * c12
        e20 = damp * c20
        e21 = damp * c21
        e22 = damp * c22
        return _three_pool_recovery(
            e00, e01, e02, e10, e11, e12, e20, e21, e22, free, pool_b, pool_c
        )
    radius = tl.sqrt(tl.maximum(-minors * (1.0 / 3.0), 1e-300))
    argument = tl.minimum(
        tl.maximum(0.5 * determinant / (radius * radius * radius), -_ARG_LIMIT),
        _ARG_LIMIT,
    )
    angle = libdevice.acos(argument) / 3.0
    root_a = 2.0 * radius * tl.cos(angle) + third
    root_b = 2.0 * radius * tl.cos(angle - _TURN_THIRD) + third
    root_c = 2.0 * radius * tl.cos(angle - 2.0 * _TURN_THIRD) + third
    low = tl.minimum(tl.minimum(root_a, root_b), root_c)
    high = tl.maximum(tl.maximum(root_a, root_b), root_c)
    middle = tl.maximum(
        tl.minimum(root_a, root_b), tl.minimum(tl.maximum(root_a, root_b), root_c)
    )
    # Three exponentials serve every divided difference between them.
    leading = tl.exp(low)
    centre = tl.exp(middle)
    trailing = tl.exp(high)
    first = _exp_difference(low, middle, leading, centre)
    span = high - low
    second = (_exp_difference(middle, high, centre, trailing) - first) / tl.where(
        span > 0.0, span, 1.0
    )
    m00 = a00 - low
    m11 = a11 - low
    m22 = a22 - low
    n00 = a00 - middle
    n11 = a11 - middle
    n22 = a22 - middle
    p00 = m00 * n00 + a01 * a10 + a02 * a20
    p01 = m00 * a01 + a01 * n11
    p02 = m00 * a02 + a02 * n22
    p10 = a10 * n00 + m11 * a10
    p11 = a10 * a01 + m11 * n11
    p12 = a10 * a02
    p20 = a20 * n00 + m22 * a20
    p21 = a20 * a01
    p22 = a20 * a02 + m22 * n22
    d00 = leading + first * m00 + second * p00
    d01 = first * a01 + second * p01
    d02 = first * a02 + second * p02
    d10 = first * a10 + second * p10
    d11 = leading + first * m11 + second * p11
    d12 = second * p12
    d20 = first * a20 + second * p20
    d21 = second * p21
    d22 = leading + first * m22 + second * p22

    # The shifted roots sum to zero, so the sum of their squares is -2 * minors
    # and none is larger than the root of that.
    close = -2.0 * minors < _SPREAD_CUT * _SPREAD_CUT
    e00 = damp * tl.where(close, c00, d00)
    e01 = damp * tl.where(close, c01, d01)
    e02 = damp * tl.where(close, c02, d02)
    e10 = damp * tl.where(close, c10, d10)
    e11 = damp * tl.where(close, c11, d11)
    e12 = damp * tl.where(close, c12, d12)
    e20 = damp * tl.where(close, c20, d20)
    e21 = damp * tl.where(close, c21, d21)
    e22 = damp * tl.where(close, c22, d22)
    return _three_pool_recovery(
        e00, e01, e02, e10, e11, e12, e20, e21, e22, free, pool_b, pool_c
    )


@triton.jit
def _three_pool_from_table(
    table, row, atom, voxel_count, mask, attenuation, free, pool_b, pool_c
):
    """Read one interval's three-pool operator, and what each pool recovers.

    The stored row is undamped, so the washout the event carries is applied
    here and the three recoveries follow from the damped entries -- which is
    what makes one row serve every event of the same length whatever its
    washout.
    """
    base = table + row * (9 * voxel_count) + atom
    return _three_pool_recovery(
        attenuation * tl.load(base + 0 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 1 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 2 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 3 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 4 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 5 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 6 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 7 * voxel_count, mask=mask, other=0.0),
        attenuation * tl.load(base + 8 * voxel_count, mask=mask, other=0.0),
        free,
        pool_b,
        pool_c,
    )


@triton.jit
def _three_pool_from_table_jvp(
    table,
    row,
    atom,
    voxel_count,
    mask,
    r1_free,
    r1_pool_b,
    r1_bound,
    exchange_b,
    exchange_c,
    fraction_b,
    d_fraction_b,
    fraction_c,
    d_fraction_c,
    d_dt,
    attenuation,
    d_attenuation,
):
    """Read one interval's three-pool operator and a direction through it.

    The row carries the tissue's share of the direction; the interval's own is
    ``A1 C d_dt``, formed here because ``d_dt`` belongs to the event and the
    row is shared. Returns the nine entries and three recoveries with their
    tangents, in the order :func:`_three_pool_step_jvp` returns them.
    """
    free = 1.0 - fraction_b - fraction_c
    d_free = -d_fraction_b - d_fraction_c
    a00 = -exchange_b * fraction_b - exchange_c * fraction_c - r1_free
    a01 = exchange_b * free
    a02 = exchange_c * free
    a10 = exchange_b * fraction_b
    a11 = -exchange_b * free - r1_pool_b
    a20 = exchange_c * fraction_c
    a22 = -exchange_c * free - r1_bound
    base = table + row * (18 * voxel_count) + atom
    c00 = tl.load(base + 0 * voxel_count, mask=mask, other=0.0)
    c01 = tl.load(base + 1 * voxel_count, mask=mask, other=0.0)
    c02 = tl.load(base + 2 * voxel_count, mask=mask, other=0.0)
    c10 = tl.load(base + 3 * voxel_count, mask=mask, other=0.0)
    c11 = tl.load(base + 4 * voxel_count, mask=mask, other=0.0)
    c12 = tl.load(base + 5 * voxel_count, mask=mask, other=0.0)
    c20 = tl.load(base + 6 * voxel_count, mask=mask, other=0.0)
    c21 = tl.load(base + 7 * voxel_count, mask=mask, other=0.0)
    c22 = tl.load(base + 8 * voxel_count, mask=mask, other=0.0)
    # The row's tangent, plus what the event's own interval direction adds.
    t00 = tl.load(base + 9 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a00 * c00 + a01 * c10 + a02 * c20
    )
    t01 = tl.load(base + 10 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a00 * c01 + a01 * c11 + a02 * c21
    )
    t02 = tl.load(base + 11 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a00 * c02 + a01 * c12 + a02 * c22
    )
    t10 = tl.load(base + 12 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a10 * c00 + a11 * c10
    )
    t11 = tl.load(base + 13 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a10 * c01 + a11 * c11
    )
    t12 = tl.load(base + 14 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a10 * c02 + a11 * c12
    )
    t20 = tl.load(base + 15 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a20 * c00 + a22 * c20
    )
    t21 = tl.load(base + 16 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a20 * c01 + a22 * c21
    )
    t22 = tl.load(base + 17 * voxel_count, mask=mask, other=0.0) + d_dt * (
        a20 * c02 + a22 * c22
    )
    e00 = attenuation * c00
    e01 = attenuation * c01
    e02 = attenuation * c02
    e10 = attenuation * c10
    e11 = attenuation * c11
    e12 = attenuation * c12
    e20 = attenuation * c20
    e21 = attenuation * c21
    e22 = attenuation * c22
    f00 = d_attenuation * c00 + attenuation * t00
    f01 = d_attenuation * c01 + attenuation * t01
    f02 = d_attenuation * c02 + attenuation * t02
    f10 = d_attenuation * c10 + attenuation * t10
    f11 = d_attenuation * c11 + attenuation * t11
    f12 = d_attenuation * c12 + attenuation * t12
    f20 = d_attenuation * c20 + attenuation * t20
    f21 = d_attenuation * c21 + attenuation * t21
    f22 = d_attenuation * c22 + attenuation * t22
    # The equilibrium the recoveries are taken against moves with the
    # fractions, so it carries a direction of its own.
    grow_free = free - (e00 * free + e01 * fraction_b + e02 * fraction_c)
    grow_pool_b = fraction_b - (e10 * free + e11 * fraction_b + e12 * fraction_c)
    grow_bound = fraction_c - (e20 * free + e21 * fraction_b + e22 * fraction_c)
    d_grow_free = d_free - (
        f00 * free
        + f01 * fraction_b
        + f02 * fraction_c
        + e00 * d_free
        + e01 * d_fraction_b
        + e02 * d_fraction_c
    )
    d_grow_pool_b = d_fraction_b - (
        f10 * free
        + f11 * fraction_b
        + f12 * fraction_c
        + e10 * d_free
        + e11 * d_fraction_b
        + e12 * d_fraction_c
    )
    d_grow_bound = d_fraction_c - (
        f20 * free
        + f21 * fraction_b
        + f22 * fraction_c
        + e20 * d_free
        + e21 * d_fraction_b
        + e22 * d_fraction_c
    )
    return (
        e00,
        e01,
        e02,
        e10,
        e11,
        e12,
        e20,
        e21,
        e22,
        grow_free,
        grow_pool_b,
        grow_bound,
        f00,
        f01,
        f02,
        f10,
        f11,
        f12,
        f20,
        f21,
        f22,
        d_grow_free,
        d_grow_pool_b,
        d_grow_bound,
    )


@triton.jit
def _three_pool_table_jvp_kernel(
    t1,
    t1_pool_b,
    t1_bound,
    pool_b_exchange,
    bound_exchange,
    pool_b_fraction,
    bound_fraction,
    d_t1,
    d_t1_pool_b,
    d_t1_bound,
    d_pool_b_exchange,
    d_bound_exchange,
    d_pool_b_fraction,
    d_bound_fraction,
    durations,
    rows,
    table,
    voxel_count,
    BLOCK: tl.constexpr,
    narrow: tl.constexpr,
):
    """Fill one row of the three-pool operator table, value and direction.

    The row holds nine undamped entries and the nine a direction through the
    tissue gives them, both at ``d_dt`` of zero -- the interval's own share of
    the direction is ``A1 C d_dt``, which the reading event adds because
    ``d_dt`` is its own and the row's is not. Laid out ``(rows, 18, voxels)``,
    the tangent following the value.
    """
    row = tl.load(rows + tl.program_id(0))
    atom = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = atom < voxel_count
    dt = tl.load(durations + row)
    nil = 0.0 * dt
    value_t1 = tl.load(t1 + atom, mask=live, other=1.0)
    value_t1b = tl.load(t1_pool_b + atom, mask=live, other=1.0)
    value_t1c = tl.load(t1_bound + atom, mask=live, other=1.0)
    (
        e00,
        e01,
        e02,
        e10,
        e11,
        e12,
        e20,
        e21,
        e22,
        _,
        _,
        _,
        d00,
        d01,
        d02,
        d10,
        d11,
        d12,
        d20,
        d21,
        d22,
        _,
        _,
        _,
    ) = _three_pool_step_jvp(
        1000.0 / value_t1,
        -1000.0 * tl.load(d_t1 + atom, mask=live, other=0.0) / (value_t1 * value_t1),
        1000.0 / value_t1b,
        -1000.0
        * tl.load(d_t1_pool_b + atom, mask=live, other=0.0)
        / (value_t1b * value_t1b),
        1000.0 / value_t1c,
        -1000.0
        * tl.load(d_t1_bound + atom, mask=live, other=0.0)
        / (value_t1c * value_t1c),
        tl.load(pool_b_exchange + atom, mask=live, other=0.0),
        tl.load(d_pool_b_exchange + atom, mask=live, other=0.0),
        tl.load(bound_exchange + atom, mask=live, other=0.0),
        tl.load(d_bound_exchange + atom, mask=live, other=0.0),
        tl.load(pool_b_fraction + atom, mask=live, other=0.0),
        tl.load(d_pool_b_fraction + atom, mask=live, other=0.0),
        tl.load(bound_fraction + atom, mask=live, other=0.0),
        tl.load(d_bound_fraction + atom, mask=live, other=0.0),
        dt,
        nil,
        1.0 + nil,
        nil,
        narrow,
    )
    base = table + row * (18 * voxel_count) + atom
    tl.store(base + 0 * voxel_count, e00, mask=live)
    tl.store(base + 1 * voxel_count, e01, mask=live)
    tl.store(base + 2 * voxel_count, e02, mask=live)
    tl.store(base + 3 * voxel_count, e10, mask=live)
    tl.store(base + 4 * voxel_count, e11, mask=live)
    tl.store(base + 5 * voxel_count, e12, mask=live)
    tl.store(base + 6 * voxel_count, e20, mask=live)
    tl.store(base + 7 * voxel_count, e21, mask=live)
    tl.store(base + 8 * voxel_count, e22, mask=live)
    tl.store(base + 9 * voxel_count, d00, mask=live)
    tl.store(base + 10 * voxel_count, d01, mask=live)
    tl.store(base + 11 * voxel_count, d02, mask=live)
    tl.store(base + 12 * voxel_count, d10, mask=live)
    tl.store(base + 13 * voxel_count, d11, mask=live)
    tl.store(base + 14 * voxel_count, d12, mask=live)
    tl.store(base + 15 * voxel_count, d20, mask=live)
    tl.store(base + 16 * voxel_count, d21, mask=live)
    tl.store(base + 17 * voxel_count, d22, mask=live)


@triton.jit
def _three_pool_contract(
    x00,
    x01,
    x02,
    x10,
    x11,
    x12,
    x20,
    x21,
    x22,
    e11,
    e12,
    e13,
    e21,
    e22,
    e23,
    e31,
    e32,
    e33,
    rec_free,
    rec_pool_b,
    rec_bound,
    free,
    fraction_b,
    fraction_c,
):
    """Nine cotangents against nine entries, less what the recoveries take.

    A recovery is ``m - E m`` with ``m`` the equilibrium
    ``(free, fraction_b, fraction_c)``, so it differentiates through the same
    nine entries with the equilibrium contracted out of them.
    """
    return (
        e11 * x00
        + e12 * x01
        + e13 * x02
        + e21 * x10
        + e22 * x11
        + e23 * x12
        + e31 * x20
        + e32 * x21
        + e33 * x22
        - rec_free * (x00 * free + x01 * fraction_b + x02 * fraction_c)
        - rec_pool_b * (x10 * free + x11 * fraction_b + x12 * fraction_c)
        - rec_bound * (x20 * free + x21 * fraction_b + x22 * fraction_c)
    )


@triton.jit
def _three_pool_interval_adjoint(
    table,
    row,
    atom,
    voxel_count,
    mask,
    r1_free,
    r1_pool_b,
    r1_bound,
    exchange_b,
    exchange_c,
    fraction_b,
    fraction_c,
    attenuation,
    b11,
    b12,
    b13,
    b21,
    b22,
    b23,
    b31,
    b32,
    b33,
    bfree,
    bpool_b,
    bbound,
):
    """What one interval's cotangents give its length and its attenuation.

    The generator is proportional to the interval, so ``dE/d(dt) == A1 E`` and
    the length's gradient needs the operator and 27 multiplies rather than the
    eigenvalues -- which is what lets every other gradient be pooled over the
    events that share a length while this one stays per event.

    Returns the two contractions, in the order
    :func:`_three_pool_step_adjoint_jvp` returns them.
    """
    free = 1.0 - fraction_b - fraction_c
    a00 = -exchange_b * fraction_b - exchange_c * fraction_c - r1_free
    a01 = exchange_b * free
    a02 = exchange_c * free
    a10 = exchange_b * fraction_b
    a11 = -exchange_b * free - r1_pool_b
    a20 = exchange_c * fraction_c
    a22 = -exchange_c * free - r1_bound
    base = table + row * (9 * voxel_count) + atom
    c00 = tl.load(base + 0 * voxel_count, mask=mask, other=0.0)
    c01 = tl.load(base + 1 * voxel_count, mask=mask, other=0.0)
    c02 = tl.load(base + 2 * voxel_count, mask=mask, other=0.0)
    c10 = tl.load(base + 3 * voxel_count, mask=mask, other=0.0)
    c11 = tl.load(base + 4 * voxel_count, mask=mask, other=0.0)
    c12 = tl.load(base + 5 * voxel_count, mask=mask, other=0.0)
    c20 = tl.load(base + 6 * voxel_count, mask=mask, other=0.0)
    c21 = tl.load(base + 7 * voxel_count, mask=mask, other=0.0)
    c22 = tl.load(base + 8 * voxel_count, mask=mask, other=0.0)
    # A1 C, the second and third rows of A1 having no entry off their own pool.
    p00 = a00 * c00 + a01 * c10 + a02 * c20
    p01 = a00 * c01 + a01 * c11 + a02 * c21
    p02 = a00 * c02 + a01 * c12 + a02 * c22
    p10 = a10 * c00 + a11 * c10
    p11 = a10 * c01 + a11 * c11
    p12 = a10 * c02 + a11 * c12
    p20 = a20 * c00 + a22 * c20
    p21 = a20 * c01 + a22 * c21
    p22 = a20 * c02 + a22 * c22
    grad_dt = attenuation * _three_pool_contract(
        p00,
        p01,
        p02,
        p10,
        p11,
        p12,
        p20,
        p21,
        p22,
        b11,
        b12,
        b13,
        b21,
        b22,
        b23,
        b31,
        b32,
        b33,
        bfree,
        bpool_b,
        bbound,
        free,
        fraction_b,
        fraction_c,
    )
    grad_att = _three_pool_contract(
        c00,
        c01,
        c02,
        c10,
        c11,
        c12,
        c20,
        c21,
        c22,
        b11,
        b12,
        b13,
        b21,
        b22,
        b23,
        b31,
        b32,
        b33,
        bfree,
        bpool_b,
        bbound,
        free,
        fraction_b,
        fraction_c,
    )
    return grad_dt, grad_att


@triton.jit
def _three_pool_interval_adjoint_jvp(
    table,
    row,
    atom,
    voxel_count,
    mask,
    r1_free,
    d_r1_free,
    r1_pool_b,
    d_r1_pool_b,
    r1_bound,
    d_r1_bound,
    exchange_b,
    d_exchange_b,
    exchange_c,
    d_exchange_c,
    fraction_b,
    d_fraction_b,
    fraction_c,
    d_fraction_c,
    d_dt,
    attenuation,
    d_attenuation,
    b11,
    b12,
    b13,
    b21,
    b22,
    b23,
    b31,
    b32,
    b33,
    bfree,
    bpool_b,
    bbound,
    t11,
    t12,
    t13,
    t21,
    t22,
    t23,
    t31,
    t32,
    t33,
    tfree,
    tpool_b,
    tbound,
):
    """The interval and the attenuation, from a dual pair's cotangents.

    ``dE/d(dt)`` is ``A1 E``, and the direction that quantity carries follows
    from the same generator: with ``C_dot == C_row + A1 C d_dt``, the
    derivative in the interval is ``A1_dot C + A1 C_row + A1 A1 C d_dt``. So
    three products of the generator against the tabulated row serve what the
    eigenvalues would otherwise be re-formed for, and these two quantities are
    the only ones that stay per event.

    Returns the interval and attenuation gradients, value then tangent, in the
    order :func:`_three_pool_step_adjoint_jvp` returns them.
    """
    free = 1.0 - fraction_b - fraction_c
    d_free = -d_fraction_b - d_fraction_c
    a00 = -exchange_b * fraction_b - exchange_c * fraction_c - r1_free
    a01 = exchange_b * free
    a02 = exchange_c * free
    a10 = exchange_b * fraction_b
    a11 = -exchange_b * free - r1_pool_b
    a20 = exchange_c * fraction_c
    a22 = -exchange_c * free - r1_bound
    da00 = (
        -d_exchange_b * fraction_b
        - exchange_b * d_fraction_b
        - d_exchange_c * fraction_c
        - exchange_c * d_fraction_c
        - d_r1_free
    )
    da01 = d_exchange_b * free + exchange_b * d_free
    da02 = d_exchange_c * free + exchange_c * d_free
    da10 = d_exchange_b * fraction_b + exchange_b * d_fraction_b
    da11 = -d_exchange_b * free - exchange_b * d_free - d_r1_pool_b
    da20 = d_exchange_c * fraction_c + exchange_c * d_fraction_c
    da22 = -d_exchange_c * free - exchange_c * d_free - d_r1_bound
    base = table + row * (18 * voxel_count) + atom
    c00 = tl.load(base + 0 * voxel_count, mask=mask, other=0.0)
    c01 = tl.load(base + 1 * voxel_count, mask=mask, other=0.0)
    c02 = tl.load(base + 2 * voxel_count, mask=mask, other=0.0)
    c10 = tl.load(base + 3 * voxel_count, mask=mask, other=0.0)
    c11 = tl.load(base + 4 * voxel_count, mask=mask, other=0.0)
    c12 = tl.load(base + 5 * voxel_count, mask=mask, other=0.0)
    c20 = tl.load(base + 6 * voxel_count, mask=mask, other=0.0)
    c21 = tl.load(base + 7 * voxel_count, mask=mask, other=0.0)
    c22 = tl.load(base + 8 * voxel_count, mask=mask, other=0.0)
    r00 = tl.load(base + 9 * voxel_count, mask=mask, other=0.0)
    r01 = tl.load(base + 10 * voxel_count, mask=mask, other=0.0)
    r02 = tl.load(base + 11 * voxel_count, mask=mask, other=0.0)
    r10 = tl.load(base + 12 * voxel_count, mask=mask, other=0.0)
    r11 = tl.load(base + 13 * voxel_count, mask=mask, other=0.0)
    r12 = tl.load(base + 14 * voxel_count, mask=mask, other=0.0)
    r20 = tl.load(base + 15 * voxel_count, mask=mask, other=0.0)
    r21 = tl.load(base + 16 * voxel_count, mask=mask, other=0.0)
    r22 = tl.load(base + 17 * voxel_count, mask=mask, other=0.0)
    # P = A1 C, Q = A1_dot C + A1 C_row, S = A1 P.
    p00 = a00 * c00 + a01 * c10 + a02 * c20
    p01 = a00 * c01 + a01 * c11 + a02 * c21
    p02 = a00 * c02 + a01 * c12 + a02 * c22
    p10 = a10 * c00 + a11 * c10
    p11 = a10 * c01 + a11 * c11
    p12 = a10 * c02 + a11 * c12
    p20 = a20 * c00 + a22 * c20
    p21 = a20 * c01 + a22 * c21
    p22 = a20 * c02 + a22 * c22
    q00 = da00 * c00 + da01 * c10 + da02 * c20 + a00 * r00 + a01 * r10 + a02 * r20
    q01 = da00 * c01 + da01 * c11 + da02 * c21 + a00 * r01 + a01 * r11 + a02 * r21
    q02 = da00 * c02 + da01 * c12 + da02 * c22 + a00 * r02 + a01 * r12 + a02 * r22
    q10 = da10 * c00 + da11 * c10 + a10 * r00 + a11 * r10
    q11 = da10 * c01 + da11 * c11 + a10 * r01 + a11 * r11
    q12 = da10 * c02 + da11 * c12 + a10 * r02 + a11 * r12
    q20 = da20 * c00 + da22 * c20 + a20 * r00 + a22 * r20
    q21 = da20 * c01 + da22 * c21 + a20 * r01 + a22 * r21
    q22 = da20 * c02 + da22 * c22 + a20 * r02 + a22 * r22
    s00 = a00 * p00 + a01 * p10 + a02 * p20
    s01 = a00 * p01 + a01 * p11 + a02 * p21
    s02 = a00 * p02 + a01 * p12 + a02 * p22
    s10 = a10 * p00 + a11 * p10
    s11 = a10 * p01 + a11 * p11
    s12 = a10 * p02 + a11 * p12
    s20 = a20 * p00 + a22 * p20
    s21 = a20 * p01 + a22 * p21
    s22 = a20 * p02 + a22 * p22
    # The direction the tabulated operator carries, and the interval's own
    # share of it.
    d00 = r00 + p00 * d_dt
    d01 = r01 + p01 * d_dt
    d02 = r02 + p02 * d_dt
    d10 = r10 + p10 * d_dt
    d11 = r11 + p11 * d_dt
    d12 = r12 + p12 * d_dt
    d20 = r20 + p20 * d_dt
    d21 = r21 + p21 * d_dt
    d22 = r22 + p22 * d_dt
    # dE/d(dt) with the attenuation held, and the direction that carries.
    g00 = attenuation * p00
    g01 = attenuation * p01
    g02 = attenuation * p02
    g10 = attenuation * p10
    g11 = attenuation * p11
    g12 = attenuation * p12
    g20 = attenuation * p20
    g21 = attenuation * p21
    g22 = attenuation * p22
    w00 = d_attenuation * p00 + attenuation * (q00 + s00 * d_dt)
    w01 = d_attenuation * p01 + attenuation * (q01 + s01 * d_dt)
    w02 = d_attenuation * p02 + attenuation * (q02 + s02 * d_dt)
    w10 = d_attenuation * p10 + attenuation * (q10 + s10 * d_dt)
    w11 = d_attenuation * p11 + attenuation * (q11 + s11 * d_dt)
    w12 = d_attenuation * p12 + attenuation * (q12 + s12 * d_dt)
    w20 = d_attenuation * p20 + attenuation * (q20 + s20 * d_dt)
    w21 = d_attenuation * p21 + attenuation * (q21 + s21 * d_dt)
    w22 = d_attenuation * p22 + attenuation * (q22 + s22 * d_dt)
    # This kernel carries every quantity as a dual pair, so the tangent
    # returned beside a gradient is that gradient's own directional
    # derivative -- not the gradient with respect to the direction.
    grad_dt_v = _three_pool_contract(
        g00,
        g01,
        g02,
        g10,
        g11,
        g12,
        g20,
        g21,
        g22,
        b11,
        b12,
        b13,
        b21,
        b22,
        b23,
        b31,
        b32,
        b33,
        bfree,
        bpool_b,
        bbound,
        free,
        fraction_b,
        fraction_c,
    )
    grad_dt_t = (
        _three_pool_contract(
            g00,
            g01,
            g02,
            g10,
            g11,
            g12,
            g20,
            g21,
            g22,
            t11,
            t12,
            t13,
            t21,
            t22,
            t23,
            t31,
            t32,
            t33,
            tfree,
            tpool_b,
            tbound,
            free,
            fraction_b,
            fraction_c,
        )
        + _three_pool_contract(
            w00,
            w01,
            w02,
            w10,
            w11,
            w12,
            w20,
            w21,
            w22,
            b11,
            b12,
            b13,
            b21,
            b22,
            b23,
            b31,
            b32,
            b33,
            bfree,
            bpool_b,
            bbound,
            free,
            fraction_b,
            fraction_c,
        )
        - (
            bfree * (g00 * d_free + g01 * d_fraction_b + g02 * d_fraction_c)
            + bpool_b * (g10 * d_free + g11 * d_fraction_b + g12 * d_fraction_c)
            + bbound * (g20 * d_free + g21 * d_fraction_b + g22 * d_fraction_c)
        )
    )
    grad_att_v = _three_pool_contract(
        c00,
        c01,
        c02,
        c10,
        c11,
        c12,
        c20,
        c21,
        c22,
        b11,
        b12,
        b13,
        b21,
        b22,
        b23,
        b31,
        b32,
        b33,
        bfree,
        bpool_b,
        bbound,
        free,
        fraction_b,
        fraction_c,
    )
    grad_att_t = (
        _three_pool_contract(
            c00,
            c01,
            c02,
            c10,
            c11,
            c12,
            c20,
            c21,
            c22,
            t11,
            t12,
            t13,
            t21,
            t22,
            t23,
            t31,
            t32,
            t33,
            tfree,
            tpool_b,
            tbound,
            free,
            fraction_b,
            fraction_c,
        )
        + _three_pool_contract(
            d00,
            d01,
            d02,
            d10,
            d11,
            d12,
            d20,
            d21,
            d22,
            b11,
            b12,
            b13,
            b21,
            b22,
            b23,
            b31,
            b32,
            b33,
            bfree,
            bpool_b,
            bbound,
            free,
            fraction_b,
            fraction_c,
        )
        - (
            bfree * (c00 * d_free + c01 * d_fraction_b + c02 * d_fraction_c)
            + bpool_b * (c10 * d_free + c11 * d_fraction_b + c12 * d_fraction_c)
            + bbound * (c20 * d_free + c21 * d_fraction_b + c22 * d_fraction_c)
        )
    )
    return grad_dt_v, grad_att_v, grad_dt_t, grad_att_t


@triton.jit
def _three_pool_table_kernel(
    t1,
    t1_pool_b,
    t1_bound,
    pool_b_exchange,
    bound_exchange,
    pool_b_fraction,
    bound_fraction,
    durations,
    rows,
    table,
    voxel_count,
    BLOCK: tl.constexpr,
    narrow: tl.constexpr,
):
    """Fill one row of the three-pool operator table.

    The row is ``expm((K - diag(R1)) dt)`` with no washout applied, so the
    event that reads it supplies its own attenuation. Laid out
    ``(rows, 9, voxels)`` -- entry-major over the voxel axis -- so the nine
    loads an event makes are each coalesced.
    """
    row = tl.load(rows + tl.program_id(0))
    atom = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = atom < voxel_count
    dt = tl.load(durations + row)
    fraction_b = tl.load(pool_b_fraction + atom, mask=live, other=0.0)
    fraction_c = tl.load(bound_fraction + atom, mask=live, other=0.0)
    (
        e00,
        e01,
        e02,
        e10,
        e11,
        e12,
        e20,
        e21,
        e22,
        _,
        _,
        _,
    ) = _three_pool_step(
        1000.0 / tl.load(t1 + atom, mask=live, other=1.0),
        1000.0 / tl.load(t1_pool_b + atom, mask=live, other=1.0),
        1000.0 / tl.load(t1_bound + atom, mask=live, other=1.0),
        tl.load(pool_b_exchange + atom, mask=live, other=0.0),
        tl.load(bound_exchange + atom, mask=live, other=0.0),
        fraction_b,
        fraction_c,
        dt,
        # The helpers narrow their arguments, so the attenuation an undamped
        # row wants has to arrive as a tensor rather than a literal one.
        1.0 + 0.0 * dt,
        narrow,
    )
    base = table + row * (9 * voxel_count) + atom
    tl.store(base + 0 * voxel_count, e00, mask=live)
    tl.store(base + 1 * voxel_count, e01, mask=live)
    tl.store(base + 2 * voxel_count, e02, mask=live)
    tl.store(base + 3 * voxel_count, e10, mask=live)
    tl.store(base + 4 * voxel_count, e11, mask=live)
    tl.store(base + 5 * voxel_count, e12, mask=live)
    tl.store(base + 6 * voxel_count, e20, mask=live)
    tl.store(base + 7 * voxel_count, e21, mask=live)
    tl.store(base + 8 * voxel_count, e22, mask=live)


@triton.jit
def _exp_difference_jvp(lower, d_lower, upper, d_upper, low_exp, high_exp):
    """``[a, b] exp`` and its directional derivative.

    The derivative of a divided difference is the next one along,
    ``d/da [a,b] = [a,a,b]``, which near the coalescence is again a series in
    the gap's square rather than a quotient that vanishes over a vanishing
    denominator.
    """
    half = 0.5 * (upper - lower)
    d_half = 0.5 * (d_upper - d_lower)
    near = tl.abs(half) < _SINCH_CUT
    square = half * half
    d_square = 2.0 * half * d_half
    # exp(mid) * sinh(half)/half, both factors expanded about zero.
    lift = low_exp * (1.0 + half + 0.5 * square)
    d_lift = low_exp * (d_lower * (1.0 + half + 0.5 * square) + d_half + half * d_half)
    sinch = 1.0 + square / 6.0 + square * square / 120.0
    d_sinch = d_square / 6.0 + 2.0 * square * d_square / 120.0
    series = lift * sinch
    d_series = d_lift * sinch + lift * d_sinch
    gap = tl.where(near, 1.0, upper - lower)
    d_gap = tl.where(near, 0.0, d_upper - d_lower)
    quotient = (high_exp - low_exp) / gap
    d_quotient = (high_exp * d_upper - low_exp * d_lower - quotient * d_gap) / gap
    return tl.where(near, series, quotient), tl.where(near, d_series, d_quotient)


@triton.jit
def _three_pool_pieces_jvp(
    r1_free,
    d_r1_free,
    r1_pool_b,
    d_r1_pool_b,
    r1_bound,
    d_r1_bound,
    exchange_b,
    d_exchange_b,
    exchange_c,
    d_exchange_c,
    fraction_b,
    d_fraction_b,
    fraction_c,
    d_fraction_c,
    dt,
    d_dt,
    narrow: tl.constexpr = False,
):
    """The three-pool operator's shared front half, as duals.

    The generator, its two invariants, the series coefficients and the three
    roots with the divided differences between them -- everything both the
    operator and its reverse sweep are assembled from, computed once in double
    so the two cannot drift apart.
    """
    work: tl.constexpr = tl.float32 if narrow else tl.float64
    step = dt.to(work)
    d_step = d_dt.to(work)
    free = (1.0 - fraction_b - fraction_c).to(work)
    d_free = (-d_fraction_b - d_fraction_c).to(work)
    pool_b = fraction_b.to(work)
    d_pool_b = d_fraction_b.to(work)
    pool_c = fraction_c.to(work)
    d_pool_c = d_fraction_c.to(work)
    rate_b = exchange_b.to(work)
    d_rate_b = d_exchange_b.to(work)
    rate_c = exchange_c.to(work)
    d_rate_c = d_exchange_c.to(work)
    kab = rate_b * pool_b
    d_kab = d_rate_b * pool_b + rate_b * d_pool_b
    kba = rate_b * free
    d_kba = d_rate_b * free + rate_b * d_free
    kac = rate_c * pool_c
    d_kac = d_rate_c * pool_c + rate_c * d_pool_c
    kca = rate_c * free
    d_kca = d_rate_c * free + rate_c * d_free
    row_a = -kab - kac - r1_free.to(work)
    d_row_a = -d_kab - d_kac - d_r1_free.to(work)
    row_b = -kba - r1_pool_b.to(work)
    d_row_b = -d_kba - d_r1_pool_b.to(work)
    row_c = -kca - r1_bound.to(work)
    d_row_c = -d_kca - d_r1_bound.to(work)
    a00 = row_a * step
    d_a00 = d_row_a * step + row_a * d_step
    a01 = kba * step
    d_a01 = d_kba * step + kba * d_step
    a02 = kca * step
    d_a02 = d_kca * step + kca * d_step
    a10 = kab * step
    d_a10 = d_kab * step + kab * d_step
    a11 = row_b * step
    d_a11 = d_row_b * step + row_b * d_step
    a20 = kac * step
    d_a20 = d_kac * step + kac * d_step
    a22 = row_c * step
    d_a22 = d_row_c * step + row_c * d_step

    third = (a00 + a11 + a22) / 3.0
    d_third = (d_a00 + d_a11 + d_a22) / 3.0
    s00 = a00 - third
    d_s00 = d_a00 - d_third
    s11 = a11 - third
    d_s11 = d_a11 - d_third
    s22 = a22 - third
    d_s22 = d_a22 - d_third
    minors = s00 * s11 - a01 * a10 + s00 * s22 - a02 * a20 + s11 * s22
    d_minors = (
        d_s00 * s11
        + s00 * d_s11
        - d_a01 * a10
        - a01 * d_a10
        + d_s00 * s22
        + s00 * d_s22
        - d_a02 * a20
        - a02 * d_a20
        + d_s11 * s22
        + s11 * d_s22
    )
    determinant = s00 * s11 * s22 - a01 * (a10 * s22) + a02 * (-s11 * a20)
    d_determinant = (
        d_s00 * s11 * s22
        + s00 * d_s11 * s22
        + s00 * s11 * d_s22
        - d_a01 * a10 * s22
        - a01 * d_a10 * s22
        - a01 * a10 * d_s22
        - d_a02 * s11 * a20
        - a02 * d_s11 * a20
        - a02 * s11 * d_a20
    )

    # --- close together: the series reduced modulo x^3 + minors x - det ---
    flat = 1.0 + 0.0 * third
    linear = 0.0 * third
    square = 0.0 * third
    d_flat = 0.0 * third
    d_linear = 0.0 * third
    d_square = 0.0 * third
    sum_flat = flat
    sum_linear = linear
    sum_square = square
    d_sum_flat = d_flat
    d_sum_linear = d_linear
    d_sum_square = d_square
    factorial = 1.0
    for order in tl.static_range(1, 16):
        next_flat = square * determinant
        d_next_flat = d_square * determinant + square * d_determinant
        next_linear = flat - square * minors
        d_next_linear = d_flat - d_square * minors - square * d_minors
        next_square = linear
        d_next_square = d_linear
        flat = next_flat
        linear = next_linear
        square = next_square
        d_flat = d_next_flat
        d_linear = d_next_linear
        d_square = d_next_square
        factorial = factorial * order
        weight = 1.0 / factorial
        sum_flat = sum_flat + weight * flat
        sum_linear = sum_linear + weight * linear
        sum_square = sum_square + weight * square
        d_sum_flat = d_sum_flat + weight * d_flat
        d_sum_linear = d_sum_linear + weight * d_linear
        d_sum_square = d_sum_square + weight * d_square
    lift = tl.exp(third)
    d_lift = lift * d_third

    # --- far apart: the Newton form at the three roots ---
    inside = -minors * (1.0 / 3.0)
    d_inside = -d_minors * (1.0 / 3.0)
    radius = tl.sqrt(tl.maximum(inside, 1e-300))
    d_radius = tl.where(inside > 0.0, 0.5 * d_inside / radius, 0.0)
    cube = radius * radius * radius
    raw = 0.5 * determinant / cube
    d_raw = (0.5 * d_determinant - raw * 3.0 * radius * radius * d_radius) / cube
    inside_limit = (raw > -_ARG_LIMIT) & (raw < _ARG_LIMIT)
    argument = tl.minimum(tl.maximum(raw, -_ARG_LIMIT), _ARG_LIMIT)
    d_argument = tl.where(inside_limit, d_raw, 0.0)
    angle = libdevice.acos(argument) / 3.0
    d_angle = -d_argument / (
        3.0 * tl.sqrt(tl.maximum(1.0 - argument * argument, 1e-300))
    )
    root_a = 2.0 * radius * tl.cos(angle) + third
    d_root_a = (
        2.0 * d_radius * tl.cos(angle)
        - 2.0 * radius * tl.sin(angle) * d_angle
        + d_third
    )
    root_b = 2.0 * radius * tl.cos(angle - _TURN_THIRD) + third
    d_root_b = (
        2.0 * d_radius * tl.cos(angle - _TURN_THIRD)
        - 2.0 * radius * tl.sin(angle - _TURN_THIRD) * d_angle
        + d_third
    )
    root_c = 2.0 * radius * tl.cos(angle - 2.0 * _TURN_THIRD) + third
    d_root_c = (
        2.0 * d_radius * tl.cos(angle - 2.0 * _TURN_THIRD)
        - 2.0 * radius * tl.sin(angle - 2.0 * _TURN_THIRD) * d_angle
        + d_third
    )
    # Sorting is a permutation, so the tangents follow their own values.
    low = tl.minimum(tl.minimum(root_a, root_b), root_c)
    high = tl.maximum(tl.maximum(root_a, root_b), root_c)
    middle = tl.maximum(
        tl.minimum(root_a, root_b), tl.minimum(tl.maximum(root_a, root_b), root_c)
    )
    d_low = tl.where(
        root_a == low, d_root_a, tl.where(root_b == low, d_root_b, d_root_c)
    )
    d_high = tl.where(
        root_a == high, d_root_a, tl.where(root_b == high, d_root_b, d_root_c)
    )
    d_middle = tl.where(
        root_a == middle, d_root_a, tl.where(root_b == middle, d_root_b, d_root_c)
    )
    leading = tl.exp(low)
    d_leading = leading * d_low
    centre = tl.exp(middle)
    d_centre = centre * d_middle
    trailing = tl.exp(high)
    d_trailing = trailing * d_high
    first, d_first = _exp_difference_jvp(low, d_low, middle, d_middle, leading, centre)
    upper, d_upper = _exp_difference_jvp(
        middle, d_middle, high, d_high, centre, trailing
    )
    span = high - low
    d_span = d_high - d_low
    guarded = tl.where(span > 0.0, span, 1.0)
    d_guarded = tl.where(span > 0.0, d_span, 0.0)
    second = (upper - first) / guarded
    d_second = (d_upper - d_first - second * d_guarded) / guarded

    # --- the shifted generator squared, for the series branch ---
    q00 = s00 * s00 + a01 * a10 + a02 * a20
    d_q00 = 2.0 * s00 * d_s00 + d_a01 * a10 + a01 * d_a10 + d_a02 * a20 + a02 * d_a20
    q01 = a01 * (s00 + s11)
    d_q01 = d_a01 * (s00 + s11) + a01 * (d_s00 + d_s11)
    q02 = a02 * (s00 + s22)
    d_q02 = d_a02 * (s00 + s22) + a02 * (d_s00 + d_s22)
    q10 = a10 * (s00 + s11)
    d_q10 = d_a10 * (s00 + s11) + a10 * (d_s00 + d_s11)
    q11 = a10 * a01 + s11 * s11
    d_q11 = d_a10 * a01 + a10 * d_a01 + 2.0 * s11 * d_s11
    q12 = a10 * a02
    d_q12 = d_a10 * a02 + a10 * d_a02
    q20 = a20 * (s00 + s22)
    d_q20 = d_a20 * (s00 + s22) + a20 * (d_s00 + d_s22)
    q21 = a20 * a01
    d_q21 = d_a20 * a01 + a20 * d_a01
    q22 = a20 * a02 + s22 * s22
    d_q22 = d_a20 * a02 + a20 * d_a02 + 2.0 * s22 * d_s22

    return (
        free,
        d_free,
        pool_b,
        d_pool_b,
        pool_c,
        d_pool_c,
        a00,
        d_a00,
        a01,
        d_a01,
        a02,
        d_a02,
        a10,
        d_a10,
        a11,
        d_a11,
        a20,
        d_a20,
        a22,
        d_a22,
        s00,
        d_s00,
        s11,
        d_s11,
        s22,
        d_s22,
        minors,
        d_minors,
        sum_flat,
        sum_linear,
        sum_square,
        d_sum_flat,
        d_sum_linear,
        d_sum_square,
        lift,
        d_lift,
        low,
        middle,
        d_low,
        d_middle,
        leading,
        d_leading,
        first,
        d_first,
        second,
        d_second,
        determinant,
        d_determinant,
        high,
        d_high,
        radius,
        d_radius,
        cube,
        raw,
        d_raw,
        argument,
        inside_limit,
        angle,
        d_angle,
        centre,
        d_centre,
        trailing,
        d_trailing,
        guarded,
        d_guarded,
        q00,
        d_q00,
        q01,
        d_q01,
        q02,
        d_q02,
        q10,
        d_q10,
        q11,
        d_q11,
        q12,
        d_q12,
        q20,
        d_q20,
        q21,
        d_q21,
        q22,
        d_q22,
    )


@triton.jit
def _three_pool_assemble_jvp(
    free,
    d_free,
    pool_b,
    d_pool_b,
    pool_c,
    d_pool_c,
    a00,
    d_a00,
    a01,
    d_a01,
    a02,
    d_a02,
    a10,
    d_a10,
    a11,
    d_a11,
    a20,
    d_a20,
    a22,
    d_a22,
    s00,
    d_s00,
    s11,
    d_s11,
    s22,
    d_s22,
    minors,
    d_minors,
    sum_flat,
    sum_linear,
    sum_square,
    d_sum_flat,
    d_sum_linear,
    d_sum_square,
    lift,
    d_lift,
    low,
    middle,
    d_low,
    d_middle,
    leading,
    d_leading,
    first,
    d_first,
    second,
    d_second,
    determinant,
    d_determinant,
    high,
    d_high,
    radius,
    d_radius,
    cube,
    raw,
    d_raw,
    argument,
    inside_limit,
    angle,
    d_angle,
    centre,
    d_centre,
    trailing,
    d_trailing,
    guarded,
    d_guarded,
    q00,
    d_q00,
    q01,
    d_q01,
    q02,
    d_q02,
    q10,
    d_q10,
    q11,
    d_q11,
    q12,
    d_q12,
    q20,
    d_q20,
    q21,
    d_q21,
    q22,
    d_q22,
    narrow: tl.constexpr = False,
):
    """The bare three-pool operator, assembled from its shared pieces.

    Both branches are formed and one is chosen: a ``where`` evaluates each
    side, so the divisor each of them carries is guarded whether or not it
    is the side taken.

    In double, and before any attenuation -- what a reverse sweep reads,
    and what :func:`_three_pool_weigh_jvp` turns into an interval's step.
    """
    c00 = lift * (sum_flat + sum_linear * s00 + sum_square * q00)
    d_c00 = d_lift * (sum_flat + sum_linear * s00 + sum_square * q00) + lift * (
        d_sum_flat
        + d_sum_linear * s00
        + sum_linear * d_s00
        + d_sum_square * q00
        + sum_square * d_q00
    )
    c01 = lift * (sum_linear * a01 + sum_square * q01)
    d_c01 = d_lift * (sum_linear * a01 + sum_square * q01) + lift * (
        d_sum_linear * a01
        + sum_linear * d_a01
        + d_sum_square * q01
        + sum_square * d_q01
    )
    c02 = lift * (sum_linear * a02 + sum_square * q02)
    d_c02 = d_lift * (sum_linear * a02 + sum_square * q02) + lift * (
        d_sum_linear * a02
        + sum_linear * d_a02
        + d_sum_square * q02
        + sum_square * d_q02
    )
    c10 = lift * (sum_linear * a10 + sum_square * q10)
    d_c10 = d_lift * (sum_linear * a10 + sum_square * q10) + lift * (
        d_sum_linear * a10
        + sum_linear * d_a10
        + d_sum_square * q10
        + sum_square * d_q10
    )
    c11 = lift * (sum_flat + sum_linear * s11 + sum_square * q11)
    d_c11 = d_lift * (sum_flat + sum_linear * s11 + sum_square * q11) + lift * (
        d_sum_flat
        + d_sum_linear * s11
        + sum_linear * d_s11
        + d_sum_square * q11
        + sum_square * d_q11
    )
    c12 = lift * (sum_square * q12)
    d_c12 = d_lift * (sum_square * q12) + lift * (
        d_sum_square * q12 + sum_square * d_q12
    )
    c20 = lift * (sum_linear * a20 + sum_square * q20)
    d_c20 = d_lift * (sum_linear * a20 + sum_square * q20) + lift * (
        d_sum_linear * a20
        + sum_linear * d_a20
        + d_sum_square * q20
        + sum_square * d_q20
    )
    c21 = lift * (sum_square * q21)
    d_c21 = d_lift * (sum_square * q21) + lift * (
        d_sum_square * q21 + sum_square * d_q21
    )
    c22 = lift * (sum_flat + sum_linear * s22 + sum_square * q22)
    d_c22 = d_lift * (sum_flat + sum_linear * s22 + sum_square * q22) + lift * (
        d_sum_flat
        + d_sum_linear * s22
        + sum_linear * d_s22
        + d_sum_square * q22
        + sum_square * d_q22
    )

    # --- the Newton form's two factors, for the eigenvalue branch ---
    m00 = a00 - low
    d_m00 = d_a00 - d_low
    m11 = a11 - low
    d_m11 = d_a11 - d_low
    m22 = a22 - low
    d_m22 = d_a22 - d_low
    n00 = a00 - middle
    d_n00 = d_a00 - d_middle
    n11 = a11 - middle
    d_n11 = d_a11 - d_middle
    n22 = a22 - middle
    d_n22 = d_a22 - d_middle
    p00 = m00 * n00 + a01 * a10 + a02 * a20
    d_p00 = (
        d_m00 * n00
        + m00 * d_n00
        + d_a01 * a10
        + a01 * d_a10
        + d_a02 * a20
        + a02 * d_a20
    )
    p01 = a01 * (m00 + n11)
    d_p01 = d_a01 * (m00 + n11) + a01 * (d_m00 + d_n11)
    p02 = a02 * (m00 + n22)
    d_p02 = d_a02 * (m00 + n22) + a02 * (d_m00 + d_n22)
    p10 = a10 * (n00 + m11)
    d_p10 = d_a10 * (n00 + m11) + a10 * (d_n00 + d_m11)
    p11 = a10 * a01 + m11 * n11
    d_p11 = d_a10 * a01 + a10 * d_a01 + d_m11 * n11 + m11 * d_n11
    p12 = a10 * a02
    d_p12 = d_a10 * a02 + a10 * d_a02
    p20 = a20 * (n00 + m22)
    d_p20 = d_a20 * (n00 + m22) + a20 * (d_n00 + d_m22)
    p21 = a20 * a01
    d_p21 = d_a20 * a01 + a20 * d_a01
    p22 = a20 * a02 + m22 * n22
    d_p22 = d_a20 * a02 + a20 * d_a02 + d_m22 * n22 + m22 * d_n22

    e00 = leading + first * m00 + second * p00
    d_e00 = d_leading + d_first * m00 + first * d_m00 + d_second * p00 + second * d_p00
    e01 = first * a01 + second * p01
    d_e01 = d_first * a01 + first * d_a01 + d_second * p01 + second * d_p01
    e02 = first * a02 + second * p02
    d_e02 = d_first * a02 + first * d_a02 + d_second * p02 + second * d_p02
    e10 = first * a10 + second * p10
    d_e10 = d_first * a10 + first * d_a10 + d_second * p10 + second * d_p10
    e11 = leading + first * m11 + second * p11
    d_e11 = d_leading + d_first * m11 + first * d_m11 + d_second * p11 + second * d_p11
    e12 = second * p12
    d_e12 = d_second * p12 + second * d_p12
    e20 = first * a20 + second * p20
    d_e20 = d_first * a20 + first * d_a20 + d_second * p20 + second * d_p20
    e21 = second * p21
    d_e21 = d_second * p21 + second * d_p21
    e22 = leading + first * m22 + second * p22
    d_e22 = d_leading + d_first * m22 + first * d_m22 + d_second * p22 + second * d_p22

    if narrow:
        # The caller has bounded the spread, so the roots are unreachable and
        # everything that leads to them goes with this select.
        def_00, dif_00 = c00, d_c00
        def_01, dif_01 = c01, d_c01
        def_02, dif_02 = c02, d_c02
        def_10, dif_10 = c10, d_c10
        def_11, dif_11 = c11, d_c11
        def_12, dif_12 = c12, d_c12
        def_20, dif_20 = c20, d_c20
        def_21, dif_21 = c21, d_c21
        def_22, dif_22 = c22, d_c22
    else:
        close = -2.0 * minors < _SPREAD_CUT * _SPREAD_CUT
        def_00 = tl.where(close, c00, e00)
        dif_00 = tl.where(close, d_c00, d_e00)
        def_01 = tl.where(close, c01, e01)
        dif_01 = tl.where(close, d_c01, d_e01)
        def_02 = tl.where(close, c02, e02)
        dif_02 = tl.where(close, d_c02, d_e02)
        def_10 = tl.where(close, c10, e10)
        dif_10 = tl.where(close, d_c10, d_e10)
        def_11 = tl.where(close, c11, e11)
        dif_11 = tl.where(close, d_c11, d_e11)
        def_12 = tl.where(close, c12, e12)
        dif_12 = tl.where(close, d_c12, d_e12)
        def_20 = tl.where(close, c20, e20)
        dif_20 = tl.where(close, d_c20, d_e20)
        def_21 = tl.where(close, c21, e21)
        dif_21 = tl.where(close, d_c21, d_e21)
        def_22 = tl.where(close, c22, e22)
        dif_22 = tl.where(close, d_c22, d_e22)

    return (
        def_00,
        dif_00,
        def_01,
        dif_01,
        def_02,
        dif_02,
        def_10,
        dif_10,
        def_11,
        dif_11,
        def_12,
        dif_12,
        def_20,
        dif_20,
        def_21,
        dif_21,
        def_22,
        dif_22,
    )


@triton.jit
def _three_pool_weigh_jvp(
    def_00,
    dif_00,
    def_01,
    dif_01,
    def_02,
    dif_02,
    def_10,
    dif_10,
    def_11,
    dif_11,
    def_12,
    dif_12,
    def_20,
    dif_20,
    def_21,
    dif_21,
    def_22,
    dif_22,
    free,
    d_free,
    pool_b,
    d_pool_b,
    pool_c,
    d_pool_c,
    attenuation,
    d_attenuation,
    narrow: tl.constexpr = False,
):
    """An interval's step, from the bare operator and what survives it.

    The recovery is ``(I - E) m0`` rather than a solve, which is what the
    equilibrium being a fixed point of the generator buys.
    """
    work: tl.constexpr = tl.float32 if narrow else tl.float64
    damp = attenuation.to(work)
    d_damp = d_attenuation.to(work)
    w00 = damp * def_00
    dw00 = d_damp * def_00 + damp * dif_00
    w01 = damp * def_01
    dw01 = d_damp * def_01 + damp * dif_01
    w02 = damp * def_02
    dw02 = d_damp * def_02 + damp * dif_02
    w10 = damp * def_10
    dw10 = d_damp * def_10 + damp * dif_10
    w11 = damp * def_11
    dw11 = d_damp * def_11 + damp * dif_11
    w12 = damp * def_12
    dw12 = d_damp * def_12 + damp * dif_12
    w20 = damp * def_20
    dw20 = d_damp * def_20 + damp * dif_20
    w21 = damp * def_21
    dw21 = d_damp * def_21 + damp * dif_21
    w22 = damp * def_22
    dw22 = d_damp * def_22 + damp * dif_22

    grow_free = free - (w00 * free + w01 * pool_b + w02 * pool_c)
    d_grow_free = d_free - (
        dw00 * free
        + w00 * d_free
        + dw01 * pool_b
        + w01 * d_pool_b
        + dw02 * pool_c
        + w02 * d_pool_c
    )
    grow_pool_b = pool_b - (w10 * free + w11 * pool_b + w12 * pool_c)
    d_grow_pool_b = d_pool_b - (
        dw10 * free
        + w10 * d_free
        + dw11 * pool_b
        + w11 * d_pool_b
        + dw12 * pool_c
        + w12 * d_pool_c
    )
    grow_bound = pool_c - (w20 * free + w21 * pool_b + w22 * pool_c)
    d_grow_bound = d_pool_c - (
        dw20 * free
        + w20 * d_free
        + dw21 * pool_b
        + w21 * d_pool_b
        + dw22 * pool_c
        + w22 * d_pool_c
    )
    return (
        w00,
        w01,
        w02,
        w10,
        w11,
        w12,
        w20,
        w21,
        w22,
        grow_free,
        grow_pool_b,
        grow_bound,
        dw00,
        dw01,
        dw02,
        dw10,
        dw11,
        dw12,
        dw20,
        dw21,
        dw22,
        d_grow_free,
        d_grow_pool_b,
        d_grow_bound,
    )


@triton.jit
def _three_pool_step_jvp(
    r1_free,
    d_r1_free,
    r1_pool_b,
    d_r1_pool_b,
    r1_bound,
    d_r1_bound,
    exchange_b,
    d_exchange_b,
    exchange_c,
    d_exchange_c,
    fraction_b,
    d_fraction_b,
    fraction_c,
    d_fraction_c,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
    narrow: tl.constexpr = False,
):
    """The three-pool longitudinal step and its directional derivative.

    The same closed form :func:`_three_pool_step` evaluates, carried
    alongside a tangent and in the same double precision. Returns the
    nine entries and three recoveries, then their twelve tangents.
    """
    (
        free,
        d_free,
        pool_b,
        d_pool_b,
        pool_c,
        d_pool_c,
        a00,
        d_a00,
        a01,
        d_a01,
        a02,
        d_a02,
        a10,
        d_a10,
        a11,
        d_a11,
        a20,
        d_a20,
        a22,
        d_a22,
        s00,
        d_s00,
        s11,
        d_s11,
        s22,
        d_s22,
        minors,
        d_minors,
        sum_flat,
        sum_linear,
        sum_square,
        d_sum_flat,
        d_sum_linear,
        d_sum_square,
        lift,
        d_lift,
        low,
        middle,
        d_low,
        d_middle,
        leading,
        d_leading,
        first,
        d_first,
        second,
        d_second,
        determinant,
        d_determinant,
        high,
        d_high,
        radius,
        d_radius,
        cube,
        raw,
        d_raw,
        argument,
        inside_limit,
        angle,
        d_angle,
        centre,
        d_centre,
        trailing,
        d_trailing,
        guarded,
        d_guarded,
        q00,
        d_q00,
        q01,
        d_q01,
        q02,
        d_q02,
        q10,
        d_q10,
        q11,
        d_q11,
        q12,
        d_q12,
        q20,
        d_q20,
        q21,
        d_q21,
        q22,
        d_q22,
    ) = _three_pool_pieces_jvp(
        r1_free,
        d_r1_free,
        r1_pool_b,
        d_r1_pool_b,
        r1_bound,
        d_r1_bound,
        exchange_b,
        d_exchange_b,
        exchange_c,
        d_exchange_c,
        fraction_b,
        d_fraction_b,
        fraction_c,
        d_fraction_c,
        dt,
        d_dt,
        narrow,
    )
    (
        def_00,
        dif_00,
        def_01,
        dif_01,
        def_02,
        dif_02,
        def_10,
        dif_10,
        def_11,
        dif_11,
        def_12,
        dif_12,
        def_20,
        dif_20,
        def_21,
        dif_21,
        def_22,
        dif_22,
    ) = _three_pool_assemble_jvp(
        free,
        d_free,
        pool_b,
        d_pool_b,
        pool_c,
        d_pool_c,
        a00,
        d_a00,
        a01,
        d_a01,
        a02,
        d_a02,
        a10,
        d_a10,
        a11,
        d_a11,
        a20,
        d_a20,
        a22,
        d_a22,
        s00,
        d_s00,
        s11,
        d_s11,
        s22,
        d_s22,
        minors,
        d_minors,
        sum_flat,
        sum_linear,
        sum_square,
        d_sum_flat,
        d_sum_linear,
        d_sum_square,
        lift,
        d_lift,
        low,
        middle,
        d_low,
        d_middle,
        leading,
        d_leading,
        first,
        d_first,
        second,
        d_second,
        determinant,
        d_determinant,
        high,
        d_high,
        radius,
        d_radius,
        cube,
        raw,
        d_raw,
        argument,
        inside_limit,
        angle,
        d_angle,
        centre,
        d_centre,
        trailing,
        d_trailing,
        guarded,
        d_guarded,
        q00,
        d_q00,
        q01,
        d_q01,
        q02,
        d_q02,
        q10,
        d_q10,
        q11,
        d_q11,
        q12,
        d_q12,
        q20,
        d_q20,
        q21,
        d_q21,
        q22,
        d_q22,
        narrow,
    )
    (
        w00,
        w01,
        w02,
        w10,
        w11,
        w12,
        w20,
        w21,
        w22,
        grow_free,
        grow_pool_b,
        grow_bound,
        dw00,
        dw01,
        dw02,
        dw10,
        dw11,
        dw12,
        dw20,
        dw21,
        dw22,
        d_grow_free,
        d_grow_pool_b,
        d_grow_bound,
    ) = _three_pool_weigh_jvp(
        def_00,
        dif_00,
        def_01,
        dif_01,
        def_02,
        dif_02,
        def_10,
        dif_10,
        def_11,
        dif_11,
        def_12,
        dif_12,
        def_20,
        dif_20,
        def_21,
        dif_21,
        def_22,
        dif_22,
        free,
        d_free,
        pool_b,
        d_pool_b,
        pool_c,
        d_pool_c,
        attenuation,
        d_attenuation,
        narrow,
    )
    return (
        w00.to(tl.float32),
        w01.to(tl.float32),
        w02.to(tl.float32),
        w10.to(tl.float32),
        w11.to(tl.float32),
        w12.to(tl.float32),
        w20.to(tl.float32),
        w21.to(tl.float32),
        w22.to(tl.float32),
        grow_free.to(tl.float32),
        grow_pool_b.to(tl.float32),
        grow_bound.to(tl.float32),
        dw00.to(tl.float32),
        dw01.to(tl.float32),
        dw02.to(tl.float32),
        dw10.to(tl.float32),
        dw11.to(tl.float32),
        dw12.to(tl.float32),
        dw20.to(tl.float32),
        dw21.to(tl.float32),
        dw22.to(tl.float32),
        d_grow_free.to(tl.float32),
        d_grow_pool_b.to(tl.float32),
        d_grow_bound.to(tl.float32),
    )


@triton.jit
def _exp_difference_adjoint_jvp(
    lower,
    d_lower,
    upper,
    d_upper,
    exp_lower,
    d_exp_lower,
    exp_upper,
    d_exp_upper,
    seed,
    d_seed,
):
    """The reverse of :func:`_exp_difference`, onto both points, on a direction.

    Near the coalescence the slope comes from the same series the value does,
    because the difference quotient's own derivative is a cancellation divided
    by a small number twice over.
    """
    half = 0.5 * (upper - lower)
    d_half = 0.5 * (d_upper - d_lower)
    near = tl.abs(half) < _SINCH_CUT
    poly = 1.0 + half + 0.5 * half * half
    d_poly = d_half + half * d_half
    even = 1.0 + half * half / 6.0
    d_even = half * d_half / 3.0
    slope = (1.0 + half) * even + poly * half * (1.0 / 3.0)
    d_slope = (
        d_half * even
        + (1.0 + half) * d_even
        + (d_poly * half + poly * d_half) * (1.0 / 3.0)
    )
    series = exp_lower * poly * even
    d_series = d_exp_lower * poly * even + exp_lower * (d_poly * even + poly * d_even)
    swing = 0.5 * exp_lower * slope
    d_swing = 0.5 * (d_exp_lower * slope + exp_lower * d_slope)

    gap = tl.where(near, 1.0, upper - lower)
    d_gap = tl.where(near, 0.0, d_upper - d_lower)
    value = (exp_upper - exp_lower) / gap
    d_value = (d_exp_upper - d_exp_lower - value * d_gap) / gap
    far_lower = (value - exp_lower) / gap
    d_far_lower = (d_value - d_exp_lower - far_lower * d_gap) / gap
    far_upper = (exp_upper - value) / gap
    d_far_upper = (d_exp_upper - d_value - far_upper * d_gap) / gap

    to_lower = tl.where(near, series - swing, far_lower)
    d_to_lower = tl.where(near, d_series - d_swing, d_far_lower)
    to_upper = tl.where(near, swing, far_upper)
    d_to_upper = tl.where(near, d_swing, d_far_upper)
    return (
        seed * to_lower,
        d_seed * to_lower + seed * d_to_lower,
        seed * to_upper,
        d_seed * to_upper + seed * d_to_upper,
    )


@triton.jit
def _three_pool_step_adjoint_jvp(
    r1_free,
    d_r1_free,
    r1_pool_b,
    d_r1_pool_b,
    r1_bound,
    d_r1_bound,
    exchange_b,
    d_exchange_b,
    exchange_c,
    d_exchange_c,
    fraction_b,
    d_fraction_b,
    fraction_c,
    d_fraction_c,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
    bar_e00,
    d_bar_e00,
    bar_e01,
    d_bar_e01,
    bar_e02,
    d_bar_e02,
    bar_e10,
    d_bar_e10,
    bar_e11,
    d_bar_e11,
    bar_e12,
    d_bar_e12,
    bar_e20,
    d_bar_e20,
    bar_e21,
    d_bar_e21,
    bar_e22,
    d_bar_e22,
    bar_grow_free,
    d_bar_grow_free,
    bar_grow_pool_b,
    d_bar_grow_pool_b,
    bar_grow_bound,
    d_bar_grow_bound,
    free,
    d_free,
    pool_b,
    d_pool_b,
    pool_c,
    d_pool_c,
    a00,
    d_a00,
    a01,
    d_a01,
    a02,
    d_a02,
    a10,
    d_a10,
    a11,
    d_a11,
    a20,
    d_a20,
    a22,
    d_a22,
    s00,
    d_s00,
    s11,
    d_s11,
    s22,
    d_s22,
    minors,
    d_minors,
    sum_flat,
    sum_linear,
    sum_square,
    d_sum_flat,
    d_sum_linear,
    d_sum_square,
    lift,
    d_lift,
    low,
    middle,
    d_low,
    d_middle,
    leading,
    d_leading,
    first,
    d_first,
    second,
    d_second,
    determinant,
    d_determinant,
    high,
    d_high,
    radius,
    d_radius,
    cube,
    raw,
    d_raw,
    argument,
    inside_limit,
    angle,
    d_angle,
    centre,
    d_centre,
    trailing,
    d_trailing,
    guarded,
    d_guarded,
    q00,
    d_q00,
    q01,
    d_q01,
    q02,
    d_q02,
    q10,
    d_q10,
    q11,
    d_q11,
    q12,
    d_q12,
    q20,
    d_q20,
    q21,
    d_q21,
    q22,
    d_q22,
    def_00,
    dif_00,
    def_01,
    dif_01,
    def_02,
    dif_02,
    def_10,
    dif_10,
    def_11,
    dif_11,
    def_12,
    dif_12,
    def_20,
    dif_20,
    def_21,
    dif_21,
    def_22,
    dif_22,
    narrow: tl.constexpr = False,
):
    """The reverse sweep of :func:`_three_pool_step`, carried on a direction.

    Reads the pieces and the bare operator the replay already formed, so an
    interval's transcendentals are taken once for the pass rather than once
    for each direction through it, and in the same double.

    Both branches are swept, each by the algebra its own forward used, and the
    choice between them is made on the cotangents rather than on the way in --
    a ``where`` evaluates both sides, so each side's divisors are guarded.

    The series branch is a polynomial in the two invariants alone, so its
    reverse is reached by carrying the recurrence's sensitivity to those two
    forward beside it, which needs no history of the sixteen terms.

    Returned as the gradients w.r.t. ``(r1_free, r1_pool_b, r1_bound,
    exchange_b, exchange_c, fraction_b, fraction_c, dt, attenuation)`` and
    then their nine tangents.
    """
    work: tl.constexpr = tl.float32 if narrow else tl.float64
    # --- the recovery and the attenuation, which both branches share ---
    damp = attenuation.to(work)
    d_damp = d_attenuation.to(work)
    r0 = bar_grow_free.to(work)
    d_r0 = d_bar_grow_free.to(work)
    r1 = bar_grow_pool_b.to(work)
    d_r1 = d_bar_grow_pool_b.to(work)
    r2 = bar_grow_bound.to(work)
    d_r2 = d_bar_grow_bound.to(work)

    y00 = bar_e00.to(work) - r0 * free
    d_y00 = d_bar_e00.to(work) - d_r0 * free - r0 * d_free
    y01 = bar_e01.to(work) - r0 * pool_b
    d_y01 = d_bar_e01.to(work) - d_r0 * pool_b - r0 * d_pool_b
    y02 = bar_e02.to(work) - r0 * pool_c
    d_y02 = d_bar_e02.to(work) - d_r0 * pool_c - r0 * d_pool_c
    y10 = bar_e10.to(work) - r1 * free
    d_y10 = d_bar_e10.to(work) - d_r1 * free - r1 * d_free
    y11 = bar_e11.to(work) - r1 * pool_b
    d_y11 = d_bar_e11.to(work) - d_r1 * pool_b - r1 * d_pool_b
    y12 = bar_e12.to(work) - r1 * pool_c
    d_y12 = d_bar_e12.to(work) - d_r1 * pool_c - r1 * d_pool_c
    y20 = bar_e20.to(work) - r2 * free
    d_y20 = d_bar_e20.to(work) - d_r2 * free - r2 * d_free
    y21 = bar_e21.to(work) - r2 * pool_b
    d_y21 = d_bar_e21.to(work) - d_r2 * pool_b - r2 * d_pool_b
    y22 = bar_e22.to(work) - r2 * pool_c
    d_y22 = d_bar_e22.to(work) - d_r2 * pool_c - r2 * d_pool_c

    # The bare operator is read rather than the attenuation divided back out
    # of the weighed one -- a washed-out interval leaves nothing to divide by.
    bar_damp = (
        y00 * def_00
        + y01 * def_01
        + y02 * def_02
        + y10 * def_10
        + y11 * def_11
        + y12 * def_12
        + y20 * def_20
        + y21 * def_21
        + y22 * def_22
    )
    d_bar_damp = (
        d_y00 * def_00
        + y00 * dif_00
        + d_y01 * def_01
        + y01 * dif_01
        + d_y02 * def_02
        + y02 * dif_02
        + d_y10 * def_10
        + y10 * dif_10
        + d_y11 * def_11
        + y11 * dif_11
        + d_y12 * def_12
        + y12 * dif_12
        + d_y20 * def_20
        + y20 * dif_20
        + d_y21 * def_21
        + y21 * dif_21
        + d_y22 * def_22
        + y22 * dif_22
    )
    column_free = r0 * def_00 + r1 * def_10 + r2 * def_20
    d_column_free = (
        d_r0 * def_00
        + r0 * dif_00
        + d_r1 * def_10
        + r1 * dif_10
        + d_r2 * def_20
        + r2 * dif_20
    )
    column_pool_b = r0 * def_01 + r1 * def_11 + r2 * def_21
    d_column_pool_b = (
        d_r0 * def_01
        + r0 * dif_01
        + d_r1 * def_11
        + r1 * dif_11
        + d_r2 * def_21
        + r2 * dif_21
    )
    column_bound = r0 * def_02 + r1 * def_12 + r2 * def_22
    d_column_bound = (
        d_r0 * def_02
        + r0 * dif_02
        + d_r1 * def_12
        + r1 * dif_12
        + d_r2 * def_22
        + r2 * dif_22
    )
    bar_free = r0 - damp * column_free
    d_bar_free = d_r0 - d_damp * column_free - damp * d_column_free
    bar_pool_b = r1 - damp * column_pool_b
    d_bar_pool_b = d_r1 - d_damp * column_pool_b - damp * d_column_pool_b
    bar_pool_c = r2 - damp * column_bound
    d_bar_pool_c = d_r2 - d_damp * column_bound - damp * d_column_bound

    o00 = damp * y00
    d_o00 = d_damp * y00 + damp * d_y00
    o01 = damp * y01
    d_o01 = d_damp * y01 + damp * d_y01
    o02 = damp * y02
    d_o02 = d_damp * y02 + damp * d_y02
    o10 = damp * y10
    d_o10 = d_damp * y10 + damp * d_y10
    o11 = damp * y11
    d_o11 = d_damp * y11 + damp * d_y11
    o12 = damp * y12
    d_o12 = d_damp * y12 + damp * d_y12
    o20 = damp * y20
    d_o20 = d_damp * y20 + damp * d_y20
    o21 = damp * y21
    d_o21 = d_damp * y21 + damp * d_y21
    o22 = damp * y22
    d_o22 = d_damp * y22 + damp * d_y22

    # --- close together: the series in the two invariants, run backwards ---
    scale00 = o00 * lift
    d_scale00 = d_o00 * lift + o00 * d_lift
    scale01 = o01 * lift
    d_scale01 = d_o01 * lift + o01 * d_lift
    scale02 = o02 * lift
    d_scale02 = d_o02 * lift + o02 * d_lift
    scale10 = o10 * lift
    d_scale10 = d_o10 * lift + o10 * d_lift
    scale11 = o11 * lift
    d_scale11 = d_o11 * lift + o11 * d_lift
    scale12 = o12 * lift
    d_scale12 = d_o12 * lift + o12 * d_lift
    scale20 = o20 * lift
    d_scale20 = d_o20 * lift + o20 * d_lift
    scale21 = o21 * lift
    d_scale21 = d_o21 * lift + o21 * d_lift
    scale22 = o22 * lift
    d_scale22 = d_o22 * lift + o22 * d_lift

    bar_flat = scale00 + scale11 + scale22
    d_bar_flat = d_scale00 + d_scale11 + d_scale22
    bar_linear = (
        scale00 * s00
        + scale01 * a01
        + scale02 * a02
        + scale10 * a10
        + scale11 * s11
        + scale20 * a20
        + scale22 * s22
    )
    d_bar_linear = (
        d_scale00 * s00
        + scale00 * d_s00
        + d_scale01 * a01
        + scale01 * d_a01
        + d_scale02 * a02
        + scale02 * d_a02
        + d_scale10 * a10
        + scale10 * d_a10
        + d_scale11 * s11
        + scale11 * d_s11
        + d_scale20 * a20
        + scale20 * d_a20
        + d_scale22 * s22
        + scale22 * d_s22
    )
    bar_square = (
        scale00 * q00
        + scale01 * q01
        + scale02 * q02
        + scale10 * q10
        + scale11 * q11
        + scale12 * q12
        + scale20 * q20
        + scale21 * q21
        + scale22 * q22
    )
    d_bar_square = (
        d_scale00 * q00
        + scale00 * d_q00
        + d_scale01 * q01
        + scale01 * d_q01
        + d_scale02 * q02
        + scale02 * d_q02
        + d_scale10 * q10
        + scale10 * d_q10
        + d_scale11 * q11
        + scale11 * d_q11
        + d_scale12 * q12
        + scale12 * d_q12
        + d_scale20 * q20
        + scale20 * d_q20
        + d_scale21 * q21
        + scale21 * d_q21
        + d_scale22 * q22
        + scale22 * d_q22
    )
    # ``lift`` multiplies the whole bracket, so the shift it carries picks up
    # the bracket back again -- which is what the three sums contract to.
    turn_series = (
        sum_flat * bar_flat + sum_linear * bar_linear + sum_square * bar_square
    )
    d_turn_series = (
        d_sum_flat * bar_flat
        + sum_flat * d_bar_flat
        + d_sum_linear * bar_linear
        + sum_linear * d_bar_linear
        + d_sum_square * bar_square
        + sum_square * d_bar_square
    )

    g00 = sum_square * scale00
    d_g00 = d_sum_square * scale00 + sum_square * d_scale00
    g01 = sum_square * scale01
    d_g01 = d_sum_square * scale01 + sum_square * d_scale01
    g02 = sum_square * scale02
    d_g02 = d_sum_square * scale02 + sum_square * d_scale02
    g10 = sum_square * scale10
    d_g10 = d_sum_square * scale10 + sum_square * d_scale10
    g11 = sum_square * scale11
    d_g11 = d_sum_square * scale11 + sum_square * d_scale11
    g12 = sum_square * scale12
    d_g12 = d_sum_square * scale12 + sum_square * d_scale12
    g20 = sum_square * scale20
    d_g20 = d_sum_square * scale20 + sum_square * d_scale20
    g21 = sum_square * scale21
    d_g21 = d_sum_square * scale21 + sum_square * d_scale21
    g22 = sum_square * scale22
    d_g22 = d_sum_square * scale22 + sum_square * d_scale22

    # The square's reverse, ``g @ shifted^T + shifted^T @ g``.
    v00 = (
        g00 * s00
        + g01 * a01
        + g02 * a02
        + s00 * g00
        + a10 * g10
        + a20 * g20
        + sum_linear * scale00
    )
    d_v00 = (
        d_g00 * s00
        + g00 * d_s00
        + d_g01 * a01
        + g01 * d_a01
        + d_g02 * a02
        + g02 * d_a02
        + d_s00 * g00
        + s00 * d_g00
        + d_a10 * g10
        + a10 * d_g10
        + d_a20 * g20
        + a20 * d_g20
        + d_sum_linear * scale00
        + sum_linear * d_scale00
    )
    v01 = (
        g00 * a10 + g01 * s11 + s00 * g01 + a10 * g11 + a20 * g21 + sum_linear * scale01
    )
    d_v01 = (
        d_g00 * a10
        + g00 * d_a10
        + d_g01 * s11
        + g01 * d_s11
        + d_s00 * g01
        + s00 * d_g01
        + d_a10 * g11
        + a10 * d_g11
        + d_a20 * g21
        + a20 * d_g21
        + d_sum_linear * scale01
        + sum_linear * d_scale01
    )
    v02 = (
        g00 * a20 + g02 * s22 + s00 * g02 + a10 * g12 + a20 * g22 + sum_linear * scale02
    )
    d_v02 = (
        d_g00 * a20
        + g00 * d_a20
        + d_g02 * s22
        + g02 * d_s22
        + d_s00 * g02
        + s00 * d_g02
        + d_a10 * g12
        + a10 * d_g12
        + d_a20 * g22
        + a20 * d_g22
        + d_sum_linear * scale02
        + sum_linear * d_scale02
    )
    v10 = (
        g10 * s00 + g11 * a01 + g12 * a02 + a01 * g00 + s11 * g10 + sum_linear * scale10
    )
    d_v10 = (
        d_g10 * s00
        + g10 * d_s00
        + d_g11 * a01
        + g11 * d_a01
        + d_g12 * a02
        + g12 * d_a02
        + d_a01 * g00
        + a01 * d_g00
        + d_s11 * g10
        + s11 * d_g10
        + d_sum_linear * scale10
        + sum_linear * d_scale10
    )
    v11 = g10 * a10 + g11 * s11 + a01 * g01 + s11 * g11 + sum_linear * scale11
    d_v11 = (
        d_g10 * a10
        + g10 * d_a10
        + d_g11 * s11
        + g11 * d_s11
        + d_a01 * g01
        + a01 * d_g01
        + d_s11 * g11
        + s11 * d_g11
        + d_sum_linear * scale11
        + sum_linear * d_scale11
    )
    v20 = (
        g20 * s00 + g21 * a01 + g22 * a02 + a02 * g00 + s22 * g20 + sum_linear * scale20
    )
    d_v20 = (
        d_g20 * s00
        + g20 * d_s00
        + d_g21 * a01
        + g21 * d_a01
        + d_g22 * a02
        + g22 * d_a02
        + d_a02 * g00
        + a02 * d_g00
        + d_s22 * g20
        + s22 * d_g20
        + d_sum_linear * scale20
        + sum_linear * d_scale20
    )
    v22 = g20 * a20 + g22 * s22 + a02 * g02 + s22 * g22 + sum_linear * scale22
    d_v22 = (
        d_g20 * a20
        + g20 * d_a20
        + d_g22 * s22
        + g22 * d_s22
        + d_a02 * g02
        + a02 * d_g02
        + d_s22 * g22
        + s22 * d_g22
        + d_sum_linear * scale22
        + sum_linear * d_scale22
    )
    turn_series = turn_series - (v00 + v11 + v22)
    d_turn_series = d_turn_series - (d_v00 + d_v11 + d_v22)

    # The recurrence's own sensitivity to the two invariants, carried forward
    # beside it: two numbers reach the whole series, so their derivatives are
    # cheaper to push forward than the sixteen terms are to keep.
    flat = 1.0 + 0.0 * a00
    linear = 0.0 * a00
    square = 0.0 * a00
    d_flat = 0.0 * a00
    d_linear = 0.0 * a00
    d_square = 0.0 * a00
    fu = 0.0 * a00
    lu = 0.0 * a00
    su = 0.0 * a00
    d_fu = 0.0 * a00
    d_lu = 0.0 * a00
    d_su = 0.0 * a00
    fv = 0.0 * a00
    lv = 0.0 * a00
    sv = 0.0 * a00
    d_fv = 0.0 * a00
    d_lv = 0.0 * a00
    d_sv = 0.0 * a00
    slope_u_flat = 0.0 * a00
    slope_u_linear = 0.0 * a00
    slope_u_square = 0.0 * a00
    d_slope_u_flat = 0.0 * a00
    d_slope_u_linear = 0.0 * a00
    d_slope_u_square = 0.0 * a00
    slope_v_flat = 0.0 * a00
    slope_v_linear = 0.0 * a00
    slope_v_square = 0.0 * a00
    d_slope_v_flat = 0.0 * a00
    d_slope_v_linear = 0.0 * a00
    d_slope_v_square = 0.0 * a00
    factorial = 1.0
    for order in tl.static_range(1, 16):
        next_flat = square * determinant
        d_next_flat = d_square * determinant + square * d_determinant
        next_linear = flat - square * minors
        d_next_linear = d_flat - d_square * minors - square * d_minors
        next_square = linear
        d_next_square = d_linear
        next_fu = su * determinant
        d_next_fu = d_su * determinant + su * d_determinant
        next_lu = fu - su * minors - square
        d_next_lu = d_fu - d_su * minors - su * d_minors - d_square
        next_su = lu
        d_next_su = d_lu
        next_fv = sv * determinant + square
        d_next_fv = d_sv * determinant + sv * d_determinant + d_square
        next_lv = fv - sv * minors
        d_next_lv = d_fv - d_sv * minors - sv * d_minors
        next_sv = lv
        d_next_sv = d_lv
        flat = next_flat
        linear = next_linear
        square = next_square
        d_flat = d_next_flat
        d_linear = d_next_linear
        d_square = d_next_square
        fu = next_fu
        lu = next_lu
        su = next_su
        d_fu = d_next_fu
        d_lu = d_next_lu
        d_su = d_next_su
        fv = next_fv
        lv = next_lv
        sv = next_sv
        d_fv = d_next_fv
        d_lv = d_next_lv
        d_sv = d_next_sv
        factorial = factorial * order
        weight = 1.0 / factorial
        slope_u_flat = slope_u_flat + weight * fu
        slope_u_linear = slope_u_linear + weight * lu
        slope_u_square = slope_u_square + weight * su
        d_slope_u_flat = d_slope_u_flat + weight * d_fu
        d_slope_u_linear = d_slope_u_linear + weight * d_lu
        d_slope_u_square = d_slope_u_square + weight * d_su
        slope_v_flat = slope_v_flat + weight * fv
        slope_v_linear = slope_v_linear + weight * lv
        slope_v_square = slope_v_square + weight * sv
        d_slope_v_flat = d_slope_v_flat + weight * d_fv
        d_slope_v_linear = d_slope_v_linear + weight * d_lv
        d_slope_v_square = d_slope_v_square + weight * d_sv

    minors_series = (
        bar_flat * slope_u_flat
        + bar_linear * slope_u_linear
        + bar_square * slope_u_square
    )
    d_minors_series = (
        d_bar_flat * slope_u_flat
        + bar_flat * d_slope_u_flat
        + d_bar_linear * slope_u_linear
        + bar_linear * d_slope_u_linear
        + d_bar_square * slope_u_square
        + bar_square * d_slope_u_square
    )
    determinant_series = (
        bar_flat * slope_v_flat
        + bar_linear * slope_v_linear
        + bar_square * slope_v_square
    )
    d_determinant_series = (
        d_bar_flat * slope_v_flat
        + bar_flat * d_slope_v_flat
        + d_bar_linear * slope_v_linear
        + bar_linear * d_slope_v_linear
        + d_bar_square * slope_v_square
        + bar_square * d_slope_v_square
    )

    # --- far apart: back through the Newton form and the three roots ---
    m00 = a00 - low
    d_m00 = d_a00 - d_low
    m11 = a11 - low
    d_m11 = d_a11 - d_low
    m22 = a22 - low
    d_m22 = d_a22 - d_low
    n00 = a00 - middle
    d_n00 = d_a00 - d_middle
    n11 = a11 - middle
    d_n11 = d_a11 - d_middle
    n22 = a22 - middle
    d_n22 = d_a22 - d_middle
    p00 = m00 * n00 + a01 * a10 + a02 * a20
    d_p00 = (
        d_m00 * n00
        + m00 * d_n00
        + d_a01 * a10
        + a01 * d_a10
        + d_a02 * a20
        + a02 * d_a20
    )
    p01 = a01 * (m00 + n11)
    d_p01 = d_a01 * (m00 + n11) + a01 * (d_m00 + d_n11)
    p02 = a02 * (m00 + n22)
    d_p02 = d_a02 * (m00 + n22) + a02 * (d_m00 + d_n22)
    p10 = a10 * (m11 + n00)
    d_p10 = d_a10 * (m11 + n00) + a10 * (d_m11 + d_n00)
    p11 = m11 * n11 + a01 * a10
    d_p11 = d_m11 * n11 + m11 * d_n11 + d_a01 * a10 + a01 * d_a10
    p12 = a10 * a02
    d_p12 = d_a10 * a02 + a10 * d_a02
    p20 = a20 * (m22 + n00)
    d_p20 = d_a20 * (m22 + n00) + a20 * (d_m22 + d_n00)
    p21 = a20 * a01
    d_p21 = d_a20 * a01 + a20 * d_a01
    p22 = m22 * n22 + a02 * a20
    d_p22 = d_m22 * n22 + m22 * d_n22 + d_a02 * a20 + a02 * d_a20

    bar_leading = o00 + o11 + o22
    d_bar_leading = d_o00 + d_o11 + d_o22
    bar_first = (
        o00 * m00
        + o01 * a01
        + o02 * a02
        + o10 * a10
        + o11 * m11
        + o20 * a20
        + o22 * m22
    )
    d_bar_first = (
        d_o00 * m00
        + o00 * d_m00
        + d_o01 * a01
        + o01 * d_a01
        + d_o02 * a02
        + o02 * d_a02
        + d_o10 * a10
        + o10 * d_a10
        + d_o11 * m11
        + o11 * d_m11
        + d_o20 * a20
        + o20 * d_a20
        + d_o22 * m22
        + o22 * d_m22
    )
    bar_second = (
        o00 * p00
        + o01 * p01
        + o02 * p02
        + o10 * p10
        + o11 * p11
        + o12 * p12
        + o20 * p20
        + o21 * p21
        + o22 * p22
    )
    d_bar_second = (
        d_o00 * p00
        + o00 * d_p00
        + d_o01 * p01
        + o01 * d_p01
        + d_o02 * p02
        + o02 * d_p02
        + d_o10 * p10
        + o10 * d_p10
        + d_o11 * p11
        + o11 * d_p11
        + d_o12 * p12
        + o12 * d_p12
        + d_o20 * p20
        + o20 * d_p20
        + d_o21 * p21
        + o21 * d_p21
        + d_o22 * p22
        + o22 * d_p22
    )

    z00 = second * o00
    d_z00 = d_second * o00 + second * d_o00
    z01 = second * o01
    d_z01 = d_second * o01 + second * d_o01
    z02 = second * o02
    d_z02 = d_second * o02 + second * d_o02
    z10 = second * o10
    d_z10 = d_second * o10 + second * d_o10
    z11 = second * o11
    d_z11 = d_second * o11 + second * d_o11
    z12 = second * o12
    d_z12 = d_second * o12 + second * d_o12
    z20 = second * o20
    d_z20 = d_second * o20 + second * d_o20
    z21 = second * o21
    d_z21 = d_second * o21 + second * d_o21
    z22 = second * o22
    d_z22 = d_second * o22 + second * d_o22

    # ``z @ n^T``, the product's reverse onto the first factor.
    u00 = z00 * n00 + z01 * a01 + z02 * a02
    d_u00 = (
        d_z00 * n00
        + z00 * d_n00
        + d_z01 * a01
        + z01 * d_a01
        + d_z02 * a02
        + z02 * d_a02
    )
    u01 = z00 * a10 + z01 * n11
    d_u01 = d_z00 * a10 + z00 * d_a10 + d_z01 * n11 + z01 * d_n11
    u02 = z00 * a20 + z02 * n22
    d_u02 = d_z00 * a20 + z00 * d_a20 + d_z02 * n22 + z02 * d_n22
    u10 = z10 * n00 + z11 * a01 + z12 * a02
    d_u10 = (
        d_z10 * n00
        + z10 * d_n00
        + d_z11 * a01
        + z11 * d_a01
        + d_z12 * a02
        + z12 * d_a02
    )
    u11 = z10 * a10 + z11 * n11
    d_u11 = d_z10 * a10 + z10 * d_a10 + d_z11 * n11 + z11 * d_n11
    u20 = z20 * n00 + z21 * a01 + z22 * a02
    d_u20 = (
        d_z20 * n00
        + z20 * d_n00
        + d_z21 * a01
        + z21 * d_a01
        + d_z22 * a02
        + z22 * d_a02
    )
    u22 = z20 * a20 + z22 * n22
    d_u22 = d_z20 * a20 + z20 * d_a20 + d_z22 * n22 + z22 * d_n22

    # ``m^T @ z``, onto the second.
    w00 = m00 * z00 + a10 * z10 + a20 * z20
    d_w00 = (
        d_m00 * z00
        + m00 * d_z00
        + d_a10 * z10
        + a10 * d_z10
        + d_a20 * z20
        + a20 * d_z20
    )
    w01 = m00 * z01 + a10 * z11 + a20 * z21
    d_w01 = (
        d_m00 * z01
        + m00 * d_z01
        + d_a10 * z11
        + a10 * d_z11
        + d_a20 * z21
        + a20 * d_z21
    )
    w02 = m00 * z02 + a10 * z12 + a20 * z22
    d_w02 = (
        d_m00 * z02
        + m00 * d_z02
        + d_a10 * z12
        + a10 * d_z12
        + d_a20 * z22
        + a20 * d_z22
    )
    w10 = a01 * z00 + m11 * z10
    d_w10 = d_a01 * z00 + a01 * d_z00 + d_m11 * z10 + m11 * d_z10
    w11 = a01 * z01 + m11 * z11
    d_w11 = d_a01 * z01 + a01 * d_z01 + d_m11 * z11 + m11 * d_z11
    w20 = a02 * z00 + m22 * z20
    d_w20 = d_a02 * z00 + a02 * d_z00 + d_m22 * z20 + m22 * d_z20
    w22 = a02 * z02 + m22 * z22
    d_w22 = d_a02 * z02 + a02 * d_z02 + d_m22 * z22 + m22 * d_z22

    bar_low = bar_leading * leading - first * (o00 + o11 + o22) - (u00 + u11 + u22)
    d_bar_low = (
        d_bar_leading * leading
        + bar_leading * d_leading
        - d_first * (o00 + o11 + o22)
        - first * (d_o00 + d_o11 + d_o22)
        - (d_u00 + d_u11 + d_u22)
    )
    bar_middle = -(w00 + w11 + w22)
    d_bar_middle = -(d_w00 + d_w11 + d_w22)
    bar_high = 0.0 * a00
    d_bar_high = 0.0 * a00

    span = high - low
    positive = span > 0.0
    bar_upper = bar_second / guarded
    d_bar_upper = (d_bar_second - bar_upper * d_guarded) / guarded
    bar_first = bar_first - bar_upper
    d_bar_first = d_bar_first - d_bar_upper
    bar_span = tl.where(positive, -bar_upper * second, 0.0)
    d_bar_span = tl.where(positive, -d_bar_upper * second - bar_upper * d_second, 0.0)
    bar_high = bar_high + bar_span
    d_bar_high = d_bar_high + d_bar_span
    bar_low = bar_low - bar_span
    d_bar_low = d_bar_low - d_bar_span

    (
        from_first_low,
        d_from_first_low,
        from_first_middle,
        d_from_first_middle,
    ) = _exp_difference_adjoint_jvp(
        low,
        d_low,
        middle,
        d_middle,
        leading,
        d_leading,
        centre,
        d_centre,
        bar_first,
        d_bar_first,
    )
    (
        from_upper_middle,
        d_from_upper_middle,
        from_upper_high,
        d_from_upper_high,
    ) = _exp_difference_adjoint_jvp(
        middle,
        d_middle,
        high,
        d_high,
        centre,
        d_centre,
        trailing,
        d_trailing,
        bar_upper,
        d_bar_upper,
    )
    bar_low = bar_low + from_first_low
    d_bar_low = d_bar_low + d_from_first_low
    bar_middle = bar_middle + from_first_middle + from_upper_middle
    d_bar_middle = d_bar_middle + d_from_first_middle + d_from_upper_middle
    bar_high = bar_high + from_upper_high
    d_bar_high = d_bar_high + d_from_upper_high

    # The three roots come off one angle a third of a turn apart, and the
    # cosine puts them in a fixed order: the last turn is the lowest, the
    # first the highest, whatever the angle is.
    swing_low = angle - 2.0 * _TURN_THIRD
    swing_middle = angle - _TURN_THIRD
    cos_low = tl.cos(swing_low)
    cos_middle = tl.cos(swing_middle)
    cos_high = tl.cos(angle)
    sin_low = tl.sin(swing_low)
    sin_middle = tl.sin(swing_middle)
    sin_high = tl.sin(angle)
    bar_radius = 2.0 * (
        cos_low * bar_low + cos_middle * bar_middle + cos_high * bar_high
    )
    d_bar_radius = 2.0 * (
        cos_low * d_bar_low
        + cos_middle * d_bar_middle
        + cos_high * d_bar_high
        - d_angle * (sin_low * bar_low + sin_middle * bar_middle + sin_high * bar_high)
    )
    swept = sin_low * bar_low + sin_middle * bar_middle + sin_high * bar_high
    d_swept = (
        sin_low * d_bar_low
        + sin_middle * d_bar_middle
        + sin_high * d_bar_high
        + d_angle * (cos_low * bar_low + cos_middle * bar_middle + cos_high * bar_high)
    )
    bar_angle = -2.0 * radius * swept
    d_bar_angle = -2.0 * (d_radius * swept + radius * d_swept)
    turn_roots = bar_low + bar_middle + bar_high
    d_turn_roots = d_bar_low + d_bar_middle + d_bar_high

    # ``acos`` is clamped, and where it is the angle no longer moves with the
    # cubic's argument -- which is what keeps a double root differentiable.
    d_argument = tl.where(inside_limit, d_raw, 0.0)
    inner = 1.0 - argument * argument
    d_inner = -2.0 * argument * d_argument
    stem = tl.sqrt(tl.maximum(inner, 1e-300))
    tilt = -1.0 / (3.0 * stem)
    d_tilt = d_inner / (6.0 * stem * stem * stem)
    bar_raw = tl.where(inside_limit, bar_angle * tilt, 0.0)
    d_bar_raw = tl.where(inside_limit, d_bar_angle * tilt + bar_angle * d_tilt, 0.0)
    safe_radius = tl.where(radius > 1e-30, radius, 1.0)
    d_safe_radius = tl.where(radius > 1e-30, d_radius, 0.0)
    safe_cube = safe_radius * safe_radius * safe_radius
    d_safe_cube = 3.0 * safe_radius * safe_radius * d_safe_radius
    determinant_roots = 0.5 * bar_raw / safe_cube
    d_determinant_roots = (
        0.5 * d_bar_raw - determinant_roots * d_safe_cube
    ) / safe_cube
    pull = tl.where(inside_limit, -3.0 * raw * bar_raw / safe_radius, 0.0)
    d_pull = tl.where(
        inside_limit,
        (-3.0 * (d_raw * bar_raw + raw * d_bar_raw) - pull * d_safe_radius)
        / safe_radius,
        0.0,
    )
    bar_radius = bar_radius + pull
    d_bar_radius = d_bar_radius + d_pull
    minors_roots = -bar_radius / (6.0 * safe_radius)
    d_minors_roots = (-d_bar_radius - minors_roots * 6.0 * d_safe_radius) / (
        6.0 * safe_radius
    )

    # --- the branch chosen on the cotangents, not on the way in ---
    if narrow:
        bar_a00, d_bar_a00 = v00, d_v00
        bar_a01, d_bar_a01 = v01, d_v01
        bar_a02, d_bar_a02 = v02, d_v02
        bar_a10, d_bar_a10 = v10, d_v10
        bar_a11, d_bar_a11 = v11, d_v11
        bar_a20, d_bar_a20 = v20, d_v20
        bar_a22, d_bar_a22 = v22, d_v22
        bar_third, d_bar_third = turn_series, d_turn_series
        bar_minors, d_bar_minors = minors_series, d_minors_series
        bar_determinant, d_bar_determinant = (determinant_series, d_determinant_series)
    else:
        close = -2.0 * minors < _SPREAD_CUT * _SPREAD_CUT
        bar_a00 = tl.where(close, v00, first * o00 + u00 + w00)
        d_bar_a00 = tl.where(
            close, d_v00, d_first * o00 + first * d_o00 + d_u00 + d_w00
        )
        bar_a01 = tl.where(close, v01, first * o01 + u01 + w01)
        d_bar_a01 = tl.where(
            close, d_v01, d_first * o01 + first * d_o01 + d_u01 + d_w01
        )
        bar_a02 = tl.where(close, v02, first * o02 + u02 + w02)
        d_bar_a02 = tl.where(
            close, d_v02, d_first * o02 + first * d_o02 + d_u02 + d_w02
        )
        bar_a10 = tl.where(close, v10, first * o10 + u10 + w10)
        d_bar_a10 = tl.where(
            close, d_v10, d_first * o10 + first * d_o10 + d_u10 + d_w10
        )
        bar_a11 = tl.where(close, v11, first * o11 + u11 + w11)
        d_bar_a11 = tl.where(
            close, d_v11, d_first * o11 + first * d_o11 + d_u11 + d_w11
        )
        bar_a20 = tl.where(close, v20, first * o20 + u20 + w20)
        d_bar_a20 = tl.where(
            close, d_v20, d_first * o20 + first * d_o20 + d_u20 + d_w20
        )
        bar_a22 = tl.where(close, v22, first * o22 + u22 + w22)
        d_bar_a22 = tl.where(
            close, d_v22, d_first * o22 + first * d_o22 + d_u22 + d_w22
        )
        bar_third = tl.where(close, turn_series, turn_roots)
        d_bar_third = tl.where(close, d_turn_series, d_turn_roots)
        bar_minors = tl.where(close, minors_series, minors_roots)
        d_bar_minors = tl.where(close, d_minors_series, d_minors_roots)
        bar_determinant = tl.where(close, determinant_series, determinant_roots)
        d_bar_determinant = tl.where(close, d_determinant_series, d_determinant_roots)

    # --- the two invariants back onto the shifted generator ---
    cofactor00 = s11 * s22
    d_cofactor00 = d_s11 * s22 + s11 * d_s22
    cofactor11 = s00 * s22 - a02 * a20
    d_cofactor11 = d_s00 * s22 + s00 * d_s22 - d_a02 * a20 - a02 * d_a20
    cofactor22 = s00 * s11 - a01 * a10
    d_cofactor22 = d_s00 * s11 + s00 * d_s11 - d_a01 * a10 - a01 * d_a10
    shift00 = bar_minors * (s11 + s22) + bar_determinant * cofactor00
    d_shift00 = (
        d_bar_minors * (s11 + s22)
        + bar_minors * (d_s11 + d_s22)
        + d_bar_determinant * cofactor00
        + bar_determinant * d_cofactor00
    )
    shift11 = bar_minors * (s00 + s22) + bar_determinant * cofactor11
    d_shift11 = (
        d_bar_minors * (s00 + s22)
        + bar_minors * (d_s00 + d_s22)
        + d_bar_determinant * cofactor11
        + bar_determinant * d_cofactor11
    )
    shift22 = bar_minors * (s00 + s11) + bar_determinant * cofactor22
    d_shift22 = (
        d_bar_minors * (s00 + s11)
        + bar_minors * (d_s00 + d_s11)
        + d_bar_determinant * cofactor22
        + bar_determinant * d_cofactor22
    )
    shift01 = -a10 * (bar_minors + bar_determinant * s22)
    d_shift01 = -d_a10 * (bar_minors + bar_determinant * s22) - a10 * (
        d_bar_minors + d_bar_determinant * s22 + bar_determinant * d_s22
    )
    shift10 = -a01 * (bar_minors + bar_determinant * s22)
    d_shift10 = -d_a01 * (bar_minors + bar_determinant * s22) - a01 * (
        d_bar_minors + d_bar_determinant * s22 + bar_determinant * d_s22
    )
    shift02 = -a20 * (bar_minors + bar_determinant * s11)
    d_shift02 = -d_a20 * (bar_minors + bar_determinant * s11) - a20 * (
        d_bar_minors + d_bar_determinant * s11 + bar_determinant * d_s11
    )
    shift20 = -a02 * (bar_minors + bar_determinant * s11)
    d_shift20 = -d_a02 * (bar_minors + bar_determinant * s11) - a02 * (
        d_bar_minors + d_bar_determinant * s11 + bar_determinant * d_s11
    )
    bar_a00 = bar_a00 + shift00
    d_bar_a00 = d_bar_a00 + d_shift00
    bar_a01 = bar_a01 + shift01
    d_bar_a01 = d_bar_a01 + d_shift01
    bar_a02 = bar_a02 + shift02
    d_bar_a02 = d_bar_a02 + d_shift02
    bar_a10 = bar_a10 + shift10
    d_bar_a10 = d_bar_a10 + d_shift10
    bar_a11 = bar_a11 + shift11
    d_bar_a11 = d_bar_a11 + d_shift11
    bar_a20 = bar_a20 + shift20
    d_bar_a20 = d_bar_a20 + d_shift20
    bar_a22 = bar_a22 + shift22
    d_bar_a22 = d_bar_a22 + d_shift22
    bar_third = bar_third - (shift00 + shift11 + shift22)
    d_bar_third = d_bar_third - (d_shift00 + d_shift11 + d_shift22)
    bar_a00 = bar_a00 + bar_third / 3.0
    d_bar_a00 = d_bar_a00 + d_bar_third / 3.0
    bar_a11 = bar_a11 + bar_third / 3.0
    d_bar_a11 = d_bar_a11 + d_bar_third / 3.0
    bar_a22 = bar_a22 + bar_third / 3.0
    d_bar_a22 = d_bar_a22 + d_bar_third / 3.0

    # --- the generator back onto the rates, the fractions and the interval ---
    step = dt.to(work)
    d_step = d_dt.to(work)
    rate_b = exchange_b.to(work)
    d_rate_b = d_exchange_b.to(work)
    rate_c = exchange_c.to(work)
    d_rate_c = d_exchange_c.to(work)
    kab = rate_b * pool_b
    d_kab = d_rate_b * pool_b + rate_b * d_pool_b
    kba = rate_b * free
    d_kba = d_rate_b * free + rate_b * d_free
    kac = rate_c * pool_c
    d_kac = d_rate_c * pool_c + rate_c * d_pool_c
    kca = rate_c * free
    d_kca = d_rate_c * free + rate_c * d_free
    row_a = -kab - kac - r1_free.to(work)
    d_row_a = -d_kab - d_kac - d_r1_free.to(work)
    row_b = -kba - r1_pool_b.to(work)
    d_row_b = -d_kba - d_r1_pool_b.to(work)
    row_c = -kca - r1_bound.to(work)
    d_row_c = -d_kca - d_r1_bound.to(work)

    bar_step = (
        row_a * bar_a00
        + kba * bar_a01
        + kca * bar_a02
        + kab * bar_a10
        + row_b * bar_a11
        + kac * bar_a20
        + row_c * bar_a22
    )
    d_bar_step = (
        d_row_a * bar_a00
        + row_a * d_bar_a00
        + d_kba * bar_a01
        + kba * d_bar_a01
        + d_kca * bar_a02
        + kca * d_bar_a02
        + d_kab * bar_a10
        + kab * d_bar_a10
        + d_row_b * bar_a11
        + row_b * d_bar_a11
        + d_kac * bar_a20
        + kac * d_bar_a20
        + d_row_c * bar_a22
        + row_c * d_bar_a22
    )
    bar_kab = step * (bar_a10 - bar_a00)
    d_bar_kab = d_step * (bar_a10 - bar_a00) + step * (d_bar_a10 - d_bar_a00)
    bar_kba = step * (bar_a01 - bar_a11)
    d_bar_kba = d_step * (bar_a01 - bar_a11) + step * (d_bar_a01 - d_bar_a11)
    bar_kac = step * (bar_a20 - bar_a00)
    d_bar_kac = d_step * (bar_a20 - bar_a00) + step * (d_bar_a20 - d_bar_a00)
    bar_kca = step * (bar_a02 - bar_a22)
    d_bar_kca = d_step * (bar_a02 - bar_a22) + step * (d_bar_a02 - d_bar_a22)

    whole_free = bar_free + rate_b * bar_kba + rate_c * bar_kca
    d_whole_free = (
        d_bar_free
        + d_rate_b * bar_kba
        + rate_b * d_bar_kba
        + d_rate_c * bar_kca
        + rate_c * d_bar_kca
    )
    whole_pool_b = bar_pool_b + rate_b * bar_kab
    d_whole_pool_b = d_bar_pool_b + d_rate_b * bar_kab + rate_b * d_bar_kab
    whole_pool_c = bar_pool_c + rate_c * bar_kac
    d_whole_pool_c = d_bar_pool_c + d_rate_c * bar_kac + rate_c * d_bar_kac

    return (
        (-step * bar_a00).to(tl.float32),
        (-step * bar_a11).to(tl.float32),
        (-step * bar_a22).to(tl.float32),
        (pool_b * bar_kab + free * bar_kba).to(tl.float32),
        (pool_c * bar_kac + free * bar_kca).to(tl.float32),
        (whole_pool_b - whole_free).to(tl.float32),
        (whole_pool_c - whole_free).to(tl.float32),
        bar_step.to(tl.float32),
        bar_damp.to(tl.float32),
        (-d_step * bar_a00 - step * d_bar_a00).to(tl.float32),
        (-d_step * bar_a11 - step * d_bar_a11).to(tl.float32),
        (-d_step * bar_a22 - step * d_bar_a22).to(tl.float32),
        (
            d_pool_b * bar_kab
            + pool_b * d_bar_kab
            + d_free * bar_kba
            + free * d_bar_kba
        ).to(tl.float32),
        (
            d_pool_c * bar_kac
            + pool_c * d_bar_kac
            + d_free * bar_kca
            + free * d_bar_kca
        ).to(tl.float32),
        (d_whole_pool_b - d_whole_free).to(tl.float32),
        (d_whole_pool_c - d_whole_free).to(tl.float32),
        d_bar_step.to(tl.float32),
        d_bar_damp.to(tl.float32),
    )


@triton.jit
def _complex_sqrt(real, imag):
    """A square root of a complex number carried as a pair of floats.

    Which of the two it is does not matter here: the only thing that reads it
    is even in it, so the branch cut the principal root carries is unreachable.
    """
    magnitude = tl.sqrt(real * real + imag * imag)
    root_real = tl.sqrt(tl.maximum(0.5 * (magnitude + real), 0.0))
    root_imag = tl.sqrt(tl.maximum(0.5 * (magnitude - real), 0.0))
    return root_real, tl.where(imag < 0.0, -root_imag, root_imag)


@triton.jit
def _complex_exp(real, imag):
    """``e^z`` for ``z`` carried as a pair of floats."""
    scale = tl.exp(real)
    return scale * tl.cos(imag), scale * tl.sin(imag)


@triton.jit
def _two_pool_transverse_step(
    r2_free, r2_bound, exchange, bound, free, shift_hz, dt, attenuation
):
    """The transverse operator of two chemically exchanging pools.

    ``expm((K - diag(R2) - 2 pi i diag(df)) t)``, in the closed form the
    longitudinal pair uses -- the numbers have become complex, the algebra has
    not. Returned as the four entries, each a pair of floats.

    A semisolid pool holds a share of the voxel without carrying any transverse
    magnetization, so it is absent from this 2x2 and present in ``free`` -- how
    much free water the exchange sees.

    There is no recovery term: transverse magnetization relaxes toward zero.
    """
    kab = exchange * bound
    kba = exchange * free
    l11 = (-kab - r2_free) * dt
    l12 = kba * dt
    l21 = kab * dt
    l22 = (-kba - r2_bound) * dt
    # Only pool b's offset appears: pool a sits at whatever off-resonance the
    # free precession already carries the whole voxel through.
    l22_imag = -2.0 * 3.141592653589793 * shift_hz * dt

    trace_real = 0.5 * (l11 + l22)
    trace_imag = 0.5 * l22_imag
    gap_real = 0.5 * (l11 - l22)
    gap_imag = -0.5 * l22_imag
    square_real = gap_real * gap_real - gap_imag * gap_imag + l12 * l21
    square_imag = 2.0 * gap_real * gap_imag

    root_real, root_imag = _complex_sqrt(square_real, square_imag)
    upper_real, upper_imag = _complex_exp(
        trace_real + root_real, trace_imag + root_imag
    )
    lower_real, lower_imag = _complex_exp(
        trace_real - root_real, trace_imag - root_imag
    )
    cos_real = 0.5 * (upper_real + lower_real)
    cos_imag = 0.5 * (upper_imag + lower_imag)

    # ``sinh(d)/d`` by series near the origin, where the root has no
    # derivative of its own.
    turning = square_real * square_real + square_imag * square_imag > 1e-24
    half_real = 0.5 * (upper_real - lower_real)
    half_imag = 0.5 * (upper_imag - lower_imag)
    guard = tl.where(turning, root_real * root_real + root_imag * root_imag, 1.0)
    divided_real = (half_real * root_real + half_imag * root_imag) / guard
    divided_imag = (half_imag * root_real - half_real * root_imag) / guard
    plain_real, plain_imag = _complex_exp(trace_real, trace_imag)
    square2_real = square_real * square_real - square_imag * square_imag
    square2_imag = 2.0 * square_real * square_imag
    poly_real = 1.0 + square_real / 6.0 + square2_real / 120.0
    poly_imag = square_imag / 6.0 + square2_imag / 120.0
    series_real = plain_real * poly_real - plain_imag * poly_imag
    series_imag = plain_real * poly_imag + plain_imag * poly_real
    scale_real = tl.where(turning, divided_real, series_real)
    scale_imag = tl.where(turning, divided_imag, series_imag)

    off_real = scale_real * gap_real - scale_imag * gap_imag
    off_imag = scale_real * gap_imag + scale_imag * gap_real
    return (
        attenuation * (cos_real + off_real),
        attenuation * (cos_imag + off_imag),
        attenuation * scale_real * l12,
        attenuation * scale_imag * l12,
        attenuation * scale_real * l21,
        attenuation * scale_imag * l21,
        attenuation * (cos_real - off_real),
        attenuation * (cos_imag - off_imag),
    )


@triton.jit
def _complex_sqrt_jvp(real, imag, d_real, d_imag):
    """A complex square root and its directional derivative.

    The derivative divides by twice the root, so a caller keeps the origin --
    where the root has none -- on its series branch.
    """
    root_real, root_imag = _complex_sqrt(real, imag)
    guard = 2.0 * (root_real * root_real + root_imag * root_imag)
    live = guard > 0.0
    guarded = tl.where(live, guard, 1.0)
    # dz / (2 w) == dz * conj(2 w) / |2 w|^2
    tangent_real = tl.where(
        live, (d_real * root_real + d_imag * root_imag) / guarded, 0.0
    )
    tangent_imag = tl.where(
        live, (d_imag * root_real - d_real * root_imag) / guarded, 0.0
    )
    return root_real, root_imag, tangent_real, tangent_imag


@triton.jit
def _complex_exp_jvp(real, imag, d_real, d_imag):
    """``e^z`` and its directional derivative, which is ``e^z`` times it."""
    value_real, value_imag = _complex_exp(real, imag)
    return (
        value_real,
        value_imag,
        value_real * d_real - value_imag * d_imag,
        value_real * d_imag + value_imag * d_real,
    )


@triton.jit
def _two_pool_transverse_step_jvp(
    r2_free,
    d_r2_free,
    r2_bound,
    d_r2_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    free,
    d_free,
    shift_hz,
    d_shift_hz,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
):
    """The transverse operator and its directional derivative.

    The same closed form :func:`_two_pool_transverse_step` evaluates, carried
    alongside a tangent. Returned as the four entries then their four tangents,
    each a pair of floats.
    """
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    l11 = (-kab - r2_free) * dt
    d_l11 = (-d_kab - d_r2_free) * dt + (-kab - r2_free) * d_dt
    l12 = kba * dt
    d_l12 = d_kba * dt + kba * d_dt
    l21 = kab * dt
    d_l21 = d_kab * dt + kab * d_dt
    l22 = (-kba - r2_bound) * dt
    d_l22 = (-d_kba - d_r2_bound) * dt + (-kba - r2_bound) * d_dt
    turn = -2.0 * 3.141592653589793
    l22_imag = turn * shift_hz * dt
    d_l22_imag = turn * (d_shift_hz * dt + shift_hz * d_dt)

    trace_real = 0.5 * (l11 + l22)
    d_trace_real = 0.5 * (d_l11 + d_l22)
    trace_imag = 0.5 * l22_imag
    d_trace_imag = 0.5 * d_l22_imag
    gap_real = 0.5 * (l11 - l22)
    d_gap_real = 0.5 * (d_l11 - d_l22)
    gap_imag = -0.5 * l22_imag
    d_gap_imag = -0.5 * d_l22_imag

    square_real = gap_real * gap_real - gap_imag * gap_imag + l12 * l21
    d_square_real = (
        2.0 * gap_real * d_gap_real
        - 2.0 * gap_imag * d_gap_imag
        + d_l12 * l21
        + l12 * d_l21
    )
    square_imag = 2.0 * gap_real * gap_imag
    d_square_imag = 2.0 * (d_gap_real * gap_imag + gap_real * d_gap_imag)

    root_real, root_imag, d_root_real, d_root_imag = _complex_sqrt_jvp(
        square_real, square_imag, d_square_real, d_square_imag
    )
    upper_real, upper_imag, d_upper_real, d_upper_imag = _complex_exp_jvp(
        trace_real + root_real,
        trace_imag + root_imag,
        d_trace_real + d_root_real,
        d_trace_imag + d_root_imag,
    )
    lower_real, lower_imag, d_lower_real, d_lower_imag = _complex_exp_jvp(
        trace_real - root_real,
        trace_imag - root_imag,
        d_trace_real - d_root_real,
        d_trace_imag - d_root_imag,
    )
    cos_real = 0.5 * (upper_real + lower_real)
    cos_imag = 0.5 * (upper_imag + lower_imag)
    d_cos_real = 0.5 * (d_upper_real + d_lower_real)
    d_cos_imag = 0.5 * (d_upper_imag + d_lower_imag)

    turning = square_real * square_real + square_imag * square_imag > 1e-24
    half_real = 0.5 * (upper_real - lower_real)
    half_imag = 0.5 * (upper_imag - lower_imag)
    d_half_real = 0.5 * (d_upper_real - d_lower_real)
    d_half_imag = 0.5 * (d_upper_imag - d_lower_imag)
    norm = root_real * root_real + root_imag * root_imag
    guard = tl.where(turning, norm, 1.0)
    d_norm = tl.where(
        turning, 2.0 * (root_real * d_root_real + root_imag * d_root_imag), 0.0
    )
    # (a / w) with w complex: a * conj(w) / |w|^2, differentiated as a quotient.
    top_real = half_real * root_real + half_imag * root_imag
    top_imag = half_imag * root_real - half_real * root_imag
    d_top_real = (
        d_half_real * root_real
        + half_real * d_root_real
        + d_half_imag * root_imag
        + half_imag * d_root_imag
    )
    d_top_imag = (
        d_half_imag * root_real
        + half_imag * d_root_real
        - d_half_real * root_imag
        - half_real * d_root_imag
    )
    divided_real = top_real / guard
    divided_imag = top_imag / guard
    d_divided_real = (d_top_real - divided_real * d_norm) / guard
    d_divided_imag = (d_top_imag - divided_imag * d_norm) / guard

    plain_real, plain_imag, d_plain_real, d_plain_imag = _complex_exp_jvp(
        trace_real, trace_imag, d_trace_real, d_trace_imag
    )
    square2_real = square_real * square_real - square_imag * square_imag
    square2_imag = 2.0 * square_real * square_imag
    d_square2_real = (
        2.0 * square_real * d_square_real - 2.0 * square_imag * d_square_imag
    )
    d_square2_imag = 2.0 * (d_square_real * square_imag + square_real * d_square_imag)
    poly_real = 1.0 + square_real / 6.0 + square2_real / 120.0
    poly_imag = square_imag / 6.0 + square2_imag / 120.0
    d_poly_real = d_square_real / 6.0 + d_square2_real / 120.0
    d_poly_imag = d_square_imag / 6.0 + d_square2_imag / 120.0
    series_real = plain_real * poly_real - plain_imag * poly_imag
    series_imag = plain_real * poly_imag + plain_imag * poly_real
    d_series_real = (
        d_plain_real * poly_real
        + plain_real * d_poly_real
        - d_plain_imag * poly_imag
        - plain_imag * d_poly_imag
    )
    d_series_imag = (
        d_plain_real * poly_imag
        + plain_real * d_poly_imag
        + d_plain_imag * poly_real
        + plain_imag * d_poly_real
    )

    scale_real = tl.where(turning, divided_real, series_real)
    scale_imag = tl.where(turning, divided_imag, series_imag)
    d_scale_real = tl.where(turning, d_divided_real, d_series_real)
    d_scale_imag = tl.where(turning, d_divided_imag, d_series_imag)

    off_real = scale_real * gap_real - scale_imag * gap_imag
    off_imag = scale_real * gap_imag + scale_imag * gap_real
    d_off_real = (
        d_scale_real * gap_real
        + scale_real * d_gap_real
        - d_scale_imag * gap_imag
        - scale_imag * d_gap_imag
    )
    d_off_imag = (
        d_scale_real * gap_imag
        + scale_real * d_gap_imag
        + d_scale_imag * gap_real
        + scale_imag * d_gap_real
    )

    e11_real = cos_real + off_real
    e11_imag = cos_imag + off_imag
    d_e11_real = d_cos_real + d_off_real
    d_e11_imag = d_cos_imag + d_off_imag
    e22_real = cos_real - off_real
    e22_imag = cos_imag - off_imag
    d_e22_real = d_cos_real - d_off_real
    d_e22_imag = d_cos_imag - d_off_imag
    return (
        attenuation * e11_real,
        attenuation * e11_imag,
        attenuation * scale_real * l12,
        attenuation * scale_imag * l12,
        attenuation * scale_real * l21,
        attenuation * scale_imag * l21,
        attenuation * e22_real,
        attenuation * e22_imag,
        d_attenuation * e11_real + attenuation * d_e11_real,
        d_attenuation * e11_imag + attenuation * d_e11_imag,
        d_attenuation * scale_real * l12
        + attenuation * (d_scale_real * l12 + scale_real * d_l12),
        d_attenuation * scale_imag * l12
        + attenuation * (d_scale_imag * l12 + scale_imag * d_l12),
        d_attenuation * scale_real * l21
        + attenuation * (d_scale_real * l21 + scale_real * d_l21),
        d_attenuation * scale_imag * l21
        + attenuation * (d_scale_imag * l21 + scale_imag * d_l21),
        d_attenuation * e22_real + attenuation * d_e22_real,
        d_attenuation * e22_imag + attenuation * d_e22_imag,
    )


@triton.jit
def _lineshape_at(lineshape, offset_hz, bins, step):
    """How well the bound pool absorbs a pulse this far off its resonance.

    Cubic Hermite between the two knots bracketing the offset, taken in
    magnitude because the lineshape is even, and clamped at the far end. Each
    knot is two floats -- the value then its slope -- so the two a read needs
    are four contiguous ones.
    """
    last = bins - 1
    scaled = tl.minimum(tl.abs(offset_hz) / step, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    return (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )


@triton.jit
def _two_pool_step_jvp(
    r1_free,
    d_r1_free,
    r1_bound,
    d_r1_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
):
    """The two-pool operator and its directional derivative.

    The same closed form :func:`_two_pool_step` evaluates, carried alongside a
    tangent. Returned as the six outputs then their six tangents.
    """
    free = 1.0 - bound
    d_free = -d_bound
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    l11 = (-kab - r1_free) * dt
    d_l11 = (-d_kab - d_r1_free) * dt + (-kab - r1_free) * d_dt
    l12 = kba * dt
    d_l12 = d_kba * dt + kba * d_dt
    l21 = kab * dt
    d_l21 = d_kab * dt + kab * d_dt
    l22 = (-kba - r1_bound) * dt
    d_l22 = (-d_kba - d_r1_bound) * dt + (-kba - r1_bound) * d_dt

    half_trace = 0.5 * (l11 + l22)
    d_half_trace = 0.5 * (d_l11 + d_l22)
    half_gap = 0.5 * (l11 - l22)
    d_half_gap = 0.5 * (d_l11 - d_l22)
    square = half_gap * half_gap + l12 * l21
    d_square = 2.0 * half_gap * d_half_gap + d_l12 * l21 + l12 * d_l21

    turning = square > 1e-12
    root = tl.sqrt(tl.maximum(square, 0.0))
    guarded = tl.where(turning, root, 1.0)
    d_root = tl.where(turning, 0.5 * d_square / guarded, 0.0)

    upper = tl.exp(half_trace + root)
    d_upper = upper * (d_half_trace + d_root)
    lower = tl.exp(half_trace - root)
    d_lower = lower * (d_half_trace - d_root)
    cosine = 0.5 * (upper + lower)
    d_cosine = 0.5 * (d_upper + d_lower)

    # sinh(d)/d by series where the root has no derivative of its own.
    plain = tl.exp(half_trace)
    d_plain = plain * d_half_trace
    poly = 1.0 + square / 6.0 + square * square / 120.0
    d_poly = d_square / 6.0 + square * d_square / 60.0
    scale = tl.where(turning, 0.5 * (upper - lower) / guarded, plain * poly)
    d_scale = tl.where(
        turning,
        0.5 * (d_upper - d_lower) / guarded
        - 0.5 * (upper - lower) * d_root / (guarded * guarded),
        d_plain * poly + plain * d_poly,
    )

    e11 = attenuation * (cosine + scale * half_gap)
    d_e11 = d_attenuation * (cosine + scale * half_gap) + attenuation * (
        d_cosine + d_scale * half_gap + scale * d_half_gap
    )
    e12 = attenuation * scale * l12
    d_e12 = (
        d_attenuation * scale * l12
        + attenuation * d_scale * l12
        + attenuation * scale * d_l12
    )
    e21 = attenuation * scale * l21
    d_e21 = (
        d_attenuation * scale * l21
        + attenuation * d_scale * l21
        + attenuation * scale * d_l21
    )
    e22 = attenuation * (cosine - scale * half_gap)
    d_e22 = d_attenuation * (cosine - scale * half_gap) + attenuation * (
        d_cosine - d_scale * half_gap - scale * d_half_gap
    )

    grow_free = free - (e11 * free + e12 * bound)
    d_grow_free = d_free - (d_e11 * free + e11 * d_free + d_e12 * bound + e12 * d_bound)
    grow_bound = bound - (e21 * free + e22 * bound)
    d_grow_bound = d_bound - (
        d_e21 * free + e21 * d_free + d_e22 * bound + e22 * d_bound
    )
    return (
        e11,
        e12,
        e21,
        e22,
        grow_free,
        grow_bound,
        d_e11,
        d_e12,
        d_e21,
        d_e22,
        d_grow_free,
        d_grow_bound,
    )


@triton.jit
def _lineshape_at_slope(lineshape, offset_hz, bins, step):
    """The lineshape and its derivative in the *signed* offset.

    The table covers the magnitude, so the slope changes sign with the offset;
    past the last knot the read is constant and the slope is zero.
    """
    last = bins - 1
    magnitude = tl.abs(offset_hz) / step
    scaled = tl.minimum(magnitude, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    value = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )
    direction = tl.where(offset_hz < 0.0, -1.0, 1.0)
    slope = direction * (
        (6.0 * u2 - 6.0 * u) * near / step
        + (3.0 * u2 - 4.0 * u + 1.0) * near_slope
        + (-6.0 * u2 + 6.0 * u) * far / step
        + (3.0 * u2 - 2.0 * u) * far_slope
    )
    return value, tl.where(magnitude > last, 0.0, slope)


@triton.jit
def _lineshape_at_curve(lineshape, offset_hz, bins, step):
    """The lineshape, its slope and its curvature, from the same cubic.

    The table covers the magnitude, so the slope changes sign with the offset
    and the curvature does not: an even function's second derivative is even.
    """
    last = bins - 1
    magnitude = tl.abs(offset_hz) / step
    scaled = tl.minimum(magnitude, last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    base = lower.to(tl.int64) * 2
    near = tl.load(lineshape + base)
    near_slope = tl.load(lineshape + base + 1)
    far = tl.load(lineshape + base + 2)
    far_slope = tl.load(lineshape + base + 3)
    value = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * near
        + (u3 - 2.0 * u2 + u) * step * near_slope
        + (-2.0 * u3 + 3.0 * u2) * far
        + (u3 - u2) * step * far_slope
    )
    direction = tl.where(offset_hz < 0.0, -1.0, 1.0)
    slope = direction * (
        (6.0 * u2 - 6.0 * u) * near / step
        + (3.0 * u2 - 4.0 * u + 1.0) * near_slope
        + (-6.0 * u2 + 6.0 * u) * far / step
        + (3.0 * u2 - 2.0 * u) * far_slope
    )
    curve = (
        (12.0 * u - 6.0) * near / (step * step)
        + (6.0 * u - 4.0) * near_slope / step
        + (-12.0 * u + 6.0) * far / (step * step)
        + (6.0 * u - 2.0) * far_slope / step
    )
    beyond = magnitude > last
    return (
        value,
        tl.where(beyond, 0.0, slope),
        tl.where(beyond, 0.0, curve),
    )


@triton.jit
def _two_pool_step_adjoint_jvp(
    r1_free,
    d_r1_free,
    r1_bound,
    d_r1_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
    bar_e11,
    d_bar_e11,
    bar_e12,
    d_bar_e12,
    bar_e21,
    d_bar_e21,
    bar_e22,
    d_bar_e22,
    bar_free,
    d_bar_free,
    bar_bound,
    d_bar_bound,
):
    """The reverse sweep of :func:`_two_pool_step`, carried on a direction.

    Recomputes the forward rather than carrying it across the event: the whole
    thing is a handful of transcendentals once per interval, against a state
    loop that runs per dephasing order.

    Where the discriminant is small the value is still formed from the two
    eigenvalues -- a sum, which loses nothing -- but the derivative is taken
    from the series, because ``d cosh(d)/d(d^2)`` reached through
    ``(e^{t+d} - e^{t-d})/2d`` is a cancellation divided by a small number.

    Returned as the gradients w.r.t. ``(r1_free, r1_bound, exchange, bound,
    dt, attenuation)`` then their six tangents.
    """
    free = 1.0 - bound
    d_free = -d_bound
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    l11 = (-kab - r1_free) * dt
    d_l11 = (-d_kab - d_r1_free) * dt + (-kab - r1_free) * d_dt
    l12 = kba * dt
    d_l12 = d_kba * dt + kba * d_dt
    l21 = kab * dt
    d_l21 = d_kab * dt + kab * d_dt
    l22 = (-kba - r1_bound) * dt
    d_l22 = (-d_kba - d_r1_bound) * dt + (-kba - r1_bound) * d_dt

    half_trace = 0.5 * (l11 + l22)
    d_half_trace = 0.5 * (d_l11 + d_l22)
    half_gap = 0.5 * (l11 - l22)
    d_half_gap = 0.5 * (d_l11 - d_l22)
    square = half_gap * half_gap + l12 * l21
    d_square = 2.0 * half_gap * d_half_gap + d_l12 * l21 + l12 * d_l21

    turning = square > 1e-12
    root = tl.sqrt(tl.maximum(square, 0.0))
    guarded = tl.where(turning, root, 1.0)
    d_root = tl.where(turning, 0.5 * d_square / guarded, 0.0)
    upper = tl.exp(half_trace + root)
    d_upper = upper * (d_half_trace + d_root)
    lower = tl.exp(half_trace - root)
    d_lower = lower * (d_half_trace - d_root)
    cosine = 0.5 * (upper + lower)
    d_cosine = 0.5 * (d_upper + d_lower)
    plain = tl.exp(half_trace)
    d_plain = plain * d_half_trace
    poly = 1.0 + square / 6.0 + square * square / 120.0
    d_poly = d_square / 6.0 + square * d_square / 60.0
    scale = tl.where(turning, 0.5 * (upper - lower) / guarded, plain * poly)
    d_scale = tl.where(
        turning,
        0.5 * (d_upper - d_lower) / guarded
        - 0.5 * (upper - lower) * d_root / (guarded * guarded),
        d_plain * poly + plain * d_poly,
    )

    bare11 = cosine + scale * half_gap
    d_bare11 = d_cosine + d_scale * half_gap + scale * d_half_gap
    bare12 = scale * l12
    d_bare12 = d_scale * l12 + scale * d_l12
    bare21 = scale * l21
    d_bare21 = d_scale * l21 + scale * d_l21
    bare22 = cosine - scale * half_gap
    d_bare22 = d_cosine - d_scale * half_gap - scale * d_half_gap

    # The recovery reaches the operator's four entries and the two fractions.
    carried11 = bar_e11 - bar_free * free
    d_carried11 = d_bar_e11 - (d_bar_free * free + bar_free * d_free)
    carried12 = bar_e12 - bar_free * bound
    d_carried12 = d_bar_e12 - (d_bar_free * bound + bar_free * d_bound)
    carried21 = bar_e21 - bar_bound * free
    d_carried21 = d_bar_e21 - (d_bar_bound * free + bar_bound * d_free)
    carried22 = bar_e22 - bar_bound * bound
    d_carried22 = d_bar_e22 - (d_bar_bound * bound + bar_bound * d_bound)

    e11 = attenuation * bare11
    d_e11 = d_attenuation * bare11 + attenuation * d_bare11
    e12 = attenuation * bare12
    d_e12 = d_attenuation * bare12 + attenuation * d_bare12
    e21 = attenuation * bare21
    d_e21 = d_attenuation * bare21 + attenuation * d_bare21
    e22 = attenuation * bare22
    d_e22 = d_attenuation * bare22 + attenuation * d_bare22

    back_free = bar_free * (1.0 - e11) - bar_bound * e21
    d_back_free = (
        d_bar_free * (1.0 - e11)
        - bar_free * d_e11
        - (d_bar_bound * e21 + bar_bound * d_e21)
    )
    back_bound = bar_bound * (1.0 - e22) - bar_free * e12
    d_back_bound = (
        d_bar_bound * (1.0 - e22)
        - bar_bound * d_e22
        - (d_bar_free * e12 + bar_free * d_e12)
    )

    back_attenuation = (
        carried11 * bare11
        + carried12 * bare12
        + carried21 * bare21
        + carried22 * bare22
    )
    d_back_attenuation = (
        d_carried11 * bare11
        + carried11 * d_bare11
        + d_carried12 * bare12
        + carried12 * d_bare12
        + d_carried21 * bare21
        + carried21 * d_bare21
        + d_carried22 * bare22
        + carried22 * d_bare22
    )

    scaled11 = attenuation * carried11
    d_scaled11 = d_attenuation * carried11 + attenuation * d_carried11
    scaled12 = attenuation * carried12
    d_scaled12 = d_attenuation * carried12 + attenuation * d_carried12
    scaled21 = attenuation * carried21
    d_scaled21 = d_attenuation * carried21 + attenuation * d_carried21
    scaled22 = attenuation * carried22
    d_scaled22 = d_attenuation * carried22 + attenuation * d_carried22

    bar_cosine = scaled11 + scaled22
    d_bar_cosine = d_scaled11 + d_scaled22
    gap = scaled11 - scaled22
    d_gap = d_scaled11 - d_scaled22
    bar_scale = gap * half_gap + scaled12 * l12 + scaled21 * l21
    d_bar_scale = (
        d_gap * half_gap
        + gap * d_half_gap
        + d_scaled12 * l12
        + scaled12 * d_l12
        + d_scaled21 * l21
        + scaled21 * d_l21
    )
    bar_half_gap = scale * gap
    d_bar_half_gap = d_scale * gap + scale * d_gap
    bar_l12 = scale * scaled12
    d_bar_l12 = d_scale * scaled12 + scale * d_scaled12
    bar_l21 = scale * scaled21
    d_bar_l21 = d_scale * scaled21 + scale * d_scaled21

    series_trace = bar_cosine * cosine + bar_scale * scale
    d_series_trace = (
        d_bar_cosine * cosine
        + bar_cosine * d_cosine
        + d_bar_scale * scale
        + bar_scale * d_scale
    )
    cosine_poly = 0.5 + square / 12.0
    d_cosine_poly = d_square / 12.0
    scale_poly = 1.0 / 6.0 + square / 60.0
    d_scale_poly = d_square / 60.0
    series_square = plain * (bar_cosine * cosine_poly + bar_scale * scale_poly)
    d_series_square = d_plain * (
        bar_cosine * cosine_poly + bar_scale * scale_poly
    ) + plain * (
        d_bar_cosine * cosine_poly
        + bar_cosine * d_cosine_poly
        + d_bar_scale * scale_poly
        + bar_scale * d_scale_poly
    )

    inverse = tl.where(turning, 1.0 / guarded, 0.0)
    d_inverse = tl.where(turning, -d_root / (guarded * guarded), 0.0)
    bar_upper = 0.5 * (bar_cosine + bar_scale * inverse)
    d_bar_upper = 0.5 * (d_bar_cosine + d_bar_scale * inverse + bar_scale * d_inverse)
    bar_lower = 0.5 * (bar_cosine - bar_scale * inverse)
    d_bar_lower = 0.5 * (d_bar_cosine - d_bar_scale * inverse - bar_scale * d_inverse)
    root_trace = bar_upper * upper + bar_lower * lower
    d_root_trace = (
        d_bar_upper * upper
        + bar_upper * d_upper
        + d_bar_lower * lower
        + bar_lower * d_lower
    )
    bar_root = bar_upper * upper - bar_lower * lower - bar_scale * scale * inverse
    d_bar_root = (
        d_bar_upper * upper
        + bar_upper * d_upper
        - d_bar_lower * lower
        - bar_lower * d_lower
        - (
            d_bar_scale * scale * inverse
            + bar_scale * d_scale * inverse
            + bar_scale * scale * d_inverse
        )
    )
    root_square = 0.5 * bar_root * inverse
    d_root_square = 0.5 * (d_bar_root * inverse + bar_root * d_inverse)

    bar_half_trace = tl.where(turning, root_trace, series_trace)
    d_bar_half_trace = tl.where(turning, d_root_trace, d_series_trace)
    bar_square = tl.where(turning, root_square, series_square)
    d_bar_square = tl.where(turning, d_root_square, d_series_square)

    bar_half_gap += 2.0 * bar_square * half_gap
    d_bar_half_gap += 2.0 * (d_bar_square * half_gap + bar_square * d_half_gap)
    bar_l12 += bar_square * l21
    d_bar_l12 += d_bar_square * l21 + bar_square * d_l21
    bar_l21 += bar_square * l12
    d_bar_l21 += d_bar_square * l12 + bar_square * d_l12

    bar_l11 = 0.5 * (bar_half_trace + bar_half_gap)
    d_bar_l11 = 0.5 * (d_bar_half_trace + d_bar_half_gap)
    bar_l22 = 0.5 * (bar_half_trace - bar_half_gap)
    d_bar_l22 = 0.5 * (d_bar_half_trace - d_bar_half_gap)

    bar_kab = (bar_l21 - bar_l11) * dt
    d_bar_kab = (d_bar_l21 - d_bar_l11) * dt + (bar_l21 - bar_l11) * d_dt
    bar_kba = (bar_l12 - bar_l22) * dt
    d_bar_kba = (d_bar_l12 - d_bar_l22) * dt + (bar_l12 - bar_l22) * d_dt
    back_dt = (
        bar_l11 * (-kab - r1_free)
        + bar_l12 * kba
        + bar_l21 * kab
        + bar_l22 * (-kba - r1_bound)
    )
    d_back_dt = (
        d_bar_l11 * (-kab - r1_free)
        + bar_l11 * (-d_kab - d_r1_free)
        + d_bar_l12 * kba
        + bar_l12 * d_kba
        + d_bar_l21 * kab
        + bar_l21 * d_kab
        + d_bar_l22 * (-kba - r1_bound)
        + bar_l22 * (-d_kba - d_r1_bound)
    )

    back_bound += bar_kab * exchange
    d_back_bound += d_bar_kab * exchange + bar_kab * d_exchange
    back_free += bar_kba * exchange
    d_back_free += d_bar_kba * exchange + bar_kba * d_exchange

    return (
        -bar_l11 * dt,
        -bar_l22 * dt,
        bar_kab * bound + bar_kba * free,
        back_bound - back_free,
        back_dt,
        back_attenuation,
        -(d_bar_l11 * dt + bar_l11 * d_dt),
        -(d_bar_l22 * dt + bar_l22 * d_dt),
        d_bar_kab * bound + bar_kab * d_bound + d_bar_kba * free + bar_kba * d_free,
        d_back_bound - d_back_free,
        d_back_dt,
        d_back_attenuation,
    )


@triton.jit
def _washout_jvp(rate, rate_tangent, dt, dt_tangent):
    """The same fraction and its directional derivative."""
    fraction = rate * dt
    live = fraction < 1.0
    return (
        tl.where(live, 1.0 - fraction, 0.0),
        tl.where(live, -(rate_tangent * dt + rate * dt_tangent), 0.0),
    )


@triton.jit
def _damping_jvp(rate, rate_tangent, dt, dt_tangent, order):
    """Diffusion damping and its directional derivative, per state order."""
    b_factor = rate * dt
    b_tangent = rate_tangent * dt + rate * dt_tangent
    squared = order * order
    transverse_weight = squared + order + 0.3333333333333333
    damp_z = tl.exp(-b_factor * squared)
    damp_t = tl.exp(-b_factor * transverse_weight)
    return (
        damp_z,
        damp_z * (-b_tangent * squared),
        damp_t,
        damp_t * (-b_tangent * transverse_weight),
    )


@triton.jit
def _complex_mul(a_real, a_imag, b_real, b_imag):
    return a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real


@triton.jit
def _dual_mul(a_vr, a_vi, a_tr, a_ti, b_vr, b_vi, b_tr, b_ti):
    """Product of two dual complex numbers."""
    value_real, value_imag = _complex_mul(a_vr, a_vi, b_vr, b_vi)
    left_real, left_imag = _complex_mul(a_tr, a_ti, b_vr, b_vi)
    right_real, right_imag = _complex_mul(a_vr, a_vi, b_tr, b_ti)
    return value_real, value_imag, left_real + right_real, left_imag + right_imag


@triton.jit
def _dual_scale(scale_value, scale_tangent, vr, vi, tr, ti):
    """A real dual number times a complex one."""
    return (
        scale_value * vr,
        scale_value * vi,
        scale_tangent * vr + scale_value * tr,
        scale_tangent * vi + scale_value * ti,
    )


@triton.jit
def _dual_times_i(vr, vi, tr, ti):
    return -vi, vr, -ti, tr


@triton.jit
def _dual_real_conj_mul(a_vr, a_vi, a_tr, a_ti, b_vr, b_vi, b_tr, b_ti):
    """``real_part(conj(a) * b)``, the contraction an adjoint asks for."""
    value = a_vr * b_vr + a_vi * b_vi
    tangent = a_tr * b_vr + a_ti * b_vi + a_vr * b_tr + a_vi * b_ti
    return value, tangent


@triton.jit
def _dual_back(entry, d_entry, spin_vr, spin_vi, spin_tr, spin_ti, br, bi, tr, ti):
    """``conj(entry * spin)`` against a cotangent, entry and spin both dual.

    One row of a real mixing operator carried through the per-order turn, which
    is what a longitudinal cotangent walks back through.
    """
    return _dual_mul(
        entry * spin_vr,
        -(entry * spin_vi),
        d_entry * spin_vr + entry * spin_tr,
        -(d_entry * spin_vi + entry * spin_ti),
        br,
        bi,
        tr,
        ti,
    )


@triton.jit
def _dual_polar(angle_value, angle_tangent):
    """``exp(i * angle)`` for a real dual angle."""
    cosine = tl.cos(angle_value)
    sine = tl.sin(angle_value)
    return cosine, sine, -sine * angle_tangent, cosine * angle_tangent


@triton.jit
def _shift_adjoint(
    plus_bar_real,
    plus_bar_imag,
    minus_bar_real,
    minus_bar_imag,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    """Transpose of ``_shift``.

    The conjugate refill at order zero sends the incoming plus adjoint back
    onto minus, conjugated, at the index the minus shift moves it to.
    """
    carry_real = tl.where(state_mask, _first(plus_bar_real, state), 0.0)
    carry_imag = -tl.where(state_mask, _first(plus_bar_imag, state), 0.0)
    forward = (state + 1 < state_count) & state_mask
    backward = (state > 0) & state_mask
    shifted_pr = tl.where(forward, _down(plus_bar_real, state), 0.0)
    shifted_pi = tl.where(forward, _down(plus_bar_imag, state), 0.0)
    shifted_mr = tl.where(backward, _up(minus_bar_real, state), 0.0)
    shifted_mi = tl.where(backward, _up(minus_bar_imag, state), 0.0)
    shifted_mr = tl.where(state == 1, shifted_mr + carry_real, shifted_mr)
    shifted_mi = tl.where(state == 1, shifted_mi + carry_imag, shifted_mi)
    return shifted_pr, shifted_pi, shifted_mr, shifted_mi


@triton.jit
def _shift_real_adjoint(
    plus_bar,
    minus_bar,
    state,
    state_mask,
    state_count: tl.constexpr,
):
    """Transpose of ``_shift_real``.

    The ``a0 = -b0`` coupling sends the incoming plus adjoint back onto minus,
    at the index the minus shift moves it to.
    """
    carry = -tl.where(state_mask, _first(plus_bar, state), 0.0)
    shifted_plus = tl.where(
        (state + 1 < state_count) & state_mask, _down(plus_bar, state), 0.0
    )
    shifted_minus = tl.where((state > 0) & state_mask, _up(minus_bar, state), 0.0)
    shifted_minus = tl.where(state == 1, shifted_minus + carry, shifted_minus)
    return shifted_plus, shifted_minus


@triton.jit
def _table_row(profile_index, event, location, locations: tl.constexpr):
    """Which row of the stacked tables this pulse reads.

    Its own shape's block of ``locations`` rows, then the voxel's place along
    the slice.
    """
    return tl.load(profile_index + event).to(tl.int64) * locations + location


@triton.jit
def _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, mask):
    """The rotation a pulse performs at this voxel, read rather than read off.

    A tabulated pair covers a shape's every pulse because a static array
    reaches the rotation through one complex scalar; this one is integrated per
    pulse per voxel, so there is nothing to interpolate and the read is four
    floats. The row runs per train and per event, as the flip does.
    """
    row = tl.load(pair_index + event_base + event).to(tl.int64)
    entry = pairs + (row * atom_count + atom) * 4
    return (
        tl.load(entry + 0, mask=mask, other=1.0),
        tl.load(entry + 1, mask=mask, other=0.0),
        tl.load(entry + 2, mask=mask, other=0.0),
        tl.load(entry + 3, mask=mask, other=0.0),
    )


@triton.jit
def _profile_pair(profile, row, theta, bins, step):
    """The Cayley-Klein pair the transition table holds at this flip angle.

    Cubic Hermite between the two knots bracketing ``theta``, clamped at both
    ends: a cubic run off its grid leaves the unit circle. Each knot is eight
    floats -- the pair then its slope, real before imaginary -- so the two a
    read needs are sixteen contiguous ones.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step

    base = (row * bins + lower.to(tl.int64)) * 8
    pair = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        pair = pair + (h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,)
    return pair


@triton.jit
def _profile_pair_slope(profile, row, theta, bins, step):
    """The pair and its derivative in the flip angle, from the same cubic.

    The derivative of a Hermite segment is another polynomial in the same four
    knot values, so reading both costs one extra combination rather than a
    second table. Returned interleaved: each component's value then its slope,
    in the order ``a`` real, ``a`` imaginary, ``b`` real, ``b`` imaginary.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step
    # d/dtheta is d/du over the knot spacing.
    g00 = (6.0 * u2 - 6.0 * u) / step
    g10 = 3.0 * u2 - 4.0 * u + 1.0
    g01 = (6.0 * u - 6.0 * u2) / step
    g11 = 3.0 * u2 - 2.0 * u

    base = (row * bins + lower.to(tl.int64)) * 8
    read = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        read = read + (
            h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,
            g00 * near + g10 * near_slope + g01 * far + g11 * far_slope,
        )
    return read


@triton.jit
def _profile_pair_curve(profile, row, theta, bins, step):
    """The pair, its slope and its curvature in the flip angle.

    The second-order pass differentiates the read twice, and a Hermite segment
    is a cubic, so all three come from the same four knot values. Returned in
    threes per component: value, slope, curvature.
    """
    last = bins - 1
    scaled = tl.minimum(tl.maximum(theta / step, 0.0), last + 0.0)
    lower = tl.minimum(tl.floor(scaled), last - 1.0)
    u = scaled - lower
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * step
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * step
    g00 = (6.0 * u2 - 6.0 * u) / step
    g10 = 3.0 * u2 - 4.0 * u + 1.0
    g01 = (6.0 * u - 6.0 * u2) / step
    g11 = 3.0 * u2 - 2.0 * u
    c00 = (12.0 * u - 6.0) / (step * step)
    c10 = (6.0 * u - 4.0) / step
    c01 = (6.0 - 12.0 * u) / (step * step)
    c11 = (6.0 * u - 2.0) / step

    base = (row * bins + lower.to(tl.int64)) * 8
    read = ()
    for component in tl.static_range(4):
        near = tl.load(profile + base + component)
        near_slope = tl.load(profile + base + 4 + component)
        far = tl.load(profile + base + 8 + component)
        far_slope = tl.load(profile + base + 12 + component)
        read = read + (
            h00 * near + h10 * near_slope + h01 * far + h11 * far_slope,
            g00 * near + g10 * near_slope + g01 * far + g11 * far_slope,
            c00 * near + c10 * near_slope + c01 * far + c11 * far_slope,
        )
    return read


@triton.jit
def _dual_conj(z):
    """A dual complex number's conjugate, both halves."""
    return (z[0], -z[1], z[2], -z[3])


@triton.jit
def _dual_weigh(z, factor):
    """A dual complex number scaled by a real constant."""
    return (factor * z[0], factor * z[1], factor * z[2], factor * z[3])


@triton.jit
def _dual_sum(first, second, third, fourth):
    """Four dual complex numbers added."""
    return (
        first[0] + second[0] + third[0] + fourth[0],
        first[1] + second[1] + third[1] + fourth[1],
        first[2] + second[2] + third[2] + fourth[2],
        first[3] + second[3] + third[3] + fourth[3],
    )


@triton.jit
def _dual_product(x, y):
    """Two dual complex numbers multiplied."""
    return _dual_mul(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3])


@triton.jit
def _dual_add(x, y):
    """Two dual complex numbers added."""
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])


@triton.jit
def _dual_subtract(x, y):
    """One dual complex number less another."""
    return (x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3])


@triton.jit
def _dual_reciprocal(z):
    """``1/z`` for a dual complex number, and the tangent that goes with it."""
    norm = z[0] * z[0] + z[1] * z[1]
    guard = tl.where(norm > 0.0, norm, 1.0)
    value_real = z[0] / guard
    value_imag = -z[1] / guard
    square_real, square_imag = _complex_mul(
        value_real, value_imag, value_real, value_imag
    )
    tangent_real, tangent_imag = _complex_mul(square_real, square_imag, z[2], z[3])
    return value_real, value_imag, -tangent_real, -tangent_imag


@triton.jit
def _two_pool_transverse_adjoint_jvp(
    r2_free,
    d_r2_free,
    r2_bound,
    d_r2_bound,
    exchange,
    d_exchange,
    bound,
    d_bound,
    free,
    d_free,
    shift_hz,
    d_shift_hz,
    dt,
    d_dt,
    attenuation,
    d_attenuation,
    bar_e11,
    bar_e12,
    bar_e21,
    bar_e22,
):
    """The reverse sweep of :func:`_two_pool_transverse_step_jvp`.

    Every step from the four generator entries to the four operator entries is
    holomorphic, so the sweep is the longitudinal one with complex numbers in
    place of real ones and no conjugates along the way. That holds because the
    cotangents arrive as row covectors -- ``bar_e`` is the number with ``dL =
    Re(bar_e de)`` -- and only where a complex intermediate meets one of the
    real inputs is a real part taken.

    Takes the four cotangents as dual complex quadruples and returns the seven
    real gradients, each as a value and a tangent.
    """
    zero = 0.0 * dt
    kab = exchange * bound
    d_kab = d_exchange * bound + exchange * d_bound
    kba = exchange * free
    d_kba = d_exchange * free + exchange * d_free
    turn = -2.0 * 3.141592653589793
    l11 = (
        (-kab - r2_free) * dt,
        zero,
        (-d_kab - d_r2_free) * dt + (-kab - r2_free) * d_dt,
        zero,
    )
    l12 = (kba * dt, zero, d_kba * dt + kba * d_dt, zero)
    l21 = (kab * dt, zero, d_kab * dt + kab * d_dt, zero)
    l22 = (
        (-kba - r2_bound) * dt,
        turn * (shift_hz * dt),
        (-d_kba - d_r2_bound) * dt + (-kba - r2_bound) * d_dt,
        turn * (d_shift_hz * dt + shift_hz * d_dt),
    )

    half_trace = _dual_weigh(_dual_add(l11, l22), 0.5)
    half_gap = _dual_weigh(_dual_subtract(l11, l22), 0.5)
    square = _dual_add(_dual_product(half_gap, half_gap), _dual_product(l12, l21))
    delta = _complex_sqrt_jvp(square[0], square[1], square[2], square[3])
    upper = _complex_exp_jvp(*_dual_add(half_trace, delta))
    lower = _complex_exp_jvp(*_dual_subtract(half_trace, delta))
    plain = _complex_exp_jvp(half_trace[0], half_trace[1], half_trace[2], half_trace[3])
    cosine = _dual_weigh(_dual_add(upper, lower), 0.5)

    turning = square[0] * square[0] + square[1] * square[1] > 1e-24
    # Off the branch the reciprocal is taken at one instead, so a discriminant
    # at the origin never divides anything the series answer then discards.
    guarded = (
        tl.where(turning, delta[0], 1.0),
        tl.where(turning, delta[1], 0.0),
        tl.where(turning, delta[2], 0.0),
        tl.where(turning, delta[3], 0.0),
    )
    inverse = _dual_reciprocal(guarded)
    divided = _dual_product(_dual_weigh(_dual_subtract(upper, lower), 0.5), inverse)
    square2 = _dual_product(square, square)
    poly = (
        1.0 + square[0] / 6.0 + square2[0] / 120.0,
        square[1] / 6.0 + square2[1] / 120.0,
        square[2] / 6.0 + square2[2] / 120.0,
        square[3] / 6.0 + square2[3] / 120.0,
    )
    series = _dual_product(plain, poly)
    scale = (
        tl.where(turning, divided[0], series[0]),
        tl.where(turning, divided[1], series[1]),
        tl.where(turning, divided[2], series[2]),
        tl.where(turning, divided[3], series[3]),
    )

    off = _dual_product(scale, half_gap)
    bare_11 = _dual_add(cosine, off)
    bare_12 = _dual_product(scale, l12)
    bare_21 = _dual_product(scale, l21)
    bare_22 = _dual_subtract(cosine, off)

    bar_attenuation = _dual_sum(
        _dual_product(bar_e11, bare_11),
        _dual_product(bar_e12, bare_12),
        _dual_product(bar_e21, bare_21),
        _dual_product(bar_e22, bare_22),
    )
    scaled_11 = _dual_scale(attenuation, d_attenuation, *bar_e11)
    scaled_12 = _dual_scale(attenuation, d_attenuation, *bar_e12)
    scaled_21 = _dual_scale(attenuation, d_attenuation, *bar_e21)
    scaled_22 = _dual_scale(attenuation, d_attenuation, *bar_e22)

    diagonal = _dual_subtract(scaled_11, scaled_22)
    bar_cosine = _dual_add(scaled_11, scaled_22)
    bar_scale = _dual_add(
        _dual_product(diagonal, half_gap),
        _dual_add(_dual_product(scaled_12, l12), _dual_product(scaled_21, l21)),
    )
    bar_half_gap = _dual_product(scale, diagonal)
    bar_l12 = _dual_product(scale, scaled_12)
    bar_l21 = _dual_product(scale, scaled_21)

    series_trace = _dual_add(
        _dual_product(bar_cosine, cosine), _dual_product(bar_scale, scale)
    )
    series_square = _dual_product(
        plain,
        _dual_add(
            _dual_product(
                bar_cosine,
                (
                    0.5 + square[0] / 12.0,
                    square[1] / 12.0,
                    square[2] / 12.0,
                    square[3] / 12.0,
                ),
            ),
            _dual_product(
                bar_scale,
                (
                    0.16666666666666666 + square[0] / 60.0,
                    square[1] / 60.0,
                    square[2] / 60.0,
                    square[3] / 60.0,
                ),
            ),
        ),
    )
    bar_upper = _dual_weigh(
        _dual_add(bar_cosine, _dual_product(bar_scale, inverse)), 0.5
    )
    bar_lower = _dual_weigh(
        _dual_subtract(bar_cosine, _dual_product(bar_scale, inverse)), 0.5
    )
    split_trace = _dual_add(
        _dual_product(bar_upper, upper), _dual_product(bar_lower, lower)
    )
    bar_delta = _dual_subtract(
        _dual_subtract(
            _dual_product(bar_upper, upper), _dual_product(bar_lower, lower)
        ),
        _dual_product(_dual_product(bar_scale, scale), inverse),
    )
    split_square = _dual_weigh(_dual_product(bar_delta, inverse), 0.5)
    bar_half_trace = (
        tl.where(turning, split_trace[0], series_trace[0]),
        tl.where(turning, split_trace[1], series_trace[1]),
        tl.where(turning, split_trace[2], series_trace[2]),
        tl.where(turning, split_trace[3], series_trace[3]),
    )
    bar_square = (
        tl.where(turning, split_square[0], series_square[0]),
        tl.where(turning, split_square[1], series_square[1]),
        tl.where(turning, split_square[2], series_square[2]),
        tl.where(turning, split_square[3], series_square[3]),
    )

    bar_half_gap = _dual_add(
        bar_half_gap, _dual_weigh(_dual_product(bar_square, half_gap), 2.0)
    )
    bar_l12 = _dual_add(bar_l12, _dual_product(bar_square, l21))
    bar_l21 = _dual_add(bar_l21, _dual_product(bar_square, l12))
    bar_l11 = _dual_weigh(_dual_add(bar_half_trace, bar_half_gap), 0.5)
    bar_l22 = _dual_weigh(_dual_subtract(bar_half_trace, bar_half_gap), 0.5)

    bar_kab = _dual_scale(dt, d_dt, *_dual_subtract(bar_l21, bar_l11))
    bar_kba = _dual_scale(dt, d_dt, *_dual_subtract(bar_l12, bar_l22))
    slope_22 = (
        -kba - r2_bound,
        turn * shift_hz,
        -d_kba - d_r2_bound,
        turn * d_shift_hz,
    )
    bar_dt = _dual_sum(
        _dual_scale(-kab - r2_free, -d_kab - d_r2_free, *bar_l11),
        _dual_scale(kba, d_kba, *bar_l12),
        _dual_scale(kab, d_kab, *bar_l21),
        _dual_product(slope_22, bar_l22),
    )

    r2_free_bar = _dual_scale(-dt, -d_dt, *bar_l11)
    r2_bound_bar = _dual_scale(-dt, -d_dt, *bar_l22)
    exchange_bar = _dual_add(
        _dual_scale(bound, d_bound, *bar_kab),
        _dual_scale(free, d_free, *bar_kba),
    )
    bound_bar = _dual_scale(exchange, d_exchange, *bar_kab)
    free_bar = _dual_scale(exchange, d_exchange, *bar_kba)
    shift_bar = _dual_scale(turn * dt, turn * d_dt, *_dual_times_i(*bar_l22))
    return (
        r2_free_bar[0],
        r2_free_bar[2],
        r2_bound_bar[0],
        r2_bound_bar[2],
        exchange_bar[0],
        exchange_bar[2],
        bound_bar[0],
        bound_bar[2],
        free_bar[0],
        free_bar[2],
        shift_bar[0],
        shift_bar[2],
        bar_dt[0],
        bar_dt[2],
        bar_attenuation[0],
        bar_attenuation[2],
    )


@triton.jit
def _store_pair_cotangent(
    grad_value,
    grad_tangent,
    pair_index,
    event_base,
    event,
    atom,
    atom_count,
    turning,
    mask,
    state_mask,
    grad_a,
    grad_b,
):
    """Send the cotangent on one pulse's rotation to its row.

    Summed over the dephasing orders first: the pair multiplies every one of
    them, so what reaches the row is the sum. The value plane is the adjoint
    and the tangent plane its own derivative, which is the split every other
    gradient here takes.
    """
    row = tl.load(pair_index + event_base + event).to(tl.int64)
    entry = (row * atom_count + atom) * 4
    # The block is padded to a power of two, and the orders past the last one
    # carry whatever the sweep left there -- so the sum is taken over the
    # orders that exist rather than over the block.
    keep = turning & state_mask
    tl.atomic_add(
        grad_value + entry + 0,
        tl.sum(tl.where(keep, grad_a[0], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_value + entry + 1,
        tl.sum(tl.where(keep, grad_a[1], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_value + entry + 2,
        tl.sum(tl.where(keep, grad_b[0], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_value + entry + 3,
        tl.sum(tl.where(keep, grad_b[1], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_tangent + entry + 0,
        tl.sum(tl.where(keep, grad_a[2], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_tangent + entry + 1,
        tl.sum(tl.where(keep, grad_a[3], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_tangent + entry + 2,
        tl.sum(tl.where(keep, grad_b[2], 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_tangent + entry + 3,
        tl.sum(tl.where(keep, grad_b[3], 0.0), axis=1)[:, None],
        mask=mask,
    )


@triton.jit
def _store_pair_gradient(
    grad_pair,
    pair_index,
    event_base,
    event,
    atom,
    atom_count,
    turning,
    mask,
    state_mask,
    grad_ar,
    grad_ai,
    grad_br,
    grad_bi,
):
    """Send the cotangent on one pulse's rotation to its row.

    Summed over the dephasing orders first: the pair multiplies every one of
    them, so what reaches the row is the sum. The block is padded to a power of
    two and the orders past the last carry whatever the sweep left there, so
    the sum is taken over the orders that exist rather than over the block.
    """
    row = tl.load(pair_index + event_base + event).to(tl.int64)
    entry = (row * atom_count + atom) * 4
    keep = turning & state_mask
    tl.atomic_add(
        grad_pair + entry + 0,
        tl.sum(tl.where(keep, grad_ar, 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_pair + entry + 1,
        tl.sum(tl.where(keep, grad_ai, 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_pair + entry + 2,
        tl.sum(tl.where(keep, grad_br, 0.0), axis=1)[:, None],
        mask=mask,
    )
    tl.atomic_add(
        grad_pair + entry + 3,
        tl.sum(tl.where(keep, grad_bi, 0.0), axis=1)[:, None],
        mask=mask,
    )


@triton.jit
def _dynamic_pair_dual_at(
    pairs,
    pair_direction,
    pair_index,
    event_base,
    event,
    atom,
    atom_count,
    mask,
    phi_value,
    phi_tangent,
    directed: tl.constexpr,
):
    """The rotation and the direction along it, with the phase applied.

    Shaped exactly as :func:`_profiled_pair_dual` returns, so the spinor
    operator and its adjoint read one from the other without knowing which
    they were handed. A pass that follows no direction holds the rotation
    still, and ``directed`` keeps the read for one out of the kernel.
    """
    held = _dynamic_pair_at(
        pairs, pair_index, event_base, event, atom, atom_count, mask
    )
    still = held[0] * 0.0
    moved = (still, still, still, still)
    if directed:
        moved = _dynamic_pair_at(
            pair_direction,
            pair_index,
            event_base,
            event,
            atom,
            atom_count,
            mask,
        )
    a = (held[0], held[1], moved[0], moved[1])
    b = (held[2], held[3], moved[2], moved[3])
    turn = _dual_polar(-phi_value, -phi_tangent)
    return a, _dual_product(b, turn)


@triton.jit
def _profiled_pair_dual(
    profile,
    row,
    alpha_value,
    alpha_tangent,
    phi_value,
    phi_tangent,
    bins,
    step,
):
    """The pair a shaped pulse turns through, and its slope, as duals.

    The flip angle carries the tangent into the table, so the pair's tangent is
    the stored slope and the slope's own tangent is the segment's curvature.
    The RF phase turns the axis once the pair is out, and so reaches ``b``.
    """
    read = _profile_pair_curve(profile, row, alpha_value, bins, step)
    a = (read[0], read[3], read[1] * alpha_tangent, read[4] * alpha_tangent)
    slope_a = (read[1], read[4], read[2] * alpha_tangent, read[5] * alpha_tangent)
    b = (read[6], read[9], read[7] * alpha_tangent, read[10] * alpha_tangent)
    slope_b = (read[7], read[10], read[8] * alpha_tangent, read[11] * alpha_tangent)
    turn = _dual_polar(-phi_value, -phi_tangent)
    return a, _dual_product(b, turn), slope_a, _dual_product(slope_b, turn)


@triton.jit
def _spinor_adjoint_dual(a, b, sp, sm, rz, pb, mb, zb):
    """The spinor rotation's adjoint, on dual numbers.

    Returns the cotangent on the Cayley-Klein pair and the three state
    cotangents sent back through the conjugate transpose. Every entry of the
    matrix is a product of two factors drawn from the pair and its conjugate,
    so the pair's two Wirtinger halves are linear in the outer product of the
    seed with the state the rotation acted on -- which is why this is a closed
    form rather than a differentiated matrix.
    """
    t00, t01, t02, t10, t11, t12, t20, t21, t22 = _spinor_coefficients(
        a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]
    )

    conj_pb = _dual_conj(pb)
    conj_mb = _dual_conj(mb)
    conj_zb = _dual_conj(zb)
    m00 = _dual_product(conj_pb, sp)
    m01 = _dual_product(conj_pb, sm)
    m02 = _dual_product(conj_pb, rz)
    m10 = _dual_product(conj_mb, sp)
    m11 = _dual_product(conj_mb, sm)
    m12 = _dual_product(conj_mb, rz)
    m20 = _dual_product(conj_zb, sp)
    m21 = _dual_product(conj_zb, sm)
    m22 = _dual_product(conj_zb, rz)

    conj_a = _dual_conj(a)
    conj_b = _dual_conj(b)
    holding_conj_a = _dual_sum(
        _dual_weigh(_dual_product(a, m11), 2.0),
        _dual_weigh(_dual_product(b, m12), -2.0),
        _dual_product(conj_b, m21),
        _dual_product(conj_a, m22),
    )
    holding_a = _dual_sum(
        _dual_weigh(_dual_product(conj_a, m00), 2.0),
        _dual_weigh(_dual_product(conj_b, m02), -2.0),
        _dual_product(b, m20),
        _dual_product(a, m22),
    )
    holding_conj_b = _dual_sum(
        _dual_weigh(_dual_product(b, m10), -2.0),
        _dual_weigh(_dual_product(a, m12), -2.0),
        _dual_product(conj_a, m20),
        _dual_weigh(_dual_product(conj_b, m22), -1.0),
    )
    holding_b = _dual_sum(
        _dual_weigh(_dual_product(conj_b, m01), -2.0),
        _dual_weigh(_dual_product(conj_a, m02), -2.0),
        _dual_product(a, m21),
        _dual_weigh(_dual_product(b, m22), -1.0),
    )
    zero = _dual_weigh(m00, 0.0)
    grad_a = _dual_sum(_dual_conj(holding_conj_a), holding_a, zero, zero)
    grad_b = _dual_sum(_dual_conj(holding_conj_b), holding_b, zero, zero)

    next_pb = _dual_sum(
        _dual_product(_dual_conj(t00), pb),
        _dual_product(_dual_conj(t10), mb),
        _dual_product(_dual_conj(t20), zb),
        zero,
    )
    next_mb = _dual_sum(
        _dual_product(_dual_conj(t01), pb),
        _dual_product(_dual_conj(t11), mb),
        _dual_product(_dual_conj(t21), zb),
        zero,
    )
    next_zb = _dual_sum(
        _dual_product(_dual_conj(t02), pb),
        _dual_product(_dual_conj(t12), mb),
        _dual_product(_dual_conj(t22), zb),
        zero,
    )
    return grad_a, grad_b, next_pb, next_mb, next_zb


@triton.jit
def _spinor_coefficients(ar, ai, br, bi, dar, dai, dbr, dbi):
    """The rotation's nine coefficients and their tangents.

    Every entry is a product of two factors drawn from the pair and its
    conjugate, so five products carry all nine: ``a^2``, ``b^2``, ``a b``,
    ``a conj(b)`` and the norm difference.
    """
    aa_r = ar * ar - ai * ai
    aa_i = 2.0 * ar * ai
    daa_r = 2.0 * (ar * dar - ai * dai)
    daa_i = 2.0 * (dar * ai + ar * dai)

    bb_r = br * br - bi * bi
    bb_i = 2.0 * br * bi
    dbb_r = 2.0 * (br * dbr - bi * dbi)
    dbb_i = 2.0 * (dbr * bi + br * dbi)

    ab_r = ar * br - ai * bi
    ab_i = ar * bi + ai * br
    dab_r = dar * br + ar * dbr - dai * bi - ai * dbi
    dab_i = dar * bi + ar * dbi + dai * br + ai * dbr

    cross_r = ar * br + ai * bi
    cross_i = ar * bi - ai * br
    dcross_r = dar * br + ar * dbr + dai * bi + ai * dbi
    dcross_i = dar * bi + ar * dbi - dai * br - ai * dbr

    t22 = ar * ar + ai * ai - br * br - bi * bi
    dt22 = 2.0 * (ar * dar + ai * dai - br * dbr - bi * dbi)

    return (
        (aa_r, -aa_i, daa_r, -daa_i),
        (-bb_r, bb_i, -dbb_r, dbb_i),
        (-2.0 * ab_r, 2.0 * ab_i, -2.0 * dab_r, 2.0 * dab_i),
        (-bb_r, -bb_i, -dbb_r, -dbb_i),
        (aa_r, aa_i, daa_r, daa_i),
        (-2.0 * ab_r, -2.0 * ab_i, -2.0 * dab_r, -2.0 * dab_i),
        (cross_r, cross_i, dcross_r, dcross_i),
        (cross_r, -cross_i, dcross_r, -dcross_i),
        (t22, 0.0 * t22, dt22, 0.0 * dt22),
    )


@triton.jit
def _rotate_spinor_dual(
    ar,
    ai,
    br,
    bi,
    dar,
    dai,
    dbr,
    dbi,
    fp_r,
    fp_i,
    fm_r,
    fm_i,
    z_r,
    z_i,
    dfp_r,
    dfp_i,
    dfm_r,
    dfm_i,
    dz_r,
    dz_i,
):
    """The spinor rotation carrying a forward-mode tangent.

    Both the states and the pair naming the rotation move, so the tangent is
    ``T dx + dT x``.
    """
    t00, t01, t02, t10, t11, t12, t20, t21, t22 = _spinor_coefficients(
        ar, ai, br, bi, dar, dai, dbr, dbi
    )

    out_pr, out_pi = _dual_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    out_mr, out_mi = _dual_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    out_zr, out_zi = _dual_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i)

    dpr, dpi = _dual_row(t00, t01, t02, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i)
    dmr, dmi = _dual_row(t10, t11, t12, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i)
    dzr, dzi = _dual_row(t20, t21, t22, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i)
    tpr, tpi = _tangent_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    tmr, tmi = _tangent_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i)
    tzr, tzi = _tangent_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i)

    return (
        out_pr,
        out_pi,
        out_mr,
        out_mi,
        out_zr,
        out_zi,
        dpr + tpr,
        dpi + tpi,
        dmr + tmr,
        dmi + tmi,
        dzr + tzr,
        dzi + tzi,
    )


@triton.jit
def _dual_row(first, second, third, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """One row of the rotation applied to the states, values only."""
    real = (
        first[0] * fp_r
        - first[1] * fp_i
        + second[0] * fm_r
        - second[1] * fm_i
        + third[0] * z_r
        - third[1] * z_i
    )
    imag = (
        first[0] * fp_i
        + first[1] * fp_r
        + second[0] * fm_i
        + second[1] * fm_r
        + third[0] * z_i
        + third[1] * z_r
    )
    return real, imag


@triton.jit
def _tangent_row(first, second, third, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """The same row built from the coefficients' tangents instead."""
    real = (
        first[2] * fp_r
        - first[3] * fp_i
        + second[2] * fm_r
        - second[3] * fm_i
        + third[2] * z_r
        - third[3] * z_i
    )
    imag = (
        first[2] * fp_i
        + first[3] * fp_r
        + second[2] * fm_i
        + second[3] * fm_r
        + third[2] * z_i
        + third[3] * z_r
    )
    return real, imag


@triton.jit
def _spinor_adjoint(
    ar, ai, br, bi, spr, spi, smr, smi, rzr, rzi, pbr, pbi, mbr, mbi, zbr, zbi
):
    """The spinor rotation's adjoint, carrying no forward direction.

    Returns the cotangent on the Cayley-Klein pair and the three state
    cotangents sent back through the conjugate transpose. Every entry of the
    matrix is a product of two factors drawn from the pair and its conjugate,
    so the pair's two Wirtinger halves are linear in the outer product of the
    seed with the state the rotation acted on -- a closed form rather than a
    differentiated matrix.
    """
    aa_r = ar * ar - ai * ai
    aa_i = 2.0 * ar * ai
    bb_r = br * br - bi * bi
    bb_i = 2.0 * br * bi
    ab_r = ar * br - ai * bi
    ab_i = ar * bi + ai * br
    cross_r = ar * br + ai * bi
    cross_i = ar * bi - ai * br
    t00_r, t00_i = aa_r, -aa_i
    t01_r, t01_i = -bb_r, bb_i
    t02_r, t02_i = -2.0 * ab_r, 2.0 * ab_i
    t10_r, t10_i = -bb_r, -bb_i
    t11_r, t11_i = aa_r, aa_i
    t12_r, t12_i = -2.0 * ab_r, -2.0 * ab_i
    t20_r, t20_i = cross_r, cross_i
    t21_r, t21_i = cross_r, -cross_i
    t22 = ar * ar + ai * ai - br * br - bi * bi

    # ``m[i][j] = conj(seed_i) * state_j``: the outer product the pair's
    # derivative is linear in.
    m00 = _complex_mul(pbr, -pbi, spr, spi)
    m01 = _complex_mul(pbr, -pbi, smr, smi)
    m02 = _complex_mul(pbr, -pbi, rzr, rzi)
    m10 = _complex_mul(mbr, -mbi, spr, spi)
    m11 = _complex_mul(mbr, -mbi, smr, smi)
    m12 = _complex_mul(mbr, -mbi, rzr, rzi)
    m20 = _complex_mul(zbr, -zbi, spr, spi)
    m21 = _complex_mul(zbr, -zbi, smr, smi)
    m22 = _complex_mul(zbr, -zbi, rzr, rzi)

    hca = _complex_mul(ar, ai, m11[0], m11[1])
    hcb = _complex_mul(br, bi, m12[0], m12[1])
    hcc = _complex_mul(br, -bi, m21[0], m21[1])
    hcd = _complex_mul(ar, -ai, m22[0], m22[1])
    holding_conj_a_r = 2.0 * hca[0] - 2.0 * hcb[0] + hcc[0] + hcd[0]
    holding_conj_a_i = 2.0 * hca[1] - 2.0 * hcb[1] + hcc[1] + hcd[1]

    ha = _complex_mul(ar, -ai, m00[0], m00[1])
    hb = _complex_mul(br, -bi, m02[0], m02[1])
    hc = _complex_mul(br, bi, m20[0], m20[1])
    hd = _complex_mul(ar, ai, m22[0], m22[1])
    holding_a_r = 2.0 * ha[0] - 2.0 * hb[0] + hc[0] + hd[0]
    holding_a_i = 2.0 * ha[1] - 2.0 * hb[1] + hc[1] + hd[1]

    ka = _complex_mul(br, bi, m10[0], m10[1])
    kb = _complex_mul(ar, ai, m12[0], m12[1])
    kc = _complex_mul(ar, -ai, m20[0], m20[1])
    kd = _complex_mul(br, -bi, m22[0], m22[1])
    holding_conj_b_r = -2.0 * ka[0] - 2.0 * kb[0] + kc[0] - kd[0]
    holding_conj_b_i = -2.0 * ka[1] - 2.0 * kb[1] + kc[1] - kd[1]

    la = _complex_mul(br, -bi, m01[0], m01[1])
    lb = _complex_mul(ar, -ai, m02[0], m02[1])
    lc = _complex_mul(ar, ai, m21[0], m21[1])
    ld = _complex_mul(br, bi, m22[0], m22[1])
    holding_b_r = -2.0 * la[0] - 2.0 * lb[0] + lc[0] - ld[0]
    holding_b_i = -2.0 * la[1] - 2.0 * lb[1] + lc[1] - ld[1]

    grad_a_r = holding_conj_a_r + holding_a_r
    grad_a_i = -holding_conj_a_i + holding_a_i
    grad_b_r = holding_conj_b_r + holding_b_r
    grad_b_i = -holding_conj_b_i + holding_b_i

    n0 = _complex_mul(t00_r, -t00_i, pbr, pbi)
    n1 = _complex_mul(t10_r, -t10_i, mbr, mbi)
    n2 = _complex_mul(t20_r, -t20_i, zbr, zbi)
    next_pr, next_pi = n0[0] + n1[0] + n2[0], n0[1] + n1[1] + n2[1]
    n0 = _complex_mul(t01_r, -t01_i, pbr, pbi)
    n1 = _complex_mul(t11_r, -t11_i, mbr, mbi)
    n2 = _complex_mul(t21_r, -t21_i, zbr, zbi)
    next_mr, next_mi = n0[0] + n1[0] + n2[0], n0[1] + n1[1] + n2[1]
    n0 = _complex_mul(t02_r, -t02_i, pbr, pbi)
    n1 = _complex_mul(t12_r, -t12_i, mbr, mbi)
    next_zr = n0[0] + n1[0] + t22 * zbr
    next_zi = n0[1] + n1[1] + t22 * zbi
    return (
        grad_a_r,
        grad_a_i,
        grad_b_r,
        grad_b_i,
        next_pr,
        next_pi,
        next_mr,
        next_mi,
        next_zr,
        next_zi,
    )


@triton.jit
def _rotate_spinor(ar, ai, br, bi, fp_r, fp_i, fm_r, fm_i, z_r, z_i):
    """The rotation named by its Cayley-Klein pair, applied to the states.

    T = [ conj(a)^2   -conj(b)^2   -2 conj(a b) ]
        [ -b^2         a^2         -2 a b       ]
        [ conj(a) b    a conj(b)   |a|^2-|b|^2  ]
    """
    aa_r = ar * ar - ai * ai
    aa_i = 2.0 * ar * ai
    bb_r = br * br - bi * bi
    bb_i = 2.0 * br * bi
    ab_r = ar * br - ai * bi
    ab_i = ar * bi + ai * br

    t00_r, t00_i = aa_r, -aa_i
    t01_r, t01_i = -bb_r, bb_i
    t02_r, t02_i = -2.0 * ab_r, 2.0 * ab_i
    t10_r, t10_i = -bb_r, -bb_i
    t11_r, t11_i = aa_r, aa_i
    t12_r, t12_i = -2.0 * ab_r, -2.0 * ab_i
    cross_r = ar * br + ai * bi
    cross_i = ar * bi - ai * br
    t20_r, t20_i = cross_r, cross_i
    t21_r, t21_i = cross_r, -cross_i
    t22 = ar * ar + ai * ai - br * br - bi * bi

    out_pr = (
        t00_r * fp_r
        - t00_i * fp_i
        + t01_r * fm_r
        - t01_i * fm_i
        + t02_r * z_r
        - t02_i * z_i
    )
    out_pi = (
        t00_r * fp_i
        + t00_i * fp_r
        + t01_r * fm_i
        + t01_i * fm_r
        + t02_r * z_i
        + t02_i * z_r
    )
    out_mr = (
        t10_r * fp_r
        - t10_i * fp_i
        + t11_r * fm_r
        - t11_i * fm_i
        + t12_r * z_r
        - t12_i * z_i
    )
    out_mi = (
        t10_r * fp_i
        + t10_i * fp_r
        + t11_r * fm_i
        + t11_i * fm_r
        + t12_r * z_i
        + t12_i * z_r
    )
    out_zr = t20_r * fp_r - t20_i * fp_i + t21_r * fm_r - t21_i * fm_i + t22 * z_r
    out_zi = t20_r * fp_i + t20_i * fp_r + t21_r * fm_i + t21_i * fm_r + t22 * z_i
    return out_pr, out_pi, out_mr, out_mi, out_zr, out_zi


@triton.jit
def _rotation_block(
    a_value,
    a_tangent,
    b_value,
    b_tangent,
    c_value,
    c_tangent,
    d_value,
    d_tangent,
    p1r,
    p1i,
    p1tr,
    p1ti,
    p2r,
    p2i,
    p2tr,
    p2ti,
    pcr,
    pci,
    pctr,
    pcti,
):
    """Seven of the nine rotation coefficients; the rest follow by symmetry.

    ``t11`` repeats ``t00`` and ``t10`` is the conjugate of ``t01``, so the
    caller derives those. Feeding ``(cos, sin)`` gives the rotation itself and
    ``(sin, cos)`` rearranged gives its derivative in the flip angle, which is
    why this is one routine rather than two.
    """
    t00 = (a_value, 0.0 * a_value, a_tangent, 0.0 * a_tangent)
    t01 = _dual_scale(b_value, b_tangent, p2r, p2i, p2tr, p2ti)
    t02 = _dual_mul(
        0.0 * c_value, -c_value, 0.0 * c_tangent, -c_tangent, p1r, p1i, p1tr, p1ti
    )
    t12 = _dual_mul(
        0.0 * c_value, c_value, 0.0 * c_tangent, c_tangent, pcr, pci, pctr, pcti
    )
    t20 = _dual_mul(
        0.0 * c_value,
        -0.5 * c_value,
        0.0 * c_tangent,
        -0.5 * c_tangent,
        pcr,
        pci,
        pctr,
        pcti,
    )
    t21 = _dual_mul(
        0.0 * c_value,
        0.5 * c_value,
        0.0 * c_tangent,
        0.5 * c_tangent,
        p1r,
        p1i,
        p1tr,
        p1ti,
    )
    t22 = (d_value, 0.0 * d_value, d_tangent, 0.0 * d_tangent)
    return t00, t01, t02, t12, t20, t21, t22


@triton.jit
def _rotation_coefficients(a, b, c, d, p1r, p1i, p2r, p2i, pcr, pci):
    """Seven of the nine rotation coefficients; the rest follow by symmetry.

    ``t11`` repeats ``t00`` and ``t10`` is the conjugate of ``t01``, so the
    caller derives those. Feeding ``(cos, sin)`` gives the rotation itself and
    ``(sin, cos)`` rearranged gives its derivative in the flip angle, which is
    why this is one routine rather than two.
    """
    t00 = (a, 0.0 * a)
    t01 = (b * p2r, b * p2i)
    t02 = _complex_mul(0.0 * c, -c, p1r, p1i)
    t12 = _complex_mul(0.0 * c, c, pcr, pci)
    t20 = _complex_mul(0.0 * c, -0.5 * c, pcr, pci)
    t21 = _complex_mul(0.0 * c, 0.5 * c, p1r, p1i)
    t22 = (d, 0.0 * d)
    return t00, t01, t02, t12, t20, t21, t22


@triton.jit(do_not_specialize=["profile_bins", "lineshape_bins"])
def _epg_vjp_kernel(
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
    exchange_rate,
    t1_bound,
    pool_b_fraction,
    pool_b_exchange,
    t1_pool_b,
    t2_pool_b,
    pool_b_shift,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    lineshape,
    profile,
    profile_index,
    pairs,
    pair_index,
    duration_row,
    pool_table,
    pool_bars,
    pool_durations,
    row_count,
    grad_pair,
    grad_output_real,
    grad_output_imag,
    grad_tissue,
    grad_flip,
    grad_phase,
    grad_duration,
    trajectory_r,
    trajectory_i,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    shim_rows,
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shimmed: tl.constexpr,
    locations: tl.constexpr,
    profiled: tl.constexpr,
    profile_bins,
    dynamic: tl.constexpr,
    broadened: tl.constexpr,
    lineshape_bins,
    pools: tl.constexpr,
    narrow: tl.constexpr,
    tabulated: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    local = problem - problem_base
    record_stride = (
        7 if pools == 3 else (6 if pools == 2 else (4 if pools == 1 else 3))
    ) * state_count
    trajectory = local * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count
    bound_plane = 3 * state_count
    bplus_plane = 4 * state_count
    bminus_plane = 5 * state_count
    semisolid_plane = 6 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    pvr = empty
    pvi = empty
    mvr = empty
    mvi = empty
    zvr = empty + tl.where(state == 0, 1.0, 0.0)
    zvi = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_b1_phase = 0.0
    atom_b0 = 0.0
    if off_axis:
        atom_b1_phase = tl.load(b1_phase + scalar_atom, mask=active_atom, other=0.0)
        atom_b0 = tl.load(b0 + scalar_atom, mask=active_atom, other=0.0)
    atom_inv = 1.0
    if inverting:
        atom_inv = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    atom_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
    atom_flow = 0.0
    direction = 0.0
    atom_washout = 0.0
    if moving:
        atom_velocity = tl.load(velocity + scalar_atom, mask=active_atom, other=0.0)
        atom_flow = atom_velocity * flow_scale
        # |v| has no derivative at the origin, so a still voxel contributes
        # none.
        direction = (atom_velocity > 0.0).to(tl.float32) - (atom_velocity < 0.0).to(
            tl.float32
        )
        atom_washout = tl.abs(atom_velocity) * washout_scale
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    r1_value = 1000.0 / atom_t1
    r2_value = 1000.0 / atom_t2

    location = atom % locations
    # A semisolid pool rides along as a plane of its own: the pulse deposits
    # into it and it exchanges with the free water, so the reverse sweep cannot
    # replay it from the free pool's.
    atom_bound = 0.0
    atom_exchange = 0.0
    atom_t1b = 1.0
    atom_t2b = 1.0
    atom_shift = 0.0
    r1b_value = 0.0
    r2b_value = 0.0
    atom_semisolid = 0.0
    atom_semisolid_exchange = 0.0
    atom_t1c = 1.0
    r1c_value = 0.0
    atom_free = 1.0
    poolvr = empty
    poolvi = empty
    bpvr = empty
    bpvi = empty
    bmvr = empty
    bmvi = empty
    semivr = empty
    semivi = empty
    if pools == 1:
        atom_bound = tl.load(bound_fraction + scalar_atom, mask=active_atom, other=0.0)
        atom_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        atom_t1b = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        r1b_value = 1000.0 / atom_t1b
    if pools == 2 or pools == 3:
        atom_bound = tl.load(pool_b_fraction + scalar_atom, mask=active_atom, other=0.0)
        atom_exchange = tl.load(
            pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        atom_t1b = tl.load(t1_pool_b + scalar_atom, mask=active_atom, other=1.0)
        r1b_value = 1000.0 / atom_t1b
        atom_t2b = tl.load(t2_pool_b + scalar_atom, mask=active_atom, other=1.0)
        r2b_value = 1000.0 / atom_t2b
        atom_shift = tl.load(pool_b_shift + scalar_atom, mask=active_atom, other=0.0)
    if pools == 3:
        # The semisolid pool takes the rows a run with it alone would take, so
        # the two second pools never contend for one.
        atom_semisolid = tl.load(
            bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_semisolid_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        atom_t1c = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        r1c_value = 1000.0 / atom_t1c
        semivr = empty + tl.where(state == 0, atom_semisolid + 0.0, 0.0)
    if pools > 0:
        # The fractions split the equilibrium at t = 0.
        atom_free = 1.0 - atom_bound - atom_semisolid
        zvr = empty + tl.where(state == 0, atom_free, 0.0)
        poolvr = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_r + slot, pvr, mask=state_mask)
        tl.store(trajectory_i + slot, pvi, mask=state_mask)
        tl.store(trajectory_r + slot + minus_plane, mvr, mask=state_mask)
        tl.store(trajectory_i + slot + minus_plane, mvi, mask=state_mask)
        tl.store(trajectory_r + slot + long_plane, zvr, mask=state_mask)
        tl.store(trajectory_i + slot + long_plane, zvi, mask=state_mask)
        if pools > 0:
            tl.store(trajectory_r + slot + bound_plane, poolvr, mask=state_mask)
            tl.store(trajectory_i + slot + bound_plane, poolvi, mask=state_mask)
        if pools == 2 or pools == 3:
            tl.store(trajectory_r + slot + bplus_plane, bpvr, mask=state_mask)
            tl.store(trajectory_i + slot + bplus_plane, bpvi, mask=state_mask)
            tl.store(trajectory_r + slot + bminus_plane, bmvr, mask=state_mask)
            tl.store(trajectory_i + slot + bminus_plane, bmvi, mask=state_mask)
        if pools == 3:
            tl.store(trajectory_r + slot + semisolid_plane, semivr, mask=state_mask)
            tl.store(trajectory_i + slot + semisolid_plane, semivi, mask=state_mask)

        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        wout_value = 1.0
        if moving:
            wout_value = _washout(atom_washout, dt_value)
        e1_value = tl.exp(-r1_value * dt_value) * wout_value
        e2_value = tl.exp(-r2_value * dt_value) * wout_value
        damp_z = 1.0
        damp_t = 1.0
        if diffusing:
            damp_z, damp_t = _damping(atom_damping, dt_value, order)
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = 1.0 - e1_value
        bare1_value = e1_value
        bare2_value = e2_value
        e1_value = bare1_value * damp_z
        e2_value = bare2_value * damp_t
        turn_t = 0.0
        szr, szi = 1.0, 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, dt_value, order)
            szr, szi = tl.cos(turn_z), tl.sin(turn_z)
        qr, qi = 1.0, 0.0
        if off_axis or moving:
            angle_value = -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
            qr, qi = tl.cos(angle_value), tl.sin(angle_value)
        ovr, ovi = e2_value * qr, e2_value * qi
        lvr, lvi = e1_value * szr, e1_value * szi

        if pools == 2 or pools == 3:
            # With an exchanging pool the transverse relaxation sits inside the
            # operator instead of in the scalar the free pool alone multiplies.
            across = _two_pool_transverse_step_jvp(
                r2_value,
                0.0,
                r2b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                atom_free,
                0.0,
                atom_shift,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
            )
            a11r, a11i = across[0], across[1]
            a12r, a12i = across[2], across[3]
            a21r, a21i = across[4], across[5]
            a22r, a22i = across[6], across[7]
            carr, cari = damp_t * qr, damp_t * qi
            f11r, f11i = _complex_mul(a11r, a11i, pvr, pvi)
            f12r, f12i = _complex_mul(a12r, a12i, bpvr, bpvi)
            g21r, g21i = _complex_mul(a21r, a21i, pvr, pvi)
            g22r, g22i = _complex_mul(a22r, a22i, bpvr, bpvi)
            # ``F-`` takes the conjugate of the operator entry by entry, not its
            # transpose: it is the conjugate state following the conjugate map.
            h11r, h11i = _complex_mul(a11r, -a11i, mvr, mvi)
            h12r, h12i = _complex_mul(a12r, -a12i, bmvr, bmvi)
            k21r, k21i = _complex_mul(a21r, -a21i, mvr, mvi)
            k22r, k22i = _complex_mul(a22r, -a22i, bmvr, bmvi)
            pvr, pvi = _complex_mul(f11r + f12r, f11i + f12i, carr, cari)
            bpvr, bpvi = _complex_mul(g21r + g22r, g21i + g22i, carr, cari)
            mvr, mvi = _complex_mul(h11r + h12r, h11i + h12i, carr, -cari)
            bmvr, bmvi = _complex_mul(k21r + k22r, k21i + k22i, carr, -cari)
        else:
            pvr, pvi = _complex_mul(ovr, ovi, pvr, pvi)
            mvr, mvi = _complex_mul(ovr, -ovi, mvr, mvi)
        if pools == 3:
            # Three pools mix through a 3x3 formed once for the interval; each
            # second pool exchanges with the free water and not with the other.
            nil = 0.0 * dt_value
            hold_value = wout_value + nil
            if tabulated:
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                ) = _three_pool_from_table(
                    pool_table,
                    tl.load(
                        duration_row + event_base + event,
                        mask=active_atom,
                        other=0,
                    ),
                    atom,
                    atom_count,
                    active_atom,
                    hold_value,
                    atom_free,
                    atom_bound,
                    atom_semisolid,
                )
            else:
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    _dw11,
                    _dw12,
                    _dw13,
                    _dw21,
                    _dw22,
                    _dw23,
                    _dw31,
                    _dw32,
                    _dw33,
                    _dgf,
                    _dgb,
                    _dgs,
                ) = _three_pool_step_jvp(
                    r1_value,
                    nil,
                    r1b_value,
                    nil,
                    r1c_value,
                    nil,
                    atom_exchange,
                    nil,
                    atom_semisolid_exchange,
                    nil,
                    atom_bound,
                    nil,
                    atom_semisolid,
                    nil,
                    dt_value,
                    nil,
                    hold_value,
                    nil,
                    narrow,
                )
            spin_r, spin_i = damp_z * szr, damp_z * szi
            mix_fr = w11 * zvr + w12 * poolvr + w13 * semivr
            mix_fi = w11 * zvi + w12 * poolvi + w13 * semivi
            mix_br = w21 * zvr + w22 * poolvr + w23 * semivr
            mix_bi = w21 * zvi + w22 * poolvi + w23 * semivi
            mix_cr = w31 * zvr + w32 * poolvr + w33 * semivr
            mix_ci = w31 * zvi + w32 * poolvi + w33 * semivi
            zvr, zvi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            poolvr, poolvi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            semivr, semivi = _complex_mul(spin_r, spin_i, mix_cr, mix_ci)
            zvr += tl.where(state == 0, grow_free, 0.0)
            poolvr += tl.where(state == 0, grow_pool_b, 0.0)
            semivr += tl.where(state == 0, grow_semisolid, 0.0)
        elif pools > 0:
            # The pools exchange while they relax, so the longitudinal step is a
            # 2x2 the interval forms once and the per-order damping and turn
            # multiply. Read from the dual helper with no direction to follow:
            # what only its tangents reach, the compiler drops.
            (
                pe11,
                pe12,
                pe21,
                pe22,
                prec_f,
                prec_b,
                _d11,
                _d12,
                _d21,
                _d22,
                _drf,
                _drb,
            ) = _two_pool_step_jvp(
                r1_value,
                0.0,
                r1b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
            )
            spin_r, spin_i = damp_z * szr, damp_z * szi
            mix_fr = pe11 * zvr + pe12 * poolvr
            mix_fi = pe11 * zvi + pe12 * poolvi
            mix_br = pe21 * zvr + pe22 * poolvr
            mix_bi = pe21 * zvi + pe22 * poolvi
            zvr, zvi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            poolvr, poolvi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            zvr += tl.where(state == 0, prec_f, 0.0)
            poolvr += tl.where(state == 0, prec_b, 0.0)
        else:
            zvr, zvi = _complex_mul(lvr, lvi, zvr, zvi)
            zvr += tl.where(state == 0, recovery_value, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count)
        pvr = tl.where(pre_shift, svr, pvr)
        pvi = tl.where(pre_shift, svi, pvi)
        mvr = tl.where(pre_shift, wvr, mvr)
        mvi = tl.where(pre_shift, wvi, mvi)
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count
            )
            bpvr = tl.where(pre_shift, svr, bpvr)
            bpvi = tl.where(pre_shift, svi, bpvi)
            bmvr = tl.where(pre_shift, wvr, bmvr)
            bmvi = tl.where(pre_shift, wvi, bmvi)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        zvr = tl.where(invert, -atom_inv * zvr, zvr)
        zvi = tl.where(invert, -atom_inv * zvi, zvi)
        if pools == 2 or pools == 3:
            # A chemically exchanging pool is free water and inverts like any
            # other; a semisolid one is saturated instead, by the pulse's own
            # saturation term.
            poolvr = tl.where(invert, -atom_inv * poolvr, poolvr)
            poolvi = tl.where(invert, -atom_inv * poolvi, poolvi)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_phase = _event_value(phase, event_base, event, active_atom, single_train)
        pulse_b1 = atom_b1
        pulse_b1_phase = atom_b1_phase
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            if off_axis:
                pulse_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )
        alpha_value = event_flip * pulse_b1
        phi_value = event_phase + pulse_b1_phase
        if pools == 1 or pools == 3:
            # The pool absorbs the power the pulse deposits, read at the offset
            # the pulse is played less the voxel's own.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, _shape_slope = _lineshape_at_slope(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            absorbed_value = tl.exp(power_value * shape_value)
            saturating = is_rf & ~is_inversion
            if pools == 1:
                poolvr = tl.where(saturating, absorbed_value * poolvr, poolvr)
                poolvi = tl.where(saturating, absorbed_value * poolvi, poolvi)
            else:
                semivr = tl.where(saturating, absorbed_value * semivr, semivr)
                semivi = tl.where(saturating, absorbed_value * semivi, semivi)
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        p1r, p1i = tl.cos(phi_value), tl.sin(phi_value)
        p2r, p2i = _complex_mul(p1r, p1i, p1r, p1i)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_coefficients(
            0.5 * (1.0 + cos_value),
            0.5 * (1.0 - cos_value),
            sin_value,
            cos_value,
            p1r,
            p1i,
            p2r,
            p2i,
            p1r,
            -p1i,
        )
        a0 = _complex_mul(t00[0], t00[1], pvr, pvi)
        a1 = _complex_mul(t01[0], t01[1], mvr, mvi)
        a2 = _complex_mul(t02[0], t02[1], zvr, zvi)
        b0_ = _complex_mul(t01[0], -t01[1], pvr, pvi)
        b1_ = _complex_mul(t00[0], t00[1], mvr, mvi)
        b2 = _complex_mul(t12[0], t12[1], zvr, zvi)
        c0 = _complex_mul(t20[0], t20[1], pvr, pvi)
        c1 = _complex_mul(t21[0], t21[1], mvr, mvi)
        c2 = _complex_mul(t22[0], t22[1], zvr, zvi)

        turned_pr = a0[0] + a1[0] + a2[0]
        turned_pi = a0[1] + a1[1] + a2[1]
        turned_mr = b0_[0] + b1_[0] + b2[0]
        turned_mi = b0_[1] + b1_[1] + b2[1]
        turned_zr = c0[0] + c1[0] + c2[0]
        turned_zi = c0[1] + c1[1] + c2[1]
        if profiled or dynamic:
            if dynamic:
                pair = _dynamic_pair_at(
                    pairs,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                )
                shaped_ar, shaped_ai = pair[0], pair[1]
                # The pair is integrated at zero RF phase, so the event's own
                # phase turns the axis afterwards.
                shaped_br, shaped_bi = _complex_mul(pair[2], pair[3], p1r, -p1i)
            else:
                shaped_ar, shaped_ai, shaped_br, shaped_bi = _profile_pair(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha_value,
                    profile_bins,
                    profile_step,
                )
                shaped_br, shaped_bi = _complex_mul(shaped_br, shaped_bi, p1r, -p1i)
            (
                turned_pr,
                turned_pi,
                turned_mr,
                turned_mi,
                turned_zr,
                turned_zi,
            ) = _rotate_spinor(
                shaped_ar,
                shaped_ai,
                shaped_br,
                shaped_bi,
                pvr,
                pvi,
                mvr,
                mvi,
                zvr,
                zvi,
            )

        rotate = is_rf & ~is_inversion
        if pools == 2 or pools == 3:
            # The same pulse, the same rotation. A chemical shift moves where a
            # pool precesses, not what a pulse does to it.
            e0 = _complex_mul(t00[0], t00[1], bpvr, bpvi)
            e1_ = _complex_mul(t01[0], t01[1], bmvr, bmvi)
            e2_ = _complex_mul(t02[0], t02[1], poolvr, poolvi)
            f0 = _complex_mul(t01[0], -t01[1], bpvr, bpvi)
            f1 = _complex_mul(t00[0], t00[1], bmvr, bmvi)
            f2 = _complex_mul(t12[0], t12[1], poolvr, poolvi)
            h0 = _complex_mul(t20[0], t20[1], bpvr, bpvi)
            h1 = _complex_mul(t21[0], t21[1], bmvr, bmvi)
            h2 = _complex_mul(t22[0], t22[1], poolvr, poolvi)
            spun_pr, spun_pi = e0[0] + e1_[0] + e2_[0], e0[1] + e1_[1] + e2_[1]
            spun_mr, spun_mi = f0[0] + f1[0] + f2[0], f0[1] + f1[1] + f2[1]
            spun_zr, spun_zi = h0[0] + h1[0] + h2[0], h0[1] + h1[1] + h2[1]
            if profiled or dynamic:
                (
                    spun_pr,
                    spun_pi,
                    spun_mr,
                    spun_mi,
                    spun_zr,
                    spun_zi,
                ) = _rotate_spinor(
                    shaped_ar,
                    shaped_ai,
                    shaped_br,
                    shaped_bi,
                    bpvr,
                    bpvi,
                    bmvr,
                    bmvi,
                    poolvr,
                    poolvi,
                )
            bpvr = tl.where(rotate, spun_pr, bpvr)
            bpvi = tl.where(rotate, spun_pi, bpvi)
            bmvr = tl.where(rotate, spun_mr, bmvr)
            bmvi = tl.where(rotate, spun_mi, bmvi)
            poolvr = tl.where(rotate, spun_zr, poolvr)
            poolvi = tl.where(rotate, spun_zi, poolvi)
        pvr = tl.where(rotate, turned_pr, pvr)
        pvi = tl.where(rotate, turned_pi, pvi)
        mvr = tl.where(rotate, turned_mr, mvr)
        mvi = tl.where(rotate, turned_mi, mvi)
        zvr = tl.where(rotate, turned_zr, zvr)
        zvi = tl.where(rotate, turned_zi, zvi)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count
            )
            spoil_b = (event_action & 8) != 0
            bpvr = tl.where(spoil_b, 0.0, tl.where(do_shift, svr, bpvr))
            bpvi = tl.where(spoil_b, 0.0, tl.where(do_shift, svi, bpvi))
            bmvr = tl.where(spoil_b, 0.0, tl.where(do_shift, wvr, bmvr))
            bmvi = tl.where(spoil_b, 0.0, tl.where(do_shift, wvi, bmvi))
        svr, svi, wvr, wvi = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count)
        pvr = tl.where(do_shift, svr, pvr)
        pvi = tl.where(do_shift, svi, pvi)
        mvr = tl.where(do_shift, wvr, mvr)
        mvi = tl.where(do_shift, wvi, mvi)
        spoil = (event_action & 8) != 0
        pvr = tl.where(spoil, 0.0, pvr)
        pvi = tl.where(spoil, 0.0, pvi)
        mvr = tl.where(spoil, 0.0, mvr)
        mvi = tl.where(spoil, 0.0, mvi)

    # ---- reverse ----
    pbvr = empty
    pbvi = empty
    mbvr = empty
    mbvi = empty
    zbvr = empty
    zbvi = empty
    zero = tl.zeros((problems, 1), tl.float32)
    g_diffv = zero
    g_flowv = zero
    g_washv = zero
    g_t1v = zero
    g_t2v = zero
    g_m0v = zero
    g_b1v = zero
    g_b1pv = zero
    g_b0v = zero
    g_invv = zero
    g_boundv = zero
    g_exchv = zero
    g_t1bv = zero
    g_t2bv = zero
    g_shiftv = zero
    g_semiv = zero
    g_sexchv = zero
    g_t1cv = zero
    poolbr = empty
    poolbi = empty
    semibr = empty
    semibi = empty
    ubvr = empty
    ubvi = empty
    wbvr = empty
    wbvi = empty

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        xpvr = tl.load(trajectory_r + slot, mask=state_mask, other=0.0)
        xpvi = tl.load(trajectory_i + slot, mask=state_mask, other=0.0)
        xmvr = tl.load(trajectory_r + slot + minus_plane, mask=state_mask, other=0.0)
        xmvi = tl.load(trajectory_i + slot + minus_plane, mask=state_mask, other=0.0)
        xzvr = tl.load(trajectory_r + slot + long_plane, mask=state_mask, other=0.0)
        xzvi = tl.load(trajectory_i + slot + long_plane, mask=state_mask, other=0.0)
        xbvr = empty
        xbvi = empty
        xcvr = empty
        xcvi = empty
        xbpvr = empty
        xbpvi = empty
        xbmvr = empty
        xbmvi = empty
        rbpvr = empty
        rbpvi = empty
        rbmvr = empty
        rbmvi = empty
        if pools > 0:
            xbvr = tl.load(
                trajectory_r + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbvi = tl.load(
                trajectory_i + slot + bound_plane, mask=state_mask, other=0.0
            )
        if pools == 2 or pools == 3:
            xbpvr = tl.load(
                trajectory_r + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbpvi = tl.load(
                trajectory_i + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbmvr = tl.load(
                trajectory_r + slot + bminus_plane, mask=state_mask, other=0.0
            )
            xbmvi = tl.load(
                trajectory_i + slot + bminus_plane, mask=state_mask, other=0.0
            )
        if pools == 3:
            xcvr = tl.load(
                trajectory_r + slot + semisolid_plane, mask=state_mask, other=0.0
            )
            xcvi = tl.load(
                trajectory_i + slot + semisolid_plane, mask=state_mask, other=0.0
            )

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        wout_value = 1.0
        if moving:
            wout_value = _washout(atom_washout, dt_value)
        dry1_value = tl.exp(-r1_value * dt_value)
        dry2_value = tl.exp(-r2_value * dt_value)
        e1_value = dry1_value * wout_value
        e2_value = dry2_value * wout_value
        damp_z = 1.0
        damp_t = 1.0
        if diffusing:
            damp_z, damp_t = _damping(atom_damping, dt_value, order)
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = 1.0 - e1_value
        bare1_value = e1_value
        bare2_value = e2_value
        e1_value = bare1_value * damp_z
        e2_value = bare2_value * damp_t
        turn_t = 0.0
        szr, szi = 1.0, 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, dt_value, order)
            szr, szi = tl.cos(turn_z), tl.sin(turn_z)
        qr, qi = 1.0, 0.0
        if off_axis or moving:
            angle_value = -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
            qr, qi = tl.cos(angle_value), tl.sin(angle_value)
        ovr, ovi = e2_value * qr, e2_value * qi
        lvr, lvi = e1_value * szr, e1_value * szi

        # Replay the intra-event stages from the recorded entry state.
        if pools == 2 or pools == 3:
            # With an exchanging pool the transverse relaxation sits inside the
            # operator instead of in the scalar the free pool alone multiplies.
            across = _two_pool_transverse_step_jvp(
                r2_value,
                0.0,
                r2b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                atom_free,
                0.0,
                atom_shift,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
            )
            a11r, a11i = across[0], across[1]
            a12r, a12i = across[2], across[3]
            a21r, a21i = across[4], across[5]
            a22r, a22i = across[6], across[7]
            carr, cari = damp_t * qr, damp_t * qi
            f11r, f11i = _complex_mul(a11r, a11i, xpvr, xpvi)
            f12r, f12i = _complex_mul(a12r, a12i, xbpvr, xbpvi)
            g21r, g21i = _complex_mul(a21r, a21i, xpvr, xpvi)
            g22r, g22i = _complex_mul(a22r, a22i, xbpvr, xbpvi)
            # ``F-`` takes the conjugate of the operator entry by entry, not its
            # transpose: it is the conjugate state following the conjugate map.
            h11r, h11i = _complex_mul(a11r, -a11i, xmvr, xmvi)
            h12r, h12i = _complex_mul(a12r, -a12i, xbmvr, xbmvi)
            k21r, k21i = _complex_mul(a21r, -a21i, xmvr, xmvi)
            k22r, k22i = _complex_mul(a22r, -a22i, xbmvr, xbmvi)
            rpvr, rpvi = _complex_mul(f11r + f12r, f11i + f12i, carr, cari)
            rbpvr, rbpvi = _complex_mul(g21r + g22r, g21i + g22i, carr, cari)
            rmvr, rmvi = _complex_mul(h11r + h12r, h11i + h12i, carr, -cari)
            rbmvr, rbmvi = _complex_mul(k21r + k22r, k21i + k22i, carr, -cari)
        else:
            rpvr, rpvi = _complex_mul(ovr, ovi, xpvr, xpvi)
            rmvr, rmvi = _complex_mul(ovr, -ovi, xmvr, xmvi)

        rbvr = empty
        rbvi = empty
        rcvr = empty
        rcvi = empty
        if pools == 3:
            nil = 0.0 * dt_value
            hold_value = wout_value + nil
            if tabulated:
                # The walk back needs the operator itself, which the row
                # already holds -- and pooling the cotangents took what
                # the eigenvalues were formed for, so nothing here reads
                # them.
                pool_row = tl.load(
                    duration_row + event_base + event,
                    mask=active_atom,
                    other=0,
                )
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                ) = _three_pool_from_table(
                    pool_table,
                    pool_row,
                    atom,
                    atom_count,
                    active_atom,
                    hold_value,
                    atom_free,
                    atom_bound,
                    atom_semisolid,
                )
            else:
                # The pieces and the bare operator are kept rather than
                # the step alone: the walk back pushes the cotangents
                # through them, and forming them once for the interval
                # is what keeps this kernel a size a compiler will take.
                (
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                ) = _three_pool_pieces_jvp(
                    r1_value,
                    nil,
                    r1b_value,
                    nil,
                    r1c_value,
                    nil,
                    atom_exchange,
                    nil,
                    atom_semisolid_exchange,
                    nil,
                    atom_bound,
                    nil,
                    atom_semisolid,
                    nil,
                    dt_value,
                    nil,
                    narrow,
                )
                (
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                ) = _three_pool_assemble_jvp(
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                    narrow,
                )
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    _dw11,
                    _dw12,
                    _dw13,
                    _dw21,
                    _dw22,
                    _dw23,
                    _dw31,
                    _dw32,
                    _dw33,
                    _dgf,
                    _dgb,
                    _dgs,
                ) = _three_pool_weigh_jvp(
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    hold_value,
                    nil,
                    narrow,
                )
                # The operator is O(1) once formed, so the per-order loop below
                # takes it at the width the states are carried in.
                w11 = w11.to(tl.float32)
                w12 = w12.to(tl.float32)
                w13 = w13.to(tl.float32)
                w21 = w21.to(tl.float32)
                w22 = w22.to(tl.float32)
                w23 = w23.to(tl.float32)
                w31 = w31.to(tl.float32)
                w32 = w32.to(tl.float32)
                w33 = w33.to(tl.float32)
                grow_free = grow_free.to(tl.float32)
                grow_pool_b = grow_pool_b.to(tl.float32)
                grow_semisolid = grow_semisolid.to(tl.float32)
            spin_r, spin_i = damp_z * szr, damp_z * szi
            mix_fr = w11 * xzvr + w12 * xbvr + w13 * xcvr
            mix_fi = w11 * xzvi + w12 * xbvi + w13 * xcvi
            mix_br = w21 * xzvr + w22 * xbvr + w23 * xcvr
            mix_bi = w21 * xzvi + w22 * xbvi + w23 * xcvi
            mix_cr = w31 * xzvr + w32 * xbvr + w33 * xcvr
            mix_ci = w31 * xzvi + w32 * xbvi + w33 * xcvi
            rzvr, rzvi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            rbvr, rbvi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            rcvr, rcvi = _complex_mul(spin_r, spin_i, mix_cr, mix_ci)
            rzvr += tl.where(state == 0, grow_free, 0.0)
            rbvr += tl.where(state == 0, grow_pool_b, 0.0)
            rcvr += tl.where(state == 0, grow_semisolid, 0.0)
        elif pools > 0:
            (
                pe11,
                pe12,
                pe21,
                pe22,
                prec_f,
                prec_b,
                _d11,
                _d12,
                _d21,
                _d22,
                _drf,
                _drb,
            ) = _two_pool_step_jvp(
                r1_value,
                0.0,
                r1b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
            )
            spin_r, spin_i = damp_z * szr, damp_z * szi
            mix_fr = pe11 * xzvr + pe12 * xbvr
            mix_fi = pe11 * xzvi + pe12 * xbvi
            mix_br = pe21 * xzvr + pe22 * xbvr
            mix_bi = pe21 * xzvi + pe22 * xbvi
            rzvr, rzvi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            rbvr, rbvi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            rzvr += tl.where(state == 0, prec_f, 0.0)
            rbvr += tl.where(state == 0, prec_b, 0.0)
        else:
            rzvr, rzvi = _complex_mul(lvr, lvi, xzvr, xzvi)
            rzvr += tl.where(state == 0, recovery_value, 0.0)

        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(
            rpvr, rpvi, rmvr, rmvi, state, state_mask, state_count
        )
        spvr = tl.where(pre_shift, svr, rpvr)
        spvi = tl.where(pre_shift, svi, rpvi)
        smvr = tl.where(pre_shift, wvr, rmvr)
        smvi = tl.where(pre_shift, wvi, rmvi)
        sbpvr = empty
        sbpvi = empty
        sbmvr = empty
        sbmvi = empty
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                rbpvr, rbpvi, rbmvr, rbmvi, state, state_mask, state_count
            )
            sbpvr = tl.where(pre_shift, svr, rbpvr)
            sbpvi = tl.where(pre_shift, svi, rbpvi)
            sbmvr = tl.where(pre_shift, wvr, rbmvr)
            sbmvi = tl.where(pre_shift, wvi, rbmvi)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count
        )
        trailing = do_shift & ~spoil
        pbvr = tl.where(spoil, 0.0, tl.where(trailing, avr, pbvr))
        pbvi = tl.where(spoil, 0.0, tl.where(trailing, avi, pbvi))
        mbvr = tl.where(spoil, 0.0, tl.where(trailing, bvr, mbvr))
        mbvi = tl.where(spoil, 0.0, tl.where(trailing, bvi, mbvi))
        if pools == 2 or pools == 3:
            avr, avi, bvr, bvi = _shift_adjoint(
                ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count
            )
            ubvr = tl.where(spoil, 0.0, tl.where(trailing, avr, ubvr))
            ubvi = tl.where(spoil, 0.0, tl.where(trailing, avi, ubvi))
            wbvr = tl.where(spoil, 0.0, tl.where(trailing, bvr, wbvr))
            wbvi = tl.where(spoil, 0.0, tl.where(trailing, bvi, wbvi))

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_phase = _event_value(phase, event_base, event, active_atom, single_train)
        pulse_b1 = atom_b1
        pulse_b1_phase = atom_b1_phase
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            if off_axis:
                pulse_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )

        # ---- recorded sample ----
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed_mask = active_atom & record & (out >= 0)
        seed_real = tl.load(
            grad_output_real + problem * output_count + out, mask=seed_mask, other=0.0
        )
        seed_imag = tl.load(
            grad_output_imag + problem * output_count + out, mask=seed_mask, other=0.0
        )
        dvr, dvi = tl.cos(-event_phase), tl.sin(-event_phase)
        # grad_m0 = Re(conj(seed) * recorded * demodulation)
        recr, reci = spvr, spvi
        if pools == 2 or pools == 3:
            recr, reci = spvr + sbpvr, spvi + sbpvi
        wr, wi = _complex_mul(recr, reci, dvr, dvi)
        g_m0v += tl.sum(
            tl.where(state == 0, seed_real * wr + seed_imag * wi, 0.0), axis=1
        )[:, None]
        # grad_phase = Re(conj(seed) * m0 * recorded * (-i) * demodulation)
        yr, yi = atom_m0 * recr, atom_m0 * reci
        yr, yi = yi, -yr
        yr, yi = _complex_mul(yr, yi, dvr, dvi)
        tl.atomic_add(
            grad_phase + event_base + event,
            tl.sum(tl.where(state == 0, seed_real * yr + seed_imag * yi, 0.0), axis=1)[
                :, None
            ],
            mask=seed_mask,
        )
        # fplus_bar[0] += conj(m0 * demodulation) * seed
        kr, ki = atom_m0 * dvr, atom_m0 * dvi
        sr, si = _complex_mul(kr, -ki, seed_real, seed_imag)
        pbvr += tl.where(state == 0, sr, 0.0)
        pbvi += tl.where(state == 0, si, 0.0)
        if pools == 2 or pools == 3:
            ubvr += tl.where(state == 0, sr, 0.0)
            ubvi += tl.where(state == 0, si, 0.0)

        # ---- RF adjoint ----
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        g_invv += tl.sum(tl.where(invert, zbvr * -rzvr + zbvi * -rzvi, 0.0), axis=1)[
            :, None
        ]
        zbvr = tl.where(invert, -atom_inv * zbvr, zbvr)
        zbvi = tl.where(invert, -atom_inv * zbvi, zbvi)

        alpha_value = event_flip * pulse_b1
        phi_value = event_phase + pulse_b1_phase
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        p1r, p1i = tl.cos(phi_value), tl.sin(phi_value)
        p2r, p2i = _complex_mul(p1r, p1i, p1r, p1i)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_coefficients(
            0.5 * (1.0 + cos_value),
            0.5 * (1.0 - cos_value),
            sin_value,
            cos_value,
            p1r,
            p1i,
            p2r,
            p2i,
            p1r,
            -p1i,
        )
        d00, d01, d02, d12, d20, d21, d22 = _rotation_coefficients(
            -0.5 * sin_value,
            0.5 * sin_value,
            cos_value,
            -sin_value,
            p1r,
            p1i,
            p2r,
            p2i,
            p1r,
            -p1i,
        )

        sat_alpha_v = zero
        sat_b0_v = zero
        if pools == 1 or pools == 3:
            # The pulse scales every order of the pool by one real number, so
            # its cotangent is a single sum over the states it multiplied.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, shape_slope = _lineshape_at_slope(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            absorbed_value = tl.exp(power_value * shape_value)
            if pools == 1:
                per_state = poolbr * rbvr + poolbi * rbvi
            else:
                per_state = semibr * rcvr + semibi * rcvi
            grad_absorbed = tl.sum(per_state, axis=1)[:, None]
            grad_exponent = grad_absorbed * absorbed_value
            twice = event_saturation * 2.0
            sat_alpha_v = grad_exponent * (twice * alpha_value * shape_value)
            # The lineshape is read at the pulse's offset from the voxel, so a
            # step in the voxel's own off-resonance moves the read the other way.
            sat_b0_v = -grad_exponent * (power_value * shape_slope)
            saturating = is_rf & ~is_inversion
            if pools == 1:
                poolbr = tl.where(saturating, absorbed_value * poolbr, poolbr)
                poolbi = tl.where(saturating, absorbed_value * poolbi, poolbi)
            else:
                semibr = tl.where(saturating, absorbed_value * semibr, semibr)
                semibi = tl.where(saturating, absorbed_value * semibi, semibi)

        # d/dalpha, contracted with the adjoint.
        row0 = _complex_mul(d00[0], d00[1], spvr, spvi)
        add1 = _complex_mul(d01[0], d01[1], smvr, smvi)
        add2 = _complex_mul(d02[0], d02[1], rzvr, rzvi)
        alpha_v = pbvr * (row0[0] + add1[0] + add2[0])
        alpha_v += pbvi * (row0[1] + add1[1] + add2[1])
        row0 = _complex_mul(d01[0], -d01[1], spvr, spvi)
        add1 = _complex_mul(d00[0], d00[1], smvr, smvi)
        add2 = _complex_mul(d12[0], d12[1], rzvr, rzvi)
        alpha_v += mbvr * (row0[0] + add1[0] + add2[0])
        alpha_v += mbvi * (row0[1] + add1[1] + add2[1])
        row0 = _complex_mul(d20[0], d20[1], spvr, spvi)
        add1 = _complex_mul(d21[0], d21[1], smvr, smvi)
        add2 = _complex_mul(d22[0], d22[1], rzvr, rzvi)
        alpha_v += zbvr * (row0[0] + add1[0] + add2[0])
        alpha_v += zbvi * (row0[1] + add1[1] + add2[1])

        # d/dphi, where only the phase factors carry the dependence.
        u1 = _complex_mul(t01[0], t01[1], smvr, smvi)
        u2 = _complex_mul(t02[0], t02[1], rzvr, rzvi)
        ur, ui = -(2.0 * u1[1] + u2[1]), 2.0 * u1[0] + u2[0]
        phi_v = pbvr * ur + pbvi * ui
        u1 = _complex_mul(t01[0], -t01[1], spvr, spvi)
        u2 = _complex_mul(t12[0], t12[1], rzvr, rzvi)
        ur, ui = 2.0 * u1[1] + u2[1], -2.0 * u1[0] - u2[0]
        phi_v += mbvr * ur + mbvi * ui
        u1 = _complex_mul(t20[0], t20[1], spvr, spvi)
        u2 = _complex_mul(t21[0], t21[1], smvr, smvi)
        ur, ui = -(u2[1] - u1[1]), u2[0] - u1[0]
        phi_v += zbvr * ur + zbvi * ui

        if profiled or dynamic:
            slope_ar, slope_ai, slope_br, slope_bi = 0.0, 0.0, 0.0, 0.0
            if dynamic:
                pair = _dynamic_pair_at(
                    pairs,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                )
                shaped_ar, shaped_ai = pair[0], pair[1]
                shaped_br, shaped_bi = _complex_mul(pair[2], pair[3], p1r, -p1i)
            else:
                (
                    shaped_ar,
                    slope_ar,
                    shaped_ai,
                    slope_ai,
                    shaped_br,
                    slope_br,
                    shaped_bi,
                    slope_bi,
                ) = _profile_pair_slope(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha_value,
                    profile_bins,
                    profile_step,
                )
                shaped_br, shaped_bi = _complex_mul(shaped_br, shaped_bi, p1r, -p1i)
                slope_br, slope_bi = _complex_mul(slope_br, slope_bi, p1r, -p1i)
            (
                grad_ar,
                grad_ai,
                grad_br,
                grad_bi,
                shaped_pbr,
                shaped_pbi,
                shaped_mbr,
                shaped_mbi,
                shaped_zbr,
                shaped_zbi,
            ) = _spinor_adjoint(
                shaped_ar,
                shaped_ai,
                shaped_br,
                shaped_bi,
                spvr,
                spvi,
                smvr,
                smvi,
                rzvr,
                rzvi,
                pbvr,
                pbvi,
                mbvr,
                mbvi,
                zbvr,
                zbvi,
            )
            if dynamic:
                # The flip is inside the pair rather than read against it, so
                # it has no gradient here: the cotangent goes out on the
                # rotation and whatever integrated it carries the rest. ``b``
                # was turned by the phase after the pair came out, so the
                # cotangent turns back the other way.
                alpha_v = alpha_v * 0.0
                back_r, back_i = _complex_mul(grad_br, grad_bi, p1r, p1i)
                _store_pair_gradient(
                    grad_pair,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    is_rf & ~is_inversion,
                    active_atom,
                    state_mask,
                    grad_ar,
                    grad_ai,
                    back_r,
                    back_i,
                )
            else:
                alpha_v = grad_ar * slope_ar + grad_ai * slope_ai
                alpha_v += grad_br * slope_br + grad_bi * slope_bi
            # d(b e^{-i phi})/dphi is -i times it, and nothing else moves.
            phi_v = grad_br * shaped_bi - grad_bi * shaped_br
            if pools == 2 or pools == 3:
                (
                    pool_ar,
                    pool_ai,
                    pool_pair_br,
                    pool_pair_bi,
                    pool_shaped_pbr,
                    pool_shaped_pbi,
                    pool_shaped_mbr,
                    pool_shaped_mbi,
                    pool_shaped_zbr,
                    pool_shaped_zbi,
                ) = _spinor_adjoint(
                    shaped_ar,
                    shaped_ai,
                    shaped_br,
                    shaped_bi,
                    sbpvr,
                    sbpvi,
                    sbmvr,
                    sbmvi,
                    rbvr,
                    rbvi,
                    ubvr,
                    ubvi,
                    wbvr,
                    wbvi,
                    poolbr,
                    poolbi,
                )
                if dynamic:
                    # The same pulse turned this pool, so its cotangent lands
                    # on the same row.
                    back_r, back_i = _complex_mul(pool_pair_br, pool_pair_bi, p1r, p1i)
                    _store_pair_gradient(
                        grad_pair,
                        pair_index,
                        event_base,
                        event,
                        atom,
                        atom_count,
                        is_rf & ~is_inversion,
                        active_atom,
                        state_mask,
                        pool_ar,
                        pool_ai,
                        back_r,
                        back_i,
                    )
                else:
                    alpha_v += pool_ar * slope_ar + pool_ai * slope_ai
                    alpha_v += pool_pair_br * slope_br + pool_pair_bi * slope_bi
                phi_v += pool_pair_br * shaped_bi - pool_pair_bi * shaped_br

        if (pools == 2 or pools == 3) and not profiled and not dynamic:
            # The same pulse turns the exchanging pool, so its cotangent adds to
            # the flip and phase the free pool already left.
            row0 = _complex_mul(d00[0], d00[1], sbpvr, sbpvi)
            add1 = _complex_mul(d01[0], d01[1], sbmvr, sbmvi)
            add2 = _complex_mul(d02[0], d02[1], rbvr, rbvi)
            alpha_v += ubvr * (row0[0] + add1[0] + add2[0])
            alpha_v += ubvi * (row0[1] + add1[1] + add2[1])
            row0 = _complex_mul(d01[0], -d01[1], sbpvr, sbpvi)
            add1 = _complex_mul(d00[0], d00[1], sbmvr, sbmvi)
            add2 = _complex_mul(d12[0], d12[1], rbvr, rbvi)
            alpha_v += wbvr * (row0[0] + add1[0] + add2[0])
            alpha_v += wbvi * (row0[1] + add1[1] + add2[1])
            row0 = _complex_mul(d20[0], d20[1], sbpvr, sbpvi)
            add1 = _complex_mul(d21[0], d21[1], sbmvr, sbmvi)
            add2 = _complex_mul(d22[0], d22[1], rbvr, rbvi)
            alpha_v += poolbr * (row0[0] + add1[0] + add2[0])
            alpha_v += poolbi * (row0[1] + add1[1] + add2[1])
            u1 = _complex_mul(t01[0], t01[1], sbmvr, sbmvi)
            u2 = _complex_mul(t02[0], t02[1], rbvr, rbvi)
            ur, ui = -(2.0 * u1[1] + u2[1]), 2.0 * u1[0] + u2[0]
            phi_v += ubvr * ur + ubvi * ui
            u1 = _complex_mul(t01[0], -t01[1], sbpvr, sbpvi)
            u2 = _complex_mul(t12[0], t12[1], rbvr, rbvi)
            ur, ui = 2.0 * u1[1] + u2[1], -2.0 * u1[0] - u2[0]
            phi_v += wbvr * ur + wbvi * ui
            u1 = _complex_mul(t20[0], t20[1], sbpvr, sbpvi)
            u2 = _complex_mul(t21[0], t21[1], sbmvr, sbmvi)
            ur, ui = -(u2[1] - u1[1]), u2[0] - u1[0]
            phi_v += poolbr * ur + poolbi * ui

        rotate = is_rf & ~is_inversion
        grad_alpha_v = tl.sum(tl.where(rotate, alpha_v, 0.0), axis=1)[:, None]
        grad_phi_v = tl.sum(tl.where(rotate, phi_v, 0.0), axis=1)[:, None]
        if pools == 1 or pools == 3:
            turning = tl.where(rotate, 1.0, 0.0)
            grad_alpha_v += sat_alpha_v * turning
            g_b0v += sat_b0_v * turning

        # Conjugate transpose of the rotation.
        n0 = _complex_mul(t00[0], -t00[1], pbvr, pbvi)
        n1 = _complex_mul(t01[0], t01[1], mbvr, mbvi)
        n2 = _complex_mul(t20[0], -t20[1], zbvr, zbvi)
        q0 = _complex_mul(t01[0], -t01[1], pbvr, pbvi)
        q1 = _complex_mul(t00[0], -t00[1], mbvr, mbvi)
        q2 = _complex_mul(t21[0], -t21[1], zbvr, zbvi)
        w0 = _complex_mul(t02[0], -t02[1], pbvr, pbvi)
        w1 = _complex_mul(t12[0], -t12[1], mbvr, mbvi)
        w2 = _complex_mul(t22[0], -t22[1], zbvr, zbvi)
        back_pr = n0[0] + n1[0] + n2[0]
        back_pi = n0[1] + n1[1] + n2[1]
        back_mr = q0[0] + q1[0] + q2[0]
        back_mi = q0[1] + q1[1] + q2[1]
        back_zr = w0[0] + w1[0] + w2[0]
        back_zi = w0[1] + w1[1] + w2[1]
        if profiled or dynamic:
            # A shaped pulse turned the states, so its own adjoint is what
            # goes back rather than the instant rotation's.
            back_pr, back_pi = shaped_pbr, shaped_pbi
            back_mr, back_mi = shaped_mbr, shaped_mbi
            back_zr, back_zi = shaped_zbr, shaped_zbi
        pbvr = tl.where(rotate, back_pr, pbvr)
        pbvi = tl.where(rotate, back_pi, pbvi)
        mbvr = tl.where(rotate, back_mr, mbvr)
        mbvi = tl.where(rotate, back_mi, mbvi)
        zbvr = tl.where(rotate, back_zr, zbvr)
        zbvi = tl.where(rotate, back_zi, zbvi)

        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip + event_base + event,
            grad_alpha_v * pulse_b1,
            mask=writes_flip,
        )
        tl.atomic_add(grad_phase + event_base + event, grad_phi_v, mask=writes_flip)
        if shimmed:
            # A pulse's transmit gradient belongs to the shim it drives, so with
            # several it lands in that shim's row rather than in a register
            # summed over the whole train.
            tl.atomic_add(
                grad_tissue + _B1_ROW * atom_count + row + atom,
                grad_alpha_v * event_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue + (_B1_PHASE_ROW + shim_rows - 1) * atom_count + row + atom,
                grad_phi_v,
                mask=writes_flip,
            )
        else:
            g_b1v += grad_alpha_v * event_flip
            g_b1pv += grad_phi_v

        if pools == 2 or pools == 3:
            n0 = _complex_mul(t00[0], -t00[1], ubvr, ubvi)
            n1 = _complex_mul(t01[0], t01[1], wbvr, wbvi)
            n2 = _complex_mul(t20[0], -t20[1], poolbr, poolbi)
            q0 = _complex_mul(t01[0], -t01[1], ubvr, ubvi)
            q1 = _complex_mul(t00[0], -t00[1], wbvr, wbvi)
            q2 = _complex_mul(t21[0], -t21[1], poolbr, poolbi)
            w0 = _complex_mul(t02[0], -t02[1], ubvr, ubvi)
            w1 = _complex_mul(t12[0], -t12[1], wbvr, wbvi)
            w2 = _complex_mul(t22[0], -t22[1], poolbr, poolbi)
            pool_back_pr = n0[0] + n1[0] + n2[0]
            pool_back_pi = n0[1] + n1[1] + n2[1]
            pool_back_mr = q0[0] + q1[0] + q2[0]
            pool_back_mi = q0[1] + q1[1] + q2[1]
            pool_back_zr = w0[0] + w1[0] + w2[0]
            pool_back_zi = w0[1] + w1[1] + w2[1]
            if profiled or dynamic:
                # A shaped pulse turned this pool too, so its own adjoint is
                # what goes back rather than the instant rotation's.
                pool_back_pr, pool_back_pi = pool_shaped_pbr, pool_shaped_pbi
                pool_back_mr, pool_back_mi = pool_shaped_mbr, pool_shaped_mbi
                pool_back_zr, pool_back_zi = pool_shaped_zbr, pool_shaped_zbi
            ubvr = tl.where(rotate, pool_back_pr, ubvr)
            ubvi = tl.where(rotate, pool_back_pi, ubvi)
            wbvr = tl.where(rotate, pool_back_mr, wbvr)
            wbvi = tl.where(rotate, pool_back_mi, wbvi)
            poolbr = tl.where(rotate, pool_back_zr, poolbr)
            poolbi = tl.where(rotate, pool_back_zi, poolbi)
            # An inversion turns the exchanging pool's longitudinal state as
            # well, so the efficiency carries what both left behind.
            g_invv += tl.sum(
                tl.where(invert, poolbr * -rbvr + poolbi * -rbvi, 0.0), axis=1
            )[:, None]
            poolbr = tl.where(invert, -atom_inv * poolbr, poolbr)
            poolbi = tl.where(invert, -atom_inv * poolbi, poolbi)
            avr, avi, bvr, bvi = _shift_adjoint(
                ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count
            )
            ubvr = tl.where(pre_shift, avr, ubvr)
            ubvi = tl.where(pre_shift, avi, ubvi)
            wbvr = tl.where(pre_shift, bvr, wbvr)
            wbvi = tl.where(pre_shift, bvi, wbvi)
        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count
        )
        pbvr = tl.where(pre_shift, avr, pbvr)
        pbvi = tl.where(pre_shift, avi, pbvi)
        mbvr = tl.where(pre_shift, bvr, mbvr)
        mbvi = tl.where(pre_shift, bvi, mbvi)

        # ---- relaxation and off-resonance adjoint ----
        # The damping is homogeneous of degree one in every transverse state it
        # acts on, so its gradient times the damping itself is the cotangent
        # taken against the states the interval leaves.
        pq = _complex_mul(qr, qi, xpvr, xpvi)
        mq = _complex_mul(qr, -qi, xmvr, xmvi)
        bare_cot_v = pbvr * pq[0] + pbvi * pq[1]
        bare_cot_v += mbvr * mq[0] + mbvi * mq[1]
        grad_e2_v = tl.sum(bare_cot_v * damp_t, axis=1)[:, None]
        cot2_v = bare_cot_v * bare2_value * damp_t
        pool_angle_v = empty
        if pools == 2 or pools == 3:
            # With an exchanging pool the damping sits inside the operator, so
            # the cotangent the interval leaves is taken against the states it
            # produced rather than against a scalar the free pool multiplies.
            plus_r = pbvr * rpvr + pbvi * rpvi + ubvr * rbpvr + ubvi * rbpvi
            plus_i = pbvr * rpvi - pbvi * rpvr + ubvr * rbpvi - ubvi * rbpvr
            minus_r = mbvr * rmvr + mbvi * rmvi + wbvr * rbmvr + wbvi * rbmvi
            minus_i = mbvr * rmvi - mbvi * rmvr + wbvr * rbmvi - wbvi * rbmvr
            cot2_v = plus_r + minus_r
            pool_angle_v = minus_i - plus_i
        if pools == 2 or pools == 3:
            # ``F-`` follows the conjugate of the operator, so its cotangent
            # lands on the entry itself rather than on the conjugate of it.
            def_r, def_i = carr, cari
            t11r, t11i = _complex_mul(
                pbvr * xpvr + pbvi * xpvi + mbvr * xmvr + mbvi * xmvi,
                pbvr * xpvi - pbvi * xpvr - mbvr * xmvi + mbvi * xmvr,
                def_r,
                def_i,
            )
            t12r, t12i = _complex_mul(
                pbvr * xbpvr + pbvi * xbpvi + mbvr * xbmvr + mbvi * xbmvi,
                pbvr * xbpvi - pbvi * xbpvr - mbvr * xbmvi + mbvi * xbmvr,
                def_r,
                def_i,
            )
            t21r, t21i = _complex_mul(
                ubvr * xpvr + ubvi * xpvi + wbvr * xmvr + wbvi * xmvi,
                ubvr * xpvi - ubvi * xpvr - wbvr * xmvi + wbvi * xmvr,
                def_r,
                def_i,
            )
            t22r, t22i = _complex_mul(
                ubvr * xbpvr + ubvi * xbpvi + wbvr * xbmvr + wbvi * xbmvi,
                ubvr * xbpvi - ubvi * xbpvr - wbvr * xbmvi + wbvi * xbmvr,
                def_r,
                def_i,
            )
            bar11 = (
                tl.sum(t11r, axis=1)[:, None],
                tl.sum(t11i, axis=1)[:, None],
                zero,
                zero,
            )
            bar12 = (
                tl.sum(t12r, axis=1)[:, None],
                tl.sum(t12i, axis=1)[:, None],
                zero,
                zero,
            )
            bar21 = (
                tl.sum(t21r, axis=1)[:, None],
                tl.sum(t21i, axis=1)[:, None],
                zero,
                zero,
            )
            bar22 = (
                tl.sum(t22r, axis=1)[:, None],
                tl.sum(t22i, axis=1)[:, None],
                zero,
                zero,
            )
            (
                back_r2,
                _q1,
                back_r2b,
                _q2,
                back_xexch,
                _q3,
                back_xbound,
                _q4,
                back_xfree,
                _q5,
                back_shift,
                _q6,
                back_xdt,
                _q7,
                back_xatt,
                _q8,
            ) = _two_pool_transverse_adjoint_jvp(
                r2_value,
                0.0,
                r2b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                atom_free,
                0.0,
                atom_shift,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
                bar11,
                bar12,
                bar21,
                bar22,
            )
            g_t2v += back_r2 * (-1000.0 / (atom_t2 * atom_t2))
            g_t2bv += back_r2b * (-1000.0 / (atom_t2b * atom_t2b))
            g_exchv += back_xexch
            # The free fraction is one less the pool's, so what reaches it
            # arrives at the pool's own with the sign turned.
            g_boundv += back_xbound - back_xfree
            if pools == 3:
                # The free share is one less both fractions, so what the
                # transverse operator leaves on it reaches the semisolid too.
                g_semiv -= back_xfree
            g_shiftv += back_shift
            xversal_dt = back_xdt
            xversal_att = back_xatt
            # The pool's transverse cotangents go back through the same
            # operator, transposed.
            ur, ui = _complex_mul(a11r, -a11i, pbvr, pbvi)
            vr_, vi_ = _complex_mul(a21r, -a21i, ubvr, ubvi)
            nub_pr, nub_pi = _complex_mul(ur + vr_, ui + vi_, carr, -cari)
            ur, ui = _complex_mul(a12r, -a12i, pbvr, pbvi)
            vr_, vi_ = _complex_mul(a22r, -a22i, ubvr, ubvi)
            nub_qr, nub_qi = _complex_mul(ur + vr_, ui + vi_, carr, -cari)
            ur, ui = _complex_mul(a11r, a11i, mbvr, mbvi)
            vr_, vi_ = _complex_mul(a21r, a21i, wbvr, wbvi)
            nwb_pr, nwb_pi = _complex_mul(ur + vr_, ui + vi_, carr, cari)
            ur, ui = _complex_mul(a12r, a12i, mbvr, mbvi)
            vr_, vi_ = _complex_mul(a22r, a22i, wbvr, wbvi)
            nwb_qr, nwb_qi = _complex_mul(ur + vr_, ui + vi_, carr, cari)
            pbvr, pbvi = nub_pr, nub_pi
            ubvr, ubvi = nub_qr, nub_qi
            mbvr, mbvi = nwb_pr, nwb_pi
            wbvr, wbvi = nwb_qr, nwb_qi

        per_angle_v = pool_angle_v
        if pools != 2 and pools != 3 and (off_axis or moving):
            po = _complex_mul(ovr, ovi, xpvr, xpvi)
            mo = _complex_mul(ovr, -ovi, xmvr, xmvi)
            # A turn of the transverse states and the off-resonance angle are
            # the same derivative; only the weight each order carries differs.
            per_angle_v = pbvr * -po[1] + pbvi * po[0]
            per_angle_v -= mbvr * -mo[1] + mbvi * mo[0]
        grad_angle_v = zero
        if off_axis or moving:
            grad_angle_v = tl.sum(per_angle_v, axis=1)[:, None]

        e1_v = empty
        grad_e1_v = zero
        long_damp_v = empty
        attenuation_v = zero
        two_pool_dt_v = zero
        if pools == 2 or pools == 3:
            attenuation_v += xversal_att
            two_pool_dt_v += xversal_dt
        zangle_v = empty
        if pools == 3:
            # The nine entries of the mixing operator and the three
            # recoveries, summed over the orders that share them, then pushed
            # back through the closed form once for the whole interval.
            spun_fr, spun_fi = _complex_mul(spin_r, spin_i, xzvr, xzvi)
            spun_br, spun_bi = _complex_mul(spin_r, spin_i, xbvr, xbvi)
            spun_cr, spun_ci = _complex_mul(spin_r, spin_i, xcvr, xcvi)
            e11_v = zbvr * spun_fr + zbvi * spun_fi
            e12_v = zbvr * spun_br + zbvi * spun_bi
            e13_v = zbvr * spun_cr + zbvi * spun_ci
            e21_v = poolbr * spun_fr + poolbi * spun_fi
            e22_v = poolbr * spun_br + poolbi * spun_bi
            e23_v = poolbr * spun_cr + poolbi * spun_ci
            e31_v = semibr * spun_fr + semibi * spun_fi
            e32_v = semibr * spun_br + semibi * spun_bi
            e33_v = semibr * spun_cr + semibi * spun_ci
            if tabulated:
                # Every gradient but the interval's own is linear in these
                # twelve, so the events sharing a length pool them here and
                # pay the closed form once each after the walk back.
                bar11 = tl.sum(e11_v, axis=1)[:, None]
                bar12 = tl.sum(e12_v, axis=1)[:, None]
                bar13 = tl.sum(e13_v, axis=1)[:, None]
                bar21 = tl.sum(e21_v, axis=1)[:, None]
                bar22 = tl.sum(e22_v, axis=1)[:, None]
                bar23 = tl.sum(e23_v, axis=1)[:, None]
                bar31 = tl.sum(e31_v, axis=1)[:, None]
                bar32 = tl.sum(e32_v, axis=1)[:, None]
                bar33 = tl.sum(e33_v, axis=1)[:, None]
                bar_free = tl.sum(tl.where(state == 0, zbvr, nil), axis=1)[:, None]
                bar_pool_b = tl.sum(tl.where(state == 0, poolbr, nil), axis=1)[:, None]
                bar_bound = tl.sum(tl.where(state == 0, semibr, nil), axis=1)[:, None]
                held = pool_bars + (local * row_count + pool_row) * 12
                tl.store(
                    held + 0,
                    tl.load(held + 0, mask=active_atom, other=0.0) + bar11,
                    mask=active_atom,
                )
                tl.store(
                    held + 1,
                    tl.load(held + 1, mask=active_atom, other=0.0) + bar12,
                    mask=active_atom,
                )
                tl.store(
                    held + 2,
                    tl.load(held + 2, mask=active_atom, other=0.0) + bar13,
                    mask=active_atom,
                )
                tl.store(
                    held + 3,
                    tl.load(held + 3, mask=active_atom, other=0.0) + bar21,
                    mask=active_atom,
                )
                tl.store(
                    held + 4,
                    tl.load(held + 4, mask=active_atom, other=0.0) + bar22,
                    mask=active_atom,
                )
                tl.store(
                    held + 5,
                    tl.load(held + 5, mask=active_atom, other=0.0) + bar23,
                    mask=active_atom,
                )
                tl.store(
                    held + 6,
                    tl.load(held + 6, mask=active_atom, other=0.0) + bar31,
                    mask=active_atom,
                )
                tl.store(
                    held + 7,
                    tl.load(held + 7, mask=active_atom, other=0.0) + bar32,
                    mask=active_atom,
                )
                tl.store(
                    held + 8,
                    tl.load(held + 8, mask=active_atom, other=0.0) + bar33,
                    mask=active_atom,
                )
                tl.store(
                    held + 9,
                    tl.load(held + 9, mask=active_atom, other=0.0) + bar_free,
                    mask=active_atom,
                )
                tl.store(
                    held + 10,
                    tl.load(held + 10, mask=active_atom, other=0.0) + bar_pool_b,
                    mask=active_atom,
                )
                tl.store(
                    held + 11,
                    tl.load(held + 11, mask=active_atom, other=0.0) + bar_bound,
                    mask=active_atom,
                )
                back_dt, back_att = _three_pool_interval_adjoint(
                    pool_table,
                    pool_row,
                    atom,
                    atom_count,
                    active_atom,
                    r1_value,
                    r1b_value,
                    r1c_value,
                    atom_exchange,
                    atom_semisolid_exchange,
                    atom_bound,
                    atom_semisolid,
                    hold_value,
                    bar11,
                    bar12,
                    bar13,
                    bar21,
                    bar22,
                    bar23,
                    bar31,
                    bar32,
                    bar33,
                    bar_free,
                    bar_pool_b,
                    bar_bound,
                )
                attenuation_v += back_att
                two_pool_dt_v += back_dt
            else:
                (
                    back_r1,
                    back_r1b,
                    back_r1c,
                    back_exch,
                    back_sexch,
                    back_bound,
                    back_semi,
                    back_dt,
                    back_att,
                    _q1,
                    _q2,
                    _q3,
                    _q4,
                    _q5,
                    _q6,
                    _q7,
                    _q8,
                    _q9,
                ) = _three_pool_step_adjoint_jvp(
                    r1_value,
                    nil,
                    r1b_value,
                    nil,
                    r1c_value,
                    nil,
                    atom_exchange,
                    nil,
                    atom_semisolid_exchange,
                    nil,
                    atom_bound,
                    nil,
                    atom_semisolid,
                    nil,
                    dt_value,
                    nil,
                    hold_value,
                    nil,
                    tl.sum(e11_v, axis=1)[:, None],
                    nil,
                    tl.sum(e12_v, axis=1)[:, None],
                    nil,
                    tl.sum(e13_v, axis=1)[:, None],
                    nil,
                    tl.sum(e21_v, axis=1)[:, None],
                    nil,
                    tl.sum(e22_v, axis=1)[:, None],
                    nil,
                    tl.sum(e23_v, axis=1)[:, None],
                    nil,
                    tl.sum(e31_v, axis=1)[:, None],
                    nil,
                    tl.sum(e32_v, axis=1)[:, None],
                    nil,
                    tl.sum(e33_v, axis=1)[:, None],
                    nil,
                    tl.sum(tl.where(state == 0, zbvr, nil), axis=1)[:, None],
                    nil,
                    tl.sum(tl.where(state == 0, poolbr, nil), axis=1)[:, None],
                    nil,
                    tl.sum(tl.where(state == 0, semibr, nil), axis=1)[:, None],
                    nil,
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                    narrow,
                )
                g_t1v += back_r1 * (-1000.0 / (atom_t1 * atom_t1))
                g_t1bv += back_r1b * (-1000.0 / (atom_t1b * atom_t1b))
                g_t1cv += back_r1c * (-1000.0 / (atom_t1c * atom_t1c))
                g_exchv += back_exch
                g_sexchv += back_sexch
                g_boundv += back_bound
                g_semiv += back_semi
                # Both halves of the interval reach the same two, so the
                # transverse pass has already put its share here.
                attenuation_v += back_att
                two_pool_dt_v += back_dt
            # All three pools take the same per-order damping and turn, so each
            # collects the cotangent of the mixture that reached it.
            sfr, sfi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            sbr, sbi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            scr, sci = _complex_mul(spin_r, spin_i, mix_cr, mix_ci)
            long_damp_v = (
                (zbvr * sfr + zbvi * sfi)
                + (poolbr * sbr + poolbi * sbi)
                + (semibr * scr + semibi * sci)
            )
            if moving:
                zangle_v = (
                    (zbvr * -sfi + zbvi * sfr)
                    + (poolbr * -sbi + poolbi * sbr)
                    + (semibr * -sci + semibi * scr)
                )
            col_fr, col_fi = _complex_mul(w11 * spin_r, -(w11 * spin_i), zbvr, zbvi)
            part_r, part_i = _complex_mul(w21 * spin_r, -(w21 * spin_i), poolbr, poolbi)
            col_fr, col_fi = col_fr + part_r, col_fi + part_i
            part_r, part_i = _complex_mul(w31 * spin_r, -(w31 * spin_i), semibr, semibi)
            col_fr, col_fi = col_fr + part_r, col_fi + part_i
            col_br, col_bi = _complex_mul(w12 * spin_r, -(w12 * spin_i), zbvr, zbvi)
            part_r, part_i = _complex_mul(w22 * spin_r, -(w22 * spin_i), poolbr, poolbi)
            col_br, col_bi = col_br + part_r, col_bi + part_i
            part_r, part_i = _complex_mul(w32 * spin_r, -(w32 * spin_i), semibr, semibi)
            col_br, col_bi = col_br + part_r, col_bi + part_i
            col_cr, col_ci = _complex_mul(w13 * spin_r, -(w13 * spin_i), zbvr, zbvi)
            part_r, part_i = _complex_mul(w23 * spin_r, -(w23 * spin_i), poolbr, poolbi)
            col_cr, col_ci = col_cr + part_r, col_ci + part_i
            part_r, part_i = _complex_mul(w33 * spin_r, -(w33 * spin_i), semibr, semibi)
            col_cr, col_ci = col_cr + part_r, col_ci + part_i
            zbvr, zbvi = col_fr, col_fi
            poolbr, poolbi = col_br, col_bi
            semibr, semibi = col_cr, col_ci
        elif pools > 0:
            # The four entries of the exchange operator and the two recoveries,
            # summed over the orders that share them, then pushed back through
            # the closed form once for the whole interval.
            spun_fr, spun_fi = _complex_mul(spin_r, spin_i, xzvr, xzvi)
            spun_br, spun_bi = _complex_mul(spin_r, spin_i, xbvr, xbvi)
            bar_e11 = tl.sum(zbvr * spun_fr + zbvi * spun_fi, axis=1)[:, None]
            bar_e12 = tl.sum(zbvr * spun_br + zbvi * spun_bi, axis=1)[:, None]
            bar_e21 = tl.sum(poolbr * spun_fr + poolbi * spun_fi, axis=1)[:, None]
            bar_e22 = tl.sum(poolbr * spun_br + poolbi * spun_bi, axis=1)[:, None]
            rec_f = tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            rec_b = tl.sum(tl.where(state == 0, poolbr, 0.0), axis=1)[:, None]
            (
                back_r1,
                back_r1b,
                back_exch,
                back_bound,
                back_dt,
                back_att,
                _t1,
                _t2,
                _t3,
                _t4,
                _t5,
                _t6,
            ) = _two_pool_step_adjoint_jvp(
                r1_value,
                0.0,
                r1b_value,
                0.0,
                atom_exchange,
                0.0,
                atom_bound,
                0.0,
                dt_value,
                0.0,
                wout_value,
                0.0,
                bar_e11,
                0.0,
                bar_e12,
                0.0,
                bar_e21,
                0.0,
                bar_e22,
                0.0,
                rec_f,
                0.0,
                rec_b,
                0.0,
            )
            # r1 = 1000/t1, so a rate gradient reaches the time through the
            # square of it.
            g_t1v += back_r1 * (-1000.0 / (atom_t1 * atom_t1))
            g_t1bv += back_r1b * (-1000.0 / (atom_t1b * atom_t1b))
            g_exchv += back_exch
            g_boundv += back_bound
            # Both halves of the interval reach the same two, so the
            # transverse pass has already put its share here.
            attenuation_v += back_att
            two_pool_dt_v += back_dt
            # Both pools take the same per-order damping and turn, so each
            # collects the cotangent of the mixture that reached it.
            sfr, sfi = _complex_mul(spin_r, spin_i, mix_fr, mix_fi)
            sbr, sbi = _complex_mul(spin_r, spin_i, mix_br, mix_bi)
            long_damp_v = (zbvr * sfr + zbvi * sfi) + (poolbr * sbr + poolbi * sbi)
            if moving:
                zangle_v = (zbvr * -sfi + zbvi * sfr) + (poolbr * -sbi + poolbi * sbr)
            back_zr, back_zi = _complex_mul(pe11 * spin_r, -(pe11 * spin_i), zbvr, zbvi)
            cross_zr, cross_zi = _complex_mul(
                pe21 * spin_r, -(pe21 * spin_i), poolbr, poolbi
            )
            back_br, back_bi = _complex_mul(pe12 * spin_r, -(pe12 * spin_i), zbvr, zbvi)
            cross_br, cross_bi = _complex_mul(
                pe22 * spin_r, -(pe22 * spin_i), poolbr, poolbi
            )
            poolbr = back_br + cross_br
            poolbi = back_bi + cross_bi
            zbvr = back_zr + cross_zr
            zbvi = back_zi + cross_zi
        else:
            spun = _complex_mul(szr, szi, xzvr, xzvi)
            e1_v = zbvr * spun[0] + zbvi * spun[1]
            grad_e1_v = tl.sum(e1_v * damp_z, axis=1)[:, None]
            grad_e1_v -= tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            # The longitudinal states turn too, and by a whole order rather
            # than the transverse half-order more.
            if moving:
                zo = _complex_mul(lvr, lvi, xzvr, xzvi)
                zangle_v = zbvr * -zo[1] + zbvi * zo[0]
            long_damp_v = e1_v * bare1_value * damp_z
            zbvr, zbvi = _complex_mul(lvr, -lvi, zbvr, zbvi)

        spread_v = zero
        if diffusing:
            # The rate and the interval multiply every order's b-weight, so
            # both take a weighted sum rather than one scalar. Order zero
            # carries no longitudinal weight, which keeps recovery out of this.
            weighted_v = long_damp_v * longitudinal_weight + cot2_v * transverse_weight
            spread_v = tl.sum(weighted_v, axis=1)[:, None]
            g_diffv += -spread_v * dt_value

        wound_v = zero
        wash_v = zero
        if moving:
            wound_v = tl.sum(per_angle_v * (order + 0.5) + zangle_v * order, axis=1)[
                :, None
            ]
            g_flowv += -wound_v * dt_value
            # Washout scales both relaxation factors, so its gradient is the
            # one they already carry, taken against the factors before that
            # scaling. Past the clamp the interval has replaced the voxel
            # outright and nothing further depends on the rate.
            live = (atom_washout * dt_value < 1.0).to(tl.float32)
            transverse_dry = (
                zero if pools == 2 or pools == 3 else grad_e2_v * dry2_value
            )
            wash_v = -live * (grad_e1_v * dry1_value + transverse_dry + attenuation_v)
            g_washv += wash_v * dt_value

        if pools != 2 and pools != 3:
            pbvr, pbvi = _complex_mul(ovr, -ovi, pbvr, pbvi)
            mbvr, mbvi = _complex_mul(ovr, ovi, mbvr, mbvi)

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        g_t1v += grad_e1_v * (bare1_value * dt_value * inverse1_value)
        if pools != 2 and pools != 3:
            g_t2v += grad_e2_v * (bare2_value * dt_value * inverse2_value)

        turn = -2.0 * 3.141592653589793
        g_b0v += grad_angle_v * (turn * dt_value)

        duration_v = -grad_e1_v * (r1_value * bare1_value)
        if pools != 2 and pools != 3:
            duration_v -= grad_e2_v * (r2_value * bare2_value)
        duration_v += grad_angle_v * (turn * atom_b0) + two_pool_dt_v
        duration_v += -spread_v * atom_damping - wound_v * atom_flow
        duration_v += wash_v * atom_washout
        tl.atomic_add(grad_duration + event_base + event, duration_v, mask=active_atom)

    if pools == 3 and tabulated:
        # One closed form per distinct length rather than one per event. The
        # walk back pooled the cotangents the eigenvalues are pushed through,
        # and the closed form is linear in them, so the pieces of the sum are
        # the sum of the pieces.
        for row in range(0, row_count):
            held = pool_bars + (local * row_count + row) * 12
            row_dt = tl.load(pool_durations + row) + zero
            one_att = _washout(atom_washout, row_dt) if moving else 1.0 + 0.0 * row_dt
            nil = 0.0 * row_dt
            (
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
            ) = _three_pool_pieces_jvp(
                r1_value,
                nil,
                r1b_value,
                nil,
                r1c_value,
                nil,
                atom_exchange,
                nil,
                atom_semisolid_exchange,
                nil,
                atom_bound,
                nil,
                atom_semisolid,
                nil,
                row_dt,
                nil,
                narrow,
            )
            (
                three_def_00,
                three_dif_00,
                three_def_01,
                three_dif_01,
                three_def_02,
                three_dif_02,
                three_def_10,
                three_dif_10,
                three_def_11,
                three_dif_11,
                three_def_12,
                three_dif_12,
                three_def_20,
                three_dif_20,
                three_def_21,
                three_dif_21,
                three_def_22,
                three_dif_22,
            ) = _three_pool_assemble_jvp(
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
                narrow,
            )
            (
                back_r1,
                back_r1b,
                back_r1c,
                back_exch,
                back_sexch,
                back_bound,
                back_semi,
                back_dt,
                back_att,
                _q1,
                _q2,
                _q3,
                _q4,
                _q5,
                _q6,
                _q7,
                _q8,
                _q9,
            ) = _three_pool_step_adjoint_jvp(
                r1_value,
                nil,
                r1b_value,
                nil,
                r1c_value,
                nil,
                atom_exchange,
                nil,
                atom_semisolid_exchange,
                nil,
                atom_bound,
                nil,
                atom_semisolid,
                nil,
                row_dt,
                nil,
                one_att,
                nil,
                tl.load(held + 0, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 1, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 2, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 3, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 4, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 5, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 6, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 7, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 8, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 9, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 10, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 11, mask=active_atom, other=0.0),
                nil,
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
                three_def_00,
                three_dif_00,
                three_def_01,
                three_dif_01,
                three_def_02,
                three_dif_02,
                three_def_10,
                three_dif_10,
                three_def_11,
                three_dif_11,
                three_def_12,
                three_dif_12,
                three_def_20,
                three_dif_20,
                three_def_21,
                three_dif_21,
                three_def_22,
                three_dif_22,
                narrow,
            )
            g_t1v += back_r1 * (-1000.0 / (atom_t1 * atom_t1))
            g_t1bv += back_r1b * (-1000.0 / (atom_t1b * atom_t1b))
            g_t1cv += back_r1c * (-1000.0 / (atom_t1c * atom_t1c))
            g_exchv += back_exch
            g_sexchv += back_sexch
            g_boundv += back_bound
            g_semiv += back_semi

    velocity_v = g_flowv * flow_scale + g_washv * direction * washout_scale
    values = (
        g_t1v,
        g_t2v,
        g_m0v,
        g_b1v,
        g_b1pv,
        g_b0v,
        g_invv,
        g_diffv,
        velocity_v,
    )
    if pools > 0:
        # The fraction also sets where each pool starts, which the walk back
        # reaches last.
        g_boundv += tl.sum(tl.where(state == 0, poolbr - zbvr, 0.0), axis=1)[:, None]
    if pools == 3:
        g_semiv += tl.sum(tl.where(state == 0, semibr - zbvr, 0.0), axis=1)[:, None]
        semisolid_row = _BOUND_ROW + 2 * (shim_rows - 1)
        tl.atomic_add(
            grad_tissue + semisolid_row * atom_count + atom,
            g_semiv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (semisolid_row + 1) * atom_count + atom,
            g_sexchv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (semisolid_row + 2) * atom_count + atom,
            g_t1cv,
            mask=active_atom,
        )
    if pools == 2 or pools == 3:
        base_row = _POOL_B_ROW + 2 * (shim_rows - 1)
        tl.atomic_add(
            grad_tissue + base_row * atom_count + atom,
            g_boundv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 1) * atom_count + atom,
            g_exchv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 2) * atom_count + atom,
            g_t1bv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 3) * atom_count + atom,
            g_t2bv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 4) * atom_count + atom,
            g_shiftv,
            mask=active_atom,
        )
    if pools == 1:
        base_row = _BOUND_ROW + 2 * (shim_rows - 1)
        tl.atomic_add(
            grad_tissue + base_row * atom_count + atom,
            g_boundv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 1) * atom_count + atom,
            g_exchv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue + (base_row + 2) * atom_count + atom,
            g_t1bv,
            mask=active_atom,
        )
    for parameter in tl.static_range(_FREE_POOL_COUNT):
        # The transmit pair went to its shim's row above when there is more
        # than one; the rest sit past whatever rows that pair took.
        if not shimmed or (parameter != _B1_ROW and parameter != _B1_PHASE_ROW):
            plane = (
                parameter if parameter < _B1_ROW else parameter + 2 * (shim_rows - 1)
            )
            tl.atomic_add(
                grad_tissue + plane * atom_count + atom,
                values[parameter],
                mask=active_atom,
            )


@triton.jit(do_not_specialize=["profile_bins", "lineshape_bins"])
def _epg_vjp_jvp_kernel(
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
    exchange_rate,
    t1_bound,
    pool_b_fraction,
    pool_b_exchange,
    t1_pool_b,
    t2_pool_b,
    pool_b_shift,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
    pairs,
    pair_index,
    pair_direction,
    grad_pair_value,
    grad_pair_tangent,
    dot_t1,
    dot_t2,
    dot_m0,
    dot_b1,
    dot_b1_phase,
    dot_b0,
    dot_inversion_efficiency,
    dot_diffusion,
    dot_velocity,
    dot_bound_fraction,
    dot_exchange_rate,
    dot_t1_bound,
    dot_pool_b_fraction,
    dot_pool_b_exchange,
    dot_t1_pool_b,
    dot_t2_pool_b,
    dot_pool_b_shift,
    dot_duration,
    dot_flip,
    dot_phase,
    duration_row,
    pool_table,
    pool_bars,
    pool_durations,
    row_count,
    grad_output_real,
    grad_output_imag,
    grad_tissue_value,
    grad_tissue_tangent,
    grad_flip_value,
    grad_flip_tangent,
    grad_phase_value,
    grad_phase_tangent,
    grad_duration_value,
    grad_duration_tangent,
    trajectory_vr,
    trajectory_vi,
    trajectory_tr,
    trajectory_ti,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shim_rows,
    shimmed: tl.constexpr,
    locations: tl.constexpr,
    profiled: tl.constexpr,
    profile_bins,
    dynamic: tl.constexpr,
    directed: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    broadened: tl.constexpr,
    lineshape_bins,
    pools: tl.constexpr,
    narrow: tl.constexpr,
    tabulated: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations
    local = problem - problem_base
    # A second pool rides along as planes of its own: it enters an event as its
    # own vector and the RF operator acts on it, so the reverse sweep cannot
    # replay it from the free pool's. A semisolid pool adds one plane, a
    # chemically exchanging one three, and the two together add four.
    record_stride = (
        7 if pools == 3 else (6 if pools == 2 else (4 if pools == 1 else 3))
    ) * state_count
    trajectory = local * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count
    bound_plane = 3 * state_count
    bplus_plane = 4 * state_count
    bminus_plane = 5 * state_count
    semisolid_plane = 6 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    pvr = empty
    pvi = empty
    ptr = empty
    pti = empty
    mvr = empty
    mvi = empty
    mtr = empty
    mti = empty
    bvr = empty
    bvi = empty
    btr = empty
    bti = empty
    bpvr = empty
    bpvi = empty
    bptr = empty
    bpti = empty
    bmvr = empty
    bmvi = empty
    bmtr = empty
    bmti = empty
    cvr = empty
    cvi = empty
    ctr = empty
    cti = empty
    atom_bound = 0.0
    d_boundf = 0.0
    atom_exchange = 0.0
    d_exchange = 0.0
    atom_t1b = 1.0
    d_t1b = 0.0
    r1b_value = 0.0
    r1b_tangent = 0.0
    atom_t2b = 1.0
    d_t2b = 0.0
    r2b_value = 0.0
    r2b_tangent = 0.0
    atom_shift = 0.0
    d_shift = 0.0
    atom_semisolid = 0.0
    d_semisolidf = 0.0
    atom_semisolid_exchange = 0.0
    d_semisolid_exchange = 0.0
    r1c_value = 0.0
    r1c_tangent = 0.0
    if pools == 1:
        atom_bound = tl.load(bound_fraction + scalar_atom, mask=active_atom, other=0.0)
        d_boundf = tl.load(
            dot_bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        d_exchange = tl.load(
            dot_exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        atom_t1b = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        d_t1b = tl.load(dot_t1_bound + scalar_atom, mask=active_atom, other=0.0)
        r1b_value = 1000.0 / atom_t1b
        r1b_tangent = -1000.0 * d_t1b / (atom_t1b * atom_t1b)
    if pools == 2 or pools == 3:
        atom_bound = tl.load(pool_b_fraction + scalar_atom, mask=active_atom, other=0.0)
        d_boundf = tl.load(
            dot_pool_b_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(
            pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        d_exchange = tl.load(
            dot_pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        atom_t1b = tl.load(t1_pool_b + scalar_atom, mask=active_atom, other=1.0)
        d_t1b = tl.load(dot_t1_pool_b + scalar_atom, mask=active_atom, other=0.0)
        r1b_value = 1000.0 / atom_t1b
        r1b_tangent = -1000.0 * d_t1b / (atom_t1b * atom_t1b)
        atom_t2b = tl.load(t2_pool_b + scalar_atom, mask=active_atom, other=1.0)
        d_t2b = tl.load(dot_t2_pool_b + scalar_atom, mask=active_atom, other=0.0)
        r2b_value = 1000.0 / atom_t2b
        r2b_tangent = -1000.0 * d_t2b / (atom_t2b * atom_t2b)
        atom_shift = tl.load(pool_b_shift + scalar_atom, mask=active_atom, other=0.0)
        d_shift = tl.load(dot_pool_b_shift + scalar_atom, mask=active_atom, other=0.0)
    if pools == 3:
        atom_semisolid = tl.load(
            bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        d_semisolidf = tl.load(
            dot_bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_semisolid_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        d_semisolid_exchange = tl.load(
            dot_exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        held_semisolid = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        d_semisolid_t1 = tl.load(
            dot_t1_bound + scalar_atom, mask=active_atom, other=0.0
        )
        r1c_value = 1000.0 / held_semisolid
        r1c_tangent = -1000.0 * d_semisolid_t1 / (held_semisolid * held_semisolid)
        cvr = empty + tl.where(state == 0, atom_semisolid + 0.0, 0.0)
        ctr = empty + tl.where(state == 0, d_semisolidf + 0.0, 0.0)
    if pools > 0:
        atom_free = 1.0 - atom_bound - atom_semisolid
        d_free = -d_boundf - d_semisolidf
        zvr = empty + tl.where(state == 0, atom_free, 0.0)
        ztr = empty + tl.where(state == 0, d_free, 0.0)
        bvr = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
        btr = empty + tl.where(state == 0, d_boundf + 0.0, 0.0)
    else:
        atom_free = 1.0 + 0.0 * atom_bound
        d_free = 0.0 * atom_bound
        zvr = empty + tl.where(state == 0, 1.0, 0.0)
        ztr = empty
    zvi = empty
    zti = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_b1_phase = 0.0
    atom_b0 = 0.0
    if off_axis:
        atom_b1_phase = tl.load(b1_phase + scalar_atom, mask=active_atom, other=0.0)
        atom_b0 = tl.load(b0 + scalar_atom, mask=active_atom, other=0.0)
    atom_inv = 1.0
    if inverting:
        atom_inv = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    d_t1 = tl.load(dot_t1 + atom, mask=active_atom, other=0.0)
    d_t2 = tl.load(dot_t2 + atom, mask=active_atom, other=0.0)
    d_m0 = 0.0
    if density:
        d_m0 = tl.load(dot_m0 + scalar_atom, mask=active_atom, other=0.0)
    d_b1 = 0.0
    if transmit:
        d_b1 = tl.load(dot_b1 + scalar_atom, mask=active_atom, other=0.0)
    d_b1_phase = 0.0
    d_b0 = 0.0
    if off_axis:
        d_b1_phase = tl.load(dot_b1_phase + scalar_atom, mask=active_atom, other=0.0)
        d_b0 = tl.load(dot_b0 + scalar_atom, mask=active_atom, other=0.0)
    d_inv = 0.0
    if inverting:
        d_inv = tl.load(
            dot_inversion_efficiency + scalar_atom, mask=active_atom, other=0.0
        )
    atom_damping = 0.0
    d_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
        d_damping = tl.load(dot_diffusion + scalar_atom, mask=active_atom, other=0.0)
    atom_flow = 0.0
    d_flow = 0.0
    direction = 0.0
    atom_washout = 0.0
    d_washout = 0.0
    if moving:
        atom_velocity = tl.load(velocity + scalar_atom, mask=active_atom, other=0.0)
        d_velocity = tl.load(dot_velocity + scalar_atom, mask=active_atom, other=0.0)
        atom_flow = atom_velocity * flow_scale
        d_flow = d_velocity * flow_scale
        # |v| has no derivative at the origin, so a still voxel contributes
        # none.
        direction = (atom_velocity > 0.0).to(tl.float32) - (atom_velocity < 0.0).to(
            tl.float32
        )
        atom_washout = tl.abs(atom_velocity) * washout_scale
        d_washout = direction * d_velocity * washout_scale
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    r1_value = 1000.0 / atom_t1
    r1_tangent = -1000.0 * d_t1 / (atom_t1 * atom_t1)
    r2_value = 1000.0 / atom_t2
    r2_tangent = -1000.0 * d_t2 / (atom_t2 * atom_t2)

    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_vr + slot, pvr, mask=state_mask)
        tl.store(trajectory_vi + slot, pvi, mask=state_mask)
        tl.store(trajectory_tr + slot, ptr, mask=state_mask)
        tl.store(trajectory_ti + slot, pti, mask=state_mask)
        tl.store(trajectory_vr + slot + minus_plane, mvr, mask=state_mask)
        tl.store(trajectory_vi + slot + minus_plane, mvi, mask=state_mask)
        tl.store(trajectory_tr + slot + minus_plane, mtr, mask=state_mask)
        tl.store(trajectory_ti + slot + minus_plane, mti, mask=state_mask)
        tl.store(trajectory_vr + slot + long_plane, zvr, mask=state_mask)
        tl.store(trajectory_vi + slot + long_plane, zvi, mask=state_mask)
        tl.store(trajectory_tr + slot + long_plane, ztr, mask=state_mask)
        tl.store(trajectory_ti + slot + long_plane, zti, mask=state_mask)
        if pools > 0:
            tl.store(trajectory_vr + slot + bound_plane, bvr, mask=state_mask)
            tl.store(trajectory_vi + slot + bound_plane, bvi, mask=state_mask)
            tl.store(trajectory_tr + slot + bound_plane, btr, mask=state_mask)
            tl.store(trajectory_ti + slot + bound_plane, bti, mask=state_mask)
        if pools == 2 or pools == 3:
            tl.store(trajectory_vr + slot + bplus_plane, bpvr, mask=state_mask)
            tl.store(trajectory_vi + slot + bplus_plane, bpvi, mask=state_mask)
            tl.store(trajectory_tr + slot + bplus_plane, bptr, mask=state_mask)
            tl.store(trajectory_ti + slot + bplus_plane, bpti, mask=state_mask)
            tl.store(trajectory_vr + slot + bminus_plane, bmvr, mask=state_mask)
            tl.store(trajectory_vi + slot + bminus_plane, bmvi, mask=state_mask)
            tl.store(trajectory_tr + slot + bminus_plane, bmtr, mask=state_mask)
            tl.store(trajectory_ti + slot + bminus_plane, bmti, mask=state_mask)
        if pools == 3:
            tl.store(trajectory_vr + slot + semisolid_plane, cvr, mask=state_mask)
            tl.store(trajectory_vi + slot + semisolid_plane, cvi, mask=state_mask)
            tl.store(trajectory_tr + slot + semisolid_plane, ctr, mask=state_mask)
            tl.store(trajectory_ti + slot + semisolid_plane, cti, mask=state_mask)

        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        dt_tangent = _event_value(
            dot_duration, event_base, event, active_atom, single_train
        )
        wout_value = 1.0
        wout_tangent = 0.0
        if moving:
            wout_value, wout_tangent = _washout_jvp(
                atom_washout, d_washout, dt_value, dt_tangent
            )
        dry1_value = tl.exp(-r1_value * dt_value)
        dry1_tangent = -dry1_value * (r1_value * dt_tangent + r1_tangent * dt_value)
        dry2_value = tl.exp(-r2_value * dt_value)
        dry2_tangent = -dry2_value * (r2_value * dt_tangent + r2_tangent * dt_value)
        e1_value = dry1_value * wout_value
        e1_tangent = dry1_tangent * wout_value + dry1_value * wout_tangent
        e2_value = dry2_value * wout_value
        e2_tangent = dry2_tangent * wout_value + dry2_value * wout_tangent
        damp_z = 1.0
        damp_z_tangent = 0.0
        damp_t = 1.0
        damp_t_tangent = 0.0
        if diffusing:
            damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
                atom_damping, d_damping, dt_value, dt_tangent, order
            )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t
        turn_t = 0.0
        dturn_t = 0.0
        szr, szi, sztr, szti = 1.0, 0.0, 0.0, 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, dt_value, order)
            d_turn = d_flow * dt_value + atom_flow * dt_tangent
            dturn_z = -order * d_turn
            dturn_t = -(order + 0.5) * d_turn
            szr, szi, sztr, szti = _dual_polar(turn_z, dturn_z)
        qr, qi, qtr, qti = 1.0, 0.0, 0.0, 0.0
        if off_axis or moving:
            angle_value = -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
            angle_tangent = (
                -2.0 * 3.141592653589793 * (d_b0 * dt_value + atom_b0 * dt_tangent)
                + dturn_t
            )
            qr, qi, qtr, qti = _dual_polar(angle_value, angle_tangent)
        ovr, ovi, otr, oti = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti)
        lvr, lvi, ltr, lti = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti)

        # The damping and the off-resonance turn both pools take; with an
        # exchanging one the relaxation itself sits inside the operator instead
        # of in the scalar the free pool alone multiplies by.
        carried = _dual_scale(damp_t, damp_t_tangent, qr, qi, qtr, qti)
        if pools == 2 or pools == 3:
            across = _two_pool_transverse_step_jvp(
                r2_value,
                r2_tangent,
                r2b_value,
                r2b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                atom_free,
                d_free,
                atom_shift,
                d_shift,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
            )
            a11 = (across[0], across[1], across[8], across[9])
            a12 = (across[2], across[3], across[10], across[11])
            a21 = (across[4], across[5], across[12], across[13])
            a22 = (across[6], across[7], across[14], across[15])
            free_plus = (pvr, pvi, ptr, pti)
            pool_plus = (bpvr, bpvi, bptr, bpti)
            free_minus = (mvr, mvi, mtr, mti)
            pool_minus = (bmvr, bmvi, bmtr, bmti)
            conjugated = _dual_conj(carried)
            # ``F-`` takes the conjugate of the operator entry by entry, not
            # its transpose: it is the conjugate state following the conjugate
            # map.
            pvr, pvi, ptr, pti = _dual_product(
                _dual_add(_dual_product(a11, free_plus), _dual_product(a12, pool_plus)),
                carried,
            )
            bpvr, bpvi, bptr, bpti = _dual_product(
                _dual_add(_dual_product(a21, free_plus), _dual_product(a22, pool_plus)),
                carried,
            )
            mvr, mvi, mtr, mti = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(a11), free_minus),
                    _dual_product(_dual_conj(a12), pool_minus),
                ),
                conjugated,
            )
            bmvr, bmvi, bmtr, bmti = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(a21), free_minus),
                    _dual_product(_dual_conj(a22), pool_minus),
                ),
                conjugated,
            )
        else:
            pvr, pvi, ptr, pti = _dual_mul(ovr, ovi, otr, oti, pvr, pvi, ptr, pti)
            mvr, mvi, mtr, mti = _dual_mul(ovr, -ovi, otr, -oti, mvr, mvi, mtr, mti)
        if pools == 3:
            # Three pools mix through a 3x3 formed in double, tangent and all:
            # a direction through an operator this ill-conditioned needs the
            # width as much as the value does.
            if tabulated:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_t11,
                    d_t12,
                    d_t13,
                    d_t21,
                    d_t22,
                    d_t23,
                    d_t31,
                    d_t32,
                    d_t33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_from_table_jvp(
                    pool_table,
                    tl.load(
                        duration_row + event_base + event,
                        mask=active_atom,
                        other=0,
                    ),
                    atom,
                    atom_count,
                    active_atom,
                    r1_value,
                    r1b_value,
                    r1c_value,
                    atom_exchange,
                    atom_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_tangent,
                    wout_value,
                    wout_tangent,
                )
            else:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_t11,
                    d_t12,
                    d_t13,
                    d_t21,
                    d_t22,
                    d_t23,
                    d_t31,
                    d_t32,
                    d_t33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_step_jvp(
                    r1_value,
                    r1_tangent,
                    r1b_value,
                    r1b_tangent,
                    r1c_value,
                    r1c_tangent,
                    atom_exchange,
                    d_exchange,
                    atom_semisolid_exchange,
                    d_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_value,
                    dt_tangent,
                    wout_value,
                    wout_tangent,
                    narrow,
                )
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            was_free = (zvr, zvi, ztr, zti)
            was_pool_b = (bvr, bvi, btr, bti)
            was_semisolid = (cvr, cvi, ctr, cti)
            mixed_free = _dual_add(
                _dual_add(
                    _dual_scale(
                        t11, d_t11, was_free[0], was_free[1], was_free[2], was_free[3]
                    ),
                    _dual_scale(
                        t12,
                        d_t12,
                        was_pool_b[0],
                        was_pool_b[1],
                        was_pool_b[2],
                        was_pool_b[3],
                    ),
                ),
                _dual_scale(
                    t13,
                    d_t13,
                    was_semisolid[0],
                    was_semisolid[1],
                    was_semisolid[2],
                    was_semisolid[3],
                ),
            )
            mixed_pool_b = _dual_add(
                _dual_add(
                    _dual_scale(
                        t21, d_t21, was_free[0], was_free[1], was_free[2], was_free[3]
                    ),
                    _dual_scale(
                        t22,
                        d_t22,
                        was_pool_b[0],
                        was_pool_b[1],
                        was_pool_b[2],
                        was_pool_b[3],
                    ),
                ),
                _dual_scale(
                    t23,
                    d_t23,
                    was_semisolid[0],
                    was_semisolid[1],
                    was_semisolid[2],
                    was_semisolid[3],
                ),
            )
            mixed_semisolid = _dual_add(
                _dual_add(
                    _dual_scale(
                        t31, d_t31, was_free[0], was_free[1], was_free[2], was_free[3]
                    ),
                    _dual_scale(
                        t32,
                        d_t32,
                        was_pool_b[0],
                        was_pool_b[1],
                        was_pool_b[2],
                        was_pool_b[3],
                    ),
                ),
                _dual_scale(
                    t33,
                    d_t33,
                    was_semisolid[0],
                    was_semisolid[1],
                    was_semisolid[2],
                    was_semisolid[3],
                ),
            )
            zvr, zvi, ztr, zti = _dual_mul(
                spin[0],
                spin[1],
                spin[2],
                spin[3],
                mixed_free[0],
                mixed_free[1],
                mixed_free[2],
                mixed_free[3],
            )
            bvr, bvi, btr, bti = _dual_mul(
                spin[0],
                spin[1],
                spin[2],
                spin[3],
                mixed_pool_b[0],
                mixed_pool_b[1],
                mixed_pool_b[2],
                mixed_pool_b[3],
            )
            cvr, cvi, ctr, cti = _dual_mul(
                spin[0],
                spin[1],
                spin[2],
                spin[3],
                mixed_semisolid[0],
                mixed_semisolid[1],
                mixed_semisolid[2],
                mixed_semisolid[3],
            )
            zvr += tl.where(state == 0, grow_free, 0.0)
            ztr += tl.where(state == 0, d_grow_free, 0.0)
            bvr += tl.where(state == 0, grow_pool_b, 0.0)
            btr += tl.where(state == 0, d_grow_pool_b, 0.0)
            cvr += tl.where(state == 0, grow_semisolid, 0.0)
            ctr += tl.where(state == 0, d_grow_semisolid, 0.0)
        elif pools > 0:
            # The exchange operator is a property of the interval, not of a
            # dephasing order, so it is formed once and the per-order damping
            # and turn multiply it.
            (
                pe11,
                pe12,
                pe21,
                pe22,
                prec_f,
                prec_b,
                de11,
                de12,
                de21,
                de22,
                drec_f,
                drec_b,
            ) = _two_pool_step_jvp(
                r1_value,
                r1_tangent,
                r1b_value,
                r1b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
            )
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            free_part = _dual_scale(pe11, de11, zvr, zvi, ztr, zti)
            cross_in = _dual_scale(pe12, de12, bvr, bvi, btr, bti)
            cross_out = _dual_scale(pe21, de21, zvr, zvi, ztr, zti)
            bound_part = _dual_scale(pe22, de22, bvr, bvi, btr, bti)
            zvr, zvi, ztr, zti = _dual_mul(
                spin[0],
                spin[1],
                spin[2],
                spin[3],
                free_part[0] + cross_in[0],
                free_part[1] + cross_in[1],
                free_part[2] + cross_in[2],
                free_part[3] + cross_in[3],
            )
            bvr, bvi, btr, bti = _dual_mul(
                spin[0],
                spin[1],
                spin[2],
                spin[3],
                cross_out[0] + bound_part[0],
                cross_out[1] + bound_part[1],
                cross_out[2] + bound_part[2],
                cross_out[3] + bound_part[3],
            )
            zvr += tl.where(state == 0, prec_f, 0.0)
            ztr += tl.where(state == 0, drec_f, 0.0)
            bvr += tl.where(state == 0, prec_b, 0.0)
            btr += tl.where(state == 0, drec_b, 0.0)
        else:
            zvr, zvi, ztr, zti = _dual_mul(lvr, lvi, ltr, lti, zvr, zvi, ztr, zti)
            zvr += tl.where(state == 0, recovery_value, 0.0)
            ztr += tl.where(state == 0, recovery_tangent, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count)
        str_, sti, wtr, wti = _shift(ptr, pti, mtr, mti, state, state_mask, state_count)
        pvr = tl.where(pre_shift, svr, pvr)
        pvi = tl.where(pre_shift, svi, pvi)
        ptr = tl.where(pre_shift, str_, ptr)
        pti = tl.where(pre_shift, sti, pti)
        mvr = tl.where(pre_shift, wvr, mvr)
        mvi = tl.where(pre_shift, wvi, mvi)
        mtr = tl.where(pre_shift, wtr, mtr)
        mti = tl.where(pre_shift, wti, mti)
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count
            )
            str_, sti, wtr, wti = _shift(
                bptr, bpti, bmtr, bmti, state, state_mask, state_count
            )
            bpvr = tl.where(pre_shift, svr, bpvr)
            bpvi = tl.where(pre_shift, svi, bpvi)
            bptr = tl.where(pre_shift, str_, bptr)
            bpti = tl.where(pre_shift, sti, bpti)
            bmvr = tl.where(pre_shift, wvr, bmvr)
            bmvi = tl.where(pre_shift, wvi, bmvi)
            bmtr = tl.where(pre_shift, wtr, bmtr)
            bmti = tl.where(pre_shift, wti, bmti)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, zvr, zvi, ztr, zti)
        zvr = tl.where(invert, ivr, zvr)
        zvi = tl.where(invert, ivi, zvi)
        ztr = tl.where(invert, itr, ztr)
        zti = tl.where(invert, iti, zti)
        if pools == 2 or pools == 3:
            # A semisolid pool is saturated by an adiabatic sweep rather than
            # turned over; a chemically exchanging one is free water and
            # inverts like any other.
            ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, bvr, bvi, btr, bti)
            bvr = tl.where(invert, ivr, bvr)
            bvi = tl.where(invert, ivi, bvi)
            btr = tl.where(invert, itr, btr)
            bti = tl.where(invert, iti, bti)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_dot_flip = _event_value(
            dot_flip, event_base, event, active_atom, single_train
        )
        event_phase = _event_value(phase, event_base, event, active_atom, single_train)
        event_dot_phase = _event_value(
            dot_phase, event_base, event, active_atom, single_train
        )
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = 1.0
            if transmit:
                atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            if off_axis:
                atom_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )
            d_b1 = tl.load(dot_b1 + row + atom, mask=active_atom, other=0.0)
            if off_axis:
                d_b1_phase = tl.load(
                    dot_b1_phase + row + atom, mask=active_atom, other=0.0
                )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * d_b1
        phi_value = event_phase + atom_b1_phase
        phi_tangent = event_dot_phase + d_b1_phase
        if pools == 1 or pools == 3:
            # The semisolid pool absorbs the power the pulse deposits, so it
            # reads the bare flip the transmit field gives the voxel -- not the
            # slice-shaped rotation the free pool takes from the table.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, shape_slope = _lineshape_at_slope(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            shape_tangent = shape_slope * -d_b0
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            power_tangent = event_saturation * 2.0 * alpha_value * alpha_tangent
            absorbed_value = tl.exp(power_value * shape_value)
            absorbed_tangent = absorbed_value * (
                power_tangent * shape_value + power_value * shape_tangent
            )
            saturating = is_rf & ~is_inversion
            if pools == 1:
                sat_b = _dual_scale(
                    absorbed_value, absorbed_tangent, bvr, bvi, btr, bti
                )
                bvr = tl.where(saturating, sat_b[0], bvr)
                bvi = tl.where(saturating, sat_b[1], bvi)
                btr = tl.where(saturating, sat_b[2], btr)
                bti = tl.where(saturating, sat_b[3], bti)
            else:
                sat_c = _dual_scale(
                    absorbed_value, absorbed_tangent, cvr, cvi, ctr, cti
                )
                cvr = tl.where(saturating, sat_c[0], cvr)
                cvi = tl.where(saturating, sat_c[1], cvi)
                ctr = tl.where(saturating, sat_c[2], ctr)
                cti = tl.where(saturating, sat_c[3], cti)
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        cos_tangent = -sin_value * alpha_tangent
        sin_tangent = cos_value * alpha_tangent
        p1r, p1i, p1tr, p1ti = _dual_polar(phi_value, phi_tangent)
        p2r, p2i, p2tr, p2ti = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_block(
            0.5 * (1.0 + cos_value),
            0.5 * cos_tangent,
            0.5 * (1.0 - cos_value),
            -0.5 * cos_tangent,
            sin_value,
            sin_tangent,
            cos_value,
            cos_tangent,
            p1r,
            p1i,
            p1tr,
            p1ti,
            p2r,
            p2i,
            p2tr,
            p2ti,
            p1r,
            -p1i,
            p1tr,
            -p1ti,
        )
        a0 = _dual_mul(t00[0], t00[1], t00[2], t00[3], pvr, pvi, ptr, pti)
        a1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], mvr, mvi, mtr, mti)
        a2 = _dual_mul(t02[0], t02[1], t02[2], t02[3], zvr, zvi, ztr, zti)
        b0_ = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], pvr, pvi, ptr, pti)
        b1_ = _dual_mul(t00[0], t00[1], t00[2], t00[3], mvr, mvi, mtr, mti)
        b2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], zvr, zvi, ztr, zti)
        c0 = _dual_mul(t20[0], t20[1], t20[2], t20[3], pvr, pvi, ptr, pti)
        c1 = _dual_mul(t21[0], t21[1], t21[2], t21[3], mvr, mvi, mtr, mti)
        c2 = _dual_mul(t22[0], t22[1], t22[2], t22[3], zvr, zvi, ztr, zti)

        turned_pvr = a0[0] + a1[0] + a2[0]
        turned_pvi = a0[1] + a1[1] + a2[1]
        turned_ptr = a0[2] + a1[2] + a2[2]
        turned_pti = a0[3] + a1[3] + a2[3]
        turned_mvr = b0_[0] + b1_[0] + b2[0]
        turned_mvi = b0_[1] + b1_[1] + b2[1]
        turned_mtr = b0_[2] + b1_[2] + b2[2]
        turned_mti = b0_[3] + b1_[3] + b2[3]
        turned_zvr = c0[0] + c1[0] + c2[0]
        turned_zvi = c0[1] + c1[1] + c2[1]
        turned_ztr = c0[2] + c1[2] + c2[2]
        turned_zti = c0[3] + c1[3] + c2[3]
        if profiled or dynamic:
            if dynamic:
                shaped_a, shaped_b = _dynamic_pair_dual_at(
                    pairs,
                    pair_direction,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                    phi_value,
                    phi_tangent,
                    directed,
                )
            else:
                shaped_a, shaped_b, _, _ = _profiled_pair_dual(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha_value,
                    alpha_tangent,
                    phi_value,
                    phi_tangent,
                    profile_bins,
                    profile_step,
                )
            (
                turned_pvr,
                turned_pvi,
                turned_mvr,
                turned_mvi,
                turned_zvr,
                turned_zvi,
                turned_ptr,
                turned_pti,
                turned_mtr,
                turned_mti,
                turned_ztr,
                turned_zti,
            ) = _rotate_spinor_dual(
                shaped_a[0],
                shaped_a[1],
                shaped_b[0],
                shaped_b[1],
                shaped_a[2],
                shaped_a[3],
                shaped_b[2],
                shaped_b[3],
                pvr,
                pvi,
                mvr,
                mvi,
                zvr,
                zvi,
                ptr,
                pti,
                mtr,
                mti,
                ztr,
                zti,
            )

        rotate = is_rf & ~is_inversion
        if pools == 2 or pools == 3:
            # The same pulse, the same rotation. A chemical shift moves where a
            # pool precesses, not what a pulse does to it.
            e0 = _dual_mul(t00[0], t00[1], t00[2], t00[3], bpvr, bpvi, bptr, bpti)
            e1_ = _dual_mul(t01[0], t01[1], t01[2], t01[3], bmvr, bmvi, bmtr, bmti)
            e2_ = _dual_mul(t02[0], t02[1], t02[2], t02[3], bvr, bvi, btr, bti)
            f0 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], bpvr, bpvi, bptr, bpti)
            f1 = _dual_mul(t00[0], t00[1], t00[2], t00[3], bmvr, bmvi, bmtr, bmti)
            f2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], bvr, bvi, btr, bti)
            h0 = _dual_mul(t20[0], t20[1], t20[2], t20[3], bpvr, bpvi, bptr, bpti)
            h1 = _dual_mul(t21[0], t21[1], t21[2], t21[3], bmvr, bmvi, bmtr, bmti)
            h2 = _dual_mul(t22[0], t22[1], t22[2], t22[3], bvr, bvi, btr, bti)
            spun_pvr = e0[0] + e1_[0] + e2_[0]
            spun_pvi = e0[1] + e1_[1] + e2_[1]
            spun_ptr = e0[2] + e1_[2] + e2_[2]
            spun_pti = e0[3] + e1_[3] + e2_[3]
            spun_mvr = f0[0] + f1[0] + f2[0]
            spun_mvi = f0[1] + f1[1] + f2[1]
            spun_mtr = f0[2] + f1[2] + f2[2]
            spun_mti = f0[3] + f1[3] + f2[3]
            spun_zvr = h0[0] + h1[0] + h2[0]
            spun_zvi = h0[1] + h1[1] + h2[1]
            spun_ztr = h0[2] + h1[2] + h2[2]
            spun_zti = h0[3] + h1[3] + h2[3]
            if profiled or dynamic:
                (
                    spun_pvr,
                    spun_pvi,
                    spun_mvr,
                    spun_mvi,
                    spun_zvr,
                    spun_zvi,
                    spun_ptr,
                    spun_pti,
                    spun_mtr,
                    spun_mti,
                    spun_ztr,
                    spun_zti,
                ) = _rotate_spinor_dual(
                    shaped_a[0],
                    shaped_a[1],
                    shaped_b[0],
                    shaped_b[1],
                    shaped_a[2],
                    shaped_a[3],
                    shaped_b[2],
                    shaped_b[3],
                    bpvr,
                    bpvi,
                    bmvr,
                    bmvi,
                    bvr,
                    bvi,
                    bptr,
                    bpti,
                    bmtr,
                    bmti,
                    btr,
                    bti,
                )
            bpvr = tl.where(rotate, spun_pvr, bpvr)
            bpvi = tl.where(rotate, spun_pvi, bpvi)
            bptr = tl.where(rotate, spun_ptr, bptr)
            bpti = tl.where(rotate, spun_pti, bpti)
            bmvr = tl.where(rotate, spun_mvr, bmvr)
            bmvi = tl.where(rotate, spun_mvi, bmvi)
            bmtr = tl.where(rotate, spun_mtr, bmtr)
            bmti = tl.where(rotate, spun_mti, bmti)
            bvr = tl.where(rotate, spun_zvr, bvr)
            bvi = tl.where(rotate, spun_zvi, bvi)
            btr = tl.where(rotate, spun_ztr, btr)
            bti = tl.where(rotate, spun_zti, bti)
        pvr = tl.where(rotate, turned_pvr, pvr)
        pvi = tl.where(rotate, turned_pvi, pvi)
        ptr = tl.where(rotate, turned_ptr, ptr)
        pti = tl.where(rotate, turned_pti, pti)
        mvr = tl.where(rotate, turned_mvr, mvr)
        mvi = tl.where(rotate, turned_mvi, mvi)
        mtr = tl.where(rotate, turned_mtr, mtr)
        mti = tl.where(rotate, turned_mti, mti)
        zvr = tl.where(rotate, turned_zvr, zvr)
        zvi = tl.where(rotate, turned_zvi, zvi)
        ztr = tl.where(rotate, turned_ztr, ztr)
        zti = tl.where(rotate, turned_zti, zti)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        svr, svi, wvr, wvi = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count)
        str_, sti, wtr, wti = _shift(ptr, pti, mtr, mti, state, state_mask, state_count)
        pvr = tl.where(do_shift, svr, pvr)
        pvi = tl.where(do_shift, svi, pvi)
        ptr = tl.where(do_shift, str_, ptr)
        pti = tl.where(do_shift, sti, pti)
        mvr = tl.where(do_shift, wvr, mvr)
        mvi = tl.where(do_shift, wvi, mvi)
        mtr = tl.where(do_shift, wtr, mtr)
        mti = tl.where(do_shift, wti, mti)
        spoil = (event_action & 8) != 0
        pvr = tl.where(spoil, 0.0, pvr)
        pvi = tl.where(spoil, 0.0, pvi)
        ptr = tl.where(spoil, 0.0, ptr)
        pti = tl.where(spoil, 0.0, pti)
        mvr = tl.where(spoil, 0.0, mvr)
        mvi = tl.where(spoil, 0.0, mvi)
        mtr = tl.where(spoil, 0.0, mtr)
        mti = tl.where(spoil, 0.0, mti)
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count
            )
            str_, sti, wtr, wti = _shift(
                bptr, bpti, bmtr, bmti, state, state_mask, state_count
            )
            bpvr = tl.where(spoil, 0.0, tl.where(do_shift, svr, bpvr))
            bpvi = tl.where(spoil, 0.0, tl.where(do_shift, svi, bpvi))
            bptr = tl.where(spoil, 0.0, tl.where(do_shift, str_, bptr))
            bpti = tl.where(spoil, 0.0, tl.where(do_shift, sti, bpti))
            bmvr = tl.where(spoil, 0.0, tl.where(do_shift, wvr, bmvr))
            bmvi = tl.where(spoil, 0.0, tl.where(do_shift, wvi, bmvi))
            bmtr = tl.where(spoil, 0.0, tl.where(do_shift, wtr, bmtr))
            bmti = tl.where(spoil, 0.0, tl.where(do_shift, wti, bmti))

    # ---- reverse ----
    pbvr = empty
    pbvi = empty
    pbtr = empty
    pbti = empty
    mbvr = empty
    mbvi = empty
    mbtr = empty
    mbti = empty
    zbvr = empty
    zbvi = empty
    zbtr = empty
    zbti = empty
    bbvr = empty
    bbvi = empty
    bbtr = empty
    bbti = empty
    ubvr = empty
    ubvi = empty
    ubtr = empty
    ubti = empty
    wbvr = empty
    wbvi = empty
    wbtr = empty
    wbti = empty
    cbvr = empty
    cbvi = empty
    cbtr = empty
    cbti = empty
    zero = tl.zeros((problems, 1), tl.float32)
    g_boundv = zero
    g_boundt = zero
    g_exchv = zero
    g_excht = zero
    g_t1bv = zero
    g_t1bt = zero
    g_t2bv = zero
    g_t2bt = zero
    g_shiftv = zero
    g_shiftt = zero
    g_semiv = zero
    g_semit = zero
    g_sexchv = zero
    g_sexcht = zero
    g_t1cv = zero
    g_t1ct = zero
    g_diffv = zero
    g_difft = zero
    g_flowv = zero
    g_flowt = zero
    g_washv = zero
    g_washt = zero
    g_t1v = zero
    g_t1t = zero
    g_t2v = zero
    g_t2t = zero
    g_m0v = zero
    g_m0t = zero
    g_b1v = zero
    g_b1t = zero
    g_b1pv = zero
    g_b1pt = zero
    g_b0v = zero
    g_b0t = zero
    g_invv = zero
    g_invt = zero

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        xpvr = tl.load(trajectory_vr + slot, mask=state_mask, other=0.0)
        xpvi = tl.load(trajectory_vi + slot, mask=state_mask, other=0.0)
        xptr = tl.load(trajectory_tr + slot, mask=state_mask, other=0.0)
        xpti = tl.load(trajectory_ti + slot, mask=state_mask, other=0.0)
        xmvr = tl.load(trajectory_vr + slot + minus_plane, mask=state_mask, other=0.0)
        xmvi = tl.load(trajectory_vi + slot + minus_plane, mask=state_mask, other=0.0)
        xmtr = tl.load(trajectory_tr + slot + minus_plane, mask=state_mask, other=0.0)
        xmti = tl.load(trajectory_ti + slot + minus_plane, mask=state_mask, other=0.0)
        xzvr = tl.load(trajectory_vr + slot + long_plane, mask=state_mask, other=0.0)
        xzvi = tl.load(trajectory_vi + slot + long_plane, mask=state_mask, other=0.0)
        xztr = tl.load(trajectory_tr + slot + long_plane, mask=state_mask, other=0.0)
        xzti = tl.load(trajectory_ti + slot + long_plane, mask=state_mask, other=0.0)
        xbvr = empty
        xbvi = empty
        xbtr = empty
        xbti = empty
        xbpvr = empty
        xbpvi = empty
        xbptr = empty
        xbpti = empty
        xbmvr = empty
        xbmvi = empty
        xbmtr = empty
        xbmti = empty
        xcvr = empty
        xcvi = empty
        xctr = empty
        xcti = empty
        if pools > 0:
            xbvr = tl.load(
                trajectory_vr + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbvi = tl.load(
                trajectory_vi + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbtr = tl.load(
                trajectory_tr + slot + bound_plane, mask=state_mask, other=0.0
            )
            xbti = tl.load(
                trajectory_ti + slot + bound_plane, mask=state_mask, other=0.0
            )
        if pools == 2 or pools == 3:
            xbpvr = tl.load(
                trajectory_vr + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbpvi = tl.load(
                trajectory_vi + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbptr = tl.load(
                trajectory_tr + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbpti = tl.load(
                trajectory_ti + slot + bplus_plane, mask=state_mask, other=0.0
            )
            xbmvr = tl.load(
                trajectory_vr + slot + bminus_plane, mask=state_mask, other=0.0
            )
            xbmvi = tl.load(
                trajectory_vi + slot + bminus_plane, mask=state_mask, other=0.0
            )
            xbmtr = tl.load(
                trajectory_tr + slot + bminus_plane, mask=state_mask, other=0.0
            )
            xbmti = tl.load(
                trajectory_ti + slot + bminus_plane, mask=state_mask, other=0.0
            )
        if pools == 3:
            xcvr = tl.load(
                trajectory_vr + slot + semisolid_plane, mask=state_mask, other=0.0
            )
            xcvi = tl.load(
                trajectory_vi + slot + semisolid_plane, mask=state_mask, other=0.0
            )
            xctr = tl.load(
                trajectory_tr + slot + semisolid_plane, mask=state_mask, other=0.0
            )
            xcti = tl.load(
                trajectory_ti + slot + semisolid_plane, mask=state_mask, other=0.0
            )

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        dt_tangent = _event_value(
            dot_duration, event_base, event, active_atom, single_train
        )
        wout_value = 1.0
        wout_tangent = 0.0
        if moving:
            wout_value, wout_tangent = _washout_jvp(
                atom_washout, d_washout, dt_value, dt_tangent
            )
        dry1_value = tl.exp(-r1_value * dt_value)
        dry1_tangent = -dry1_value * (r1_value * dt_tangent + r1_tangent * dt_value)
        dry2_value = tl.exp(-r2_value * dt_value)
        dry2_tangent = -dry2_value * (r2_value * dt_tangent + r2_tangent * dt_value)
        e1_value = dry1_value * wout_value
        e1_tangent = dry1_tangent * wout_value + dry1_value * wout_tangent
        e2_value = dry2_value * wout_value
        e2_tangent = dry2_tangent * wout_value + dry2_value * wout_tangent
        damp_z = 1.0
        damp_z_tangent = 0.0
        damp_t = 1.0
        damp_t_tangent = 0.0
        if diffusing:
            damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
                atom_damping, d_damping, dt_value, dt_tangent, order
            )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t
        turn_t = 0.0
        dturn_t = 0.0
        szr, szi, sztr, szti = 1.0, 0.0, 0.0, 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, dt_value, order)
            d_turn = d_flow * dt_value + atom_flow * dt_tangent
            dturn_z = -order * d_turn
            dturn_t = -(order + 0.5) * d_turn
            szr, szi, sztr, szti = _dual_polar(turn_z, dturn_z)
        qr, qi, qtr, qti = 1.0, 0.0, 0.0, 0.0
        if off_axis or moving:
            angle_value = -2.0 * 3.141592653589793 * (atom_b0 * dt_value) + turn_t
            angle_tangent = (
                -2.0 * 3.141592653589793 * (d_b0 * dt_value + atom_b0 * dt_tangent)
                + dturn_t
            )
            qr, qi, qtr, qti = _dual_polar(angle_value, angle_tangent)
        ovr, ovi, otr, oti = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti)
        lvr, lvi, ltr, lti = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti)

        # Replay the intra-event stages from the recorded entry state.
        carried = _dual_scale(damp_t, damp_t_tangent, qr, qi, qtr, qti)
        rbpvr = empty
        rbpvi = empty
        rbptr = empty
        rbpti = empty
        rbmvr = empty
        rbmvi = empty
        rbmtr = empty
        rbmti = empty
        a11 = (empty, empty, empty, empty)
        a12 = (empty, empty, empty, empty)
        a21 = (empty, empty, empty, empty)
        a22 = (empty, empty, empty, empty)
        if pools == 2 or pools == 3:
            across = _two_pool_transverse_step_jvp(
                r2_value,
                r2_tangent,
                r2b_value,
                r2b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                atom_free,
                d_free,
                atom_shift,
                d_shift,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
            )
            a11 = (across[0], across[1], across[8], across[9])
            a12 = (across[2], across[3], across[10], across[11])
            a21 = (across[4], across[5], across[12], across[13])
            a22 = (across[6], across[7], across[14], across[15])
            free_plus = (xpvr, xpvi, xptr, xpti)
            pool_plus = (xbpvr, xbpvi, xbptr, xbpti)
            free_minus = (xmvr, xmvi, xmtr, xmti)
            pool_minus = (xbmvr, xbmvi, xbmtr, xbmti)
            conjugated = _dual_conj(carried)
            rpvr, rpvi, rptr, rpti = _dual_product(
                _dual_add(_dual_product(a11, free_plus), _dual_product(a12, pool_plus)),
                carried,
            )
            rbpvr, rbpvi, rbptr, rbpti = _dual_product(
                _dual_add(_dual_product(a21, free_plus), _dual_product(a22, pool_plus)),
                carried,
            )
            rmvr, rmvi, rmtr, rmti = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(a11), free_minus),
                    _dual_product(_dual_conj(a12), pool_minus),
                ),
                conjugated,
            )
            rbmvr, rbmvi, rbmtr, rbmti = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(a21), free_minus),
                    _dual_product(_dual_conj(a22), pool_minus),
                ),
                conjugated,
            )
        else:
            rpvr, rpvi, rptr, rpti = _dual_mul(
                ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti
            )
            rmvr, rmvi, rmtr, rmti = _dual_mul(
                ovr, -ovi, otr, -oti, xmvr, xmvi, xmtr, xmti
            )
        rbvr = empty
        rbvi = empty
        rbtr = empty
        rbti = empty
        rcvr = empty
        rcvi = empty
        rctr = empty
        rcti = empty
        if pools == 3:
            if tabulated:
                # The walk back needs the operator and the direction
                # through it, which the row already holds -- and pooling
                # the cotangents took what the eigenvalues were formed
                # for, so nothing here reads them.
                pool_row = tl.load(
                    duration_row + event_base + event,
                    mask=active_atom,
                    other=0,
                )
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_w11,
                    d_w12,
                    d_w13,
                    d_w21,
                    d_w22,
                    d_w23,
                    d_w31,
                    d_w32,
                    d_w33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_from_table_jvp(
                    pool_table,
                    pool_row,
                    atom,
                    atom_count,
                    active_atom,
                    r1_value,
                    r1b_value,
                    r1c_value,
                    atom_exchange,
                    atom_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_tangent,
                    wout_value,
                    wout_tangent,
                )
            else:
                # The pieces and the bare operator are kept rather than
                # the step alone: the walk back pushes the cotangents
                # through them, and forming them once for the interval
                # is what keeps this kernel a size a compiler will take.
                (
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                ) = _three_pool_pieces_jvp(
                    r1_value,
                    r1_tangent,
                    r1b_value,
                    r1b_tangent,
                    r1c_value,
                    r1c_tangent,
                    atom_exchange,
                    d_exchange,
                    atom_semisolid_exchange,
                    d_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_value,
                    dt_tangent,
                    narrow,
                )
                (
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                ) = _three_pool_assemble_jvp(
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                    narrow,
                )
                (
                    w11,
                    w12,
                    w13,
                    w21,
                    w22,
                    w23,
                    w31,
                    w32,
                    w33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_w11,
                    d_w12,
                    d_w13,
                    d_w21,
                    d_w22,
                    d_w23,
                    d_w31,
                    d_w32,
                    d_w33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_weigh_jvp(
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    wout_value,
                    wout_tangent,
                    narrow,
                )
                # The operator is O(1) once formed, so the per-order loop below
                # takes it at the width the states are carried in.
                w11 = w11.to(tl.float32)
                w12 = w12.to(tl.float32)
                w13 = w13.to(tl.float32)
                w21 = w21.to(tl.float32)
                w22 = w22.to(tl.float32)
                w23 = w23.to(tl.float32)
                w31 = w31.to(tl.float32)
                w32 = w32.to(tl.float32)
                w33 = w33.to(tl.float32)
                grow_free = grow_free.to(tl.float32)
                grow_pool_b = grow_pool_b.to(tl.float32)
                grow_semisolid = grow_semisolid.to(tl.float32)
                d_w11 = d_w11.to(tl.float32)
                d_w12 = d_w12.to(tl.float32)
                d_w13 = d_w13.to(tl.float32)
                d_w21 = d_w21.to(tl.float32)
                d_w22 = d_w22.to(tl.float32)
                d_w23 = d_w23.to(tl.float32)
                d_w31 = d_w31.to(tl.float32)
                d_w32 = d_w32.to(tl.float32)
                d_w33 = d_w33.to(tl.float32)
                d_grow_free = d_grow_free.to(tl.float32)
                d_grow_pool_b = d_grow_pool_b.to(tl.float32)
                d_grow_semisolid = d_grow_semisolid.to(tl.float32)
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            mixed_free = _dual_add(
                _dual_add(
                    _dual_scale(w11, d_w11, xzvr, xzvi, xztr, xzti),
                    _dual_scale(w12, d_w12, xbvr, xbvi, xbtr, xbti),
                ),
                _dual_scale(w13, d_w13, xcvr, xcvi, xctr, xcti),
            )
            mixed_bound = _dual_add(
                _dual_add(
                    _dual_scale(w21, d_w21, xzvr, xzvi, xztr, xzti),
                    _dual_scale(w22, d_w22, xbvr, xbvi, xbtr, xbti),
                ),
                _dual_scale(w23, d_w23, xcvr, xcvi, xctr, xcti),
            )
            mixed_semisolid = _dual_add(
                _dual_add(
                    _dual_scale(w31, d_w31, xzvr, xzvi, xztr, xzti),
                    _dual_scale(w32, d_w32, xbvr, xbvi, xbtr, xbti),
                ),
                _dual_scale(w33, d_w33, xcvr, xcvi, xctr, xcti),
            )
            rzvr, rzvi, rztr, rzti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_free
            )
            rbvr, rbvi, rbtr, rbti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_bound
            )
            rcvr, rcvi, rctr, rcti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_semisolid
            )
            rzvr += tl.where(state == 0, grow_free, 0.0)
            rztr += tl.where(state == 0, d_grow_free, 0.0)
            rbvr += tl.where(state == 0, grow_pool_b, 0.0)
            rbtr += tl.where(state == 0, d_grow_pool_b, 0.0)
            rcvr += tl.where(state == 0, grow_semisolid, 0.0)
            rctr += tl.where(state == 0, d_grow_semisolid, 0.0)
        elif pools > 0:
            (
                pe11,
                pe12,
                pe21,
                pe22,
                prec_f,
                prec_b,
                de11,
                de12,
                de21,
                de22,
                drec_f,
                drec_b,
            ) = _two_pool_step_jvp(
                r1_value,
                r1_tangent,
                r1b_value,
                r1b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
            )
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti)
            free_part = _dual_scale(pe11, de11, xzvr, xzvi, xztr, xzti)
            cross_in = _dual_scale(pe12, de12, xbvr, xbvi, xbtr, xbti)
            cross_out = _dual_scale(pe21, de21, xzvr, xzvi, xztr, xzti)
            bound_part = _dual_scale(pe22, de22, xbvr, xbvi, xbtr, xbti)
            mixed_free = (
                free_part[0] + cross_in[0],
                free_part[1] + cross_in[1],
                free_part[2] + cross_in[2],
                free_part[3] + cross_in[3],
            )
            mixed_bound = (
                cross_out[0] + bound_part[0],
                cross_out[1] + bound_part[1],
                cross_out[2] + bound_part[2],
                cross_out[3] + bound_part[3],
            )
            rzvr, rzvi, rztr, rzti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_free
            )
            rbvr, rbvi, rbtr, rbti = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_bound
            )
            rzvr += tl.where(state == 0, prec_f, 0.0)
            rztr += tl.where(state == 0, drec_f, 0.0)
            rbvr += tl.where(state == 0, prec_b, 0.0)
            rbtr += tl.where(state == 0, drec_b, 0.0)
        else:
            rzvr, rzvi, rztr, rzti = _dual_mul(
                lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti
            )
            rzvr += tl.where(state == 0, recovery_value, 0.0)
            rztr += tl.where(state == 0, recovery_tangent, 0.0)

        pre_shift = (event_action & 1) != 0
        svr, svi, wvr, wvi = _shift(
            rpvr, rpvi, rmvr, rmvi, state, state_mask, state_count
        )
        str_, sti, wtr, wti = _shift(
            rptr, rpti, rmtr, rmti, state, state_mask, state_count
        )
        spvr = tl.where(pre_shift, svr, rpvr)
        spvi = tl.where(pre_shift, svi, rpvi)
        sptr = tl.where(pre_shift, str_, rptr)
        spti = tl.where(pre_shift, sti, rpti)
        smvr = tl.where(pre_shift, wvr, rmvr)
        smvi = tl.where(pre_shift, wvi, rmvi)
        smtr = tl.where(pre_shift, wtr, rmtr)
        smti = tl.where(pre_shift, wti, rmti)
        sbpvr = rbpvr
        sbpvi = rbpvi
        sbptr = rbptr
        sbpti = rbpti
        sbmvr = rbmvr
        sbmvi = rbmvi
        sbmtr = rbmtr
        sbmti = rbmti
        if pools == 2 or pools == 3:
            svr, svi, wvr, wvi = _shift(
                rbpvr, rbpvi, rbmvr, rbmvi, state, state_mask, state_count
            )
            str_, sti, wtr, wti = _shift(
                rbptr, rbpti, rbmtr, rbmti, state, state_mask, state_count
            )
            sbpvr = tl.where(pre_shift, svr, rbpvr)
            sbpvi = tl.where(pre_shift, svi, rbpvi)
            sbptr = tl.where(pre_shift, str_, rbptr)
            sbpti = tl.where(pre_shift, sti, rbpti)
            sbmvr = tl.where(pre_shift, wvr, rbmvr)
            sbmvi = tl.where(pre_shift, wvi, rbmvi)
            sbmtr = tl.where(pre_shift, wtr, rbmtr)
            sbmti = tl.where(pre_shift, wti, rbmti)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count
        )
        atr, ati, btr, bti = _shift_adjoint(
            pbtr, pbti, mbtr, mbti, state, state_mask, state_count
        )
        trailing = do_shift & ~spoil
        pbvr = tl.where(spoil, 0.0, tl.where(trailing, avr, pbvr))
        pbvi = tl.where(spoil, 0.0, tl.where(trailing, avi, pbvi))
        pbtr = tl.where(spoil, 0.0, tl.where(trailing, atr, pbtr))
        pbti = tl.where(spoil, 0.0, tl.where(trailing, ati, pbti))
        mbvr = tl.where(spoil, 0.0, tl.where(trailing, bvr, mbvr))
        mbvi = tl.where(spoil, 0.0, tl.where(trailing, bvi, mbvi))
        mbtr = tl.where(spoil, 0.0, tl.where(trailing, btr, mbtr))
        mbti = tl.where(spoil, 0.0, tl.where(trailing, bti, mbti))
        if pools == 2 or pools == 3:
            avr, avi, bvr, bvi = _shift_adjoint(
                ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count
            )
            atr, ati, btr, bti = _shift_adjoint(
                ubtr, ubti, wbtr, wbti, state, state_mask, state_count
            )
            ubvr = tl.where(spoil, 0.0, tl.where(trailing, avr, ubvr))
            ubvi = tl.where(spoil, 0.0, tl.where(trailing, avi, ubvi))
            ubtr = tl.where(spoil, 0.0, tl.where(trailing, atr, ubtr))
            ubti = tl.where(spoil, 0.0, tl.where(trailing, ati, ubti))
            wbvr = tl.where(spoil, 0.0, tl.where(trailing, bvr, wbvr))
            wbvi = tl.where(spoil, 0.0, tl.where(trailing, bvi, wbvi))
            wbtr = tl.where(spoil, 0.0, tl.where(trailing, btr, wbtr))
            wbti = tl.where(spoil, 0.0, tl.where(trailing, bti, wbti))

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_dot_flip = _event_value(
            dot_flip, event_base, event, active_atom, single_train
        )
        event_phase = _event_value(phase, event_base, event, active_atom, single_train)
        event_dot_phase = _event_value(
            dot_phase, event_base, event, active_atom, single_train
        )

        # ---- recorded sample ----
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed_mask = active_atom & record & (out >= 0)
        seed_real = tl.load(
            grad_output_real + problem * output_count + out, mask=seed_mask, other=0.0
        )
        seed_imag = tl.load(
            grad_output_imag + problem * output_count + out, mask=seed_mask, other=0.0
        )
        dvr, dvi, dtr, dti = _dual_polar(-event_phase, -event_dot_phase)
        # A coil sees the whole voxel, so what it records is the sum over pools.
        recorded = (spvr, spvi, sptr, spti)
        if pools == 2 or pools == 3:
            recorded = _dual_add(recorded, (sbpvr, sbpvi, sbptr, sbpti))
        # grad_m0 = Re(conj(seed) * recorded * demodulation)
        wr, wi, wtr_, wti_ = _dual_mul(*recorded, dvr, dvi, dtr, dti)
        m0_value, m0_tangent = _dual_real_conj_mul(
            seed_real,
            seed_imag,
            0.0 * seed_real,
            0.0 * seed_imag,
            wr,
            wi,
            wtr_,
            wti_,
        )
        g_m0v += tl.sum(tl.where(state == 0, m0_value, 0.0), axis=1)[:, None]
        g_m0t += tl.sum(tl.where(state == 0, m0_tangent, 0.0), axis=1)[:, None]
        # grad_phase = Re(conj(seed) * m0 * recorded * (-i) * demodulation)
        yr, yi, ytr, yti = _dual_scale(atom_m0, d_m0, *recorded)
        yr, yi, ytr, yti = _dual_times_i(yr, yi, ytr, yti)
        yr, yi, ytr, yti = -yr, -yi, -ytr, -yti
        yr, yi, ytr, yti = _dual_mul(yr, yi, ytr, yti, dvr, dvi, dtr, dti)
        phase_value, phase_tangent = _dual_real_conj_mul(
            seed_real, seed_imag, 0.0 * seed_real, 0.0 * seed_imag, yr, yi, ytr, yti
        )
        tl.atomic_add(
            grad_phase_value + event_base + event,
            tl.sum(tl.where(state == 0, phase_value, 0.0), axis=1)[:, None],
            mask=seed_mask,
        )
        tl.atomic_add(
            grad_phase_tangent + event_base + event,
            tl.sum(tl.where(state == 0, phase_tangent, 0.0), axis=1)[:, None],
            mask=seed_mask,
        )
        # fplus_bar[0] += conj(m0 * demodulation) * seed
        kr, ki, ktr, kti = _dual_scale(atom_m0, d_m0, dvr, dvi, dtr, dti)
        sr, si, stg_r, stg_i = _dual_mul(
            kr,
            -ki,
            ktr,
            -kti,
            seed_real,
            seed_imag,
            0.0 * seed_real,
            0.0 * seed_imag,
        )
        pbvr += tl.where(state == 0, sr, 0.0)
        pbvi += tl.where(state == 0, si, 0.0)
        pbtr += tl.where(state == 0, stg_r, 0.0)
        pbti += tl.where(state == 0, stg_i, 0.0)
        if pools == 2 or pools == 3:
            ubvr += tl.where(state == 0, sr, 0.0)
            ubvi += tl.where(state == 0, si, 0.0)
            ubtr += tl.where(state == 0, stg_r, 0.0)
            ubti += tl.where(state == 0, stg_i, 0.0)

        # ---- RF adjoint ----
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inv_value, inv_tangent = _dual_real_conj_mul(
            zbvr, zbvi, zbtr, zbti, -rzvr, -rzvi, -rztr, -rzti
        )
        g_invv += tl.sum(tl.where(invert, inv_value, 0.0), axis=1)[:, None]
        g_invt += tl.sum(tl.where(invert, inv_tangent, 0.0), axis=1)[:, None]
        ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, zbvr, zbvi, zbtr, zbti)
        zbvr = tl.where(invert, ivr, zbvr)
        zbvi = tl.where(invert, ivi, zbvi)
        zbtr = tl.where(invert, itr, zbtr)
        zbti = tl.where(invert, iti, zbti)
        if pools == 2 or pools == 3:
            pool_v, pool_t = _dual_real_conj_mul(
                bbvr, bbvi, bbtr, bbti, -rbvr, -rbvi, -rbtr, -rbti
            )
            g_invv += tl.sum(tl.where(invert, pool_v, 0.0), axis=1)[:, None]
            g_invt += tl.sum(tl.where(invert, pool_t, 0.0), axis=1)[:, None]
            ivr, ivi, itr, iti = _dual_scale(-atom_inv, -d_inv, bbvr, bbvi, bbtr, bbti)
            bbvr = tl.where(invert, ivr, bbvr)
            bbvi = tl.where(invert, ivi, bbvi)
            bbtr = tl.where(invert, itr, bbtr)
            bbti = tl.where(invert, iti, bbti)

        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = 1.0
            if transmit:
                atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            if off_axis:
                atom_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )
            d_b1 = tl.load(dot_b1 + row + atom, mask=active_atom, other=0.0)
            if off_axis:
                d_b1_phase = tl.load(
                    dot_b1_phase + row + atom, mask=active_atom, other=0.0
                )
        alpha_value = event_flip * atom_b1
        alpha_tangent = event_dot_flip * atom_b1 + event_flip * d_b1
        phi_value = event_phase + atom_b1_phase
        phi_tangent = event_dot_phase + d_b1_phase
        sat_alpha_v = zero
        sat_alpha_t = zero
        sat_b0_v = zero
        sat_b0_t = zero
        if broadened:
            # The pulse scales every order of the bound pool by one real
            # number, so its cotangent is a single sum over the states it
            # multiplied. The lineshape's own slope is differentiated too,
            # which is what the curvature the reader returns is for.
            offset_value = tl.load(rf_frequency + event) - atom_b0
            shape_value, shape_slope, shape_curve = _lineshape_at_curve(
                lineshape, offset_value, lineshape_bins, lineshape_step
            )
            shape_tangent = shape_slope * -d_b0
            slope_tangent = shape_curve * -d_b0
            event_saturation = tl.load(saturation + event)
            power_value = event_saturation * alpha_value * alpha_value
            power_tangent = event_saturation * 2.0 * alpha_value * alpha_tangent
            absorbed_value = tl.exp(power_value * shape_value)
            absorbed_tangent = absorbed_value * (
                power_tangent * shape_value + power_value * shape_tangent
            )
            if pools == 1:
                held_bar = (bbvr, bbvi, bbtr, bbti)
                held_state = (rbvr, rbvi, rbtr, rbti)
            else:
                held_bar = (cbvr, cbvi, cbtr, cbti)
                held_state = (rcvr, rcvi, rctr, rcti)
            per_state_v, per_state_t = _dual_real_conj_mul(*held_bar, *held_state)
            grad_absorbed_v = tl.sum(per_state_v, axis=1)[:, None]
            grad_absorbed_t = tl.sum(per_state_t, axis=1)[:, None]
            grad_exponent_v = grad_absorbed_v * absorbed_value
            grad_exponent_t = (
                grad_absorbed_t * absorbed_value + grad_absorbed_v * absorbed_tangent
            )
            twice = event_saturation * 2.0
            sat_alpha_v = grad_exponent_v * (twice * alpha_value * shape_value)
            sat_alpha_t = grad_exponent_t * (
                twice * alpha_value * shape_value
            ) + grad_exponent_v * twice * (
                alpha_tangent * shape_value + alpha_value * shape_tangent
            )
            # The lineshape is read at the pulse's offset from the voxel, so a
            # step in the voxel's own off-resonance moves the read the other
            # way.
            sat_b0_v = -grad_exponent_v * (power_value * shape_slope)
            sat_b0_t = -(
                grad_exponent_t * (power_value * shape_slope)
                + grad_exponent_v
                * (power_tangent * shape_slope + power_value * slope_tangent)
            )
            damped = _dual_scale(absorbed_value, absorbed_tangent, *held_bar)
            saturating = is_rf & ~is_inversion
            if pools == 1:
                bbvr = tl.where(saturating, damped[0], bbvr)
                bbvi = tl.where(saturating, damped[1], bbvi)
                bbtr = tl.where(saturating, damped[2], bbtr)
                bbti = tl.where(saturating, damped[3], bbti)
            else:
                cbvr = tl.where(saturating, damped[0], cbvr)
                cbvi = tl.where(saturating, damped[1], cbvi)
                cbtr = tl.where(saturating, damped[2], cbtr)
                cbti = tl.where(saturating, damped[3], cbti)
        cos_value = tl.cos(alpha_value)
        sin_value = tl.sin(alpha_value)
        cos_tangent = -sin_value * alpha_tangent
        sin_tangent = cos_value * alpha_tangent
        p1r, p1i, p1tr, p1ti = _dual_polar(phi_value, phi_tangent)
        p2r, p2i, p2tr, p2ti = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti)
        t00, t01, t02, t12, t20, t21, t22 = _rotation_block(
            0.5 * (1.0 + cos_value),
            0.5 * cos_tangent,
            0.5 * (1.0 - cos_value),
            -0.5 * cos_tangent,
            sin_value,
            sin_tangent,
            cos_value,
            cos_tangent,
            p1r,
            p1i,
            p1tr,
            p1ti,
            p2r,
            p2i,
            p2tr,
            p2ti,
            p1r,
            -p1i,
            p1tr,
            -p1ti,
        )
        # The flip angle reaches a shaped pulse's rotation through the slope
        # stored beside it, or not at all when the rotation is read per voxel,
        # so the operator's derivative in the flip is only built where the
        # pulse is a flip and a phase.
        alpha_v = empty
        alpha_t = empty
        phi_v = empty
        phi_t = empty
        alpha_b_v = empty
        alpha_b_t = empty
        phi_b_v = empty
        phi_b_t = empty
        if not profiled and not dynamic:
            d00, d01, d02, d12, d20, d21, d22 = _rotation_block(
                -0.5 * sin_value,
                -0.5 * sin_tangent,
                0.5 * sin_value,
                0.5 * sin_tangent,
                cos_value,
                cos_tangent,
                -sin_value,
                -sin_tangent,
                p1r,
                p1i,
                p1tr,
                p1ti,
                p2r,
                p2i,
                p2tr,
                p2ti,
                p1r,
                -p1i,
                p1tr,
                -p1ti,
            )

            # d/dalpha, contracted with the adjoint.
            row0 = _dual_mul(d00[0], d00[1], d00[2], d00[3], spvr, spvi, sptr, spti)
            add1 = _dual_mul(d01[0], d01[1], d01[2], d01[3], smvr, smvi, smtr, smti)
            add2 = _dual_mul(d02[0], d02[1], d02[2], d02[3], rzvr, rzvi, rztr, rzti)
            alpha_v, alpha_t = _dual_real_conj_mul(
                pbvr,
                pbvi,
                pbtr,
                pbti,
                row0[0] + add1[0] + add2[0],
                row0[1] + add1[1] + add2[1],
                row0[2] + add1[2] + add2[2],
                row0[3] + add1[3] + add2[3],
            )
            row0 = _dual_mul(d01[0], -d01[1], d01[2], -d01[3], spvr, spvi, sptr, spti)
            add1 = _dual_mul(d00[0], d00[1], d00[2], d00[3], smvr, smvi, smtr, smti)
            add2 = _dual_mul(d12[0], d12[1], d12[2], d12[3], rzvr, rzvi, rztr, rzti)
            part_v, part_t = _dual_real_conj_mul(
                mbvr,
                mbvi,
                mbtr,
                mbti,
                row0[0] + add1[0] + add2[0],
                row0[1] + add1[1] + add2[1],
                row0[2] + add1[2] + add2[2],
                row0[3] + add1[3] + add2[3],
            )
            alpha_v += part_v
            alpha_t += part_t
            row0 = _dual_mul(d20[0], d20[1], d20[2], d20[3], spvr, spvi, sptr, spti)
            add1 = _dual_mul(d21[0], d21[1], d21[2], d21[3], smvr, smvi, smtr, smti)
            add2 = _dual_mul(d22[0], d22[1], d22[2], d22[3], rzvr, rzvi, rztr, rzti)
            part_v, part_t = _dual_real_conj_mul(
                zbvr,
                zbvi,
                zbtr,
                zbti,
                row0[0] + add1[0] + add2[0],
                row0[1] + add1[1] + add2[1],
                row0[2] + add1[2] + add2[2],
                row0[3] + add1[3] + add2[3],
            )
            alpha_v += part_v
            alpha_t += part_t

            # d/dphi, where only the phase factors carry the dependence.
            u1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], smvr, smvi, smtr, smti)
            u2 = _dual_mul(t02[0], t02[1], t02[2], t02[3], rzvr, rzvi, rztr, rzti)
            ur, ui, utr, uti = _dual_times_i(
                2.0 * u1[0] + u2[0],
                2.0 * u1[1] + u2[1],
                2.0 * u1[2] + u2[2],
                2.0 * u1[3] + u2[3],
            )
            phi_v, phi_t = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, ur, ui, utr, uti)
            u1 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], spvr, spvi, sptr, spti)
            u2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], rzvr, rzvi, rztr, rzti)
            ur, ui, utr, uti = _dual_times_i(
                -2.0 * u1[0] - u2[0],
                -2.0 * u1[1] - u2[1],
                -2.0 * u1[2] - u2[2],
                -2.0 * u1[3] - u2[3],
            )
            part_v, part_t = _dual_real_conj_mul(
                mbvr, mbvi, mbtr, mbti, ur, ui, utr, uti
            )
            phi_v += part_v
            phi_t += part_t
            u1 = _dual_mul(t20[0], t20[1], t20[2], t20[3], spvr, spvi, sptr, spti)
            u2 = _dual_mul(t21[0], t21[1], t21[2], t21[3], smvr, smvi, smtr, smti)
            ur, ui, utr, uti = _dual_times_i(
                u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2], u2[3] - u1[3]
            )
            part_v, part_t = _dual_real_conj_mul(
                zbvr, zbvi, zbtr, zbti, ur, ui, utr, uti
            )
            phi_v += part_v
            phi_t += part_t

            # The same pulse turns the exchanging pool, so its cotangent adds to
            # the flip and phase the free pool already left.
            alpha_b_v = 0.0 * alpha_v
            alpha_b_t = 0.0 * alpha_v
            phi_b_v = 0.0 * alpha_v
            phi_b_t = 0.0 * alpha_v
            if pools == 2 or pools == 3:
                row0 = _dual_mul(
                    d00[0], d00[1], d00[2], d00[3], sbpvr, sbpvi, sbptr, sbpti
                )
                add1 = _dual_mul(
                    d01[0], d01[1], d01[2], d01[3], sbmvr, sbmvi, sbmtr, sbmti
                )
                add2 = _dual_mul(d02[0], d02[1], d02[2], d02[3], rbvr, rbvi, rbtr, rbti)
                alpha_b_v, alpha_b_t = _dual_real_conj_mul(
                    ubvr,
                    ubvi,
                    ubtr,
                    ubti,
                    row0[0] + add1[0] + add2[0],
                    row0[1] + add1[1] + add2[1],
                    row0[2] + add1[2] + add2[2],
                    row0[3] + add1[3] + add2[3],
                )
                row0 = _dual_mul(
                    d01[0], -d01[1], d01[2], -d01[3], sbpvr, sbpvi, sbptr, sbpti
                )
                add1 = _dual_mul(
                    d00[0], d00[1], d00[2], d00[3], sbmvr, sbmvi, sbmtr, sbmti
                )
                add2 = _dual_mul(d12[0], d12[1], d12[2], d12[3], rbvr, rbvi, rbtr, rbti)
                part_v, part_t = _dual_real_conj_mul(
                    wbvr,
                    wbvi,
                    wbtr,
                    wbti,
                    row0[0] + add1[0] + add2[0],
                    row0[1] + add1[1] + add2[1],
                    row0[2] + add1[2] + add2[2],
                    row0[3] + add1[3] + add2[3],
                )
                alpha_b_v += part_v
                alpha_b_t += part_t
                row0 = _dual_mul(
                    d20[0], d20[1], d20[2], d20[3], sbpvr, sbpvi, sbptr, sbpti
                )
                add1 = _dual_mul(
                    d21[0], d21[1], d21[2], d21[3], sbmvr, sbmvi, sbmtr, sbmti
                )
                add2 = _dual_mul(d22[0], d22[1], d22[2], d22[3], rbvr, rbvi, rbtr, rbti)
                part_v, part_t = _dual_real_conj_mul(
                    bbvr,
                    bbvi,
                    bbtr,
                    bbti,
                    row0[0] + add1[0] + add2[0],
                    row0[1] + add1[1] + add2[1],
                    row0[2] + add1[2] + add2[2],
                    row0[3] + add1[3] + add2[3],
                )
                alpha_b_v += part_v
                alpha_b_t += part_t

                u1 = _dual_mul(
                    t01[0], t01[1], t01[2], t01[3], sbmvr, sbmvi, sbmtr, sbmti
                )
                u2 = _dual_mul(t02[0], t02[1], t02[2], t02[3], rbvr, rbvi, rbtr, rbti)
                ur, ui, utr, uti = _dual_times_i(
                    2.0 * u1[0] + u2[0],
                    2.0 * u1[1] + u2[1],
                    2.0 * u1[2] + u2[2],
                    2.0 * u1[3] + u2[3],
                )
                phi_b_v, phi_b_t = _dual_real_conj_mul(
                    ubvr, ubvi, ubtr, ubti, ur, ui, utr, uti
                )
                u1 = _dual_mul(
                    t01[0], -t01[1], t01[2], -t01[3], sbpvr, sbpvi, sbptr, sbpti
                )
                u2 = _dual_mul(t12[0], t12[1], t12[2], t12[3], rbvr, rbvi, rbtr, rbti)
                ur, ui, utr, uti = _dual_times_i(
                    -2.0 * u1[0] - u2[0],
                    -2.0 * u1[1] - u2[1],
                    -2.0 * u1[2] - u2[2],
                    -2.0 * u1[3] - u2[3],
                )
                part_v, part_t = _dual_real_conj_mul(
                    wbvr, wbvi, wbtr, wbti, ur, ui, utr, uti
                )
                phi_b_v += part_v
                phi_b_t += part_t
                u1 = _dual_mul(
                    t20[0], t20[1], t20[2], t20[3], sbpvr, sbpvi, sbptr, sbpti
                )
                u2 = _dual_mul(
                    t21[0], t21[1], t21[2], t21[3], sbmvr, sbmvi, sbmtr, sbmti
                )
                ur, ui, utr, uti = _dual_times_i(
                    u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2], u2[3] - u1[3]
                )
                part_v, part_t = _dual_real_conj_mul(
                    bbvr, bbvi, bbtr, bbti, ur, ui, utr, uti
                )
                phi_b_v += part_v
                phi_b_t += part_t

        if profiled or dynamic:
            shaped_slope_a = (empty, empty, empty, empty)
            shaped_slope_b = (empty, empty, empty, empty)
            if dynamic:
                shaped_a, shaped_b = _dynamic_pair_dual_at(
                    pairs,
                    pair_direction,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                    phi_value,
                    phi_tangent,
                    directed,
                )
            else:
                shaped_a, shaped_b, shaped_slope_a, shaped_slope_b = (
                    _profiled_pair_dual(
                        profile,
                        _table_row(profile_index, event, location, locations),
                        alpha_value,
                        alpha_tangent,
                        phi_value,
                        phi_tangent,
                        profile_bins,
                        profile_step,
                    )
                )
            grad_a, grad_b, shaped_pb, shaped_mb, shaped_zb = _spinor_adjoint_dual(
                shaped_a,
                shaped_b,
                (spvr, spvi, sptr, spti),
                (smvr, smvi, smtr, smti),
                (rzvr, rzvi, rztr, rzti),
                (pbvr, pbvi, pbtr, pbti),
                (mbvr, mbvi, mbtr, mbti),
                (zbvr, zbvi, zbtr, zbti),
            )
            if dynamic:
                # The flip is inside the pair rather than read against it, so
                # it has no gradient here: the cotangent goes out on the
                # rotation and whatever integrated it carries the rest. ``b``
                # was turned by the phase after the pair came out, so the
                # cotangent turns back the other way.
                alpha_v = empty
                alpha_t = empty
                back = _dual_product(
                    grad_b, _dual_conj(_dual_polar(-phi_value, -phi_tangent))
                )
                _store_pair_cotangent(
                    grad_pair_value,
                    grad_pair_tangent,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    is_rf & ~is_inversion,
                    active_atom,
                    state_mask,
                    grad_a,
                    back,
                )
            else:
                alpha_v, alpha_t = _dual_real_conj_mul(
                    grad_a[0],
                    grad_a[1],
                    grad_a[2],
                    grad_a[3],
                    shaped_slope_a[0],
                    shaped_slope_a[1],
                    shaped_slope_a[2],
                    shaped_slope_a[3],
                )
                part_v, part_t = _dual_real_conj_mul(
                    grad_b[0],
                    grad_b[1],
                    grad_b[2],
                    grad_b[3],
                    shaped_slope_b[0],
                    shaped_slope_b[1],
                    shaped_slope_b[2],
                    shaped_slope_b[3],
                )
                alpha_v += part_v
                alpha_t += part_t
            # d(b e^{-i phi})/dphi is -i times it, and nothing else moves.
            turn_r, turn_i, turn_tr, turn_ti = _dual_times_i(
                shaped_b[0], shaped_b[1], shaped_b[2], shaped_b[3]
            )
            phi_v, phi_t = _dual_real_conj_mul(
                grad_b[0],
                grad_b[1],
                grad_b[2],
                grad_b[3],
                -turn_r,
                -turn_i,
                -turn_tr,
                -turn_ti,
            )
            if pools == 2 or pools == 3:
                pool_a, pool_b_pair, shaped_ub, shaped_wb, shaped_bb = (
                    _spinor_adjoint_dual(
                        shaped_a,
                        shaped_b,
                        (sbpvr, sbpvi, sbptr, sbpti),
                        (sbmvr, sbmvi, sbmtr, sbmti),
                        (rbvr, rbvi, rbtr, rbti),
                        (ubvr, ubvi, ubtr, ubti),
                        (wbvr, wbvi, wbtr, wbti),
                        (bbvr, bbvi, bbtr, bbti),
                    )
                )
                if dynamic:
                    # The same pulse turned this pool, so its cotangent lands
                    # on the same row.
                    pool_back = _dual_product(
                        pool_b_pair,
                        _dual_conj(_dual_polar(-phi_value, -phi_tangent)),
                    )
                    _store_pair_cotangent(
                        grad_pair_value,
                        grad_pair_tangent,
                        pair_index,
                        event_base,
                        event,
                        atom,
                        atom_count,
                        is_rf & ~is_inversion,
                        active_atom,
                        state_mask,
                        pool_a,
                        pool_back,
                    )
                else:
                    alpha_b_v, alpha_b_t = _dual_real_conj_mul(
                        pool_a[0],
                        pool_a[1],
                        pool_a[2],
                        pool_a[3],
                        shaped_slope_a[0],
                        shaped_slope_a[1],
                        shaped_slope_a[2],
                        shaped_slope_a[3],
                    )
                    part_v, part_t = _dual_real_conj_mul(
                        pool_b_pair[0],
                        pool_b_pair[1],
                        pool_b_pair[2],
                        pool_b_pair[3],
                        shaped_slope_b[0],
                        shaped_slope_b[1],
                        shaped_slope_b[2],
                        shaped_slope_b[3],
                    )
                    alpha_b_v += part_v
                    alpha_b_t += part_t
                phi_b_v, phi_b_t = _dual_real_conj_mul(
                    pool_b_pair[0],
                    pool_b_pair[1],
                    pool_b_pair[2],
                    pool_b_pair[3],
                    -turn_r,
                    -turn_i,
                    -turn_tr,
                    -turn_ti,
                )
        alpha_v += alpha_b_v
        alpha_t += alpha_b_t
        phi_v += phi_b_v
        phi_t += phi_b_t

        rotate = is_rf & ~is_inversion
        grad_alpha_v = tl.sum(tl.where(rotate, alpha_v, 0.0), axis=1)[:, None]
        grad_alpha_t = tl.sum(tl.where(rotate, alpha_t, 0.0), axis=1)[:, None]
        grad_phi_v = tl.sum(tl.where(rotate, phi_v, 0.0), axis=1)[:, None]
        grad_phi_t = tl.sum(tl.where(rotate, phi_t, 0.0), axis=1)[:, None]
        if pools == 1 or pools == 3:
            turning = tl.where(rotate, 1.0, 0.0)
            grad_alpha_v += sat_alpha_v * turning
            grad_alpha_t += sat_alpha_t * turning
            g_b0v += sat_b0_v * turning
            g_b0t += sat_b0_t * turning

        # Conjugate transpose of the rotation.
        n0 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], pbvr, pbvi, pbtr, pbti)
        n1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], mbvr, mbvi, mbtr, mbti)
        n2 = _dual_mul(t20[0], -t20[1], t20[2], -t20[3], zbvr, zbvi, zbtr, zbti)
        q0 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], pbvr, pbvi, pbtr, pbti)
        q1 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], mbvr, mbvi, mbtr, mbti)
        q2 = _dual_mul(t21[0], -t21[1], t21[2], -t21[3], zbvr, zbvi, zbtr, zbti)
        w0 = _dual_mul(t02[0], -t02[1], t02[2], -t02[3], pbvr, pbvi, pbtr, pbti)
        w1 = _dual_mul(t12[0], -t12[1], t12[2], -t12[3], mbvr, mbvi, mbtr, mbti)
        w2 = _dual_mul(t22[0], -t22[1], t22[2], -t22[3], zbvr, zbvi, zbtr, zbti)

        back_pb = (
            n0[0] + n1[0] + n2[0],
            n0[1] + n1[1] + n2[1],
            n0[2] + n1[2] + n2[2],
            n0[3] + n1[3] + n2[3],
        )
        back_mb = (
            q0[0] + q1[0] + q2[0],
            q0[1] + q1[1] + q2[1],
            q0[2] + q1[2] + q2[2],
            q0[3] + q1[3] + q2[3],
        )
        back_zb = (
            w0[0] + w1[0] + w2[0],
            w0[1] + w1[1] + w2[1],
            w0[2] + w1[2] + w2[2],
            w0[3] + w1[3] + w2[3],
        )
        if pools == 2 or pools == 3:
            n0 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], ubvr, ubvi, ubtr, ubti)
            n1 = _dual_mul(t01[0], t01[1], t01[2], t01[3], wbvr, wbvi, wbtr, wbti)
            n2 = _dual_mul(t20[0], -t20[1], t20[2], -t20[3], bbvr, bbvi, bbtr, bbti)
            q0 = _dual_mul(t01[0], -t01[1], t01[2], -t01[3], ubvr, ubvi, ubtr, ubti)
            q1 = _dual_mul(t00[0], -t00[1], t00[2], -t00[3], wbvr, wbvi, wbtr, wbti)
            q2 = _dual_mul(t21[0], -t21[1], t21[2], -t21[3], bbvr, bbvi, bbtr, bbti)
            w0 = _dual_mul(t02[0], -t02[1], t02[2], -t02[3], ubvr, ubvi, ubtr, ubti)
            w1 = _dual_mul(t12[0], -t12[1], t12[2], -t12[3], wbvr, wbvi, wbtr, wbti)
            w2 = _dual_mul(t22[0], -t22[1], t22[2], -t22[3], bbvr, bbvi, bbtr, bbti)
            back_ub = (
                n0[0] + n1[0] + n2[0],
                n0[1] + n1[1] + n2[1],
                n0[2] + n1[2] + n2[2],
                n0[3] + n1[3] + n2[3],
            )
            back_wb = (
                q0[0] + q1[0] + q2[0],
                q0[1] + q1[1] + q2[1],
                q0[2] + q1[2] + q2[2],
                q0[3] + q1[3] + q2[3],
            )
            back_bb = (
                w0[0] + w1[0] + w2[0],
                w0[1] + w1[1] + w2[1],
                w0[2] + w1[2] + w2[2],
                w0[3] + w1[3] + w2[3],
            )
            if profiled or dynamic:
                back_ub = shaped_ub
                back_wb = shaped_wb
                back_bb = shaped_bb
            ubvr = tl.where(rotate, back_ub[0], ubvr)
            ubvi = tl.where(rotate, back_ub[1], ubvi)
            ubtr = tl.where(rotate, back_ub[2], ubtr)
            ubti = tl.where(rotate, back_ub[3], ubti)
            wbvr = tl.where(rotate, back_wb[0], wbvr)
            wbvi = tl.where(rotate, back_wb[1], wbvi)
            wbtr = tl.where(rotate, back_wb[2], wbtr)
            wbti = tl.where(rotate, back_wb[3], wbti)
            bbvr = tl.where(rotate, back_bb[0], bbvr)
            bbvi = tl.where(rotate, back_bb[1], bbvi)
            bbtr = tl.where(rotate, back_bb[2], bbtr)
            bbti = tl.where(rotate, back_bb[3], bbti)
        if profiled or dynamic:
            back_pb = shaped_pb
            back_mb = shaped_mb
            back_zb = shaped_zb

        pbvr = tl.where(rotate, back_pb[0], pbvr)
        pbvi = tl.where(rotate, back_pb[1], pbvi)
        pbtr = tl.where(rotate, back_pb[2], pbtr)
        pbti = tl.where(rotate, back_pb[3], pbti)
        mbvr = tl.where(rotate, back_mb[0], mbvr)
        mbvi = tl.where(rotate, back_mb[1], mbvi)
        mbtr = tl.where(rotate, back_mb[2], mbtr)
        mbti = tl.where(rotate, back_mb[3], mbti)
        zbvr = tl.where(rotate, back_zb[0], zbvr)
        zbvi = tl.where(rotate, back_zb[1], zbvi)
        zbtr = tl.where(rotate, back_zb[2], zbtr)
        zbti = tl.where(rotate, back_zb[3], zbti)

        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip_value + event_base + event,
            grad_alpha_v * atom_b1,
            mask=writes_flip,
        )
        tl.atomic_add(
            grad_flip_tangent + event_base + event,
            grad_alpha_t * atom_b1 + grad_alpha_v * d_b1,
            mask=writes_flip,
        )
        tl.atomic_add(
            grad_phase_value + event_base + event, grad_phi_v, mask=writes_flip
        )
        tl.atomic_add(
            grad_phase_tangent + event_base + event, grad_phi_t, mask=writes_flip
        )
        # A pulse's transmit gradient belongs to the shim it drives, so with
        # several it lands in that shim's row here rather than in a register
        # summed over the whole train. ``row`` is the offset of the row the
        # replay above read.
        if shimmed:
            tl.atomic_add(
                grad_tissue_value + _B1_ROW * atom_count + row + atom,
                grad_alpha_v * event_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_tangent + _B1_ROW * atom_count + row + atom,
                grad_alpha_t * event_flip + grad_alpha_v * event_dot_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_value
                + (_B1_PHASE_ROW + shim_rows - 1) * atom_count
                + row
                + atom,
                grad_phi_v,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_tangent
                + (_B1_PHASE_ROW + shim_rows - 1) * atom_count
                + row
                + atom,
                grad_phi_t,
                mask=writes_flip,
            )
        else:
            g_b1v += grad_alpha_v * event_flip
            g_b1t += grad_alpha_t * event_flip + grad_alpha_v * event_dot_flip
            g_b1pv += grad_phi_v
            g_b1pt += grad_phi_t

        avr, avi, bvr, bvi = _shift_adjoint(
            pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count
        )
        atr, ati, btr, bti = _shift_adjoint(
            pbtr, pbti, mbtr, mbti, state, state_mask, state_count
        )
        pbvr = tl.where(pre_shift, avr, pbvr)
        pbvi = tl.where(pre_shift, avi, pbvi)
        pbtr = tl.where(pre_shift, atr, pbtr)
        pbti = tl.where(pre_shift, ati, pbti)
        mbvr = tl.where(pre_shift, bvr, mbvr)
        mbvi = tl.where(pre_shift, bvi, mbvi)
        mbtr = tl.where(pre_shift, btr, mbtr)
        mbti = tl.where(pre_shift, bti, mbti)
        if pools == 2 or pools == 3:
            avr, avi, bvr, bvi = _shift_adjoint(
                ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count
            )
            atr, ati, btr, bti = _shift_adjoint(
                ubtr, ubti, wbtr, wbti, state, state_mask, state_count
            )
            ubvr = tl.where(pre_shift, avr, ubvr)
            ubvi = tl.where(pre_shift, avi, ubvi)
            ubtr = tl.where(pre_shift, atr, ubtr)
            ubti = tl.where(pre_shift, ati, ubti)
            wbvr = tl.where(pre_shift, bvr, wbvr)
            wbvi = tl.where(pre_shift, bvi, wbvi)
            wbtr = tl.where(pre_shift, btr, wbtr)
            wbti = tl.where(pre_shift, bti, wbti)

        # ---- relaxation and off-resonance adjoint ----
        grad_e2_v = zero
        grad_e2_t = zero
        attenuation_v = zero
        attenuation_t = zero
        two_pool_dt_v = zero
        two_pool_dt_t = zero
        # The damping is homogeneous of degree one in every transverse state it
        # acts on, so its gradient times the damping itself is the cotangent
        # taken against the states the interval leaves. With one pool that is
        # the same thing as the relaxation factor's own gradient, scaled.
        if pools == 2 or pools == 3:
            plus_side = _dual_add(
                _dual_product(
                    _dual_conj((pbvr, pbvi, pbtr, pbti)),
                    (rpvr, rpvi, rptr, rpti),
                ),
                _dual_product(
                    _dual_conj((ubvr, ubvi, ubtr, ubti)),
                    (rbpvr, rbpvi, rbptr, rbpti),
                ),
            )
            minus_side = _dual_add(
                _dual_product(
                    _dual_conj((mbvr, mbvi, mbtr, mbti)),
                    (rmvr, rmvi, rmtr, rmti),
                ),
                _dual_product(
                    _dual_conj((wbvr, wbvi, wbtr, wbti)),
                    (rbmvr, rbmvi, rbmtr, rbmti),
                ),
            )
            damped = _dual_add(plus_side, minus_side)
            wound = _dual_times_i(*_dual_subtract(plus_side, minus_side))
            cot2_v = damped[0]
            cot2_t = damped[2]
            per_angle_v = wound[0]
            per_angle_t = wound[2]
        else:
            pq = _dual_mul(qr, qi, qtr, qti, xpvr, xpvi, xptr, xpti)
            mq = _dual_mul(qr, -qi, qtr, -qti, xmvr, xmvi, xmtr, xmti)
            e2_v, e2_t = _dual_real_conj_mul(
                pbvr, pbvi, pbtr, pbti, pq[0], pq[1], pq[2], pq[3]
            )
            part_v, part_t = _dual_real_conj_mul(
                mbvr, mbvi, mbtr, mbti, mq[0], mq[1], mq[2], mq[3]
            )
            bare_cot_v = e2_v + part_v
            bare_cot_t = e2_t + part_t
            grad_e2_v = tl.sum(bare_cot_v * damp_t, axis=1)[:, None]
            grad_e2_t = tl.sum(
                bare_cot_v * damp_t_tangent + bare_cot_t * damp_t, axis=1
            )[:, None]

            per_angle_v = empty
            per_angle_t = empty
            if off_axis or moving:
                po = _dual_mul(ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti)
                po = _dual_times_i(po[0], po[1], po[2], po[3])
                mo = _dual_mul(ovr, -ovi, otr, -oti, xmvr, xmvi, xmtr, xmti)
                mo = _dual_times_i(mo[0], mo[1], mo[2], mo[3])
                angle_v, angle_t = _dual_real_conj_mul(
                    pbvr, pbvi, pbtr, pbti, po[0], po[1], po[2], po[3]
                )
                part_v, part_t = _dual_real_conj_mul(
                    mbvr, mbvi, mbtr, mbti, mo[0], mo[1], mo[2], mo[3]
                )
                per_angle_v = angle_v - part_v
                per_angle_t = angle_t - part_t
            cot2_v = bare_cot_v * bare2_value * damp_t
            cot2_t = (
                bare_cot_t * bare2_value * damp_t
                + bare_cot_v * bare2_tangent * damp_t
                + bare_cot_v * bare2_value * damp_t_tangent
            )
        # A turn of the transverse states and the off-resonance angle are the
        # same derivative; only the weight each order carries differs.
        grad_angle_v = zero
        grad_angle_t = zero
        if off_axis or moving:
            grad_angle_v = tl.sum(per_angle_v, axis=1)[:, None]
            grad_angle_t = tl.sum(per_angle_t, axis=1)[:, None]

        grad_e1_v = zero
        grad_e1_t = zero
        if pools == 3:
            # The nine entries of the mixing operator and the three recoveries,
            # summed over the orders that share them, then pushed back through
            # the closed form once for the whole interval and in double.
            free_bar = (zbvr, zbvi, zbtr, zbti)
            bound_bar = (bbvr, bbvi, bbtr, bbti)
            semi_bar = (cbvr, cbvi, cbtr, cbti)
            spun_free = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], xzvr, xzvi, xztr, xzti
            )
            spun_bound = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], xbvr, xbvi, xbtr, xbti
            )
            spun_semi = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], xcvr, xcvi, xctr, xcti
            )
            e11_v, e11_t = _dual_real_conj_mul(*free_bar, *spun_free)
            e12_v, e12_t = _dual_real_conj_mul(*free_bar, *spun_bound)
            e13_v, e13_t = _dual_real_conj_mul(*free_bar, *spun_semi)
            e21_v, e21_t = _dual_real_conj_mul(*bound_bar, *spun_free)
            e22_v, e22_t = _dual_real_conj_mul(*bound_bar, *spun_bound)
            e23_v, e23_t = _dual_real_conj_mul(*bound_bar, *spun_semi)
            e31_v, e31_t = _dual_real_conj_mul(*semi_bar, *spun_free)
            e32_v, e32_t = _dual_real_conj_mul(*semi_bar, *spun_bound)
            e33_v, e33_t = _dual_real_conj_mul(*semi_bar, *spun_semi)
            if tabulated:
                # Every gradient but the interval's own and the
                # attenuation's is linear in these cotangents, so the
                # events sharing a length pool them here and pay the
                # closed form once each after the walk back. The tangent
                # gradient carries a third term in the event's own
                # interval direction, which pools as the value cotangents
                # weighted by it.
                bar11_v = tl.sum(e11_v, axis=1)[:, None]
                bar12_v = tl.sum(e12_v, axis=1)[:, None]
                bar13_v = tl.sum(e13_v, axis=1)[:, None]
                bar21_v = tl.sum(e21_v, axis=1)[:, None]
                bar22_v = tl.sum(e22_v, axis=1)[:, None]
                bar23_v = tl.sum(e23_v, axis=1)[:, None]
                bar31_v = tl.sum(e31_v, axis=1)[:, None]
                bar32_v = tl.sum(e32_v, axis=1)[:, None]
                bar33_v = tl.sum(e33_v, axis=1)[:, None]
                barfree_v = tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
                barpool_v = tl.sum(tl.where(state == 0, bbvr, 0.0), axis=1)[:, None]
                barbound_v = tl.sum(tl.where(state == 0, cbvr, 0.0), axis=1)[:, None]
                bar11_t = tl.sum(e11_t, axis=1)[:, None]
                bar12_t = tl.sum(e12_t, axis=1)[:, None]
                bar13_t = tl.sum(e13_t, axis=1)[:, None]
                bar21_t = tl.sum(e21_t, axis=1)[:, None]
                bar22_t = tl.sum(e22_t, axis=1)[:, None]
                bar23_t = tl.sum(e23_t, axis=1)[:, None]
                bar31_t = tl.sum(e31_t, axis=1)[:, None]
                bar32_t = tl.sum(e32_t, axis=1)[:, None]
                bar33_t = tl.sum(e33_t, axis=1)[:, None]
                barfree_t = tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]
                barpool_t = tl.sum(tl.where(state == 0, bbtr, 0.0), axis=1)[:, None]
                barbound_t = tl.sum(tl.where(state == 0, cbtr, 0.0), axis=1)[:, None]
                held = pool_bars + (local * row_count + pool_row) * 36
                tl.store(
                    held + 0,
                    tl.load(held + 0, mask=active_atom, other=0.0) + bar11_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 1,
                    tl.load(held + 1, mask=active_atom, other=0.0) + bar12_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 2,
                    tl.load(held + 2, mask=active_atom, other=0.0) + bar13_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 3,
                    tl.load(held + 3, mask=active_atom, other=0.0) + bar21_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 4,
                    tl.load(held + 4, mask=active_atom, other=0.0) + bar22_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 5,
                    tl.load(held + 5, mask=active_atom, other=0.0) + bar23_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 6,
                    tl.load(held + 6, mask=active_atom, other=0.0) + bar31_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 7,
                    tl.load(held + 7, mask=active_atom, other=0.0) + bar32_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 8,
                    tl.load(held + 8, mask=active_atom, other=0.0) + bar33_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 9,
                    tl.load(held + 9, mask=active_atom, other=0.0) + barfree_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 10,
                    tl.load(held + 10, mask=active_atom, other=0.0) + barpool_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 11,
                    tl.load(held + 11, mask=active_atom, other=0.0) + barbound_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 12,
                    tl.load(held + 12, mask=active_atom, other=0.0) + bar11_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 13,
                    tl.load(held + 13, mask=active_atom, other=0.0) + bar12_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 14,
                    tl.load(held + 14, mask=active_atom, other=0.0) + bar13_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 15,
                    tl.load(held + 15, mask=active_atom, other=0.0) + bar21_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 16,
                    tl.load(held + 16, mask=active_atom, other=0.0) + bar22_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 17,
                    tl.load(held + 17, mask=active_atom, other=0.0) + bar23_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 18,
                    tl.load(held + 18, mask=active_atom, other=0.0) + bar31_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 19,
                    tl.load(held + 19, mask=active_atom, other=0.0) + bar32_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 20,
                    tl.load(held + 20, mask=active_atom, other=0.0) + bar33_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 21,
                    tl.load(held + 21, mask=active_atom, other=0.0) + barfree_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 22,
                    tl.load(held + 22, mask=active_atom, other=0.0) + barpool_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 23,
                    tl.load(held + 23, mask=active_atom, other=0.0) + barbound_t,
                    mask=active_atom,
                )
                tl.store(
                    held + 24,
                    tl.load(held + 24, mask=active_atom, other=0.0)
                    + dt_tangent * bar11_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 25,
                    tl.load(held + 25, mask=active_atom, other=0.0)
                    + dt_tangent * bar12_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 26,
                    tl.load(held + 26, mask=active_atom, other=0.0)
                    + dt_tangent * bar13_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 27,
                    tl.load(held + 27, mask=active_atom, other=0.0)
                    + dt_tangent * bar21_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 28,
                    tl.load(held + 28, mask=active_atom, other=0.0)
                    + dt_tangent * bar22_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 29,
                    tl.load(held + 29, mask=active_atom, other=0.0)
                    + dt_tangent * bar23_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 30,
                    tl.load(held + 30, mask=active_atom, other=0.0)
                    + dt_tangent * bar31_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 31,
                    tl.load(held + 31, mask=active_atom, other=0.0)
                    + dt_tangent * bar32_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 32,
                    tl.load(held + 32, mask=active_atom, other=0.0)
                    + dt_tangent * bar33_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 33,
                    tl.load(held + 33, mask=active_atom, other=0.0)
                    + dt_tangent * barfree_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 34,
                    tl.load(held + 34, mask=active_atom, other=0.0)
                    + dt_tangent * barpool_v,
                    mask=active_atom,
                )
                tl.store(
                    held + 35,
                    tl.load(held + 35, mask=active_atom, other=0.0)
                    + dt_tangent * barbound_v,
                    mask=active_atom,
                )
                (
                    back_dt_v,
                    back_att_v,
                    back_dt_t,
                    back_att_t,
                ) = _three_pool_interval_adjoint_jvp(
                    pool_table,
                    pool_row,
                    atom,
                    atom_count,
                    active_atom,
                    r1_value,
                    r1_tangent,
                    r1b_value,
                    r1b_tangent,
                    r1c_value,
                    r1c_tangent,
                    atom_exchange,
                    d_exchange,
                    atom_semisolid_exchange,
                    d_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_tangent,
                    wout_value,
                    wout_tangent,
                    bar11_v,
                    bar12_v,
                    bar13_v,
                    bar21_v,
                    bar22_v,
                    bar23_v,
                    bar31_v,
                    bar32_v,
                    bar33_v,
                    barfree_v,
                    barpool_v,
                    barbound_v,
                    bar11_t,
                    bar12_t,
                    bar13_t,
                    bar21_t,
                    bar22_t,
                    bar23_t,
                    bar31_t,
                    bar32_t,
                    bar33_t,
                    barfree_t,
                    barpool_t,
                    barbound_t,
                )
                attenuation_v = back_att_v
                attenuation_t = back_att_t
                two_pool_dt_v = back_dt_v
                two_pool_dt_t = back_dt_t
            else:
                (
                    back_r1_v,
                    back_r1b_v,
                    back_r1c_v,
                    back_exch_v,
                    back_sexch_v,
                    back_bound_v,
                    back_semi_v,
                    back_dt_v,
                    back_att_v,
                    back_r1_t,
                    back_r1b_t,
                    back_r1c_t,
                    back_exch_t,
                    back_sexch_t,
                    back_bound_t,
                    back_semi_t,
                    back_dt_t,
                    back_att_t,
                ) = _three_pool_step_adjoint_jvp(
                    r1_value,
                    r1_tangent,
                    r1b_value,
                    r1b_tangent,
                    r1c_value,
                    r1c_tangent,
                    atom_exchange,
                    d_exchange,
                    atom_semisolid_exchange,
                    d_semisolid_exchange,
                    atom_bound,
                    d_boundf,
                    atom_semisolid,
                    d_semisolidf,
                    dt_value,
                    dt_tangent,
                    wout_value,
                    wout_tangent,
                    tl.sum(e11_v, axis=1)[:, None],
                    tl.sum(e11_t, axis=1)[:, None],
                    tl.sum(e12_v, axis=1)[:, None],
                    tl.sum(e12_t, axis=1)[:, None],
                    tl.sum(e13_v, axis=1)[:, None],
                    tl.sum(e13_t, axis=1)[:, None],
                    tl.sum(e21_v, axis=1)[:, None],
                    tl.sum(e21_t, axis=1)[:, None],
                    tl.sum(e22_v, axis=1)[:, None],
                    tl.sum(e22_t, axis=1)[:, None],
                    tl.sum(e23_v, axis=1)[:, None],
                    tl.sum(e23_t, axis=1)[:, None],
                    tl.sum(e31_v, axis=1)[:, None],
                    tl.sum(e31_t, axis=1)[:, None],
                    tl.sum(e32_v, axis=1)[:, None],
                    tl.sum(e32_t, axis=1)[:, None],
                    tl.sum(e33_v, axis=1)[:, None],
                    tl.sum(e33_t, axis=1)[:, None],
                    tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None],
                    tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None],
                    tl.sum(tl.where(state == 0, bbvr, 0.0), axis=1)[:, None],
                    tl.sum(tl.where(state == 0, bbtr, 0.0), axis=1)[:, None],
                    tl.sum(tl.where(state == 0, cbvr, 0.0), axis=1)[:, None],
                    tl.sum(tl.where(state == 0, cbtr, 0.0), axis=1)[:, None],
                    three_free,
                    three_d_free,
                    three_pool_b,
                    three_d_pool_b,
                    three_pool_c,
                    three_d_pool_c,
                    three_a00,
                    three_d_a00,
                    three_a01,
                    three_d_a01,
                    three_a02,
                    three_d_a02,
                    three_a10,
                    three_d_a10,
                    three_a11,
                    three_d_a11,
                    three_a20,
                    three_d_a20,
                    three_a22,
                    three_d_a22,
                    three_s00,
                    three_d_s00,
                    three_s11,
                    three_d_s11,
                    three_s22,
                    three_d_s22,
                    three_minors,
                    three_d_minors,
                    three_sum_flat,
                    three_sum_linear,
                    three_sum_square,
                    three_d_sum_flat,
                    three_d_sum_linear,
                    three_d_sum_square,
                    three_lift,
                    three_d_lift,
                    three_low,
                    three_middle,
                    three_d_low,
                    three_d_middle,
                    three_leading,
                    three_d_leading,
                    three_first,
                    three_d_first,
                    three_second,
                    three_d_second,
                    three_determinant,
                    three_d_determinant,
                    three_high,
                    three_d_high,
                    three_radius,
                    three_d_radius,
                    three_cube,
                    three_raw,
                    three_d_raw,
                    three_argument,
                    three_inside_limit,
                    three_angle,
                    three_d_angle,
                    three_centre,
                    three_d_centre,
                    three_trailing,
                    three_d_trailing,
                    three_guarded,
                    three_d_guarded,
                    three_q00,
                    three_d_q00,
                    three_q01,
                    three_d_q01,
                    three_q02,
                    three_d_q02,
                    three_q10,
                    three_d_q10,
                    three_q11,
                    three_d_q11,
                    three_q12,
                    three_d_q12,
                    three_q20,
                    three_d_q20,
                    three_q21,
                    three_d_q21,
                    three_q22,
                    three_d_q22,
                    three_def_00,
                    three_dif_00,
                    three_def_01,
                    three_dif_01,
                    three_def_02,
                    three_dif_02,
                    three_def_10,
                    three_dif_10,
                    three_def_11,
                    three_dif_11,
                    three_def_12,
                    three_dif_12,
                    three_def_20,
                    three_dif_20,
                    three_def_21,
                    three_dif_21,
                    three_def_22,
                    three_dif_22,
                    narrow,
                )
                slope1_v = -1000.0 / (atom_t1 * atom_t1)
                slope1_t = 2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
                slope1b_v = -1000.0 / (atom_t1b * atom_t1b)
                slope1b_t = 2000.0 * d_t1b / (atom_t1b * atom_t1b * atom_t1b)
                slope1c_v = -1000.0 / (held_semisolid * held_semisolid)
                slope1c_t = (
                    2000.0
                    * d_semisolid_t1
                    / (held_semisolid * held_semisolid * held_semisolid)
                )
                g_t1v += back_r1_v * slope1_v
                g_t1t += back_r1_t * slope1_v + back_r1_v * slope1_t
                g_t1bv += back_r1b_v * slope1b_v
                g_t1bt += back_r1b_t * slope1b_v + back_r1b_v * slope1b_t
                g_t1cv += back_r1c_v * slope1c_v
                g_t1ct += back_r1c_t * slope1c_v + back_r1c_v * slope1c_t
                g_exchv += back_exch_v
                g_excht += back_exch_t
                g_sexchv += back_sexch_v
                g_sexcht += back_sexch_t
                g_boundv += back_bound_v
                g_boundt += back_bound_t
                g_semiv += back_semi_v
                g_semit += back_semi_t
                attenuation_v = back_att_v
                attenuation_t = back_att_t
                two_pool_dt_v = back_dt_v
                two_pool_dt_t = back_dt_t
            turned_free = _dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_free)
            turned_bound = _dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_bound)
            turned_semi = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], *mixed_semisolid
            )
            damp_pair_v, damp_pair_t = _dual_real_conj_mul(*free_bar, *turned_free)
            other_v, other_t = _dual_real_conj_mul(*bound_bar, *turned_bound)
            stuck_v, stuck_t = _dual_real_conj_mul(*semi_bar, *turned_semi)
            long_damp_v = damp_pair_v + other_v + stuck_v
            long_damp_t = damp_pair_t + other_t + stuck_t
            zangle_v, zangle_t = _dual_real_conj_mul(
                *free_bar, *_dual_times_i(*turned_free)
            )
            part_v, part_t = _dual_real_conj_mul(
                *bound_bar, *_dual_times_i(*turned_bound)
            )
            zangle_v += part_v
            zangle_t += part_t
            part_v, part_t = _dual_real_conj_mul(
                *semi_bar, *_dual_times_i(*turned_semi)
            )
            zangle_v += part_v
            zangle_t += part_t
            col_free = _dual_add(
                _dual_add(
                    _dual_back(w11, d_w11, *spin, *free_bar),
                    _dual_back(w21, d_w21, *spin, *bound_bar),
                ),
                _dual_back(w31, d_w31, *spin, *semi_bar),
            )
            col_bound = _dual_add(
                _dual_add(
                    _dual_back(w12, d_w12, *spin, *free_bar),
                    _dual_back(w22, d_w22, *spin, *bound_bar),
                ),
                _dual_back(w32, d_w32, *spin, *semi_bar),
            )
            col_semi = _dual_add(
                _dual_add(
                    _dual_back(w13, d_w13, *spin, *free_bar),
                    _dual_back(w23, d_w23, *spin, *bound_bar),
                ),
                _dual_back(w33, d_w33, *spin, *semi_bar),
            )
            zbvr, zbvi, zbtr, zbti = col_free
            bbvr, bbvi, bbtr, bbti = col_bound
            cbvr, cbvi, cbtr, cbti = col_semi
        elif pools > 0:
            # The four entries of the exchange operator and the two recoveries,
            # summed over the orders that share them, then pushed back through
            # the closed form once for the whole interval.
            free_bar = (zbvr, zbvi, zbtr, zbti)
            bound_bar = (bbvr, bbvi, bbtr, bbti)
            spun_free = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], xzvr, xzvi, xztr, xzti
            )
            spun_bound = _dual_mul(
                spin[0], spin[1], spin[2], spin[3], xbvr, xbvi, xbtr, xbti
            )
            e11_v, e11_t = _dual_real_conj_mul(*free_bar, *spun_free)
            e12_v, e12_t = _dual_real_conj_mul(*free_bar, *spun_bound)
            e21_v, e21_t = _dual_real_conj_mul(*bound_bar, *spun_free)
            e22_v, e22_t = _dual_real_conj_mul(*bound_bar, *spun_bound)
            bar_e11_v = tl.sum(e11_v, axis=1)[:, None]
            bar_e11_t = tl.sum(e11_t, axis=1)[:, None]
            bar_e12_v = tl.sum(e12_v, axis=1)[:, None]
            bar_e12_t = tl.sum(e12_t, axis=1)[:, None]
            bar_e21_v = tl.sum(e21_v, axis=1)[:, None]
            bar_e21_t = tl.sum(e21_t, axis=1)[:, None]
            bar_e22_v = tl.sum(e22_v, axis=1)[:, None]
            bar_e22_t = tl.sum(e22_t, axis=1)[:, None]
            rec_f_v = tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            rec_f_t = tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]
            rec_b_v = tl.sum(tl.where(state == 0, bbvr, 0.0), axis=1)[:, None]
            rec_b_t = tl.sum(tl.where(state == 0, bbtr, 0.0), axis=1)[:, None]
            (
                back_r1_v,
                back_r1b_v,
                back_exch_v,
                back_bound_v,
                back_dt_v,
                back_att_v,
                back_r1_t,
                back_r1b_t,
                back_exch_t,
                back_bound_t,
                back_dt_t,
                back_att_t,
            ) = _two_pool_step_adjoint_jvp(
                r1_value,
                r1_tangent,
                r1b_value,
                r1b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
                bar_e11_v,
                bar_e11_t,
                bar_e12_v,
                bar_e12_t,
                bar_e21_v,
                bar_e21_t,
                bar_e22_v,
                bar_e22_t,
                rec_f_v,
                rec_f_t,
                rec_b_v,
                rec_b_t,
            )
            # r1 = 1000/t1, so a rate gradient reaches the time through the
            # square of it.
            slope1_v = -1000.0 / (atom_t1 * atom_t1)
            slope1_t = 2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
            slope1b_v = -1000.0 / (atom_t1b * atom_t1b)
            slope1b_t = 2000.0 * d_t1b / (atom_t1b * atom_t1b * atom_t1b)
            g_t1v += back_r1_v * slope1_v
            g_t1t += back_r1_t * slope1_v + back_r1_v * slope1_t
            g_t1bv += back_r1b_v * slope1b_v
            g_t1bt += back_r1b_t * slope1b_v + back_r1b_v * slope1b_t
            g_exchv += back_exch_v
            g_excht += back_exch_t
            g_boundv += back_bound_v
            g_boundt += back_bound_t
            attenuation_v = back_att_v
            attenuation_t = back_att_t
            two_pool_dt_v = back_dt_v
            two_pool_dt_t = back_dt_t
            # Both pools take the same per-order damping and turn, so each
            # collects the cotangent of the mixture that reached it.
            damp_pair_v, damp_pair_t = _dual_real_conj_mul(
                *free_bar, *_dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_free)
            )
            other_v, other_t = _dual_real_conj_mul(
                *bound_bar, *_dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_bound)
            )
            long_damp_v = damp_pair_v + other_v
            long_damp_t = damp_pair_t + other_t
            spun_mix_free = _dual_times_i(
                *_dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_free)
            )
            spun_mix_bound = _dual_times_i(
                *_dual_mul(spin[0], spin[1], spin[2], spin[3], *mixed_bound)
            )
            zangle_v, zangle_t = _dual_real_conj_mul(*free_bar, *spun_mix_free)
            part_v, part_t = _dual_real_conj_mul(*bound_bar, *spun_mix_bound)
            zangle_v += part_v
            zangle_t += part_t
            back_z = _dual_mul(
                pe11 * spin[0],
                -(pe11 * spin[1]),
                de11 * spin[0] + pe11 * spin[2],
                -(de11 * spin[1] + pe11 * spin[3]),
                *free_bar,
            )
            cross_z = _dual_mul(
                pe21 * spin[0],
                -(pe21 * spin[1]),
                de21 * spin[0] + pe21 * spin[2],
                -(de21 * spin[1] + pe21 * spin[3]),
                *bound_bar,
            )
            back_b = _dual_mul(
                pe12 * spin[0],
                -(pe12 * spin[1]),
                de12 * spin[0] + pe12 * spin[2],
                -(de12 * spin[1] + pe12 * spin[3]),
                *free_bar,
            )
            cross_b = _dual_mul(
                pe22 * spin[0],
                -(pe22 * spin[1]),
                de22 * spin[0] + pe22 * spin[2],
                -(de22 * spin[1] + pe22 * spin[3]),
                *bound_bar,
            )
            next_zbvr = back_z[0] + cross_z[0]
            next_zbvi = back_z[1] + cross_z[1]
            next_zbtr = back_z[2] + cross_z[2]
            next_zbti = back_z[3] + cross_z[3]
            bbvr = back_b[0] + cross_b[0]
            bbvi = back_b[1] + cross_b[1]
            bbtr = back_b[2] + cross_b[2]
            bbti = back_b[3] + cross_b[3]
            zbvr = next_zbvr
            zbvi = next_zbvi
            zbtr = next_zbtr
            zbti = next_zbti
        else:
            spun = _dual_mul(szr, szi, sztr, szti, xzvr, xzvi, xztr, xzti)
            e1_v, e1_t = _dual_real_conj_mul(
                zbvr, zbvi, zbtr, zbti, spun[0], spun[1], spun[2], spun[3]
            )
            grad_e1_v = tl.sum(e1_v * damp_z, axis=1)[:, None]
            grad_e1_t = tl.sum(e1_v * damp_z_tangent + e1_t * damp_z, axis=1)[:, None]
            grad_e1_v -= tl.sum(tl.where(state == 0, zbvr, 0.0), axis=1)[:, None]
            grad_e1_t -= tl.sum(tl.where(state == 0, zbtr, 0.0), axis=1)[:, None]
            long_damp_v = e1_v * bare1_value * damp_z
            long_damp_t = (
                e1_t * bare1_value * damp_z
                + e1_v * bare1_tangent * damp_z
                + e1_v * bare1_value * damp_z_tangent
            )
            # The longitudinal states turn too, and by a whole order rather
            # than the transverse half-order more.
            zo = _dual_mul(lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti)
            zo = _dual_times_i(zo[0], zo[1], zo[2], zo[3])
            zangle_v, zangle_t = _dual_real_conj_mul(
                zbvr, zbvi, zbtr, zbti, zo[0], zo[1], zo[2], zo[3]
            )
            zbvr, zbvi, zbtr, zbti = _dual_mul(
                lvr, -lvi, ltr, -lti, zbvr, zbvi, zbtr, zbti
            )

        next_pb = (pbvr, pbvi, pbtr, pbti)
        next_mb = (mbvr, mbvi, mbtr, mbti)
        next_ub = (ubvr, ubvi, ubtr, ubti)
        next_wb = (wbvr, wbvi, wbtr, wbti)
        if pools == 2 or pools == 3:
            # The four entries of the transverse operator, summed over the
            # orders that share them, then pushed back through the closed form
            # once for the whole interval. ``F-`` follows the conjugate of the
            # operator, so its cotangent lands on the entry itself rather than
            # on the conjugate of it.
            ap = (pbvr, pbvi, pbtr, pbti)
            am = (mbvr, mbvi, mbtr, mbti)
            aub = (ubvr, ubvi, ubtr, ubti)
            awb = (wbvr, wbvi, wbtr, wbti)
            fp = (xpvr, xpvi, xptr, xpti)
            fm = (xmvr, xmvi, xmtr, xmti)
            bp = (xbpvr, xbpvi, xbptr, xbpti)
            bm = (xbmvr, xbmvi, xbmtr, xbmti)
            term11 = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(ap), fp),
                    _dual_product(am, _dual_conj(fm)),
                ),
                carried,
            )
            term12 = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(ap), bp),
                    _dual_product(am, _dual_conj(bm)),
                ),
                carried,
            )
            term21 = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(aub), fp),
                    _dual_product(awb, _dual_conj(fm)),
                ),
                carried,
            )
            term22 = _dual_product(
                _dual_add(
                    _dual_product(_dual_conj(aub), bp),
                    _dual_product(awb, _dual_conj(bm)),
                ),
                carried,
            )
            bar11 = (
                tl.sum(term11[0], axis=1)[:, None],
                tl.sum(term11[1], axis=1)[:, None],
                tl.sum(term11[2], axis=1)[:, None],
                tl.sum(term11[3], axis=1)[:, None],
            )
            bar12 = (
                tl.sum(term12[0], axis=1)[:, None],
                tl.sum(term12[1], axis=1)[:, None],
                tl.sum(term12[2], axis=1)[:, None],
                tl.sum(term12[3], axis=1)[:, None],
            )
            bar21 = (
                tl.sum(term21[0], axis=1)[:, None],
                tl.sum(term21[1], axis=1)[:, None],
                tl.sum(term21[2], axis=1)[:, None],
                tl.sum(term21[3], axis=1)[:, None],
            )
            bar22 = (
                tl.sum(term22[0], axis=1)[:, None],
                tl.sum(term22[1], axis=1)[:, None],
                tl.sum(term22[2], axis=1)[:, None],
                tl.sum(term22[3], axis=1)[:, None],
            )
            (
                back_r2_v,
                back_r2_t,
                back_r2b_v,
                back_r2b_t,
                back_xexch_v,
                back_xexch_t,
                back_xbound_v,
                back_xbound_t,
                back_xfree_v,
                back_xfree_t,
                back_shift_v,
                back_shift_t,
                back_xdt_v,
                back_xdt_t,
                back_xatt_v,
                back_xatt_t,
            ) = _two_pool_transverse_adjoint_jvp(
                r2_value,
                r2_tangent,
                r2b_value,
                r2b_tangent,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_boundf,
                atom_free,
                d_free,
                atom_shift,
                d_shift,
                dt_value,
                dt_tangent,
                wout_value,
                wout_tangent,
                bar11,
                bar12,
                bar21,
                bar22,
            )
            slope2_v = -1000.0 / (atom_t2 * atom_t2)
            slope2_t = 2000.0 * d_t2 / (atom_t2 * atom_t2 * atom_t2)
            slope2b_v = -1000.0 / (atom_t2b * atom_t2b)
            slope2b_t = 2000.0 * d_t2b / (atom_t2b * atom_t2b * atom_t2b)
            g_t2v += back_r2_v * slope2_v
            g_t2t += back_r2_t * slope2_v + back_r2_v * slope2_t
            g_t2bv += back_r2b_v * slope2b_v
            g_t2bt += back_r2b_t * slope2b_v + back_r2b_v * slope2b_t
            g_exchv += back_xexch_v
            g_excht += back_xexch_t
            # The free water is what both second pools leave, so a cotangent
            # on it reaches each of their fractions turned over.
            g_boundv += back_xbound_v - back_xfree_v
            g_boundt += back_xbound_t - back_xfree_t
            if pools == 3:
                g_semiv -= back_xfree_v
                g_semit -= back_xfree_t
            g_shiftv += back_shift_v
            g_shiftt += back_shift_t
            attenuation_v += back_xatt_v
            attenuation_t += back_xatt_t
            two_pool_dt_v += back_xdt_v
            two_pool_dt_t += back_xdt_t
            step11 = _dual_product(a11, carried)
            step12 = _dual_product(a12, carried)
            step21 = _dual_product(a21, carried)
            step22 = _dual_product(a22, carried)
            next_pb = _dual_add(
                _dual_product(_dual_conj(step11), ap),
                _dual_product(_dual_conj(step21), aub),
            )
            next_ub = _dual_add(
                _dual_product(_dual_conj(step12), ap),
                _dual_product(_dual_conj(step22), aub),
            )
            next_mb = _dual_add(_dual_product(step11, am), _dual_product(step21, awb))
            next_wb = _dual_add(_dual_product(step12, am), _dual_product(step22, awb))

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum rather than one scalar. Order zero carries no
        # longitudinal weight, which keeps recovery out of this.
        spread_v = zero
        spread_t = zero
        if diffusing:
            weighted_v = long_damp_v * longitudinal_weight + cot2_v * transverse_weight
            weighted_t = long_damp_t * longitudinal_weight + cot2_t * transverse_weight
            spread_v = tl.sum(weighted_v, axis=1)[:, None]
            spread_t = tl.sum(weighted_t, axis=1)[:, None]
            g_diffv += -spread_v * dt_value
            g_difft += -(spread_v * dt_tangent + spread_t * dt_value)

        wound_v = zero
        wound_t = zero
        if moving:
            wound_v = tl.sum(per_angle_v * (order + 0.5) + zangle_v * order, axis=1)[
                :, None
            ]
            wound_t = tl.sum(per_angle_t * (order + 0.5) + zangle_t * order, axis=1)[
                :, None
            ]
            g_flowv += -wound_v * dt_value
            g_flowt += -(wound_v * dt_tangent + wound_t * dt_value)

        # Washout scales both relaxation factors, so its gradient is the one
        # they already carry, taken against the factors before that scaling.
        # Past the clamp the interval has replaced the voxel outright and
        # nothing further depends on the rate.
        wash_v = zero
        wash_t = zero
        if moving:
            live = (atom_washout * dt_value < 1.0).to(tl.float32)
            wash_v = -live * (
                grad_e1_v * dry1_value + grad_e2_v * dry2_value + attenuation_v
            )
            wash_t = -live * (
                grad_e1_v * dry1_tangent
                + grad_e1_t * dry1_value
                + grad_e2_v * dry2_tangent
                + grad_e2_t * dry2_value
                + attenuation_t
            )
            g_washv += wash_v * dt_value
            g_washt += wash_v * dt_tangent + wash_t * dt_value

        if pools == 2 or pools == 3:
            pbvr, pbvi, pbtr, pbti = next_pb
            mbvr, mbvi, mbtr, mbti = next_mb
            ubvr, ubvi, ubtr, ubti = next_ub
            wbvr, wbvi, wbtr, wbti = next_wb
        else:
            pbvr, pbvi, pbtr, pbti = _dual_mul(
                ovr, -ovi, otr, -oti, pbvr, pbvi, pbtr, pbti
            )
            mbvr, mbvi, mbtr, mbti = _dual_mul(
                ovr, ovi, otr, oti, mbvr, mbvi, mbtr, mbti
            )

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse1_tangent = -2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        inverse2_tangent = -2000.0 * d_t2 / (atom_t2 * atom_t2 * atom_t2)
        scale1_value = bare1_value * dt_value * inverse1_value
        scale1_tangent = bare1_tangent * dt_value * inverse1_value
        scale1_tangent += bare1_value * dt_tangent * inverse1_value
        scale1_tangent += bare1_value * dt_value * inverse1_tangent
        scale2_value = bare2_value * dt_value * inverse2_value
        scale2_tangent = bare2_tangent * dt_value * inverse2_value
        scale2_tangent += bare2_value * dt_tangent * inverse2_value
        scale2_tangent += bare2_value * dt_value * inverse2_tangent
        g_t1v += grad_e1_v * scale1_value
        g_t1t += grad_e1_v * scale1_tangent + grad_e1_t * scale1_value
        g_t2v += grad_e2_v * scale2_value
        g_t2t += grad_e2_v * scale2_tangent + grad_e2_t * scale2_value

        turn = -2.0 * 3.141592653589793
        g_b0v += grad_angle_v * (turn * dt_value)
        g_b0t += grad_angle_v * (turn * dt_tangent) + grad_angle_t * (turn * dt_value)

        decay1_value = r1_value * bare1_value
        decay1_tangent = r1_value * bare1_tangent + r1_tangent * bare1_value
        decay2_value = r2_value * bare2_value
        decay2_tangent = r2_value * bare2_tangent + r2_tangent * bare2_value
        duration_v = -grad_e1_v * decay1_value - grad_e2_v * decay2_value
        duration_v += grad_angle_v * (turn * atom_b0) + two_pool_dt_v
        duration_t = -(grad_e1_v * decay1_tangent + grad_e1_t * decay1_value)
        duration_t -= grad_e2_v * decay2_tangent + grad_e2_t * decay2_value
        duration_t += grad_angle_v * (turn * d_b0) + grad_angle_t * (turn * atom_b0)
        duration_t += two_pool_dt_t
        duration_v += -spread_v * atom_damping - wound_v * atom_flow
        duration_t += -(spread_v * d_damping + spread_t * atom_damping)
        duration_t += -(wound_v * d_flow + wound_t * atom_flow)
        duration_v += wash_v * atom_washout
        duration_t += wash_v * d_washout + wash_t * atom_washout
        tl.atomic_add(
            grad_duration_value + event_base + event, duration_v, mask=active_atom
        )
        tl.atomic_add(
            grad_duration_tangent + event_base + event, duration_t, mask=active_atom
        )

    if pools == 3 and tabulated:
        # One closed form per distinct length rather than one per event,
        # run twice. The walk back pooled the cotangents the eigenvalues
        # are pushed through and the closed form is linear in them, so the
        # pieces of the sum are the sum of the pieces. A gradient's own
        # direction depends on the interval as well, and a row is shared
        # by events whose interval directions differ -- so the second pass
        # takes that dependence alone, driven by the cotangents the walk
        # back weighted by each event's direction and read at a unit one.
        for row in range(0, row_count):
            held = pool_bars + (local * row_count + row) * 36
            row_dt = tl.load(pool_durations + row) + zero
            nil = 0.0 * row_dt
            unit = 1.0 + nil
            one_att = unit
            att_rate = nil
            att_span = nil
            if moving:
                one_att, att_rate = _washout_jvp(atom_washout, d_washout, row_dt, nil)
                _held_att, att_span = _washout_jvp(atom_washout, nil, row_dt, unit)
            (
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
            ) = _three_pool_pieces_jvp(
                r1_value,
                r1_tangent,
                r1b_value,
                r1b_tangent,
                r1c_value,
                r1c_tangent,
                atom_exchange,
                d_exchange,
                atom_semisolid_exchange,
                d_semisolid_exchange,
                atom_bound,
                d_boundf,
                atom_semisolid,
                d_semisolidf,
                row_dt,
                nil,
                narrow,
            )
            (
                three_def_00,
                three_dif_00,
                three_def_01,
                three_dif_01,
                three_def_02,
                three_dif_02,
                three_def_10,
                three_dif_10,
                three_def_11,
                three_dif_11,
                three_def_12,
                three_dif_12,
                three_def_20,
                three_dif_20,
                three_def_21,
                three_dif_21,
                three_def_22,
                three_dif_22,
            ) = _three_pool_assemble_jvp(
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
                narrow,
            )
            (
                back_r1_v,
                back_r1b_v,
                back_r1c_v,
                back_exch_v,
                back_sexch_v,
                back_bound_v,
                back_semi_v,
                _row_dt_v,
                _row_att_v,
                back_r1_t,
                back_r1b_t,
                back_r1c_t,
                back_exch_t,
                back_sexch_t,
                back_bound_t,
                back_semi_t,
                _row_dt_t,
                _row_att_t,
            ) = _three_pool_step_adjoint_jvp(
                r1_value,
                r1_tangent,
                r1b_value,
                r1b_tangent,
                r1c_value,
                r1c_tangent,
                atom_exchange,
                d_exchange,
                atom_semisolid_exchange,
                d_semisolid_exchange,
                atom_bound,
                d_boundf,
                atom_semisolid,
                d_semisolidf,
                row_dt,
                nil,
                one_att,
                att_rate,
                tl.load(held + 0, mask=active_atom, other=0.0),
                tl.load(held + 12, mask=active_atom, other=0.0),
                tl.load(held + 1, mask=active_atom, other=0.0),
                tl.load(held + 13, mask=active_atom, other=0.0),
                tl.load(held + 2, mask=active_atom, other=0.0),
                tl.load(held + 14, mask=active_atom, other=0.0),
                tl.load(held + 3, mask=active_atom, other=0.0),
                tl.load(held + 15, mask=active_atom, other=0.0),
                tl.load(held + 4, mask=active_atom, other=0.0),
                tl.load(held + 16, mask=active_atom, other=0.0),
                tl.load(held + 5, mask=active_atom, other=0.0),
                tl.load(held + 17, mask=active_atom, other=0.0),
                tl.load(held + 6, mask=active_atom, other=0.0),
                tl.load(held + 18, mask=active_atom, other=0.0),
                tl.load(held + 7, mask=active_atom, other=0.0),
                tl.load(held + 19, mask=active_atom, other=0.0),
                tl.load(held + 8, mask=active_atom, other=0.0),
                tl.load(held + 20, mask=active_atom, other=0.0),
                tl.load(held + 9, mask=active_atom, other=0.0),
                tl.load(held + 21, mask=active_atom, other=0.0),
                tl.load(held + 10, mask=active_atom, other=0.0),
                tl.load(held + 22, mask=active_atom, other=0.0),
                tl.load(held + 11, mask=active_atom, other=0.0),
                tl.load(held + 23, mask=active_atom, other=0.0),
                three_free,
                three_d_free,
                three_pool_b,
                three_d_pool_b,
                three_pool_c,
                three_d_pool_c,
                three_a00,
                three_d_a00,
                three_a01,
                three_d_a01,
                three_a02,
                three_d_a02,
                three_a10,
                three_d_a10,
                three_a11,
                three_d_a11,
                three_a20,
                three_d_a20,
                three_a22,
                three_d_a22,
                three_s00,
                three_d_s00,
                three_s11,
                three_d_s11,
                three_s22,
                three_d_s22,
                three_minors,
                three_d_minors,
                three_sum_flat,
                three_sum_linear,
                three_sum_square,
                three_d_sum_flat,
                three_d_sum_linear,
                three_d_sum_square,
                three_lift,
                three_d_lift,
                three_low,
                three_middle,
                three_d_low,
                three_d_middle,
                three_leading,
                three_d_leading,
                three_first,
                three_d_first,
                three_second,
                three_d_second,
                three_determinant,
                three_d_determinant,
                three_high,
                three_d_high,
                three_radius,
                three_d_radius,
                three_cube,
                three_raw,
                three_d_raw,
                three_argument,
                three_inside_limit,
                three_angle,
                three_d_angle,
                three_centre,
                three_d_centre,
                three_trailing,
                three_d_trailing,
                three_guarded,
                three_d_guarded,
                three_q00,
                three_d_q00,
                three_q01,
                three_d_q01,
                three_q02,
                three_d_q02,
                three_q10,
                three_d_q10,
                three_q11,
                three_d_q11,
                three_q12,
                three_d_q12,
                three_q20,
                three_d_q20,
                three_q21,
                three_d_q21,
                three_q22,
                three_d_q22,
                three_def_00,
                three_dif_00,
                three_def_01,
                three_dif_01,
                three_def_02,
                three_dif_02,
                three_def_10,
                three_dif_10,
                three_def_11,
                three_dif_11,
                three_def_12,
                three_dif_12,
                three_def_20,
                three_dif_20,
                three_def_21,
                three_dif_21,
                three_def_22,
                three_dif_22,
                narrow,
            )
            (
                alt_free,
                alt_d_free,
                alt_pool_b,
                alt_d_pool_b,
                alt_pool_c,
                alt_d_pool_c,
                alt_a00,
                alt_d_a00,
                alt_a01,
                alt_d_a01,
                alt_a02,
                alt_d_a02,
                alt_a10,
                alt_d_a10,
                alt_a11,
                alt_d_a11,
                alt_a20,
                alt_d_a20,
                alt_a22,
                alt_d_a22,
                alt_s00,
                alt_d_s00,
                alt_s11,
                alt_d_s11,
                alt_s22,
                alt_d_s22,
                alt_minors,
                alt_d_minors,
                alt_sum_flat,
                alt_sum_linear,
                alt_sum_square,
                alt_d_sum_flat,
                alt_d_sum_linear,
                alt_d_sum_square,
                alt_lift,
                alt_d_lift,
                alt_low,
                alt_middle,
                alt_d_low,
                alt_d_middle,
                alt_leading,
                alt_d_leading,
                alt_first,
                alt_d_first,
                alt_second,
                alt_d_second,
                alt_determinant,
                alt_d_determinant,
                alt_high,
                alt_d_high,
                alt_radius,
                alt_d_radius,
                alt_cube,
                alt_raw,
                alt_d_raw,
                alt_argument,
                alt_inside_limit,
                alt_angle,
                alt_d_angle,
                alt_centre,
                alt_d_centre,
                alt_trailing,
                alt_d_trailing,
                alt_guarded,
                alt_d_guarded,
                alt_q00,
                alt_d_q00,
                alt_q01,
                alt_d_q01,
                alt_q02,
                alt_d_q02,
                alt_q10,
                alt_d_q10,
                alt_q11,
                alt_d_q11,
                alt_q12,
                alt_d_q12,
                alt_q20,
                alt_d_q20,
                alt_q21,
                alt_d_q21,
                alt_q22,
                alt_d_q22,
            ) = _three_pool_pieces_jvp(
                r1_value,
                nil,
                r1b_value,
                nil,
                r1c_value,
                nil,
                atom_exchange,
                nil,
                atom_semisolid_exchange,
                nil,
                atom_bound,
                nil,
                atom_semisolid,
                nil,
                row_dt,
                unit,
                narrow,
            )
            (
                alt_def_00,
                alt_dif_00,
                alt_def_01,
                alt_dif_01,
                alt_def_02,
                alt_dif_02,
                alt_def_10,
                alt_dif_10,
                alt_def_11,
                alt_dif_11,
                alt_def_12,
                alt_dif_12,
                alt_def_20,
                alt_dif_20,
                alt_def_21,
                alt_dif_21,
                alt_def_22,
                alt_dif_22,
            ) = _three_pool_assemble_jvp(
                alt_free,
                alt_d_free,
                alt_pool_b,
                alt_d_pool_b,
                alt_pool_c,
                alt_d_pool_c,
                alt_a00,
                alt_d_a00,
                alt_a01,
                alt_d_a01,
                alt_a02,
                alt_d_a02,
                alt_a10,
                alt_d_a10,
                alt_a11,
                alt_d_a11,
                alt_a20,
                alt_d_a20,
                alt_a22,
                alt_d_a22,
                alt_s00,
                alt_d_s00,
                alt_s11,
                alt_d_s11,
                alt_s22,
                alt_d_s22,
                alt_minors,
                alt_d_minors,
                alt_sum_flat,
                alt_sum_linear,
                alt_sum_square,
                alt_d_sum_flat,
                alt_d_sum_linear,
                alt_d_sum_square,
                alt_lift,
                alt_d_lift,
                alt_low,
                alt_middle,
                alt_d_low,
                alt_d_middle,
                alt_leading,
                alt_d_leading,
                alt_first,
                alt_d_first,
                alt_second,
                alt_d_second,
                alt_determinant,
                alt_d_determinant,
                alt_high,
                alt_d_high,
                alt_radius,
                alt_d_radius,
                alt_cube,
                alt_raw,
                alt_d_raw,
                alt_argument,
                alt_inside_limit,
                alt_angle,
                alt_d_angle,
                alt_centre,
                alt_d_centre,
                alt_trailing,
                alt_d_trailing,
                alt_guarded,
                alt_d_guarded,
                alt_q00,
                alt_d_q00,
                alt_q01,
                alt_d_q01,
                alt_q02,
                alt_d_q02,
                alt_q10,
                alt_d_q10,
                alt_q11,
                alt_d_q11,
                alt_q12,
                alt_d_q12,
                alt_q20,
                alt_d_q20,
                alt_q21,
                alt_d_q21,
                alt_q22,
                alt_d_q22,
                narrow,
            )
            (
                _span_r1_v,
                _span_r1b_v,
                _span_r1c_v,
                _span_exch_v,
                _span_sexch_v,
                _span_bound_v,
                _span_semi_v,
                _span_dt_v,
                _span_att_v,
                span_r1_t,
                span_r1b_t,
                span_r1c_t,
                span_exch_t,
                span_sexch_t,
                span_bound_t,
                span_semi_t,
                _span_dt_t,
                _span_att_t,
            ) = _three_pool_step_adjoint_jvp(
                r1_value,
                nil,
                r1b_value,
                nil,
                r1c_value,
                nil,
                atom_exchange,
                nil,
                atom_semisolid_exchange,
                nil,
                atom_bound,
                nil,
                atom_semisolid,
                nil,
                row_dt,
                unit,
                one_att,
                att_span,
                tl.load(held + 24, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 25, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 26, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 27, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 28, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 29, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 30, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 31, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 32, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 33, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 34, mask=active_atom, other=0.0),
                nil,
                tl.load(held + 35, mask=active_atom, other=0.0),
                nil,
                alt_free,
                alt_d_free,
                alt_pool_b,
                alt_d_pool_b,
                alt_pool_c,
                alt_d_pool_c,
                alt_a00,
                alt_d_a00,
                alt_a01,
                alt_d_a01,
                alt_a02,
                alt_d_a02,
                alt_a10,
                alt_d_a10,
                alt_a11,
                alt_d_a11,
                alt_a20,
                alt_d_a20,
                alt_a22,
                alt_d_a22,
                alt_s00,
                alt_d_s00,
                alt_s11,
                alt_d_s11,
                alt_s22,
                alt_d_s22,
                alt_minors,
                alt_d_minors,
                alt_sum_flat,
                alt_sum_linear,
                alt_sum_square,
                alt_d_sum_flat,
                alt_d_sum_linear,
                alt_d_sum_square,
                alt_lift,
                alt_d_lift,
                alt_low,
                alt_middle,
                alt_d_low,
                alt_d_middle,
                alt_leading,
                alt_d_leading,
                alt_first,
                alt_d_first,
                alt_second,
                alt_d_second,
                alt_determinant,
                alt_d_determinant,
                alt_high,
                alt_d_high,
                alt_radius,
                alt_d_radius,
                alt_cube,
                alt_raw,
                alt_d_raw,
                alt_argument,
                alt_inside_limit,
                alt_angle,
                alt_d_angle,
                alt_centre,
                alt_d_centre,
                alt_trailing,
                alt_d_trailing,
                alt_guarded,
                alt_d_guarded,
                alt_q00,
                alt_d_q00,
                alt_q01,
                alt_d_q01,
                alt_q02,
                alt_d_q02,
                alt_q10,
                alt_d_q10,
                alt_q11,
                alt_d_q11,
                alt_q12,
                alt_d_q12,
                alt_q20,
                alt_d_q20,
                alt_q21,
                alt_d_q21,
                alt_q22,
                alt_d_q22,
                alt_def_00,
                alt_dif_00,
                alt_def_01,
                alt_dif_01,
                alt_def_02,
                alt_dif_02,
                alt_def_10,
                alt_dif_10,
                alt_def_11,
                alt_dif_11,
                alt_def_12,
                alt_dif_12,
                alt_def_20,
                alt_dif_20,
                alt_def_21,
                alt_dif_21,
                alt_def_22,
                alt_dif_22,
                narrow,
            )
            slope1_v = -1000.0 / (atom_t1 * atom_t1)
            slope1_t = 2000.0 * d_t1 / (atom_t1 * atom_t1 * atom_t1)
            slope1b_v = -1000.0 / (atom_t1b * atom_t1b)
            slope1b_t = 2000.0 * d_t1b / (atom_t1b * atom_t1b * atom_t1b)
            slope1c_v = -1000.0 / (held_semisolid * held_semisolid)
            slope1c_t = (
                2000.0
                * d_semisolid_t1
                / (held_semisolid * held_semisolid * held_semisolid)
            )
            row_r1_t = back_r1_t + span_r1_t
            row_r1b_t = back_r1b_t + span_r1b_t
            row_r1c_t = back_r1c_t + span_r1c_t
            g_t1v += back_r1_v * slope1_v
            g_t1t += row_r1_t * slope1_v + back_r1_v * slope1_t
            g_t1bv += back_r1b_v * slope1b_v
            g_t1bt += row_r1b_t * slope1b_v + back_r1b_v * slope1b_t
            g_t1cv += back_r1c_v * slope1c_v
            g_t1ct += row_r1c_t * slope1c_v + back_r1c_v * slope1c_t
            g_exchv += back_exch_v
            g_excht += back_exch_t + span_exch_t
            g_sexchv += back_sexch_v
            g_sexcht += back_sexch_t + span_sexch_t
            g_boundv += back_bound_v
            g_boundt += back_bound_t + span_bound_t
            g_semiv += back_semi_v
            g_semit += back_semi_t + span_semi_t

    velocity_v = g_flowv * flow_scale + g_washv * direction * washout_scale
    velocity_t = g_flowt * flow_scale + g_washt * direction * washout_scale
    if pools > 0:
        # The fraction also sets where each pool starts, which the walk back
        # reaches last.
        g_boundv += tl.sum(tl.where(state == 0, bbvr - zbvr, 0.0), axis=1)[:, None]
        g_boundt += tl.sum(tl.where(state == 0, bbtr - zbtr, 0.0), axis=1)[:, None]
    if pools == 3:
        g_semiv += tl.sum(tl.where(state == 0, cbvr - zbvr, 0.0), axis=1)[:, None]
        g_semit += tl.sum(tl.where(state == 0, cbtr - zbtr, 0.0), axis=1)[:, None]
    if pools == 1:
        base_row = _BOUND_ROW + 2 * (shim_rows - 1)
        tl.atomic_add(
            grad_tissue_value + base_row * atom_count + atom,
            g_boundv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + base_row * atom_count + atom,
            g_boundt,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_value + (base_row + 1) * atom_count + atom,
            g_exchv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + (base_row + 1) * atom_count + atom,
            g_excht,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_value + (base_row + 2) * atom_count + atom,
            g_t1bv,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + (base_row + 2) * atom_count + atom,
            g_t1bt,
            mask=active_atom,
        )
    if pools == 3:
        semisolid_row = _BOUND_ROW + 2 * (shim_rows - 1)
        stuck = (g_semiv, g_sexchv, g_t1cv)
        stuck_tangents = (g_semit, g_sexcht, g_t1ct)
        for offset in tl.static_range(3):
            tl.atomic_add(
                grad_tissue_value + (semisolid_row + offset) * atom_count + atom,
                stuck[offset],
                mask=active_atom,
            )
            tl.atomic_add(
                grad_tissue_tangent + (semisolid_row + offset) * atom_count + atom,
                stuck_tangents[offset],
                mask=active_atom,
            )
    if pools == 2 or pools == 3:
        base_row = _POOL_B_ROW + 2 * (shim_rows - 1)
        rows = (g_boundv, g_exchv, g_t1bv, g_t2bv, g_shiftv)
        tangent_rows = (g_boundt, g_excht, g_t1bt, g_t2bt, g_shiftt)
        for offset in tl.static_range(5):
            tl.atomic_add(
                grad_tissue_value + (base_row + offset) * atom_count + atom,
                rows[offset],
                mask=active_atom,
            )
            tl.atomic_add(
                grad_tissue_tangent + (base_row + offset) * atom_count + atom,
                tangent_rows[offset],
                mask=active_atom,
            )
    values = (
        g_t1v,
        g_t2v,
        g_m0v,
        g_b1v,
        g_b1pv,
        g_b0v,
        g_invv,
        g_diffv,
        velocity_v,
    )
    tangents = (
        g_t1t,
        g_t2t,
        g_m0t,
        g_b1t,
        g_b1pt,
        g_b0t,
        g_invt,
        g_difft,
        velocity_t,
    )
    for parameter in tl.static_range(_FREE_POOL_COUNT):
        # The transmit pair went to its shim's row above when there is more
        # than one; the rest sit past whatever rows that pair took.
        if not shimmed or (parameter != _B1_ROW and parameter != _B1_PHASE_ROW):
            plane = (
                parameter if parameter < _B1_ROW else parameter + 2 * (shim_rows - 1)
            )
            tl.atomic_add(
                grad_tissue_value + plane * atom_count + atom,
                values[parameter],
                mask=active_atom,
            )
            tl.atomic_add(
                grad_tissue_tangent + plane * atom_count + atom,
                tangents[parameter],
                mask=active_atom,
            )


@triton.jit
def _epg_real_vjp_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    shim_index,
    dot_t1,
    dot_t2,
    dot_m0,
    dot_b1,
    dot_inversion_efficiency,
    dot_diffusion,
    dot_duration,
    dot_flip,
    grad_output_imag,
    grad_tissue_value,
    grad_tissue_tangent,
    grad_flip_value,
    grad_flip_tangent,
    grad_duration_value,
    grad_duration_tangent,
    trajectory_value,
    trajectory_tangent,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shim_rows,
    shimmed: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    # The grid rounds up to whole tiles, so the last program of a wave reaches
    # past it. Those problems are real, but their trajectory rows belong to a
    # later launch and do not exist yet.
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    # The trajectory holds the state entering every event: three planes of
    # configuration orders, for the value and the tangent alike.
    record_stride = 3 * state_count
    trajectory = (problem - problem_base) * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    plus_value = empty
    plus_tangent = empty
    minus_value = empty
    minus_tangent = empty
    long_value = empty + tl.where(state == 0, 1.0, 0.0)
    long_tangent = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    atom_dot_t1 = tl.load(dot_t1 + atom, mask=active_atom, other=0.0)
    atom_dot_t2 = tl.load(dot_t2 + atom, mask=active_atom, other=0.0)
    atom_dot_m0 = 0.0
    if density:
        atom_dot_m0 = tl.load(dot_m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_dot_b1 = 0.0
    if transmit:
        atom_dot_b1 = tl.load(dot_b1 + scalar_atom, mask=active_atom, other=0.0)
    atom_dot_inversion = 0.0
    if inverting:
        atom_dot_inversion = tl.load(
            dot_inversion_efficiency + scalar_atom, mask=active_atom, other=0.0
        )
    atom_damping = 0.0
    atom_dot_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
        atom_dot_damping = tl.load(
            dot_diffusion + scalar_atom, mask=active_atom, other=0.0
        )
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    rate1_value = 1000.0 / atom_t1
    rate1_tangent = -1000.0 * atom_dot_t1 / (atom_t1 * atom_t1)
    rate2_value = 1000.0 / atom_t2
    rate2_tangent = -1000.0 * atom_dot_t2 / (atom_t2 * atom_t2)

    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_value + slot, plus_value, mask=state_mask)
        tl.store(trajectory_value + slot + minus_plane, minus_value, mask=state_mask)
        tl.store(trajectory_value + slot + long_plane, long_value, mask=state_mask)
        tl.store(trajectory_tangent + slot, plus_tangent, mask=state_mask)
        tl.store(
            trajectory_tangent + slot + minus_plane, minus_tangent, mask=state_mask
        )
        tl.store(trajectory_tangent + slot + long_plane, long_tangent, mask=state_mask)

        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        dt_tangent = _event_value(
            dot_duration, event_base, event, active_atom, single_train
        )
        e1_value = tl.exp(-rate1_value * dt_value)
        e1_tangent = -e1_value * (rate1_value * dt_tangent + rate1_tangent * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        e2_tangent = -e2_value * (rate2_value * dt_tangent + rate2_tangent * dt_value)
        damp_z = 1.0
        damp_z_tangent = 0.0
        damp_t = 1.0
        damp_t_tangent = 0.0
        if diffusing:
            damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
                atom_damping, atom_dot_damping, dt_value, dt_tangent, order
            )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t

        plus_tangent = plus_value * e2_tangent + plus_tangent * e2_value
        plus_value = plus_value * e2_value
        minus_tangent = minus_value * e2_tangent + minus_tangent * e2_value
        minus_value = minus_value * e2_value
        long_tangent = long_value * e1_tangent + long_tangent * e1_value
        long_value = long_value * e1_value
        long_value += tl.where(state == 0, recovery_value, 0.0)
        long_tangent += tl.where(state == 0, recovery_tangent, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, state, state_mask, state_count
        )
        shifted_pt, shifted_mt = _shift_real(
            plus_tangent, minus_tangent, state, state_mask, state_count
        )
        plus_value = tl.where(pre_shift, shifted_pv, plus_value)
        minus_value = tl.where(pre_shift, shifted_mv, minus_value)
        plus_tangent = tl.where(pre_shift, shifted_pt, plus_tangent)
        minus_tangent = tl.where(pre_shift, shifted_mt, minus_tangent)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inverted_value = -atom_inversion * long_value
        inverted_tangent = -atom_inversion * long_tangent
        inverted_tangent -= atom_dot_inversion * long_value
        long_value = tl.where(invert, inverted_value, long_value)
        long_tangent = tl.where(invert, inverted_tangent, long_tangent)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_dot_flip = _event_value(
            dot_flip, event_base, event, active_atom, single_train
        )
        pulse_b1 = atom_b1
        pulse_dot_b1 = atom_dot_b1
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse the row of the shim it drives.
        if shimmed:
            shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + shim_row + atom, mask=active_atom, other=1.0)
            pulse_dot_b1 = tl.load(
                dot_b1 + shim_row + atom, mask=active_atom, other=0.0
            )
        alpha_value = event_flip * pulse_b1
        alpha_tangent = event_dot_flip * pulse_b1 + event_flip * pulse_dot_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        cosine_tangent = -sine_value * alpha_tangent
        sine_tangent = cosine_value * alpha_tangent
        chs_value = 0.5 * (1.0 + cosine_value)
        chs_tangent = 0.5 * cosine_tangent
        shs_value = 0.5 * (1.0 - cosine_value)
        shs_tangent = -0.5 * cosine_tangent
        half_sine_value = 0.5 * sine_value
        half_sine_tangent = 0.5 * sine_tangent

        rotated_pv = chs_value * plus_value + shs_value * minus_value
        rotated_pv -= sine_value * long_value
        rotated_pt = chs_value * plus_tangent + chs_tangent * plus_value
        rotated_pt += shs_value * minus_tangent + shs_tangent * minus_value
        rotated_pt -= sine_value * long_tangent + sine_tangent * long_value
        rotated_mv = shs_value * plus_value + chs_value * minus_value
        rotated_mv += sine_value * long_value
        rotated_mt = shs_value * plus_tangent + shs_tangent * plus_value
        rotated_mt += chs_value * minus_tangent + chs_tangent * minus_value
        rotated_mt += sine_value * long_tangent + sine_tangent * long_value
        rotated_zv = half_sine_value * plus_value - half_sine_value * minus_value
        rotated_zv += cosine_value * long_value
        rotated_zt = half_sine_value * plus_tangent + half_sine_tangent * plus_value
        rotated_zt -= half_sine_value * minus_tangent + half_sine_tangent * minus_value
        rotated_zt += cosine_value * long_tangent + cosine_tangent * long_value

        rotate = is_rf & ~is_inversion
        plus_value = tl.where(rotate, rotated_pv, plus_value)
        plus_tangent = tl.where(rotate, rotated_pt, plus_tangent)
        minus_value = tl.where(rotate, rotated_mv, minus_value)
        minus_tangent = tl.where(rotate, rotated_mt, minus_tangent)
        long_value = tl.where(rotate, rotated_zv, long_value)
        long_tangent = tl.where(rotate, rotated_zt, long_tangent)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, state, state_mask, state_count
        )
        shifted_pt, shifted_mt = _shift_real(
            plus_tangent, minus_tangent, state, state_mask, state_count
        )
        plus_value = tl.where(do_shift, shifted_pv, plus_value)
        minus_value = tl.where(do_shift, shifted_mv, minus_value)
        plus_tangent = tl.where(do_shift, shifted_pt, plus_tangent)
        minus_tangent = tl.where(do_shift, shifted_mt, minus_tangent)
        spoil = (event_action & 8) != 0
        plus_value = tl.where(spoil, 0.0, plus_value)
        minus_value = tl.where(spoil, 0.0, minus_value)
        plus_tangent = tl.where(spoil, 0.0, plus_tangent)
        minus_tangent = tl.where(spoil, 0.0, minus_tangent)

    plus_bar_value = empty
    plus_bar_tangent = empty
    minus_bar_value = empty
    minus_bar_tangent = empty
    long_bar_value = empty
    long_bar_tangent = empty
    zero = tl.zeros((problems, 1), tl.float32)
    grad_t1_value = zero
    grad_t1_tangent = zero
    grad_t2_value = zero
    grad_t2_tangent = zero
    grad_m0_value = zero
    grad_m0_tangent = zero
    grad_b1_value = zero
    grad_b1_tangent = zero
    grad_inversion_value = zero
    grad_inversion_tangent = zero
    grad_damping_value = zero
    grad_damping_tangent = zero

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        entry_pv = tl.load(trajectory_value + slot, mask=state_mask, other=0.0)
        entry_mv = tl.load(
            trajectory_value + slot + minus_plane, mask=state_mask, other=0.0
        )
        entry_zv = tl.load(
            trajectory_value + slot + long_plane, mask=state_mask, other=0.0
        )
        entry_pt = tl.load(trajectory_tangent + slot, mask=state_mask, other=0.0)
        entry_mt = tl.load(
            trajectory_tangent + slot + minus_plane, mask=state_mask, other=0.0
        )
        entry_zt = tl.load(
            trajectory_tangent + slot + long_plane, mask=state_mask, other=0.0
        )

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        dt_tangent = _event_value(
            dot_duration, event_base, event, active_atom, single_train
        )
        e1_value = tl.exp(-rate1_value * dt_value)
        e1_tangent = -e1_value * (rate1_value * dt_tangent + rate1_tangent * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        e2_tangent = -e2_value * (rate2_value * dt_tangent + rate2_tangent * dt_value)
        damp_z = 1.0
        damp_z_tangent = 0.0
        damp_t = 1.0
        damp_t_tangent = 0.0
        if diffusing:
            damp_z, damp_z_tangent, damp_t, damp_t_tangent = _damping_jvp(
                atom_damping, atom_dot_damping, dt_value, dt_tangent, order
            )
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value, recovery_tangent = 1.0 - e1_value, -e1_tangent
        bare1_value, bare1_tangent = e1_value, e1_tangent
        bare2_value, bare2_tangent = e2_value, e2_tangent
        e1_tangent = e1_tangent * damp_z + bare1_value * damp_z_tangent
        e1_value = bare1_value * damp_z
        e2_tangent = e2_tangent * damp_t + bare2_value * damp_t_tangent
        e2_value = bare2_value * damp_t

        # Replay the intra-event stages from the recorded entry state.
        stage_pv = entry_pv * e2_value
        stage_pt = entry_pv * e2_tangent + entry_pt * e2_value
        stage_mv = entry_mv * e2_value
        stage_mt = entry_mv * e2_tangent + entry_mt * e2_value
        stage_zv = entry_zv * e1_value + tl.where(state == 0, recovery_value, 0.0)
        stage_zt = entry_zv * e1_tangent + entry_zt * e1_value
        stage_zt += tl.where(state == 0, recovery_tangent, 0.0)

        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            stage_pv, stage_mv, state, state_mask, state_count
        )
        shifted_pt, shifted_mt = _shift_real(
            stage_pt, stage_mt, state, state_mask, state_count
        )
        stage_pv = tl.where(pre_shift, shifted_pv, stage_pv)
        stage_mv = tl.where(pre_shift, shifted_mv, stage_mv)
        stage_pt = tl.where(pre_shift, shifted_pt, stage_pt)
        stage_mt = tl.where(pre_shift, shifted_mt, stage_mt)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, state, state_mask, state_count
        )
        adjoint_pt, adjoint_mt = _shift_real_adjoint(
            plus_bar_tangent, minus_bar_tangent, state, state_mask, state_count
        )
        trailing = do_shift & ~spoil
        plus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_pv, plus_bar_value)
        )
        minus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_mv, minus_bar_value)
        )
        plus_bar_tangent = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_pt, plus_bar_tangent)
        )
        minus_bar_tangent = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_mt, minus_bar_tangent)
        )

        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        inversion_gain = -tl.sum(
            tl.where(invert, long_bar_value * stage_zv, 0.0), axis=1
        )[:, None]
        inversion_gain_tangent = -tl.sum(
            tl.where(
                invert,
                long_bar_value * stage_zt + long_bar_tangent * stage_zv,
                0.0,
            ),
            axis=1,
        )[:, None]
        grad_inversion_value += inversion_gain
        grad_inversion_tangent += inversion_gain_tangent
        inverted_bar_value = -atom_inversion * long_bar_value
        inverted_bar_tangent = (
            -atom_inversion * long_bar_tangent - atom_dot_inversion * long_bar_value
        )
        long_bar_value = tl.where(invert, inverted_bar_value, long_bar_value)
        long_bar_tangent = tl.where(invert, inverted_bar_tangent, long_bar_tangent)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_dot_flip = _event_value(
            dot_flip, event_base, event, active_atom, single_train
        )
        pulse_b1 = atom_b1
        pulse_dot_b1 = atom_dot_b1
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse the row of the shim it drives.
        if shimmed:
            shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + shim_row + atom, mask=active_atom, other=1.0)
            pulse_dot_b1 = tl.load(
                dot_b1 + shim_row + atom, mask=active_atom, other=0.0
            )
        alpha_value = event_flip * pulse_b1
        alpha_tangent = event_dot_flip * pulse_b1 + event_flip * pulse_dot_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        cosine_tangent = -sine_value * alpha_tangent
        sine_tangent = cosine_value * alpha_tangent
        chs_value = 0.5 * (1.0 + cosine_value)
        chs_tangent = 0.5 * cosine_tangent
        shs_value = 0.5 * (1.0 - cosine_value)
        shs_tangent = -0.5 * cosine_tangent
        half_sine_value = 0.5 * sine_value
        half_sine_tangent = 0.5 * sine_tangent

        # d/dalpha of each output row, contracted with the adjoint.
        row_p_value = half_sine_value * stage_mv - half_sine_value * stage_pv
        row_p_value -= cosine_value * stage_zv
        row_p_tangent = half_sine_value * stage_mt + half_sine_tangent * stage_mv
        row_p_tangent -= half_sine_value * stage_pt + half_sine_tangent * stage_pv
        row_p_tangent -= cosine_value * stage_zt + cosine_tangent * stage_zv
        row_m_value = half_sine_value * stage_pv - half_sine_value * stage_mv
        row_m_value += cosine_value * stage_zv
        row_m_tangent = half_sine_value * stage_pt + half_sine_tangent * stage_pv
        row_m_tangent -= half_sine_value * stage_mt + half_sine_tangent * stage_mv
        row_m_tangent += cosine_value * stage_zt + cosine_tangent * stage_zv
        row_z_value = 0.5 * cosine_value * stage_pv - 0.5 * cosine_value * stage_mv
        row_z_value -= sine_value * stage_zv
        row_z_tangent = 0.5 * (cosine_value * stage_pt + cosine_tangent * stage_pv)
        row_z_tangent -= 0.5 * (cosine_value * stage_mt + cosine_tangent * stage_mv)
        row_z_tangent -= sine_value * stage_zt + sine_tangent * stage_zv

        alpha_bar_terms_value = plus_bar_value * row_p_value
        alpha_bar_terms_value += minus_bar_value * row_m_value
        alpha_bar_terms_value += long_bar_value * row_z_value
        alpha_bar_terms_tangent = plus_bar_value * row_p_tangent
        alpha_bar_terms_tangent += plus_bar_tangent * row_p_value
        alpha_bar_terms_tangent += minus_bar_value * row_m_tangent
        alpha_bar_terms_tangent += minus_bar_tangent * row_m_value
        alpha_bar_terms_tangent += long_bar_value * row_z_tangent
        alpha_bar_terms_tangent += long_bar_tangent * row_z_value
        rotate = is_rf & ~is_inversion
        grad_alpha_value = tl.sum(tl.where(rotate, alpha_bar_terms_value, 0.0), axis=1)[
            :, None
        ]
        grad_alpha_tangent = tl.sum(
            tl.where(rotate, alpha_bar_terms_tangent, 0.0), axis=1
        )[:, None]

        # Transpose of the rotation.
        rotated_pbv = chs_value * plus_bar_value + shs_value * minus_bar_value
        rotated_pbv += half_sine_value * long_bar_value
        rotated_pbt = chs_value * plus_bar_tangent + chs_tangent * plus_bar_value
        rotated_pbt += shs_value * minus_bar_tangent + shs_tangent * minus_bar_value
        rotated_pbt += half_sine_value * long_bar_tangent
        rotated_pbt += half_sine_tangent * long_bar_value
        rotated_mbv = shs_value * plus_bar_value + chs_value * minus_bar_value
        rotated_mbv -= half_sine_value * long_bar_value
        rotated_mbt = shs_value * plus_bar_tangent + shs_tangent * plus_bar_value
        rotated_mbt += chs_value * minus_bar_tangent + chs_tangent * minus_bar_value
        rotated_mbt -= half_sine_value * long_bar_tangent
        rotated_mbt -= half_sine_tangent * long_bar_value
        rotated_zbv = -sine_value * plus_bar_value + sine_value * minus_bar_value
        rotated_zbv += cosine_value * long_bar_value
        rotated_zbt = -sine_value * plus_bar_tangent - sine_tangent * plus_bar_value
        rotated_zbt += sine_value * minus_bar_tangent + sine_tangent * minus_bar_value
        rotated_zbt += cosine_value * long_bar_tangent + cosine_tangent * long_bar_value

        plus_bar_value = tl.where(rotate, rotated_pbv, plus_bar_value)
        plus_bar_tangent = tl.where(rotate, rotated_pbt, plus_bar_tangent)
        minus_bar_value = tl.where(rotate, rotated_mbv, minus_bar_value)
        minus_bar_tangent = tl.where(rotate, rotated_mbt, minus_bar_tangent)
        long_bar_value = tl.where(rotate, rotated_zbv, long_bar_value)
        long_bar_tangent = tl.where(rotate, rotated_zbt, long_bar_tangent)

        flip_gain_value = grad_alpha_value * pulse_b1
        flip_gain_tangent = grad_alpha_tangent * pulse_b1
        flip_gain_tangent += grad_alpha_value * pulse_dot_b1
        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip_value + event_base + event, flip_gain_value, mask=writes_flip
        )
        tl.atomic_add(
            grad_flip_tangent + event_base + event, flip_gain_tangent, mask=writes_flip
        )
        if shimmed:
            # A pulse's transmit gradient belongs to the shim it drives, so
            # with several it lands in that shim's row rather than in a
            # register summed over the whole train.
            tl.atomic_add(
                grad_tissue_value + 3 * atom_count + shim_row + atom,
                grad_alpha_value * event_flip,
                mask=writes_flip,
            )
            tl.atomic_add(
                grad_tissue_tangent + 3 * atom_count + shim_row + atom,
                grad_alpha_tangent * event_flip + grad_alpha_value * event_dot_flip,
                mask=writes_flip,
            )
        else:
            grad_b1_value += tl.where(rotate, grad_alpha_value * event_flip, 0.0)
            grad_b1_tangent += tl.where(
                rotate,
                grad_alpha_tangent * event_flip + grad_alpha_value * event_dot_flip,
                0.0,
            )

        # The sample is i * m0 * plus[0]; only the imaginary seed acts.
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed = tl.load(
            grad_output_imag + problem * output_count + out,
            mask=active_atom & record & (out >= 0),
            other=0.0,
        )
        grad_m0_value += tl.sum(tl.where(state == 0, seed * stage_pv, 0.0), axis=1)[
            :, None
        ]
        grad_m0_tangent += tl.sum(tl.where(state == 0, seed * stage_pt, 0.0), axis=1)[
            :, None
        ]
        plus_bar_value += tl.where(state == 0, seed * atom_m0, 0.0)
        plus_bar_tangent += tl.where(state == 0, seed * atom_dot_m0, 0.0)

        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, state, state_mask, state_count
        )
        adjoint_pt, adjoint_mt = _shift_real_adjoint(
            plus_bar_tangent, minus_bar_tangent, state, state_mask, state_count
        )
        plus_bar_value = tl.where(pre_shift, adjoint_pv, plus_bar_value)
        minus_bar_value = tl.where(pre_shift, adjoint_mv, minus_bar_value)
        plus_bar_tangent = tl.where(pre_shift, adjoint_pt, plus_bar_tangent)
        minus_bar_tangent = tl.where(pre_shift, adjoint_mt, minus_bar_tangent)

        cot2_value = plus_bar_value * entry_pv + minus_bar_value * entry_mv
        cot2_tangent = (
            plus_bar_value * entry_pt
            + plus_bar_tangent * entry_pv
            + minus_bar_value * entry_mt
            + minus_bar_tangent * entry_mv
        )
        cot1_value = long_bar_value * entry_zv
        cot1_tangent = long_bar_value * entry_zt + long_bar_tangent * entry_zv
        grad_e2_value = tl.sum(cot2_value * damp_t, axis=1)[:, None]
        grad_e2_tangent = tl.sum(
            cot2_value * damp_t_tangent + cot2_tangent * damp_t, axis=1
        )[:, None]
        grad_e1_value = tl.sum(cot1_value * damp_z, axis=1)[:, None]
        grad_e1_value -= tl.sum(tl.where(state == 0, long_bar_value, 0.0), axis=1)[
            :, None
        ]
        grad_e1_tangent = tl.sum(
            cot1_value * damp_z_tangent + cot1_tangent * damp_z, axis=1
        )[:, None]
        grad_e1_tangent -= tl.sum(tl.where(state == 0, long_bar_tangent, 0.0), axis=1)[
            :, None
        ]

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum. Order zero has no longitudinal weight, which
        # keeps recovery out of this.
        spread_value = zero
        spread_tangent = zero
        if diffusing:
            weighted_value = (
                cot1_value * bare1_value * damp_z * longitudinal_weight
                + cot2_value * bare2_value * damp_t * transverse_weight
            )
            weighted_tangent = (
                cot1_tangent * bare1_value * damp_z
                + cot1_value * bare1_tangent * damp_z
                + cot1_value * bare1_value * damp_z_tangent
            ) * longitudinal_weight + (
                cot2_tangent * bare2_value * damp_t
                + cot2_value * bare2_tangent * damp_t
                + cot2_value * bare2_value * damp_t_tangent
            ) * transverse_weight
            spread_value = tl.sum(weighted_value, axis=1)[:, None]
            spread_tangent = tl.sum(weighted_tangent, axis=1)[:, None]
            grad_damping_value += -spread_value * dt_value
            grad_damping_tangent += -(
                spread_value * dt_tangent + spread_tangent * dt_value
            )

        plus_bar_tangent = plus_bar_value * e2_tangent + plus_bar_tangent * e2_value
        plus_bar_value = plus_bar_value * e2_value
        minus_bar_tangent = minus_bar_value * e2_tangent + minus_bar_tangent * e2_value
        minus_bar_value = minus_bar_value * e2_value
        long_bar_tangent = long_bar_value * e1_tangent + long_bar_tangent * e1_value
        long_bar_value = long_bar_value * e1_value

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse1_tangent = -2000.0 * atom_dot_t1 / (atom_t1 * atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        inverse2_tangent = -2000.0 * atom_dot_t2 / (atom_t2 * atom_t2 * atom_t2)
        scale1_value = bare1_value * dt_value * inverse1_value
        scale1_tangent = bare1_tangent * dt_value * inverse1_value
        scale1_tangent += bare1_value * dt_tangent * inverse1_value
        scale1_tangent += bare1_value * dt_value * inverse1_tangent
        scale2_value = bare2_value * dt_value * inverse2_value
        scale2_tangent = bare2_tangent * dt_value * inverse2_value
        scale2_tangent += bare2_value * dt_tangent * inverse2_value
        scale2_tangent += bare2_value * dt_value * inverse2_tangent
        grad_t1_value += grad_e1_value * scale1_value
        grad_t1_tangent += grad_e1_value * scale1_tangent
        grad_t1_tangent += grad_e1_tangent * scale1_value
        grad_t2_value += grad_e2_value * scale2_value
        grad_t2_tangent += grad_e2_value * scale2_tangent
        grad_t2_tangent += grad_e2_tangent * scale2_value

        decay1_value = rate1_value * bare1_value
        decay1_tangent = rate1_value * bare1_tangent + rate1_tangent * bare1_value
        decay2_value = rate2_value * bare2_value
        decay2_tangent = rate2_value * bare2_tangent + rate2_tangent * bare2_value
        duration_gain_value = -grad_e1_value * decay1_value
        duration_gain_value -= grad_e2_value * decay2_value
        duration_gain_tangent = -(
            grad_e1_value * decay1_tangent + grad_e1_tangent * decay1_value
        )
        duration_gain_tangent -= (
            grad_e2_value * decay2_tangent + grad_e2_tangent * decay2_value
        )
        duration_gain_value += -spread_value * atom_damping
        duration_gain_tangent += -(
            spread_value * atom_dot_damping + spread_tangent * atom_damping
        )
        tl.atomic_add(
            grad_duration_value + event_base + event,
            duration_gain_value,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_duration_tangent + event_base + event,
            duration_gain_tangent,
            mask=active_atom,
        )

    tl.atomic_add(grad_tissue_value + atom, grad_t1_value, mask=active_atom)
    tl.atomic_add(grad_tissue_tangent + atom, grad_t1_tangent, mask=active_atom)
    tl.atomic_add(
        grad_tissue_value + atom_count + atom, grad_t2_value, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_tangent + atom_count + atom, grad_t2_tangent, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_value + 2 * atom_count + atom, grad_m0_value, mask=active_atom
    )
    tl.atomic_add(
        grad_tissue_tangent + 2 * atom_count + atom, grad_m0_tangent, mask=active_atom
    )
    if not shimmed:
        tl.atomic_add(
            grad_tissue_value + 3 * atom_count + atom,
            grad_b1_value,
            mask=active_atom,
        )
        tl.atomic_add(
            grad_tissue_tangent + 3 * atom_count + atom,
            grad_b1_tangent,
            mask=active_atom,
        )
    # The transmit pair takes a row per shim each in the plane the complex
    # path allocates, so the rows past it move even though this kernel leaves
    # the transmit phase at zero throughout.
    past_transmit = 2 * (shim_rows - 1)
    tl.atomic_add(
        grad_tissue_value + (6 + past_transmit) * atom_count + atom,
        grad_inversion_value,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_tangent + (6 + past_transmit) * atom_count + atom,
        grad_inversion_tangent,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_value + (7 + past_transmit) * atom_count + atom,
        grad_damping_value,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue_tangent + (7 + past_transmit) * atom_count + atom,
        grad_damping_tangent,
        mask=active_atom,
    )


@triton.jit
def _epg_real_vjp_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    shim_index,
    grad_output_imag,
    grad_tissue,
    grad_flip,
    grad_duration,
    trajectory_value,
    problem_base,
    problem_end,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shim_rows,
    shimmed: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = problem_base + tl.program_id(0) * problems
    problem = problem + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    # The grid rounds up to whole tiles, so the last program of a wave reaches
    # past it. Those problems are real, but their trajectory rows belong to a
    # later launch and do not exist yet.
    active_atom = problem < problem_end
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    # The trajectory holds the state entering every event: three planes of
    # configuration orders.
    record_stride = 3 * state_count
    trajectory = (problem - problem_base) * event_count * record_stride + state
    minus_plane = state_count
    long_plane = 2 * state_count

    empty = tl.zeros((problems, block_states), tl.float32)
    plus_value = empty
    minus_value = empty
    long_value = empty + tl.where(state == 0, 1.0, 0.0)

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    atom_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
    order = state.to(tl.float32)
    longitudinal_weight = order * order
    transverse_weight = longitudinal_weight + order + 0.3333333333333333
    rate1_value = 1000.0 / atom_t1
    rate2_value = 1000.0 / atom_t2

    event_base = train * event_count
    for event in range(0, event_count):
        slot = trajectory + event * record_stride
        tl.store(trajectory_value + slot, plus_value, mask=state_mask)
        tl.store(trajectory_value + slot + minus_plane, minus_value, mask=state_mask)
        tl.store(trajectory_value + slot + long_plane, long_value, mask=state_mask)

        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        e1_value = tl.exp(-rate1_value * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        damp_z = 1.0
        damp_t = 1.0
        if diffusing:
            damp_z, damp_t = _damping(atom_damping, dt_value, order)
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = 1.0 - e1_value
        bare1_value = e1_value
        bare2_value = e2_value
        e1_value = bare1_value * damp_z
        e2_value = bare2_value * damp_t

        plus_value = plus_value * e2_value
        minus_value = minus_value * e2_value
        long_value = long_value * e1_value
        long_value += tl.where(state == 0, recovery_value, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, state, state_mask, state_count
        )
        plus_value = tl.where(pre_shift, shifted_pv, plus_value)
        minus_value = tl.where(pre_shift, shifted_mv, minus_value)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        long_value = tl.where(invert, -atom_inversion * long_value, long_value)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        pulse_b1 = atom_b1
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse the row of the shim it drives.
        if shimmed:
            shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + shim_row + atom, mask=active_atom, other=1.0)
        alpha_value = event_flip * pulse_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        chs_value = 0.5 * (1.0 + cosine_value)
        shs_value = 0.5 * (1.0 - cosine_value)
        half_sine_value = 0.5 * sine_value

        rotated_pv = chs_value * plus_value + shs_value * minus_value
        rotated_pv -= sine_value * long_value
        rotated_mv = shs_value * plus_value + chs_value * minus_value
        rotated_mv += sine_value * long_value
        rotated_zv = half_sine_value * plus_value - half_sine_value * minus_value
        rotated_zv += cosine_value * long_value

        rotate = is_rf & ~is_inversion
        plus_value = tl.where(rotate, rotated_pv, plus_value)
        minus_value = tl.where(rotate, rotated_mv, minus_value)
        long_value = tl.where(rotate, rotated_zv, long_value)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pv, shifted_mv = _shift_real(
            plus_value, minus_value, state, state_mask, state_count
        )
        plus_value = tl.where(do_shift, shifted_pv, plus_value)
        minus_value = tl.where(do_shift, shifted_mv, minus_value)
        spoil = (event_action & 8) != 0
        plus_value = tl.where(spoil, 0.0, plus_value)
        minus_value = tl.where(spoil, 0.0, minus_value)

    plus_bar_value = empty
    minus_bar_value = empty
    long_bar_value = empty
    zero = tl.zeros((problems, 1), tl.float32)
    grad_t1_value = zero
    grad_t2_value = zero
    grad_m0_value = zero
    grad_b1_value = zero
    grad_inversion_value = zero
    grad_damping_value = zero

    for reverse in range(0, event_count):
        event = event_count - 1 - reverse
        slot = trajectory + event * record_stride
        entry_pv = tl.load(trajectory_value + slot, mask=state_mask, other=0.0)
        entry_mv = tl.load(
            trajectory_value + slot + minus_plane, mask=state_mask, other=0.0
        )
        entry_zv = tl.load(
            trajectory_value + slot + long_plane, mask=state_mask, other=0.0
        )

        event_action = tl.load(action + event).to(tl.int32)
        event_kind = tl.load(kind + event)
        dt_value = _event_value(duration, event_base, event, active_atom, single_train)
        e1_value = tl.exp(-rate1_value * dt_value)
        e2_value = tl.exp(-rate2_value * dt_value)
        damp_z = 1.0
        damp_t = 1.0
        if diffusing:
            damp_z, damp_t = _damping(atom_damping, dt_value, order)
        # Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = 1.0 - e1_value
        bare1_value = e1_value
        bare2_value = e2_value
        e1_value = bare1_value * damp_z
        e2_value = bare2_value * damp_t

        # Replay the intra-event stages from the recorded entry state.
        stage_pv = entry_pv * e2_value
        stage_mv = entry_mv * e2_value
        stage_zv = entry_zv * e1_value + tl.where(state == 0, recovery_value, 0.0)

        pre_shift = (event_action & 1) != 0
        shifted_pv, shifted_mv = _shift_real(
            stage_pv, stage_mv, state, state_mask, state_count
        )
        stage_pv = tl.where(pre_shift, shifted_pv, stage_pv)
        stage_mv = tl.where(pre_shift, shifted_mv, stage_mv)

        # Undo the trailing spoil or shift.
        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        spoil = (event_action & 8) != 0
        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, state, state_mask, state_count
        )
        trailing = do_shift & ~spoil
        plus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_pv, plus_bar_value)
        )
        minus_bar_value = tl.where(
            spoil, 0.0, tl.where(trailing, adjoint_mv, minus_bar_value)
        )

        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        grad_inversion_value += -tl.sum(
            tl.where(invert, long_bar_value * stage_zv, 0.0), axis=1
        )[:, None]
        long_bar_value = tl.where(
            invert, -atom_inversion * long_bar_value, long_bar_value
        )

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        pulse_b1 = atom_b1
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse the row of the shim it drives.
        if shimmed:
            shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
            if transmit:
                pulse_b1 = tl.load(b1 + shim_row + atom, mask=active_atom, other=1.0)
        alpha_value = event_flip * pulse_b1
        cosine_value = tl.cos(alpha_value)
        sine_value = tl.sin(alpha_value)
        chs_value = 0.5 * (1.0 + cosine_value)
        shs_value = 0.5 * (1.0 - cosine_value)
        half_sine_value = 0.5 * sine_value

        # d/dalpha of each output row, contracted with the adjoint.
        row_p_value = half_sine_value * stage_mv - half_sine_value * stage_pv
        row_p_value -= cosine_value * stage_zv
        row_m_value = half_sine_value * stage_pv - half_sine_value * stage_mv
        row_m_value += cosine_value * stage_zv
        row_z_value = 0.5 * cosine_value * stage_pv - 0.5 * cosine_value * stage_mv
        row_z_value -= sine_value * stage_zv

        alpha_bar_terms_value = plus_bar_value * row_p_value
        alpha_bar_terms_value += minus_bar_value * row_m_value
        alpha_bar_terms_value += long_bar_value * row_z_value
        rotate = is_rf & ~is_inversion
        grad_alpha_value = tl.sum(tl.where(rotate, alpha_bar_terms_value, 0.0), axis=1)[
            :, None
        ]

        # Transpose of the rotation.
        rotated_pbv = chs_value * plus_bar_value + shs_value * minus_bar_value
        rotated_pbv += half_sine_value * long_bar_value
        rotated_mbv = shs_value * plus_bar_value + chs_value * minus_bar_value
        rotated_mbv -= half_sine_value * long_bar_value
        rotated_zbv = -sine_value * plus_bar_value + sine_value * minus_bar_value
        rotated_zbv += cosine_value * long_bar_value

        plus_bar_value = tl.where(rotate, rotated_pbv, plus_bar_value)
        minus_bar_value = tl.where(rotate, rotated_mbv, minus_bar_value)
        long_bar_value = tl.where(rotate, rotated_zbv, long_bar_value)

        writes_flip = active_atom & rotate
        tl.atomic_add(
            grad_flip + event_base + event,
            grad_alpha_value * pulse_b1,
            mask=writes_flip,
        )
        if shimmed:
            # A pulse's transmit gradient belongs to the shim it drives, so
            # with several it lands in that shim's row rather than in a
            # register summed over the whole train.
            tl.atomic_add(
                grad_tissue + 3 * atom_count + shim_row + atom,
                grad_alpha_value * event_flip,
                mask=writes_flip,
            )
        else:
            grad_b1_value += tl.where(rotate, grad_alpha_value * event_flip, 0.0)

        # The sample is i * m0 * plus[0]; only the imaginary seed acts.
        record = ((event_action & 32) != 0) & (event_kind == 2)
        out = tl.load(output_index + event)
        seed = tl.load(
            grad_output_imag + problem * output_count + out,
            mask=active_atom & record & (out >= 0),
            other=0.0,
        )
        grad_m0_value += tl.sum(tl.where(state == 0, seed * stage_pv, 0.0), axis=1)[
            :, None
        ]
        plus_bar_value += tl.where(state == 0, seed * atom_m0, 0.0)

        adjoint_pv, adjoint_mv = _shift_real_adjoint(
            plus_bar_value, minus_bar_value, state, state_mask, state_count
        )
        plus_bar_value = tl.where(pre_shift, adjoint_pv, plus_bar_value)
        minus_bar_value = tl.where(pre_shift, adjoint_mv, minus_bar_value)

        cot2_value = plus_bar_value * entry_pv + minus_bar_value * entry_mv
        cot1_value = long_bar_value * entry_zv
        grad_e2_value = tl.sum(cot2_value * damp_t, axis=1)[:, None]
        grad_e1_value = tl.sum(cot1_value * damp_z, axis=1)[:, None]
        grad_e1_value -= tl.sum(tl.where(state == 0, long_bar_value, 0.0), axis=1)[
            :, None
        ]

        # The rate and the interval multiply every order's b-weight, so both
        # take a weighted sum. Order zero has no longitudinal weight, which
        # keeps recovery out of this.
        spread_value = zero
        if diffusing:
            weighted_value = (
                cot1_value * bare1_value * damp_z * longitudinal_weight
                + cot2_value * bare2_value * damp_t * transverse_weight
            )
            spread_value = tl.sum(weighted_value, axis=1)[:, None]
            grad_damping_value += -spread_value * dt_value

        plus_bar_value = plus_bar_value * e2_value
        minus_bar_value = minus_bar_value * e2_value
        long_bar_value = long_bar_value * e1_value

        inverse1_value = 1000.0 / (atom_t1 * atom_t1)
        inverse2_value = 1000.0 / (atom_t2 * atom_t2)
        grad_t1_value += grad_e1_value * (bare1_value * dt_value * inverse1_value)
        grad_t2_value += grad_e2_value * (bare2_value * dt_value * inverse2_value)

        duration_gain_value = -grad_e1_value * (rate1_value * bare1_value)
        duration_gain_value -= grad_e2_value * (rate2_value * bare2_value)
        duration_gain_value += -spread_value * atom_damping
        tl.atomic_add(
            grad_duration + event_base + event,
            duration_gain_value,
            mask=active_atom,
        )

    tl.atomic_add(grad_tissue + atom, grad_t1_value, mask=active_atom)
    tl.atomic_add(grad_tissue + atom_count + atom, grad_t2_value, mask=active_atom)
    tl.atomic_add(grad_tissue + 2 * atom_count + atom, grad_m0_value, mask=active_atom)
    if not shimmed:
        tl.atomic_add(
            grad_tissue + 3 * atom_count + atom, grad_b1_value, mask=active_atom
        )
    # The transmit pair takes a row per shim each in the plane the complex
    # path allocates, so the rows past it move even though this kernel leaves
    # the transmit phase at zero throughout.
    past_transmit = 2 * (shim_rows - 1)
    tl.atomic_add(
        grad_tissue + (6 + past_transmit) * atom_count + atom,
        grad_inversion_value,
        mask=active_atom,
    )
    tl.atomic_add(
        grad_tissue + (7 + past_transmit) * atom_count + atom,
        grad_damping_value,
        mask=active_atom,
    )


@triton.jit
def _epg_real_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    shim_index,
    output_real,
    output_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shimmed: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count

    empty = tl.zeros((problems, block_states), tl.float32)
    plus = empty
    minus = empty
    longitudinal = empty + tl.where(state == 0, 1.0, 0.0)

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    rate1 = 1000.0 / atom_t1
    rate2 = 1000.0 / atom_t2
    atom_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
    order = state.to(tl.float32)

    # The relaxation factors depend on the event only through its duration, and
    # a train repeats its intervals: an interval as long as the last one reuses
    # the factors rather than taking the two exponentials again. Where several
    # trains share the program the durations differ across its lanes and there
    # is nothing uniform to compare, so only a single-train launch memoizes.
    last_dt = -1.0
    e1 = rate1 * 0.0 + 1.0
    e2 = rate2 * 0.0 + 1.0

    event_base = train * event_count
    # Two events to an iteration. A repetition is several events -- a pulse,
    # a sample, an interval -- so the loop runs longer than the sequence is
    # repetitions, and unrolling lets one back-edge and one set of event
    # bookkeeping serve two of them. Two is where it stops paying: four was
    # measured slower, and the body is already large enough that widening it
    # costs registers.
    for event in tl.range(0, event_count, loop_unroll_factor=2):
        # Read here rather than through the helper: one train gives a duration
        # the whole program shares, and the skip and the memo below both want
        # to compare it as the single number it is.
        if single_train:
            dt = tl.load(duration + event)
        else:
            dt = tl.load(duration + event_base + event, mask=active_atom, other=0.0)
        # An event of no duration relaxes nothing: both factors are one and the
        # recovery term is zero. Half the events of a spoiled repetition are
        # instantaneous, and reducing over the trains this program carries makes
        # that a branch the whole program agrees on rather than a tile of
        # multiplies by one.
        if single_train:
            relaxes = dt != 0.0
        else:
            relaxes = tl.max(dt) != 0.0
        if relaxes:
            if single_train:
                if dt != last_dt:
                    e1 = tl.exp(-rate1 * dt)
                    e2 = tl.exp(-rate2 * dt)
                    last_dt = dt
            else:
                e1 = tl.exp(-rate1 * dt)
                e2 = tl.exp(-rate2 * dt)
            damp_z = 1.0
            damp_t = 1.0
            if diffusing:
                damp_z, damp_t = _damping(atom_damping, dt, order)
            recovery = 1.0 - e1
            plus *= e2 * damp_t
            minus *= e2 * damp_t
            longitudinal = longitudinal * (e1 * damp_z) + tl.where(
                state == 0, recovery, 0.0
            )

        # Every flag below is read from a per-event array with no atom index, so
        # it is uniform across the program and can steer real control flow. A
        # `tl.where` would make every event pay for every operator: a spoiled
        # repetition is four events and needs one rotation and one shift.
        event_action = tl.load(action + event).to(tl.int32)
        if (event_action & 1) != 0:
            plus, minus = _shift_real(plus, minus, state, state_mask, state_count)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        if is_rf and is_inversion:
            longitudinal = -atom_inversion * longitudinal
        elif is_rf:
            alpha = _event_value(flip, event_base, event, active_atom, single_train)
            pulse_b1 = atom_b1
            # One shim is the whole sequence's transmit field, loaded once
            # above; several give each pulse the row of the shim it drives.
            if shimmed and transmit:
                shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
                pulse_b1 = tl.load(b1 + shim_row + atom, mask=active_atom, other=1.0)
            alpha *= pulse_b1
            cosine = tl.cos(alpha)
            sine = tl.sin(alpha)
            cosine_half_sq = 0.5 * (1.0 + cosine)
            sine_half_sq = 0.5 * (1.0 - cosine)
            half_sine = 0.5 * sine
            rotated_p = (
                cosine_half_sq * plus + sine_half_sq * minus - sine * longitudinal
            )
            rotated_m = (
                sine_half_sq * plus + cosine_half_sq * minus + sine * longitudinal
            )
            longitudinal = half_sine * plus - half_sine * minus + cosine * longitudinal
            plus = rotated_p
            minus = rotated_m

        if ((event_action & 32) != 0) and event_kind == 2:
            out = tl.load(output_index + event)
            output_offset = problem * output_count + out
            output_mask = active_atom & (state == 0) & (out >= 0)
            tl.store(output_real + output_offset + state, empty, mask=output_mask)
            tl.store(
                output_imag + output_offset + state, atom_m0 * plus, mask=output_mask
            )

        if ((event_action & 2) != 0) or ((event_action & 16) != 0):
            plus, minus = _shift_real(plus, minus, state, state_mask, state_count)
        if (event_action & 8) != 0:
            plus = empty
            minus = empty


@triton.jit
def _epg_real_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    inversion_efficiency,
    diffusion,
    duration,
    kind,
    flip,
    action,
    output_index,
    shim_index,
    tangent_t1,
    tangent_t2,
    tangent_m0,
    tangent_b1,
    tangent_inversion_efficiency,
    tangent_diffusion,
    tangent_duration,
    tangent_flip,
    output_real,
    output_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shimmed: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count

    empty = tl.zeros((problems, block_states), tl.float32)
    plus = empty
    minus = empty
    longitudinal = empty + tl.where(state == 0, 1.0, 0.0)
    dot_plus = empty
    dot_minus = empty
    dot_longitudinal = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    dot_t1 = tl.load(tangent_t1 + atom, mask=active_atom, other=0.0)
    dot_t2 = tl.load(tangent_t2 + atom, mask=active_atom, other=0.0)
    dot_m0 = 0.0
    if density:
        dot_m0 = tl.load(tangent_m0 + scalar_atom, mask=active_atom, other=0.0)
    dot_b1 = 0.0
    if transmit:
        dot_b1 = tl.load(tangent_b1 + scalar_atom, mask=active_atom, other=0.0)
    dot_inversion = 0.0
    if inverting:
        dot_inversion = tl.load(
            tangent_inversion_efficiency + scalar_atom, mask=active_atom, other=0.0
        )
    rate1 = 1000.0 / atom_t1
    rate2 = 1000.0 / atom_t2
    atom_damping = 0.0
    dot_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
        dot_damping = tl.load(
            tangent_diffusion + scalar_atom, mask=active_atom, other=0.0
        )
    order = state.to(tl.float32)

    event_base = train * event_count
    # Two events to an iteration. A repetition is several events -- a pulse,
    # a sample, an interval -- so the loop runs longer than the sequence is
    # repetitions, and unrolling lets one back-edge and one set of event
    # bookkeeping serve two of them. Two is where it stops paying: four was
    # measured slower, and the body is already large enough that widening it
    # costs registers.
    for event in tl.range(0, event_count, loop_unroll_factor=2):
        dt = _event_value(duration, event_base, event, active_atom, single_train)
        dot_dt = _event_value(
            tangent_duration, event_base, event, active_atom, single_train
        )
        # An event of no duration relaxes nothing, and carries no tangent along
        # the relaxation either: both factors are one and both their derivatives
        # are zero.
        if tl.max(dt) != 0.0 or tl.max(dot_dt) != 0.0:
            e1 = tl.exp(-rate1 * dt)
            e2 = tl.exp(-rate2 * dt)
            dot_e1 = e1 * (1000.0 * dt * dot_t1 / (atom_t1 * atom_t1) - rate1 * dot_dt)
            dot_e2 = e2 * (1000.0 * dt * dot_t2 / (atom_t2 * atom_t2) - rate2 * dot_dt)
            damp_z = 1.0
            ddamp_z = 0.0
            damp_t = 1.0
            ddamp_t = 0.0
            if diffusing:
                damp_z, ddamp_z, damp_t, ddamp_t = _damping_jvp(
                    atom_damping, dot_damping, dt, dot_dt, order
                )
            # Order zero is undamped, so the recovery keeps the bare factor.
            recovery, dot_recovery = 1.0 - e1, -dot_e1
            dot_e1 = dot_e1 * damp_z + e1 * ddamp_z
            e1 = e1 * damp_z
            dot_e2 = dot_e2 * damp_t + e2 * ddamp_t
            e2 = e2 * damp_t
            dot_plus = dot_plus * e2 + plus * dot_e2
            dot_minus = dot_minus * e2 + minus * dot_e2
            dot_longitudinal = dot_longitudinal * e1 + longitudinal * dot_e1
            dot_longitudinal += tl.where(state == 0, dot_recovery, 0.0)
            plus *= e2
            minus *= e2
            longitudinal = longitudinal * e1 + tl.where(state == 0, recovery, 0.0)

        # Every flag below is read from a per-event array with no atom index, so
        # it is uniform across the program and can steer real control flow.
        event_action = tl.load(action + event).to(tl.int32)
        if (event_action & 1) != 0:
            plus, minus = _shift_real(plus, minus, state, state_mask, state_count)
            dot_plus, dot_minus = _shift_real(
                dot_plus, dot_minus, state, state_mask, state_count
            )

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        if is_rf and is_inversion:
            dot_longitudinal = (
                -atom_inversion * dot_longitudinal - dot_inversion * longitudinal
            )
            longitudinal = -atom_inversion * longitudinal
        elif is_rf:
            event_flip = _event_value(
                flip, event_base, event, active_atom, single_train
            )
            dot_flip = _event_value(
                tangent_flip, event_base, event, active_atom, single_train
            )
            pulse_b1 = atom_b1
            pulse_dot_b1 = dot_b1
            # One shim is the whole sequence's transmit field, loaded once above;
            # several give each pulse the row of the shim it drives.
            if shimmed:
                shim_row = tl.load(shim_index + event).to(tl.int64) * atom_count
                if transmit:
                    pulse_b1 = tl.load(
                        b1 + shim_row + atom, mask=active_atom, other=1.0
                    )
                pulse_dot_b1 = tl.load(
                    tangent_b1 + shim_row + atom, mask=active_atom, other=0.0
                )
            alpha = event_flip * pulse_b1
            dot_alpha = dot_flip * pulse_b1 + event_flip * pulse_dot_b1
            cosine = tl.cos(alpha)
            sine = tl.sin(alpha)
            cosine_half_sq = 0.5 * (1.0 + cosine)
            sine_half_sq = 0.5 * (1.0 - cosine)
            half_sine = 0.5 * sine
            dot_cosine = -sine * dot_alpha
            dot_sine = cosine * dot_alpha
            dot_cosine_half_sq = -0.5 * sine * dot_alpha
            dot_sine_half_sq = 0.5 * sine * dot_alpha
            dot_half_sine = 0.5 * cosine * dot_alpha

            rotated_dp = cosine_half_sq * dot_plus + dot_cosine_half_sq * plus
            rotated_dp += sine_half_sq * dot_minus + dot_sine_half_sq * minus
            rotated_dp -= sine * dot_longitudinal + dot_sine * longitudinal
            rotated_dm = sine_half_sq * dot_plus + dot_sine_half_sq * plus
            rotated_dm += cosine_half_sq * dot_minus + dot_cosine_half_sq * minus
            rotated_dm += sine * dot_longitudinal + dot_sine * longitudinal
            rotated_dz = half_sine * dot_plus + dot_half_sine * plus
            rotated_dz -= half_sine * dot_minus + dot_half_sine * minus
            rotated_dz += cosine * dot_longitudinal + dot_cosine * longitudinal
            rotated_p = (
                cosine_half_sq * plus + sine_half_sq * minus - sine * longitudinal
            )
            rotated_m = (
                sine_half_sq * plus + cosine_half_sq * minus + sine * longitudinal
            )
            rotated_z = half_sine * plus - half_sine * minus + cosine * longitudinal

            plus = rotated_p
            minus = rotated_m
            longitudinal = rotated_z
            dot_plus = rotated_dp
            dot_minus = rotated_dm
            dot_longitudinal = rotated_dz

        if ((event_action & 32) != 0) and event_kind == 2:
            out = tl.load(output_index + event)
            output_offset = problem * output_count + out
            output_mask = active_atom & (state == 0) & (out >= 0)
            signal_imag = dot_m0 * plus + atom_m0 * dot_plus
            tl.store(output_real + output_offset + state, empty, mask=output_mask)
            tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        if ((event_action & 2) != 0) or ((event_action & 16) != 0):
            plus, minus = _shift_real(plus, minus, state, state_mask, state_count)
            dot_plus, dot_minus = _shift_real(
                dot_plus, dot_minus, state, state_mask, state_count
            )
        if (event_action & 8) != 0:
            plus = empty
            minus = empty
            dot_plus = empty
            dot_minus = empty


@triton.jit
def _rotate_flip_phase(
    cosine,
    sine,
    cos_phi,
    sin_phi,
    cos_2phi,
    sin_2phi,
    fp_r,
    fp_i,
    fm_r,
    fm_i,
    z_r,
    z_i,
):
    """One pool through a hard pulse named by its flip angle and phase.

    Pulled out of the kernel body so a second pool can take the same rotation:
    a chemical shift moves where a pool precesses, not what a pulse does to it.
    """
    cosine_half_sq = 0.5 * (1.0 + cosine)
    sine_half_sq = 0.5 * (1.0 - cosine)

    rotated_pr = cosine_half_sq * fp_r
    rotated_pr += sine_half_sq * (cos_2phi * fm_r - sin_2phi * fm_i)
    rotated_pr += sine * (sin_phi * z_r + cos_phi * z_i)
    rotated_pi = cosine_half_sq * fp_i
    rotated_pi += sine_half_sq * (sin_2phi * fm_r + cos_2phi * fm_i)
    rotated_pi += sine * (sin_phi * z_i - cos_phi * z_r)

    rotated_mr = sine_half_sq * (cos_2phi * fp_r + sin_2phi * fp_i)
    rotated_mr += cosine_half_sq * fm_r
    rotated_mr += sine * (sin_phi * z_r - cos_phi * z_i)
    rotated_mi = sine_half_sq * (-sin_2phi * fp_r + cos_2phi * fp_i)
    rotated_mi += cosine_half_sq * fm_i
    rotated_mi += sine * (cos_phi * z_r + sin_phi * z_i)

    rotated_zr = -0.5 * sine * (sin_phi * fp_r - cos_phi * fp_i)
    rotated_zr += -0.5 * sine * (sin_phi * fm_r + cos_phi * fm_i)
    rotated_zr += cosine * z_r
    rotated_zi = -0.5 * sine * (cos_phi * fp_r + sin_phi * fp_i)
    rotated_zi += 0.5 * sine * (cos_phi * fm_r - sin_phi * fm_i)
    rotated_zi += cosine * z_i
    return (
        rotated_pr,
        rotated_pi,
        rotated_mr,
        rotated_mi,
        rotated_zr,
        rotated_zi,
    )


@triton.jit(do_not_specialize=["profile_bins", "lineshape_bins"])
def _epg_kernel(
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
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
    pairs,
    pair_index,
    duration_row,
    pool_table,
    output_real,
    output_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shim_rows,
    shimmed: tl.constexpr,
    locations: tl.constexpr,
    profiled: tl.constexpr,
    profile_bins,
    dynamic: tl.constexpr,
    broadened: tl.constexpr,
    lineshape_bins,
    pools: tl.constexpr,
    narrow: tl.constexpr,
    tabulated: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    # A partial block carries lanes with no problem behind them, and they must
    # take no part in a reduction or a store.
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations

    empty = tl.zeros((problems, block_states), tl.float32)
    fplus_real = empty
    fplus_imag = empty
    fminus_real = empty
    fminus_imag = empty
    # A second pool holds its own share of the equilibrium. The semisolid one
    # carries longitudinal states alone -- nothing dephases it, so it reaches
    # the higher orders only through exchange with the free pool's -- while the
    # chemically exchanging one carries a transverse pair of its own.
    #
    # ``bound`` is whichever second pool the longitudinal step pairs the free
    # water with -- the semisolid one when it is the only one, the exchanging
    # one otherwise -- and ``semisolid`` is the third, which only a three-pool
    # run carries.
    atom_bound = 0.0
    atom_exchange = 0.0
    atom_r1_bound = 0.0
    atom_r2_bound = 0.0
    atom_shift = 0.0
    atom_semisolid = 0.0
    atom_semisolid_exchange = 0.0
    atom_r1_semisolid = 0.0
    if pools == 1:
        atom_bound = tl.load(bound_fraction + scalar_atom, mask=active_atom, other=0.0)
        atom_exchange = tl.load(
            bound_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        atom_r1_bound = 1000.0 / tl.load(
            t1_bound + scalar_atom, mask=active_atom, other=1.0
        )
    if pools == 2 or pools == 3:
        atom_bound = tl.load(pool_b_fraction + scalar_atom, mask=active_atom, other=0.0)
        atom_exchange = tl.load(
            pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        atom_r1_bound = 1000.0 / tl.load(
            t1_pool_b + scalar_atom, mask=active_atom, other=1.0
        )
        atom_r2_bound = 1000.0 / tl.load(
            t2_pool_b + scalar_atom, mask=active_atom, other=1.0
        )
        atom_shift = tl.load(pool_b_shift + scalar_atom, mask=active_atom, other=0.0)
    if pools == 3:
        atom_semisolid = tl.load(
            bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_semisolid_exchange = tl.load(
            bound_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        atom_r1_semisolid = 1000.0 / tl.load(
            t1_bound + scalar_atom, mask=active_atom, other=1.0
        )
    # A semisolid pool holds a share of the voxel without carrying any
    # transverse magnetization, so the 2x2 below is blind to it and the
    # exchange inside that 2x2 is not.
    atom_free = 1.0 - atom_bound - atom_semisolid
    longitudinal_real = empty + tl.where(state == 0, atom_free, 0.0)
    longitudinal_imag = empty
    bound_real = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
    bound_imag = empty
    semisolid_real = empty + tl.where(state == 0, atom_semisolid + 0.0, 0.0)
    semisolid_imag = empty
    bplus_real = empty
    bplus_imag = empty
    bminus_real = empty
    bminus_imag = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_b1_phase = 0.0
    atom_b0 = 0.0
    if off_axis:
        atom_b1_phase = tl.load(b1_phase + scalar_atom, mask=active_atom, other=0.0)
        atom_b0 = tl.load(b0 + scalar_atom, mask=active_atom, other=0.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    atom_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
    atom_flow = 0.0
    atom_washout = 0.0
    if moving:
        atom_velocity = tl.load(velocity + scalar_atom, mask=active_atom, other=0.0)
        atom_flow = atom_velocity * flow_scale
        atom_washout = tl.abs(atom_velocity) * washout_scale
    order = state.to(tl.float32)

    event_base = train * event_count
    for event in range(0, event_count):
        dt = _event_value(duration, event_base, event, active_atom, single_train)
        wout = 1.0
        if moving:
            wout = _washout(atom_washout, dt)
        e1 = tl.exp(-(1000.0 / atom_t1) * dt) * wout
        e2 = tl.exp(-(1000.0 / atom_t2) * dt) * wout
        damp_z = 1.0
        damp_t = 1.0
        if diffusing:
            damp_z, damp_t = _damping(atom_damping, dt, order)
        turn_z = 0.0
        turn_t = 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, dt, order)
        recovery = 1.0 - e1
        e1 = e1 * damp_z
        e2 = e2 * damp_t
        off_cos = 1.0
        off_sin = 0.0
        if off_axis or moving:
            # Flow winds the transverse states through the same rotation
            # off-resonance does, so the two phases add before either is taken.
            off_phase = -2.0 * 3.141592653589793 * atom_b0 * dt + turn_t
            off_cos = tl.cos(off_phase)
            off_sin = tl.sin(off_phase)
        if pools == 2 or pools == 3:
            # Both pools take the same off-resonance and the same per-order
            # damping; what separates them is the chemical shift, which the
            # exchange operator already carries.
            (
                x11r,
                x11i,
                x12r,
                x12i,
                x21r,
                x21i,
                x22r,
                x22i,
            ) = _two_pool_transverse_step(
                1000.0 / atom_t2,
                atom_r2_bound,
                atom_exchange,
                atom_bound,
                atom_free,
                atom_shift,
                dt,
                wout,
            )
            mixed_pr = (
                x11r * fplus_real
                - x11i * fplus_imag
                + x12r * bplus_real
                - x12i * bplus_imag
            )
            mixed_pi = (
                x11r * fplus_imag
                + x11i * fplus_real
                + x12r * bplus_imag
                + x12i * bplus_real
            )
            mixed_br = (
                x21r * fplus_real
                - x21i * fplus_imag
                + x22r * bplus_real
                - x22i * bplus_imag
            )
            mixed_bi = (
                x21r * fplus_imag
                + x21i * fplus_real
                + x22r * bplus_imag
                + x22i * bplus_real
            )
            # ``F-`` follows the conjugate of the operator entry by entry, not
            # its transpose: it is the conjugate state, and the map it takes is
            # the conjugate map.
            mixed_mr = (
                x11r * fminus_real
                + x11i * fminus_imag
                + x12r * bminus_real
                + x12i * bminus_imag
            )
            mixed_mi = (
                x11r * fminus_imag
                - x11i * fminus_real
                + x12r * bminus_imag
                - x12i * bminus_real
            )
            mixed_nr = (
                x21r * fminus_real
                + x21i * fminus_imag
                + x22r * bminus_real
                + x22i * bminus_imag
            )
            mixed_ni = (
                x21r * fminus_imag
                - x21i * fminus_real
                + x22r * bminus_imag
                - x22i * bminus_real
            )
            fplus_real = damp_t * (mixed_pr * off_cos - mixed_pi * off_sin)
            fplus_imag = damp_t * (mixed_pr * off_sin + mixed_pi * off_cos)
            bplus_real = damp_t * (mixed_br * off_cos - mixed_bi * off_sin)
            bplus_imag = damp_t * (mixed_br * off_sin + mixed_bi * off_cos)
            fminus_real = damp_t * (mixed_mr * off_cos + mixed_mi * off_sin)
            fminus_imag = damp_t * (-mixed_mr * off_sin + mixed_mi * off_cos)
            bminus_real = damp_t * (mixed_nr * off_cos + mixed_ni * off_sin)
            bminus_imag = damp_t * (-mixed_nr * off_sin + mixed_ni * off_cos)
        else:
            old_real = fplus_real
            fplus_real = e2 * (old_real * off_cos - fplus_imag * off_sin)
            fplus_imag = e2 * (old_real * off_sin + fplus_imag * off_cos)
            old_real = fminus_real
            fminus_real = e2 * (old_real * off_cos + fminus_imag * off_sin)
            fminus_imag = e2 * (-old_real * off_sin + fminus_imag * off_cos)
        # The longitudinal states carry a phase of their own, which nothing
        # else in the state machine gives them.
        turn_cos = 1.0
        turn_sin = 0.0
        if moving:
            turn_cos = tl.cos(turn_z)
            turn_sin = tl.sin(turn_z)
        if pools == 3:
            # Three pools mix through a 3x3 formed in double; every pool takes
            # the same per-order damping and flow phase, their order-n states
            # describing one dephasing configuration.
            if tabulated:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                ) = _three_pool_from_table(
                    pool_table,
                    tl.load(
                        duration_row + event_base + event,
                        mask=active_atom,
                        other=0,
                    ),
                    atom,
                    atom_count,
                    active_atom,
                    wout,
                    atom_free,
                    atom_bound,
                    atom_semisolid,
                )
            else:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                ) = _three_pool_step(
                    1000.0 / atom_t1,
                    atom_r1_bound,
                    atom_r1_semisolid,
                    atom_exchange,
                    atom_semisolid_exchange,
                    atom_bound,
                    atom_semisolid,
                    dt,
                    wout,
                    narrow,
                )
            free_real = (
                t11 * longitudinal_real + t12 * bound_real + t13 * semisolid_real
            )
            free_imag = (
                t11 * longitudinal_imag + t12 * bound_imag + t13 * semisolid_imag
            )
            held_real = (
                t21 * longitudinal_real + t22 * bound_real + t23 * semisolid_real
            )
            held_imag = (
                t21 * longitudinal_imag + t22 * bound_imag + t23 * semisolid_imag
            )
            stuck_real = (
                t31 * longitudinal_real + t32 * bound_real + t33 * semisolid_real
            )
            stuck_imag = (
                t31 * longitudinal_imag + t32 * bound_imag + t33 * semisolid_imag
            )
            longitudinal_real = damp_z * (free_real * turn_cos - free_imag * turn_sin)
            longitudinal_imag = damp_z * (free_real * turn_sin + free_imag * turn_cos)
            bound_real = damp_z * (held_real * turn_cos - held_imag * turn_sin)
            bound_imag = damp_z * (held_real * turn_sin + held_imag * turn_cos)
            semisolid_real = damp_z * (stuck_real * turn_cos - stuck_imag * turn_sin)
            semisolid_imag = damp_z * (stuck_real * turn_sin + stuck_imag * turn_cos)
            longitudinal_real += tl.where(state == 0, grow_free, 0.0)
            bound_real += tl.where(state == 0, grow_pool_b, 0.0)
            semisolid_real += tl.where(state == 0, grow_semisolid, 0.0)
        elif pools > 0:
            # The exchange operator is a property of the interval, not of a
            # dephasing order, so it is formed once and the per-order damping
            # multiplies it. Both pools take that damping and the flow phase:
            # their order-n states describe one dephasing configuration, and a
            # second pool has no diffusion coefficient of its own to damp by.
            e11, e12, e21, e22, grow_free, grow_bound = _two_pool_step(
                1000.0 / atom_t1, atom_r1_bound, atom_exchange, atom_bound, dt, wout
            )
            free_real = e11 * longitudinal_real + e12 * bound_real
            free_imag = e11 * longitudinal_imag + e12 * bound_imag
            held_real = e21 * longitudinal_real + e22 * bound_real
            held_imag = e21 * longitudinal_imag + e22 * bound_imag
            longitudinal_real = damp_z * (free_real * turn_cos - free_imag * turn_sin)
            longitudinal_imag = damp_z * (free_real * turn_sin + free_imag * turn_cos)
            bound_real = damp_z * (held_real * turn_cos - held_imag * turn_sin)
            bound_imag = damp_z * (held_real * turn_sin + held_imag * turn_cos)
            longitudinal_real += tl.where(state == 0, grow_free, 0.0)
            bound_real += tl.where(state == 0, grow_bound, 0.0)
        else:
            old_real = longitudinal_real
            longitudinal_real = e1 * (
                old_real * turn_cos - longitudinal_imag * turn_sin
            )
            longitudinal_imag = e1 * (
                old_real * turn_sin + longitudinal_imag * turn_cos
            )
            longitudinal_real += tl.where(state == 0, recovery, 0.0)

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fplus_real,
            fplus_imag,
            fminus_real,
            fminus_imag,
            state,
            state_mask,
            state_count,
        )
        fplus_real = tl.where(pre_shift, shifted_pr, fplus_real)
        fplus_imag = tl.where(pre_shift, shifted_pi, fplus_imag)
        fminus_real = tl.where(pre_shift, shifted_mr, fminus_real)
        fminus_imag = tl.where(pre_shift, shifted_mi, fminus_imag)
        if pools == 2 or pools == 3:
            b_pr, b_pi, b_mr, b_mi = _shift(
                bplus_real,
                bplus_imag,
                bminus_real,
                bminus_imag,
                state,
                state_mask,
                state_count,
            )
            bplus_real = tl.where(pre_shift, b_pr, bplus_real)
            bplus_imag = tl.where(pre_shift, b_pi, bplus_imag)
            bminus_real = tl.where(pre_shift, b_mr, bminus_real)
            bminus_imag = tl.where(pre_shift, b_mi, bminus_imag)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        longitudinal_real = tl.where(
            invert, -atom_inversion * longitudinal_real, longitudinal_real
        )
        longitudinal_imag = tl.where(
            invert, -atom_inversion * longitudinal_imag, longitudinal_imag
        )
        if pools == 2 or pools == 3:
            # A chemically exchanging pool is free water and turns over like
            # any other; a semisolid one is saturated instead, which its own
            # saturation term already carries.
            bound_real = tl.where(invert, -atom_inversion * bound_real, bound_real)
            bound_imag = tl.where(invert, -atom_inversion * bound_imag, bound_imag)

        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = 1.0
            if transmit:
                atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            if off_axis:
                atom_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )
        alpha = (
            _event_value(flip, event_base, event, active_atom, single_train) * atom_b1
        )
        phi = (
            _event_value(phase, event_base, event, active_atom, single_train)
            + atom_b1_phase
        )
        if profiled or dynamic:
            # Either pair is built at zero RF phase, which turns the rotation
            # axis and so reaches ``b`` alone.
            if dynamic:
                # Already integrated at this pulse's own flip, so the flip is
                # inside the pair rather than read against it.
                pair = _dynamic_pair_at(
                    pairs,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                )
            else:
                pair = _profile_pair(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha,
                    profile_bins,
                    profile_step,
                )
            turn_r = tl.cos(phi)
            turn_i = -tl.sin(phi)
            spun_br = pair[2] * turn_r - pair[3] * turn_i
            spun_bi = pair[2] * turn_i + pair[3] * turn_r
            (shaped_pr, shaped_pi, shaped_mr, shaped_mi, shaped_zr, shaped_zi) = (
                _rotate_spinor(
                    pair[0],
                    pair[1],
                    spun_br,
                    spun_bi,
                    fplus_real,
                    fplus_imag,
                    fminus_real,
                    fminus_imag,
                    longitudinal_real,
                    longitudinal_imag,
                )
            )
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        cos_phi = tl.cos(phi)
        sin_phi = tl.sin(phi)
        cos_2phi = tl.cos(2.0 * phi)
        sin_2phi = tl.sin(2.0 * phi)

        (
            rotated_pr,
            rotated_pi,
            rotated_mr,
            rotated_mi,
            rotated_zr,
            rotated_zi,
        ) = _rotate_flip_phase(
            cosine,
            sine,
            cos_phi,
            sin_phi,
            cos_2phi,
            sin_2phi,
            fplus_real,
            fplus_imag,
            fminus_real,
            fminus_imag,
            longitudinal_real,
            longitudinal_imag,
        )

        if pools == 2 or pools == 3:
            (
                b_rot_pr,
                b_rot_pi,
                b_rot_mr,
                b_rot_mi,
                b_rot_zr,
                b_rot_zi,
            ) = _rotate_flip_phase(
                cosine,
                sine,
                cos_phi,
                sin_phi,
                cos_2phi,
                sin_2phi,
                bplus_real,
                bplus_imag,
                bminus_real,
                bminus_imag,
                bound_real,
                bound_imag,
            )
            if profiled or dynamic:
                # The same pulse, the same rotation: a chemical shift moves
                # where a pool precesses, not what a pulse does to it.
                (
                    b_rot_pr,
                    b_rot_pi,
                    b_rot_mr,
                    b_rot_mi,
                    b_rot_zr,
                    b_rot_zi,
                ) = _rotate_spinor(
                    pair[0],
                    pair[1],
                    spun_br,
                    spun_bi,
                    bplus_real,
                    bplus_imag,
                    bminus_real,
                    bminus_imag,
                    bound_real,
                    bound_imag,
                )
        if profiled or dynamic:
            rotated_pr = shaped_pr
            rotated_pi = shaped_pi
            rotated_mr = shaped_mr
            rotated_mi = shaped_mi
            rotated_zr = shaped_zr
            rotated_zi = shaped_zi

        rotate = is_rf & ~is_inversion
        if pools == 2 or pools == 3:
            bplus_real = tl.where(rotate, b_rot_pr, bplus_real)
            bplus_imag = tl.where(rotate, b_rot_pi, bplus_imag)
            bminus_real = tl.where(rotate, b_rot_mr, bminus_real)
            bminus_imag = tl.where(rotate, b_rot_mi, bminus_imag)
            bound_real = tl.where(rotate, b_rot_zr, bound_real)
            bound_imag = tl.where(rotate, b_rot_zi, bound_imag)
        if pools == 1 or pools == 3:
            # The semisolid pool absorbs the power the pulse deposits, so it
            # reads the bare flip the transmit field gives the voxel -- not the
            # slice-shaped rotation the free pool takes from the table.
            offset = tl.load(rf_frequency + event) - atom_b0
            absorbed = tl.exp(
                tl.load(saturation + event)
                * alpha
                * alpha
                * _lineshape_at(lineshape, offset, lineshape_bins, lineshape_step)
            )
            if pools == 1:
                bound_real = tl.where(rotate, absorbed * bound_real, bound_real)
                bound_imag = tl.where(rotate, absorbed * bound_imag, bound_imag)
            else:
                semisolid_real = tl.where(
                    rotate, absorbed * semisolid_real, semisolid_real
                )
                semisolid_imag = tl.where(
                    rotate, absorbed * semisolid_imag, semisolid_imag
                )
        fplus_real = tl.where(rotate, rotated_pr, fplus_real)
        fplus_imag = tl.where(rotate, rotated_pi, fplus_imag)
        fminus_real = tl.where(rotate, rotated_mr, fminus_real)
        fminus_imag = tl.where(rotate, rotated_mi, fminus_imag)
        longitudinal_real = tl.where(rotate, rotated_zr, longitudinal_real)
        longitudinal_imag = tl.where(rotate, rotated_zi, longitudinal_imag)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        adc_phase = _event_value(phase, event_base, event, active_atom, single_train)
        adc_cos = tl.cos(adc_phase)
        adc_sin = tl.sin(adc_phase)
        # A coil sees the whole voxel, so what it records is the sum over
        # pools; each pool's share is already in its own state.
        read_real = fplus_real
        read_imag = fplus_imag
        if pools == 2 or pools == 3:
            read_real = fplus_real + bplus_real
            read_imag = fplus_imag + bplus_imag
        signal_real = atom_m0 * (read_real * adc_cos + read_imag * adc_sin)
        signal_imag = atom_m0 * (read_imag * adc_cos - read_real * adc_sin)
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        tl.store(output_real + output_offset + state, signal_real, mask=output_mask)
        tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fplus_real,
            fplus_imag,
            fminus_real,
            fminus_imag,
            state,
            state_mask,
            state_count,
        )
        fplus_real = tl.where(do_shift, shifted_pr, fplus_real)
        fplus_imag = tl.where(do_shift, shifted_pi, fplus_imag)
        fminus_real = tl.where(do_shift, shifted_mr, fminus_real)
        fminus_imag = tl.where(do_shift, shifted_mi, fminus_imag)
        spoil = (event_action & 8) != 0
        fplus_real = tl.where(spoil, 0.0, fplus_real)
        fplus_imag = tl.where(spoil, 0.0, fplus_imag)
        fminus_real = tl.where(spoil, 0.0, fminus_real)
        fminus_imag = tl.where(spoil, 0.0, fminus_imag)
        if pools == 2 or pools == 3:
            b_pr, b_pi, b_mr, b_mi = _shift(
                bplus_real,
                bplus_imag,
                bminus_real,
                bminus_imag,
                state,
                state_mask,
                state_count,
            )
            bplus_real = tl.where(spoil, 0.0, tl.where(do_shift, b_pr, bplus_real))
            bplus_imag = tl.where(spoil, 0.0, tl.where(do_shift, b_pi, bplus_imag))
            bminus_real = tl.where(spoil, 0.0, tl.where(do_shift, b_mr, bminus_real))
            bminus_imag = tl.where(spoil, 0.0, tl.where(do_shift, b_mi, bminus_imag))


@triton.jit
def _rotate_flip_phase_jvp(
    cosine,
    dcosine,
    sine,
    dsine,
    cos_phi,
    dcos_phi,
    sin_phi,
    dsin_phi,
    cos_2phi,
    dcos_2phi,
    sin_2phi,
    dsin_2phi,
    fpr,
    fpi,
    fmr,
    fmi,
    zr,
    zi,
    dfpr,
    dfpi,
    dfmr,
    dfmi,
    dzr,
    dzi,
):
    """One pool through a hard pulse, carried alongside a tangent.

    Pulled out of the kernel body so a second pool can take the same
    rotation: a chemical shift moves where a pool precesses, not what a
    pulse does to it.
    """
    ch = 0.5 * (1.0 + cosine)
    sh = 0.5 * (1.0 - cosine)
    dch = 0.5 * dcosine
    dsh = -0.5 * dcosine
    pr_a = cos_2phi * fmr - sin_2phi * fmi
    dpr_a = dcos_2phi * fmr + cos_2phi * dfmr
    dpr_a -= dsin_2phi * fmi + sin_2phi * dfmi
    pr_b = sin_phi * zr + cos_phi * zi
    dpr_b = dsin_phi * zr + sin_phi * dzr + dcos_phi * zi + cos_phi * dzi
    rotated_pr = ch * fpr + sh * pr_a + sine * pr_b
    rotated_dpr = dch * fpr + ch * dfpr + dsh * pr_a + sh * dpr_a
    rotated_dpr += dsine * pr_b + sine * dpr_b

    pi_a = sin_2phi * fmr + cos_2phi * fmi
    dpi_a = dsin_2phi * fmr + sin_2phi * dfmr
    dpi_a += dcos_2phi * fmi + cos_2phi * dfmi
    pi_b = sin_phi * zi - cos_phi * zr
    dpi_b = dsin_phi * zi + sin_phi * dzi - dcos_phi * zr - cos_phi * dzr
    rotated_pi = ch * fpi + sh * pi_a + sine * pi_b
    rotated_dpi = dch * fpi + ch * dfpi + dsh * pi_a + sh * dpi_a
    rotated_dpi += dsine * pi_b + sine * dpi_b

    mr_a = cos_2phi * fpr + sin_2phi * fpi
    dmr_a = dcos_2phi * fpr + cos_2phi * dfpr
    dmr_a += dsin_2phi * fpi + sin_2phi * dfpi
    mr_b = sin_phi * zr - cos_phi * zi
    dmr_b = dsin_phi * zr + sin_phi * dzr - dcos_phi * zi - cos_phi * dzi
    rotated_mr = sh * mr_a + ch * fmr + sine * mr_b
    rotated_dmr = dsh * mr_a + sh * dmr_a + dch * fmr + ch * dfmr
    rotated_dmr += dsine * mr_b + sine * dmr_b

    mi_a = -sin_2phi * fpr + cos_2phi * fpi
    dmi_a = -dsin_2phi * fpr - sin_2phi * dfpr
    dmi_a += dcos_2phi * fpi + cos_2phi * dfpi
    mi_b = cos_phi * zr + sin_phi * zi
    dmi_b = dcos_phi * zr + cos_phi * dzr + dsin_phi * zi + sin_phi * dzi
    rotated_mi = sh * mi_a + ch * fmi + sine * mi_b
    rotated_dmi = dsh * mi_a + sh * dmi_a + dch * fmi + ch * dfmi
    rotated_dmi += dsine * mi_b + sine * dmi_b

    zr_a = sin_phi * fpr - cos_phi * fpi
    dzr_a = dsin_phi * fpr + sin_phi * dfpr - dcos_phi * fpi - cos_phi * dfpi
    zr_b = sin_phi * fmr + cos_phi * fmi
    dzr_b = dsin_phi * fmr + sin_phi * dfmr + dcos_phi * fmi + cos_phi * dfmi
    rotated_zr = -0.5 * sine * zr_a - 0.5 * sine * zr_b + cosine * zr
    rotated_dzr = -0.5 * (dsine * zr_a + sine * dzr_a)
    rotated_dzr -= 0.5 * (dsine * zr_b + sine * dzr_b)
    rotated_dzr += dcosine * zr + cosine * dzr

    zi_a = cos_phi * fpr + sin_phi * fpi
    dzi_a = dcos_phi * fpr + cos_phi * dfpr + dsin_phi * fpi + sin_phi * dfpi
    zi_b = cos_phi * fmr - sin_phi * fmi
    dzi_b = dcos_phi * fmr + cos_phi * dfmr - dsin_phi * fmi - sin_phi * dfmi
    rotated_zi = -0.5 * sine * zi_a + 0.5 * sine * zi_b + cosine * zi
    rotated_dzi = -0.5 * (dsine * zi_a + sine * dzi_a)
    rotated_dzi += 0.5 * (dsine * zi_b + sine * dzi_b)
    rotated_dzi += dcosine * zi + cosine * dzi

    return (
        rotated_pr,
        rotated_pi,
        rotated_mr,
        rotated_mi,
        rotated_zr,
        rotated_zi,
        rotated_dpr,
        rotated_dpi,
        rotated_dmr,
        rotated_dmi,
        rotated_dzr,
        rotated_dzi,
    )


@triton.jit(do_not_specialize=["profile_bins", "lineshape_bins"])
def _epg_jvp_kernel(
    t1,
    t2,
    m0,
    b1,
    b1_phase,
    b0,
    inversion_efficiency,
    diffusion,
    velocity,
    # The bound pool's three. A bound pool is outside the real subspace this
    # kernel stands for, so the dispatch never sends one here; the pointers
    # hold the ABI's shape.
    bound_fraction,
    exchange_rate,
    t1_bound,
    # The chemically exchanging pool's five, on the same terms.
    pool_b_fraction,
    pool_b_exchange,
    t1_pool_b,
    t2_pool_b,
    pool_b_shift,
    duration,
    kind,
    flip,
    phase,
    action,
    output_index,
    shim_index,
    tangent_t1,
    tangent_t2,
    tangent_m0,
    tangent_b1,
    tangent_b1_phase,
    tangent_b0,
    tangent_inversion_efficiency,
    tangent_diffusion,
    tangent_velocity,
    tangent_bound_fraction,
    tangent_exchange_rate,
    tangent_t1_bound,
    tangent_pool_b_fraction,
    tangent_pool_b_exchange,
    tangent_t1_pool_b,
    tangent_t2_pool_b,
    tangent_pool_b_shift,
    tangent_duration,
    tangent_flip,
    tangent_phase,
    saturation,
    rf_frequency,
    profile,
    profile_index,
    lineshape,
    pairs,
    pair_index,
    pair_direction,
    duration_row,
    pool_table,
    output_real,
    output_imag,
    atom_count,
    train_count,
    event_count,
    output_count,
    flow_scale,
    washout_scale,
    profile_step,
    lineshape_step,
    state_count: tl.constexpr,
    single_train: tl.constexpr,
    atom_stride: tl.constexpr,
    shim_rows,
    shimmed: tl.constexpr,
    locations: tl.constexpr,
    profiled: tl.constexpr,
    profile_bins,
    dynamic: tl.constexpr,
    broadened: tl.constexpr,
    lineshape_bins,
    pools: tl.constexpr,
    narrow: tl.constexpr,
    tabulated: tl.constexpr,
    off_axis: tl.constexpr,
    moving: tl.constexpr,
    diffusing: tl.constexpr,
    transmit: tl.constexpr,
    density: tl.constexpr,
    inverting: tl.constexpr,
    block_states: tl.constexpr,
    problems: tl.constexpr,
):
    problem = tl.program_id(0) * problems + tl.arange(0, problems)[:, None]
    state = tl.arange(0, block_states)[None, :]
    active_atom = problem < train_count * atom_count
    # A partial block carries lanes with no problem behind them, and they must
    # take no part in a reduction or a store.
    state_mask = (state < state_count) & active_atom
    atom = problem % atom_count
    # A property given as one value for the whole tissue is read at one
    # address by every voxel, which is a stride of zero through it.
    scalar_atom = atom * atom_stride
    train = problem // atom_count
    # Voxels are spread over the slice voxel-major, so a voxel's place along
    # the slice is its index modulo the profile's width. One pulse shape holds
    # that many consecutive rows, and the event says which shape it drives.
    location = atom % locations

    empty = tl.zeros((problems, block_states), tl.float32)
    fpr = empty
    fpi = empty
    fmr = empty
    fmi = empty
    # Equilibrium is split between the pools, so a direction along the bound
    # fraction moves magnetization from one to the other before a single event
    # has run.
    atom_bound = 0.0
    d_bound = 0.0
    atom_exchange = 0.0
    d_exchange = 0.0
    atom_r1_bound = 0.0
    d_r1_bound = 0.0
    atom_r2_bound = 0.0
    d_r2_bound = 0.0
    atom_shift = 0.0
    d_shift = 0.0
    atom_semisolid = 0.0
    d_semisolid = 0.0
    atom_semisolid_exchange = 0.0
    d_semisolid_exchange = 0.0
    atom_r1_semisolid = 0.0
    d_r1_semisolid = 0.0
    if pools == 1:
        atom_bound = tl.load(bound_fraction + scalar_atom, mask=active_atom, other=0.0)
        d_bound = tl.load(
            tangent_bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        d_exchange = tl.load(
            tangent_exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        held_t1 = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        atom_r1_bound = 1000.0 / held_t1
        d_r1_bound = (
            -1000.0
            * tl.load(tangent_t1_bound + scalar_atom, mask=active_atom, other=0.0)
            / (held_t1 * held_t1)
        )
    if pools == 2 or pools == 3:
        atom_bound = tl.load(pool_b_fraction + scalar_atom, mask=active_atom, other=0.0)
        d_bound = tl.load(
            tangent_pool_b_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_exchange = tl.load(
            pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        d_exchange = tl.load(
            tangent_pool_b_exchange + scalar_atom, mask=active_atom, other=0.0
        )
        held_t1 = tl.load(t1_pool_b + scalar_atom, mask=active_atom, other=1.0)
        atom_r1_bound = 1000.0 / held_t1
        d_r1_bound = (
            -1000.0
            * tl.load(tangent_t1_pool_b + scalar_atom, mask=active_atom, other=0.0)
            / (held_t1 * held_t1)
        )
        held_t2 = tl.load(t2_pool_b + scalar_atom, mask=active_atom, other=1.0)
        atom_r2_bound = 1000.0 / held_t2
        d_r2_bound = (
            -1000.0
            * tl.load(tangent_t2_pool_b + scalar_atom, mask=active_atom, other=0.0)
            / (held_t2 * held_t2)
        )
        atom_shift = tl.load(pool_b_shift + scalar_atom, mask=active_atom, other=0.0)
        d_shift = tl.load(
            tangent_pool_b_shift + scalar_atom, mask=active_atom, other=0.0
        )
    if pools == 3:
        atom_semisolid = tl.load(
            bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        d_semisolid = tl.load(
            tangent_bound_fraction + scalar_atom, mask=active_atom, other=0.0
        )
        atom_semisolid_exchange = tl.load(
            exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        d_semisolid_exchange = tl.load(
            tangent_exchange_rate + scalar_atom, mask=active_atom, other=0.0
        )
        held_semisolid = tl.load(t1_bound + scalar_atom, mask=active_atom, other=1.0)
        atom_r1_semisolid = 1000.0 / held_semisolid
        d_r1_semisolid = (
            -1000.0
            * tl.load(tangent_t1_bound + scalar_atom, mask=active_atom, other=0.0)
            / (held_semisolid * held_semisolid)
        )
    atom_free = 1.0 - atom_bound - atom_semisolid
    d_free = -d_bound - d_semisolid
    zr = empty + tl.where(state == 0, atom_free, 0.0)
    zi = empty
    br = empty + tl.where(state == 0, atom_bound + 0.0, 0.0)
    bi = empty
    cr = empty + tl.where(state == 0, atom_semisolid + 0.0, 0.0)
    ci = empty
    dcr = empty + tl.where(state == 0, d_semisolid + 0.0, 0.0)
    dci = empty
    dfpr = empty
    dfpi = empty
    dfmr = empty
    dfmi = empty
    dzr = empty + tl.where(state == 0, -d_bound - d_semisolid, 0.0)
    dzi = empty
    dbr = empty + tl.where(state == 0, d_bound + 0.0, 0.0)
    dbi = empty
    bpr = empty
    bpi = empty
    bmr = empty
    bmi = empty
    dbpr = empty
    dbpi = empty
    dbmr = empty
    dbmi = empty

    atom_t1 = tl.load(t1 + atom, mask=active_atom, other=1.0)
    atom_t2 = tl.load(t2 + atom, mask=active_atom, other=1.0)
    atom_m0 = 1.0
    if density:
        atom_m0 = tl.load(m0 + scalar_atom, mask=active_atom, other=0.0)
    atom_b1 = 1.0
    if transmit:
        atom_b1 = tl.load(b1 + scalar_atom, mask=active_atom, other=1.0)
    atom_b1_phase = 0.0
    atom_b0 = 0.0
    if off_axis:
        atom_b1_phase = tl.load(b1_phase + scalar_atom, mask=active_atom, other=0.0)
        atom_b0 = tl.load(b0 + scalar_atom, mask=active_atom, other=0.0)
    atom_inversion = 1.0
    if inverting:
        atom_inversion = tl.load(
            inversion_efficiency + scalar_atom, mask=active_atom, other=1.0
        )
    atom_damping = 0.0
    d_damping = 0.0
    if diffusing:
        atom_damping = tl.load(diffusion + scalar_atom, mask=active_atom, other=0.0)
        d_damping = tl.load(
            tangent_diffusion + scalar_atom, mask=active_atom, other=0.0
        )
    atom_flow = 0.0
    d_flow = 0.0
    atom_washout = 0.0
    d_washout = 0.0
    if moving:
        atom_velocity = tl.load(velocity + scalar_atom, mask=active_atom, other=0.0)
        d_velocity = tl.load(
            tangent_velocity + scalar_atom, mask=active_atom, other=0.0
        )
        atom_flow = atom_velocity * flow_scale
        d_flow = d_velocity * flow_scale
        # |v| has no derivative at the origin, so a still voxel contributes
        # none.
        direction = (atom_velocity > 0.0).to(tl.float32) - (atom_velocity < 0.0).to(
            tl.float32
        )
        atom_washout = tl.abs(atom_velocity) * washout_scale
        d_washout = direction * d_velocity * washout_scale
    order = state.to(tl.float32)
    dt1 = tl.load(tangent_t1 + atom, mask=active_atom, other=0.0)
    dt2 = tl.load(tangent_t2 + atom, mask=active_atom, other=0.0)
    dm0 = 0.0
    if density:
        dm0 = tl.load(tangent_m0 + scalar_atom, mask=active_atom, other=0.0)
    db1 = 0.0
    if transmit:
        db1 = tl.load(tangent_b1 + scalar_atom, mask=active_atom, other=0.0)
    db1_phase = 0.0
    db0 = 0.0
    if off_axis:
        db1_phase = tl.load(tangent_b1_phase + scalar_atom, mask=active_atom, other=0.0)
        db0 = tl.load(tangent_b0 + scalar_atom, mask=active_atom, other=0.0)
    dinversion = 0.0
    if inverting:
        dinversion = tl.load(
            tangent_inversion_efficiency + scalar_atom, mask=active_atom, other=0.0
        )

    event_base = train * event_count
    for event in range(0, event_count):
        event_dt = _event_value(duration, event_base, event, active_atom, single_train)
        ddt = _event_value(
            tangent_duration, event_base, event, active_atom, single_train
        )
        r1 = 1000.0 / atom_t1
        r2 = 1000.0 / atom_t2
        wout = 1.0
        dwout = 0.0
        if moving:
            wout, dwout = _washout_jvp(atom_washout, d_washout, event_dt, ddt)
        dry1 = tl.exp(-r1 * event_dt)
        dry2 = tl.exp(-r2 * event_dt)
        e1 = dry1 * wout
        e2 = dry2 * wout
        de1 = (
            e1 * (1000.0 * event_dt * dt1 / (atom_t1 * atom_t1) - r1 * ddt)
            + dry1 * dwout
        )
        de2 = (
            e2 * (1000.0 * event_dt * dt2 / (atom_t2 * atom_t2) - r2 * ddt)
            + dry2 * dwout
        )
        damp_z = 1.0
        ddamp_z = 0.0
        damp_t = 1.0
        ddamp_t = 0.0
        if diffusing:
            damp_z, ddamp_z, damp_t, ddamp_t = _damping_jvp(
                atom_damping, d_damping, event_dt, ddt, order
            )
        # Order zero is undamped, so the recovery term keeps the bare factor.
        recovery, drecovery = 1.0 - e1, -de1
        de1 = de1 * damp_z + e1 * ddamp_z
        e1 = e1 * damp_z
        de2 = de2 * damp_t + e2 * ddamp_t
        e2 = e2 * damp_t
        turn_z = 0.0
        turn_t = 0.0
        dturn_z = 0.0
        dturn_t = 0.0
        if moving:
            turn_z, turn_t = _flow(atom_flow, event_dt, order)
            d_turn = d_flow * event_dt + atom_flow * ddt
            dturn_z = -order * d_turn
            dturn_t = -(order + 0.5) * d_turn
        off_cos = 1.0
        off_sin = 0.0
        doff_cos = 0.0
        doff_sin = 0.0
        if off_axis or moving:
            # Flow winds the transverse states through the same rotation
            # off-resonance does, so the two phases add before either is taken.
            off_phase = -2.0 * 3.141592653589793 * atom_b0 * event_dt + turn_t
            doff_phase = (
                -2.0 * 3.141592653589793 * (db0 * event_dt + atom_b0 * ddt) + dturn_t
            )
            off_cos = tl.cos(off_phase)
            off_sin = tl.sin(off_phase)
            doff_cos = -off_sin * doff_phase
            doff_sin = off_cos * doff_phase

        if pools == 2 or pools == 3:
            # Both pools take the same off-resonance and per-order damping;
            # what separates them is the chemical shift, which the exchange
            # operator already carries.
            (
                x11r,
                x11i,
                x12r,
                x12i,
                x21r,
                x21i,
                x22r,
                x22i,
                d11r,
                d11i,
                d12r,
                d12i,
                d21r,
                d21i,
                d22r,
                d22i,
            ) = _two_pool_transverse_step_jvp(
                r2,
                -1000.0 * dt2 / (atom_t2 * atom_t2),
                atom_r2_bound,
                d_r2_bound,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_bound,
                atom_free,
                d_free,
                atom_shift,
                d_shift,
                event_dt,
                ddt,
                wout,
                dwout,
            )
            mix_pr = x11r * fpr - x11i * fpi + x12r * bpr - x12i * bpi
            mix_pi = x11r * fpi + x11i * fpr + x12r * bpi + x12i * bpr
            dmix_pr = (
                d11r * fpr
                + x11r * dfpr
                - d11i * fpi
                - x11i * dfpi
                + d12r * bpr
                + x12r * dbpr
                - d12i * bpi
                - x12i * dbpi
            )
            dmix_pi = (
                d11r * fpi
                + x11r * dfpi
                + d11i * fpr
                + x11i * dfpr
                + d12r * bpi
                + x12r * dbpi
                + d12i * bpr
                + x12i * dbpr
            )
            mix_br = x21r * fpr - x21i * fpi + x22r * bpr - x22i * bpi
            mix_bi = x21r * fpi + x21i * fpr + x22r * bpi + x22i * bpr
            dmix_br = (
                d21r * fpr
                + x21r * dfpr
                - d21i * fpi
                - x21i * dfpi
                + d22r * bpr
                + x22r * dbpr
                - d22i * bpi
                - x22i * dbpi
            )
            dmix_bi = (
                d21r * fpi
                + x21r * dfpi
                + d21i * fpr
                + x21i * dfpr
                + d22r * bpi
                + x22r * dbpi
                + d22i * bpr
                + x22i * dbpr
            )
            # ``F-`` follows the conjugate of the operator entry by entry.
            mix_mr = x11r * fmr + x11i * fmi + x12r * bmr + x12i * bmi
            mix_mi = x11r * fmi - x11i * fmr + x12r * bmi - x12i * bmr
            dmix_mr = (
                d11r * fmr
                + x11r * dfmr
                + d11i * fmi
                + x11i * dfmi
                + d12r * bmr
                + x12r * dbmr
                + d12i * bmi
                + x12i * dbmi
            )
            dmix_mi = (
                d11r * fmi
                + x11r * dfmi
                - d11i * fmr
                - x11i * dfmr
                + d12r * bmi
                + x12r * dbmi
                - d12i * bmr
                - x12i * dbmr
            )
            mix_nr = x21r * fmr + x21i * fmi + x22r * bmr + x22i * bmi
            mix_ni = x21r * fmi - x21i * fmr + x22r * bmi - x22i * bmr
            dmix_nr = (
                d21r * fmr
                + x21r * dfmr
                + d21i * fmi
                + x21i * dfmi
                + d22r * bmr
                + x22r * dbmr
                + d22i * bmi
                + x22i * dbmi
            )
            dmix_ni = (
                d21r * fmi
                + x21r * dfmi
                - d21i * fmr
                - x21i * dfmr
                + d22r * bmi
                + x22r * dbmi
                - d22i * bmr
                - x22i * dbmr
            )
            # The damping and off-resonance both pools share, applied after.
            carry_r = damp_t * off_cos
            carry_i = damp_t * off_sin
            dcarry_r = ddamp_t * off_cos + damp_t * doff_cos
            dcarry_i = ddamp_t * off_sin + damp_t * doff_sin
            fpr = mix_pr * carry_r - mix_pi * carry_i
            fpi = mix_pr * carry_i + mix_pi * carry_r
            dfpr = (
                dmix_pr * carry_r
                + mix_pr * dcarry_r
                - dmix_pi * carry_i
                - mix_pi * dcarry_i
            )
            dfpi = (
                dmix_pr * carry_i
                + mix_pr * dcarry_i
                + dmix_pi * carry_r
                + mix_pi * dcarry_r
            )
            bpr = mix_br * carry_r - mix_bi * carry_i
            bpi = mix_br * carry_i + mix_bi * carry_r
            dbpr = (
                dmix_br * carry_r
                + mix_br * dcarry_r
                - dmix_bi * carry_i
                - mix_bi * dcarry_i
            )
            dbpi = (
                dmix_br * carry_i
                + mix_br * dcarry_i
                + dmix_bi * carry_r
                + mix_bi * dcarry_r
            )
            fmr = mix_mr * carry_r + mix_mi * carry_i
            fmi = -mix_mr * carry_i + mix_mi * carry_r
            dfmr = (
                dmix_mr * carry_r
                + mix_mr * dcarry_r
                + dmix_mi * carry_i
                + mix_mi * dcarry_i
            )
            dfmi = (
                -dmix_mr * carry_i
                - mix_mr * dcarry_i
                + dmix_mi * carry_r
                + mix_mi * dcarry_r
            )
            bmr = mix_nr * carry_r + mix_ni * carry_i
            bmi = -mix_nr * carry_i + mix_ni * carry_r
            dbmr = (
                dmix_nr * carry_r
                + mix_nr * dcarry_r
                + dmix_ni * carry_i
                + mix_ni * dcarry_i
            )
            dbmi = (
                -dmix_nr * carry_i
                - mix_nr * dcarry_i
                + dmix_ni * carry_r
                + mix_ni * dcarry_r
            )
        else:
            old_fpr = fpr
            old_fpi = fpi
            old_dfpr = dfpr
            old_dfpi = dfpi
            fpr = e2 * (old_fpr * off_cos - old_fpi * off_sin)
            fpi = e2 * (old_fpr * off_sin + old_fpi * off_cos)
            dfpr = de2 * (old_fpr * off_cos - old_fpi * off_sin)
            dfpr += e2 * (
                old_dfpr * off_cos
                + old_fpr * doff_cos
                - old_dfpi * off_sin
                - old_fpi * doff_sin
            )
            dfpi = de2 * (old_fpr * off_sin + old_fpi * off_cos)
            dfpi += e2 * (
                old_dfpr * off_sin
                + old_fpr * doff_sin
                + old_dfpi * off_cos
                + old_fpi * doff_cos
            )

            old_fmr = fmr
            old_fmi = fmi
            old_dfmr = dfmr
            old_dfmi = dfmi
            fmr = e2 * (old_fmr * off_cos + old_fmi * off_sin)
            fmi = e2 * (-old_fmr * off_sin + old_fmi * off_cos)
            dfmr = de2 * (old_fmr * off_cos + old_fmi * off_sin)
            dfmr += e2 * (
                old_dfmr * off_cos
                + old_fmr * doff_cos
                + old_dfmi * off_sin
                + old_fmi * doff_sin
            )
            dfmi = de2 * (-old_fmr * off_sin + old_fmi * off_cos)
            dfmi += e2 * (
                -old_dfmr * off_sin
                - old_fmr * doff_sin
                + old_dfmi * off_cos
                + old_fmi * doff_cos
            )

        # The longitudinal states carry a phase of their own, which nothing
        # else in the state machine gives them.
        turn_cos = 1.0
        turn_sin = 0.0
        dturn_cos = 0.0
        dturn_sin = 0.0
        if moving:
            turn_cos = tl.cos(turn_z)
            turn_sin = tl.sin(turn_z)
            dturn_cos = -turn_sin * dturn_z
            dturn_sin = turn_cos * dturn_z
        old_zr = zr
        old_zi = zi
        old_dzr = dzr
        old_dzi = dzi
        spun_zr = old_zr * turn_cos - old_zi * turn_sin
        spun_zi = old_zr * turn_sin + old_zi * turn_cos
        dspun_zr = (
            old_dzr * turn_cos
            + old_zr * dturn_cos
            - old_dzi * turn_sin
            - old_zi * dturn_sin
        )
        dspun_zi = (
            old_dzr * turn_sin
            + old_zr * dturn_sin
            + old_dzi * turn_cos
            + old_zi * dturn_cos
        )
        if pools == 3:
            # Three pools mix through a 3x3 formed in double, tangent and all:
            # a direction through an operator this ill-conditioned needs the
            # width as much as the value does.
            if tabulated:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_t11,
                    d_t12,
                    d_t13,
                    d_t21,
                    d_t22,
                    d_t23,
                    d_t31,
                    d_t32,
                    d_t33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_from_table_jvp(
                    pool_table,
                    tl.load(
                        duration_row + event_base + event,
                        mask=active_atom,
                        other=0,
                    ),
                    atom,
                    atom_count,
                    active_atom,
                    r1,
                    atom_r1_bound,
                    atom_r1_semisolid,
                    atom_exchange,
                    atom_semisolid_exchange,
                    atom_bound,
                    d_bound,
                    atom_semisolid,
                    d_semisolid,
                    ddt,
                    wout,
                    dwout,
                )
            else:
                (
                    t11,
                    t12,
                    t13,
                    t21,
                    t22,
                    t23,
                    t31,
                    t32,
                    t33,
                    grow_free,
                    grow_pool_b,
                    grow_semisolid,
                    d_t11,
                    d_t12,
                    d_t13,
                    d_t21,
                    d_t22,
                    d_t23,
                    d_t31,
                    d_t32,
                    d_t33,
                    d_grow_free,
                    d_grow_pool_b,
                    d_grow_semisolid,
                ) = _three_pool_step_jvp(
                    r1,
                    -1000.0 * dt1 / (atom_t1 * atom_t1),
                    atom_r1_bound,
                    d_r1_bound,
                    atom_r1_semisolid,
                    d_r1_semisolid,
                    atom_exchange,
                    d_exchange,
                    atom_semisolid_exchange,
                    d_semisolid_exchange,
                    atom_bound,
                    d_bound,
                    atom_semisolid,
                    d_semisolid,
                    event_dt,
                    ddt,
                    wout,
                    dwout,
                    narrow,
                )
            spun_hr = br * turn_cos - bi * turn_sin
            spun_hi = br * turn_sin + bi * turn_cos
            dspun_hr = dbr * turn_cos + br * dturn_cos - dbi * turn_sin - bi * dturn_sin
            dspun_hi = dbr * turn_sin + br * dturn_sin + dbi * turn_cos + bi * dturn_cos
            spun_cr = cr * turn_cos - ci * turn_sin
            spun_ci = cr * turn_sin + ci * turn_cos
            dspun_cr = dcr * turn_cos + cr * dturn_cos - dci * turn_sin - ci * dturn_sin
            dspun_ci = dcr * turn_sin + cr * dturn_sin + dci * turn_cos + ci * dturn_cos
            free_r = t11 * spun_zr + t12 * spun_hr + t13 * spun_cr
            free_i = t11 * spun_zi + t12 * spun_hi + t13 * spun_ci
            held_r = t21 * spun_zr + t22 * spun_hr + t23 * spun_cr
            held_i = t21 * spun_zi + t22 * spun_hi + t23 * spun_ci
            stuck_r = t31 * spun_zr + t32 * spun_hr + t33 * spun_cr
            stuck_i = t31 * spun_zi + t32 * spun_hi + t33 * spun_ci
            d_free_r = (
                d_t11 * spun_zr
                + t11 * dspun_zr
                + d_t12 * spun_hr
                + t12 * dspun_hr
                + d_t13 * spun_cr
                + t13 * dspun_cr
            )
            d_free_i = (
                d_t11 * spun_zi
                + t11 * dspun_zi
                + d_t12 * spun_hi
                + t12 * dspun_hi
                + d_t13 * spun_ci
                + t13 * dspun_ci
            )
            d_held_r = (
                d_t21 * spun_zr
                + t21 * dspun_zr
                + d_t22 * spun_hr
                + t22 * dspun_hr
                + d_t23 * spun_cr
                + t23 * dspun_cr
            )
            d_held_i = (
                d_t21 * spun_zi
                + t21 * dspun_zi
                + d_t22 * spun_hi
                + t22 * dspun_hi
                + d_t23 * spun_ci
                + t23 * dspun_ci
            )
            d_stuck_r = (
                d_t31 * spun_zr
                + t31 * dspun_zr
                + d_t32 * spun_hr
                + t32 * dspun_hr
                + d_t33 * spun_cr
                + t33 * dspun_cr
            )
            d_stuck_i = (
                d_t31 * spun_zi
                + t31 * dspun_zi
                + d_t32 * spun_hi
                + t32 * dspun_hi
                + d_t33 * spun_ci
                + t33 * dspun_ci
            )
            zr = damp_z * free_r + tl.where(state == 0, grow_free, 0.0)
            zi = damp_z * free_i
            dzr = (
                ddamp_z * free_r
                + damp_z * d_free_r
                + tl.where(state == 0, d_grow_free, 0.0)
            )
            dzi = ddamp_z * free_i + damp_z * d_free_i
            br = damp_z * held_r + tl.where(state == 0, grow_pool_b, 0.0)
            bi = damp_z * held_i
            dbr = (
                ddamp_z * held_r
                + damp_z * d_held_r
                + tl.where(state == 0, d_grow_pool_b, 0.0)
            )
            dbi = ddamp_z * held_i + damp_z * d_held_i
            cr = damp_z * stuck_r + tl.where(state == 0, grow_semisolid, 0.0)
            ci = damp_z * stuck_i
            dcr = (
                ddamp_z * stuck_r
                + damp_z * d_stuck_r
                + tl.where(state == 0, d_grow_semisolid, 0.0)
            )
            dci = ddamp_z * stuck_i + damp_z * d_stuck_i
        elif pools > 0:
            # The exchange operator belongs to the interval, not to a dephasing
            # order, so it is formed once and carries its own tangent; the
            # per-order damping multiplies both pools, whose order-n states
            # describe one dephasing configuration.
            (
                e11,
                e12,
                e21,
                e22,
                grow_free,
                grow_bound,
                d_e11,
                d_e12,
                d_e21,
                d_e22,
                d_grow_free,
                d_grow_bound,
            ) = _two_pool_step_jvp(
                r1,
                -1000.0 * dt1 / (atom_t1 * atom_t1),
                atom_r1_bound,
                d_r1_bound,
                atom_exchange,
                d_exchange,
                atom_bound,
                d_bound,
                event_dt,
                ddt,
                wout,
                dwout,
            )
            old_br = br
            old_bi = bi
            old_dbr = dbr
            old_dbi = dbi
            spun_hr = old_br * turn_cos - old_bi * turn_sin
            spun_hi = old_br * turn_sin + old_bi * turn_cos
            dspun_hr = (
                old_dbr * turn_cos
                + old_br * dturn_cos
                - old_dbi * turn_sin
                - old_bi * dturn_sin
            )
            dspun_hi = (
                old_dbr * turn_sin
                + old_br * dturn_sin
                + old_dbi * turn_cos
                + old_bi * dturn_cos
            )
            free_r = e11 * spun_zr + e12 * spun_hr
            free_i = e11 * spun_zi + e12 * spun_hi
            held_r = e21 * spun_zr + e22 * spun_hr
            held_i = e21 * spun_zi + e22 * spun_hi
            d_free_r = (
                d_e11 * spun_zr + e11 * dspun_zr + d_e12 * spun_hr + e12 * dspun_hr
            )
            d_free_i = (
                d_e11 * spun_zi + e11 * dspun_zi + d_e12 * spun_hi + e12 * dspun_hi
            )
            d_held_r = (
                d_e21 * spun_zr + e21 * dspun_zr + d_e22 * spun_hr + e22 * dspun_hr
            )
            d_held_i = (
                d_e21 * spun_zi + e21 * dspun_zi + d_e22 * spun_hi + e22 * dspun_hi
            )
            zr = damp_z * free_r + tl.where(state == 0, grow_free, 0.0)
            zi = damp_z * free_i
            dzr = (
                ddamp_z * free_r
                + damp_z * d_free_r
                + tl.where(state == 0, d_grow_free, 0.0)
            )
            dzi = ddamp_z * free_i + damp_z * d_free_i
            br = damp_z * held_r + tl.where(state == 0, grow_bound, 0.0)
            bi = damp_z * held_i
            dbr = (
                ddamp_z * held_r
                + damp_z * d_held_r
                + tl.where(state == 0, d_grow_bound, 0.0)
            )
            dbi = ddamp_z * held_i + damp_z * d_held_i
        else:
            dzr = dspun_zr * e1 + spun_zr * de1 + tl.where(state == 0, drecovery, 0.0)
            dzi = dspun_zi * e1 + spun_zi * de1
            zr = spun_zr * e1 + tl.where(state == 0, recovery, 0.0)
            zi = spun_zi * e1

        event_action = tl.load(action + event).to(tl.int32)
        pre_shift = (event_action & 1) != 0
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fpr, fpi, fmr, fmi, state, state_mask, state_count
        )
        shifted_dpr, shifted_dpi, shifted_dmr, shifted_dmi = _shift(
            dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count
        )
        fpr = tl.where(pre_shift, shifted_pr, fpr)
        fpi = tl.where(pre_shift, shifted_pi, fpi)
        fmr = tl.where(pre_shift, shifted_mr, fmr)
        fmi = tl.where(pre_shift, shifted_mi, fmi)
        dfpr = tl.where(pre_shift, shifted_dpr, dfpr)
        dfpi = tl.where(pre_shift, shifted_dpi, dfpi)
        dfmr = tl.where(pre_shift, shifted_dmr, dfmr)
        dfmi = tl.where(pre_shift, shifted_dmi, dfmi)
        if pools == 2 or pools == 3:
            s_bpr, s_bpi, s_bmr, s_bmi = _shift(
                bpr, bpi, bmr, bmi, state, state_mask, state_count
            )
            s_dbpr, s_dbpi, s_dbmr, s_dbmi = _shift(
                dbpr, dbpi, dbmr, dbmi, state, state_mask, state_count
            )
            bpr = tl.where(pre_shift, s_bpr, bpr)
            bpi = tl.where(pre_shift, s_bpi, bpi)
            bmr = tl.where(pre_shift, s_bmr, bmr)
            bmi = tl.where(pre_shift, s_bmi, bmi)
            dbpr = tl.where(pre_shift, s_dbpr, dbpr)
            dbpi = tl.where(pre_shift, s_dbpi, dbpi)
            dbmr = tl.where(pre_shift, s_dbmr, dbmr)
            dbmi = tl.where(pre_shift, s_dbmi, dbmi)

        event_kind = tl.load(kind + event)
        is_rf = event_kind == 1
        is_inversion = (event_action & 4) != 0
        invert = is_rf & is_inversion
        dzr = tl.where(invert, -dinversion * zr - atom_inversion * dzr, dzr)
        dzi = tl.where(invert, -dinversion * zi - atom_inversion * dzi, dzi)
        zr = tl.where(invert, -atom_inversion * zr, zr)
        zi = tl.where(invert, -atom_inversion * zi, zi)
        if pools == 2 or pools == 3:
            # A chemically exchanging pool is free water and turns over like
            # any other; a semisolid one is saturated instead.
            dbr = tl.where(invert, -dinversion * br - atom_inversion * dbr, dbr)
            dbi = tl.where(invert, -dinversion * bi - atom_inversion * dbi, dbi)
            br = tl.where(invert, -atom_inversion * br, br)
            bi = tl.where(invert, -atom_inversion * bi, bi)

        event_flip = _event_value(flip, event_base, event, active_atom, single_train)
        event_phase = _event_value(phase, event_base, event, active_atom, single_train)
        # One shim is the whole sequence's transmit field, loaded once above;
        # several give each pulse a row of its own.
        if shimmed:
            row = tl.load(shim_index + event).to(tl.int64) * atom_count
            atom_b1 = 1.0
            if transmit:
                atom_b1 = tl.load(b1 + row + atom, mask=active_atom, other=1.0)
            db1 = tl.load(tangent_b1 + row + atom, mask=active_atom, other=0.0)
            if off_axis:
                atom_b1_phase = tl.load(
                    b1_phase + row + atom, mask=active_atom, other=0.0
                )
                db1_phase = tl.load(
                    tangent_b1_phase + row + atom, mask=active_atom, other=0.0
                )
        alpha = event_flip * atom_b1
        dalpha = (
            _event_value(tangent_flip, event_base, event, active_atom, single_train)
            * atom_b1
            + event_flip * db1
        )
        phi = event_phase + atom_b1_phase
        dphi = (
            _event_value(tangent_phase, event_base, event, active_atom, single_train)
            + db1_phase
        )
        if pools == 1 or pools == 3:
            # The semisolid pool absorbs the power the pulse deposits, so it reads
            # the bare flip the transmit field gives the voxel. The offset
            # reaches it through the voxel's own off-resonance, which is where
            # the lineshape's slope enters a forward direction.
            offset = tl.load(rf_frequency + event) - atom_b0
            shape, shape_slope = _lineshape_at_slope(
                lineshape, offset, lineshape_bins, lineshape_step
            )
            deposited = tl.load(saturation + event)
            absorbed = tl.exp(deposited * alpha * alpha * shape)
            d_exponent = deposited * (
                2.0 * alpha * dalpha * shape - alpha * alpha * shape_slope * db0
            )
            saturating = is_rf & ~is_inversion
            if pools == 1:
                dbr = tl.where(saturating, absorbed * (dbr + br * d_exponent), dbr)
                dbi = tl.where(saturating, absorbed * (dbi + bi * d_exponent), dbi)
                br = tl.where(saturating, absorbed * br, br)
                bi = tl.where(saturating, absorbed * bi, bi)
            else:
                dcr = tl.where(saturating, absorbed * (dcr + cr * d_exponent), dcr)
                dci = tl.where(saturating, absorbed * (dci + ci * d_exponent), dci)
                cr = tl.where(saturating, absorbed * cr, cr)
                ci = tl.where(saturating, absorbed * ci, ci)
        if profiled or dynamic:
            if dynamic:
                # The array was resolved outside the kernel, so a direction
                # along it arrives already carried through the pulse integral.
                held = _dynamic_pair_at(
                    pairs,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                )
                moved = _dynamic_pair_at(
                    pair_direction,
                    pair_index,
                    event_base,
                    event,
                    atom,
                    atom_count,
                    active_atom,
                )
                pair_ar, pair_ai, pair_br, pair_bi = held
                dot_ar, dot_ai, dot_br, dot_bi = moved
            else:
                read = _profile_pair_slope(
                    profile,
                    _table_row(profile_index, event, location, locations),
                    alpha,
                    profile_bins,
                    profile_step,
                )
                # The flip angle carries the tangent into the table.
                pair_ar, pair_ai = read[0], read[2]
                pair_br, pair_bi = read[4], read[6]
                dot_ar, dot_ai = read[1] * dalpha, read[3] * dalpha
                dot_br, dot_bi = read[5] * dalpha, read[7] * dalpha
            # The RF phase turns the axis after the pair comes out, and so
            # reaches ``b`` alone.
            turn_r = tl.cos(phi)
            turn_i = -tl.sin(phi)
            spun_br = pair_br * turn_r - pair_bi * turn_i
            spun_bi = pair_br * turn_i + pair_bi * turn_r
            slope_br = dot_br
            slope_bi = dot_bi
            (
                shaped_pr,
                shaped_pi,
                shaped_mr,
                shaped_mi,
                shaped_zr,
                shaped_zi,
                shaped_dpr,
                shaped_dpi,
                shaped_dmr,
                shaped_dmi,
                shaped_dzr,
                shaped_dzi,
            ) = _rotate_spinor_dual(
                pair_ar,
                pair_ai,
                spun_br,
                spun_bi,
                dot_ar,
                dot_ai,
                slope_br * turn_r - slope_bi * turn_i + dphi * spun_bi,
                slope_br * turn_i + slope_bi * turn_r - dphi * spun_br,
                fpr,
                fpi,
                fmr,
                fmi,
                zr,
                zi,
                dfpr,
                dfpi,
                dfmr,
                dfmi,
                dzr,
                dzi,
            )
            if pools == 2 or pools == 3:
                # The same pulse, the same rotation.
                (
                    held_pr,
                    held_pi,
                    held_mr,
                    held_mi,
                    held_zr,
                    held_zi,
                    held_dpr,
                    held_dpi,
                    held_dmr,
                    held_dmi,
                    held_dzr,
                    held_dzi,
                ) = _rotate_spinor_dual(
                    pair_ar,
                    pair_ai,
                    spun_br,
                    spun_bi,
                    dot_ar,
                    dot_ai,
                    slope_br * turn_r - slope_bi * turn_i + dphi * spun_bi,
                    slope_br * turn_i + slope_bi * turn_r - dphi * spun_br,
                    bpr,
                    bpi,
                    bmr,
                    bmi,
                    br,
                    bi,
                    dbpr,
                    dbpi,
                    dbmr,
                    dbmi,
                    dbr,
                    dbi,
                )
        cosine = tl.cos(alpha)
        sine = tl.sin(alpha)
        dcosine = -sine * dalpha
        dsine = cosine * dalpha
        cos_phi = tl.cos(phi)
        sin_phi = tl.sin(phi)
        cos_2phi = tl.cos(2.0 * phi)
        sin_2phi = tl.sin(2.0 * phi)
        dcos_phi = -sin_phi * dphi
        dsin_phi = cos_phi * dphi
        dcos_2phi = -2.0 * sin_2phi * dphi
        dsin_2phi = 2.0 * cos_2phi * dphi

        (
            rotated_pr,
            rotated_pi,
            rotated_mr,
            rotated_mi,
            rotated_zr,
            rotated_zi,
            rotated_dpr,
            rotated_dpi,
            rotated_dmr,
            rotated_dmi,
            rotated_dzr,
            rotated_dzi,
        ) = _rotate_flip_phase_jvp(
            cosine,
            dcosine,
            sine,
            dsine,
            cos_phi,
            dcos_phi,
            sin_phi,
            dsin_phi,
            cos_2phi,
            dcos_2phi,
            sin_2phi,
            dsin_2phi,
            fpr,
            fpi,
            fmr,
            fmi,
            zr,
            zi,
            dfpr,
            dfpi,
            dfmr,
            dfmi,
            dzr,
            dzi,
        )
        if pools == 2 or pools == 3:
            (
                b_pr,
                b_pi,
                b_mr,
                b_mi,
                b_zr,
                b_zi,
                b_dpr,
                b_dpi,
                b_dmr,
                b_dmi,
                b_dzr,
                b_dzi,
            ) = _rotate_flip_phase_jvp(
                cosine,
                dcosine,
                sine,
                dsine,
                cos_phi,
                dcos_phi,
                sin_phi,
                dsin_phi,
                cos_2phi,
                dcos_2phi,
                sin_2phi,
                dsin_2phi,
                bpr,
                bpi,
                bmr,
                bmi,
                br,
                bi,
                dbpr,
                dbpi,
                dbmr,
                dbmi,
                dbr,
                dbi,
            )
        if profiled or dynamic:
            rotated_pr = shaped_pr
            rotated_pi = shaped_pi
            rotated_mr = shaped_mr
            rotated_mi = shaped_mi
            rotated_zr = shaped_zr
            rotated_zi = shaped_zi
            rotated_dpr = shaped_dpr
            rotated_dpi = shaped_dpi
            rotated_dmr = shaped_dmr
            rotated_dmi = shaped_dmi
            rotated_dzr = shaped_dzr
            rotated_dzi = shaped_dzi
            if pools == 2 or pools == 3:
                b_pr = held_pr
                b_pi = held_pi
                b_mr = held_mr
                b_mi = held_mi
                b_zr = held_zr
                b_zi = held_zi
                b_dpr = held_dpr
                b_dpi = held_dpi
                b_dmr = held_dmr
                b_dmi = held_dmi
                b_dzr = held_dzr
                b_dzi = held_dzi

        rotate = is_rf & ~is_inversion
        fpr = tl.where(rotate, rotated_pr, fpr)
        fpi = tl.where(rotate, rotated_pi, fpi)
        fmr = tl.where(rotate, rotated_mr, fmr)
        fmi = tl.where(rotate, rotated_mi, fmi)
        zr = tl.where(rotate, rotated_zr, zr)
        zi = tl.where(rotate, rotated_zi, zi)
        dfpr = tl.where(rotate, rotated_dpr, dfpr)
        dfpi = tl.where(rotate, rotated_dpi, dfpi)
        dfmr = tl.where(rotate, rotated_dmr, dfmr)
        dfmi = tl.where(rotate, rotated_dmi, dfmi)
        dzr = tl.where(rotate, rotated_dzr, dzr)
        dzi = tl.where(rotate, rotated_dzi, dzi)
        if pools == 2 or pools == 3:
            bpr = tl.where(rotate, b_pr, bpr)
            bpi = tl.where(rotate, b_pi, bpi)
            bmr = tl.where(rotate, b_mr, bmr)
            bmi = tl.where(rotate, b_mi, bmi)
            br = tl.where(rotate, b_zr, br)
            bi = tl.where(rotate, b_zi, bi)
            dbpr = tl.where(rotate, b_dpr, dbpr)
            dbpi = tl.where(rotate, b_dpi, dbpi)
            dbmr = tl.where(rotate, b_dmr, dbmr)
            dbmi = tl.where(rotate, b_dmi, dbmi)
            dbr = tl.where(rotate, b_dzr, dbr)
            dbi = tl.where(rotate, b_dzi, dbi)

        record = ((event_action & 32) != 0) & (event_kind == 2)
        adc_cos = tl.cos(event_phase)
        adc_sin = tl.sin(event_phase)
        dadc_phase = _event_value(
            tangent_phase, event_base, event, active_atom, single_train
        )
        dadc_cos = -adc_sin * dadc_phase
        dadc_sin = adc_cos * dadc_phase
        read_r = fpr
        read_i = fpi
        dread_r = dfpr
        dread_i = dfpi
        if pools == 2 or pools == 3:
            read_r = fpr + bpr
            read_i = fpi + bpi
            dread_r = dfpr + dbpr
            dread_i = dfpi + dbpi
        signal_real = dm0 * (read_r * adc_cos + read_i * adc_sin)
        signal_real += atom_m0 * (
            dread_r * adc_cos
            + read_r * dadc_cos
            + dread_i * adc_sin
            + read_i * dadc_sin
        )
        signal_imag = dm0 * (read_i * adc_cos - read_r * adc_sin)
        signal_imag += atom_m0 * (
            dread_i * adc_cos
            + read_i * dadc_cos
            - dread_r * adc_sin
            - read_r * dadc_sin
        )
        out = tl.load(output_index + event)
        output_offset = problem * output_count + out
        output_mask = active_atom & (state == 0) & record & (out >= 0)
        tl.store(output_real + output_offset + state, signal_real, mask=output_mask)
        tl.store(output_imag + output_offset + state, signal_imag, mask=output_mask)

        do_shift = ((event_action & 2) != 0) | ((event_action & 16) != 0)
        shifted_pr, shifted_pi, shifted_mr, shifted_mi = _shift(
            fpr, fpi, fmr, fmi, state, state_mask, state_count
        )
        shifted_dpr, shifted_dpi, shifted_dmr, shifted_dmi = _shift(
            dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count
        )
        fpr = tl.where(do_shift, shifted_pr, fpr)
        fpi = tl.where(do_shift, shifted_pi, fpi)
        fmr = tl.where(do_shift, shifted_mr, fmr)
        fmi = tl.where(do_shift, shifted_mi, fmi)
        dfpr = tl.where(do_shift, shifted_dpr, dfpr)
        dfpi = tl.where(do_shift, shifted_dpi, dfpi)
        dfmr = tl.where(do_shift, shifted_dmr, dfmr)
        dfmi = tl.where(do_shift, shifted_dmi, dfmi)
        spoil = (event_action & 8) != 0
        fpr = tl.where(spoil, 0.0, fpr)
        fpi = tl.where(spoil, 0.0, fpi)
        fmr = tl.where(spoil, 0.0, fmr)
        fmi = tl.where(spoil, 0.0, fmi)
        dfpr = tl.where(spoil, 0.0, dfpr)
        dfpi = tl.where(spoil, 0.0, dfpi)
        dfmr = tl.where(spoil, 0.0, dfmr)
        dfmi = tl.where(spoil, 0.0, dfmi)
        if pools == 2 or pools == 3:
            s_bpr, s_bpi, s_bmr, s_bmi = _shift(
                bpr, bpi, bmr, bmi, state, state_mask, state_count
            )
            s_dbpr, s_dbpi, s_dbmr, s_dbmi = _shift(
                dbpr, dbpi, dbmr, dbmi, state, state_mask, state_count
            )
            bpr = tl.where(spoil, 0.0, tl.where(do_shift, s_bpr, bpr))
            bpi = tl.where(spoil, 0.0, tl.where(do_shift, s_bpi, bpi))
            bmr = tl.where(spoil, 0.0, tl.where(do_shift, s_bmr, bmr))
            bmi = tl.where(spoil, 0.0, tl.where(do_shift, s_bmi, bmi))
            dbpr = tl.where(spoil, 0.0, tl.where(do_shift, s_dbpr, dbpr))
            dbpi = tl.where(spoil, 0.0, tl.where(do_shift, s_dbpi, dbpi))
            dbmr = tl.where(spoil, 0.0, tl.where(do_shift, s_dbmr, dbmr))
            dbmi = tl.where(spoil, 0.0, tl.where(do_shift, s_dbmi, dbmi))


def _pool_flag(lineshape: Any, exchanging: bool) -> int:
    """Which pools a launch is to carry, as the kernels' own constexpr reads it.

    Kept in one place so a launcher cannot describe the tissue one way and the
    kernel read it another.
    """
    if lineshape is not None and exchanging:
        return 3
    if exchanging:
        return 2
    return 1 if lineshape is not None else 0


# The operator table holds nine entries per voxel per distinct interval and
# the adjoint's cotangent table twelve, both float32.
_TABLE_FLOATS_PER_ROW = 9
_BAR_FLOATS_PER_ROW = 12

# What the two tables may take of what the card can spare. The trajectory is
# the larger claim on the same memory and is allocated after them.
_TABLE_SHARE = 0.25
_TABLE_FLOOR_BYTES = 64 << 20


def _three_pool_table_bytes(
    tissue: tuple[torch.Tensor, ...],
    rows: int,
    *,
    problems: int | None,
    dual: bool,
) -> int:
    """What the tables would take -- the operator's, and the adjoint's bars.

    The operator table holds a row of voxels; the cotangent table holds a row
    of problems, which is voxels times trains cut to what one chunk carries.
    ``problems`` of ``None`` is a caller that builds no cotangent table.

    A ``dual`` launch stores the operator twice over, value and direction, and
    pools its cotangents three times over: the value bars, the tangent bars,
    and the value bars weighted by each event's own interval direction.
    """
    entries = _TABLE_FLOATS_PER_ROW * (2 if dual else 1)
    total = int(tissue[0].numel()) * int(rows) * entries
    if problems is not None:
        pooled = _BAR_FLOATS_PER_ROW * (3 if dual else 1)
        total += int(problems) * int(rows) * pooled
    return total * 4


def _table_budget(device: torch.device) -> int:
    """How many bytes the three-pool tables may claim on this device."""
    if device.type != "cuda":
        return _TABLE_FLOOR_BYTES
    free, _total = torch.cuda.mem_get_info(device)
    return max(_TABLE_FLOOR_BYTES, int(free * _TABLE_SHARE))


def _three_pool_table_jvp(
    tissue: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    durations: torch.Tensor,
) -> torch.Tensor:
    """The three-pool operator and a direction through it, per distinct length.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    tangents:
        The directions along them, in the same order.
    durations:
        The distinct interval lengths, in seconds.

    Returns
    -------
    torch.Tensor
        ``(rows, 18, voxels)`` float32, undamped and at ``d_dt`` of zero.
    """
    voxels = int(tissue[0].numel())
    rows = int(durations.numel())
    table = torch.empty(
        (rows, 18, voxels), dtype=torch.float32, device=tissue[0].device
    )
    order = (0, 14, 11, 13, 10, 12, 9)
    block = min(1024, triton.next_power_of_2(max(voxels, 1)))
    spread = durations.abs().to(torch.float64) * three_pool_spread_rate(tissue)
    for picked, narrow in (
        (torch.nonzero(spread <= NARROW_SPREAD).flatten(), True),
        (torch.nonzero(spread > NARROW_SPREAD).flatten(), False),
    ):
        if picked.numel() == 0:
            continue
        _three_pool_table_jvp_kernel[(picked.numel(), triton.cdiv(voxels, block))](
            *(tissue[index] for index in order),
            *(tangents[index] for index in order),
            durations.to(torch.float32),
            picked.to(torch.int32),
            table,
            voxels,
            BLOCK=block,
            narrow=narrow,
            num_warps=4,
        )
    return table


def _tabulate_three_pool(
    tissue: tuple[torch.Tensor, ...],
    duration: torch.Tensor,
    *,
    pools: int,
    narrow: bool,
    problems: int | None = None,
    tangents: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """The operator table an event loop should read, or ``None`` to form it.

    Only a wide launch has anything to gain: under ``narrow`` the operator is
    already 504 float32 instructions and forming it per event costs less than
    a round trip through memory.

    A wide launch is wide because of its longest interval, and one preparation
    delay is enough -- so the events that pay the roots in double are mostly
    events whose own length would have taken the series. Splitting the table by
    row is what lets each interval take the branch its own spread asks for,
    which is worth more than sharing a row between events and does not need a
    row to be shared at all.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    duration:
        The packed event durations, in seconds.
    pools:
        Which pool model the launch carries.
    narrow:
        Whether every interval keeps the eigenvalues close together.
    problems:
        How many problems a chunk of the adjoint carries, which is the height
        of the cotangent table it allocates. ``None`` for a caller that builds
        no such table.
    tangents:
        The directions along the tissue, for a caller that follows one. The
        table then carries the direction through the operator beside its
        value, at twice the width.

    Returns
    -------
    tuple
        The per-event row index, the table and the distinct lengths, or
        ``(None, None, None)``.
    """
    if pools != 3 or narrow:
        return None, None, None
    distinct, inverse = torch.unique(duration.detach(), return_inverse=True)
    # A row costs a formation, an event costs one too, so a train whose
    # lengths are all different has nothing to gain and a table to write.
    if distinct.numel() >= duration.numel():
        return None, None, None
    if _three_pool_table_bytes(
        tissue, distinct.numel(), problems=problems, dual=tangents is not None
    ) > _table_budget(tissue[0].device):
        # A pathological train has as many lengths as events, and the tables
        # grow with their product. Forming the operator per event is slower
        # and always fits, so that is what an unbounded one falls back to.
        return None, None, None
    lengths = distinct.to(torch.float32).contiguous()
    if tangents is not None:
        built = _three_pool_table_jvp(tissue, tangents, lengths)
    else:
        built = _three_pool_table(tissue, lengths)
    return inverse.reshape(duration.shape).to(torch.int32), built, lengths


def _three_pool_table(
    tissue: tuple[torch.Tensor, ...], durations: torch.Tensor
) -> torch.Tensor:
    """The three-pool operator for each distinct interval, over every voxel.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    durations:
        The distinct interval lengths, in seconds.

    Returns
    -------
    torch.Tensor
        ``(rows, 9, voxels)`` float32, undamped -- the reading event applies
        its own washout.
    """
    (
        t1,
        _t2,
        _m0,
        _b1,
        _b1_phase,
        _b0,
        _inversion,
        _diffusion,
        _velocity,
        bound_fraction,
        bound_exchange,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        _t2_pool_b,
        _pool_b_shift,
    ) = tissue
    voxels = t1.numel()
    rows = durations.numel()
    table = torch.empty((rows, 9, voxels), dtype=torch.float32, device=t1.device)
    # The spread a row reaches is its own length times the rate, so the split
    # is exact per row rather than one verdict for the whole table.
    spread = durations.abs().to(torch.float64) * three_pool_spread_rate(tissue)
    block = min(1024, triton.next_power_of_2(max(voxels, 1)))
    narrow_rows = torch.nonzero(spread <= NARROW_SPREAD, as_tuple=False).flatten()
    wide_rows = torch.nonzero(spread > NARROW_SPREAD, as_tuple=False).flatten()
    for picked, narrow in ((narrow_rows, True), (wide_rows, False)):
        if picked.numel() == 0:
            continue
        _three_pool_table_kernel[(picked.numel(), triton.cdiv(voxels, block))](
            t1,
            t1_pool_b,
            t1_bound,
            pool_b_exchange,
            bound_exchange,
            pool_b_fraction,
            bound_fraction,
            durations.to(torch.float32),
            picked.to(torch.int32),
            table,
            voxels,
            BLOCK=block,
            narrow=narrow,
            num_warps=4,
        )
    return table


# Elements of the state tile one program carries. Four to a lane keeps enough
# independent arithmetic in flight to cover the latency of the chain the event
# loop is, and a wider tile spends registers without buying more of it; both
# halves of that were measured over 8 to 64 configuration orders.
_TILE_ELEMENTS = 64


def _atom_stride(*tuples: tuple[torch.Tensor, ...]) -> int:
    """How far to step through a property to reach one voxel's value.

    Zero where every optional property was given as one value for the whole
    tissue: each is then read at one address by every voxel and needs no room
    per voxel. The relaxation times lead each tuple and are stepped by one
    whatever this says, since a tissue is its two relaxation times before it is
    anything else.

    One stride serves the values and the directions followed beside them, so a
    pass carrying tangents is asked about both: a direction laid out per voxel
    has to be stepped through even where the value it follows is one number.
    """
    return (
        0 if all(value.numel() <= 1 for values in tuples for value in values[2:]) else 1
    )


def _problems_per_program(block_states: int) -> int:
    """How many independent problems to carry on one program's lane axis.

    A warp's lanes cost about the same whether they are used or not, so packing
    several problems into one program is close to free.

    It depends on the state count alone, and deliberately not on how many
    problems the launch has. A run cut into chunks would otherwise compile a
    different tile from the same run whole, and the two tiles reassociate their
    arithmetic differently -- so a streamed volume would answer a little
    differently from an unstreamed one, which is a difference a caller has no
    way to account for. ``tests/sequence/test_both_pools.py`` pins that.

    The result indexes a ``tl.arange``, so it must be a power of two.
    """
    widest = max(1, _TILE_ELEMENTS // block_states)
    return 1 << (widest.bit_length() - 1)


def _output_shape(
    train_count: int, atom_count: int, output_count: int
) -> tuple[int, ...]:
    """Signal shape, matching what the CPU kernels return."""
    if train_count == 1:
        return (atom_count, output_count)
    return (train_count, atom_count, output_count)


def _only_scalars(flags: dict) -> dict:
    """The switches a real-subspace kernel takes.

    Off-resonance and flow are not in its representation to begin with -- it
    carries three real planes where the complex kernels carry four -- so it is
    given the terms that survive that reduction and no others.
    """
    return {
        name: flags[name] for name in ("diffusing", "transmit", "density", "inverting")
    }


def simulate(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> torch.Tensor:
    """Run a packed state machine on CUDA and return complex signals.

    ``real_axis`` of 1 selects the real-subspace kernel; see
    ``real_subspace_axis`` for when that is legitimate.
    """
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_into(
        tissue,
        events,
        output_real,
        output_imag,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        features=features,
    )
    return torch.complex(output_real, output_imag)


def simulate_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> None:
    """Run the forward machine into buffers the caller owns.

    Streaming reuses one set of buffers per chunk, so allocating here would put
    an allocation in the loop -- and an allocation that reaches ``cudaMalloc``
    synchronizes the device, which is exactly what the streams exist to avoid.

    ``atom_count`` is given rather than taken from ``tissue`` because a chunk's
    buffers are sized for the largest chunk and the last one is shorter.
    """
    (
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
    ) = tissue
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
    train_count = _train_count(events)
    shims = _shim_count(tissue)
    pools = _pool_flag(lineshape, exchanging)
    block_states = triton.next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(block_states)
    grid = (triton.cdiv(total, problems),)
    # A kernel argument has to be a tensor even where the branch reading it is
    # compiled out, so an unprofiled launch passes one it already has.
    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    duration_row, pool_table, _lengths = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow
    )

    if real_axis == 1:
        _epg_real_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            output_real,
            output_imag,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shimmed=_shim_count(tissue) > 1,
            **_only_scalars(_feature_flags(features, geometry)),
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return

    _epg_kernel[grid](
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
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        t1 if pairs is None else pairs,
        kind if pair_rows is None else pair_rows,
        kind if duration_row is None else duration_row,
        t1 if pool_table is None else pool_table,
        output_real,
        output_imag,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        single_train=train_count == 1,
        atom_stride=_atom_stride(tissue),
        shim_rows=shims,
        shimmed=shims > 1,
        locations=1 if profile is None else profile.points,
        profiled=profile is not None and profile.bins > 0,
        profile_bins=0 if profile is None else profile.bins,
        dynamic=dynamic is not None,
        broadened=lineshape is not None and lineshape.bins > 0,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
        pools=pools,
        narrow=narrow,
        tabulated=pool_table is not None,
        **_feature_flags(features, geometry),
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
) -> torch.Tensor:
    """Run one fused state-machine Jacobian-vector product on CUDA.

    ``real_axis`` of 1 selects the real-subspace kernel, which produces no
    derivative along ``b1_phase``, ``b0`` or the RF phase -- seeds along those
    directions leave the subspace, so the caller must rule them out.
    """
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_jvp_into(
        tissue,
        events,
        tissue_tangents,
        event_tangents,
        output_real,
        output_imag,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
    )
    return torch.complex(output_real, output_imag)


def simulate_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
) -> None:
    """Run one Jacobian-vector product into buffers the caller owns.

    See ``simulate_into`` for why the streaming path needs this.
    """
    (
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
    ) = tissue
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
    tangent_duration, tangent_flip, tangent_phase = event_tangents
    train_count = _train_count(events)
    pools = _pool_flag(lineshape, exchanging)
    shims = _shim_count(tissue)
    block_states = triton.next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(block_states)
    grid = (triton.cdiv(total, problems),)

    if real_axis == 1:
        _epg_real_jvp_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            tissue_tangents[0],
            tissue_tangents[1],
            tissue_tangents[2],
            tissue_tangents[3],
            tissue_tangents[6],
            tissue_tangents[7],
            tangent_duration,
            tangent_flip,
            output_real,
            output_imag,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue, tissue_tangents),
            shimmed=shims > 1,
            **_only_scalars(_feature_flags(features, geometry)),
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        return

    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    pair_direction = (
        None if dynamic_direction is None else dynamic_direction.to(t1.device)
    )
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    duration_row, pool_table, _lengths = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow, tangents=tissue_tangents
    )
    _epg_jvp_kernel[grid](
        *tissue,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        *tissue_tangents,
        tangent_duration,
        tangent_flip,
        tangent_phase,
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        t1 if pairs is None else pairs,
        kind if pair_rows is None else pair_rows,
        t1 if pair_direction is None else pair_direction,
        kind if duration_row is None else duration_row,
        t1 if pool_table is None else pool_table,
        output_real,
        output_imag,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        single_train=train_count == 1,
        atom_stride=_atom_stride(tissue, tissue_tangents),
        shim_rows=shims,
        shimmed=shims > 1,
        locations=1 if profile is None else profile.points,
        profiled=profile is not None and profile.bins > 0,
        profile_bins=0 if profile is None else profile.bins,
        dynamic=dynamic is not None,
        broadened=lineshape is not None and lineshape.bins > 0,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
        pools=pools,
        narrow=narrow,
        tabulated=pool_table is not None,
        **_feature_flags(features, geometry),
        block_states=block_states,
        problems=problems,
        num_warps=1,
    )


# How much device memory the recorded trajectory may hold at once. Beyond this
# the problems are run in waves, which the gradient buffers absorb because they
# accumulate rather than being written.
_TRAJECTORY_BUDGET_BYTES = 256 << 20


def _trajectory_wave(
    event_count: int, state_count: int, total: int, planes: int, blocks: int = 3
) -> int:
    """How many problems can record their trajectory in one launch."""
    per_problem = event_count * blocks * state_count * planes * 4
    return max(1, min(total, _TRAJECTORY_BUDGET_BYTES // max(1, per_problem)))


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
    exchanging: bool = False,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint on CUDA, for a whole volume on one device.

    Returns the gradients in the differentiable-input order -- every tissue
    property, then event duration, flip and phase, and the pair's cotangent
    where one is given. A shard takes this same kernel a level down and a
    streamed volume has chunked launchers of its own;
    :func:`torchsim.sequence._accelerators` decides which route a run takes.

    Carrying no forward direction, this records two trajectory planes per
    recorded state where that pass records four, and holds one state where it
    holds a dual.
    """
    (
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
        exchange_rate,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
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
    ) = events[:9]
    atom_count = t1.numel()
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)
    device = t1.device
    shims = max(1, b1.numel() // atom_count) if atom_count else 1
    table = None if profile is None else profile.packed(device)
    table_rows = None if profile is None else profile.rows().to(device)
    pairs = None if dynamic is None else dynamic.packed(device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, event_count).to(device)
    )
    grad_pair = None if dynamic is None else torch.zeros_like(pairs)
    locations = 1 if profile is None else profile.points
    absorption = None if lineshape is None else lineshape.packed(device)

    grad_tissue = torch.zeros(
        tissue_gradient_height(shims) * atom_count,
        dtype=torch.float32,
        device=device,
    )
    grad_flip = torch.zeros_like(flip)
    grad_phase = torch.zeros_like(phase)
    grad_duration = torch.zeros_like(duration)
    grad_output = grad_output.resolve_conj()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()

    # A semisolid pool records a plane of its own beside the three the free
    # water keeps; a chemically exchanging one three, and the two together
    # four.
    pools = _pool_flag(lineshape, exchanging)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    blocks = 7 if pools == 3 else (6 if pools == 2 else (4 if pools == 1 else 3))
    wave = _trajectory_wave(event_count, state_count, total, 2, blocks)
    duration_row, pool_table, pool_durations = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow, problems=wave
    )
    row_count = 0 if pool_durations is None else pool_durations.numel()
    pool_bars = None
    if pool_table is not None:
        # A slot per problem the chunk carries, so the walk back accumulates
        # into memory it owns and no two programs contend for a row. The
        # chunks run one after another, so one chunk's worth is enough.
        pool_bars = torch.zeros(
            wave * row_count * 12, dtype=torch.float32, device=device
        )
    trajectory = [
        torch.empty(
            (wave, event_count * blocks * state_count),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(2)
    ]

    problems = _problems_per_program(block_states)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        if pool_bars is not None:
            # The slots are per chunk, so each chunk starts from nothing.
            pool_bars.zero_()
        _epg_vjp_kernel[(triton.cdiv(span, problems),)](
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
            exchange_rate,
            t1_bound,
            pool_b_fraction,
            pool_b_exchange,
            t1_pool_b,
            t2_pool_b,
            pool_b_shift,
            duration,
            kind,
            flip,
            phase,
            action,
            output_index,
            shim_index,
            saturation,
            rf_frequency,
            absorption,
            table,
            table_rows,
            pairs,
            pair_rows,
            duration_row,
            pool_table,
            pool_bars,
            pool_durations,
            row_count,
            grad_pair,
            grad_real,
            grad_imag,
            grad_tissue,
            grad_flip,
            grad_phase,
            grad_duration,
            *trajectory,
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            geometry.flow_scale,
            geometry.washout_scale,
            shims,
            1.0 if profile is None else profile.step,
            1.0 if lineshape is None else lineshape.step,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shimmed=shims > 1,
            locations=locations,
            profiled=profile is not None and profile.bins > 0,
            profile_bins=0 if profile is None else profile.bins,
            dynamic=dynamic is not None,
            broadened=lineshape is not None and lineshape.bins > 0,
            lineshape_bins=0 if lineshape is None else lineshape.bins,
            pools=pools,
            narrow=narrow,
            tabulated=pool_table is not None,
            block_states=block_states,
            problems=problems,
            num_warps=1,
            **_feature_flags(features, geometry),
        )
    voxel = tuple(
        grad_tissue[base * atom_count : (base + rows) * atom_count]
        for base, rows in zip(
            tissue_gradient_bases(shims), tissue_gradient_rows(shims), strict=True
        )
    )
    if dynamic is not None:
        return (*voxel, grad_duration, grad_flip, grad_phase, grad_pair)
    return (*voxel, grad_duration, grad_flip, grad_phase)


def simulate_real_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint through the real subspace, on CUDA.

    Returns the gradients in the differentiable-input order -- every tissue
    property, then event duration, flip and phase. The representation divides
    the RF phase out, so transmit phase, off-resonance, velocity and RF phase
    come back at zero and callers must not ask for those.

    Carrying no forward direction, this records one trajectory plane where the
    forward-over-reverse pass records two, and holds one state where it holds a
    dual.
    """
    (
        t1,
        t2,
        m0,
        b1,
        _b1_phase,
        _b0,
        inversion_efficiency,
        diffusion,
        *_rest,
    ) = tissue
    duration, kind, flip, phase, action, output_index, shim_index = events[:7]
    atom_count = t1.numel()
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)
    device = t1.device
    shims = _shim_count(tissue)

    grad_tissue = torch.zeros(
        tissue_gradient_height(shims) * atom_count,
        dtype=torch.float32,
        device=device,
    )
    grad_flip = torch.zeros_like(flip)
    grad_duration = torch.zeros_like(duration)
    grad_phase = torch.zeros_like(phase)
    grad_imag = grad_output.resolve_conj().imag.contiguous()

    wave = _trajectory_wave(event_count, state_count, total, 1)
    trajectory = torch.empty(
        (wave, event_count * 3 * state_count), dtype=torch.float32, device=device
    )

    problems = _problems_per_program(block_states)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        _epg_real_vjp_kernel[(triton.cdiv(span, problems),)](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            grad_imag,
            grad_tissue,
            grad_flip,
            grad_duration,
            trajectory,
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shim_rows=shims,
            shimmed=shims > 1,
            **_only_scalars(_feature_flags(features, NO_GEOMETRY)),
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
    voxel = tuple(
        grad_tissue[base * atom_count : (base + rows) * atom_count]
        for base, rows in zip(
            tissue_gradient_bases(shims), tissue_gradient_rows(shims), strict=True
        )
    )
    return (*voxel, grad_duration, grad_flip, grad_phase)


class AdjointBuffers:
    """Device memory a forward-over-reverse pass writes into.

    Sized for ``chunk`` voxels and reusable for any narrower one. Per-voxel
    gradients are cleared before each pass; per-event gradients accumulate over
    every pass the buffers serve and are read out with ``event_gradients``.

    ``real_axis`` of 1 halves the state planes, so buffers built for one
    representation cannot be handed to the other.
    """

    def __init__(
        self,
        events: tuple[torch.Tensor, ...],
        chunk: int,
        *,
        state_count: int,
        output_count: int,
        real_axis: int | None = None,
        shims: int = 1,
        pools: int = 0,
    ) -> None:
        (
            duration,
            kind,
            flip,
            phase,
            _action,
            _output_index,
            _shim,
            _saturation,
            _rf_frequency,
        ) = events
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.planes = 2 if real_axis == 1 else 4
        # A bound pool records a fourth block of states per event: the RF
        # operator scales it, so the reverse sweep cannot replay it from the
        # free pool's.
        self.blocks = 3 + (4 if pools == 3 else (3 if pools == 2 else pools))
        self.chunk = chunk
        self.shims = shims
        self.rows = tissue_gradient_height(shims)
        self.state_count = state_count
        self.output_count = output_count
        self.train_count = train_count
        # One dual accumulator per plane: value is the gradient w.r.t. the
        # tangent inputs, tangent the gradient w.r.t. the primal ones.
        self.tissue = [
            torch.zeros(self.rows * chunk, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.flip = [torch.zeros_like(flip) for _ in range(2)]
        self.duration = [torch.zeros_like(duration) for _ in range(2)]
        self.phase = [torch.zeros_like(phase) for _ in range(2)]
        self.cotangent = [
            torch.empty(
                train_count * chunk * output_count,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self.wave = _trajectory_wave(
            event_count,
            state_count,
            train_count * chunk,
            self.planes,
            self.blocks,
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * self.blocks * state_count),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.planes)
        ]

    def tissue_gradients(self, atom_count: int) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-voxel gradients of the last pass, one entry per parameter.

        Each is flat and as wide as the buffer it belongs to, so the transmit
        pair spans every shim. Ordered to match ``event_gradients``: tangent
        plane first.
        """
        return tuple(
            tuple(
                self.tissue[plane][base * atom_count : (base + rows) * atom_count]
                for base, rows in zip(
                    tissue_gradient_bases(self.shims),
                    tissue_gradient_rows(self.shims),
                    strict=True,
                )
            )
            for plane in (1, 0)
        )

    def event_gradients(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-event gradients summed over every pass so far.

        Ordered ``(duration, flip, phase)`` to match the tail of the
        differentiable-input order, tangent plane first.
        """
        return tuple(
            (self.duration[plane], self.flip[plane], self.phase[plane])
            for plane in (1, 0)
        )


class GradientBuffers:
    """Device memory a first-order adjoint writes into.

    Half of what the forward-over-reverse pass needs: one accumulator per
    gradient rather than a dual, and one trajectory plane per real state rather
    than a plane per component of one. Sized for ``chunk`` voxels and reusable
    for any narrower one; per-event gradients accumulate over every pass the
    buffers serve.

    ``real_axis`` of 1 halves the planes again, so buffers built for one
    representation cannot be handed to the other.
    """

    def __init__(
        self,
        events: tuple[torch.Tensor, ...],
        chunk: int,
        *,
        state_count: int,
        output_count: int,
        real_axis: int | None = None,
    ) -> None:
        duration, kind, flip, phase = events[:4]
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.real_axis = real_axis
        self.planes = 1 if real_axis == 1 else 2
        self.chunk = chunk
        self.rows = tissue_gradient_height(1)
        self.state_count = state_count
        self.output_count = output_count
        self.train_count = train_count
        self.tissue = torch.zeros(self.rows * chunk, dtype=torch.float32, device=device)
        self.flip = torch.zeros_like(flip)
        self.duration = torch.zeros_like(duration)
        self.phase = torch.zeros_like(phase)
        self.cotangent = [
            torch.empty(
                train_count * chunk * output_count,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self.wave = _trajectory_wave(
            event_count, state_count, train_count * chunk, self.planes
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * 3 * state_count),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.planes)
        ]

    def tissue_gradients(self, atom_count: int) -> tuple[torch.Tensor, ...]:
        """The per-voxel gradients of the last pass, one entry per parameter."""
        return tuple(
            self.tissue[base * atom_count : (base + rows) * atom_count]
            for base, rows in zip(
                tissue_gradient_bases(1), tissue_gradient_rows(1), strict=True
            )
        )

    def event_gradients(self) -> tuple[torch.Tensor, ...]:
        """``(duration, flip, phase)``, summed over every pass so far."""
        return (self.duration, self.flip, self.phase)


def simulate_vjp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: GradientBuffers,
    *,
    state_count: int,
    output_count: int,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """One chunk of a first-order adjoint, into buffers the caller owns.

    ``grad_output`` is already on the device. Returns the per-voxel gradients
    of this chunk; the per-event ones accumulate in ``buffers``.
    """
    (
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
        exchange_rate,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
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
    ) = events[:9]
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)

    buffers.tissue.zero_()
    grad_output = grad_output.resolve_conj()
    size = total * output_count
    grad_real = buffers.cotangent[0][:size]
    grad_imag = buffers.cotangent[1][:size]
    grad_real.copy_(grad_output.real.reshape(-1))
    grad_imag.copy_(grad_output.imag.reshape(-1))

    problems = _problems_per_program(block_states)
    for base in range(0, total, buffers.wave):
        span = min(buffers.wave, total - base)
        _epg_vjp_kernel[(triton.cdiv(span, problems),)](
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
            exchange_rate,
            t1_bound,
            pool_b_fraction,
            pool_b_exchange,
            t1_pool_b,
            t2_pool_b,
            pool_b_shift,
            duration,
            kind,
            flip,
            phase,
            action,
            output_index,
            shim_index,
            saturation,
            rf_frequency,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            grad_real,
            grad_imag,
            buffers.tissue,
            buffers.flip,
            buffers.phase,
            buffers.duration,
            *buffers.trajectory,
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            geometry.flow_scale,
            geometry.washout_scale,
            1,
            1.0,
            1.0,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shimmed=False,
            locations=1,
            profiled=False,
            profile_bins=0,
            dynamic=False,
            broadened=False,
            lineshape_bins=0,
            pools=0,
            narrow=False,
            tabulated=False,
            block_states=block_states,
            problems=problems,
            num_warps=1,
            **_feature_flags(features, geometry),
        )
    return buffers.tissue_gradients(atom_count)


def simulate_real_vjp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: GradientBuffers,
    *,
    state_count: int,
    output_count: int,
    atom_count: int,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """The same, for a train the real subspace covers."""
    (
        t1,
        t2,
        m0,
        b1,
        _b1_phase,
        _b0,
        inversion_efficiency,
        diffusion,
        *_rest,
    ) = tissue
    duration, kind, flip, _phase, action, output_index, shim_index = events[:7]
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)

    buffers.tissue.zero_()
    size = total * output_count
    grad_imag = buffers.cotangent[1][:size]
    grad_imag.copy_(grad_output.resolve_conj().imag.reshape(-1))

    problems = _problems_per_program(block_states)
    for base in range(0, total, buffers.wave):
        span = min(buffers.wave, total - base)
        _epg_real_vjp_kernel[(triton.cdiv(span, problems),)](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            grad_imag,
            buffers.tissue,
            buffers.flip,
            buffers.duration,
            buffers.trajectory[0],
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            # The streamed route carries one shim, as the complex one does:
            # ``GradientBuffers`` sizes its gradient plane for a single row.
            shim_rows=1,
            shimmed=False,
            block_states=block_states,
            problems=problems,
            num_warps=1,
            **_only_scalars(_feature_flags(features, NO_GEOMETRY)),
        )
    return buffers.tissue_gradients(atom_count)


def simulate_vjp_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: AdjointBuffers,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    dynamic_gradients: tuple[torch.Tensor, torch.Tensor] | None = None,
    features: frozenset[str] | None = None,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Forward-over-reverse for one chunk of voxels, into caller-owned buffers.

    Returns the per-voxel gradients of this chunk -- tangent plane first, one
    entry per tissue parameter, views into ``buffers`` that the next call
    overwrites. The per-event gradients accumulate inside ``buffers`` instead,
    because every chunk contributes to all of them.

    ``atom_count`` is this chunk's width, which may be narrower than the one
    the buffers were built for.
    """
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        _bound_fraction,
        _bound_exchange,
        _t1_bound,
        _pool_b_fraction,
        _pool_b_exchange,
        _t1_pool_b,
        _t2_pool_b,
        _pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        _saturation,
        _rf_frequency,
    ) = events
    train_count = _train_count(events)
    pools = _pool_flag(lineshape, exchanging)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = triton.next_power_of_2(state_count)
    real = real_axis == 1

    grad_output = grad_output.resolve_conj()
    size = total * output_count
    grad_real, grad_imag = (
        plane[:size].view(grad_output.shape) for plane in buffers.cotangent
    )
    grad_real.copy_(grad_output.real)
    grad_imag.copy_(grad_output.imag)
    grad_tissue = [plane[: buffers.rows * atom_count] for plane in buffers.tissue]
    for plane in grad_tissue:
        plane.zero_()
    grad_flip, grad_duration, grad_phase = buffers.flip, buffers.duration, buffers.phase
    trajectory = buffers.trajectory
    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    pair_direction = (
        None if dynamic_direction is None else dynamic_direction.to(t1.device)
    )
    grad_pair_value = None if dynamic_gradients is None else dynamic_gradients[0]
    grad_pair_tangent = None if dynamic_gradients is None else dynamic_gradients[1]
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    wave = buffers.wave
    duration_row, pool_table, pool_durations = _tabulate_three_pool(
        tissue,
        duration,
        pools=pools,
        narrow=narrow,
        tangents=tangents,
        problems=wave,
    )
    row_count = 0 if pool_durations is None else pool_durations.numel()
    pool_bars = None
    if pool_table is not None:
        # A slot per problem the chunk carries, so the walk back accumulates
        # into memory it owns and no two programs contend for a row. Three
        # sets of twelve: the value cotangents, their directions, and the
        # value cotangents weighted by each event's own interval direction.
        pool_bars = torch.zeros(
            wave * row_count * 36, dtype=torch.float32, device=t1.device
        )

    problems = _problems_per_program(block_states)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        if pool_bars is not None:
            # The slots are per chunk, so each chunk starts from nothing.
            pool_bars.zero_()
        grid = (triton.cdiv(span, problems),)
        shape = dict(
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue, tangents),
            block_states=block_states,
            problems=problems,
            num_warps=1,
        )
        if real:
            _epg_real_vjp_jvp_kernel[grid](
                t1,
                t2,
                m0,
                b1,
                inversion_efficiency,
                diffusion,
                duration,
                kind,
                flip,
                action,
                output_index,
                shim_index,
                tangents[0],
                tangents[1],
                tangents[2],
                tangents[3],
                tangents[6],
                tangents[7],
                tangents[_DURATION_SEED],
                tangents[_FLIP_SEED],
                grad_imag,
                *grad_tissue,
                *grad_flip,
                *grad_duration,
                *trajectory,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                shim_rows=_shim_count(tissue),
                shimmed=_shim_count(tissue) > 1,
                **_only_scalars(_feature_flags(features, geometry)),
                **shape,
            )
        else:
            _epg_vjp_jvp_kernel[grid](
                *tissue,
                *events,
                t1 if table is None else table,
                kind if table_rows is None else table_rows,
                t1 if absorption is None else absorption,
                t1 if pairs is None else pairs,
                kind if pair_rows is None else pair_rows,
                t1 if pair_direction is None else pair_direction,
                t1 if grad_pair_value is None else grad_pair_value,
                t1 if grad_pair_tangent is None else grad_pair_tangent,
                *tangents,
                kind if duration_row is None else duration_row,
                t1 if pool_table is None else pool_table,
                t1 if pool_bars is None else pool_bars,
                t1 if pool_durations is None else pool_durations,
                row_count,
                grad_real,
                grad_imag,
                *grad_tissue,
                *grad_flip,
                *grad_phase,
                *grad_duration,
                *trajectory,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                geometry.flow_scale,
                geometry.washout_scale,
                1.0 if profile is None else profile.step,
                1.0 if lineshape is None else lineshape.step,
                shim_rows=_shim_count(tissue),
                shimmed=_shim_count(tissue) > 1,
                locations=1 if profile is None else profile.points,
                profiled=profile is not None and profile.bins > 0,
                profile_bins=0 if profile is None else profile.bins,
                dynamic=dynamic is not None,
                directed=dynamic_direction is not None,
                broadened=lineshape is not None and lineshape.bins > 0,
                lineshape_bins=0 if lineshape is None else lineshape.bins,
                pools=pools,
                narrow=narrow,
                tabulated=pool_table is not None,
                **_feature_flags(features, geometry),
                **shape,
            )

    # Plane 1 is the tangent part -> d/d(primal inputs); plane 0 the value part.
    return buffers.tissue_gradients(atom_count)


def simulate_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through the state machine on CUDA.

    ``tangents`` follows the differentiable-input order -- every tissue
    property, then event duration, flip and phase -- and the two returned
    tuples, gradients with respect to the primal inputs then to the tangent
    inputs, follow it too.

    ``real_axis`` of 1 selects the real-subspace adjoint. That representation
    divides the RF phase out, so it leaves ``b1_phase``, ``b0`` and ``phase`` at
    zero and callers must not ask for those; the complex adjoint produces every
    one of them.

    Gradients land through atomic accumulation, so repeated runs agree to
    floating-point tolerance rather than bit for bit.
    """
    atom_count = tissue[0].numel()
    gradients = None
    if dynamic is not None:
        held = dynamic.packed(tissue[0].device)
        gradients = (torch.zeros_like(held), torch.zeros_like(held))
    buffers = AdjointBuffers(
        events,
        atom_count,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        shims=_shim_count(tissue),
        pools=_pool_flag(lineshape, exchanging),
    )
    voxel_grads = simulate_vjp_jvp_into(
        tissue,
        events,
        tangents,
        grad_output,
        buffers,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        dynamic_gradients=gradients,
        features=features,
    )
    sides = tuple(
        (*voxels, *per_event)
        for voxels, per_event in zip(
            voxel_grads, buffers.event_gradients(), strict=True
        )
    )
    if gradients is None:
        return sides
    # The value plane is the adjoint and the tangent plane its own derivative,
    # which is the split the tissue gradients take; the sides come back in the
    # order the caller reads them, curvature first.
    return (*sides[0], gradients[1]), (*sides[1], gradients[0])
