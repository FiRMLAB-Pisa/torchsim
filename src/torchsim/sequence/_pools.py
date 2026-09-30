"""Exchanging pools as relaxation and exchange operators tabulated per interval.

A tissue with more than one chemically exchanging pool beside the free water
reaches the kernels as the operators its intervals apply rather than as the
properties those operators are formed from. Over an interval of length ``tau``
the longitudinal states of the ``n`` pools -- the free water, the exchanging
pools and a semisolid pool if there is one -- relax and exchange as

``z(tau) = eq + exp(A tau) (z(0) - eq)``

and the transverse states of the ``m`` pools that carry any as
``F+(tau) = exp(C tau) F+(0)``, with ``F-`` taking the conjugate operator. The
exponentials are formed here in double precision for every distinct interval
length and every voxel, and the kernels multiply by them. Everything a voxel
does to all of its pools alike -- off-resonance, diffusion, flow and washout --
stays in the kernels.

The tables are ordinary tensors of the tissue, so autograd carries a
derivative through them to every property they were formed from, to any order
the kernels reach. A derivative along an interval's length is the kernels' own:
the tables carry ``dT/dtau`` and ``d2T/dtau2`` beside ``T`` when the durations
are differentiated, and an event reads the row of its own length.
"""

from __future__ import annotations

__all__ = ["PoolLayout", "PoolTables", "tabulate", "generators"]

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.forward_ad import unpack_dual

from ._parameters import EXTRA_POOLS, POOL_NAMES, TISSUE_NAMES

# Pools from which a tissue is tabulated rather than handed to the closed forms
# of the two- and three-pool kernels. The tests lower it to one to hold the
# tables to those closed forms.
TABULATE_FROM = 2

_TISSUE = {name: index for index, name in enumerate(TISSUE_NAMES)}
_POOLS = {name: index for index, name in enumerate(POOL_NAMES)}


@dataclass(frozen=True)
class PoolLayout:
    """How a pool table is laid out, which both kernels read it by.

    Per voxel the table holds the equilibrium of the ``n`` longitudinal pools,
    then one row per distinct interval length of ``n * n`` longitudinal
    operator entries, the ``n`` entries of that operator applied to the
    equilibrium, and ``m * m`` transverse entries as real and imaginary pairs.
    A table carrying slopes repeats every row as its first derivative along
    the interval and then again as its second. A voxel's entries are
    contiguous, since every kernel reads one voxel's operators at a time.

    Attributes
    ----------
    exchanging:
        Chemically exchanging pools beside the free water.
    semisolid:
        Whether the longitudinal pools end with a semisolid one.
    rows:
        Distinct interval lengths.
    blocks:
        One for the operators alone, three with their slopes.
    """

    exchanging: int
    semisolid: bool
    rows: int
    blocks: int = 1

    @property
    def longitudinal(self) -> int:
        return 1 + self.exchanging + int(self.semisolid)

    @property
    def transverse(self) -> int:
        return 1 + self.exchanging

    @property
    def row_width(self) -> int:
        n, m = self.longitudinal, self.transverse
        return n * n + n + 2 * m * m

    @property
    def width(self) -> int:
        return self.longitudinal + self.rows * self.row_width * self.blocks


@dataclass(frozen=True)
class PoolTables:
    """A tabulated tissue, and which row each event reads.

    Attributes
    ----------
    layout:
        How ``values`` is laid out.
    values:
        ``(voxels, width)`` float32.
    index:
        The row each event reads, int32, shaped like the event durations.
    direction:
        A direction along ``values``, for a pass that follows one.
    """

    layout: PoolLayout
    values: torch.Tensor
    index: torch.Tensor
    direction: torch.Tensor | None = None


def carries_derivative(value: torch.Tensor) -> bool:
    """Whether a gradient or a forward direction is tracked through ``value``."""
    return bool(value.requires_grad) or unpack_dual(value).tangent is not None


def generators(
    tissue: tuple[torch.Tensor, ...],
    extras: tuple[torch.Tensor, ...] | None,
    *,
    exchanging: int,
    semisolid: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The longitudinal and transverse generators, and the equilibrium.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    extras:
        The prepared buffers of pools C to E, in ``POOL_NAMES`` order, or
        ``None`` for a tissue with pool B alone.
    exchanging:
        How many exchanging pools to carry, from pool B on.
    semisolid:
        Whether to carry the semisolid pool as the last longitudinal one.

    Returns
    -------
    tuple
        ``A`` as ``(voxels, n, n)`` float64 in 1/s, ``C`` as ``(voxels, m, m)``
        complex128 in 1/s, and the equilibrium as ``(voxels, n)`` float64.
    """
    voxels = tissue[_TISSUE["t1_ms"]].numel()

    def take(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float64).reshape(-1).expand(voxels)

    def pool(letter: str, field: str) -> torch.Tensor:
        if letter == "b":
            return take(tissue[_TISSUE[field.format("b")]])
        if extras is None:
            raise ValueError(f"pool {letter} is carried but was not prepared")
        return take(extras[_POOLS[field.format(letter)]])

    letters = ("b", *EXTRA_POOLS)[:exchanging]
    fraction = [pool(letter, "pool_{}_fraction") for letter in letters]
    exchange = [pool(letter, "pool_{}_exchange_hz") for letter in letters]
    r1 = [1000.0 / pool(letter, "t1_pool_{}_ms") for letter in letters]
    r2 = [1000.0 / pool(letter, "t2_pool_{}_ms") for letter in letters]
    shift = [pool(letter, "pool_{}_shift_hz") for letter in letters]
    zero = torch.zeros(voxels, dtype=torch.float64, device=fraction[0].device)
    semisolid_fraction = take(tissue[_TISSUE["bound_fraction"]]) if semisolid else zero
    free = 1.0 - sum(fraction) - semisolid_fraction
    # A pool leaves the free water at its own share of the exchange rate and
    # returns at the free water's, which is what conserves the total.
    outward = [rate * share for rate, share in zip(exchange, fraction, strict=True)]
    inward = [rate * free for rate in exchange]
    r1_free = 1000.0 / take(tissue[_TISSUE["t1_ms"]])
    r2_free = 1000.0 / take(tissue[_TISSUE["t2_ms"]])

    n = 1 + exchanging + int(semisolid)
    rows = [[zero] * n for _ in range(n)]
    rows[0][0] = -sum(outward) - r1_free
    for pool_index in range(exchanging):
        at = 1 + pool_index
        rows[0][at] = inward[pool_index]
        rows[at][0] = outward[pool_index]
        rows[at][at] = -inward[pool_index] - r1[pool_index]
    if semisolid:
        rate = take(tissue[_TISSUE["bound_exchange_hz"]])
        at = n - 1
        rows[0][0] = rows[0][0] - rate * semisolid_fraction
        rows[0][at] = rate * free
        rows[at][0] = rate * semisolid_fraction
        rows[at][at] = -rate * free - 1000.0 / take(tissue[_TISSUE["t1_bound_ms"]])
    longitudinal = torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)

    # The semisolid pool has no transverse magnetization, so it is absent from
    # the transverse generator and from the free water's loss along it.
    m = 1 + exchanging
    across = [[zero.to(torch.complex128)] * m for _ in range(m)]
    across[0][0] = (-sum(outward) - r2_free).to(torch.complex128)
    for pool_index in range(exchanging):
        at = 1 + pool_index
        across[0][at] = inward[pool_index].to(torch.complex128)
        across[at][0] = outward[pool_index].to(torch.complex128)
        across[at][at] = torch.complex(
            -inward[pool_index] - r2[pool_index], -2.0 * torch.pi * shift[pool_index]
        )
    transverse = torch.stack([torch.stack(row, dim=-1) for row in across], dim=-2)

    equilibrium = torch.stack(
        [free, *fraction, *((semisolid_fraction,) if semisolid else ())], dim=-1
    )
    return longitudinal, transverse, equilibrium


def tabulate(
    tissue: tuple[torch.Tensor, ...],
    extras: tuple[torch.Tensor, ...] | None,
    duration: torch.Tensor,
    *,
    exchanging: int,
    semisolid: bool,
    slopes: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, PoolLayout]:
    """The operators every distinct interval applies, for every voxel.

    Parameters
    ----------
    tissue, extras, exchanging, semisolid:
        As :func:`generators` takes them.
    duration:
        The packed event durations, in seconds.
    slopes:
        Whether to carry the derivatives along the interval, which a pass
        differentiating the durations reads. ``None`` carries them when the
        durations carry a derivative.

    Returns
    -------
    tuple
        The table, the row index each event reads and the layout.
    """
    if slopes is None:
        slopes = carries_derivative(duration)
    lengths, inverse = torch.unique(duration.detach(), return_inverse=True)
    longitudinal, transverse, equilibrium = generators(
        tissue, extras, exchanging=exchanging, semisolid=semisolid
    )
    tau = lengths.to(device=longitudinal.device, dtype=torch.float64)
    voxels = equilibrium.shape[0]
    exponent = torch.linalg.matrix_exp(longitudinal[None] * tau[:, None, None, None])
    across = torch.linalg.matrix_exp(
        transverse[None] * tau[:, None, None, None].to(torch.complex128)
    )

    def row(z_operator: torch.Tensor, t_operator: torch.Tensor) -> torch.Tensor:
        restored = (z_operator @ equilibrium[None, :, :, None]).squeeze(-1)
        return torch.cat(
            (
                z_operator.flatten(-2),
                restored,
                torch.view_as_real(t_operator).flatten(-3),
            ),
            dim=-1,
        )

    blocks = [row(exponent, across)]
    if slopes:
        blocks.append(row(longitudinal[None] @ exponent, transverse[None] @ across))
        squared = longitudinal @ longitudinal
        blocks.append(
            row(squared[None] @ exponent, (transverse @ transverse)[None] @ across)
        )
    table = torch.cat(
        (
            equilibrium,
            *(block.permute(1, 0, 2).reshape(voxels, -1) for block in blocks),
        ),
        dim=-1,
    ).to(torch.float32)
    layout = PoolLayout(
        exchanging=exchanging,
        semisolid=semisolid,
        rows=int(lengths.numel()),
        blocks=len(blocks),
    )
    index = inverse.reshape(duration.shape).to(torch.int32).contiguous()
    return table.contiguous(), index, layout


def pools_of(
    values: torch.Tensor | None, index: torch.Tensor | None, layout: Any
) -> PoolTables | None:
    """The tables a kernel reads, from the pieces an autograd node carries."""
    if values is None or not isinstance(layout, PoolLayout):
        return None
    return PoolTables(layout=layout, values=values, index=index)
