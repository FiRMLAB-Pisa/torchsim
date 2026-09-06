"""The Triton kernels, held to what they specialize or to an oracle.

Triton's CPU interpreter runs the kernels in Python over host tensors, so
these reach the plumbing without a GPU and without invalidating a compile
cache: the argument alignment of the launchers, the trajectory planes a pool
model claims, the operator table's row indexing, and the washout and
recoveries the reading event applies.

They are slow -- a minute each, since the interpreter walks every element in
Python -- so they are opt-in: ``pytest -m interpreted``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent.parent
ROOT = TESTS.parent
TIMEOUT_S = 900


def _run(case: str) -> subprocess.CompletedProcess[str]:
    """One case, in a process that sets the interpreter flag before Triton."""
    environment = dict(os.environ)
    # Triton reads this at import, so it cannot be set from inside a test.
    environment["TRITON_INTERPRET"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(TESTS)])
    return subprocess.run(
        [sys.executable, "-m", "utils.interpreted", case],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=ROOT,
        env=environment,
    )


@pytest.mark.interpreted
@pytest.mark.parametrize(
    "case",
    [
        "narrow",
        "wide",
        "chunked",
        "unread",
        "streamed",
        "washed",
        "shimmed",
        "profiled",
        "one_pool",
        "two_pools",
        "real",
        "real_shimmed",
        "spoiled",
        "narrowed",
    ],
)
def test_the_kernels_agree_with_what_they_specialize(case: str) -> None:
    """Each case runs one launch two ways and holds the two to each other.

    The table cases differ only in where the operator comes from; the pool
    cases hold a kernel to an oracle sharing no code with it, and to the pass
    it specializes.

    ``narrow`` forces a table onto a train the launch-wide gate calls narrow,
    so every row takes the series and the two arms agree to the bit. ``wide``
    is the case that ships -- an inversion makes the launch decline the gate,
    and the table carries series rows beside a roots row. ``chunked`` is the
    same launch cut into chunks, which is what tells a chunk-local index for
    the cotangent table from a global one. ``unread`` poisons ``acos`` so a
    narrow launch that touched the three roots could not come back finite.
    ``streamed`` runs the chunked launcher, whose fixed positional list is
    what a grown kernel signature misaligns first. ``washed`` gives the
    interval a washout, so a pooled row has to carry its own attenuation
    rather than one. ``shimmed`` drives a transmit row per shim, which is
    what tells the three-pool row index from the shim row it sits beside.
    ``profiled`` turns a shaped pulse through its own table, which is the
    one launch that answers whether a table is read separately from how
    many knots it holds.
    ``one_pool`` and ``two_pools`` reach the pool models the table cases
    never do, against the packed reference and against the
    forward-over-reverse pass. ``real`` reaches the real-subspace kernels,
    which carry three real planes where every other case here carries four
    components -- against the reference, against the complex adjoint, and
    with the gradients the representation cannot hold held to exactly zero.
    ``real_shimmed`` gives those kernels a transmit row per shim and leaves
    the last row undriven, so a layout that is merely wide enough cannot pass
    for one that reads the index.
    """
    finished = _run(case)
    assert finished.returncode == 0, (
        f"{case} case failed:\n{finished.stdout}\n{finished.stderr}"
    )
    # A case that fell out before its checks prints no terminator.
    assert finished.stdout.rstrip().endswith("checked"), finished.stdout
