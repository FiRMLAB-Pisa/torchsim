"""The Triton kernels for tabulated pools, held to the C++ ones.

Each case runs one pass of both backends on the same tables in Triton's CPU
interpreter, in a process of its own because Triton reads the interpreter flag
at import. The C++ kernels are held to the state machine written out in torch
by ``test_many_pools.py``, so agreeing with them is agreeing with that.

They take seconds each rather than milliseconds, so they are opt-in:
``pytest -m interpreted``.
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

PASSES = ("forward", "jvp", "vjp", "vjp_jvp")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["TRITON_INTERPRET"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(TESTS)])
    return subprocess.run(
        [sys.executable, "-m", "utils.interpreted_pools", *arguments],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=ROOT,
        env=environment,
    )


def _held(*arguments: str) -> None:
    finished = _run(*arguments)
    assert finished.returncode == 0, (
        f"{' '.join(arguments)} failed:\n{finished.stdout}\n{finished.stderr}"
    )
    # A case that fell out before its comparison prints no terminator.
    assert finished.stdout.rstrip().endswith("checked"), finished.stdout


@pytest.mark.interpreted
@pytest.mark.parametrize("pass_name", PASSES)
@pytest.mark.parametrize(
    ("pools", "rotation"),
    [("2f", "instant"), ("3s", "profile"), ("4s", "dynamic"), ("4s", "shimmed")],
)
def test_the_triton_kernels_agree_with_the_cpp_ones(pass_name, pools, rotation):
    """Two to four exchanging pools, with and without a semisolid one, turned
    by a hard pulse, a tabulated one, a rotation per voxel and a transmit row
    per shim."""
    _held(pass_name, pools, rotation)


@pytest.mark.interpreted
@pytest.mark.parametrize("pass_name", ("vjp", "vjp_jvp"))
def test_an_adjoint_in_waves_over_two_trains_agrees_with_the_cpp_one(pass_name):
    """Every problem recorded in a wave of its own, over two trains of
    different lengths: the trajectory and the per-problem table cotangents are
    indexed from the wave's base, and summed over the trains."""
    _held(pass_name, "4s", "dynamic", "trains", "chunked")


@pytest.mark.interpreted
@pytest.mark.parametrize("pass_name", PASSES)
def test_the_kernels_agree_with_every_optional_term_off(pass_name):
    """A tissue declaring nothing and a sequence moving nothing compiles every
    optional term out of both kernels."""
    _held(pass_name, "3s", "instant", "undeclared")
