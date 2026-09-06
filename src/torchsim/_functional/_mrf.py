"""MR Fingerprinting simulator."""

__all__ = ["mrf_sim"]

import numpy.typing as npt
import torch

from ..simulators.mrf import MRFSimulator
from ._run import evaluated


def mrf_sim(
    flip: npt.ArrayLike,
    TR: float | npt.ArrayLike,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    diff: str | tuple[str] = None,
    B1: float | npt.ArrayLike = 1.0,
    inv_efficiency: float | npt.ArrayLike = 1.0,
    M0: float | npt.ArrayLike = 1.0,
    TI: float = 0.0,
    states: int = 10,
    nreps: int = 1,
    device: str | torch.device = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    SSFP MR Fingerprinting simulator wrapper.

    Parameters
    ----------
    flip : float | npt.ArrayLike
        Flip angle train in degrees.
    TR : float | npt.ArrayLike
        Repetition time in milliseconds.
    T1 : float | npt.ArrayLike
        Longitudinal relaxation time in milliseconds.
    T2 : float | npt.ArrayLike
        Transverse relaxation time in milliseconds.
    diff : str | tuple[str], optional
        Arguments to get the signal derivative with respect to.
        The default is ``None`` (no differentation).
    B1 : float | npt.ArrayLike, optional
        Flip angle scaling map, default is ``1.0``.
    inv_efficiency : float | npt.ArrayLike, optional
        Inversion efficiency map, default is ``1.0``.
    M0 : float or array-like, optional
        Proton density scaling factor, default is ``1.0``.
    TI : float | npt.ArrayLike, optional
        Inversion time in milliseconds.
        The default is ``0.0``.
    states : int, optional
        Number of EPG states to be retained.
        The default is ``10``.
    nreps : int, optional
        Number of simulation repetitions.
        The default is ``1``.
    device : str | torch.device, optional
        Computational device for simulation.
        The default is ``None`` (infer from input).

    Returns
    -------
    sig : npt.ArrayLike
        Signal evolution of shape ``(..., len(flip))``.
    jac : npt.ArrayLike
        Derivatives of signal wrt ``diff`` parameters,
        of shape ``(..., len(diff), len(flip))``.
        Not returned if ``diff`` is ``None``.

    """
    return evaluated(
        MRFSimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B1=B1,
        inv_efficiency=inv_efficiency,
        flip=flip,
        TR=TR,
        TI=TI,
        states=states,
        repetitions=nreps,
    )
