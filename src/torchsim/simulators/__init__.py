"""The sequences that ship with TorchSim, as simulators.

Each names its protocol arguments at construction and its tissue properties
at the call, so parameter inference, sequence optimization and a
reconstruction pipeline take all of them the same way.
"""

from __future__ import annotations

__all__ = [
    "ASLSimulator",
    "DiffusionSimulator",
    "DoubleAngleSimulator",
    "FSESimulator",
    "HSFPSimulator",
    "IRMultiGradientEchoSimulator",
    "IRbSSFPSimulator",
    "InversionRecoverySimulator",
    "LookLockerSimulator",
    "LorentzianSimulator",
    "MOLLISimulator",
    "MP2RAGESimulator",
    "MPRAGESimulator",
    "MPnRAGESimulator",
    "MRFSimulator",
    "MultiEchoSimulator",
    "MultiGradientEchoSimulator",
    "SPGRSimulator",
    "SpinEchoSimulator",
    "bSSFPSimulator",
]

from .asl import ASLSimulator
from .bssfp import bSSFPSimulator
from .cest import LorentzianSimulator
from .diffusion import DiffusionSimulator
from .fse import FSESimulator
from .gradient_echo import IRMultiGradientEchoSimulator, MultiGradientEchoSimulator
from .hsfp import HSFPSimulator
from .looklocker import IRbSSFPSimulator, LookLockerSimulator, MOLLISimulator
from .mp2rage import MP2RAGESimulator
from .mpnrage import MPnRAGESimulator
from .mprage import MPRAGESimulator
from .mrf import MRFSimulator
from .relaxometry import (
    DoubleAngleSimulator,
    InversionRecoverySimulator,
    MultiEchoSimulator,
)
from .spgr import SPGRSimulator
from .spin_echo import SpinEchoSimulator
