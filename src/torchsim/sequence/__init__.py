"""Sequence descriptions and differentiable state-machine simulation."""

from __future__ import annotations

__all__ = [
    "SequenceDescription",
    "ShimDefinition",
    "execution",
    "FSEReadout",
    "Operator",
    "bSSFPReadout",
    "Delay",
    "Dephase",
    "Excitation",
    "rf_definition",
    "compose_spinor",
    "Inversion",
    "offload",
    "Readout",
    "Refocusing",
    "Saturation",
    "SPGRReadout",
    "Spoil",
    "SSFPEchoReadout",
    "SSFPFidReadout",
]

from ._accelerators import distribute, execution, offload  # noqa: F401
from ._builders import (
    fse_description,  # noqa: F401
    mpnrage_description,  # noqa: F401
    mprage_description,  # noqa: F401
    mrf_description,  # noqa: F401
    spgr_description,  # noqa: F401
)
from ._description import (
    AdcRole,  # noqa: F401
    EventAction,  # noqa: F401
    EventType,  # noqa: F401
    RfDefinition,  # noqa: F401
    RfMode,  # noqa: F401
    RfShape,  # noqa: F401
    RfUse,  # noqa: F401
    SequenceDescription,
    SequenceEvent,  # noqa: F401
    ShimDefinition,
    decompress_shape,  # noqa: F401
    ideal_rf_definition,  # noqa: F401
    rf_definition,
)
from ._operators import (
    Delay,
    Dephase,
    Excitation,
    FSEReadout,
    Inversion,
    Operator,
    Readout,
    Refocusing,
    Saturation,
    SPGRReadout,
    Spoil,
    SSFPEchoReadout,
    SSFPFidReadout,
    bSSFPReadout,
    compose,  # noqa: F401
    module,  # noqa: F401
    operator,  # noqa: F401
    operator_names,  # noqa: F401
    register_operator,  # noqa: F401
)
from ._simulation import (
    EpgEngine,  # noqa: F401
    SimulationResult,  # noqa: F401
    TissueProperties,  # noqa: F401
    simulate_subspace,  # noqa: F401
)
from ._transition import (
    ExactSliceProfile,  # noqa: F401
    compose_spinor,
    exact_slice_profile,  # noqa: F401
)
