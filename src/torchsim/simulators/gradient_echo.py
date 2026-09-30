"""Multi-gradient-echo water and fat, in closed form."""

from __future__ import annotations

__all__ = ["IRMultiGradientEchoSimulator", "MultiGradientEchoSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence._array import arrays
from ._contrast import across_contrasts

#: The proton gyromagnetic ratio over 2 pi, in Hz/T.
GAMMA_BAR_HZ_PER_T = 42.57747892e6

#: Multi-peak fat spectra: chemical shift from water in ppm, and relative
#: amplitude, one row per peak.
FAT_SPECTRA: dict[str, tuple[tuple[float, float], ...]] = {
    # Middleton, M. S., Hamilton, G., Bydder, M., Sirlin, C. B., "How much fat
    # is under the water peak in liver fat MR spectroscopy?", Proc. ISMRM 17
    # (2009), p. 4331.
    "middleton2009": (
        (-3.80, 0.087),
        (-3.40, 0.693),
        (-2.60, 0.128),
        (-1.94, 0.004),
        (-0.39, 0.039),
        (+0.60, 0.048),
    ),
    # Hamilton, G., Yokoo, T., Bydder, M., et al., "In vivo characterization of
    # the liver fat 1H MR spectrum", NMR in Biomedicine 24.7 (2011),
    # pp. 784-790. https://doi.org/10.1002/nbm.1622
    "hamilton2011": (
        (-3.80, 0.086),
        (-3.40, 0.537),
        (-2.60, 0.165),
        (-1.94, 0.046),
        (-0.39, 0.052),
        (+0.60, 0.114),
    ),
}


def fat_modulation(
    TE: torch.Tensor,  # noqa: N803
    field_strength: float | torch.Tensor,
    spectrum: str,
) -> torch.Tensor:
    """Return the complex fat signal at each echo time, relative to water.

    Parameters
    ----------
    TE:
        Echo times in milliseconds.
    field_strength:
        Main field in tesla.
    spectrum:
        A key of :data:`FAT_SPECTRA`.

    Returns
    -------
    torch.Tensor
        ``sum_p a_p exp(i 2 pi gamma_bar B0 delta_p TE)``, complex, shaped as
        ``TE``.

    Raises
    ------
    ValueError
        If ``spectrum`` is not a key of :data:`FAT_SPECTRA`.
    """
    if spectrum not in FAT_SPECTRA:
        raise ValueError(
            f"fat spectrum {spectrum!r} is not one of {sorted(FAT_SPECTRA)}"
        )
    echo_s = TE * 1e-3
    total = torch.zeros((), dtype=torch.complex64, device=echo_s.device)
    for shift_ppm, amplitude in FAT_SPECTRA[spectrum]:
        shift_hz = GAMMA_BAR_HZ_PER_T * field_strength * shift_ppm * 1e-6
        total = total + amplitude * torch.exp(1j * 2 * torch.pi * shift_hz * echo_s)
    return total


def _water_and_fat(
    held: Mapping[str, Any],
    water: Any,
    fat: Any,
    TE: torch.Tensor,  # noqa: N803
    field_strength: float | torch.Tensor,
    fat_spectrum: str,
) -> torch.Tensor:
    """Water and fat longitudinal terms, read out at the echo times."""
    echo_s = TE * 1e-3
    fraction = held.get("fat_fraction", 0.0)
    phase = torch.pi / 180.0 * torch.as_tensor(held.get("fat_phase", 0.0))
    water_decay = torch.exp(-1e3 / held["T2star"] * echo_s) if "T2star" in held else 1.0
    if "fat_T2star" in held:
        fat_decay = torch.exp(-1e3 / held["fat_T2star"] * echo_s)
    else:
        fat_decay = water_decay
    fat_signal = torch.exp(1j * phase) * fat_modulation(
        TE, field_strength, fat_spectrum
    )
    # The off-resonance map follows the Freeman-Hill convention and this
    # expression the Ernst-Anderson one, which turn the opposite way; the fat
    # spectrum's shifts are chemical shifts and turn with the positive sign.
    turn = torch.exp(-1j * 2 * torch.pi * held.get("B0", 0.0) * echo_s)
    signal = (
        1 - fraction
    ) * water * water_decay + fraction * fat * fat_signal * fat_decay
    return held.get("M0", 1.0) * signal * turn


class MultiGradientEchoSimulator(Simulator):
    """Water and fat, dephasing and decaying across a gradient-echo train.

    Fat is a multi-peak spectrum [1]_ at a fixed set of chemical shifts, so
    its signal at each echo is a known complex modulation and what is left to
    fit per voxel is how much fat there is, the transverse decay and the
    off-resonance. Each model BART's multi-echo fits use is a subset of the
    properties: water and fat with a shared or separate T2*, water alone with
    T2*, and off-resonance alone.

    ``fat_fraction`` and ``fat_phase`` span the water and fat signals: with a
    complex scale in front, water is ``(1 - fat_fraction)`` of it and fat
    ``fat_fraction`` of it rotated by ``fat_phase``. Leaving ``T2star`` out
    means no decay, and leaving ``fat_T2star`` out gives fat water's T2*.

    References
    ----------
    .. [1] Hamilton, G., Yokoo, T., Bydder, M., et al., "In vivo
       characterization of the liver fat 1H MR spectrum", NMR in Biomedicine
       24.7 (2011), pp. 784-790. https://doi.org/10.1002/nbm.1622

    Examples
    --------
    .. exec::

        from torchsim.simulators import MultiGradientEchoSimulator

        sequence = MultiGradientEchoSimulator(TE=(1.2, 2.4, 3.6, 4.8, 6.0, 7.2))
        signal = sequence.simulate(fat_fraction=(0.0, 0.2), T2star=30.0, B0=15.0)
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T2star": None,
            "M0": None,
            "B0": None,
            "fat_fraction": None,
            "fat_phase": None,
            "fat_T2star": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TE: float | npt.ArrayLike,
        field_strength: float | npt.ArrayLike = 3.0,
        fat_spectrum: str = "hamilton2011",
    ) -> torch.Tensor:
        """Return the transverse magnetization at each echo time.

        Parameters
        ----------
        properties:
            ``T2star`` and ``fat_T2star`` in milliseconds, ``B0`` in Hz,
            ``fat_fraction`` as a fraction of the signal, ``fat_phase`` in
            degrees, and ``M0`` as a scaling.
        TE:
            Echo times in milliseconds.
        field_strength:
            Main field in tesla, which sets the fat peaks' frequencies.
        fat_spectrum:
            A key of :data:`~torchsim.simulators.gradient_echo.FAT_SPECTRA`.
        """
        held = across_contrasts(properties, TE)
        return _water_and_fat(held, 1.0, 1.0, TE, field_strength, fat_spectrum)


class IRMultiGradientEchoSimulator(Simulator):
    """A multi-gradient-echo train read at a series of inversion times.

    Water and fat each recover from the inversion along their own apparent
    T1, ``1 - (1 + inv_efficiency) exp(-TI / T1)``, as a Look-Locker readout
    does, and each echo train then dephases and decays as in
    :class:`MultiGradientEchoSimulator`. ``TI`` and ``TE`` give one contrast
    per entry, so a train of echoes after each of several inversion times is
    the flattened grid of the two.

    ``fat_T1`` and ``fat_inv_efficiency`` default to water's, which is the
    model with one recovery for both.

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import IRMultiGradientEchoSimulator

        TI, TE = torch.meshgrid(
            torch.tensor([20.0, 200.0, 800.0, 2000.0]),
            torch.tensor([1.2, 2.4, 3.6]),
            indexing="ij",
        )
        sequence = IRMultiGradientEchoSimulator(TI=TI.flatten(), TE=TE.flatten())
        signal = sequence.simulate(T1=900.0, fat_T1=300.0, fat_fraction=0.2)
        print(signal.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": None,
            "T2star": None,
            "M0": None,
            "B0": None,
            "inv_efficiency": None,
            "fat_fraction": None,
            "fat_phase": None,
            "fat_T1": None,
            "fat_T2star": None,
            "fat_inv_efficiency": None,
        },
    )

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        TI: float | npt.ArrayLike,
        TE: float | npt.ArrayLike,
        field_strength: float | npt.ArrayLike = 3.0,
        fat_spectrum: str = "hamilton2011",
    ) -> torch.Tensor:
        """Return the transverse magnetization at each inversion and echo time.

        Parameters
        ----------
        properties:
            ``T1``, ``fat_T1``, ``T2star`` and ``fat_T2star`` in milliseconds,
            ``B0`` in Hz, ``inv_efficiency`` and ``fat_inv_efficiency`` as
            fractions of a perfect inversion, ``fat_fraction`` as a fraction
            of the signal, ``fat_phase`` in degrees, and ``M0`` as a scaling.
        TI:
            Inversion times in milliseconds.
        TE:
            Echo times in milliseconds.
        field_strength:
            Main field in tesla, which sets the fat peaks' frequencies.
        fat_spectrum:
            A key of :data:`~torchsim.simulators.gradient_echo.FAT_SPECTRA`.
        """
        held = across_contrasts(properties, TI, TE)
        inversion_s = TI * 1e-3
        water_rate = 1e3 / held["T1"]
        fat_rate = 1e3 / held["fat_T1"] if "fat_T1" in held else water_rate
        efficiency = held.get("inv_efficiency", 1.0)
        fat_efficiency = held.get("fat_inv_efficiency", efficiency)
        water = 1 - (1 + efficiency) * torch.exp(-water_rate * inversion_s)
        fat = 1 - (1 + fat_efficiency) * torch.exp(-fat_rate * inversion_s)
        return _water_and_fat(held, water, fat, TE, field_strength, fat_spectrum)
