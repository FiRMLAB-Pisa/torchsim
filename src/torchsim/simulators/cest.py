"""The Z-spectrum: Lorentzian lines in closed form, and pulsed saturation simulated."""

from __future__ import annotations

__all__ = ["CESTSimulator", "LorentzianSimulator"]

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt
import torch

from ..model import Simulator, SpinPhysics
from ..sequence import Delay, Excitation, Readout, SampledPulse, Spoil
from ..sequence._array import arrays, as_torch
from ._contrast import across_contrasts
from ._pulses import sample_count
from .gradient_echo import GAMMA_BAR_HZ_PER_T


class LorentzianSimulator(Simulator):
    """A Z-spectrum as equilibrium minus one Lorentzian line per pool.

    Each pool saturates the water signal along a Lorentzian line centred on
    its own offset [1]_, and the lines add:

    ``M0 (1 - sum_i A_i (W_i / 2)^2 / ((W_i / 2)^2 + (offset - shift_i)^2))``

    Pool ``i``, counted from one, exposes ``pool{i}_amplitude``,
    ``pool{i}_width`` (the full width at half maximum) and ``pool{i}_shift``.

    Parameters
    ----------
    pools:
        How many Lorentzian lines.
    **values
        Properties or sequence arguments to fix, as for any simulator.

    References
    ----------
    .. [1] Zaiss, M., Schmitt, B., Bachert, P., "Quantitative separation of
       CEST effect from magnetization transfer and spillover effects by
       Lorentzian-line-fit analysis of z-spectra", Journal of Magnetic
       Resonance 211.2 (2011), pp. 149-155.
       https://doi.org/10.1016/j.jmr.2011.05.001

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import LorentzianSimulator

        sequence = LorentzianSimulator(2, offsets=torch.linspace(-5.0, 5.0, 41))
        signal = sequence.simulate(
            pool1_amplitude=0.9, pool1_width=1.5, pool1_shift=0.0,
            pool2_amplitude=0.1, pool2_width=2.0, pool2_shift=3.5,
        )
        print(signal.shape)

    """

    properties = ("M0",)
    pools: int = 1

    def __init__(self, pools: int = 1, **values: Any) -> None:
        self.pools = int(pools)
        if self.pools < 1:
            raise ValueError(f"a Z-spectrum needs at least one pool, got {pools}")
        self.properties = (
            "M0",
            *(
                f"pool{index}_{what}"
                for index in range(1, self.pools + 1)
                for what in ("amplitude", "width", "shift")
            ),
        )
        super().__init__(**values)

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Evaluate the closed form, no state machine and no description."""
        return self._signal(properties, **arrays(self.played(**sequence)))

    def _signal(
        self,
        properties: Mapping[str, Any],
        *,
        offsets: npt.ArrayLike,
    ) -> torch.Tensor:
        """Return the saturated water signal at each offset.

        Parameters
        ----------
        properties:
            ``pool{i}_amplitude`` as a fraction of ``M0``, ``pool{i}_width``
            and ``pool{i}_shift`` in the unit of ``offsets``, and ``M0`` as a
            scaling.
        offsets:
            Saturation frequency offsets from water, conventionally in ppm.
        """
        held = across_contrasts(properties, offsets)
        remaining = 1.0
        for index in range(1, self.pools + 1):
            amplitude = held[f"pool{index}_amplitude"]
            half = held[f"pool{index}_width"] / 2
            distance = offsets - held[f"pool{index}_shift"]
            remaining = remaining - amplitude * half**2 / (half**2 + distance**2)
        return held.get("M0", 1.0) * remaining


class CESTSimulator(Simulator):
    """A Z-spectrum simulated through a train of saturation pulses, pool by pool.

    At each offset the magnetization starts from equilibrium, a train of
    rectangular pulses saturates it while the free water and every exchanging
    pool precess at their own frequency, relax and exchange, and the
    longitudinal magnetization left once a spoiled delay has passed is the
    point of the spectrum. The pools are the Bloch-McConnell model's [1]_:
    ``poolB_*`` to ``poolE_*`` beside the free water, each a fraction of
    ``M0`` with its own exchange rate, relaxation times and chemical shift in
    Hz. Each pulse starts at the phase the one before it ended at, so the
    train is continuous in phase over the time the pulses play and not over
    the delays between them, which is how a Pulseq CEST protocol accumulates
    it. This is the sequence BART's ``sim`` plays as ``CEST``.

    References
    ----------
    .. [1] Zaiss, M., Bachert, P., "Chemical exchange saturation transfer
       (CEST) and MR Z-spectroscopy in vivo: a review of theoretical
       approaches and methods", Physics in Medicine and Biology 58.22 (2013),
       pp. R221-R269. https://doi.org/10.1088/0031-9155/58/22/R221

    Examples
    --------
    .. exec::

        import torch
        from torchsim.simulators import CESTSimulator

        sequence = CESTSimulator(offsets=torch.linspace(-5.0, 5.0, 21))
        spectrum = sequence.simulate(
            T1=1300.0, T2=75.0,
            poolB_fraction=0.001, poolB_exchange=1000.0, poolB_T1=1300.0,
            poolB_T2=10.0, poolB_shift=3.5 * 127.7,
        )
        print(spectrum.shape)

    """

    model = SpinPhysics(
        properties={
            "T1": "t1_ms",
            "T2": "t2_ms",
            "M0": "m0",
            "B1": "b1",
            "B0": "b0_hz",
        },
    )
    # The pulses are played in the rotating frame and the delay before the
    # read spoils, so no order but zero ever holds anything.
    states = 1

    def layout(
        self,
        *,
        offsets: npt.ArrayLike,
        field_strength: float = 3.0,
        B1sat: float = 1.0,
        npulses: int = 1,
        pulse_duration: float = 100.0,
        interpulse_delay: float = 0.0,
        recovery: float = 6.5,
        dwell: float = 0.01,
    ) -> list:
        """Return one saturation train and read per offset.

        Parameters
        ----------
        offsets : array-like
            Saturation frequency offsets from the free water, in ppm.
        field_strength : float, optional
            Main field in tesla, which turns an offset in ppm into Hz.
        B1sat : float, optional
            Amplitude of each saturation pulse in microtesla.
        npulses : int, optional
            Saturation pulses per offset.
        pulse_duration : float, optional
            Duration of each pulse in milliseconds.
        interpulse_delay : float, optional
            Time between two pulses in milliseconds.
        recovery : float, optional
            Time in milliseconds from the end of the last pulse to the read,
            over which the transverse magnetization is spoiled.
        dwell : float, optional
            How long each sample of a pulse is held, in milliseconds.

        Raises
        ------
        ValueError
            If a pulse is not a whole number of samples.
        """
        dwell_s = 1e-3 * dwell
        count = sample_count(1e-3 * pulse_duration, dwell_s)
        nutation = 2 * torch.pi * GAMMA_BAR_HZ_PER_T * 1e-6 * as_torch(B1sat)
        waveform = nutation * torch.ones(count, dtype=torch.float64)
        offset_hz = (
            GAMMA_BAR_HZ_PER_T
            * 1e-6
            * field_strength
            * torch.atleast_1d(as_torch(offsets)).to(torch.float64)
        )
        parts: list = []
        pulse_s = count * dwell_s
        for pulse in range(int(npulses)):
            # Each pulse takes up the phase the one before it ended at.
            carried = -2 * torch.pi * offset_hz * pulse * pulse_s
            parts.append(
                SampledPulse(waveform, dwell_s, phase_rad=carried, offset_hz=offset_hz)
            )
            if pulse < npulses - 1 and interpulse_delay > 0:
                parts.append(Delay(1e-3 * interpulse_delay))
        parts += [Spoil(), Delay(1e-3 * recovery), Excitation(torch.pi / 2), Readout()]
        return parts

    def evaluate(self, properties: Mapping[str, Any], **sequence: Any) -> torch.Tensor:
        """Return the longitudinal magnetization, offsets last."""
        # One train per offset, each read once: (offsets, voxels) or, for a
        # single voxel, (offsets,).
        signal = super().evaluate(properties, **sequence)[..., 0].movedim(0, -1)
        # The read is a pi/2 pulse, which the transmit field scales and turns
        # like every other, so it lays Mz along -i exp(i phase) sin(b1 pi/2).
        tissue = self.model.tissue(properties)
        b1 = torch.as_tensor(tissue.b1, device=signal.device)
        phase = torch.as_tensor(tissue.b1_phase_rad, device=signal.device)
        read = -1j * torch.sin(b1 * torch.pi / 2) * torch.exp(1j * phase)
        batch = self._batch(properties)
        if batch:
            read = torch.broadcast_to(read, batch).reshape(-1, 1)
        return (signal / read.to(signal.dtype)).real
