"""
======================
Writing a Signal Model
======================

The scope of this notebook is to write a simulator TorchSim does not ship.

There are two things to say. **Which operator plays each kind of event** --
what an excitation is, what a sample is -- and **what order they are played
in**. The base class does the rest: it resolves the layout into an event
stream, rebinds values onto it, holds the derivatives, places the work on a
device, and reads a sequence back from a description a scanner streamed.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# The base class, and the operators a layout is written from.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# Every figure is drawn at the width of the documentation column, so none of
# them is scaled on the way in and type is the same size throughout.
PAGE_WIDTH = 8.6  # inches

# Figures are read at gallery scale, so the type sizes are set once here.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.6),
        "savefig.dpi": 110,
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 19,
        "figure.constrained_layout.use": True,
    }
)


def key(axes, ncols=1):
    """The legend above what it describes, clear of the curves and the titles.

    Takes a figure, where every panel is showing the same series, and puts one
    legend over the whole of it. Takes an axis, or several, where the panels
    differ, and puts a legend over each -- every titled panel in the figure
    then ends up with the same padding, so the titles line up whether or not
    that panel carries one, which is only known once it has been laid out.
    """
    if hasattr(axes, "add_subplot"):
        handles, labels = axes.axes[0].get_legend_handles_labels()
        return axes.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=ncols,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.4,
        )
    axes = [axes] if hasattr(axes, "get_legend_handles_labels") else list(axes)
    figure = axes[0].figure
    legends = [
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=ncols,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.6,
            columnspacing=1.4,
        )
        for axis in axes
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tallest = max(legend.get_window_extent(renderer).height for legend in legends)
    for axis in figure.axes:
        if axis.get_title():
            axis.set_title(axis.get_title(), pad=72.0 * tallest / figure.dpi + 4.0)
    return legends


# sphinx_gallery_end_ignore

import numpy as np
import torch

from torchsim import (
    Delay,
    Excitation,
    Inversion,
    SPGRReadout,
    SSFPFidReadout,
    Spoil,
)
from torchsim.model import Simulator

# %%
# Handlers
# --------
# A simulator says what plays each kind of event by naming it. The five
# readouts are what distinguishes one steady-state family from another: an
# :func:`~torchsim.SSFPFidReadout` winds one configuration order after every
# sample, an :func:`~torchsim.SSFPEchoReadout` winds it before, an
# :func:`~torchsim.SPGRReadout` spoils, a :func:`~torchsim.bSSFPReadout` leaves
# the states where they were, and an :func:`~torchsim.FSEReadout` samples a
# spin echo the refocusing pulses have already crushed around.
#
# Naming one is the whole of choosing between them, and it is also what says
# how an arriving stream is to be read -- the transport carries no gradients,
# so the handler is where the dephasing lives.
#
# Nothing is said here about the tissue. Every property a voxel has can be
# given to any simulator, and giving one is what turns its term on.
#

# %%
# Layout
# ------
# An :class:`~torchsim.model.Simulator` is the protocol. You do not
# write timestamps: ``layout`` returns the *operators* of one repetition in
# order, and the simulator turns the span each one holds into the timestamps a
# description carries.
#
# ``self.operators`` is the handler set named above. What ``layout`` produces
# is an event stream whose events carry their own action word, and from there
# the path is the fused one -- packing, the feature mask, offload and sharding.


class SSFPMRF(Simulator):
    """An Inversion, then one Excitation and one sample per repetition."""

    excitation = Excitation
    inversion = Inversion
    readout = SSFPFidReadout
    states = 10

    def layout(self, *, flip, TR, TI=0.0):
        """Return one repetition's operators, in the order they are played."""
        angles = torch.deg2rad(torch.as_tensor(flip))
        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for index in range(angles.numel()):
            parts.append(self.operators.excitation(angles[index]))
            parts.append(self.operators.readout(duration_s=TR * 1e-3))
        return parts


# %%
# Running the simulator
# ---------------------
# Property and sequence arguments are given together and told apart by
# ``properties``. A scalar property is one voxel; an array is a map, and every
# voxel runs at once.
#
flip = np.concatenate(
    (np.linspace(5.0, 60.0, 350), np.linspace(60.0, 1.0, 350), np.ones(180))
)

sequence = SSFPMRF(flip=flip, TR=10.0, TI=20.0)
signal = sequence.simulate(T1=1000.0, T2=100.0)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(signal))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")
# sphinx_gallery_end_ignore

# %%
# The same call over a parameter map returns one row per voxel:
#
signals = sequence.simulate(
    T1=torch.tensor([500.0, 1000.0, 1500.0]),
    T2=torch.tensor([50.0, 100.0, 150.0]),
)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(signals.T))
plt.xlabel("TR index")
plt.ylabel("signal magnitude [a.u.]")
# sphinx_gallery_end_ignore

# %%
# Forward-mode derivatives
# ------------------------
# A Bloch simulation records far more samples than it takes parameters, so a
# derivative with respect to tissue is cheapest taken forwards: one directional
# derivative per property yields every voxel's derivative at once, and the cost
# is one pass per property rather than per voxel.
#
# That is what :meth:`~torchsim.model.Simulator.jacobian` does. A single name
# collapses the parameter axis; a sequence of names keeps it.
#
signal, jacobian = sequence.jacobian(("T1", "T2"), T1=1000.0, T2=100.0)

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(abs(jacobian.T))
plt.xlabel("TR index")
plt.ylabel("signal jacobian [a.u.]")
# sphinx_gallery_end_ignore

# %%
# Reverse-mode derivatives
# ------------------------
# The simulator optimization problem runs the other way: one scalar cost,
# many sequence parameters. That is reverse mode, and it is deliberately not
# wrapped -- build a cost on the signal and call ``backward()``. The engine
# reads which of its inputs carry a gradient and picks its kernel from that, so
# a layer here would only hide the choice.
#
schedule = torch.tensor(flip, dtype=torch.float32, requires_grad=True)
recorded = sequence.simulate(T1=1000.0, T2=100.0, flip=schedule)
loss = -recorded.abs().square().sum()
loss.backward()

# sphinx_gallery_start_ignore
plt.figure()
plt.plot(schedule.grad)
plt.xlabel("TR index")
plt.ylabel("d(loss) / d(flip) [1/deg]")
# sphinx_gallery_end_ignore

# %%
# Additional physics
# ------------------
# Nothing was declared about the tissue, and nothing had to be. Naming a
# property in the call is what turns its term on, so a second exchanging pool
# is four more names in the call and no change to the simulator:
#
two_pool = sequence.simulate(
    T1=1000.0,
    T2=100.0,
    poolB_fraction=0.2,
    poolB_exchange=20.0,
    poolB_T1=500.0,
    poolB_T2=20.0,
)

# sphinx_gallery_start_ignore
print(
    "  a second pool moves the train by "
    f"{float(np.abs(two_pool - signal).max() / np.abs(signal).max()):.1%}"
)
# sphinx_gallery_end_ignore

# %%
# What the whole vocabulary is, and what each term does to a train, is
# :ref:`the expanded-physics example
# <sphx_glr_generated_autoexamples_01-framework_02-expanded-physics.py>`.
#
# Validation against a closed form
# ---------------------------------
# A fingerprinting train has no closed form to check against, so here is a
# second simulator that does. Saturation recovery destroys whatever
# magnetization was there, waits, and reads what has come back -- and what it
# reads at a saturation time is :math:`M_0 \sin\alpha \, (1 - e^{-T_S/T_1})`,
# because the saturation leaves nothing behind and the recovery is undisturbed
# until the readout.
#
# The state machine has never been told what sequence these events add up to.
# It plays them, so agreeing with the closed form is a check rather than a
# tautology.
#
# ``@`` composes two operators into one, so the saturation reads as the single
# thing a physicist would name rather than as a pulse and a spoiler.


class SaturationRecovery(Simulator):
    """Saturate, wait, read what recovered -- once per saturation time.

    Parameters
    ----------
    TS : array-like
        Saturation times, in milliseconds, one per block.
    flip : float
        The readout flip angle, in degrees.
    phases : float or array-like, optional
        The readout phase, in degrees.
    """

    excitation = Excitation
    readout = SPGRReadout
    # Every block begins by destroying the transverse magnetization, so nothing
    # is carried in a dephased configuration and one order is the whole state.
    states = 1

    def layout(self, *, TS, flip, phases=0.0):
        """Return one saturate-wait-read block per saturation time."""
        waits = torch.atleast_1d(torch.as_tensor(TS)) * 1e-3
        angle = torch.deg2rad(torch.as_tensor(flip)).broadcast_to(waits.shape)
        turn = torch.deg2rad(torch.as_tensor(phases)).broadcast_to(waits.shape)

        saturate = Excitation(torch.pi / 2) @ Spoil()
        parts = []
        for index in range(waits.numel()):
            parts.append(saturate)
            parts.append(Delay(waits[index]))
            parts.append(self.operators.excitation(angle[index], turn[index]))
            parts.append(self.operators.readout(turn[index]))
        return parts


# %%
#
SATURATION_TIMES = torch.tensor([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0])
FLIP_DEG = 10.0
T1_MS = torch.tensor([830.0, 1330.0, 4000.0])
NAMES = ("white matter", "grey matter", "CSF")

recovery = SaturationRecovery(TS=SATURATION_TIMES, flip=FLIP_DEG)
recovered = recovery.simulate(T1=830.0, T2=80.0, M0=1.0).abs()
closed_form = torch.sin(torch.deg2rad(torch.tensor(FLIP_DEG))) * (
    1.0 - torch.exp(-SATURATION_TIMES / 830.0)
)

times = torch.logspace(1.3, 3.7, 60)
curves = SaturationRecovery(TS=times, flip=FLIP_DEG).simulate(
    T1=T1_MS, T2=torch.tensor([80.0, 110.0, 2000.0]), M0=1.0
)

# sphinx_gallery_start_ignore
print(
    "  largest disagreement with the closed form: "
    f"{float((recovered - closed_form).abs().max()):.2e}"
)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.0))
for row, name in enumerate(NAMES):
    axis.plot(times.numpy(), curves[row].abs().numpy(), label=name)
    analytic = torch.sin(torch.deg2rad(torch.tensor(FLIP_DEG))) * (
        1.0 - torch.exp(-times / T1_MS[row])
    )
    axis.plot(times.numpy(), analytic.numpy(), "--k", lw=0.8)
axis.set(
    xlabel="saturation time [ms]",
    ylabel="|signal|",
    xscale="log",
    title="simulated, against the closed form (dashed)",
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
# sphinx_gallery_end_ignore

# %%
# Inspecting the event stream
# ---------------------------
# :meth:`~torchsim.model.Simulator.describe` returns the event stream, which is
# the same object a sequence arriving from a scanner is read into: a timestamp
# and an action word on every event. It is worth looking at once, because a
# sequence that plays the wrong thing is far easier to see here than in the
# signal it produces.
#
description = recovery.describe(TS=SATURATION_TIMES, flip=FLIP_DEG)

# sphinx_gallery_start_ignore
print(
    f"  {len(description.events)} events, "
    f"{description.tr_duration_us * 1e-3:.0f} ms long"
)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
description.plot(axis)
axis.set(title="one saturate-wait-read block per saturation time", ylim=(-8, 100))
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
# Building from a description
# ---------------------------
# A description that came from somewhere else -- an MRD file, a Pulseq export,
# a scanner's own stream -- becomes a simulator through
# :meth:`~torchsim.model.Simulator.from_description`. No layout is walked: the
# events are re-emitted through this model's handlers, which is what puts the
# gradients back that the transport does not carry.
#
# It does not reproduce this sequence exactly, and the gap is the lesson. A
# handler reinstates what a pulse or a sample implies; the spoiler inside the
# saturation block is neither, so nothing reinstates it -- and a stream
# arriving from a scanner could not have carried it either. A preparation that
# has to survive the round trip belongs in a pulse the wire can name.
#
arrived = SaturationRecovery.from_description(description, states=1)

# sphinx_gallery_start_ignore
print(
    "  the same stream, read back through the handlers, differs by "
    f"{float((arrived.simulate(T1=830.0, T2=80.0, M0=1.0).abs() - recovered).abs().max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
# Functional wrapper
# ------------------
# The shipped models come with one, and yours can too:


def ssfp_mrf_sim(flip, TR, T1, T2, TI=0.0, diff=None):
    """Simulate an inversion-prepared SSFP train, and differentiate it."""
    sequence = SSFPMRF(flip=flip, TR=TR, TI=TI)
    if diff is None:
        return sequence.simulate(T1=T1, T2=T2)
    return sequence.jacobian(diff, T1=T1, T2=T2)


signal, jacobian = ssfp_mrf_sim(flip, 10.0, 1000.0, 100.0, diff=("T1", "T2"))
# signal is (repetitions,); jacobian is (2, repetitions), one row per property
