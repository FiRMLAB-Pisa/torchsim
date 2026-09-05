"""
===========
Basic Usage
===========

The scope of this notebook is to showcase the basic functionalities of Torchsim,
including how to simulate a signal, calculating derivatives etc.

This example uses a fast spin echo that ships with TorchSim.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

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
import math
import time

import numpy as np
import torch

import torchsim
from torchsim.simulators import FSESimulator

# sphinx_gallery_start_ignore
from pathlib import Path

# The example sequence, wherever the gallery is running this file from.
SEQ_FILE = next(
    candidate
    for candidate in (Path("fse.seq"), Path("examples/01-framework/fse.seq"))
    if candidate.exists()
)
# sphinx_gallery_end_ignore

# %%
#
# Forward simulation
# ------------------
#
# First, we need to create a Simulator object instance. The constructor
# accepts the sequence parameters (in this example, number of echoes and echo spacing)
# and the tissue properties (T1, and T2).
#
# The simulator is parallelized: if N (T1, T2) pairs are provided,
# the corresponding N signals are computed in parallel. Properties support broadcasting:
# if we pass a list of T2s but just a single T1, the same T1 is used for all atoms.
#
ECHOES = 48
ESP_MS = 5.0

T1_MS = 1000.0
T2_MS = torch.tensor([80.0, 110.0, 2000.0])

simulator = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)

flip = torch.full((ECHOES,), 180.0)
signal180 = simulator.simulate(flip=flip)  # (3, 48): one row per tissue

# %%
#
# Changing the system parameters (e.g., simulating a refocusing train with 60° flip angle)
# will affect the resulting signal evolution
#
signal60 = simulator.simulate(flip=torch.full((ECHOES,), 60.0))

# sphinx_gallery_start_ignore
NAMES = ("white matter", "grey matter", "CSF")
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
for axis, values, title in (
    (axes[0], signal180, "a 180 degree train"),
    (axes[1], signal60, "a 60 degree train"),
):
    for row, name in enumerate(NAMES):
        axis.plot(values[row].abs().numpy(), label=name)
    axis.set(xlabel="Echo", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
key(figure, ncols=3)
# sphinx_gallery_end_ignore

# %%
#
# Derivative with respect to tissue parameters
# --------------------------------------------
#
# Torchsim allows to efficiently evaluate the derivative
# of the signal wrt input parameters, via :meth:`~torchsim.model.SignalModel.jacobian`.
# The desired derivatives can be specified by string:
#
signal, dT2 = simulator.jacobian("T2", flip=flip)  # dT2 is (3, 48) too

# %%
#
# Since the number of echoes is typically much larger than the number of differentiation parameters,
# forward mode differentiation is more efficient than the more common backward propagation.
#
# Here you can see a comparison with finite differences derivatives:
#
STEP_MS = 1.0
signal_plus = simulator.simulate(flip=flip, T2=T2_MS + STEP_MS)
dT2_finite = (signal_plus - signal) / STEP_MS

# sphinx_gallery_start_ignore
discrepancy = (dT2 - dT2_finite).abs().max() / dT2.abs().max()
print(f"largest disagreement with a {STEP_MS} ms step: {float(discrepancy):.2e}")
# sphinx_gallery_end_ignore


# %%
#
# Approaching steady state
# ------------------------
#
# Everything so far started from equilibrium. A scanner does not: it plays the
# train over and over, and what it records is the state the train has settled
# into. Reaching that by playing it out is hundreds of repetitions, every one
# of them the full cost of the sequence.
#
# ``repetitions="auto"`` reads the limit off a handful of playings instead. A
# settled signal is a constant plus decaying modes, so finitely many terms fix
# where it is going, and the answer is the one that running there arrives at.
#
settling = FSESimulator(ESP=ESP_MS, TR=500.0, T1=T1_MS, T2=T2_MS, states=20)
first_pass = settling.simulate(flip=flip)
settled = settling.simulate(flip=flip, repetitions="auto")
played_out = settling.simulate(flip=flip, repetitions=200)

# sphinx_gallery_start_ignore
print(f"  one playing          {float(first_pass[0].abs().max()):.5f}")
print(f'  repetitions="auto"   {float(settled[0].abs().max()):.5f}')
print(f"  200 playings         {float(played_out[0].abs().max()):.5f}")
print(
    "  the last two agree to "
    f"{float((settled - played_out).abs().max() / played_out.abs().max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# The first playing is wrong by a fraction that a short TR makes large, which
# is what a simulation of a steady-state sequence gets wrong if it starts from
# equilibrium and stops. Every shipped simulator takes the setting, and it
# costs a fraction of what running there costs.
#


# %%
#
# Functional wrapper
# ------------------
#
# Every sequence that ships also has a function, for the case where there is
# nothing to reuse: it takes the protocol and the tissue together, returns the
# signal, and returns the derivative too if ``diff`` names a property. It is
# the object above with the construction folded in, so the answer is the same
# to the bit.
#
signal, dT2 = torchsim.fse_sim(
    flip=flip, ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, diff="T2"
)

# sphinx_gallery_start_ignore
reference, reference_dT2 = simulator.jacobian("T2", flip=flip)
print(
    f"agrees with the simulator to "
    f"{float((torch.as_tensor(dT2) - reference_dT2).abs().max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# Reach for it when a sequence is simulated once and nothing about it is being
# varied. The object is what a loop wants: it resolves the event stream on its
# first call and rebinds only the numbers that change afterwards, which is
# worth about eight times the whole call to a design or a dictionary sweep.
#

# %%
#
# Performance tweaking
# --------------------------
#
# Everything so far took the defaults. Four settings decide what the run costs
# and how exact it is, and each is given to the constructor or to the call.
#
# ``states`` is how many configuration orders are carried. A refocused train
# winds one order per interval, and a pulse that is not a perfect 180 degrees
# splits the magnetization down every pathway those orders describe, so the
# answer is only as good as the number kept. Too few is a wrong answer rather
# than a slow one. Held against a train carrying far more than it needs:
#
signal_ref = simulator.simulate(flip=flip)
converged = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, states=64).simulate(
    flip=flip
)
ORDERS = (4, 10, 16, 32, 48)
truncated = [
    FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS, states=orders)
    for orders in ORDERS
]

# sphinx_gallery_start_ignore
for orders, sequence in zip(ORDERS, truncated, strict=True):
    drift = float((sequence.simulate(flip=flip) - converged).abs().max())
    print(
        f"  {orders:2d} orders   {drift / float(converged.abs().max()):.1e} from converged"
    )
# sphinx_gallery_end_ignore

# %%
#
# It lands exactly at forty-eight, which is the number of echoes: a train that
# winds one order per interval can populate one more pathway per echo and no
# more, so carrying more orders than the train has intervals changes nothing.
# A spoiled sequence is the other case -- it discards the transverse orders
# every repetition, so a handful is enough however long it runs.
#
# The shipped default is chosen for the refocused trains these simulators are
# written for, and a 60 degree train is not one of them. It is the first
# setting to raise when a signal looks wrong late in an echo train, and the
# check above -- run once against a larger number -- is how you find out rather
# than assume.

# %%
#
# A simulator is worth holding on to. The structure of a sequence -- the order
# of its events, which of them record, how far it winds -- is settled the first
# time it runs and the numbers are rebound onto it afterwards, so a sweep that
# changes only flip angles never walks the event stream again. That happens by
# itself; what it is worth grows with the sequence, and a 500-repetition
# fingerprinting schedule is where it decides whether a dictionary sweep takes
# minutes or hours.
#
held = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)
rebuilt = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)

for name, candidate in (("held", held), ("rebuilt anew", rebuilt)):
    candidate.simulate(flip=flip)  # the first call is where the structure is read
    start = time.perf_counter()
    for _ in range(20):
        if name != "held":
            candidate = FSESimulator(ESP=ESP_MS, TR=3000.0, T1=T1_MS, T2=T2_MS)
        candidate.simulate(flip=flip)
    print(f"  {name:14s} {1e3 * (time.perf_counter() - start) / 20:6.2f} ms a call")

# %%
#
# ``execution`` says where the work goes. ``"auto"`` weighs the problem against
# what the cards have free -- work too small to repay a launch stays on the
# host, work that fits crosses in one piece, work that does not is streamed
# through in chunks. Naming a device insists on it, and a block settles it for
# everything inside:
#
with torchsim.execution("cpu"):
    on_the_host = simulator.simulate(flip=flip)

print(
    f"  forced onto the host, agrees to {float((on_the_host - signal_ref).abs().max()):.1e}"
)

# %%
#
# ``stream`` and ``budget_bytes`` are for a volume rather than a dictionary:
# the first insists the run be cut into chunks even where it would have fit,
# the second caps what a chunk may hold. A whole-brain map at a hundred
# thousand voxels is the case they exist for, and it is written like this:
#
if torch.cuda.is_available():
    with torchsim.execution("cuda", stream=True, budget_bytes=1 << 28):
        streamed = simulator.simulate(flip=flip)
    agreement = float((streamed.cpu() - signal_ref).abs().max())
    print(f"  streamed through a card in 256 MiB chunks, agrees to {agreement:.1e}")
else:
    print("  no card here, so the streamed run is skipped")

# %%
#
# Simulating a sequence you did not write
# ---------------------------------------
#
# Every call so far named the sequence: ``ESP``, ``TR``, the flip train. That
# is fine when you wrote it, and wrong when it came from somewhere else -- a
# scan does not need its parameters retyped, it needs them read.
#
# The MRD client streams a **sequence description**, having taken it from the
# Pulseq sequence the scanner is running. It is a repetition's worth of events:
# time passing, an RF pulse, an ADC window, each with a timestamp and the
# numbers that kind of event carries, plus the pulse shapes and transmit shims
# they name.
#
# Echo spacing, echo train length, refocusing angle, TR and pulse shapes are
# all in there. Nothing about the sequence has to be given again.
#
described = simulator.describe(flip=flip, ESP=ESP_MS, TR=3000.0)

# sphinx_gallery_start_ignore
print(
    f"  {len(described.events)} events over "
    f"{described.tr_duration_us * 1e-3:.0f} ms, "
    f"{len(described.rf_definitions)} pulse shape, "
    f"{len(described.shim_definitions)} shims"
)
# sphinx_gallery_end_ignore

# %%
#
# You do not have to read it. ``describe`` here only shows what a stream looks
# like; :meth:`~torchsim.SequenceDescription.plot` draws one, which is the way
# to check that a layout laid down what you meant.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.axis("off")
axis.set(xlim=(0, 1), ylim=(0, 1))


def _box(x, y, text, colour, size=13.0):
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colour, edgecolor="none"),
    )


def _arrow(start_at, end_at, label=None, side="right"):
    axis.annotate(
        "",
        xy=end_at,
        xytext=start_at,
        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.8, shrinkA=2, shrinkB=2),
    )
    if label:
        middle = ((start_at[0] + end_at[0]) / 2, (start_at[1] + end_at[1]) / 2)
        axis.text(
            middle[0] + (0.015 if side == "right" else -0.015),
            middle[1],
            label,
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=11.5,
            color="0.35",
            style="italic",
        )


_box(0.22, 0.92, "a Pulseq design\n.seq, or a script", "tab:blue")
_box(
    0.78,
    0.92,
    "the scanner's SEQDESC stream\nwaveforms 999 / 1000 / 1002 / 1005",
    "tab:orange",
    11.5,
)
_box(0.50, 0.46, "SequenceDescription", "0.25", 15.0)
axis.text(
    0.50,
    0.32,
    "events  ·  rf_definitions  ·  shim_definitions",
    ha="center",
    fontsize=11.5,
    color="0.35",
)
_box(0.50, 0.12, "Simulator.from_description", "tab:green", 14.0)
_arrow((0.22, 0.80), (0.44, 0.56), "sequence_descriptor()", side="left")
_arrow((0.78, 0.80), (0.56, 0.56), "decode_sequence_description()", side="right")
_arrow((0.50, 0.27), (0.50, 0.19))

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
FIRST = 6
described.plot(axis, upto_s=FIRST * ESP_MS * 1e-3)
axis.set(title=f"the first {FIRST} echoes of the description above", ylim=(-8, 105))
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# A description normally arrives; it is not typed. Writing one by hand once is
# worth it only to see that there is nothing else in it --
# :meth:`~torchsim.SequenceDescription.from_operators` lays them out.
#
by_hand = torchsim.SequenceDescription.from_operators(
    torchsim.Excitation(math.pi / 2, math.pi / 2),
    *[
        part
        for _ in range(ECHOES)
        for part in (
            torchsim.Delay(0.5 * ESP_MS * 1e-3),
            torchsim.Refocusing(math.radians(60.0), 0.0),
            torchsim.Delay(0.5 * ESP_MS * 1e-3),
            torchsim.Readout(0.0),
        )
    ],
)

print(
    f"  written by hand: {len(by_hand.events)} events over "
    f"{by_hand.tr_duration_us * 1e-3:.0f} ms"
)

# %%
#
# :meth:`~torchsim.model.Simulator.from_description` runs one, and the only
# thing given to it is the tissue. The events are already concrete -- each
# carries the action word saying whether it winds, spoils or records -- so no
# layout is walked and no sequence parameter is inferred.
#
# Which simulator you call it on is the whole of what you choose, and it is
# not a formality. A description says an RF pulse was played, tagged with the
# use its designer gave it, and an ADC window was opened. It says nothing
# about the gradients between them, because the transport carries none.
#
# The dephasing lives in the handlers instead: a refocused train crushes
# either side of its refocusing pulses, an unbalanced one winds an order after
# every sample, a spoiled one discards the transverse states. Naming
# ``FSESimulator`` is how you say which of those the events are to be read as.
#
from_stream = FSESimulator.from_description(described, states=10, T1=T1_MS, T2=T2_MS)
streamed_signal = from_stream.simulate()

# %%
#
# It differentiates like anything else, because the derivative follows from the
# events and not from who wrote them:
#
_, streamed_dT2 = from_stream.jacobian("T2")

# sphinx_gallery_start_ignore
print(f"  signal {tuple(streamed_signal.shape)}, dT2 {tuple(streamed_dT2.shape)}")
# sphinx_gallery_end_ignore

# %%
#
# One thing to know when comparing it against the shipped simulator: a
# description carries the events, and a simulator may carry physics *around*
# them. :class:`~torchsim.simulators.FSESimulator` folds in the recovery
# between one train and the next in closed form, which is not an event and so
# is not in the stream. The shape of the train is the same; the driven
# equilibrium the shipped object adds does not come along.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(1, 1, figsize=(PAGE_WIDTH, 3.2))
axis.plot(signal_ref[0].abs().numpy(), "-k", lw=2, label="FSESimulator")
axis.plot(
    streamed_signal[0].abs().numpy(), "--r", lw=2, label="from the description alone"
)
axis.set(
    xlabel="Echo", ylabel="|signal|", title="white matter, the same train both ways"
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
# From a Pulseq file
# ------------------
#
# The stream a scanner sends is read off the Pulseq sequence it is running, so
# the same events can be read from the ``.seq`` file directly -- which is what
# to do when the scan has not been run yet. The file is parsed by pypulseq,
# which also computes its trajectory: ``pip install torchsim[pulseq]``.
#
# The file states how many blocks one repetition holds, in its ``TRSize``
# definition, so nothing is searched for. What is read off the trajectory is
# what Pulseq does not write down: which ADC sample each readout passes
# through k = 0 in, which is the echo the timestamp goes on.
#
train = FSESimulator.from_pulseq(SEQ_FILE, states=20)
from_file = train.simulate(T1=T1_MS, T2=T2_MS)

# sphinx_gallery_start_ignore
read = train.describe()
spacing = np.diff([event.timestamp_us for event in read.adc_events]) * 1e-3
print(f"  read from {SEQ_FILE.name}:")
print(f"    {len(read.events)} events over {read.tr_duration_us * 1e-3:.0f} ms")
print(f"    {len(read.adc_events)} echoes, {spacing.mean():.2f} ms apart")
print(f"    {len(read.rf_definitions)} pulse shapes")

figure, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 5.4))
read.plot(axes[0])
axes[0].set(title=f"one repetition of {SEQ_FILE.name}", ylim=(-8, 200))
key(axes[0], ncols=2)
axes[1].plot(np.cumsum(np.r_[spacing[0], spacing]), from_file[0].abs().numpy(), "o-")
axes[1].set(
    xlabel="time after excitation [ms]",
    ylabel="|signal|",
    title="white matter, at the echo times the file puts them at",
)
axes[1].grid(alpha=0.3)
# sphinx_gallery_end_ignore

# %%
#
# Echo spacing, echo train length, refocusing angle and pulse shapes were never
# named. The two things given were the tissue and which simulator to read the
# events as.
#

# %%
#
# Next steps
# ----------
#
# The derivative with respect to tissue is what a fit descends and what a
# model-based reconstruction pushes through an encoding operator; the parameter
# inference and model-based imaging notebooks do both. Differentiating with
# respect to the schedule instead is what designs a protocol, and is the
# subject of the sequence optimization notebooks.
#
# When the sequence you want is not one of the ones that ship, the next
# notebooks say what to write instead.
#
