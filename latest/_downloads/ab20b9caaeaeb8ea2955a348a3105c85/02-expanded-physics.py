"""
================
Expanded Physics
================

The scope of this notebook is to show the physics a simulator can carry beyond
T1 and T2: the transmit field and the array that produces it, off resonance,
an imperfect inversion, a second exchanging pool, a bound pool, diffusion and
flow, and the shaped pulse a scanner actually plays.

Each term is a tissue property. Naming one in a call is what turns it on, and
what a voxel is not given costs nothing.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# Every simulator accepts any tissue property, whether or not it declares one.
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
import math
from pathlib import Path

import numpy as np
import torch

import torchsim
from torchsim import SPGRReadout, ShimDefinition, bSSFPReadout
from torchsim.simulators import FSESimulator, MRFSimulator

# %%
# Test sequence
# -------------
#
# An inversion-prepared fingerprinting train: four hundred repetitions at a
# fixed TR, with a flip angle that rises and falls smoothly. It drives every
# coherence pathway, so most terms below can be shown on it. Where a term needs
# a different readout to be visible, only the readout is changed.
#
FLIP_DEG = np.concatenate((np.linspace(5.0, 55.0, 200), np.linspace(55.0, 5.0, 200)))
TRAIN = dict(flip=FLIP_DEG, TR=10.0, TI=20.0, states=20)
WATER = dict(T1=1000.0, T2=80.0)

fingerprinting = MRFSimulator(**TRAIN)
baseline = fingerprinting.simulate(**WATER)

# %%
#
# ``MRFSimulator`` names T1, T2, M0, a transmit scaling and an inversion
# efficiency. Every other field a voxel has can be given to it anyway, and
# giving one is what asks for its physics:
#

# sphinx_gallery_start_ignore
print(f"  declared: {', '.join(fingerprinting.exposes)}")
print(f"  accepted: {', '.join(fingerprinting.accepts)}")
print(f"  the sequence is written in: {', '.join(fingerprinting.variables)}")
# sphinx_gallery_end_ignore

# %%
# Transmit field
# --------------
#
# ``B1`` scales the flip angle a voxel turns. Because the nominal angle changes
# every repetition, a transmit error distorts the trajectory rather than
# scaling it -- which is what makes B1 estimable alongside T1 and T2, and what
# makes ignoring it a bias in both.
#
transmit = torch.tensor([0.7, 0.85, 1.0, 1.15])
scaled = fingerprinting.simulate(**WATER, B1=transmit)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6))
for row, value in enumerate(transmit):
    axis.plot(abs(scaled[row]), label=f"B1 = {float(value):.2f}")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="a transmit error reshapes the trajectory",
)
axis.grid(alpha=0.3)
key(axis, ncols=4)
# sphinx_gallery_end_ignore

# %%
# Transmit array and RF shim
# --------------------------
#
# On a parallel-transmit system the field is the complex sum of what several
# channels put on the voxel. ``B1`` and ``B1phase`` then carry one row per
# channel, and a :class:`~torchsim.ShimDefinition` gives the amplitude and
# phase each channel is driven at.
#
# Four channels whose sensitivities sit a quarter turn apart, driven alike,
# cancel exactly. The array is summed as a complex field before the state
# machine sees it, so a single pair of per-voxel buffers reaches the kernels.
#
CHANNELS, VOXELS = 4, 3
sensitivity = torch.full((CHANNELS, VOXELS), 1.0 / CHANNELS)
sensitivity_phase = (
    (torch.arange(CHANNELS)[:, None] * 2.0 * math.pi / CHANNELS)
    .expand(CHANNELS, VOXELS)
    .contiguous()
    .float()
)
ARRAY = dict(
    T1=torch.linspace(600.0, 1400.0, VOXELS),
    T2=torch.linspace(40.0, 120.0, VOXELS),
    B1=sensitivity,
    B1phase=sensitivity_phase,
)


def first_echo(step_rad):
    """What a shim holding each channel one more step behind leaves per voxel."""
    shim = ShimDefinition(
        0,
        (1.0,) * CHANNELS,
        tuple(float(-channel * step_rad) for channel in range(CHANNELS)),
    )
    train = FSESimulator(
        ESP=5.0,
        flip=torch.full((8,), 150.0),
        states=12,
        shims={0: shim},
    )
    return train.simulate(**ARRAY)[..., 0].abs()


steps_rad = torch.linspace(0.0, 2.0 * math.pi, 61)
swept = torch.stack([first_echo(float(step)) for step in steps_rad])

# sphinx_gallery_start_ignore
print(f"  driven alike     {[round(float(v), 4) for v in first_echo(0.0)]}")
print(
    f"  counter-rotated  "
    f"{[round(float(v), 4) for v in first_echo(2.0 * math.pi / CHANNELS)]}"
)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6))
degrees = torch.rad2deg(steps_rad).numpy()
for voxel in range(VOXELS):
    axis.plot(degrees, swept[:, voxel].numpy(), label=f"voxel {voxel + 1}")
for marked, text in ((0.0, "driven\nalike"), (360.0 / CHANNELS, "counter-\nrotated")):
    axis.axvline(marked, color="0.55", linestyle=":", linewidth=1.4)
    axis.annotate(text, xy=(marked + 8.0, 0.05), fontsize=13, color="0.35")
axis.set(
    xlabel="phase step between channels [deg]",
    ylabel="first echo [a.u.]",
    title="four channels a quarter turn apart, and the shim that finds them",
    xlim=(0.0, 360.0),
    xticks=[0, 90, 180, 270, 360],
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
# sphinx_gallery_end_ignore

# %%
#
# A shim belongs to the pulse rather than to the sequence: each RF event names
# the shim it is driven on, so an excitation and a refocusing pulse can use
# different ones.
#

# %%
# Shaped RF pulse and slice profile
# ---------------------------------
#
# The pulses above are instantaneous. A real slice-selective excitation turns a
# different angle at each position in the slice, and because that is a Bloch
# response rather than a scaling, it cannot be folded into the flip angle: the
# pulse must be integrated.
#
# TorchSim takes the complex envelope, one row per transmit channel, as it
# comes off a Pulseq block or an MRD sequence description. This one is an SLR
# 90 degree pulse, 2 ms long over a 5 mm slice, saved beside this file.
#

# sphinx_gallery_start_ignore
# The gallery runs an example from its own directory; a shell may not.
DATA = Path("data") if Path("data").is_dir() else Path(__file__).parent / "data"
# sphinx_gallery_end_ignore
waveform = np.load(DATA / "slr90.npz")
excitation = torchsim.rf_definition(
    waveform["samples"],
    dwell_s=float(waveform["dwell_s"]),
    bandwidth_hz=float(waveform["bandwidth_hz"]),
)

# %%
#
# ``pulse`` is the waveform the RF events drive; ``across_slice`` is how many
# positions to integrate it at. Without the second, the pulse is evaluated at
# the slice centre only, which reproduces the hard-pulse answer.
#
REFOCUSED = dict(ESP=5.0, TR=3000.0, T1=830.0, T2=80.0, states=48)
angles = torch.full((48,), 150.0)

hard = FSESimulator(**REFOCUSED).simulate(flip=angles)
centre = FSESimulator(**REFOCUSED, pulse=excitation).simulate(flip=angles)
across = FSESimulator(**REFOCUSED, pulse=excitation, across_slice=21).simulate(
    flip=angles
)

# %%
#
# The table holds the flip a spin turns at each position: flat across the
# passband and falling away outside it.
#

# sphinx_gallery_start_ignore
from torchsim.sequence._transition import transition_table  # noqa: E402

positions = torch.linspace(-1.0, 1.0, 121, dtype=torch.float64)
table = transition_table(excitation, positions, bins=64, rf_raster_time_s=1e-6)
_a, b = table.at(
    torch.arange(positions.numel()),
    torch.full((positions.numel(),), 0.5 * math.pi, dtype=torch.float64),
)
turned_deg = torch.rad2deg(2.0 * torch.arcsin(b.abs().clamp(max=1.0)))

figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.4))
time_ms = 1e3 * np.arange(waveform["samples"].size) * float(waveform["dwell_s"])
axes[0].plot(time_ms, np.abs(waveform["samples"]) / np.abs(waveform["samples"]).max())
axes[0].set(xlabel="time [ms]", ylabel="envelope [a.u.]", title="the pulse")
axes[1].plot(positions.numpy(), turned_deg.numpy())
axes[1].axhline(90.0, color="0.6", linestyle=":")
axes[1].set(
    xlabel="position [slice thicknesses]",
    ylabel="flip turned [deg]",
    title="what it turns, where",
)
for axis in axes:
    axis.grid(alpha=0.3)

print(f"  at the slice centre     {float(hard.abs().max()):.4f} (hard pulse)")
print(f"  the shaped pulse there  {float(centre.abs().max()):.4f}")
print(f"  averaged over the slice {float(across.abs().max()):.4f}")
print(
    f"  the profile costs       "
    f"{100 * (1 - float(across.abs().max()) / float(hard.abs().max())):.0f}% "
    f"of the signal"
)
# sphinx_gallery_end_ignore

# %%
#
# At the centre the shaped pulse reproduces the hard-pulse answer, which
# confirms the envelope is scaled correctly. Averaged across the slice it is
# much smaller, and for a refocused train the difference is not a scaling: the
# slice edges see a smaller refocusing angle and therefore a different balance
# of coherence pathways.
#
# Exchange and magnetization transfer
# -----------------------------------
#
# A second free pool -- myelin water beside intra- and extracellular water --
# is recorded along with the first. A bound pool has a T2 of tens of
# microseconds and is never recorded, but exchanges with the water that is.
#
# Five properties describe an exchanging free pool and three a bound pool. Both
# are shown on a spoiled train driven to steady state, using the white matter
# models of Malik et al. (Magn. Reson. Med. 2018).
#
REPETITIONS, TR_MS, SPOILED_FLIP, SPOILING_STEP = 200, 5.0, 10.0, 117.0
index = np.arange(REPETITIONS)
SPOILED_TRAIN = dict(
    flip=np.full(REPETITIONS, SPOILED_FLIP),
    phases=SPOILING_STEP * index * (index + 1) / 2.0,
    TR=TR_MS,
    TI=0.0,
    states=40,
)
WHITE_MATTER = dict(T1=779.0, T2=45.0)
FREE = dict(poolB_exchange=2.0, poolB_T1=500.0, poolB_T2=20.0)
BOUND = dict(bound_exchange=4.3, bound_T1=779.0)


class SpoiledMRF(MRFSimulator):
    """The same train, read with a spoiled gradient echo."""

    readout = SPGRReadout


spoiled = SpoiledMRF(**SPOILED_TRAIN)
one_pool = spoiled.simulate(**WHITE_MATTER)
with_free = spoiled.simulate(**WHITE_MATTER, poolB_fraction=0.2, **FREE)
with_bound = spoiled.simulate(**WHITE_MATTER, bound_fraction=0.117, **BOUND)

# %%
#
# The pool fraction is a tissue property, so sweeping it is one call over a
# voxel axis. At zero fraction both must return the single-pool answer.
#
fractions = torch.linspace(0.0, 0.3, 31)
free_sweep = spoiled.simulate(**WHITE_MATTER, poolB_fraction=fractions, **FREE)
bound_sweep = spoiled.simulate(**WHITE_MATTER, bound_fraction=fractions, **BOUND)

# sphinx_gallery_start_ignore
SERIES = (
    ("one pool", "0.25", one_pool),
    ("a second free pool", "tab:orange", with_free),
    ("a bound pool", "tab:green", with_bound),
)
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6))
for label, colour, values in SERIES:
    axes[0].plot(abs(np.asarray(values).reshape(-1)), color=colour, label=label)
axes[0].set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="approach to the steady state",
    xlim=(0, REPETITIONS),
)
settled = float(abs(np.asarray(one_pool).reshape(-1)[-1]))
axes[1].axhline(settled, color="0.25", label="one pool")
axes[1].plot(fractions.numpy(), abs(free_sweep[:, -1]), color="tab:orange")
axes[1].plot(fractions.numpy(), abs(bound_sweep[:, -1]), color="tab:green")
axes[1].set(
    xlabel="second-pool fraction",
    ylabel="settled signal [a.u.]",
    title="what the fraction does",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(figure, ncols=3)

print(f"  one pool settles at   {settled:.5f}")
print(
    f"  a second free pool    {float(abs(np.asarray(with_free).reshape(-1)[-1])):.5f}"
)
print(
    f"  a bound pool          {float(abs(np.asarray(with_bound).reshape(-1)[-1])):.5f}"
)
print(
    "  at a fraction of nothing the two rejoin it, to "
    f"{float(max(abs(free_sweep[0, -1] - settled), abs(bound_sweep[0, -1] - settled))):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# The two move the signal in opposite directions: filling a second free pool
# raises it, since that pool is recorded too, while filling a bound pool lowers
# it, since the magnetization parked there is never read.
#
# The properties are independent, so a voxel can carry both -- eight names in
# one call.
#
three_pool = spoiled.simulate(
    **WHITE_MATTER, poolB_fraction=0.2, **FREE, bound_fraction=0.117, **BOUND
)

# sphinx_gallery_start_ignore
print(
    f"  both pools together   {float(abs(np.asarray(three_pool).reshape(-1)[-1])):.5f}"
)
# sphinx_gallery_end_ignore

# %%
# Off resonance
# -------------
#
# ``B0`` turns the transverse states between events. Whether that reaches the
# signal depends on the sequence: a train that dephases by a whole
# configuration order every repetition separates the orders and is insensitive to
# it, while a balanced train bands. Only the readout is changed below.
#
BALANCED_TR_MS = 10.0
offsets_hz = torch.linspace(-150.0, 150.0, 121)


class BalancedMRF(MRFSimulator):
    """The same train, read with a fully refocused steady state."""

    readout = bSSFPReadout


banded = BalancedMRF(
    flip=np.full(64, 20.0),
    TR=BALANCED_TR_MS,
    TI=0.0,
    states=20,
).simulate(T1=1000.0, T2=80.0, B0=offsets_hz, repetitions="auto")

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
axis.plot(offsets_hz, abs(banded[:, -1]))
for band in (-100.0, 0.0, 100.0):
    axis.axvline(band, color="0.6", linestyle=":", linewidth=1.2)
axis.set(
    xlabel="off resonance [Hz]",
    ylabel="signal magnitude [a.u.]",
    title=f"balanced, TR = {BALANCED_TR_MS:.0f} ms: bands every "
    f"{1e3 / BALANCED_TR_MS:.0f} Hz",
)
axis.grid(alpha=0.3)
print(f"  nulls sit {1e3 / BALANCED_TR_MS:.0f} Hz apart, which is 1 / TR")
# sphinx_gallery_end_ignore

# %%
# Inversion efficiency
# --------------------
#
# ``inv_efficiency`` is the fraction of magnetization the inversion pulse
# turns over. It affects the front of the train, where the inversion sets the
# contrast.
#
efficiencies = torch.tensor([1.0, 0.9, 0.8])
inverted = fingerprinting.simulate(**WATER, inv_efficiency=efficiencies)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
for row, value in enumerate(efficiencies):
    axis.plot(abs(inverted[row][:80]), label=f"efficiency {float(value):.1f}")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="an imperfect inversion is a transient, not a scaling",
)
axis.grid(alpha=0.3)
key(axis, ncols=3)
print(
    f"  the first repetition falls to {float(inverted[2][0].abs() / inverted[0][0].abs()):.3f} of the ideal;"
)
print(
    f"  by the four-hundredth the three agree to "
    f"{float((inverted[2][-1] - inverted[0][-1]).abs() / inverted[0][-1].abs()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
# Diffusion and flow
# ------------------
#
# Both are read off the winding a gradient has put on a configuration order, so
# the sequence must say how much winding an order stands for. That is two
# simulator arguments rather than tissue properties: ``crusher_dephasing_rad``,
# the turn one crusher puts across a voxel, and ``voxel_size_m``, the distance
# it puts it across. Without them an order has no physical extent and neither
# term does anything.
#
MOMENT = dict(crusher_dephasing_rad=4.0 * math.pi, voxel_size_m=1e-3)
moving = MRFSimulator(**TRAIN, **MOMENT)

diffusivities = torch.tensor([0.0, 1.0, 2.0, 3.0])
velocities = torch.tensor([0.0, 0.01, 0.03, 0.05])
diffusing = moving.simulate(**WATER, D=diffusivities)
flowing = moving.simulate(**WATER, v=velocities)

# %%
#
# The train is only mildly diffusion-weighted, so diffusion is drawn as a ratio
# to a voxel that does not diffuse. Flow is large enough to read directly.
#

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
undiffused = diffusing[0].abs()
readable = undiffused > 0.15 * undiffused.max()
for row in range(1, len(diffusivities)):
    ratio = torch.where(readable, diffusing[row].abs() / undiffused, torch.nan)
    axis.plot(ratio.numpy(), label=f"D = {float(diffusivities[row]):.0f}")
axis.axhline(1.0, color="0.35", lw=1.2)
axis.set(
    xlabel="repetition",
    ylabel="relative to no diffusion",
    title=r"diffusion in $\mu$m$^2$/ms, over a "
    f"{MOMENT['crusher_dephasing_rad'] / math.pi:.0f}"
    r"$\pi$ crusher across a "
    f"{1e3 * MOMENT['voxel_size_m']:.0f} mm voxel",
    ylim=(0.75, 1.2),
)
axis.grid(alpha=0.3)
key(axis, ncols=3)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
for row in range(len(velocities)):
    axis.plot(abs(flowing[row]), label=f"v = {100 * float(velocities[row]):.0f} cm/s")
axis.set(
    xlabel="repetition",
    ylabel="signal magnitude [a.u.]",
    title="flow through the same voxel",
)
axis.grid(alpha=0.3)
key(axis, ncols=4)

print(
    "  diffusion damps the higher orders, so it costs signal where the train "
    "is built from them:"
)
print(
    f"    D = 3 departs from D = 0 by "
    f"{float((diffusing[3] - diffusing[0]).abs().max() / diffusing[0].abs().max()):.0%}"
)
print(
    "  flow carries winding out of the voxel and brings unsaturated "
    "magnetization in, so it reshapes the train:"
)
print(
    f"    5 cm/s departs by "
    f"{float((flowing[3] - flowing[0]).abs().max() / flowing[0].abs().max()):.1f}"
    f"x the unflowed signal"
)
# sphinx_gallery_end_ignore

# %%
# Cost of an unused property
# --------------------------
#
# Nothing. A property held at the value where it has no effect -- unit
# transmit, no off resonance, an empty pool -- is reported absent, and its term
# is left out of the kernel that is compiled and run.
#
idle = fingerprinting.simulate(
    **WATER,
    B0=0.0,
    D=0.0,
    v=0.0,
    poolB_fraction=0.0,
    bound_fraction=0.0,
    inv_efficiency=1.0,
)

# sphinx_gallery_start_ignore
print(
    "  six more fields named, none of them doing anything: agrees to "
    f"{float(np.abs(np.asarray(idle) - np.asarray(baseline)).max()):.1e}"
)
# sphinx_gallery_end_ignore

# %%
#
# Every term above is one the kernels already carry. A term they do not -- a
# third free pool, a gradient moment that varies down the train -- is a change
# to the engine rather than a name in a call.
