"""
====================
MP2RAGE lookup table
====================

The scope of this notebook is to map T1 from a two-block MP2RAGE, by
interpolating along a curve and by matching the same curve, and to sweep the
number of points to show which of the two is limited by it.

With a single unknown a dictionary degenerates: the atoms lie on a curve rather
than filling a space. Interpolating between the two nearest then costs nothing
and takes the grid spacing out of the answer, which is what
:class:`~torchsim.LookupTable` does.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl cmap

# %%
#
# The problem is stated over a simulator carrying the sequence and filled in
# by an estimator. :func:`~torchsim.execution` decides where that work runs,
# and the timings below are taken inside it.
#

# sphinx_gallery_start_ignore
import csv
import warnings
from pathlib import Path

import brainweb_dl
import matplotlib.pyplot as plt
from cmap import Colormap
from brainweb_dl import get_mri

warnings.filterwarnings("ignore")


# Fuderer et al. (Magn. Reson. Med. 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()

# Colormap, window and unit per parameter. Both relaxation windows stop well
# short of CSF, so that white and grey matter -- 500 against 833 ms in T1, 70
# against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "T1 [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "T2 [ms]"),
    "M0": ("gray", (0.0, 1.0), "M0"),
}


def panel(axis, values, cmap, limits, title=None, ylabel=None):
    """One map without ticks; the handle is what a row shares a colorbar from."""
    handle = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    return handle


def scalebar(handle, axes, label):
    """One colorbar for a group of panels, so none gives up width to its own."""
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


# Every panel on this page is drawn at the same size, so any two figures can be
# read against each other. The side is set by the widest grid, which fills the
# documentation column; a figure with fewer columns is narrower, not larger.
PAGE_WIDTH = 8.6  # inches, the width of the documentation column
BAR_WIDTH = 0.8  # what one colorbar takes out of it
PANEL = (PAGE_WIDTH - 1 * BAR_WIDTH) / 3  # one image panel


def canvas(rows, columns, shape, *, bars=1, extra=0.6):
    """A grid of image panels, in the proportion of the images.

    ``bars`` is how many colorbars a row carries and ``extra`` the height left
    over the panels, for titles and for a figure title where there is one.
    """
    return plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(
            columns * PANEL + bars * BAR_WIDTH,
            PANEL * shape[0] / shape[1] * rows + extra,
        ),
    )


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
import time

import numpy as np
import torch

import torchsim
from torchsim.estimators import DictionaryMatcher, LookupTable
from torchsim.simulators import MP2RAGESimulator

# %%
#
# Phantom
# -------
#
# BrainWeb subject 0, slice 90: an axial slice at 1 mm through the lateral
# ventricles. BrainWeb publishes fuzzy memberships rather than labels, so each
# voxel holds a fraction of each tissue, and the relaxation times are weighted
# by those fractions. A third of the voxels are mixtures, so the truth is a
# continuum and not four values.
#

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
tissues = list(csv.DictReader(table.open()))
tissue_T1 = np.array([float(row["T1 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(row["PD (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5

# A mixed voxel is given the relaxation time its tissues average to. That is
# the parameter a fit can actually return: no single T1 explains a voxel that
# is half one tissue and half another.
share = np.maximum(occupancy, 1e-6)
T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

truth = torch.as_tensor(T1_true[mask].copy())
density = torch.as_tensor(M0_true[mask].copy())

figure, axes = canvas(1, 2, mask.shape, bars=2, extra=1.1)
for axis, values, name in ((axes[0, 0], T1_true, "T1"), (axes[0, 1], M0_true, "M0")):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
# sphinx_gallery_end_ignore

# %%
#
# Protocol
# --------
#
# One inversion, two spoiled gradient-echo blocks read at two inversion times.
# Each block samples the centre of k-space of its own shot train, so a voxel
# contributes two numbers. The train spoils after every readout, so T1 is the
# only tissue property that moves them.
#
PROTOCOL = dict(
    TI=(800.0, 2700.0),
    flip=(4.0, 5.0),
    TRspgr=6.7,
    TRmp2rage=6000.0,
    nshots=128,
)
INVERSION_EFFICIENCY = 0.96

simulator = MP2RAGESimulator(**PROTOCOL, inv_efficiency=INVERSION_EFFICIENCY)

# %%
#
# Signal curve
# ------------
#
# Neither block alone says T1: both carry the proton density and the receive
# gain. The unified combination divides that scale out, leaving a number
# between -0.5 and 0.5 that depends on T1 alone. Which combination is monotonic
# belongs to the sequence, so it is given rather than assumed.
#


def unified(blocks):
    """The MP2RAGE unified image: scale-free, and a function of T1 alone."""
    return (blocks[..., 0] * blocks[..., 1]) / blocks.square().sum(-1).clamp_min(1e-12)


# %%
#
# The curve is not monotonic over every T1, and where it turns back it has no
# inverse. The table keeps the longest monotonic run and reports what it spans,
# so the invertible range is a number rather than an assumption.
#
sweep = torch.arange(50.0, 6000.0, 10.0)
curve = unified(simulator.simulate(T1=sweep, M0=1.0))

# sphinx_gallery_start_ignore
turning = int(curve.argmin()) if curve[0] > curve[-1] else int(curve.argmax())

figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3))
blocks = simulator.simulate(T1=sweep, M0=1.0)
axes[0].plot(
    sweep.numpy(), blocks[:, 0].numpy(), label=f"TI = {PROTOCOL['TI'][0]:.0f} ms"
)
axes[0].plot(
    sweep.numpy(), blocks[:, 1].numpy(), label=f"TI = {PROTOCOL['TI'][1]:.0f} ms"
)
axes[0].set(xlabel="T1 [ms]", ylabel="magnetization", title="the two blocks")

axes[1].plot(sweep.numpy(), curve.numpy(), color="crimson")
axes[1].axvline(float(sweep[turning]), color="k", ls="--", lw=1)
axes[1].set(
    xlabel="T1 [ms]",
    ylabel="unified image",
    title="the curve a T1 is read off",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(axes[0], ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# Measurement
# -----------
#
# Both blocks, at the true T1 and proton density of every brain voxel, with
# noise at half a percent of the peak magnetization.
#
clean = simulator.simulate(T1=truth, M0=density)
NOISE_STD = float(0.005 * clean.abs().max())

generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)


# sphinx_gallery_start_ignore
def footprint(problem):
    """MiB the fitted estimator itself holds."""
    held = sum(t.numel() * t.element_size() for t in problem.buffers())
    return held / 2**20


def mapped(problem, passes=3):
    """Map the slice a few times: the quickest pass, and what it held."""
    on_device = torch.cuda.is_available()
    with torchsim.execution():
        problem(measured[:64])
        if on_device:
            torch.cuda.reset_peak_memory_stats()
        best = float("inf")
        for _ in range(passes):
            start = time.perf_counter()
            maps = problem(measured)
            best = min(best, time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated() / 2**20 if on_device else float("nan")
    return maps, best, peak


def estimated(make, points):
    """Fit this method over a grid of this many points, then map the slice."""
    grid = torch.linspace(50.0, 6000.0, points)
    problem = make(simulator.bind(M0=1.0))
    start = time.perf_counter()
    problem.fit(T1=grid, seed=0)
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return problem, found, training, timing, footprint(problem), peak


def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


def held(megabytes):
    """A measurement in MiB, or a dash where there was no card to make it."""
    return f"{megabytes:6.0f} MiB" if np.isfinite(megabytes) else f"{'--':>10}"


# sphinx_gallery_end_ignore

# %%
#
# Two estimators
# --------------
#
# Both are given the same T1 grid. The match scores the two-block signal
# against every atom and takes the nearest; the table reduces both blocks to
# the unified number and interpolates along the curve, so ``combine`` is all it
# is told. Neither is given the range in advance.
#
grid = torch.linspace(50.0, 6000.0, 60)

table = LookupTable(simulator.bind(M0=1.0), combine=unified).fit(T1=grid, seed=0)

maps = table.map(measured)  # {"T1": ...}, one value per voxel

match = DictionaryMatcher(simulator.bind(M0=1.0)).fit(T1=grid, seed=0)

# %%
#
# Sweeping the grid separates the method from the sampling. Times are the best
# of three passes over the slice.
#

# sphinx_gallery_start_ignore
POINTS = (30, 60, 120, 250, 500, 1000, 2000)
matched = {}
looked_up = {}
for points in POINTS:
    _, found, _, timing, _, _ = estimated(DictionaryMatcher, points)
    matched[points] = (error(found["T1"], truth), timing)
    problem, found, _, timing, _, _ = estimated(
        lambda acq: LookupTable(acq, combine=unified), points
    )
    looked_up[points] = (error(found["T1"], truth), timing)

print(f"\n{'points':>7}{'match':>10}{'table':>10}{'match':>10}{'table':>10}")
print(f"{'':>7}{'error':>10}{'error':>10}{'time':>10}{'time':>10}")
print("-" * 47)
for points in POINTS:
    print(
        f"{points:>7}{matched[points][0]:9.2f}%{looked_up[points][0]:9.2f}%"
        f"{1e3 * matched[points][1]:8.1f}ms{1e3 * looked_up[points][1]:8.1f}ms"
    )
# sphinx_gallery_end_ignore

# %%
#
# The table is at its floor from the coarsest grid and does not move again. The
# match starts an order of magnitude worse and climbs to the same place, paying
# for it in points: its search is one comparison per atom per voxel, where the
# table's binary search grows with the logarithm.
#
# The floor both reach is the noise. A fine enough grid matches a table
# exactly; the table's advantage is that it was never told how fine.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3))
axes[0].plot(POINTS, [matched[n][0] for n in POINTS], "-o", label="match")
axes[0].plot(POINTS, [looked_up[n][0] for n in POINTS], "-*", label="lookup table")
axes[0].set(
    xlabel="Points on the curve",
    ylabel="Median relative error [%]",
    xscale="log",
    yscale="log",
    title="what the grid costs",
)
axes[1].plot(POINTS, [1e3 * matched[n][1] for n in POINTS], "-o", label="match")
axes[1].plot(
    POINTS, [1e3 * looked_up[n][1] for n in POINTS], "-*", label="lookup table"
)
axes[1].set(
    xlabel="Points on the curve",
    ylabel="Time to map the slice [ms]",
    xscale="log",
    yscale="log",
    title="what it costs to pay it",
)
for axis in axes:
    axis.grid(alpha=0.3)
key(figure, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# Maps
# ----
#
# At the point count each needs: the table at sixty, the match at a grid fine
# enough not to limit it.
#

# sphinx_gallery_start_ignore
TABLE_POINTS = 60
MATCH_POINTS = 2000

problem, table_maps, table_training, table_time, table_model, table_peak = estimated(
    lambda acq: LookupTable(acq, combine=unified), TABLE_POINTS
)
_, match_maps, match_training, match_time, match_model, match_peak = estimated(
    DictionaryMatcher, MATCH_POINTS
)
print(
    f"the table keeps {problem.points} of {TABLE_POINTS} points -- the "
    f"monotonic run -- and spans unified "
    f"{problem.span[0]:.2f} to {problem.span[1]:.2f}"
)
# sphinx_gallery_end_ignore

# %%
#
# Neither estimates M0. Both answer with a T1, and the two blocks it predicts
# are a shape the measurement is a multiple of, so the multiple is one inner
# product per voxel.
#


def proton_density(maps):
    """The scale the measurement is, of the blocks the answer predicts."""
    predicted = simulator.simulate(T1=maps["T1"], M0=1.0)
    return (predicted * measured).sum(-1) / predicted.square().sum(-1).clamp_min(1e-12)


M0_map = proton_density(maps)

# sphinx_gallery_start_ignore
estimates = {
    "lookup": (table_maps, proton_density(table_maps)),
    "match": (match_maps, proton_density(match_maps)),
}

print(
    f"\n{'method':<24}{'train':>9}{'map':>9}{'model':>10}{'peak':>10}{'T1':>8}{'M0':>8}"
)
print("-" * 78)
for short, name, training, timing, model, peak in (
    (
        "lookup",
        f"lookup, {TABLE_POINTS} points",
        table_training,
        table_time,
        table_model,
        table_peak,
    ),
    (
        "match",
        f"match, {MATCH_POINTS} atoms",
        match_training,
        match_time,
        match_model,
        match_peak,
    ),
):
    found, m0 = estimates[short]
    print(
        f"{name:<24}{training:8.2f}s{1e3 * timing:7.1f}ms"
        f"{model:6.2f} MiB{held(peak)}"
        f"{error(found['T1'], truth):7.2f}%{error(m0, density):7.2f}%"
    )
# sphinx_gallery_end_ignore

# %%


# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


panels = [
    ("T1", T1_true, {name: found["T1"] for name, (found, _) in estimates.items()}),
    ("M0", M0_true, {name: m0 for name, (_, m0) in estimates.items()}),
]

figure, axes = canvas(len(panels), 1 + len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    cmap, limits, label = STYLE[name]
    panel(
        axes[row, 0],
        reference,
        cmap,
        limits,
        ylabel=label,
        title="truth" if row == 0 else None,
    )
    for column, (method, values) in enumerate(found.items(), start=1):
        handle = panel(
            axes[row, column],
            painted(values),
            cmap,
            limits,
            title=method if row == 0 else None,
        )
    scalebar(handle, axes[row], "")

# The errors, each parameter on a scale of its own: an error map read at the
# scale of the map it came from is a black rectangle.
figure, axes = canvas(len(panels), len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    label = STYLE[name][2]
    residuals = {
        method: np.abs(painted(values) - reference) for method, values in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
    for column, (method, values) in enumerate(residuals.items()):
        handle = panel(
            axes[row, column],
            values,
            "inferno",
            (0.0, top or 1.0),
            title=f"\u0394 {method}" if row == 0 else None,
        )
    scalebar(handle, axes[row], f"|error|, {label}")
# sphinx_gallery_end_ignore
