"""
===================
Dictionary matching
===================

The scope of this notebook is to map a brain slice by exhaustive dictionary
matching, and to show the two ways of making that affordable: working in the
low-rank basis the train spans, and clustering the dictionary so that most
atoms are never scored.

A dictionary spans every combination of the parameters, so its size is the
product of the grids and each atom is as long as the train. The two savings are
independent and multiply; what each costs and what each gets wrong is read off
the same slice.
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
PANEL = (PAGE_WIDTH - 1 * BAR_WIDTH) / 4  # one image panel


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
from torchsim import (
    Subspace,
)
from torchsim.estimators import DictionaryMatcher
from torchsim.simulators import MRFSimulator

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
tissue_T2 = np.array([float(row["T2 (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(row["PD (ms)"]) for row in tissues])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5

# A mixed voxel is given the relaxation times its tissues average to. That is
# the parameter a fit can actually return: no single T1 explains a voxel that
# is half one tissue and half another, and pretending otherwise would make the
# error being reported partly the phantom's.
share = np.maximum(occupancy, 1e-6)
T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

figure, axes = canvas(1, 3, mask.shape, bars=3, extra=1.1)
for axis, values, name in (
    (axes[0, 0], T1_true, "T1"),
    (axes[0, 1], T2_true, "T2"),
    (axes[0, 2], M0_true, "M0"),
):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values, cmap, limits), axis, label)
figure.suptitle("BrainWeb subject 0, slice 90")
# sphinx_gallery_end_ignore

# %%
#
# Sequence
# --------
#
# Four hundred repetitions after an inversion, at a fixed repetition time and a
# flip angle that varies smoothly along the train. A schedule that jumped about
# would give trajectories differing by noise rather than by physics.
#
CONTRASTS = 400
TR_MS = 10.0
TI_MS = 20.0

repetition = torch.arange(CONTRASTS, dtype=torch.float32)
flip = 10.0 + 50.0 * torch.sin(torch.pi * repetition / CONTRASTS) ** 2

simulator = MRFSimulator(flip=flip, TR=TR_MS, TI=TI_MS, states=20, M0=1.0)

# %%
#
# The readouts wind the states on rather than rewinding them, so the trajectory
# comes back real to within 3e-8. That halves both the dictionary and the
# arithmetic that searches it.
#
fingerprints = simulator.simulate(
    T1=torch.tensor([500.0, 833.0, 2569.0]), T2=torch.tensor([70.0, 83.0, 329.0])
).real

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3))
axes[0].plot(repetition.numpy(), flip.numpy())
axes[0].set(xlabel="Repetition", ylabel="Flip angle [deg]", title="the schedule")
axes[0].grid(alpha=0.3)
for row, name in enumerate(("white matter", "grey matter", "CSF")):
    axes[1].plot(repetition.numpy(), fingerprints[row].numpy(force=True), label=name)
axes[1].set(xlabel="Repetition", ylabel="signal", title="fingerprints")
axes[1].grid(alpha=0.3)
key(axes[1], ncols=3)
# sphinx_gallery_end_ignore

# %%
#
# The measurement, with noise at 2% of the peak fingerprint. The estimators are
# told the same number: one trained for more noise than the scan has learns to
# distrust the data and answers with the prior.
#
NOISE_STD = float(0.02 * fingerprints.max())

truth = {
    "T1": torch.as_tensor(T1_true[mask].copy()),
    "T2": torch.as_tensor(T2_true[mask].copy()),
}
density = torch.as_tensor(M0_true[mask].copy())
clean = simulator.simulate(**truth).real * density[:, None]
generator = torch.Generator().manual_seed(42)
measured = clean + NOISE_STD * torch.randn(clean.shape, generator=generator)


# sphinx_gallery_start_ignore
def footprint(problem):
    """MiB the fitted estimator itself holds -- a dictionary, or a regression."""
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


def log_uniform(low, high, count):
    """``count`` draws spread evenly in the logarithm."""
    span = torch.rand(count, generator=prior)
    return torch.exp(np.log(low) + span * (np.log(high) - np.log(low)))


# sphinx_gallery_end_ignore

# %%
#
# Problem statement
# -----------------
#
# What is unknown, over what range, and at what noise level. Both relaxation
# times span more than a decade, so the grid is logarithmic: uniform spacing
# would spend most of it on long T1, where the trajectories are nearly
# parallel.
#
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)

# %%
#
# Subspace rank
# -------------
#
# A basis fitted to simulated trajectories says how much of their energy each
# rank keeps. One minus that fraction is the relative squared error of
# projecting through the basis and back.
#
training_signals, _, _ = (
    DictionaryMatcher(simulator)
    .fit(
        T1=log_uniform(*T1_RANGE, SAMPLES),
        T2=log_uniform(*T2_RANGE, SAMPLES),
        noise_std=NOISE_STD,
        seed=0,
    )
    .training_set(SAMPLES)
)
training_signals = training_signals.real

RANK = 4

# sphinx_gallery_start_ignore
ranks = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
outside = [1 - Subspace.fit(training_signals, rank).retained for rank in ranks]
noise_energy = (
    NOISE_STD**2 * CONTRASTS / float(training_signals.square().sum(-1).mean())
)

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.59))
axis.semilogy(ranks, outside, "o-", label="left outside the basis")
axis.axhline(noise_energy, color="crimson", linestyle="--", label="added by the noise")
axis.axvline(RANK, color="0.5", linewidth=1)
axis.set(
    xlabel="rank",
    ylabel="relative energy",
    title="what projecting through the basis loses",
)
axis.grid(alpha=0.3)
key(axis, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# Four directions out of four hundred leave less outside the basis than the
# noise puts in, and every contrast dropped is arithmetic the match avoids.
#
# Dictionary
# ----------
#
# The dictionary spans the parameters jointly, so its size is the product of
# the grids: twenty thousand atoms for two parameters, on a grid fine enough
# that the spacing is not what limits the answer. A third parameter multiplies
# it again.
#
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")

# sphinx_gallery_start_ignore
ATOMS = grid_t1.numel()
start = time.perf_counter()
# sphinx_gallery_end_ignore
full = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), seed=0
)

maps = full.map(measured)  # {"T1": ..., "T2": ...}, one value per voxel

# sphinx_gallery_start_ignore
full_training = time.perf_counter() - start
full_maps, full_matching, full_peak = mapped(full)
full_model = footprint(full)
# sphinx_gallery_end_ignore

# %%
#
# Matching in the subspace
# ------------------------
#
# ``rank`` is the whole change. The dictionary is fitted, projected and stored
# in four directions instead of four hundred, and the measurement is projected
# the same way before scoring.
#

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
low = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), seed=0, rank=RANK
)

# sphinx_gallery_start_ignore
low_training = time.perf_counter() - start
low_maps, low_matching, low_peak = mapped(low)
low_model = footprint(low)
# sphinx_gallery_end_ignore

# %%
#
# Clustered dictionary
# --------------------
#
# Compressing shortened every inner product; grouping cuts how many are taken.
# Neighbouring tissues give nearly parallel signals, so the atoms cluster, and
# a voxel scored against one representative per group rules out most groups
# before any atom inside them is touched.
#
# The clustering is done in the compressed basis, so a group is entered without
# leaving the space the measurement is already in.
#
GROUPS = 32

# sphinx_gallery_start_ignore
start = time.perf_counter()
# sphinx_gallery_end_ignore
grouped = DictionaryMatcher(simulator, groups=GROUPS).fit(
    T1=grid_t1.reshape(-1),
    T2=grid_t2.reshape(-1),
    seed=0,
    rank=RANK,
)

# sphinx_gallery_start_ignore
group_training = time.perf_counter() - start
group_maps, group_matching, group_peak = mapped(grouped)
group_model = footprint(grouped)

grouping = grouped.grouping
normalized = low.subspace.project(measured)
normalized = normalized / normalized.norm(dim=-1, keepdim=True)
survivors = grouping.survivors(normalized, grouped.prune)
print(
    f"{GROUPS} groups of {grouping.sizes[0]} atoms; "
    f"{float(survivors.sum(-1).float().mean()):.1f} still open per voxel"
)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
# The match works the density out on its way -- the score is a cosine, so the
# measurement's length over the atom's is the scale -- and returns it as M0.
estimates = {
    "full": (full_maps, full_maps["M0"]),
    f"rank {RANK}": (low_maps, low_maps["M0"]),
    "+ groups": (group_maps, group_maps["M0"]),
}
# sphinx_gallery_end_ignore

# %%
#
# Cost and accuracy
# -----------------
#
# Best of three passes each, after a warm-up. **model** is what the fitted
# estimator carries between volumes; **peak** is the high-water mark on the
# card, which decides whether a volume fits or has to be streamed, and a dash
# on a machine with no card.
#


# sphinx_gallery_start_ignore
def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


def held(megabytes):
    """A measurement in MiB, or a dash where there was no card to make it."""
    return f"{megabytes:6.0f} MiB" if np.isfinite(megabytes) else f"{'--':>10}"


print(
    f"\n{'method':<28}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}"
    f"{'T1':>8}{'T2':>8}{'M0':>8}"
)
print("-" * 88)
rows = [
    (
        "full",
        "match, 400 contrasts",
        full_training,
        full_matching,
        full_model,
        full_peak,
    ),
    (
        f"rank {RANK}",
        f"match, rank {RANK}",
        low_training,
        low_matching,
        low_model,
        low_peak,
    ),
    (
        "+ groups",
        f"match, rank {RANK} + groups",
        group_training,
        group_matching,
        group_model,
        group_peak,
    ),
]
for key, name, training, timing, model, peak in rows:
    found, m0 = estimates[key]
    print(
        f"{name:<28}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{held(peak)}"
        f"{error(found['T1'], truth['T1']):7.1f}%"
        f"{error(found['T2'], truth['T2']):7.1f}%"
        f"{error(m0, density):7.1f}%"
    )
# sphinx_gallery_end_ignore

# %%
#
# Maps
# ----
#


# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


panels = [
    ("T1", T1_true, {name: maps["T1"] for name, (maps, _) in estimates.items()}),
    ("T2", T2_true, {name: maps["T2"] for name, (maps, _) in estimates.items()}),
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

# Each parameter's errors on a scale of their own: an error map read at the
# scale of the map it came from is a black rectangle.
figure, axes = canvas(len(panels), len(estimates), mask.shape)
for row, (name, reference, found) in enumerate(panels):
    label = STYLE[name][2]
    residuals = {
        method: np.abs(painted(values) - reference) for method, values in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())
    for column, (method, values) in enumerate(residuals.items()):
        error = panel(
            axes[row, column],
            values,
            "inferno",
            (0.0, top or 1.0),
            title=f"\u0394 {method}" if row == 0 else None,
        )
    unit = label[label.find(" [") :] if "[" in label else ""
    scalebar(error, axes[row], f"|error|{unit}")
# sphinx_gallery_end_ignore
