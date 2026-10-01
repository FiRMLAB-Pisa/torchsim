"""
=============================
PERK: kernel ridge regression
=============================

The scope of this notebook is to map a brain slice with PERK, to show what the
size of the regression buys, and to read the error bar it reports.

PERK never builds a dictionary. It is a kernel regression trained on signals
drawn from a prior rather than laid on a grid, and at inference it projects a
signal onto a fixed set of random Fourier features and reads the answer off a
linear combination of them. Its cost per voxel does not depend on how many
parameters are unknown, and the training is paid once.
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
PANEL = (PAGE_WIDTH - 3 * BAR_WIDTH) / 3  # one image panel


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
from torchsim.estimators import PERK, DictionaryMatcher
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
# The readouts wind the states on rather than rewinding them, so nothing
# returns transverse magnetization to the imaginary axis: the trajectory comes
# back real to within 3e-8, which halves both the dictionary and the
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
# The measurement, with noise at 2% of the peak fingerprint. One number sets
# it, and the same number is what the estimators are told to expect -- an
# estimator trained for more noise than the scan has learns to distrust the
# data and answers with the prior instead.
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
# times span more than a decade, so the prior is drawn logarithmically:
# uniform sampling would spend most of the budget on long T1, where the
# trajectories are nearly parallel.
#
T1_RANGE = (200.0, 5000.0)
T2_RANGE = (20.0, 600.0)
SAMPLES = 20_000
prior = torch.Generator().manual_seed(11)

# %%
#
# Subspace basis
# --------------
#
# Four hundred contrasts do not span four hundred directions. A basis fitted to
# simulated trajectories says how many they do span; one minus the energy it
# keeps is the relative squared error of projecting through it and back.
#
training_signals, _, _ = (
    PERK(simulator)
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
# Four directions leave less outside the basis than the noise puts in, so
# rank 4 is used from here on.
#
# Training
# --------
#
# Twenty thousand parameter pairs drawn from the prior, simulated, and given
# the noise the scan has. Fitting is a linear solve against the features.
#


FEATURES = 1000

perk = PERK(simulator, n_features=FEATURES, regularization=1e-6, normalize=True).fit(
    T1=log_uniform(*T1_RANGE, SAMPLES),
    T2=log_uniform(*T2_RANGE, SAMPLES),
    noise_std=NOISE_STD,
    seed=0,
    rank=RANK,
    samples=SAMPLES,
)

maps = perk(measured)  # {"T1": ..., "T2": ...}, one value per voxel


# sphinx_gallery_start_ignore
def regressed(features):
    """Train a regression of this size, then map the slice."""
    start = time.perf_counter()
    problem = PERK(
        simulator, n_features=features, regularization=1e-6, normalize=True
    ).fit(
        T1=log_uniform(*T1_RANGE, SAMPLES),
        T2=log_uniform(*T2_RANGE, SAMPLES),
        noise_std=NOISE_STD,
        seed=0,
        rank=RANK,
        samples=SAMPLES,
    )
    training = time.perf_counter() - start
    found, timing, peak = mapped(problem)
    return found, training, timing, footprint(problem), peak


regressions = {features: regressed(features) for features in (500, 1000, 4000)}
perk_maps, perk_training, perk_mapping, perk_model, perk_peak = regressions[FEATURES]
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
# A compressed dictionary match over a grid fine enough that the spacing is not
# what limits it, kept only as the reference point in the table below. What a
# match is and what it costs is the subject of the first example in this
# section, not of this one.
T1_GRID = torch.logspace(np.log10(T1_RANGE[0]), np.log10(T1_RANGE[1]), 200)
T2_GRID = torch.logspace(np.log10(T2_RANGE[0]), np.log10(T2_RANGE[1]), 100)
grid_t1, grid_t2 = torch.meshgrid(T1_GRID, T2_GRID, indexing="ij")

start = time.perf_counter()
matched = DictionaryMatcher(simulator).fit(
    T1=grid_t1.reshape(-1), T2=grid_t2.reshape(-1), rank=RANK, seed=0
)
match_training = time.perf_counter() - start
match_maps, match_mapping, match_peak = mapped(matched)
match_model = footprint(matched)
estimates = {
    "match": (match_maps, match_maps["M0"]),
    "PERK": (perk_maps, perk_maps["M0"]),
}
print(
    f"dictionary: {grid_t1.numel()} atoms at rank {RANK}; "
    f"regression: {FEATURES} features from {SAMPLES} training draws"
)
# sphinx_gallery_end_ignore

# %%
#
# Cost and accuracy
# -----------------
#
# Best of three passes each, after a warm-up. **model** is what the fitted
# estimator carries between volumes; **peak** is the high-water mark on the
# card while the slice was mapped, and a dash on a machine with no card. The
# dictionary row is the reference point.
#


# sphinx_gallery_start_ignore
def error(estimate, reference):
    """Median relative error, in percent."""
    return float(100 * ((estimate - reference).abs() / reference).median())


def held(megabytes):
    """A measurement in MiB, or a dash where there was no card to make it."""
    return f"{megabytes:6.0f} MiB" if np.isfinite(megabytes) else f"{'--':>10}"


print(
    f"\n{'method':<26}{'train':>8}{'map':>8}{'model':>10}{'peak':>10}{'T1':>8}{'T2':>8}"
)
print("-" * 78)
rows = [
    (
        f"match, rank {RANK}",
        match_training,
        match_mapping,
        match_model,
        match_peak,
        match_maps,
    )
]
for features, (found, training, timing, model, peak) in regressions.items():
    rows.append((f"PERK, {features} features", training, timing, model, peak, found))

for name, training, timing, model, peak, found in rows:
    print(
        f"{name:<26}{training:7.1f}s{timing:7.2f}s"
        f"{model:6.1f} MiB{held(peak)}"
        f"{error(found['T1'], truth['T1']):7.1f}%"
        f"{error(found['T2'], truth['T2']):7.1f}%"
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
    ("T1", T1_true, {name: found["T1"] for name, (found, _) in estimates.items()}),
    ("T2", T2_true, {name: found["T2"] for name, (found, _) in estimates.items()}),
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

# %%
#
# Uncertainty
# -----------
#
# ``uncertainty=True`` returns a second set of maps: how far the answer is
# expected to sit from the truth. PERK learns that at training, from the
# residuals of its own fit, so reporting it is a matrix multiply rather than a
# rerun of the volume.
#
maps, spread = perk(measured, uncertainty=True)

# %%
#
# Read against the Cramer-Rao bound, the lowest standard deviation an unbiased
# estimate could reach from this train at this noise. The bound belongs to the
# sequence, so the gap is what the method loses.
#
_signal, sensitivity = simulator.jacobian("T1 T2".split(), **truth)
sensitivity = sensitivity.real * density[:, None, None]
floor = torchsim.crlb(sensitivity, noise_variance=NOISE_STD**2, singular="infinite")
bound = {"T1": floor[:, 0].sqrt(), "T2": floor[:, 1].sqrt()}

# sphinx_gallery_start_ignore
relative = {name: 100.0 * spread[name] / maps[name].clamp_min(1e-6) for name in bound}

print(f"{'':<6}{'PERK':>10}{'CRLB':>10}{'PERK':>9}   (median over the brain)")
for name in ("T1", "T2"):
    print(
        f"{name:<6}{float(spread[name].median()):7.1f} ms"
        f"{float(bound[name].median()):7.1f} ms"
        f"{float(relative[name].median()):8.1f}%"
    )

figure, axes = canvas(2, 3, mask.shape)
for row, name in enumerate(("T1", "T2")):
    cmap, _limits, label = STYLE[name]
    absolute = (("PERK", spread[name]), ("CRLB", bound[name]))
    top = max(
        float(np.percentile(values.numpy(force=True), 98)) for _, values in absolute
    )
    for column, (title, values) in enumerate(absolute):
        handle = panel(
            axes[row, column],
            painted(values),
            cmap,
            (0.0, top or 1.0),
            title=title if row == 0 else None,
            ylabel=label if column == 0 else None,
        )
    figure.colorbar(handle, ax=axes[row, :2], label="ms", shrink=0.92, aspect=20)
    share = panel(
        axes[row, 2],
        painted(relative[name]),
        cmap,
        (0.0, float(np.percentile(relative[name].numpy(force=True), 98)) or 1.0),
        title="PERK, relative" if row == 0 else None,
    )
    figure.colorbar(share, ax=axes[row, 2], label="%", shrink=0.92, aspect=20)
# sphinx_gallery_end_ignore

# %%
#
# Each row uses its own parameter's colormap. The two absolute panels share a
# scale; the third is the same spread as a percentage of the relaxation time,
# which is what says whether ten milliseconds is tight.
#
# Both are largest in CSF, whose long T1 this train resolves least. The gap
# between them is not: T1 sits within a small multiple of the bound, T2
# several times above it, and it is T2 whose error moved when the feature
# count was swept. One is a sequence to redesign, the other a regression to
# enlarge.
#
# The number is not the noise alone. A regression trained on a prior answers
# with the prior where the data is weak, and is wrong the same way in every
# realization, so repeating the scan would never show that part.
