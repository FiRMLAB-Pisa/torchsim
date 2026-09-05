"""
=========================================
Designing a joint relaxometry protocol
=========================================

The scope of this notebook is to choose the flip angles of a DESPOT protocol
so that a joint fit of T1 and T2 is as precise as possible [1]_.

DESPOT estimates T1 from spoiled gradient-echo scans at different flip angles
and T2 from balanced SSFP scans. Fitting them jointly uses all the data for
both parameters. The cost is a Cramer-Rao bound: the lowest variance an
unbiased estimate can have, given the derivative of each signal with respect to
every parameter estimated.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim brainweb-dl cmap

# %%
#
# The two closed-form sequences being designed, the three pieces a design is
# stated in, and :func:`~torchsim.crlb`, which is the cost.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from cmap import Colormap


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

import torch

import torchsim
from torchsim.optim import Bounded, SequenceDesign
from torchsim.simulators import SPGRSimulator, bSSFPSimulator

# %%
#
# Sequences
# ---------
#
# A design problem is three pieces, and the first is a simulator with the
# tissue it is designed for already fixed on it, so only the parameters under
# design are left to give. Both sequences here are closed forms, constructed
# and called exactly as a state-machine sequence would be.
#

# White and grey matter at 3 T -- the design is for both at once.
T1_MS = torch.tensor([830.0, 1330.0])
T2_MS = torch.tensor([80.0, 110.0])
# Noise standard deviation, as a fraction of the fully relaxed magnetization.
NOISE = 0.005

spgr = SPGRSimulator(TE=2.0, TR=6.0, T1=T1_MS, T2star=T2_MS, M0=1.0, B0=0.0)
ssfp = bSSFPSimulator(TE=2.5, TR=5.0, T1=T1_MS, T2=T2_MS, M0=1.0, B0=0.0)

# %%
#
# Cost
# ----
#
# Four parameters are estimated jointly: T1, T2, the proton density and the
# off-resonance. The last two are nuisances, estimated because they affect the
# data rather than because the design is for them.
#
# The two sequences carry different information. The spoiled steady state in
# closed form depends on T2\* rather than T2, so its T2 row is exactly zero.
# Each block is blind to something and the Fisher matrix adds them up.
#
JOINT = ("T1", "T2", "M0", "B0")


def rows(simulator, **design):
    """The Jacobian rows for every joint parameter, zero where the block is blind."""
    present = [name for name in JOINT if name in simulator.exposes]
    _, jacobian = simulator.jacobian(present, **design)
    placed = jacobian.new_zeros(jacobian.shape[:-2] + (len(JOINT), jacobian.shape[-1]))
    where = torch.tensor([JOINT.index(name) for name in present])
    return placed.index_copy(-2, where, jacobian)


def bounds(spgr_flip, ssfp_flip):
    """The Cramer-Rao bound on each joint parameter, for each design tissue."""
    together = torch.cat(
        (rows(spgr, flip=spgr_flip), rows(ssfp, flip=ssfp_flip)), dim=-1
    )
    return torchsim.crlb(together, noise_variance=NOISE**2)


# %%
#
# Dividing each bound by its own parameter squared makes the terms
# dimensionless, so a 100 ms T2 and a 1000 ms T1 are weighted by how well they
# are known rather than by how large they are. The logarithm makes the gradient
# relative, so the design does not depend on the noise level.
#


def precision(spgr_flip, ssfp_flip):
    """Relative variance of T1 and T2, averaged over the design tissues."""
    bound = bounds(spgr_flip, ssfp_flip)
    relative = bound[..., 0] / T1_MS**2 + bound[..., 1] / T2_MS**2
    return relative.mean().log()


# %%
#
# Design
# ------
#
# Four scans of each kind, starting from a spread of angles. The limits are
# what the scanner will play and are enforced exactly.
#
spgr_start = torch.tensor([2.0, 4.0, 8.0, 16.0])
ssfp_start = torch.tensor([10.0, 20.0, 40.0, 60.0])

design = SequenceDesign(
    precision,
    spgr_flip=Bounded(spgr_start, 1.0, 40.0),
    ssfp_flip=Bounded(ssfp_start, 1.0, 70.0),
)

start = time.time()
result = design.minimize(iterations=120, learning_rate=0.3)
design_time = time.time() - start

spgr_designed = result.parameters["spgr_flip"]
ssfp_designed = result.parameters["ssfp_flip"]

# %%
#
# What it buys, as the standard deviation of each estimate in percent of the
# value itself.
#

# sphinx_gallery_start_ignore
for label, angles_pair in (
    ("published spread", (spgr_start, ssfp_start)),
    ("designed", (spgr_designed, ssfp_designed)),
):
    bound = bounds(*angles_pair)
    sigma_t1 = 100.0 * bound[..., 0].sqrt() / T1_MS
    sigma_t2 = 100.0 * bound[..., 1].sqrt() / T2_MS
    print(
        f"{label:18s} "
        f"sigma(T1)/T1 = {sigma_t1[0]:.1f}%, {sigma_t1[1]:.1f}%   "
        f"sigma(T2)/T2 = {sigma_t2[0]:.1f}%, {sigma_t2[1]:.1f}%"
    )
print(f"designed in {design_time:.1f} s")
# sphinx_gallery_end_ignore

# %%
#
# Optimized schedule
# ------------------
#
# What the scanner is handed: eight scans, four spoiled and four balanced, each
# differing only in flip angle. Before and after.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), sharey=True)
for axis, start_angles, designed, title in (
    (axes[0], spgr_start, spgr_designed, "SPGR block"),
    (axes[1], ssfp_start, ssfp_designed, "bSSFP block"),
):
    index = torch.arange(1, start_angles.numel() + 1)
    axis.bar(
        index - 0.19, start_angles, width=0.36, color="grey", label="published spread"
    )
    axis.bar(
        index + 0.19, designed.detach(), width=0.36, color="crimson", label="designed"
    )
    for position, angle in zip(index, designed.detach(), strict=False):
        axis.annotate(
            f"{float(angle):.0f}",
            (float(position) + 0.19, float(angle)),
            ha="center",
            va="bottom",
        )
    axis.set(xlabel="Acquisition", title=title, xticks=index.tolist())
    axis.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Flip angle [deg]")
axes[0].margins(y=0.15)  # headroom for the annotation over the tallest bar
key(figure, ncols=2)
# sphinx_gallery_end_ignore

# %%
#
# Optimized flip angles
# ---------------------
#
# The design collapses eight distinct angles onto three and repeats them. The
# information sits at a few places on each curve, and a fixed number of scans
# is best spent there rather than sampling the curve evenly.
#
# The SPGR angle lands above the Ernst angle of both tissues, where the curve
# separates the two T1 values most sharply; the peak itself is where the signal
# is largest and says least. The two bSSFP angles sit either side of the
# steady-state maximum, which is what makes the pair sensitive to T2. The upper
# one is against its limit rather than at an interior optimum, and that limit
# is what the deposited RF power allows.
#

# sphinx_gallery_start_ignore
sweep = torch.linspace(1.0, 70.0, 200)
figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.6))

for axis, simulator, start_angles, designed, title in (
    (axes[0], spgr, spgr_start, spgr_designed, "SPGR"),
    (axes[1], ssfp, ssfp_start, ssfp_designed, "bSSFP"),
):
    curve = simulator.simulate(flip=sweep).abs()
    axis.plot(sweep, curve[0], label="T1/T2 = 830/80 ms")
    axis.plot(sweep, curve[1], label="T1/T2 = 1330/110 ms")
    sampled = simulator.simulate(flip=start_angles).abs()
    axis.plot(start_angles, sampled[0], "o", color="grey", label="start")
    sampled = simulator.simulate(flip=designed).abs()
    axis.plot(designed, sampled[0], "*", ms=14, color="crimson", label="designed")
    axis.set(xlabel="Flip angle [deg]", ylabel="|signal|", title=title)
    axis.grid(alpha=0.3)
key(figure, ncols=4)

axes[2].plot(result.loss.cpu())
axes[2].set(xlabel="Iteration", ylabel="log relative CRLB", title="convergence")
axes[2].grid(alpha=0.3)
# sphinx_gallery_end_ignore

# %%
#
# Effect on the maps
# ------------------
#
# The phantom is the one the parameter-inference examples map: BrainWeb
# subject 0, slice 90, whose fuzzy memberships give a T1, a T2 and a proton
# density known at every voxel. Both protocols are played on it and compared
# against the truth rather than against each other.
#

# sphinx_gallery_start_ignore
import csv
from pathlib import Path

import brainweb_dl
import numpy as np
from brainweb_dl import get_mri

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
# drawn from its first row down.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
mask = occupancy > 0.5
share = np.maximum(occupancy, 1e-6)

T1_true = np.where(mask, fractions @ tissue_T1 / share, 0.0).astype(np.float32)
T2_true = np.where(mask, fractions @ tissue_T2 / share, 0.0).astype(np.float32)
M0_true = np.where(mask, fractions @ tissue_PD, 0.0).astype(np.float32)

truth = {
    "T1": torch.as_tensor(T1_true[mask].copy()),
    "T2": torch.as_tensor(T2_true[mask].copy()),
    "M0": torch.as_tensor(M0_true[mask].copy()),
}
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
# The two blocks are one experiment and are fitted as one: a
# :class:`~torchsim.model.SignalModel` that plays each and concatenates what
# they record. The fit is the one thing held fixed between the protocols --
# the same nonlinear least squares over the same four unknowns, from the same
# guess.
#
from torchsim.estimators import NonlinearLeastSquares
from torchsim.model import SignalModel


class JointRelaxometry(SignalModel):
    """Both blocks at fixed flip angles, as one signal model."""

    properties = ("T1", "T2", "M0", "B0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence):
        """The two blocks, end to end along the contrast axis."""
        T1, T2 = properties["T1"], properties["T2"]
        M0 = properties.get("M0", 1.0)
        B0 = properties.get("B0", 0.0)
        return torch.cat(
            (
                self.spoiled.simulate(T1=T1, T2star=T2, M0=M0, B0=B0),
                self.balanced.simulate(T1=T1, T2=T2, M0=M0, B0=B0),
            ),
            dim=-1,
        )


# %%
#
# The noise is independent on the real and imaginary channels, each at the
# standard deviation the bound was computed with.
#
UNKNOWN = {
    "T1": (200.0, 5000.0),
    "T2": (20.0, 600.0),
    "M0": (0.1, 2.0),
    "B0": (-50.0, 50.0),
}
generator = torch.Generator().manual_seed(7)


joint = JointRelaxometry(spgr_designed, ssfp_designed)

# sphinx_gallery_start_ignore
clean = joint.simulate(B0=0.0, **truth)
noise = torch.randn((2, *clean.shape), generator=generator, dtype=torch.float32)
# sphinx_gallery_end_ignore
measured = clean + NOISE * torch.complex(noise[0], noise[1])

problem = NonlinearLeastSquares(
    joint,
    bounds=UNKNOWN,
    initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
).fit(UNKNOWN, noise_std=NOISE, seed=0)

maps = problem(measured)  # {"T1": ..., "T2": ..., "M0": ..., "B0": ...}


# sphinx_gallery_start_ignore
def mapped(spgr_flip, ssfp_flip):
    """Play both blocks over the slice, then fit every voxel."""
    block = JointRelaxometry(spgr_flip, ssfp_flip)
    exact = block.simulate(B0=0.0, **truth)
    draw = torch.randn((2, *exact.shape), generator=generator, dtype=torch.float32)
    seen = exact + NOISE * torch.complex(draw[0], draw[1])

    fit = NonlinearLeastSquares(
        block,
        bounds=UNKNOWN,
        initial={"T1": 1000.0, "T2": 100.0, "M0": 1.0, "B0": 0.0},
    ).fit(UNKNOWN, noise_std=NOISE, seed=0)
    start = time.time()
    found = fit(seen)
    return found, time.time() - start


before, before_seconds = mapped(spgr_start, ssfp_start)
after, after_seconds = mapped(spgr_designed, ssfp_designed)
print(f"{2 * int(mask.sum())} joint fits in {before_seconds + after_seconds:.1f} s")
# sphinx_gallery_end_ignore

# %%
#
# The bound was computed for two tissues; the slice has thousands. Evaluated at
# every voxel's own relaxation times it becomes a predicted precision map,
# which is what the measured error is read against.
#


def predicted_sigma(spgr_flip, ssfp_flip):
    """Relative standard deviation the bound allows, voxel by voxel."""
    at_voxel = tuple(
        sequence.bind(T1=truth["T1"], M0=1.0, B0=0.0, **{name: truth["T2"]})
        for sequence, name in (
            (SPGRSimulator(TE=2.0, TR=6.0), "T2star"),
            (bSSFPSimulator(TE=2.5, TR=5.0), "T2"),
        )
    )
    together = torch.cat(
        (rows(at_voxel[0], flip=spgr_flip), rows(at_voxel[1], flip=ssfp_flip)), dim=-1
    )
    bound = torchsim.crlb(together, noise_variance=NOISE**2)
    return {
        "T1": bound[..., 0].sqrt() / truth["T1"],
        "T2": bound[..., 1].sqrt() / truth["T2"],
    }


# sphinx_gallery_start_ignore
expected = {
    "published spread": predicted_sigma(spgr_start, ssfp_start),
    "designed": predicted_sigma(spgr_designed, ssfp_designed),
}
# sphinx_gallery_end_ignore

# %%
#
# What the design bought, over the brain rather than two tissues. No unbiased
# estimator beats the bound and a good one approaches it, so the two columns
# agreeing is the check that the design optimized the right thing.
#
# Both are root-mean-square, because a bound is a standard deviation: the
# median absolute error is about two thirds of one and would flatter the
# estimator by that factor.
#

# sphinx_gallery_start_ignore
found = {"published spread": before, "designed": after}


def rms(values):
    """The second moment, which is what a standard deviation is."""
    return 100.0 * float(values.square().mean().sqrt())


print(f"\n{'':18}{'T1':>26}{'T2':>26}")
print(f"{'':18}{'measured':>13}{'bound':>13}{'measured':>13}{'bound':>13}")
for label, maps in found.items():
    line = f"{label:18}"
    for name in ("T1", "T2"):
        error = (maps[name] - truth[name]) / truth[name]
        line += f"{rms(error):12.1f}%{rms(expected[label][name]):12.1f}%"
    print(line)
# sphinx_gallery_end_ignore

# %%
#
# The maps, and the error each protocol leaves. The designed protocol gives the
# same picture with less noise in it, which is what a precision design buys.
#


# sphinx_gallery_start_ignore
def painted(values):
    """A flat vector of brain voxels, back in the shape of the slice."""
    canvas = np.zeros(mask.shape, dtype=np.float32)
    canvas[mask] = values.numpy(force=True)
    return canvas


for name in ("T1", "T2"):
    reference = T1_true if name == "T1" else T2_true
    cmap, limits, label = STYLE[name]
    residuals = {
        method: np.abs(painted(maps[name]) - reference)
        for method, maps in found.items()
    }
    top = max(float(np.percentile(values[mask], 98)) for values in residuals.values())

    figure, axes = canvas(2, 1 + len(found), mask.shape)
    panel(axes[0, 0], reference, cmap, limits, title="truth")
    axes[1, 0].set_visible(False)
    for column, method in enumerate(found, start=1):
        estimate = panel(
            axes[0, column], painted(found[method][name]), cmap, limits, title=method
        )
        error = panel(axes[1, column], residuals[method], "inferno", (0.0, top or 1.0))
    scalebar(estimate, axes[0], label)
    scalebar(error, axes[1, 1:], f"|error|, {label}")
# sphinx_gallery_end_ignore

# %%
#
# References
# ----------
#
# .. [1] Teixeira, R. P. A. G., Malik, S. J., Hajnal, J. V., "Joint system
#    relaxometry (JSR) and Cramer-Rao lower bound optimization of sequence
#    parameters: a framework for enhanced precision of DESPOT T1 and T2
#    estimation", Magnetic Resonance in Medicine 79.1 (2018), pp. 234-245.
#    https://doi.org/10.1002/mrm.26670
