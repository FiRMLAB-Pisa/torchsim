"""
==============================================
Nonlinear inversion from k-space
==============================================

The scope of this notebook is to reconstruct T2 maps straight from k-space,
with the signal model inside the forward operator, and to say where the time
and the memory go.

Physics-based reconstruction removes the intermediate images. The forward
operator is a chain

.. math::

   F = P \\, \\mathcal{F} \\, C \\, M

-- sampling, Fourier encoding, coil sensitivities, and the **signal model** --
and the parameter maps are solved for directly against the k-space that was
measured. Only the last factor changes with the sequence, and it is the only
one TorchSim supplies: :class:`~torchsim.recon.ModelOperator` turns any
simulator into it, and the encoding comes from mri-nufft.

Unlike a subspace this stays nonlinear, so it needs a starting guess and a loop
around it, and it pays for that with a model of any number of parameters where
a basis would have to span their product. The comparison here is against
gridding, by iteratively regularized Gauss-Newton.

Wang X, Tan Z, Scholand N, Roeloffs V, Uecker M. *Physics-based reconstruction
methods for magnetic resonance imaging.* Phil Trans R Soc A 379:20200196
(2021).
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim brainweb-dl cmap mri-nufft[finufft,cufinufft] deepinv

# %%
#
# The phantom is BrainWeb's, reached through ``brainweb-dl``: ``get_mri``
# fetches the fuzzy tissue memberships, and the package ships the table of
# relaxation times that goes with them -- which is what the two standard
# library imports read.
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

# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
from brainweb_dl import get_mri

# %%
#
# The Fourier encoding is not TorchSim's and never will be. ``mri-nufft``
# supplies the radial trajectory and the non-uniform transform that plays it;
# ``deepinv`` supplies the :class:`~deepinv.physics.LinearPhysics` base class
# the encoding operator is written against, and the linear solver a
# Gauss-Newton step hands its linearized problem to. Anything exposing ``A``
# and ``A_adjoint`` composes with what TorchSim supplies.
#
import mrinufft
from deepinv.physics import LinearPhysics
from mrinufft.trajectories import initialize_2D_radial

# %%
#
# From TorchSim: the sequence, the estimator the contrast-then-fit routes
# need, and :class:`~torchsim.recon.ModelOperator`, which is the signal
# model as a factor of the forward operator.
#
import time

import numpy as np
import torch

from torchsim.estimators import DictionaryMatcher
from torchsim.recon import GaussNewton, ModelOperator, Schedule, iterative
from torchsim.simulators import MultiEchoSimulator


# %%
#
# What the experiment is: a 96 matrix read as 16 radial spokes per echo, eight
# echoes, and the rank the baseline's estimator compresses to.
#
SIZE = 96
ECHOES = 8
SPOKES = 16
SAMPLES = 192
RANK = 3

# The GPU transform is used when it is both installed and usable; the
# simulation follows it, so the images and the operator meet on one device.
on_gpu = torch.cuda.is_available() and mrinufft.check_backend("cufinufft")
device = "cuda" if on_gpu else "cpu"
backend = "cufinufft" if on_gpu else "finufft"

# %%
#
# Phantom
# -------
#
# BrainWeb subject 0, slice 90, resampled to the matrix reconstructed here.
# BrainWeb publishes fuzzy memberships rather than labels, so weighting the
# tabulated relaxation times by them gives a T2 map whose mixed voxels sit
# between the pure ones, known everywhere.
#

# sphinx_gallery_start_ignore
BRAIN_TISSUES = (1, 2, 3, 8)  # CSF, grey matter, white matter, glial matter
SLICE = 90

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
rows = list(csv.DictReader(table.open()))
tissue_T2 = np.array([float(r["T2 (ms)"]) for r in rows])[list(BRAIN_TISSUES)]
tissue_PD = np.array([float(r["PD (ms)"]) for r in rows])[list(BRAIN_TISSUES)]

fractions = get_mri(sub_id=0, contrast="fuzzy")[SLICE].astype(np.float32)
fractions = fractions[..., list(BRAIN_TISSUES)]
# BrainWeb's first in-plane axis runs posterior to anterior, and an image is
# drawn from its first row down. Flipping here puts anterior at the top of
# every figure below rather than in each one of them.
fractions = np.flipud(fractions).copy()
occupancy = fractions.sum(-1)
share = np.maximum(occupancy, 1e-6)


def resampled(values):
    """The slice at the matrix size this example reconstructs."""
    grid = torch.as_tensor(np.asarray(values, np.float32))[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0].to(device)


T2_true = resampled(np.where(occupancy > 0.5, fractions @ tissue_T2 / share, 0.0))
M0_true = resampled(np.where(occupancy > 0.5, fractions @ tissue_PD, 0.0))
brain = resampled((occupancy > 0.5).astype(np.float32)) > 0.5
T2_true = torch.where(
    brain, T2_true.clamp(20.0, 400.0), torch.tensor(20.0, device=device)
)

# sphinx_gallery_end_ignore

# %%
#
# Sequence and sampling
# ---------------------
#
# A multi-echo spin echo on a golden-angle radial trajectory that rotates
# between echoes. Sixteen spokes per echo across a 96-sample matrix is roughly
# ninefold undersampled. The protocol stays on the host;
# :class:`~torchsim.recon.ModelOperator` takes it wherever the maps are.
#
TE = torch.linspace(10.0, 150.0, ECHOES)
simulator = MultiEchoSimulator(TE=TE)

images = (
    torch.as_tensor(simulator.to(device).simulate(T2=T2_true)).to(torch.complex64)
    * M0_true.to(torch.complex64)[..., None]
)

trajectory = (
    initialize_2D_radial(SPOKES * ECHOES, SAMPLES, tilt="golden")
    .astype(np.float32)
    .reshape(ECHOES, SPOKES * SAMPLES, 2)
)

build = mrinufft.get_operator(backend)
per_echo = [
    build(trajectory[echo], (SIZE, SIZE), n_coils=1, squeeze_dims=False, density=True)
    for echo in range(ECHOES)
]


class RadialEncoding(LinearPhysics):
    """``(batch, echoes, x, y)`` images to k-space, one trajectory per echo.

    This is the whole of ``P F C`` for this experiment, and none of it is
    TorchSim's: it wraps mri-nufft, which is what a real pipeline would do
    with its own trajectory, its own density compensation and its own coils.
    """

    def A(self, x, **kwargs):
        return torch.stack(
            [per_echo[e].op(x[:, e][:, None])[:, 0] for e in range(ECHOES)], 1
        )

    def A_adjoint(self, y, **kwargs):
        return torch.stack(
            [per_echo[e].adj_op(y[:, e][:, None])[:, 0] for e in range(ECHOES)], 1
        )


encoding = RadialEncoding()
kspace = encoding.A(images.movedim(-1, 0)[None])

# The k-space is scaled so the adjoint image peaks at one. Every damping
# weight below is then a number about the model rather than about the
# receiver gain, which is what makes one choice of it transferable.
gridded = encoding.A_adjoint(kspace)[0].movedim(0, -1)
scale = float(gridded.abs().max())
kspace, gridded = kspace / scale, gridded / scale

# sphinx_gallery_start_ignore
undersampling = (0.5 * np.pi * SIZE) / SPOKES
print(f"{SPOKES} spokes per echo: {undersampling:.0f}x undersampled")
# sphinx_gallery_end_ignore

# %%
#
# The maps on the left are what every route recovers; the spokes on the right
# are all that is measured of them, one echo's worth, rotated by the golden
# angle from the echo before.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.58))
for axis, values, name, title in (
    (axes[0], T2_true, "T2", "ground truth"),
    (axes[1], M0_true, "M0", "proton density"),
):
    cmap, limits, label = STYLE[name]
    handle = panel(axis, values.cpu().numpy(), cmap, limits, title=title)
    axis.figure.colorbar(handle, ax=axis, label=label, fraction=0.046, pad=0.03)
    axis.set_box_aspect(1)

for echo in (0, ECHOES // 2, ECHOES - 1):
    arm = trajectory[echo].reshape(SPOKES, SAMPLES, 2)
    for spoke in range(SPOKES):
        axes[2].plot(
            arm[spoke, :, 0],
            arm[spoke, :, 1],
            lw=0.4,
            color=plt.cm.plasma(echo / (ECHOES - 1)),
        )
axes[2].set(
    xlabel="$k_x$",
    ylabel="$k_y$",
    title=f"{SPOKES} spokes per echo, 3 of {ECHOES} shown",
)
# An image keeps its own aspect and a line plot fills whatever it is given, so
# the box each panel is drawn into has to be fixed for the row to line up.
axes[2].set_box_aspect(1)
# A colorbar on the trajectory would have nothing to scale, but the panel has
# to lose the same width as the two beside it or the images shrink.
bar = figure.colorbar(handle, ax=axes[2], fraction=0.046, pad=0.03)
bar.ax.set_visible(False)
# sphinx_gallery_end_ignore

# %%
#
# Estimator for the baseline
# --------------------------
#
# The baseline reconstructs images and then fits them, so it needs an
# estimator, stated over a compressed basis: three directions hold essentially
# all of an eight-echo exponential. The nonlinear route has no such step --
# its answer is the maps.
#
grid = torch.linspace(20.0, 400.0, 500)
mapping = DictionaryMatcher(simulator).fit(T2=grid, M0=1.0, rank=RANK, seed=0)

# sphinx_gallery_start_ignore
print(f"rank {RANK} of {ECHOES} contrasts keeps {mapping.subspace.retained:.6f}")


def clock():
    """Wall clock, with the card caught up first."""
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def report(name, seconds, found):
    """One route's cost and its error over the brain.

    Every route is timed the same way -- reconstruction *and* fit -- because
    that is what a pipeline costs. A route that reconstructs quickly and then
    fits eight images is not a quick route.
    """
    error = (found[brain] - T2_true[brain]).abs()
    print(
        f"{name:<26} {seconds:5.1f}s   "
        f"T2 error {float(error.mean()):5.1f} ms "
        f"({100 * float((error / T2_true[brain]).mean()):4.1f}%)"
    )
    return found


# sphinx_gallery_end_ignore

# %%
#
# Baseline reconstruction
# -----------------------
#
# Gridding is the adjoint with a density weighting -- one pass, smooth, biased
# -- and the estimator above turns its eight images into a T2 map. Sixteen
# spokes of 192 samples is 3072 measurements against 9216 unknowns, so each
# echo alone is underdetermined and iterating has nothing to converge to.
# Accuracy comes from a constraint across the echoes, which is the model.
#

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
adjoint = mapping(gridded)["T2"]

# sphinx_gallery_start_ignore
report("adjoint per echo", clock() - started, adjoint)
# sphinx_gallery_end_ignore

# %%
#
# Nonlinear model
# ---------------
#
# The signal model stays inside the forward operator and the maps are solved
# for against k-space directly. Two things are declared:
#
# * **what is unknown** -- ``T2``, plus the complex amplitude the operator
#   carries for it, which is proton density and receive phase together;
# * **what T2 may be** -- a box bound, kept by solving for a transformed
#   variable so no iterate leaves it. That matters more here than in a fit:
#   the model is evaluated at every voxel to predict every k-space sample, so
#   one unphysical voxel corrupts the whole residual.
#
# An equality constraint would be written into the model instead.
#
operator = ModelOperator(simulator, "T2", bounds={"T2": (20.0, 400.0)})

# The amplitude starts from the first gridded echo, which is nearly free and
# is most of what makes the first Newton step sensible.
initial = operator.initial((1, SIZE, SIZE), T2=100.0).to(device)
initial[0, ..., 1] = gridded[..., 0].real
initial[0, ..., 2] = gridded[..., 0].imag

# %%
#
# An iteratively regularized Gauss-Newton: linearize, solve, step, lower the
# damping. TorchSim supplies the loop and the derivative but not the linear
# solver -- :func:`~torchsim.recon.iterative` hands the linearized problem to
# the same deepinv routine the baseline called. A proximal solver under a
# wavelet prior is a change to that one argument.
#

# sphinx_gallery_start_ignore
started = clock()
# sphinx_gallery_end_ignore
found = GaussNewton(
    Schedule(initial=1e-3, factor=0.5, minimum=1e-7),
    solve=iterative(max_iter=20),
    max_iterations=8,
).minimize(operator, kspace, initial, encoding=encoding)

# No fit afterwards: the maps are what was solved for.
modelled = operator.split(found.x)["T2"][0]

# sphinx_gallery_start_ignore
report("model-based", clock() - started, modelled)
print(
    f"residual {float(found.cost[0]):.3e} -> {float(found.cost[-1]):.3e}, "
    f"damping {float(found.damping[0]):.0e} -> {float(found.damping[-1]):.0e}"
)
# sphinx_gallery_end_ignore

# %%
#
# Timing
# ------
#
# Each conjugate-gradient step costs one product with the Jacobian and one with
# its adjoint, and each is the encoding operator once and the model once.
# Timing the four says which half a faster reconstruction would come from; here
# they are comparable.
#
# Neither product builds the Jacobian. That is a memory argument: the blocks
# are ``voxels x channels x contrasts`` where a signal is ``voxels x
# contrasts``, so what is not held is the channel count times the signal, every
# iteration.
#
tangent = torch.randn_like(initial)
predicted = operator.A_jvp(initial, tangent)
adjoint_image = encoding.A_adjoint(kspace).movedim(1, -1)


# sphinx_gallery_start_ignore
def timed(call, repeats=5):
    """Wall clock, after a warm-up, because the first call plans transforms."""
    call()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if device == "cuda":
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / repeats


print(f"per conjugate-gradient step, {operator.channels} channels solved for:")
print(f"  model    J  v   {timed(lambda: operator.A_jvp(initial, tangent)):6.1f} ms")
print(
    f"  model    J^H v  {timed(lambda: operator.A_vjp(initial, adjoint_image)):6.1f} ms"
)
print(
    f"  encoding A      {timed(lambda: encoding.A(predicted.movedim(-1, 1))):6.1f} ms"
)
print(f"  encoding A^H    {timed(lambda: encoding.A_adjoint(kspace)):6.1f} ms")

blocks = operator.jacobian(initial)
print(
    f"the Jacobian this avoids holding: "
    f"{blocks.numel() * 8 / 2**20:.1f} MiB, against "
    f"{predicted.numel() * 8 / 2**20:.1f} MiB for a signal"
)
# sphinx_gallery_end_ignore

# %%
#
# Maps
# ----
#

# sphinx_gallery_start_ignore
shown = (
    ("truth", T2_true),
    ("adjoint", adjoint),
    ("model-based", modelled),
)

cmap, limits, label = STYLE["T2"]
figure, axes = canvas(2, len(shown), T2_true.shape)
axes[1, 0].set_visible(False)
for column, (title, values) in enumerate(shown):
    picture = torch.where(brain, values, torch.tensor(0.0, device=device))
    estimate = panel(axes[0, column], picture.cpu().numpy(), cmap, limits, title=title)
    if column == 0:
        continue
    difference = torch.where(
        brain, (values - T2_true).abs(), torch.tensor(0.0, device=device)
    )
    error = panel(axes[1, column], difference.cpu().numpy(), "inferno", (0, 80))
scalebar(estimate, axes[0], label)
scalebar(error, axes[1, 1:], f"|error|, {label}")
# sphinx_gallery_end_ignore

# %%
#
# Writing a different model
# -------------------------
#
# The model is the only thing above that names a relaxation time, and it is an
# ordinary :class:`~torchsim.model.Simulator` -- the same object the fitting
# and sequence-design notebooks use. Water-fat separation, T2* with a field
# map, a Look-Locker inversion recovery: each is a different ``evaluate``, and
# the operator, the loop and the encoding are unchanged.
#
