"""
=============================
Synthetic MR fingerprinting
=============================

The scope of this notebook is to build a training pair end to end: what a
scanner would measure from a fingerprinting exam, and the maps it came from.

A subject is segmented into tissue classes; each class is given an M0, a T1 and
a T2; one voxel per class is simulated by extended phase graphs; every voxel of
a class is handed its class's evolution; the volume is weighted by birdcage
sensitivities and pushed through a frame-wise non-uniform Fourier transform;
the k-space is brought back and coil-combined. The undersampled series and the
fully sampled one are the pair, with the ground-truth maps and the
segmentation.

Only the simulation is TorchSim's. The phantom, the coils and the encoding come
from torchio, deepmriprep, SigPy and mri-nufft.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install torchsim torchio deepmriprep sigpy cmap mri-nufft[finufft,cufinufft]

# %%
#
# Only one step of this pipeline is TorchSim's. Four other packages do the
# rest, and each has exactly one job here:
#
# * ``torchio`` fetches a T1-weighted IXI subject and gives it as tensors,
#   which is where the anatomy comes from;
# * ``deepmriprep`` segments that anatomy into grey matter, white matter and
#   CSF with a U-Net -- torch throughout, and it hands back probabilities
#   rather than labels;
# * ``sigpy.mri`` generates birdcage coil sensitivities to weight the volume
#   with;
# * ``mri-nufft`` supplies the spiral trajectory and the non-uniform Fourier
#   transform that plays it, forwards and back.
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
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
PHASE = Colormap("colorcet:CET_C6").to_matplotlib()

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


def domain(axis, values, title=None):
    """A complex map the way coil sensitivities are read: phase in colour,
    magnitude in opacity, so an unsupported corner reads as background rather
    than as a phase."""
    values = torch.as_tensor(values).cpu()
    rgba = PHASE(((values.angle() / (2 * np.pi)) + 0.5).numpy())
    magnitude = values.abs().numpy()
    rgba[..., -1] = magnitude / max(magnitude.max(), 1e-12)
    axis.set_facecolor("black")
    axis.imshow(rgba)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_box_aspect(1)
    if title is not None:
        axis.set_title(title)


def scalebar(handle, axes, label):
    """One colorbar for a group of panels, so none gives up width to its own."""
    axes = list(np.ravel(axes))
    axes[0].figure.colorbar(handle, ax=axes, label=label, shrink=0.92, aspect=20)


# Every panel on this page is drawn at the same size, so any two figures can be
# read against each other. The side is set by the widest grid, which fills the
# documentation column; a figure with fewer columns is narrower, not larger.
PAGE_WIDTH = 8.6  # inches, the width of the documentation column
BAR_WIDTH = 0.8  # what one colorbar takes out of it
PANEL = (PAGE_WIDTH - BAR_WIDTH) / 5.5  # one image panel


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
import mrinufft
from deepmriprep import Preprocess
import sigpy.mri as smri
import torchio as tio
from mrinufft.trajectories import initialize_2D_spiral

# %%
#
# TorchSim's part is the third step: one fingerprinting simulation per tissue
# class, rather than one per voxel.
#
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from torchsim.simulators import MRFSimulator


# %%
#
# What the pair will be: a 128 matrix, four hundred frames, one spiral arm of
# 768 samples per frame, and eight receive channels.
#
SIZE = 128
FRAMES = 400
SAMPLES = 768
COILS = 8
SLICE = 110  # axial, at the level of the lateral ventricles

# The GPU transform is used when it is both installed and usable; the
# simulation follows it, so the images and the operator meet on one device.
on_gpu = torch.cuda.is_available() and mrinufft.check_backend("cufinufft")
device = "cuda" if on_gpu else "cpu"
backend = "cufinufft" if on_gpu else "finufft"

# %%
#
# Subject
# -------
#
# One IXI subject through torchio: a T1-weighted volume at 1.5 T, and the only
# measurement this notebook starts from. The segmentation reads it; a table
# supplies the tissue properties a contrast cannot give.
#
# ``download=True`` fetches the archive once, a few hundred MB, into a cache
# under the home directory.
#
CACHE = Path.home() / ".cache" / "torchsim" / "ixi-tiny"
subject = tio.datasets.IXITiny(str(CACHE), download=True)[0]


# sphinx_gallery_start_ignore
def slab(image):
    """One axial slice of a volume, at the matrix this example works in."""
    values = np.asarray(image.numpy(), dtype=np.float32).squeeze()[:, :, SLICE].T
    grid = torch.as_tensor(np.flip(values, 0).copy())[None, None]
    return torch.nn.functional.interpolate(
        grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False
    )[0, 0]


# sphinx_gallery_end_ignore

# %%
#
# Tissue classes
# --------------
#
# ``deepmriprep`` segments the head into grey matter, white matter and CSF with
# a U-Net, in torch and on the same card as everything else. It returns three
# probability maps rather than one label per voxel, which is the difference
# between a phantom with partial volume in it and one without.
#
NAMES = ("grey matter", "white matter", "CSF")

segmentation = Preprocess().run(
    str(subject.image.path),
    output_paths={
        name: f"{tempfile.gettempdir()}/{name}.nii.gz" for name in ("p1", "p2", "p3")
    },
    run_all=False,
)

# sphinx_gallery_start_ignore
# The probability maps come back on their own grid, and carry the affine that
# places them in the subject's coordinates, so resampling by it is all the
# alignment they need. The rest puts every volume in RAS on a 1 mm isotropic
# grid, so that one index is the same axial slice everywhere and a square
# slice is a square image.
for key, name in zip(("p1", "p2", "p3"), NAMES, strict=False):
    subject.add_image(
        tio.ScalarImage(
            tensor=torch.as_tensor(segmentation[key].get_fdata())[None].float(),
            affine=segmentation[key].affine,
        ),
        name,
    )
subject = tio.Compose(
    [tio.ToCanonical(), tio.Resample("image"), tio.Resample(1), tio.CropOrPad(240)]
)(subject)

anatomy = slab(subject.image)
fractions = torch.stack([slab(subject[name]) for name in NAMES], dim=-1)
fractions = fractions.clamp(0.0, 1.0)
occupancy = fractions.sum(-1)
brain = occupancy > 0.5

mixed = int(((fractions.max(-1).values < 0.99) & brain).sum())
print(
    f"{SIZE}x{SIZE} slice, {int(brain.sum())} brain voxels; "
    f"{100 * mixed / int(brain.sum()):.0f}% are a mixture of two tissues or more"
)
# sphinx_gallery_end_ignore

# %%
#
# The contrast the subject arrived as, and the three probabilities the network
# made of it:
#

# sphinx_gallery_start_ignore
# Four panels across the column leave room for a word each; the paragraph above
# says which contrast and which tissue.
figure, axes = canvas(1, 4, (SIZE, SIZE))
panel(axes[0, 0], anatomy.cpu(), "gray", (0, float(anatomy.max())), title="T1w")
for column, name in enumerate(("grey", "white", "CSF"), start=1):
    handle = panel(
        axes[0, column], fractions[..., column - 1].cpu(), "magma", (0, 1), title=name
    )
scalebar(handle, axes[0, 1:], "probability")
# sphinx_gallery_end_ignore

# %%
#
# Each class is given the three numbers a simulation needs, tabulated at 1.5 T.
# None could be read off this subject: a T1-weighted volume is a contrast, not
# a map. The measurement supplies where each tissue is and in what proportion;
# the table supplies what each class is taken to be. A relaxometry protocol on
# the same subject would fill the table from data instead.
#
NAMES_M0 = torch.tensor([0.80, 0.70, 1.00])  # relative proton density
NAMES_T1 = torch.tensor([1100.0, 650.0, 4000.0])  # ms, at 1.5 T
NAMES_T2 = torch.tensor([95.0, 70.0, 2000.0])  # ms, at 1.5 T

dominant = fractions > 0.5
class_M0 = NAMES_M0
class_T1 = NAMES_T1
class_T2 = NAMES_T2

# sphinx_gallery_start_ignore
print(f"\n{'tissue':<14} {'voxels':>8}   {'M0':>4} {'T1 (ms)':>8} {'T2 (ms)':>8}")
for k, name in enumerate(NAMES):
    print(
        f"{name:<14} {int(dominant[..., k].sum()):8d}   "
        f"{class_M0[k]:4.2f} {class_T1[k]:8.0f} {class_T2[k]:8.1f}"
    )
# sphinx_gallery_end_ignore

# %%
#
# Simulation per class
# --------------------
#
# An inversion, then four hundred repetitions whose flip angle sweeps.
# :class:`~torchsim.simulators.MRFSimulator` takes arrays of tissue properties,
# so the whole table is one call: three extended phase graph runs rather than
# sixteen thousand.
#
schedule = 5.0 + 55.0 * torch.sin(torch.linspace(0.0, 4 * torch.pi, FRAMES)).abs()
simulator = MRFSimulator(TR=12.0, TI=20.0, T1=class_T1, T2=class_T2)

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
per_class = torch.as_tensor(simulator.simulate(flip=schedule))

# sphinx_gallery_start_ignore
print(
    f"{len(NAMES)} classes x {FRAMES} frames in "
    f"{time.perf_counter() - started:.1f}s -> {tuple(per_class.shape)}"
)
# sphinx_gallery_end_ignore

# %%
#
# Whole-brain volume
# ------------------
#
# A voxel that is part grey matter and part CSF produces the sum of what the
# two do, weighted by how much of each is present -- not the signal of their
# averaged relaxation times. The mixing happens on the signals, which is one
# matrix product against the per-class evolutions.
#
# Averaging the parameters and simulating once is wrong wherever a voxel is not
# pure: an inversion-prepared train is markedly nonlinear in T1.
#
weights = (fractions * class_M0).to(per_class.dtype)
series = (weights @ per_class) * brain[..., None]

# The maps that go out as ground truth are the mixture averages, which is what
# a single-compartment fit of this data could return at best.
share = occupancy.clamp_min(1e-6)
truth_M0 = (fractions @ class_M0) * brain
truth_T1 = (fractions @ class_T1) / share * brain
truth_T2 = (fractions @ class_T2) / share * brain

# %%
#
# Coils and encoding
# ------------------
#
# Birdcage sensitivities from SigPy, then one spiral arm per frame rotated by
# the golden angle. A single arm of 768 samples against a 128 x 128 matrix is
# twenty-one-fold undersampled, which is how MRF is run.
#
sensitivities = torch.as_tensor(smri.birdcage_maps((COILS, SIZE, SIZE))).to(
    torch.complex64
)
trajectory = initialize_2D_spiral(
    FRAMES, SAMPLES, tilt="golden", nb_revolutions=8
).astype(np.float32)

build = mrinufft.get_operator(backend)
# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
arms = [
    build(
        trajectory[frame], (SIZE, SIZE), n_coils=COILS, squeeze_dims=False, density=True
    )
    for frame in range(FRAMES)
]
# sphinx_gallery_start_ignore
print(f"{FRAMES} arms built in {time.perf_counter() - started:.1f}s")
# sphinx_gallery_end_ignore

coil_series = (sensitivities[:, None] * series.movedim(-1, 0)[None]).to(device)

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
kspace = torch.stack(
    [arms[frame].op(coil_series[:, frame][None])[0] for frame in range(FRAMES)]
)
# sphinx_gallery_start_ignore
print(f"forward NUFFT {time.perf_counter() - started:.1f}s -> {tuple(kspace.shape)}")
# sphinx_gallery_end_ignore

# %%
#
# Eight birdcage sensitivities, and one spiral arm per frame rotated so that
# consecutive frames sample different parts of k-space.
#

# sphinx_gallery_start_ignore
# The trajectory is a plot rather than a map, so it gets a panel and a half.
# That makes this the widest row on the page, and PANEL is set from it.
figure = plt.figure(figsize=(5.5 * PANEL + BAR_WIDTH, PANEL + 1.05))
grid = figure.add_gridspec(1, 6, width_ratios=(1, 1, 1, 1, BAR_WIDTH / PANEL, 1.5))
for index in range(4):
    domain(
        figure.add_subplot(grid[0, index]), sensitivities[index], title=f"coil {index}"
    )

bar = figure.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), cmap=PHASE),
    cax=figure.add_subplot(grid[0, 4]),
)
bar.set_label("phase [rad]")
bar.set_ticks([-np.pi, 0.0, np.pi], labels=["$-\\pi$", "0", "$\\pi$"])

axis = figure.add_subplot(grid[0, 5])
for frame in (0, FRAMES // 2, FRAMES - 1):
    axis.plot(
        trajectory[frame, :, 0],
        trajectory[frame, :, 1],
        lw=0.5,
        color=plt.cm.plasma(frame / (FRAMES - 1)),
    )
# Three of the arms, coloured first to last: consecutive frames are rotated by
# the golden angle, so each one samples somewhere the last did not.
axis.set(xlabel="$k_x$", ylabel="$k_y$", title=f"3 of {FRAMES} arms")
axis.set_box_aspect(1)
# sphinx_gallery_end_ignore

# %%
#
# Reconstruction
# --------------
#
# Adjoint per frame, then a sensitivity-weighted coil combination, available
# here because the maps are known. A real pipeline would estimate them.
#

# sphinx_gallery_start_ignore
started = time.perf_counter()
# sphinx_gallery_end_ignore
folded = torch.stack(
    [arms[frame].adj_op(kspace[frame][None])[0] for frame in range(FRAMES)]
)
weights = sensitivities.to(device)
combined = (folded * weights.conj()[None]).sum(1) / (
    weights.abs().square().sum(0)[None] + 1e-6
)
# sphinx_gallery_start_ignore
print(f"adjoint and combine {time.perf_counter() - started:.1f}s")
# sphinx_gallery_end_ignore

# %%
#
# Data pair
# ---------
#
# Each frame is one spiral arm, so the aliasing is worse than the signal. What
# survives is the time course, and that is what a fingerprinting reconstruction
# reads.
#

# sphinx_gallery_start_ignore
reference = (series.movedim(-1, 0) * brain).to(device)
reference = reference / reference.abs().max()
undersampled = combined / combined.abs().max()
here = brain.to(device)

frame_error = float(
    (undersampled[:, here] - reference[:, here]).abs().mean()
    / reference[:, here].abs().mean()
)
courses = undersampled[:, here].movedim(0, -1)
truth_courses = reference[:, here].movedim(0, -1)
courses = courses / courses.norm(dim=-1, keepdim=True)
truth_courses = truth_courses / truth_courses.norm(dim=-1, keepdim=True)
agreement = (courses.conj() * truth_courses).sum(-1).abs()

print(f"per-frame error inside the brain : {100 * frame_error:5.1f}%")
print(
    f"time-course agreement            : median {float(agreement.median()):.3f}, "
    f"tenth percentile {float(agreement.quantile(0.1)):.3f}"
)
# sphinx_gallery_end_ignore

# %%
#
# Fifty percent wrong frame by frame, and the fingerprints still line up above
# 0.86 for nine voxels in ten. That gap is the premise of the method: the input
# is what the scanner gives, artefacts and all, and the target is the curve
# underneath.
#

# %%
#
# Phantom summary
# ---------------
#

# sphinx_gallery_start_ignore
figure, axes = canvas(1, 3, (SIZE, SIZE), bars=3)
for axis, name, values in (
    (axes[0, 0], "T1", truth_T1),
    (axes[0, 1], "T2", truth_T2),
    (axes[0, 2], "M0", truth_M0),
):
    cmap, limits, label = STYLE[name]
    scalebar(panel(axis, values.cpu(), cmap, limits), axis, label)
# sphinx_gallery_end_ignore

# %%
#
# Inspecting the pair
# -------------------
#

# sphinx_gallery_start_ignore
# The row says which of the pair it is and the column what is being shown, so
# neither has to be repeated in six headings over a three-inch panel.
figure, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 5.02))
for column, frame in enumerate((0, FRAMES // 2)):
    axes[0, column].imshow(reference[frame].abs().cpu(), cmap="gray")
    axes[1, column].imshow(undersampled[frame].abs().cpu(), cmap="gray")
    axes[0, column].set_title(f"frame {frame}")
for axis in axes[:, :2].ravel():
    axis.set_xticks([]), axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(False)

voxel = torch.nonzero(here)[int(here.sum()) // 2]
row, column = int(voxel[0]), int(voxel[1])
axes[0, 2].plot(reference[:, row, column].abs().cpu(), lw=1.0)
axes[0, 2].set_title("one voxel")
axes[1, 2].plot(undersampled[:, row, column].abs().cpu(), lw=0.7)
for axis, name in ((axes[0, 0], "fully sampled"), (axes[1, 0], "one spiral arm")):
    axis.set_ylabel(name)
for axis in axes[:, 2]:
    axis.set_xlabel("frame")
# An image keeps its own aspect and a line plot fills whatever it is given, so
# a row of both ends up with panels of different heights unless the box each
# one is drawn into is fixed.
for panel in axes.ravel():
    panel.set_box_aspect(1)
# sphinx_gallery_end_ignore

# %%
#
# Exporting
# ---------
#
# The pair, the ground truth, the segmentation, and the schedule and trajectory
# that produced them. It goes to a temporary directory here so that a
# documentation build leaves no archive behind.
#
contents = {
    "undersampled": undersampled.cpu().numpy(),
    "reference": reference.cpu().numpy(),
    "M0": truth_M0.numpy(),
    "T1": truth_T1.numpy(),
    "T2": truth_T2.numpy(),
    "tissue_probabilities": fractions.numpy(),
    "flip_angles_deg": schedule.numpy(),
    "trajectory": trajectory,
}

# sphinx_gallery_start_ignore
with tempfile.TemporaryDirectory() as folder:
    archive = Path(folder) / "synthetic_mrf.npz"
    np.savez_compressed(archive, **contents)
    reloaded = np.load(archive)
    print(f"{archive.name}  {archive.stat().st_size / 2**20:.1f} MiB")
    for name in contents:
        array = reloaded[name]
        print(f"  {name:<16} {str(array.shape):<20} {array.dtype}")
# sphinx_gallery_end_ignore

# %%
#
# Command-line use
# ----------------
#
# Everything above is fixed except three inputs:
#
# * **a T1-weighted NIfTI**, which replaces the torchio subject and is what
#   the segmentation was trained on, so the CSF map stops being the weak one;
# * **a matfile** carrying the trajectory and the flip-angle schedule, read
#   instead of the spiral and the sine generated here;
# * **an output path**, replacing the temporary directory.
#
# The tissue table is the one thing decided rather than read. A relaxometry
# protocol on the same subject would fill it from data, which is what the
# parameter-inference notebooks do.
