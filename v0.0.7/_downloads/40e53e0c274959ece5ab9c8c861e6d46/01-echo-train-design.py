"""
=======================================
Designing echo trains for image quality
=======================================

The scope of this notebook is to design refocusing flip angles for image
quality rather than for precision: first a single echo train, then a whole
segmented 3D protocol in which each shot carries its own repetition time, echo
train length and angles.

T2 decay across a long train modulates k-space, and that modulation is a point
spread function, so the refocusing angles control the resolution of the image
[1]_. Only the cost distinguishes this from a precision design; the simulator,
the bounded parameters and the loop are the same.

"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install torchsim

# %%
#
# A design is a simulator with its tissue fixed on it, a cost written on what
# it records, and the bounded parameters :class:`~torchsim.SequenceDesign`
# drives.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np


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
import time

import torch

from torchsim.optim import Bounded, SequenceDesign
from torchsim.simulators import FSESimulator

# %%
#
# Tissues
# -------
#
# A PD-weighted knee protocol is read for the separation between fluid and
# cartilage, so the design is for all three tissues at once. Designing for one
# of them would tailor the train to it.
#
TISSUES = {
    #            cartilage  muscle  synovial fluid
    "T1": [1200.0, 1420.0, 3600.0],
    "T2": [35.0, 30.0, 250.0],
}
CARTILAGE, MUSCLE, FLUID = 0, 1, 2

# %%
#
# Blurring
# --------
#
# With the echo index running along one k-space direction, the magnitude of
# the echo train *is* the k-space modulation, and the width of its Fourier
# transform is the blur it adds. That width can be read off the modulation
# without transforming anything: the second moment of :math:`|\mathcal{F}w|^2`
# is the energy in the slope of :math:`w` relative to the energy in
# :math:`w` itself.
#
# Written this way a train that stops early simply contributes fewer terms, so
# trains of different lengths compare on the same footing -- which is what the
# second half of this example needs.
#


def blur(signal, acquired):
    """The width, in pixels, of the point spread a train produces.

    Parameters
    ----------
    signal:
        ``(shots, tissues, echoes)`` echo train magnitudes.
    acquired:
        ``(shots, echoes)``, one where the shot is still acquiring.

    Returns
    -------
    torch.Tensor
        ``(shots, tissues)``.
    """
    pair = acquired[:, None, :-1] * acquired[:, None, 1:]
    step = torch.diff(signal, dim=-1) * pair
    energy = (signal * acquired[:, None, :]).square().sum(-1).clamp_min(1e-12)
    lines = acquired.sum(-1)[:, None]
    return lines / (2 * torch.pi) * (step.square().sum(-1) / energy).sqrt()


# %%
#
# One train
# =========
#
# A 120-echo train has 120 angles but three degrees of freedom: the
# **minimum**, which sets how much the train is spoiled by flow and motion; the
# angle at the **centre of k-space**, which sets the image contrast; and the
# **maximum**, which the deposited RF power limits.
#
# The train blends between them: it starts at the maximum, drops to the minimum
# as the pseudo steady state is established, passes through the centre-of-
# k-space angle where k-space is sampled, and ramps back up.
#
ESP_MS = 5.0
ECHOES = 120
CENTRE_ECHO = 24

one_train = FSESimulator(ESP=ESP_MS, states=12, **TISSUES)
echo = torch.arange(1, ECHOES + 1, dtype=torch.float32)


def ramp(index, start, stop, first, last):
    """A smooth step from ``first`` to ``last`` between two echo indices."""
    span = ((index - start) / (stop - start).clamp_min(1e-3)).clamp(0.0, 1.0)
    return first + (last - first) * span.square() * (3.0 - 2.0 * span)


def shape(index, control, length, centre_echo):
    """The refocusing angles of a train of ``length`` echoes.

    Parameters
    ----------
    index:
        The echo indices to evaluate at, one-based.
    control:
        ``(shots, 3)`` -- the minimum, centre-of-k-space and maximum angles.
    length:
        ``(shots, 1)`` echo train length.
    centre_echo:
        The echo that samples the centre of the shot's k-space band.
    """
    low, middle, high = control[:, 0:1], control[:, 1:2], control[:, 2:3]
    settled = torch.full_like(low, 5.0)
    sampled = torch.full_like(low, float(centre_echo))
    return torch.where(
        index <= 5,
        ramp(index, torch.ones_like(low), settled, high, low),
        torch.where(
            index <= centre_echo,
            ramp(index, settled, sampled, low, middle),
            ramp(index, sampled, length, middle, high),
        ),
    )


# %%
#
# The cost: the image should be sharp, fluid should stand out from cartilage,
# and the RF power should stay where the scanner will accept it.
#
LOWEST = torch.tensor([20.0, 30.0, 60.0])
HIGHEST = torch.tensor([90.0, 160.0, 170.0])
PRESCRIBED = torch.tensor([[50.0, 90.0, 150.0]])

ALWAYS = torch.ones(1, ECHOES)


def power(flip, acquired, TR_ms):
    """Deposited RF power, per shot.

    Refocusing energy divided by the time it is spread over, relative to a
    train of 180 degree pulses. Energy per second is what a scanner limits,
    so a train that ends early gets no credit for the echoes it never played
    and none for a repetition time it does not take.
    """
    energy = ((flip / 180.0).square() * acquired).sum(-1)
    return energy / (TR_ms.squeeze(-1) * 1e-3)


#: What the prescribed train already deposits. RF power is a limit the
#: scanner enforces, so the cost may spend up to it and no further.
POWER_BUDGET = power(
    shape(echo, PRESCRIBED, torch.full((1, 1), float(ECHOES)), CENTRE_ECHO),
    ALWAYS,
    torch.full((1, 1), 1800.0),
).mean()


def single_train(control):
    """Sharpness and contrast from one train of a fixed length."""
    flip = shape(echo, control, torch.full_like(control[:, :1], ECHOES), CENTRE_ECHO)
    signal = one_train.simulate(flip=flip, TR=1800.0).abs()
    at_centre = signal[:, :, CENTRE_ECHO - 1]
    contrast = at_centre[:, FLUID] - at_centre[:, CARTILAGE]
    deposited = power(flip, ALWAYS, torch.full_like(control[:, :1], 1800.0))
    return (
        blur(signal, ALWAYS).mean()
        - 12.0 * contrast.mean()
        + 20.0 * torch.relu(deposited.mean() / POWER_BUDGET - 1.0)
    )


# %%
#
# A conventional prescription to start from: 50 degree minimum, 90 degree
# centre-of-k-space angle, 150 degree maximum. The limits are what the scanner
# will play, and :class:`~torchsim.Bounded` holds them exactly.
#
design = SequenceDesign(single_train, control=Bounded(PRESCRIBED, LOWEST, HIGHEST))

# sphinx_gallery_start_ignore
# One call resolves the protocol's structure, which is then held; the clock
# below measures the design itself.
design.minimize(iterations=1)
start = time.perf_counter()
# sphinx_gallery_end_ignore
one = design.minimize(iterations=25, learning_rate=0.3)

# sphinx_gallery_start_ignore
one_elapsed = time.perf_counter() - start
print(
    f"one train of {ECHOES} echoes designed in {one_elapsed:.2f} s, "
    f"{1000 * one_elapsed / 25:.1f} ms per iteration"
)
# sphinx_gallery_end_ignore

# %%
#
# What it did:
#
LENGTH = torch.full((1, 1), float(ECHOES))
prescribed_flip = shape(echo, PRESCRIBED, LENGTH, CENTRE_ECHO)
designed_flip = shape(echo, one.parameters["control"], LENGTH, CENTRE_ECHO)
prescribed_signal = one_train.simulate(flip=prescribed_flip, TR=1800.0).abs()
designed_signal = one_train.simulate(flip=designed_flip, TR=1800.0).abs()

# sphinx_gallery_start_ignore
for label, angles, signal in (
    ("prescribed", PRESCRIBED, prescribed_signal),
    ("designed", one.parameters["control"], designed_signal),
):
    at_centre = signal[:, :, CENTRE_ECHO - 1]
    print(
        f"{label:<11}"
        f" min {float(angles[0, 0]):>5.1f}  centre {float(angles[0, 1]):>5.1f}"
        f"  max {float(angles[0, 2]):>5.1f} deg"
        f" | blur {float(blur(signal, ALWAYS).mean()):>4.2f} px"
        f" | fluid - cartilage "
        f"{float(at_centre[0, FLUID] - at_centre[0, CARTILAGE]):>5.3f}"
    )

figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.7))
echo_index = np.arange(1, ECHOES + 1)
pixel = np.arange(ECHOES) - ECHOES // 2

axes[0].plot(
    echo_index, prescribed_flip[0].numpy(force=True), "k--", label="prescribed"
)
axes[0].plot(echo_index, designed_flip[0].numpy(force=True), label="designed")
axes[0].set(xlabel="Echo #", ylabel="Refocusing angle [deg]", title="the train")
axes[0].grid(alpha=0.3)

# Dashed is prescribed and solid designed, as in the panel beside it, so the
# colour is free to carry the tissue and the legend stays two entries long.
for tissue, name, colour in (
    (CARTILAGE, "cartilage", "tab:blue"),
    (FLUID, "synovial fluid", "tab:orange"),
):
    axes[1].plot(
        echo_index,
        prescribed_signal[0, tissue].numpy(force=True),
        "--",
        color=colour,
        alpha=0.6,
    )
    axes[1].plot(
        echo_index,
        designed_signal[0, tissue].numpy(force=True),
        "-",
        color=colour,
        label=name,
    )
axes[1].set(xlabel="Echo #", ylabel="|signal|", title="k-space modulation")
axes[1].grid(alpha=0.3)


def spread(modulation):
    """The point spread function itself, for drawing."""
    return torch.fft.fftshift(torch.fft.fft(modulation, dim=-1).abs().square(), dim=-1)


for label, signal in (("prescribed", prescribed_signal), ("designed", designed_signal)):
    psf = spread(signal[0, CARTILAGE])
    axes[2].semilogy(
        pixel,
        (psf / psf.max()).numpy(force=True),
        "k--" if label == "prescribed" else "-",
        label=f"{label}, {float(blur(signal, ALWAYS)[0, CARTILAGE]):.2f} px",
    )
axes[2].set(
    xlabel="Pixel",
    ylabel="PSF (normalized)",
    title="point spread",
    ylim=(1e-5, 2.0),
)
axes[2].grid(alpha=0.3)
key(axes, ncols=1)
# An image keeps its own aspect and a line plot fills whatever it is given, so
# the box each panel is drawn into has to be fixed for the row to line up.
axes[2].set_box_aspect(1)
# sphinx_gallery_end_ignore

# %%
#
# Optimized schedule
# ------------------
#
# The echo index runs along one k-space direction, so forming the image is a
# multiplication: transform each tissue's contribution along the phase-encode
# axis, weight every line by the train at the echo that sampled it, and
# transform back. A fast-decaying train weights the edges of k-space down and
# comes back smeared.
#
# The phantom is a cartoon knee: a cartilage band with a two-pixel joint line,
# and four fluid bars five, three, two and one pixels thick. No noise is added,
# so what differs between the two images is the train alone.
#

# sphinx_gallery_start_ignore
SIZE = ECHOES

rows, columns = torch.meshgrid(
    torch.arange(SIZE, dtype=torch.float32),
    torch.arange(SIZE, dtype=torch.float32),
    indexing="ij",
)
middle = SIZE / 2 - 0.5
inside = ((rows - middle) / 54.0) ** 2 + ((columns - middle) / 44.0) ** 2 < 1.0


def band(low, high):
    """The rows of the phantom between two heights."""
    return inside & (rows >= low) & (rows < high)


fluid_map = band(39, 41)
for top, thickness in ((70, 5), (81, 3), (90, 2), (97, 1)):
    fluid_map = fluid_map | band(top, top + thickness)
cartilage_map = band(26, 54) & ~fluid_map
muscle_map = inside & ~cartilage_map & ~fluid_map

#: One mask per tissue, in the order the simulator returns them.
PHANTOM = torch.stack([cartilage_map, muscle_map, fluid_map]).to(torch.complex64)


def imaged(signal):
    """The image a train produces, one tissue's modulation at a time.

    Parameters
    ----------
    signal:
        ``(1, tissues, echoes)`` echo train magnitudes.

    Returns
    -------
    torch.Tensor
        ``(SIZE, SIZE)``, phase encoding down the rows.
    """
    spectrum = torch.zeros(SIZE, SIZE, dtype=torch.complex64)
    for tissue, component in enumerate(PHANTOM):
        # The echo that samples the centre of k-space belongs at ky = 0.
        along_ky = torch.roll(signal[0, tissue].to(torch.complex64), -(CENTRE_ECHO - 1))
        spectrum = spectrum + torch.fft.fft(component, dim=0) * along_ky[:, None]
    return torch.fft.ifft(spectrum, dim=0).abs()


prescribed_image = imaged(prescribed_signal)
designed_image = imaged(designed_signal)
scale = float(prescribed_image.max())
# sphinx_gallery_end_ignore

# %%
#
# The joint line and the thin bars are where a point spread of a pixel or two
# shows. The ringing at every edge is the finite matrix rather than the train;
# what the design moves is the depth of the troughs between the bars.
#

# sphinx_gallery_start_ignore
VIEW = (slice(6, 114), slice(12, 108))

figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.4))
for axis, picture, title in (
    (axes[0], prescribed_image, "prescribed"),
    (axes[1], designed_image, "designed"),
):
    axis.imshow(
        (picture[VIEW] / scale).numpy(force=True),
        cmap="gray",
        vmin=0.0,
        vmax=0.85,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_box_aspect(1)

profile = slice(64, 104)
column = SIZE // 2
axes[2].plot(
    np.arange(profile.start, profile.stop),
    (prescribed_image[profile, column] / scale).numpy(force=True),
    "k--",
    label="prescribed",
)
axes[2].plot(
    np.arange(profile.start, profile.stop),
    (designed_image[profile, column] / scale).numpy(force=True),
    label="designed",
)
axes[2].set(
    xlabel="Row",
    ylabel="signal (normalized)",
    title="through the fluid bars",
)
axes[2].grid(alpha=0.3)
key(axes[2], ncols=2)
# An image keeps its own aspect and a line plot fills whatever it is given, so
# the box each panel is drawn into has to be fixed for the row to line up.
axes[2].set_box_aspect(1)
# sphinx_gallery_end_ignore

# %%
#
# Sharpness moved without giving up contrast: the fluid-to-cartilage
# difference at the centre of k-space is within a percent of where it started
# while the point spread narrowed by nearly a fifth. The power term is what
# prevents the design buying sharpness by driving the train harder than the
# scanner allows.
#
# The one-pixel bar is the limit. A point spread narrower than a pixel is not
# available, so a bar flattened in both images is flattened by the matrix.
#

# %%
#
# A whole protocol
# ================
#
# Segmented 3D TSE splits k-space over many shots, and they do not do the same
# job: the centre sets contrast, the periphery sets sharpness. Giving each shot
# its own parameters lets a protocol spend a long repetition time where
# contrast comes from and a short one where it does not [2]_.
#
# The prescription is two sets of numbers, at the **centre** and at the
# **periphery**, and a cubic transition builds every shot between them. What
# transitions is the repetition time, the echo train length and the three
# control angles.
#
ESP_SPACE_MS = 3.5
TE_MS = 28.0
TE_ECHO = round(TE_MS / ESP_SPACE_MS)
GRID = 64  # the padded echo axis, at least as long as the longest train

# 320 x 240 phase-encode matrix, CAIPIRINHA 4, elliptical scanning.
LINES = round(320 * 240 / 4 * torch.pi / 4)
BUDGET_S = 300.0

SAMPLES = 16
protocol_shots = FSESimulator(ESP=ESP_SPACE_MS, states=12, **TISSUES)
grid_echo = torch.arange(1, GRID + 1, dtype=torch.float32)

# Each sampled radius stands for the shots at that distance from the centre of
# k-space. Their number grows with radius, because that is the area element of
# the phase-encode plane.
radius = (torch.arange(SAMPLES, dtype=torch.float32) + 0.5) / SAMPLES
density = 2 * radius / (2 * radius).sum()
cubic = (3 * radius.square() - 2 * radius.pow(3))[:, None]


def transition(centre, periphery):
    """Every shot's value, cubically between the two prescribed ends."""
    return centre + (periphery - centre) * cubic


# %%
#
# Shots differ in length: the centre can afford a long train because contrast
# is decided by one echo of it, while the periphery wants a short one so its
# lines are not spread by T2 decay. A train that has ended is masked out of the
# padded echo axis.
#
# A refocusing angle of exactly zero is a corner, not a point -- what reaches
# the scanner is a magnitude, which has no sign there -- so the mask floors at
# a negligible angle.
#
FLOOR = 1e-6


def protocol(
    centre_control, edge_control, centre_length, edge_length, centre_TR, edge_TR
):
    """Every shot of the exam, from the two prescribed ends."""
    control = transition(centre_control, edge_control)
    length = transition(centre_length, edge_length)
    TR = transition(centre_TR, edge_TR)
    acquired = torch.sigmoid(length - grid_echo).clamp_min(FLOOR)
    flip = shape(grid_echo, control, length, TE_ECHO) * acquired
    return flip, TR, length, acquired


# %%
#
# Covering k-space is what ties the two ends together. A shot covers as many
# lines as its train is long, so the shot count is the lines divided by the
# average train length and the scan time is that many shots at the average
# repetition time. Lengthening the trains at the centre buys the repetition
# time there.
#


def measure(**design):
    """Everything the cost reads, from one batched simulation of all shots."""
    flip, TR, length, acquired = protocol(**design)
    signal = protocol_shots.simulate(flip=flip, TR=TR).abs()
    shots = LINES / (density * acquired.sum(-1)).sum()
    scan_s = shots * (density * TR.squeeze(-1)).sum() * 1e-3
    return signal, flip, TR, length, acquired, shots, scan_s


def deposited(flip, acquired, TR):
    """The exam's RF power, averaged over its shots."""
    return (density * power(flip, acquired, TR)).sum()


#: The power the prescription deposits, which is what the exam may spend.
PRESCRIBED_PROTOCOL = protocol(
    PRESCRIBED,
    PRESCRIBED,
    torch.tensor([[45.0]]),
    torch.tensor([[20.0]]),
    torch.tensor([[1800.0]]),
    torch.tensor([[150.0]]),
)
SPACE_POWER_BUDGET = deposited(
    PRESCRIBED_PROTOCOL[0], PRESCRIBED_PROTOCOL[3], PRESCRIBED_PROTOCOL[1]
)


def image_quality(**design):
    """Sharpness where it is decided, contrast where it is decided."""
    signal, flip, TR, length, acquired, shots, scan_s = measure(**design)
    at_centre = signal[:, :, TE_ECHO - 1]
    contrast = at_centre[:, FLUID] - at_centre[:, CARTILAGE]
    outer, inner = density * radius, density * (1.0 - radius)
    # A train must fit inside its own repetition time, with room for the
    # excitation and the fat saturation ahead of it.
    infeasible = torch.relu(
        length.squeeze(-1) * ESP_SPACE_MS + 60.0 - TR.squeeze(-1)
    ).mean()
    return (
        0.7 * (outer * blur(signal, acquired).mean(-1)).sum() / outer.sum()
        - 12.0 * (inner * contrast).sum() / inner.sum()
        + 20.0 * torch.relu(scan_s - BUDGET_S) / BUDGET_S
        + 10.0 * infeasible / 60.0
        + 20.0 * torch.relu(deposited(flip, acquired, TR) / SPACE_POWER_BUDGET - 1.0)
    )


# %%
#
# Starting from the prescription the abstract reports: a 45 echo train at
# 1800 ms at the centre, a 20 echo train at 150 ms at the periphery.
#
PRESCRIPTION = {
    "centre_control": Bounded(PRESCRIBED, LOWEST, HIGHEST),
    "edge_control": Bounded(PRESCRIBED, LOWEST, HIGHEST),
    "centre_length": Bounded(torch.tensor([[45.0]]), 12.0, 60.0),
    "edge_length": Bounded(torch.tensor([[20.0]]), 12.0, 60.0),
    "centre_TR": Bounded(torch.tensor([[1800.0]]), 150.0, 2600.0),
    "edge_TR": Bounded(torch.tensor([[150.0]]), 150.0, 2600.0),
}

design = SequenceDesign(image_quality, **PRESCRIPTION)

# sphinx_gallery_start_ignore
design.minimize(iterations=1)
start = time.perf_counter()
# sphinx_gallery_end_ignore
many = design.minimize(iterations=40, learning_rate=0.2)

# sphinx_gallery_start_ignore
many_elapsed = time.perf_counter() - start
print(
    f"a whole protocol designed in {many_elapsed:.2f} s, "
    f"{1000 * many_elapsed / 40:.1f} ms per iteration"
)
# sphinx_gallery_end_ignore

# %%
#
# Covering this matrix with the trains the prescription asks for takes a shot
# count that falls out of the arithmetic above; the abstract reports 586.
#

# sphinx_gallery_start_ignore
prescribed = {name: value.initial for name, value in PRESCRIPTION.items()}
print("\n                centre of k-space          periphery")
print("                ETL     TR         ETL     TR       shots    scan")
for label, values in (("prescribed", prescribed), ("designed", many.parameters)):
    _, _, TR, length, _, shots, scan_s = measure(**values)
    print(
        f"{label:<12}"
        f" {float(length[0, 0]):>5.1f}  {float(TR[0, 0]):>5.0f} ms"
        f"  {float(length[-1, 0]):>7.1f}  {float(TR[-1, 0]):>4.0f} ms"
        f"  {float(shots):>7.0f}  {float(scan_s) / 60:>5.2f} min"
    )

print("\nrefocusing angles      min  centre-of-band    max")
for label, values in (
    ("prescribed", {"centre_control": PRESCRIBED, "edge_control": PRESCRIBED}),
    ("designed", many.parameters),
):
    for where, name in (("centre_control", "centre"), ("edge_control", "periphery")):
        angles = values[where][0]
        print(
            f"  {label + ', ' + name:<22}"
            + "".join(f"{float(value):>7.1f}" for value in angles)
        )

signal, flip, TR, length, acquired, shots, scan_s = measure(**many.parameters)
start_signal, start_flip, start_TR, start_length, start_acquired, _, _ = measure(
    **prescribed
)
at_centre = signal[:, :, TE_ECHO - 1]
start_at_centre = start_signal[:, :, TE_ECHO - 1]
width = blur(signal, acquired).mean(-1)
start_width = blur(start_signal, start_acquired).mean(-1)

print(
    f"\nfluid - cartilage at the centre "
    f"{float(start_at_centre[0, FLUID] - start_at_centre[0, CARTILAGE]):.3f}"
    f" -> {float(at_centre[0, FLUID] - at_centre[0, CARTILAGE]):.3f}"
)
print(
    f"blur at the periphery          {float(start_width[-1]):.2f}"
    f" -> {float(width[-1]):.2f} px"
)
print(
    f"RF power                       "
    f"{float(deposited(start_flip, start_acquired, start_TR)):.2f}"
    f" -> {float(deposited(flip, acquired, TR)):.2f}"
    f"  (budget {float(SPACE_POWER_BUDGET):.2f})"
)
# sphinx_gallery_end_ignore

# %%
#
# The trains the transition produced. Each curve is one sampled distance from
# the centre of k-space; shots between them read the same curve at their own
# radius, which keeps k-space free of the discontinuities a shot-by-shot design
# would leave.
#

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 6.7))
colours = plt.cm.viridis(np.linspace(0.0, 0.85, SAMPLES))
grid_index = np.arange(1, GRID + 1)
shown = (0, 5, 10, 15)

for shot in shown:
    live = acquired[shot] > 0.5
    axes[0, 0].plot(
        grid_index[live.numpy(force=True)],
        flip[shot][live].numpy(force=True),
        color=colours[shot],
        label=f"r = {float(radius[shot]):.2f}",
    )
axes[0, 0].set(
    xlabel="Echo #", ylabel="Refocusing angle [deg]", title="designed trains"
)
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(
    radius.numpy(force=True),
    start_length[:, 0].numpy(force=True),
    "k--",
    label="prescribed",
)
axes[0, 1].plot(
    radius.numpy(force=True), length[:, 0].numpy(force=True), label="designed"
)
axes[0, 1].set(
    xlabel="k-space radius", ylabel="Echo train length", title="train length"
)
axes[0, 1].grid(alpha=0.3)

axes[0, 2].plot(
    radius.numpy(force=True),
    start_TR[:, 0].numpy(force=True),
    "k--",
    label="prescribed",
)
axes[0, 2].plot(radius.numpy(force=True), TR[:, 0].numpy(force=True), label="designed")
axes[0, 2].set(
    xlabel="k-space radius",
    ylabel="TR [ms]",
    title=f"scan {float(scan_s) / 60:.1f}/{BUDGET_S / 60:.0f} min",
)
axes[0, 2].grid(alpha=0.3)

for axis, tissue, name in (
    (axes[1, 0], CARTILAGE, "cartilage"),
    (axes[1, 1], FLUID, "synovial fluid"),
):
    for shot in shown:
        live = acquired[shot] > 0.5
        axis.plot(
            grid_index[live.numpy(force=True)],
            signal[shot, tissue][live].numpy(force=True),
            color=colours[shot],
        )
    axis.axvline(TE_ECHO, color="k", ls=":", lw=1)
    axis.set(xlabel="Echo #", ylabel="|signal|", title=name)
    axis.grid(alpha=0.3)

start_contrast = start_at_centre[:, FLUID] - start_at_centre[:, CARTILAGE]
axes[1, 2].plot(
    radius.numpy(force=True),
    start_contrast.numpy(force=True),
    "k--",
    label="prescribed",
)
axes[1, 2].plot(
    radius.numpy(force=True),
    (at_centre[:, FLUID] - at_centre[:, CARTILAGE]).numpy(force=True),
    label="designed",
)
axes[1, 2].set(xlabel="k-space radius", ylabel="fluid - cartilage", title="contrast")
axes[1, 2].grid(alpha=0.3)
# A legend wider than its panel makes the layout give the panel up for it, so
# each is columned to stay inside: four short entries in two, two long in one.
key([axes[0, 0]], ncols=2)
key([axes[0, 1], axes[0, 2], axes[1, 2]], ncols=1)
# sphinx_gallery_end_ignore

# %%
#
# References
# ----------
#
# .. [1] Busse, R. F., Brau, A. C. S., Vu, A., et al., "Effects of
#    refocusing flip angle modulation and view ordering in 3D fast spin
#    echo", Magnetic Resonance in Medicine 60.3 (2008), pp. 640-649.
#    https://doi.org/10.1002/mrm.21680
#
# .. [2] Buonincontri, G., Paul, D., Liu, W., Forman, C., Kluge, T.,
#    "Doubling the repetition time without paying the price: 3D turbo spin
#    echo with individually parameterized echo trains", Proceedings of the
#    International Society for Magnetic Resonance in Medicine (2025),
#    abstract 566-05-007.
