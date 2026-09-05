# User Guide

Everything you need to go from an empty environment to a simulated echo train,
and a map of where the rest of this documentation is.

```{contents}
:local:
:depth: 2
```

## Install

TorchSim is a PyTorch package with two compiled kernels behind it. `pip`
installs the Python side and its NumPy and SciPy dependencies for you. Two
things it cannot decide on your behalf, and which you therefore settle first:
**which build of PyTorch you want**, and **where you want the whole thing to
live**.

### An environment of its own

Install into an isolated environment rather than the system interpreter. A
simulation pins versions of PyTorch, NumPy and SciPy, and the phantom, coil
and reconstruction packages the examples reach for pin their own; keeping them
apart from the rest of the machine is what lets you delete a bad combination
by deleting a directory.

Any of these works. Pick the one you already use.

::::{tab-set}

:::{tab-item} venv
:sync: venv

Ships with Python, nothing to install first:

```sh
python -m venv ~/envs/torchsim
source ~/envs/torchsim/bin/activate     # Windows: ~\envs\torchsim\Scripts\activate
python -m pip install --upgrade pip
```
:::

:::{tab-item} conda
:sync: conda

Useful when you also want a specific Python, or non-Python libraries
beside it:

```sh
conda create -n torchsim python=3.12
conda activate torchsim
```

Install TorchSim itself with `pip` inside that environment; there is no
conda package.
:::

:::{tab-item} uv
:sync: uv

The fastest of the three, and it resolves the whole set at once:

```sh
uv venv --python 3.12 ~/envs/torchsim
source ~/envs/torchsim/bin/activate
```

Read `uv pip install` for `pip install` in everything that follows.
:::

::::

### PyTorch first

**Install PyTorch before TorchSim.** The wheel you want depends on hardware
`pip` cannot see: a CPU-only build and a CUDA build have the same name and
version, and differ only in which index they came from. Install TorchSim first
and you get whichever build the default index happens to serve, which is
usually not the one you meant.

Choose by what you will run on:

::::{tab-set}

:::{tab-item} CPU only
:sync: cpu

A laptop, a login node, or a machine with no NVIDIA card. This is also
the smallest download by a wide margin:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Everything in TorchSim runs here. The state machine has a threaded,
vectorized C++ kernel behind it, so a dictionary of a few thousand atoms
is seconds rather than minutes.
:::

:::{tab-item} NVIDIA GPU
:sync: cuda

Check which CUDA versions the current PyTorch release ships wheels for
on [pytorch.org/get-started](https://pytorch.org/get-started/locally/),
pick the one your driver supports, and install from that index --
`cu128` below stands for whichever tag you chose:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The GPU kernels are written in Triton, which comes with the Linux CUDA
wheels; you do not install it separately and you do not need the CUDA
toolkit, only a driver new enough for the build you picked.
:::

:::{tab-item} Apple silicon
:sync: mac

The default wheel is the right one:

```sh
pip install torch
```

The simulation runs on the CPU kernels. There is no Metal path: the
fused state machine exists as C++ and as Triton, and Triton has no
Apple backend.
:::

::::

Then check that the build is the one you wanted, *before* installing anything
on top of it:

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA build prints a version and `True`; a CPU build prints `None` and
`False`.

### TorchSim

```sh
pip install torchsim
```

Where a wheel exists for your platform this is the whole of it. Where one does
not, `pip` builds the two C++ extensions from source and you need a C++17
compiler on the path -- `build-essential` on Debian and Ubuntu, `gcc-c++`
on Fedora, the Command Line Tools on macOS, the Visual Studio Build Tools on
Windows. There is no pure-Python fallback: the extensions *are* the CPU state
machine, and a run that cannot find one raises rather than quietly running
something slower.

To check the install, simulate a fast spin echo train and read three tissues at
once:

```python
import torch
from torchsim.simulators import FSESimulator

acquisition = FSESimulator(
    ESP=5.0,
    TR=3000.0,
    T1=torch.tensor([830.0, 1330.0, 4000.0]),   # ms
    T2=torch.tensor([80.0, 110.0, 2000.0]),
)
signal = acquisition.simulate(flip=torch.full((48,), 60.0))
print(signal.shape)                              # torch.Size([3, 48])
```

On a card, hand it tissue that already lives there and the whole run follows:

```python
acquisition = acquisition.to("cuda")
signal = acquisition.simulate(flip=torch.full((48,), 60.0, device="cuda"))
```

The first GPU call pays for compiling the Triton kernel it needs -- tens of
seconds, once per kernel per machine, cached afterwards. A first call that
seems to hang is almost always that compile.

## Your first simulation

A simulator carries a sequence and the tissue it is being asked about; what
you pass at the call is whatever is actually varying. Asking for derivatives
alongside the signal costs one extra pass:

```python
import numpy as np
import torchsim

flip = np.concatenate(
    (np.linspace(5.0, 60.0, 300), np.linspace(60.0, 2.0, 300), np.full(280, 2.0))
)
signal, jacobian = torchsim.mrf_sim(
    flip=flip, TR=10.0, T1=1000.0, T2=100.0, diff=("T1", "T2")
)
```

`signal` is the forward pass; `jacobian` holds its derivative with respect
to T1 and T2. That derivative is what a dictionary fit, a nonlinear
least-squares map, a model-based reconstruction and a sequence design all
start from, which is why it is one keyword rather than a separate call.

Arrays go in and come back in whatever library you wrote them in -- NumPy here,
CuPy or PyTorch elsewhere -- over the same memory rather than a copy.

## Finding your way around this documentation

{doc}`explanations/epg`
: What configuration states are, why a train of pulses generates more echoes
  than it has pulses, and where relaxation, diffusion, flow and a second
  proton pool enter. Read this if EPG is new, or if you want to know what
  the simulator is actually computing.

{doc}`explanations/implementation`
: How that algorithm is realized here: a sequence as a stream of events, one
  fused kernel per voxel, derivatives taken forward or backward depending on
  what you differentiate, and the shortcuts a run takes when your sequence
  allows them.

{doc}`generated/autoexamples/index`
: Worked examples you can run, download or open in Colab. They go from
  calling a simulator that ships with TorchSim, through writing one of your
  own, to parameter inference, sequence design and model-based
  reconstruction.

{doc}`api/index`
: The reference. Start at {doc}`api/simulators` for what ships, at
  {doc}`api/model` for writing your own, and at {doc}`api/execution` for
  placing a run across devices.

{doc}`developer_guide`
: Setting up to change TorchSim: the editable install, the style the code is
  written in, the tests, and how a pull request is opened.

## Getting help, and reporting what breaks

**Ask a question** in
[Discussions](https://github.com/FiRMLAB-Pisa/torchsim/discussions). How to model
a sequence, whether a signal you got is expected, which estimator suits a
problem -- these belong there, and the answer is then findable by whoever asks
next.

**Report a bug** in
[Issues](https://github.com/FiRMLAB-Pisa/torchsim/issues/new/choose), where a
form asks for what a fix needs:

- the shortest script that reproduces it, pasted whole -- a sequence is enough
  numbers that a description of it leaves the run ambiguous;
- what you expected instead, and why. A signal that surprises you is not yet a
  bug: say which analytic case, published figure or alternative simulator you
  are comparing against;
- the full traceback, if it raises;
- the environment, as the form's command prints it -- TorchSim, PyTorch, CUDA,
  Python;
- whether you have seen it on the CPU kernels, the CUDA kernels, or both. That
  difference is often the whole diagnosis.

**Ask for a feature** -- a sequence, a physical effect, an estimator -- through
the same form chooser. Name the paper the model comes from and the figure it
would have to reproduce; that is what makes it implementable.

**Report a vulnerability** privately instead: open a draft advisory from the
repository's [Security tab](https://github.com/FiRMLAB-Pisa/torchsim/security/advisories/new),
or email the address in the [security policy](https://github.com/FiRMLAB-Pisa/torchsim/blob/main/.github/SECURITY.md).
The kernels index raw pointers, so anything reachable from ordinary arguments
that reads or writes out of bounds is worth reporting that way rather than in a
public issue. Wrong physics is a bug report, not a vulnerability.
