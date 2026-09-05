# TorchSim

TorchSim is a pure Pytorch-based MR simulator, including analytical and EPG model.

[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/torchsim/graph/badge.svg?token=l8xhIVORYm)](https://codecov.io/gh/FiRMLAB-Pisa/torchsim)
[![Tests](https://github.com/FiRMLAB-Pisa/torchsim/actions/workflows/test.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/torchsim/actions/workflows/test.yml)
[![Lint](https://github.com/FiRMLAB-Pisa/torchsim/actions/workflows/lint.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/torchsim/actions/workflows/lint.yml)
[![License](https://img.shields.io/github/license/FiRMLAB-Pisa/torchsim)](https://github.com/FiRMLAB-Pisa/torchsim/blob/main/LICENSE.txt)
[![Codefactor](https://www.codefactor.io/repository/github/FiRMLAB-Pisa/torchsim/badge)](https://www.codefactor.io/repository/github/FiRMLAB-Pisa/torchsim)
[![Documentation](https://github.com/FiRMLAB-Pisa/torchsim/actions/workflows/docs.yml/badge.svg)](https://firmlab-pisa.github.io/torchsim/)
[![PyPi](https://img.shields.io/pypi/v/torchsim)](https://pypi.org/project/torchsim)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PythonVersion](https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white)](https://python.org)

## Features

TorchSim contains tools to implement parallelized and differentiable MR simulators. Specifically, we provide

1. Automatic vectorization of across multiple atoms (e.g., voxels).
2. Automatic generation of forward and jacobian methods (based on forward-mode autodiff) to be used in parameter fitting or model-based reconstructions.
3. Support for custom manual defined jacobian methods to override auto-generated jacobian.
4. Support for advanced signal models, including diffusion, flow, magnetization transfer and chemical exchange.
5. GPU support.

## Installation

TorchSim can be installed via pip as:

```bash
pip install torchsim
```

## Basic Usage

Using TorchSim, we can quickly implement and run MR simulations.
We also provide pre-defined simulators for several applications:

```python
import numpy as np
import torchsim

# generate a flip angle pattern
flip = np.concatenate((np.linspace(5, 60.0, 300), np.linspace(60.0, 2.0, 300), np.ones(280)*2.0))
sig, jac = torchsim.mrf_sim(flip=flip, TR=10.0, T1=1000.0, T2=100.0, diff=("T1","T2"))
```

This way we obtained the forward pass signal (`sig`) as well as the jacobian
calculated with respect to `T1` and `T2`.

## Development

If you are interested in improving this project, install TorchSim in editable mode:

```bash
git clone git@github.com:FiRMLAB-Pisa/torchsim
cd torchsim
pip install -e ".[dev]"
pre-commit install
```

The install compiles the two C++ kernels, so it needs a C++17 compiler; CMake
and Ninja arrive as build-time wheels. `pre-commit` runs the formatter and the
linter -- both `ruff` -- on every commit, which is exactly what CI checks.

## Related projects

This package is inspired by the following excellent projects:

- epyg \<<https://github.com/brennerd11/EpyG>\>
- sycomore \<<https://github.com/lamyj/sycomore/>\>
- mri-sim-py \<<https://somnathrakshit.github.io/projects/project-mri-sim-py-epg/>\>
- ssfp \<<https://github.com/mckib2/ssfp>\>
- erwin \<<https://github.com/lamyj/erwin>\>
