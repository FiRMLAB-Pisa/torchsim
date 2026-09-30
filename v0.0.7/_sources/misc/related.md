# Related Projects

MR simulators are not one kind of thing, and the first question about any of
them is what it is *for*. Two families answer different questions, and a
package is usually only comparable to the others in its own.

**Image formation.** Given a pulse sequence with its real waveforms and a
phantom with positions, what raw data comes off the scanner? These integrate
the Bloch equation over isochromats that have coordinates, so gradients,
off-resonance maps, coil sensitivities, motion and the readout itself are all
in the model. JEMRIS [^1], MRiLab [^2], KomaMRI.jl [^3], CMRsim [^4] and
MRzero-Core [^5] answer this question.

**Signal evolution.** Given a train of pulses and a tissue, what does one
voxel record? No coordinates, no k-space, no image: the sequence is a stream
of events and the answer is a curve per tissue. That is what a dictionary, a
fit, a Cramer-Rao bound and a model-based reconstruction are built from, and
it is what an extended phase graph computes in a fraction of the states an
isochromat ensemble needs. sycomore [^6], epgpy [^7], EpyG [^8],
mri-sim-py [^9], snapMRF [^10] and TorchSim answer this one.
BlochSimulators.jl [^11] answers both, from the same sequence description.

**TorchSim is the second kind.** It has no gradient waveforms, no phantom
coordinates and no encoding operator of its own -- {doc}`../explanations/implementation`
lists what that rules out. What it adds instead is the derivative: the same
kernels that produce a signal produce its Jacobian with respect to tissue, and
its gradient with respect to the sequence, which is what the estimators, the
model-based reconstruction and the sequence design on these pages consume.

## The landscape

Read as of August 2026, from each project's own documentation. A blank is not
a criticism: a scanner simulator has no reason to carry a dictionary Jacobian,
and a signal-model simulator has no reason to carry a coil.

| Project | Model | Implementation | Hardware | Derivatives | Sequence input |
| --- | --- | --- | --- | --- | --- |
| [JEMRIS](https://www.jemris.org) | Isochromat, ODE solver (CVODE) | C++, MPI | CPU clusters; a GPU port published in 2025 [^12] | -- | XML, GUI |
| [MRiLab](https://leoliuf.github.io/MRiLab/) | Discrete spin, multi-pool exchange tissue model | MATLAB front end, C++/CUDA kernels | CPU threads, CUDA | -- | GUI |
| [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) | Isochromat, operator splitting and Magnus expansions | Julia | CPU threads, CUDA, AMDGPU, Metal, oneAPI | -- | Pulseq `.seq`, Julia, GUI |
| [CMRsim](https://gitlab.ethz.ch/ibt-cmr/mri_simulation/cmrsim) | Isochromat, and analytic signal models | Python, TensorFlow 2 | CPU, GPU | -- | Python |
| [MRzero-Core](https://github.com/MRsources/MRzero-Core) | Phase distribution graphs, and Bloch | Python, PyTorch, Rust core | CPU, CUDA | Automatic, reverse mode | Pulseq `.seq`, PyPulseq |
| [BlochSimulators.jl](https://github.com/MagneticResonanceImaging/BlochSimulators.jl) | Isochromat *and* EPG | Julia | CPU threads, distributed, CUDA | Finite differences, in MR-STAT [^11] | Julia |
| [sycomore](https://github.com/lamyj/sycomore) | Bloch, and EPG: regular, discrete, discrete 3D | C++ core, Python bindings | One CPU core | -- | Python |
| [epgpy](https://github.com/py-baudin/epgpy) | EPG, with 3D gradients and multi-compartment exchange | Python, NumPy or CuPy | CPU, CUDA through CuPy | Analytic, first and second order | Python |
| [EpyG](https://github.com/brennerd11/EpyG) | EPG | Python | CPU | -- | Python |
| [mri-sim-py](https://github.com/utcsilab/mri-sim-py.epg) | EPG | Python, PyTorch | CPU, CUDA | Automatic, reverse mode | Python |
| [snapMRF](https://github.com/dongwang881107/snapMRF) | EPG, with matching | CUDA C | CUDA | -- | Command line |
| **TorchSim** | EPG, and closed forms | Python, PyTorch, C++ and Triton kernels | CPU threads, CUDA, several cards | Automatic: forward, reverse, and forward over reverse | Python, or a description |

## Where each one is the better tool

**A sequence you are developing, and the image it makes.** KomaMRI, JEMRIS,
CMRsim or MRzero-Core. They read the waveforms you will play, carry the spins
through them at their coordinates, and hand back k-space. TorchSim reads a
sequence as events rather than waveforms, and stops at the signal.

**One curve, read interactively, with the model in front of you.** sycomore or
epgpy. Both are a few milliseconds for one tissue with nothing to warm up, and
sycomore's units make an expression read like the paper it came from. TorchSim
resolves the structure of a sequence before it runs one, which is seconds it
does not repay until there is a dictionary to sweep or a loop to run.

**A dictionary, a fit, a design loop, or a map solved from k-space.**
TorchSim. The batching across tissues, the derivative, and the estimators and
reconstruction that consume it are the point of the package; `benchmarks/` in
the repository measures the first two against the alternatives above and
states the agreement between them.

**A steady state that has a closed form.** Its expression, which is faster
than any of these -- and several ship here as
{doc}`closed-form simulators <../api/simulators>`.

## References

[^1]: Stöcker, T., Vahedipour, K., Pflugfelder, D., Shah, N. J., "High-
    performance computing MRI simulations", Magnetic Resonance in Medicine
    (2010). https://doi.org/10.1002/mrm.22406

[^2]: Liu, F., Velikina, J. V., Block, W. F., Kijowski, R., Samsonov, A. A.,
    "Fast realistic MRI simulations based on generalized multi-pool exchange
    tissue model", IEEE Transactions on Medical Imaging (2017).
    https://doi.org/10.1109/TMI.2016.2620961

[^3]: Castillo-Passi, C., Coronado, R., Varela-Mattatall, G., Alberola-López,
    C., Botnar, R., Irarrazaval, P., "KomaMRI.jl: An open-source framework for
    general MRI simulations with GPU acceleration", Magnetic Resonance in
    Medicine (2023). https://doi.org/10.1002/mrm.29635

[^4]: Weine, J., McGrath, C., Dirix, P., Stoeck, C. T., Kozerke, S., "CMRsim
    -- A python package for cardiovascular MR simulations incorporating
    complex motion and flow", Magnetic Resonance in Medicine (2024).
    https://doi.org/10.1002/mrm.30010

[^5]: Endres, J., Weinmüller, S., Dang, H. N., Zaiss, M., "Phase distribution
    graphs for fast, differentiable, and spatially encoded Bloch simulations
    of arbitrary MRI sequences", Magnetic Resonance in Medicine 92.3 (2024),
    pp. 1189-1204. https://doi.org/10.1002/mrm.30055

[^6]: Lamy, J., "sycomore: an MRI simulation toolkit".
    https://github.com/lamyj/sycomore

[^7]: Baudin, P., "epgpy: EPG simulations in Python".
    https://github.com/py-baudin/epgpy

[^8]: Brenner, D., "EpyG: extended phase graphs in Python".
    https://github.com/brennerd11/EpyG

[^9]: "mri-sim-py.epg: a GPU-accelerated extended phase graph algorithm for
    differentiable optimization and learning".
    https://github.com/utcsilab/mri-sim-py.epg

[^10]: Wang, D., Ostenson, J., Smith, D. S., "snapMRF: GPU-accelerated
    magnetic resonance fingerprinting dictionary generation and matching using
    extended phase graphs", Magnetic Resonance Imaging 66 (2020), pp. 248-256.
    https://doi.org/10.1016/j.mri.2019.11.015

[^11]: van der Heide, O., Sbrizzi, A., Bruijnen, T., van den Berg, C. A. T.,
    "GPU-accelerated Bloch simulations and MR-STAT reconstructions using the
    Julia programming language", Magnetic Resonance in Medicine (2024).
    https://doi.org/10.1002/mrm.30074

[^12]: "GPU-accelerated JEMRIS for extensive MRI simulations", Magnetic
    Resonance Materials in Physics, Biology and Medicine (2025).
    https://doi.org/10.1007/s10334-025-01281-z
