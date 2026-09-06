---
name: add-a-sequence
description: Write a new simulator, signal model, or sequence operator in TorchSim. Use when asked to add a sequence, a readout module, a physical effect, or a tissue property.
---

# Add a sequence to TorchSim

**There is one base class, `Simulator`, written in one of two ways.** Nothing
downstream ever asks which way.

## A signal that has to be played

Implement `layout()`: a train of pulses whose magnetization state carries from
one event to the next, which is almost every quantitative sequence.

1. Set `model` to a `SpinPhysics` — what a voxel holds, so which tissue
   properties are exposed and which terms the kernels carry, and what each kind
   of event does to it.
2. Implement `layout()`, returning the operators of one repetition in the order
   they are played.

The extended phase-graph engine, the derivative, the device placement and the
memory policy all follow, and none of them is yours to write. The two pieces
are separate so either can change without the other: giving an MRF timing a
selective excitation, or a refocused train a readout that spoils rather than
winds, is an assignment rather than a new model.

`EventOperators` names the operator each kind of event plays. Four are supplied
and a `SpinPhysics` picks one: `SPOILED` and `UNBALANCED` for gradient-spoiled
and gradient-echo trains, `BALANCED` for a fully refocused steady state,
`REFOCUSED` for a spin-echo train.

## A signal with a closed form

Implement `evaluate()` instead — a mono-exponential decay, an
inversion-recovery curve, an Ernst steady state. There is nothing to play and
no state to carry, so there is no `layout` and the `SpinPhysics` carries only
the property declaration. `SPGRSimulator` is written this way; `MP2RAGESimulator`
carries both, the closed form a lookup table is built from and the layout a
description arriving from a scanner is compared against.

## An operator of your own

An operator is a Python function that returns events. `Excitation`,
`Refocusing`, `Readout`, `Dephase`, `Spoil`, `Delay` and the readout modules
built from them speak exactly the vocabulary the kernels implement, so a module
you write — a T2 preparation, an MT saturation, a shaped pulse — reaches the
fused kernels with no change to them.

An event is a wait, an RF pulse, or an ADC. **Gradients are not events**: each
event declares its own role (`CRUSH_BEFORE`, `CRUSH_AFTER`, `SHIFT_AFTER`,
`SPOIL_AFTER`), because a Pulseq file has no gradient *use* field and a crusher
cannot be told from a phase encode by looking at waveforms. What an operator
cannot say is how much a gradient dephases: a description carries one crusher
moment for the whole sequence, and dephasing is quantized to whole
configuration orders.

## Adding a physical effect

A tissue property has an **identity** — the value at which it has no effect —
and a run reads which properties were passed to decide which terms the kernel
evaluates. Adding an effect is one line in a model's property declaration, plus
the term itself in **both** kernels.

The shared parameter ABI is `src/torchsim/sequence/_parameters.py`, read by the
Python dispatch, the C++ extension and the Triton kernels. A parameter added
there is added in all three or in none.

## What the change has to arrive with

A test that pins the new physics against something **outside** TorchSim: a
closed form, a published figure, or an isochromat summation written out in the
test itself. Comparing TorchSim to TorchSim proves the two agree, which was
never in doubt. Put it in `tests/epg/` and state the invariant in the module
docstring.

Units are public at the edges and internal underneath: a caller writes
milliseconds, degrees and Hz; a description timestamps in microseconds and
carries radians. Convert at the boundary and name the unit in the identifier.
