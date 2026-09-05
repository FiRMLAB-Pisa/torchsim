# Sequences

```{eval-rst}
.. currentmodule:: torchsim
```

Device-agnostic description of an acquisition, shared by the interpreter, the
estimators and the sequence optimizer. A description is a list of events with
the timing and the tissue interaction each one carries; nothing in it names a
vendor or a device.

This page is the description and what it is assembled from. Writing a
simulator out of these is on {doc}`model`, running one is on {doc}`execution`,
and the temporal basis a run spans is on {doc}`recon`.

## Operators

The modules a sequence is assembled from. An operator is a Python function
returning the events it plays and how long it holds the timeline, and `@`
composes two into one -- so a preparation or a readout TorchSim does not ship
reaches the fused kernels with no change to them.

The five readouts differ only in what they play around the sample, which is
what separates a spoiled, an unbalanced, a balanced and a refocused train.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Operator
   Excitation
   Refocusing
   Inversion
   Saturation
   Readout
   Delay
   Dephase
   Spoil
   bSSFPReadout
   SSFPFidReadout
   SSFPEchoReadout
   SPGRReadout
   FSEReadout
```

## Pulses and shims

An ideal pulse turns the whole voxel through one angle. {func}`rf_definition`
takes a complex envelope instead -- one row per transmit channel -- so a
selective excitation is integrated over the slice rather than scaled, and
{class}`ShimDefinition` gives the amplitude and phase each channel is driven
at. A simulator takes them as `pulse=` and `shims=`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   rf_definition
   ShimDefinition
```

## Description

The event stream itself: what a layout composes to, and what a sequence
arriving from a scanner is read into.
{meth}`SequenceDescription.from_operators` builds one directly,
{meth}`SequenceDescription.from_pulseq` reads one out of a Pulseq `.seq` file,
and {meth}`~torchsim.model.Simulator.from_description` runs either.

Reading a `.seq` file needs pypulseq, which parses the format and computes the
gradient trajectory the echo is found on: `pip install torchsim[pulseq]`.
Nothing else in the package imports it.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SequenceDescription
```
