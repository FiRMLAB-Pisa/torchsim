# Running a simulation

```{eval-rst}
.. currentmodule:: torchsim
```

The differentiable state machine a sequence description is run on, what the
transmit chain does to the pulses it asked for, and where the per-voxel work
is placed.

Simulating a volume and mapping one are both per-voxel, and both outgrow a card
long before they outgrow a host. What to do about that is one policy -- stay on
the host when a launch would not repay itself, cross whole when the problem
fits, stream through in chunks when it does not, and spread across as many
cards as there are -- and it is stated once, around the call.

## The engine

The engine runs a {class}`SequenceDescription` directly, which is what
you reach for when a builder or an interpreter handed you a description rather
than a simulator. A {class}`~torchsim.model.Simulator` wraps it and is how a
sequence is usually written; see {doc}`model`.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

```

## Transmit calibration

What the transmit chain does to the pulse the description asked for: the
amplitude a flip angle needs, and the profile a finite-bandwidth pulse leaves
across the slice.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

```

## Where the work runs

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   execution
   offload
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   utils.b1rms
```
