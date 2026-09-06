# Signal models

```{eval-rst}
.. currentmodule:: torchsim.model
```

**There is one base class, and it is written in one of two ways.**

{class}`Simulator` is the interface, and the only thing anything downstream
ever sees: the estimators, the model-based operator and the sequence optimizer
all take one and never ask how it arrives at its signal.

Implement {meth}`~Simulator.layout` when the signal has to be *played* -- a
train of pulses whose magnetization state carries from one event to the next,
which is almost every quantitative sequence. Two things are said. Which
operator plays each kind of event, by naming it in the class body, and what
order they come in. The extended phase-graph engine, the derivative, the
device placement and the memory policy all follow from that and none of them
is yours to write.

```python
class SSFPMRF(Simulator):
    excitation = Excitation
    inversion = Inversion
    readout = SSFPFidReadout
    states = 10

    def layout(self, *, flip, TR, TI=0.0):
        parts = [self.operators.inversion(duration_s=TI * 1e-3)]
        for angle in torch.deg2rad(torch.as_tensor(flip)):
            parts += [
                self.operators.excitation(angle),
                self.operators.readout(duration_s=TR * 1e-3),
            ]
        return parts
```

The six slots a class body may name are `excitation`, `refocusing`,
`inversion`, `saturation`, `readout` and `delay`; each is one of the operators
on {doc}`sequence`. Naming a different readout is the whole of the difference
between a spoiled, an unbalanced, a balanced and a refocused train, so a
variant is a subclass with one line in it. Naming one is also what says how a
stream arriving from a scanner is to be read, since
{meth}`~Simulator.from_description` re-emits its events through these same
operators.

Nothing is declared about the tissue. Every property a voxel can have may be
given to any simulator, and giving one is what turns its term on.

A sequence that came from somewhere else is read the same way:
{meth}`~Simulator.from_description` takes the stream an MRD client decodes, and
{meth}`~Simulator.from_pulseq` takes a Pulseq `.seq` file directly. Neither
walks a layout -- naming the simulator is what says how the events are played.

Implement {meth}`~Simulator.evaluate` instead when the signal has a closed
form -- a mono-exponential decay, an inversion-recovery curve, an Ernst
steady state. There is nothing to play and no state to carry, so there is no
`layout` and the `SpinPhysics` carries only the property declaration.
{class}`~torchsim.simulators.SPGRSimulator` is written this way, and
{class}`~torchsim.simulators.MP2RAGESimulator` carries both: the closed form a
lookup table is built from, and the layout a description arriving from a
scanner is compared against.

A model that composes others rather than declaring physics of its own names
its properties in the class body and writes whatever constructor suits it:

```python
class JointRelaxometry(Simulator):
    properties = ("T1", "T2", "M0")

    def __init__(self, spgr_flip, ssfp_flip):
        self.spoiled = SPGRSimulator(TE=2.0, TR=6.0, flip=spgr_flip)
        self.balanced = bSSFPSimulator(TE=2.5, TR=5.0, flip=ssfp_flip)

    def evaluate(self, properties, **sequence):
        ...
```

Either way it fixes its arguments the same way: a constructor takes the
keywords {meth}`~Simulator.simulate` takes, {meth}`~Simulator.bind` adds more
to a copy, and a call overrides either.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Simulator
```
