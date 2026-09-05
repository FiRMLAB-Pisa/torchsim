# Signal models

```{eval-rst}
.. currentmodule:: torchsim.model
```

**There are two base classes and you write exactly one of them.**

{class}`SignalModel` is the interface, and the only thing anything downstream
ever sees: the estimators, the model-based operator and the sequence optimizer
all take a `SignalModel` and never ask which kind it is. Whichever of the two
you write, that is what you have made.

Write a {class}`Simulator` when the signal has to be *played* -- a train of
pulses whose magnetization state carries from one event to the next, which is
almost every quantitative sequence. Two things are said. Which operator plays
each kind of event, by naming it in the class body, and what order they come
in, by implementing {meth}`~Simulator.layout`. The extended phase-graph engine,
the derivative, the device placement and the memory policy all follow from that
and none of them is yours to write.

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

Subclass {class}`SignalModel` directly and implement
{meth}`~SignalModel.evaluate` only when the signal has a closed form -- a
mono-exponential decay, an inversion-recovery curve -- so there is nothing to
play and no state to carry.

Either kind fixes its arguments the same way: a constructor takes the keywords
{meth}`~SignalModel.simulate` takes, {meth}`~SignalModel.bind` adds more to a
copy, and a call overrides either.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   SignalModel
   Simulator
```
