# Simulators

The sequences that ship with TorchSim. A constructor takes the keywords
{meth}`~torchsim.model.Simulator.simulate` takes and fixes them, so a
sequence and the tissue it is being asked about are written down together and
what is left to give at the call is whatever is actually varying.

## Closed form

Steady states that have an analytic expression, evaluated in one pass.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.simulators.bSSFPSimulator
   torchsim.simulators.SPGRSimulator
   torchsim.simulators.MP2RAGESimulator
   torchsim.simulators.InversionRecoverySimulator
   torchsim.simulators.MultiEchoSimulator
   torchsim.simulators.DoubleAngleSimulator
```

## State machine

Trains that have to be played out, run on the extended phase graph engine.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.simulators.FSESimulator
   torchsim.simulators.MPRAGESimulator
   torchsim.simulators.MPnRAGESimulator
   torchsim.simulators.MRFSimulator
```

## Functional wrappers

One call, protocol and tissue together, for a signal and optionally its
derivative. What the classes above do, without holding one.

### Analytical

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.bssfp_sim
   torchsim.spgr_sim
```

### Iterative

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.fse_sim
   torchsim.mprage_sim
   torchsim.mp2rage_sim
   torchsim.mpnrage_sim
   torchsim.mrf_sim
```
