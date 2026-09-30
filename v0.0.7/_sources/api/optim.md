# Sequence design

```{eval-rst}
.. currentmodule:: torchsim
```

Choosing a sequence's parameters by minimizing a cost you write. The
acquisition is a simulator with the tissue it is designed for already fixed on
it; the cost is a plain function of what it records; a
{class}`SequenceDesign` holds the parameters and runs the loop.

{class}`Bounded` holds a parameter inside the limits the scanner will play, by
optimizing a transformed variable -- no iterate is ever outside them.
{func}`crlb` turns the derivative of a signal with respect to tissue into the
lowest variance an unbiased estimate of it can have, which is the cost a
precision-driven design minimizes.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   Bounded
   SequenceDesign
   crlb
```
