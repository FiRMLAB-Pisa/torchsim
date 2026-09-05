:orphan:

.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Framework
---------

How to use TorchSim, and how to extend it.

The first notebook covers basic usage: simulating a signal, taking its
derivatives, tuning the run, and reading a sequence back from the description
a scanner streams. The second covers the physics a simulator can carry beyond
T1 and T2.

The last two are for sequences TorchSim does not ship. A **signal model** of
your own is two pieces -- a physics saying what each kind of event does, and a
simulator saying what order they are played in. An **operator** of your own --
a preparation, a readout -- is a Python function that returns events.

None of it requires touching a kernel.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to showcase the basic functionalities of Torchsim, including how to simulate a signal, calculating derivatives etc.">

.. only:: html

  .. image:: /generated/autoexamples/01-framework/images/thumb/sphx_glr_01-getting-started_thumb.png
    :alt:

  :doc:`/generated/autoexamples/01-framework/01-getting-started`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Basic Usage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to show the physics a simulator can carry beyond T1 and T2: the transmit field and the array that produces it, off resonance, an imperfect inversion, a second exchanging pool, a bound pool, diffusion and flow, and the shaped pulse a scanner actually plays.">

.. only:: html

  .. image:: /generated/autoexamples/01-framework/images/thumb/sphx_glr_02-expanded-physics_thumb.png
    :alt:

  :doc:`/generated/autoexamples/01-framework/02-expanded-physics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Expanded Physics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to write a simulator TorchSim does not ship.">

.. only:: html

  .. image:: /generated/autoexamples/01-framework/images/thumb/sphx_glr_03-writing-a-simulator_thumb.png
    :alt:

  :doc:`/generated/autoexamples/01-framework/03-writing-a-simulator`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Writing a Signal Model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to show how to add a sequence module TorchSim does not ship -- a preparation, or a readout -- without touching a kernel.">

.. only:: html

  .. image:: /generated/autoexamples/01-framework/images/thumb/sphx_glr_04-custom-operator_thumb.png
    :alt:

  :doc:`/generated/autoexamples/01-framework/04-custom-operator`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Writing a New Operator</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Parameter inference
-------------------

Estimating tissue properties from a measured volume.

Every estimator is made from the simulator it inverts and fitted over the
same statement of the problem -- what is unknown, over what range, at what
noise level -- and they differ only in how they fill it in. These examples change only that: dictionary matching over a parameter
grid, compressed and clustered; interpolation along a curve where there is a
single unknown; a nonlinear fit of the model itself; and a kernel regression
that never builds a grid at all.

Every one of them maps the same BrainWeb slice, so the answer is known
everywhere -- mixtures of tissues included -- and reports what it cost in time
and in memory alongside what it got wrong.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to map a brain slice by exhaustive dictionary matching, and to show the two ways of making that affordable: working in the low-rank basis the train spans, and clustering the dictionary so that most atoms are never scored.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_01-dictionary-matching_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/01-dictionary-matching`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dictionary matching</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to map T1 from a two-block MP2RAGE, by interpolating along a curve and by matching the same curve, and to sweep the number of points to show which of the two is limited by it.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_02-lookup-table_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/02-lookup-table`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MP2RAGE lookup table</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to map T2 from a multi-echo spin echo by nonlinear least squares, and to compare it against a dictionary match on the same slice.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_03-nonlinear-least-squares_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">T2 mapping by nonlinear least squares</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to map a brain slice with PERK, to show what the size of the regression buys, and to read the error bar it reports.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_04-perk_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/04-perk`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PERK: kernel ridge regression</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Sequence optimization
---------------------

Choosing a sequence's parameters rather than simulating the ones you were
given.

A design problem is three pieces: a simulator with the tissue it is designed
for already fixed on it; a cost, which is a plain function of what that
simulator records; and a :class:`~torchsim.SequenceDesign`, which holds the
parameters inside the limits the scanner will play and runs the loop.

Only the cost changes between the two examples here. One asks for a picture --
sharp where sharpness is decided, with contrast where contrast is decided --
and the other asks for precision, choosing flip angles so that T1 and T2 are
estimated as tightly as the scan time allows.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to design refocusing flip angles for image quality rather than for precision: first a single echo train, then a whole segmented 3D protocol in which each shot carries its own repetition time, echo train length and angles.">

.. only:: html

  .. image:: /generated/autoexamples/03-sequence-optimization/images/thumb/sphx_glr_01-echo-train-design_thumb.png
    :alt:

  :doc:`/generated/autoexamples/03-sequence-optimization/01-echo-train-design`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Designing echo trains for image quality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to choose the flip angles of a DESPOT protocol so that a joint fit of T1 and T2 is as precise as possible [1]_.">

.. only:: html

  .. image:: /generated/autoexamples/03-sequence-optimization/images/thumb/sphx_glr_02-joint-relaxometry_thumb.png
    :alt:

  :doc:`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Designing a joint relaxometry protocol</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Model-based imaging
-------------------

Reconstructing parameter maps from k-space without forming the contrast images
in between.

A quantitative scan is usually reconstructed twice: once to make one image per
contrast, and again -- voxel by voxel -- to turn those images into maps. The
first step has no idea what the second one is for. Writing the signal model
into the forward operator removes the intermediate step, and the echoes then
constrain one another instead of being recovered separately.

There are two ways to do it. A **linear subspace** writes the signal in the
low-rank basis the train spans and reconstructs the coefficients, which stays
linear and so has no local minima and no starting guess. **Nonlinear inversion**
keeps the model itself inside the operator and solves for the maps directly,
which costs more and is what a signal with no small basis needs.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to reconstruct one undersampled radial multi-echo spin echo three ways -- gridding, conjugate gradients per echo, and a linear subspace -- and to report what each costs and gets wrong.">

.. only:: html

  .. image:: /generated/autoexamples/04-model-based-imaging/images/thumb/sphx_glr_01-linear-subspace_thumb.png
    :alt:

  :doc:`/generated/autoexamples/04-model-based-imaging/01-linear-subspace`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reconstructing in a linear subspace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to reconstruct T2 maps straight from k-space, with the signal model inside the forward operator, and to say where the time and the memory go.">

.. only:: html

  .. image:: /generated/autoexamples/04-model-based-imaging/images/thumb/sphx_glr_02-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear inversion from k-space</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Miscellaneous
-------------

Everything that is a pipeline rather than a demonstration of a call.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The scope of this notebook is to build a training pair end to end: what a scanner would measure from a fingerprinting exam, and the maps it came from.">

.. only:: html

  .. image:: /generated/autoexamples/05-misc/images/thumb/sphx_glr_01-synthetic-data_thumb.png
    :alt:

  :doc:`/generated/autoexamples/05-misc/01-synthetic-data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synthetic MR fingerprinting</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /generated/autoexamples/01-framework/index.rst
   /generated/autoexamples/02-parameter-inference/index.rst
   /generated/autoexamples/03-sequence-optimization/index.rst
   /generated/autoexamples/04-model-based-imaging/index.rst
   /generated/autoexamples/05-misc/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: autoexamples_python.zip </generated/autoexamples/autoexamples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: autoexamples_jupyter.zip </generated/autoexamples/autoexamples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
