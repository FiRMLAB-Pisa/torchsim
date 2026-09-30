

.. _sphx_glr_generated_autoexamples_02-parameter-inference:

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


.. toctree::
   :hidden:

   /generated/autoexamples/02-parameter-inference/01-dictionary-matching
   /generated/autoexamples/02-parameter-inference/02-lookup-table
   /generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares
   /generated/autoexamples/02-parameter-inference/04-perk

