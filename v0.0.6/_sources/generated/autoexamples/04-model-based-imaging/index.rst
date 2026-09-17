

.. _sphx_glr_generated_autoexamples_04-model-based-imaging:

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


.. toctree::
   :hidden:

   /generated/autoexamples/04-model-based-imaging/01-linear-subspace
   /generated/autoexamples/04-model-based-imaging/02-nonlinear-inversion

