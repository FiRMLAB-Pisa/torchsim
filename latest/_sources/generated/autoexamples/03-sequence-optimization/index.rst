

.. _sphx_glr_generated_autoexamples_03-sequence-optimization:

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


.. toctree::
   :hidden:

   /generated/autoexamples/03-sequence-optimization/01-echo-train-design
   /generated/autoexamples/03-sequence-optimization/02-joint-relaxometry

