

.. _sphx_glr_generated_autoexamples_01-framework:

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


.. toctree::
   :hidden:

   /generated/autoexamples/01-framework/01-getting-started
   /generated/autoexamples/01-framework/02-expanded-physics
   /generated/autoexamples/01-framework/03-writing-a-simulator
   /generated/autoexamples/01-framework/04-custom-operator

