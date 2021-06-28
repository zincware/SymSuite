Notebooks
=========

Here are a few examples of applying SymDet to some simple systems and looking at the results.

.. toctree::
   :maxdepth: 1

   examples/double_well_investigation
   examples/SO_example
   examples/Pendulum_Example

Double Well Investigation
^^^^^^^^^^^^^^^^^^^^^^^^^
In this notebook we use the symmetry detection functionality of SymDet to identify groups of points along a double well
potential which are connected by a symmetry.

SO Examples
^^^^^^^^^^^
This notebook explores how the generator extraction functionality of SymDet can be used to compute the generators of a
symmetry group. This is applied to both the SO(2) and SO(3) groups.

Pendulum Example
^^^^^^^^^^^^^^^^
The pendulum example notebook takes what we have learned from the previous two examples and applies it to a real system,
namely, a simple pendulum. In this example, we begin by running a short simulation of an ideal pendulum under gravity.
The SymDet clustering algorithm is then applied to this raw data to identity points along the potential energy surface
of the pendulum that rae connected by symmetry. These groups are then passed into the generator extraction stages of
SymDet and finally, their generators calculated. This acts as a full end-to-end detection and characterization of
symmetry groups using machine learning.