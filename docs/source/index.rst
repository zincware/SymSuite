.. SymDet documentation master file, created by
   sphinx-quickstart on Thu May  6 11:22:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SymDet's documentation!
==================================
SymDet is a python package developed in conjunction with PhD research into extraction of fundamental physical laws from
raw data. Specifically, SymDet provides the functionality for performing the analysis described in the paper by Sven
Krippendorf and Marc Syvaeri on
`Detecting symmetries with neural networks <https://iopscience.iop.org/article/10.1088/2632-2153/abbd2d>`_.

The main idea behind this method is twofold.

1. The embedding layer of a neural network holds within it an encoded representation from which symmetries in data
can be studied.

2. By formulating the problem as regression, one can extract the generators of the Lie algebra from the point clouds
generated in this representation.

At the moment the code can perform the following tasks.

1. Use a dense neural network to construct a tSNE representation to visually identify symmetry groups.
2. Fit generators of symmetry groups using given point cloud data. I can only confirm the accuracy of this fitting
   for two dimensional data but am working to extend this to arbitrary systems.
3. Identify groups connected by symmetry in the tSNE representation and collect them for generator extraction.

More work will be done to bridge these two processes in the hope that a pipeline from raw data to generators can be
formed.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   theory
   notebooks
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
