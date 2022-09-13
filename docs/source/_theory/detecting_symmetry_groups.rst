Detecting Symmetry Groups
=========================
Now we can get into some of the fun part of this package. How do we use
machine learning to detect symmetry groups in large data-sets. This is a direct
implementation of Sven Krippendorf and Marc Syvaeri's paper on
`Detecting symmetries with neural networks <https://iopscience.iop.org/article/10.1088/2632-2153/abbd2d>`_.
and I refer readers to the original work for a full treatment of the problem
from the authors. Here I will discuss the theory and implementation as best I
can.

In a system related by some symmetry such as a harmonic oscillator potential it does not
take a human long to identify that symmetry exists. If we then train a neural
network to classify points along this potential, it seems intuitive that on some
level the machine learning model should also identify this symmetry. Unfortunately,
the representation of the symmetry will be buried somewhere in a very high-dimensional
space that we cannot simply visualize. That's where the tSNE visualization we
discussed comes in.