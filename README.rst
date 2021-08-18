|build| |docs| |madewithpython| |license|

SymDet
------

A python package to perform symmetry detection and generator extraction on
raw data. Follows the paper by Sven Krippendorf and Marc Syvaeri on
`Detecting symmetries with neural networks <https://iopscience.iop.org/article/10.1088/2632-2153/abbd2d>`_.

Notes
=====
This project is under heavy development and is therefore not available on PyPi.
I would not expect major API breaks but certainly addition of functionality.

Installation
============
There are several options for installing SymDet

PyPi
****

We host the code on PyPi and so it can be simply installed by:

.. code-block:: bash

   pip3 install symdet

Install from source
*******************

**pip installation**

.. code-block:: bash

   git clone https://github.com/zincware/SymDet.git
   cd SymDet
   pip3 install . --user

**conda installation**

.. code-block:: bash

   git clone https://github.com/zincware/SymDet.git
   cd SymDet
   conda create -n SymDet python=3.8
   conda activate SymDet
   pip3 install .

Documentation
*************

There is a live version of the documentation hosted
`here <https://symdet.readthedocs.io/en/latest/>`_. Alternatively you can
build it from source using

.. code-block:: bash

   cd Symdet/docs
   make html

You can then browse the documentation locally using your favourite browser.

Getting started
===============

As a first step I would suggest looking at the
`examples <https://github.com/zincware/SymDet/tree/main/examples>`_
directory and following along with some tutorials.
From here you may get a better idea of what you can use this package for.

Comments
========
This is a really young project and any comments or contributions would be
welcome. If you see issues in the documentation (particularly if you're a
mathematician) I would always welcome the feedback.

.. badges

.. |build| image:: https://github.com/SamTov/SymDet/actions/workflows/python-package.yml/badge.svg
    :alt: Build tests passing
    :target: https://github.com/SamTov/SymDet/blob/readme_badges/

.. |docs| image:: https://readthedocs.org/projects/symdet/badge/?version=latest&style=flat
    :alt: Build tests passing
    :target: https://symdet.readthedocs.io/en/latest/

.. |license| image:: https://img.shields.io/badge/License-EPLv2.0-purple.svg?style=flat
    :alt: Project license
    :target: https://www.gnu.org/licenses/quick-guide-gplv3.en.html

.. |madewithpython| image:: https://img.shields.io/badge/Made%20With-Python-blue.svg
    :alt: Made with python
