.. pyviability documentation master file, created by
   sphinx-quickstart on Thu Mar  2 13:54:28 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyViability's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
------------

*Disclaimer: The library is still preliminary work. As I am happy about everybody who'd consider using the library, I'll help you as much as I can and might consider writing a proper tutorial and extend the Documentation. Simpy contact me (Tim.Kittel@pik-potsdam.de).*

PyViability is small library for Computations related to Viability Theory, particular the Viability Kernel and the Capture Basin, and for the Classifications of (a) Models(') (state space) with respect to the Topology of Sustainable Management [1].

The library was developped during the work on [2] and is in a very preliminary state. I am happy for any support to improve the library. 

[1] http://www.earth-syst-dynam.net/7/21/2016/esd-7-21-2016.html

[2] https://arxiv.org/abs/1706.04542

Find the source here_.

.. _here: https://github.com/timkittel/PyViability

Setup
-----

The code was tested with **Python 3.5** under **Ubuntu Xenial** only. If you want to get it running on a different system, please contact me (Tim.Kittel@pik-potsdam.de).

To install the library, run

.. code-block:: rest

   git clone https://github.com/timkittel/PyViability.git
   cd PyViability
   pip install -e .

Tutorial
--------

Because the library is still very preliminary there is no real tutorial here, yet. Check out the `run-examples.py`, which is a script that provides a few examples. There are no real explanations to the context of the example models, but generally, you can understand the dynamics of the models by carefully analyzing the flows, that are plotted. The default flow is shown with think lines in light blue, the management flows with thin, dark blue, dotted (or dashed) lines.

As some of the constructions there are a bit tricky, the easiest is probably to simply contact me (Tim.Kittel@pik-potsdam.de). As I am happy about everybody who'd consider using the library, I'll help you as much as I can and might consider writing a proper tutorial and extend the Documentation.

API-Documentation
-----------------

* :ref:`modindex`

Search
------
* :ref:`search`
