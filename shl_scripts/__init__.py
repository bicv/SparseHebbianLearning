#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
__author__ = "Laurent Perrinet INT - CNRS"
__version__ = '2017-02-09'
__licence__ = 'GPLv2'
__all__ = []

"""
========================================================
Learning filters from natural images using sparse coding
========================================================

* When imposing a code representing patches from natural images to be sparse,
one observes the formation of filters ressembling the receptive field of simple
cells in primates primary visual cortex.  This was first proposed in the
framework of the SparseNet algorithm from Bruno Olshausen
(http://redwood.berkeley.edu/bruno/sparsenet/).

* This particular implementation has been published as Perrinet, Neural
Computation (2010) (see http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl )::

   @article{Perrinet10shl,
        Author = {Perrinet, Laurent U.},
        Title = {Role of homeostasis in learning sparse representations},
        Year = {2010}
        Url = {http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl},
        Doi = {10.1162/neco.2010.05-08-795},
        Journal = {Neural Computation},
        Volume = {22},
        Number = {7},
        Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
        Month = {July},
        }


"""
from .shl_scripts import *
from .shl_tools import *
from .shl_learn import *
