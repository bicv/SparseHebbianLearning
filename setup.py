#!/usr/bin/env python
# -*- coding: utf8 -*-
from setuptools import setup, find_packages

NAME = 'shl_scripts'
import shl_scripts
VERSION = shl_scripts.__version__ # << to change in __init__.py

setup(
    name=NAME,
    version=VERSION,
    author='Laurent PERRINET, Institut de Neurosciences de la Timone (CNRS/Aix-Marseille Université)',
    description=' This is a collection of python scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the [SparseNet algorithm from Bruno Olshausen](http://redwood.berkeley.edu/bruno/sparsenet/).',
    long_description=open('README.rst').read(),
    license='LICENSE.txt',
    keywords="Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit",
    url = 'https://github.com/laurentperrinet/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/laurentperrinet/' + NAME + '/tarball/' + VERSION,
    classifiers = ['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'License :: OSI Approved :: GNU General Public License (GPL)',
               'Operating System :: POSIX',
               'Topic :: Scientific/Engineering',
               'Topic :: Utilities',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.4',
              ],
    extras_require={
                'html' : [
                         'vispy',
                         'matplotlib'
                         'jupyter>=1.0']
    },
    # package source directory
    package_dir={'': NAME},
    packages=find_packages(NAME, exclude='docs')
)
