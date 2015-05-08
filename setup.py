from setuptools import setup, find_packages

setup(name='SHL_scripts',
      version='2.0',
      author='Laurent PERRINET, Institut de Neurosciences de la Timone (CNRS/Aix-Marseille Université)',
      description=' This is a collection of python scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the [SparseNet algorithm from Bruno Olshausen](http://redwood.berkeley.edu/bruno/sparsenet/).',
      long_description=open('README.rst').read(),
      license='LICENSE.txt',
      keywords="Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit",

      # package source directory
      package_dir={'': 'src'},
      packages=find_packages('src', exclude='docs')
)