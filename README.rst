Reproducible research : Python implementation of SparseHebbianLearning
======================================================================


.. image:: probe/shl_homeo.png
   :scale: 100%
   :alt: Set of RFs after Sparse Hebbian Learning.


Object
------

* This is a collection of python scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the SparseNet algorithm from Bruno Olshausen (http://redwood.berkeley.edu/bruno/sparsenet/).

* this has been published as Perrinet (2019) (see  https://laurentperrinet.github.io/publication/perrinet-19-hulk/ )::

   @article{Perrinet19hulk,
    abstract = {The formation of structure in the visual system, that is, of the connections between cells within neural populations, is by large an unsupervised learning process: the emergence of this architecture is mostly self-organized. In the primary visual cortex of mammals, for example, one can observe during development the formation of cells selective to localized, oriented features which results in the development of a representation of contours in area V1. We modeled such a process using sparse Hebbian learning algorithms. These algorithms alternate a coding step to encode the information with a learning step to find the proper encoder. We identified here a major difficulty of classical solutions in their ability to deduce a good representation while knowing immature encoders, and to learn good encoders with a non-optimal representation. To solve this problem, we propose to introduce a new regulation process between learning and coding, called homeostasis. It is compatible with a neuromimetic architecture and allows for a more efficient emergence of localized filters sensitive to orientation. The key to this algorithm lies in a simple adaptation mechanism based on non-linear functions that reconciles the antagonistic processes that occur at the coding and learning time scales. We tested this unsupervised algorithm with this homeostasis rule for a series of learning algorithms coupled with different neural coding algorithms. In addition, we propose a simplification of this optimal homeostasis rule by implementing a simple heuristic on the probability of activation of neurons. Compared to the optimal homeostasis rule, we show that this heuristic allows to implement a faster unsupervised learning algorithm while retaining much of its effectiveness. These results demonstrate the potential application of such a strategy in computer vision and machine learning and we illustrate it with a result in a convolutional neural network.},
    author = {Perrinet, Laurent U},
    bdsk-url-1 = {https://github.com/SpikeAI/HULK},
    date-added = {2019-09-01 16:14:10 +0300},
    date-modified = {2019-09-19 12:00:01 +0200},
    doi = {10.3390/vision3030047},
    grants = {anr-horizontal-v1,spikeai; mesocentre},
    journal = {Vision},
    keywords = {area-v1,gain control,homeostasis,matching pursuit,sparse coding,sparse hebbian learning,unsupervised learning},
    month = {sep},
    number = {3},
    pages = {47},
    time_start = {2019-04-18T13:00:00},
    title = {An adaptive homeostatic algorithm for the unsupervised learning of visual features},
    url = {https://spikeai.github.io/HULK/},
    volume = {3},
    year = {2019}
   }



* ... and initially as Perrinet, Neural Computation (2010) (see  https://laurentperrinet.github.io/publication/perrinet-10-shl )::

   @article{Perrinet10shl,
        Author = {Perrinet, Laurent U.},
        Title = {Role of homeostasis in learning sparse representations},
        Year = {2010}
        Url = {https://laurentperrinet.github.io/publication/perrinet-10-shl},
        Doi = {10.1162/neco.2010.05-08-795},
        Journal = {Neural Computation},
        Volume = {22},
        Number = {7},
        Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
        Month = {July},
        }

* all comments and bug corrections should be submitted to Laurent Perrinet at Laurent.Perrinet@univ-amu.fr
* find out updates on https://github.com/bicv/SparseHebbianLearning


Installation
-------------

* Be sure to have dependencies installed::

   pip3 install -U SLIP

* Then, either install the code directly::

   pip3 install git+https://github.com/bicv/SparseHebbianLearning.git

* or if you wish to tinker with the code, download the code @ https://github.com/bicv/SparseHebbianLearning//archive/master.zip. You may also grab it directly using the command-line::

   wget https://github.com/bicv/SparseHebbianLearning//archive/master.zip
   unzip master.zip -d SparseHebbianLearning/
   cd SparseHebbianLearning/
   ipython setup.py clean build install
   jupyter notebook

* developpers may use all the power of git with::

   git clone https://github.com/bicv/SparseHebbianLearning.git

Licence
--------

This piece of code is distributed under the terms of the GNU General Public License (GPL), check http://www.gnu.org/copyleft/gpl.html if you have not red the term of the license yet.

Contents
--------

* ``README.rst`` : this file
* ``index.ipynb`` : an introduction as a notebook
* ``src/shl_*.py`` : the class files
* ``probe*.ipynb`` : the individual experiments as notebooks
* ``database`` : the image files.

Changelog
---------

* 4.0 - 2019-06-06: finalized the code for https://laurentperrinet.github.io/publication/perrinet-19-hulk/

* 3.0 - 2017-06-06: refactored the code for https://laurentperrinet.github.io/publication/boutin-ruffier-perrinet-17-spars/

* 2.1 - 2015-10-20:
 * finalizing the code to reproduce the sparsenet algorithm

* 2.0 - 2015-05-07:
 * transform to a class to just do the Sparse Hebbian Learning (high-level) experiments (getting data from an image folder, learning, coding, analyszing)
 * use sklearn to do all the hard low-level work, in particular ``sklearn.decomposition.SparseCoder`` see http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html and http://www.cs.utexas.edu/~leif/pubs/20130626-scipy-johnson.pdf
 * The dictionary learning is tested in http://blog.invibe.net/posts/2015-05-05-reproducing-olshausens-classical-sparsenet.html and the corresponding PR is tested in http://blog.invibe.net/posts/2015-05-06-reproducing-olshausens-classical-sparsenet-part-2.html

* 1.1 - 2014-06-18:
 * documentation
 * dropped Matlab support

* 1.0 - 2011-10-27 : initial release
