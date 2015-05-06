SHL_scripts
=================

Installation
-------------

``sh
        python setup.py clean build install
``

Licence
--------


Contribute
------------Reproducible research : Python implementation of SparseHebbianLearning
======================================================================

![Animation of the formation of RFs during aSSC learning.]
(http://invibe.net/cgi-bin/index.cgi/SparseHebbianLearning?action=AttachFile&do=get&target=ssc.gif)

This piece of code is distributed under the terms of the GNU General Public License (GPL), check http://www.gnu.org/copyleft/gpl.html if you have not red the term of the license yet.

*  (!)  tl;dr : [Download the code](https://github.com/meduz/SHL_scripts/archive/master.zip). Or directly from the command-line, do

```
wget https://github.com/meduz/SHL_scripts/archive/master.zip
unzip master.zip -d SHL_scripts
cd SHL_scripts/
python learn.py
```

Object
------

* This is a collection of python scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the [SparseNet algorithm from Bruno Olshausen](http://redwood.berkeley.edu/bruno/sparsenet/).

* this has been published as Perrinet, Neural Computation (2010) (see  http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl ):

```bibtex
@article{Perrinet10shl,
    Author = {Perrinet, Laurent U.},
    Doi = {10.1162/neco.2010.05-08-795},
    Journal = {Neural Computation},
    Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
    Month = {July},
    Number = {7},
    Title = {Role of homeostasis in learning sparse representations},
    Url = {http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl},
    Volume = {22},
    Year = {2010},
    Annote = {Posted Online March 17, 2010.},
}
```

* all comments and bug corrections should be submitted to Laurent Perrinet at Laurent.Perrinet@gmail.com
* find out updates on http://invibe.net/LaurentPerrinet/SparseHebbianLearning

Get Ready!
----------

 Be sure to have :

* a computer (tested on Mac, Linux) with ``python`` + ``numpy`` (on macosx, you may consider using [HomeBrew](https://github.com/meduz/dotfiles/blob/master/init/osx_brew_python.sh),
* grab the sources from the [Download the code](https://github.com/meduz/SHL_scripts/archive/master.zip),
* These scripts should be platform independent, however, there is a heavy bias toward unix users when generating figures.

Contents
--------


 * ``README.md`` : this file
 * ``learn.py`` : the scripts (see Contents.m  for a script pointing to the different experiments)
 * ``ssc.py`` : the individual experiments
 * ``IMAGES_sparsenet.mat`` : the image files (if absent, they get automagically downloaded from [this link](http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile|this page).
* ``matlab_code`` : some obsolete matlab code


Some useful code tidbits
------------------------

* get the code with CLI  ``
wget https://github.com/meduz/SHL_scripts/archive/master.zip
``.
* decompress  ``
unzip master.zip -d SHL_scripts
``
* get to the code ``
cd SHL_scripts
``

* run the main script ``
python learn.py
``

* remove SSC related files to start over ``
rm -f IMAGES_*.mat.pdf *.hdf5
``

Changelog
---------

* in the pipes:
 * use ``sklearn.decomposition.SparseCoder`` see http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.htm and http://www.cs.utexas.edu/~leif/pubs/20130626-scipy-johnson.pdf

* 1.1 : 14-06-18
 * documentation
 * dropped Matlab support

* 1.0 : initial release, 27-Oct-2011

