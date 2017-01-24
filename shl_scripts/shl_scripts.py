#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
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

import time
import sys

toolbar_width = 40

import matplotlib
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import SparseHebbianLearning
from sklearn.feature_extraction.image import extract_patches_2d

from SLIP import Image
import numpy as np

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

class SHL(object):
    """
    Base class to define SHL experiments:
        - intializing
        - running learning
        - visualization
        - quantitative analysis

    """
    def __init__(self,
                 height=256,
                 width=256,
                 patch_size=(12, 12),
                 database = 'database/',
                 n_components=14**2,
                 learning_algorithm='omp',
                 alpha=None,
                 transform_n_nonzero_coefs=20,
                 n_iter=5000,
                 eta=.01,
                 eta_homeo=.01,
                 alpha_homeo=.01,
                 max_patches=1000,
                 batch_size=100,
                 n_image=200,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 ):
        self.height = height
        self.width = width
        self.database = database
        self.patch_size = patch_size
        self.n_components = n_components
        self.n_iter = int(n_iter/DEBUG_DOWNSCALE)
        self.max_patches = int(max_patches/DEBUG_DOWNSCALE)
        self.n_image = int(n_image/DEBUG_DOWNSCALE)
        self.batch_size = batch_size
        self.learning_algorithm = learning_algorithm
        self.alpha=alpha

        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.eta = eta
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo

        self.verbose = verbose
        # Load natural images and extract patches
        self.slip = Image({'N_X':height, 'N_Y':width,
                                        'white_n_learning' : 0,
                                        'seed': None,
                                        'white_N' : .07,
                                        'white_N_0' : .0, # olshausen = 0.
                                        'white_f_0' : .4, # olshausen = 0.2
                                        'white_alpha' : 1.4,
                                        'white_steepness' : 4.,
                                        'datapath': self.database,
                                        'do_mask':True,
                                        'N_image': n_image})

    def get_data(self, name_database='serre07_distractors', seed=None, patch_norm=True):
        if self.verbose:
            # setup toolbar
            sys.stdout.write('Extracting data...')
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
            t0 = time.time()
        imagelist = self.slip.make_imagelist(name_database=name_database)#, seed=seed)
        for filename, croparea in imagelist:
            # whitening
            image, filename_, croparea_ = self.slip.patch(name_database, filename=filename, croparea=croparea, center=False)#, , seed=seed)
            image = self.slip.whitening(image)
            # Extract all reference patches and ravel them
            data_ = extract_patches_2d(image, self.patch_size, max_patches=int(self.max_patches))#, seed=seed)
            data_ = data_.reshape(data_.shape[0], -1)
            data_ -= np.mean(data_, axis=0)
            if patch_norm:
                data_ /= np.std(data_, axis=0)
            # collect everything as a matrix
            try:
                data = np.vstack((data, data_))
            except Exception:
                data = data_.copy()
            if self.verbose:
                # update the bar
                sys.stdout.write(filename + ", ")
                sys.stdout.flush()
        if self.verbose:
            dt = time.time() - t0
            sys.stdout.write("\n")
            sys.stdout.write("Data is of shape : "+ str(data.shape))
            sys.stdout.write('done in %.2fs.' % dt)
            sys.stdout.flush()
        return data


    def learn_dico(self, name_database='serre07_distractors', **kwargs):
        data = self.get_data(name_database)
        # Learn the dictionary from reference patches
        if self.verbose: print('Learning the dictionary...', end=' ')
        t0 = time.time()
        dico = SparseHebbianLearning(eta=self.eta,
                                     n_components=self.n_components, n_iter=self.n_iter,
                                     gain_rate=self.eta_homeo, alpha_homeo=self.alpha_homeo,
                                     transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
                                     batch_size=self.batch_size, verbose=self.verbose, n_jobs=1,
                                     transform_algorithm=self.learning_algorithm, transform_alpha=self.alpha, **kwargs)
        if self.verbose: print('Training on %d patches' % len(data), end='... ')
        dico.fit(data)
        if self.verbose:
            dt = time.time() - t0
            print('done in %.2fs.' % dt)
        return dico

    def show_dico(self, dico, title=None, fname=None):
        subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
        fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)
        for i, component in enumerate(dico.components_):
            ax = fig.add_subplot(np.sqrt(self.n_components), np.sqrt(self.n_components), i + 1)
            cmax = np.max(np.abs(component))
            ax.imshow(component.reshape(self.patch_size), cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                    interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
        if title is not None:
            fig.suptitle(title, fontsize=12, backgroundcolor = 'white', color = 'k')
        #fig.tight_layout(rect=[0, 0, .9, 1])
        if not fname is None: fig.savefig(fname, dpi=200)
        return fig, ax

    def code(self, data, dico, intercept=0., coding_algorithm='omp', **kwargs):
        if self.verbose:
            print('Coding data...', end=' ')
            t0 = time.time()
        dico.set_params(transform_algorithm=coding_algorithm, **kwargs)
        sparse_code = dico.transform(data)
        V = dico.components_

        patches = np.dot(sparse_code, V)

        if coding_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()

        patches += intercept
#         patches = patches.reshape(len(data), *self.patch_size)
        if coding_algorithm == 'threshold':
            patches -= patches.min()
            patches /= patches.max()
        if self.verbose:
            dt = time.time() - t0
            print('done in %.2fs.' % dt)
        return patches

    def plot_variance(self, dico, name_database='serre07_distractors', fname=None):
        data = self.get_data(name_database)
        sparse_code = dico.transform(data)
        # code = self.code(data, dico)
        Z = np.mean(sparse_code**2)
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(self.n_components), np.mean(sparse_code**2, axis=0)/Z)#, yerr=np.std(code**2/Z, axis=0))
        ax.set_title('Variance of coefficients')
        ax.set_ylabel('Variance')
        ax.set_xlabel('#')
        ax.set_xlim(0, self.n_components)
        if not fname is None: fig.savefig(fname, dpi=200)
        return fig, ax

    def plot_variance_histogram(self, dico, name_database='serre07_distractors', fname=None):
        from scipy.stats import gamma
        import pandas as pd
        import seaborn as sns
        data = self.get_data(name_database)
        sparse_code = dico.transform(data)
        Z = np.mean(sparse_code**2)
        df = pd.DataFrame(np.mean(sparse_code**2, axis=0)/Z, columns=['Variance'])
        #code = self.code(data, dico)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        with sns.axes_style("white"):
            ax = sns.distplot(df['Variance'], kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
        ax.set_title('distribution of the mean variance of coefficients')
        ax.set_ylabel('pdf')
        ax.set_xlim(0)
        if not fname is None: fig.savefig(fname, dpi=200)
        return fig, ax

if __name__ == '__main__':
    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 0
    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='omp', verbose=verbose)
    dico = shl.learn_dico()
    fig, ax = shl.show_dico(dico)
    plt.savefig('assc.png')
    plt.show()
