#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
from encode_shl import sparse_encode
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
import os

toolbar_width = 40

import matplotlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tools_shl
# see https://github.com/bicv/SLIP/blob/master/SLIP.ipynb
from SLIP import Image

import numpy as np


import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

def touch(fname):
    open(fname, 'w').close()

class SHL(object):
    """

    Base class to define SHL experiments:
        - initialization
        - coding and learning
        - visualization
        - quantitative analysis

    """
    def __init__(self,
                 height=256,
                 width=256,
                 patch_size=(12, 12),
                 database = 'database/',
                 n_dictionary=14**2,
                 learning_algorithm='mp',
                 alpha=None,
                 l0_sparseness=10,
                 n_iter=2**14,
                 eta=.01,
                 eta_homeo=.05,
                 alpha_homeo=.2,
                 max_patches=1024,
                 batch_size=256,
                 n_image=200,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 data_cache = './data_cache',
                 ):
        self.height = height
        self.width = width
        self.database = database
        self.patch_size = patch_size
        self.n_dictionary = n_dictionary
        self.n_iter = int(n_iter/DEBUG_DOWNSCALE)
        self.max_patches = int(max_patches/DEBUG_DOWNSCALE)
        self.n_image = int(n_image/DEBUG_DOWNSCALE)
        self.batch_size = batch_size
        self.learning_algorithm = learning_algorithm
        self.alpha=alpha

        self.l0_sparseness = l0_sparseness
        self.eta = eta
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo

        self.verbose = verbose
        # assigning and create a folder for caching data
        self.data_cache = './data_cache'
        if not self.data_cache is None:
            try:
                os.mkdir(self.data_cache)
            except:
                pass

        # creating a tag related to this process
        PID, HOST = os.getpid(), os.uname()[1]
        self.LOCK = '_lock' + '_pid-' + str(PID) + '_host-' + HOST


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

    def dev_get_data(self,name_database,seed=None,patch_norm=True):
        return tools_shl.get_data(height=self.height, width=self.width, n_image=self.n_image,
                    patch_size=self.patch_size, datapath=self.database, name_database=name_database,
                    max_patches=self.max_patches, seed=None, patch_norm=True,
                    verbose=self.verbose)

    def learn_dico(self, data=None, name_database='serre07_distractors',
                   matname=None, record_each=0, **kwargs):

        if matname is None:
            if data is None: data = self.dev_get_data(name_database)
            # Learn the dictionary from reference patches
            if self.verbose: print('Learning the dictionary with algo = self.learning_algorithm', end=' ')
            t0 = time.time()
            dico = SparseHebbianLearning(eta=self.eta,
                                         fit_algorithm=self.learning_algorithm,
                                         n_dictionary=self.n_dictionary, n_iter=self.n_iter,
                                         eta_homeo=self.eta_homeo, alpha_homeo=self.alpha_homeo,
                                         l0_sparseness=self.l0_sparseness,
                                         batch_size=self.batch_size, verbose=self.verbose,
                                         fit_tol=self.alpha,
                                         record_each=record_each,
                                          **kwargs)
            if self.verbose: print('Training on %d patches' % len(data), end='... ')
            dico.fit(data)
            if self.verbose:
                dt = time.time() - t0
                print('done in %.2fs.' % dt)

        else:
            import pickle
            fmatname = os.path.join(self.data_cache, matname)
            if not(os.path.isfile(fmatname)):
                time.sleep(np.random.rand()*0.1)
                if not(os.path.isfile(fmatname + '_lock')):
                    touch(fmatname + '_lock')
                    touch(fmatname + self.LOCK)
                    dico = self.learn_dico(data=data, name_database=name_database,
                                           record_each=record_each, matname=None, **kwargs)
                    with open(fmatname, 'wb') as fp:
                        pickle.dump(dico, fp)
                    try:
                        os.remove(fmatname + self.LOCK)
                        os.remove(fmatname + '_lock')
                    except:
                        print('Coud not remove ', fmatname + LOCK)
                else:

                    print('the computation is locked')
            else:
                with open(fmatname, 'rb') as fp:
                    dico = pickle.load(fp)
        return dico

    def code(self, data, dico, coding_algorithm='mp', **kwargs):
        if self.verbose:
            print('Coding data with algorithm ', coding_algorithm,  end=' ')
            t0 = time.time()

        sparse_code = dico.transform(data, algorithm=coding_algorithm)
        patches = np.dot(sparse_code, dico.dictionary)

        if self.verbose:
            dt = time.time() - t0
            print('done in %.2fs.' % dt)
        return patches
# SparseHebbianLearning
class SparseHebbianLearning:
    """Sparse Hebbian learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Parameters
    ----------

    n_dictionary : int,
        Number of dictionary elements to extract

    eta : float
        Gives the learning parameter for the homeostatic gain.

    n_iter : int,
        total number of iterations to perform

    eta_homeo : float
        Gives the learning parameter for the homeostatic gain.

    alpha_homeo : float
        Gives the smoothing exponent  for the homeostatic gain
        If equal to 1 the homeostatic learning rule learns a linear relation to
        variance.

    dict_init : array of shape (n_dictionary, n_pixels),
        initial value of the dictionary for warm restart scenarios

    fit_algorithm : {'mp', 'lars', 'cd'}
        see sparse_encode

    batch_size : int,
        The number of samples to take in each batch.

    l0_sparseness : int, ``0.1 * n_pixels`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'`, `algorithm='mp'`  and
        `algorithm='omp'`.

    fit_tol : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `fit_tol` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `fit_tol` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='mp'` or `algorithm='omp'`, `fit_tol` is the tolerance
        parameter: the value of the reconstruction error targeted. In this case,
        it overrides `l0_sparseness`.

    verbose :
        degree of verbosity of the printed output

    Attributes
    ----------
    dictionary : array, [n_dictionary, n_pixels]
        dictionary extracted from the data


    Notes
    -----
    **References:**

    Olshausen BA, Field DJ (1996).
    Emergence of simple-cell receptive field properties by learning a sparse code for natural images.
    Nature, 381: 607-609. (http://redwood.berkeley.edu/bruno/papers/nature-paper.pdf)

    Olshausen BA, Field DJ (1997)
    Sparse Coding with an Overcomplete Basis Set: A Strategy Employed by V1?
    Vision Research, 37: 3311-3325.   (http://redwood.berkeley.edu/bruno/papers/VR.pdf)

    See also
    --------
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html

    """
    def __init__(self, fit_algorithm, n_dictionary=None, eta=0.02, n_iter=40000,
                 eta_homeo=0.001, alpha_homeo=0.02, dict_init=None,
                 batch_size=100,
                 l0_sparseness=None, fit_tol=None,
                 record_each=200, verbose=False, random_state=None):
        self.eta = eta
        self.n_dictionary = n_dictionary
        self.n_iter = n_iter
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo
        self.fit_algorithm = fit_algorithm
        self.batch_size = batch_size
        self.dict_init = dict_init
        self.l0_sparseness = l0_sparseness
        self.fit_tol = fit_tol
        self.record_each = record_each
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_pixels)
            Training vector, where n_samples in the number of samples
            and n_pixels is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        return_fn = dict_learning(
            X, self.eta, self.n_dictionary, self.l0_sparseness,
            n_iter=self.n_iter, eta_homeo=self.eta_homeo, alpha_homeo=self.alpha_homeo,
            method=self.fit_algorithm, dict_init=self.dict_init,
            batch_size=self.batch_size, record_each=self.record_each,
            verbose=self.verbose, random_state=self.random_state)

        if self.record_each==0:
            self.dictionary = return_fn
        else:
            self.dictionary, self.record = return_fn

    def transform(self, X, algorithm=None, l0_sparseness=None, fit_tol=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_pixels)
            Training vector, where n_samples in the number of samples
            and n_pixels is the number of features.

        Returns
        -------
        self : object
            Returns sparse code.
        """
        if algorithm is None:  algorithm = self.fit_algorithm
        if l0_sparseness is None:  l0_sparseness = self.l0_sparseness
        if fit_tol is None:  fit_tol = self.fit_tol
        #print("coding with algorithm : {0}".format(algorithm))
        return sparse_encode(X, self.dictionary, algorithm=algorithm,
                                fit_tol=fit_tol, l0_sparseness=l0_sparseness)

    def time_plot(self, fname=None, N_nosample=1):
        return tools_shl.time_plot(self, fname=fname, N_nosample=N_nosample)


def dict_learning(X, eta=0.02, n_dictionary=2, l0_sparseness=10, fit_tol=None, n_iter=100,
                       eta_homeo=0.01, alpha_homeo=0.02, dict_init=None,
                       batch_size=100, record_each=0, verbose=False,
                       method='mp', random_state=None):
    """
    Solves a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::


    Solves the optimization problem::

        (U^*, V^*) = argmin_{(U,V)} 0.5 || X - V^T * U ||_2^2
                                    + alpha * S( U )
                                    + alpha_homeo * H(V)

                     s. t. || U ||_0 = k

                    where S is a sparse representation cost,
                    and H a homeostatic representation cost.

    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    For instance,

        H(V) = \sum_{0 <= k < n_dictionary} (|| V_k ||_2^2 -1)^2

    Parameters
    ----------
    X: array of shape (n_samples, n_pixels)
        Data matrix.

    n_dictionary : int,
        Number of dictionary atoms to extract.

    eta : float
        Gives the learning parameter for the homeostatic gain.

    n_iter : int,
        total number of iterations to perform

    eta_homeo : float
        Gives the learning parameter for the homeostatic gain.

    alpha_homeo : float
        Gives the smoothing exponent  for the homeostatic gain
        If equal to 1 the homeostatic learning rule learns a linear relation to
        variance.

    dict_init : array of shape (n_dictionary, n_pixels),
        initial value of the dictionary for warm restart scenarios

    fit_algorithm : {'mp', 'omp', 'comp', 'lars', 'cd'}
        see sparse_encode

    batch_size : int,
        The number of samples to take in each batch.

    l0_sparseness : int, ``0.1 * n_pixels`` by default
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'`, `algorithm='mp'`  and
        `algorithm='omp'`.

    fit_tol : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `fit_tol` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `fit_tol` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='mp'` or `algorithm='omp'`, `fit_tol` is the tolerance
        parameter: the value of the reconstruction error targeted. In this case,
        it overrides `l0_sparseness`.

    record_each :
        if set to 0, it does nothing. Else it records every record_each step the
        statistics during the learning phase (variance and kurtosis of coefficients).

    verbose :
        degree of verbosity of the printed output

    Returns
    -------

    dictionary : array of shape (n_dictionary, n_pixels),
        the solutions to the dictionary learning problem

    """

    if record_each>0:
        import pandas as pd
        record = pd.DataFrame()

    if n_dictionary is None:
        n_dictionary = X.shape[1]

    t0 = time.time()
    n_samples, n_pixels = X.shape

    dictionary = np.random.randn(n_dictionary, n_pixels)
    norm = np.sqrt(np.sum(dictionary**2, axis=1))
    dictionary /= norm[:, np.newaxis]
    norm = np.sqrt(np.sum(dictionary**2, axis=1))

    if verbose == 1:
        print('[dict_learning]', end=' ')
    gain = np.ones(n_dictionary)
    mean_var = np.ones(n_dictionary)
    if method=='comp':
        mod = np.dot(np.linspace(1., 0, n_components, endpoint=True).T, np.ones((n_samples, 1)))
    else:
        mod = None

    # splits the whole dataset into batches
    n_batches = n_samples // batch_size
    X_train = X.copy()
    np.random.shuffle(X_train)
    batches = np.array_split(X_train, n_batches)
    import itertools
    # Return elements from list of batches until it is exhausted. Then repeat the sequence indefinitely.
    batches = itertools.cycle(batches)

    for ii, this_X in zip(range(n_iter), batches):
        dt = (time.time() - t0)
        if verbose > 0:
            if ii % int(n_iter//verbose + 1) == 0:
                print ("Iteration % 3i /  % 3i (elapsed time: % 3is, % 4.1fmn)"
                       % (ii, n_iter, dt, dt//60))

        # Sparse cooding
        sparse_code = sparse_encode(this_X, dictionary, algorithm=method, fit_tol=fit_tol,
                                  mod=mod, l0_sparseness=l0_sparseness)

        # Update dictionary
        residual = this_X - sparse_code @ dictionary
        residual /= n_dictionary # divide by the number of features
        dictionary += eta * sparse_code.T @ residual

        # Update
        if method=='comp': mod = update_mod(mod, dictionary.T, X.T, eta_homeo, verbose=verbose)

        # homeostasis
        norm = np.sqrt(np.sum(dictionary**2, axis=1)).T
        dictionary /= norm[:, np.newaxis]
        # Update and apply gain

        #if eta_homeo>0.:
        #    gain_ = update_gain(gain_, sparse_code, eta_homeo, verbose=verbose)
        #    gain_ /= gain_.mean()
        #    gain = gain_**alpha_homeo

        if eta_homeo>0.:
            mean_var = update_gain(mean_var, sparse_code, eta_homeo, verbose=verbose)
            gain = mean_var**alpha_homeo
            gain /= gain.mean()
            # print(np.mean(sparse_code**2, axis=0), gain, gain.mean())
            dictionary /= gain[:, np.newaxis]

        if record_each>0:
            if ii % int(record_each) == 0:
                from scipy.stats import kurtosis
                record_one = pd.DataFrame([{'kurt':kurtosis(sparse_code, axis=0),
                                            'var':np.mean(sparse_code**2, axis=0)}],
                                            index=[ii])
                record = pd.concat([record, record_one])


    if verbose > 1:
        print('Learning code...', end=' ')
    elif verbose == 1:
        print('|', end=' ')

    if verbose > 1:
        dt = (time.time() - t0)
        print('done (total time: % 3is, % 4.1fmn)' % (dt, dt / 60))

    if record_each==0:
        return dictionary
    else:
        return dictionary, record

def update_gain(gain, code, eta_homeo, verbose=False):
    """Update the estimated variance of coefficients in place.

    Following the classical SparseNet algorithm from Olshausen, we
    compute here a "gain vector" for the dictionary. This gain will
    be used to tune the weight of each dictionary element.

    The heuristics used here follows the assumption that during learning,
    some elements that learn first are more responsive to input patches
    as may be recorded by estimating their mean variance. If we were to
    keep their norm fixed, these would be more likely to be selected again,
    leading to a ``monopolistic'' distribution of dictionary elements,
    some having learned more often than others. By dividing their norm
    by their mean estimated variance, we lower the probability of elements
    with high variance to be selected again. This thus helps the learning
    to be more balanced.

    Parameters
    ----------
    gain: array of shape (n_dictionary)
        Value of the dictionary' norm at the previous iteration.

    code: array of shape (n_dictionary, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    eta_homeo: float
        Gives the learning parameter for the gain.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    gain: array of shape (n_dictionary)
        Updated value of the dictionary' norm.

    """
    if code.ndim == 1:
        code = code[:, np.newaxis]
    if eta_homeo>0.:
        n_dictionary, n_samples = code.shape
        #print (gain.shape) # assert gain.shape == n_dictionary
        gain = (1 - eta_homeo)*gain + eta_homeo * np.mean(code**2, axis=0)/np.mean(code**2)
    return gain


def update_mod(mod, dictionary, X, eta_homeo, verbose=False):
    """Update the estimated modulation function in place.

    Parameters
    ----------
    mod: array of shape (n_samples, n_components)
        Value of the modulation function at the previous iteration.

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    X: array of shape (n_samples, n_features)
        Data matrix.

    eta_homeo: float
        Gives the learning parameter for the mod.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    mod: array of shape (n_samples, n_components)
        Updated value of the modulation function.

    """
    if eta_homeo>0.:
        coef = np.dot(dictionary, X.T).T
        mod_ = -np.sort(-np.abs(coef), axis=0)
        mod = (1 - eta_homeo)*mod + eta_homeo * mod_
    return mod
'''
def mp(X, dictionary, l0_sparseness=10, fit_tol=None, verbose=0):
    """
    Matching Pursuit
    cf. https://en.wikipedia.org/wiki/Matching_pursuit

    Parameters
    ----------
    X : array of shape (n_samples, n_pixels)
        Data matrix.

    dictionary : array of shape (n_dictionary, n_pixels)
        The dictionary matrix against which to solve the sparse coding of
        the data.

    Returns
    -------
    sparse_code : array of shape (n_samples, n_dictionary)
        The sparse code

    """

    # if mod is not None:
    #     n_components, n_samples = mod.shape
    #     z = np.empty(n_components)
    # while True:
    #     if mod is None:
    #         lam = np.argmax(np.abs(alpha))
    #     else:
    #         for k in range(n_components):
    #             z[k] = np.interp(-np.abs(alpha[k]), -mod[:, k], np.linspace(0, 1., n_samples, endpoint=True))
    #         lam = np.argmax(z)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_pixels = X.shape
    n_dictionary, n_pixels = dictionary.shape
    sparse_code = np.zeros((n_samples, n_dictionary))

    corr = (X @ dictionary.T)
    Xcorr = (dictionary @ dictionary.T)
    # TODO: vectorize?
    for i_sample in range(n_samples):
        c = corr[i_sample, :].copy()
        for i_l0 in range(int(l0_sparseness)):
            ind  = np.argmax(np.abs(c))
            a_i = c[ind] / Xcorr[ind, ind]
            sparse_code[i_sample, ind] += a_i
            c -= a_i * Xcorr[ind, :]

    return sparse_code
'''

if __name__ == '__main__':
    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 0
    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', verbose=verbose)
    dico = shl.learn_dico()
    fig, ax = shl.show_dico(dico)
    plt.savefig('assc.png')
    plt.show()
