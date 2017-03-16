#!/usr/bin/env python
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
from encode_shl import sparse_encode
import time
import shl_tools
import numpy as np

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

    def plot_variance(self, data=None, algorithm=None, fname=None):
        return shl_tools.plot_variance(self, data=data, fname=fname, algorithm=algorithm)

    def plot_variance_histogram(self, data, algorithm=None, fname=None):
        return shl_tools.plot_variance_histogram(self, data=data, fname=fname, algorithm=algorithm)

    def time_plot(self, variable='kurt', fname=None, N_nosample=1):
        return shl_tools.time_plot(self, variable=variable, fname=fname, N_nosample=N_nosample)

    def show_dico(self, title=None, fname=None):
        return shl_tools.show_dico(self, title=title, fname=fname)


def dict_learning(X, eta=0.02, n_dictionary=2, l0_sparseness=10, fit_tol=None, n_iter=100,
                       eta_homeo=0.01, alpha_homeo=0.02, dict_init=None,
                       batch_size=100, record_each=0, record_num_batches = 1000, verbose=False,
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

    record_num_batches :
        number of batches used to make statistics (if -1, uses the whole training set)

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
                indx = np.random.permutation(X_train.shape[0])[:record_num_batches]
                sparse_code = sparse_encode(X_train[indx, :], dictionary, algorithm=method, fit_tol=fit_tol,
                                          mod=mod, l0_sparseness=l0_sparseness)
                record_one = pd.DataFrame([{'kurt':kurtosis(sparse_code, axis=0),
                                            'prob_active':np.mean(np.abs(sparse_code)>0, axis=0),
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
