#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
import numpy as np
import time

def sparse_encode(X, dictionary, precision=None, algorithm='mp', fit_tol=None,
                          P_cum=None, l0_sparseness=10, C=0., do_sym=True, verbose=0):
    """Generic sparse coding

    Each column of the result is the solution to a sparse coding problem.

    Parameters
    ----------
    X : array of shape (n_samples, n_pixels)
        Data matrix.

    dictionary : array of shape (n_dictionary, n_pixels)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    precision : array of shape (n_dictionary, n_pixels)
        A matrix giving for each dictionary its respective precision (inverse of variance)

    algorithm : {'mp', 'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        mp :  Matching Pursuit
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated dictionary are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than regularization
        from the projection dictionary * data'

    max_iter : int, 1000 by default
        Maximum number of iterations to perform if `algorithm='lasso_cd'`.

    verbose : int
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    Returns
    -------
    code : array of shape (n_samples, n_dictionary)
        The sparse codes

    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    #n_samples, n_pixels = X.shape

    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_pixels  # account for scaling

        from sklearn.linear_model import LassoLars

        # Not passing in verbose=max(0, verbose-1) because Lars.fit already
        # corrects the verbosity level.
        cov = np.dot(dictionary, X.T)
        lasso_lars = LassoLars(alpha=fit_tol, fit_intercept=False,
                               verbose=verbose, normalize=False,
                               precompute=None, fit_path=False)
        lasso_lars.fit(dictionary.T, X.T, Xy=cov)
        sparse_code = lasso_lars.coef_.T

    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_pixels  # account for scaling

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=fit_tol, fit_intercept=False, normalize=False,
                    precompute=None, max_iter=max_iter, warm_start=True)

        if init is not None:
            clf.coef_ = init

        clf.fit(dictionary.T, X.T, check_input=check_input)
        sparse_code = clf.coef_.T

    elif algorithm == 'lars':

        # Not passing in verbose=max(0, verbose-1) because Lars.fit already
        # corrects the verbosity level.
        from sklearn.linear_model import Lars
        cov = np.dot(dictionary, X.T)
        lars = Lars(fit_intercept=False, verbose=verbose, normalize=False,
                    precompute=None, n_nonzero_coefs=l0_sparseness,
                    fit_path=False)
        lars.fit(dictionary.T, X.T, Xy=cov)
        sparse_code = lars.coef_.T

    elif algorithm == 'threshold':
        cov = np.dot(dictionary, X.T)
        sparse_code = ((np.sign(cov) *
                    np.maximum(np.abs(cov) - regularization, 0))).T

    elif algorithm == 'omp':
        # TODO: Should verbose argument be passed to this?
        from sklearn.linear_model import orthogonal_mp_gram
        from sklearn.utils.extmath import row_norms

        cov = np.dot(dictionary, X.T)
        gram = np.dot(dictionary, dictionary.T)
        sparse_code = orthogonal_mp_gram(
            Gram=gram, Xy=cov, n_nonzero_coefs=l0_sparseness,
            tol=None, norms_squared=row_norms(X, squared=True),
            copy_Xy=False).T

    elif algorithm == 'mp':
        sparse_code = mp(X, dictionary, precision, l0_sparseness=l0_sparseness, fit_tol=fit_tol,
                            P_cum=P_cum, C=C, do_sym=do_sym, verbose=verbose)
    else:
        raise ValueError('Sparse coding method must be "mp", "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    return sparse_code

def get_rescaling(corr, nb_quant, do_sym=False, verbose=False):
    # if do_sym:
    #     corr = np.abs(corr)
    # else:
    #     corr *= corr>0

    sorted_coeffs = np.sort(corr.ravel())
    indices = [int(q*(sorted_coeffs.size-1) ) for q in np.linspace(0, 1, nb_quant, endpoint=True)]
    C_vec = sorted_coeffs[indices]
    return C_vec

def rescaling(code, C=0., do_sym=False, verbose=False):
    """
    See

    - http://blog.invibe.net/posts/2017-11-07-meul-with-a-non-parametric-homeostasis.html

    for a derivation of the following function.

    """
    if isinstance(C, (np.float, int)):
        if C==0.: print('WARNING! C is equal to zero!')
        elif C==np.inf: return C
        if do_sym:
            return 1.-np.exp(-np.abs(code)/C)
        else:
            return (1.-np.exp(-code/C))*(code>0)
    elif isinstance(C, np.ndarray):
        code_bins = np.linspace(0., 1., C.size, endpoint=True)
        return np.interp(code, C, code_bins, left=0., right=1.) * (code > 0.)

def quantile(P_cum, p_c, stick, do_fast=True):
    """
    See

    - http://blog.invibe.net/posts/2017-03-29-testing-comps-pcum.html
    - http://blog.invibe.net/posts/2017-03-29-testing-comps-fastpcum_scripted.html

    for tests of this function

    for a derivation of the following line in the fast mode, see:

    - http://blog.invibe.net/posts/2017-03-29-testing-comps-fastpcum.html

    """
    if do_fast:
        try:
            indices = (p_c * P_cum.shape[1]).astype(np.int)  # (floor) index of each p_c in the respective line of P_cum
            p = p_c * P_cum.shape[1] - indices  # ratio between floor and ceil
            floor = P_cum.ravel()[indices - (p_c == 1) + stick]  # floor, accounting for extremes, and moved on the raveled P_cum matrix
            ceil = P_cum.ravel()[indices + 1 - (p_c == 0) - (p_c == 1) -
                                (indices >= P_cum.shape[1] - 1) + stick]  # ceiling,  accounting for both extremes, and moved similarly
        except IndexError as e: # TODO : remove this debugging HACK
            print (e)
            print(P_cum.shape, np.prod(P_cum.shape), p_c, floor, p,
                  indices - (p_c == 1) + stick,
                  indices + 1 - (p_c == 0) - (p_c == 1) - (indices >= stick + P_cum.shape[1] - 1) + stick)
        return (1 - p) * floor + p * ceil
    else:
        code_bins = np.linspace(0., 1., P_cum.shape[1], endpoint=True)
        q_i = np.zeros_like(p_c)
        for i in range(P_cum.shape[0]):
            q_i[i] = np.interp(p_c[i], code_bins, P_cum[i, :], left=0., right=1.)
        return q_i

def mp(X, dictionary, precision=None, l0_sparseness=10, fit_tol=None, alpha=1., do_sym=True, P_cum=None, do_fast=True, C=0., verbose=0):
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

    precision : array of shape (n_dictionary, n_pixels)
        A matrix giving for each dictionary its respective precision (inverse of variance)

    fit_tol : criterium based on the residual error - not implem    ented yet

    Returns
    -------
    sparse_code : array of shape (n_samples, n_dictionary)
        The sparse code

    """
    # initialization
    if verbose>0:
        t0=time.time()
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_pixels = X.shape
    n_dictionary, n_pixels = dictionary.shape
    sparse_code = np.zeros((n_samples, n_dictionary))
    if not P_cum is None:
        nb_quant = P_cum.shape[1]
        stick = np.arange(n_dictionary)*nb_quant
    #if fit_tol is None: fit_tol = 0.

    # starting Matching Pursuit
    if precision is None:
        corr = (X @ dictionary.T)
        Xcorr = (dictionary @ dictionary.T)
        #SE_0 = np.sum(X*2, axis=1)
    else:
        corr = (X @ (precision*dictionary).T)
        Xcorr = (dictionary @ (precision*dictionary).T)
        #SE_0 = np.sum(X*2, axis=1)


    if not P_cum is None:
        nb_quant = P_cum.shape[1]
        stick = np.arange(n_dictionary)*nb_quant
        if C == 0.:
            C = P_cum[-1, :]
            P_cum = P_cum[:-1, :]

    # TODO: vectorize by doing all patches at the same time?
    for i_sample in range(n_samples):
        c = corr[i_sample, :].copy()
        #c_0 = corr_0[i_sample]
        #i_l0, SE = 0, SE_0
        for i_l0 in range(int(l0_sparseness)) :
        #while (i_l0 < l0_sparseness) or (SE > fit_tol * SE_0):
            q = rescaling(c, C=C, do_sym=do_sym)
            if not P_cum is None:
                q = quantile(P_cum, q, stick, do_fast=do_fast)
            ind = np.argmax(q)
            c_ind = alpha * c[ind] / Xcorr[ind, ind]
            sparse_code[i_sample, ind] += c_ind
            c -= c_ind * Xcorr[ind, :]
            #SE -= c_ind**2 # pythagora
            #i_l0 += 1
    if verbose>0:
        duration=time.time()-t0
        print('coding duration : {0}'.format(duration))
    return sparse_code
