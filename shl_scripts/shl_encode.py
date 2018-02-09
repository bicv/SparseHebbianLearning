#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
import numpy as np
import time

def sparse_encode(X, dictionary, precision=None, algorithm='mp', fit_tol=None,
                  P_cum=None, l0_sparseness=10, C=5., do_sym=False, verbose=0,
                  gain=None):
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
        alpha = 0.01, #float(regularization) / n_pixels  # account for scaling

        from sklearn.linear_model import LassoLars

        # Not passing in verbose=max(0, verbose-1) because Lars.fit already
        # corrects the verbosity level.
        cov = np.dot(dictionary, X.T)
        lasso_lars = LassoLars(fit_intercept=False, # alpha=fit_tol,
                               verbose=verbose, normalize=False,
                               precompute=None, fit_path=False)
        lasso_lars.fit(dictionary.T, X.T, Xy=cov)
        sparse_code = lasso_lars.coef_

    elif algorithm == 'lasso_cd':
        alpha = 1. # float(regularization) / n_pixels  # account for scaling
        max_iter = 1000

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=alpha, fit_intercept=False, normalize=False,
                    precompute=True, max_iter=max_iter, warm_start=True)

        clf.fit(dictionary.T, X.T, check_input=True)
        sparse_code = clf.coef_

    elif algorithm == 'lars':

        # Not passing in verbose=max(0, verbose-1) because Lars.fit already
        # corrects the verbosity level.
        from sklearn.linear_model import Lars
        cov = np.dot(dictionary, X.T)
        lars = Lars(fit_intercept=False, verbose=verbose, normalize=False,
                    precompute=True, n_nonzero_coefs=l0_sparseness,
                    fit_path=False)
        lars.fit(dictionary.T, X.T, Xy=cov)
        sparse_code = lars.coef_

    elif algorithm == 'threshold':
        regularization = 1.
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
        sparse_code = mp(X, dictionary, precision, l0_sparseness=l0_sparseness,
                         fit_tol=fit_tol, P_cum=P_cum, C=C, do_sym=do_sym,
                         verbose=verbose, gain=gain)
    else:
        raise ValueError('Sparse coding method must be "mp", "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    return sparse_code
#
# def get_rescaling(corr, nb_quant, do_sym=False, verbose=False):
#     # if do_sym:
#     #     corr = np.abs(corr)
#     # else:
#     #     corr *= corr>0
#
#     sorted_coeffs = np.sort(corr.ravel())
#     indices = [int(q*(sorted_coeffs.size-1) ) for q in np.linspace(0, 1, nb_quant, endpoint=True)]
#     C_vec = sorted_coeffs[indices]
#     return C_vec

def rectify(code, do_sym=False, verbose=False):
    """

    Simple rectification

    """
    if do_sym:
        return np.abs(code)
    else:
        return code*(code>0)

def rescaling(code, C=5., do_sym=False, verbose=False):
    """
    See

    - http://blog.invibe.net/posts/2017-11-07-meul-with-a-non-parametric-homeostasis.html

    for a derivation of the following function.

    """
    return 1.-np.exp(-rectify(code, do_sym=do_sym)/C)

    # if isinstance(C, (np.float, int)):
    #     if C == 0.: print('WARNING! C is equal to zero!')
    #     elif C == np.inf: return C
    #     if do_sym:
    #         return 1.-np.exp(-np.abs(code)/C)
    #     else:
    #         return (1.-np.exp(-code/C))*(code>0)
    # elif isinstance(C, np.ndarray):
    #     code_bins = np.linspace(0., 1., C.size, endpoint=True)
    #     return np.interp(code, C, code_bins, left=0., right=1.) * (code > 0.)

def inv_rescaling(r, C=5.):
    """
    Inverting the rescaling

    """
    if np.sum(r>=1) + np.sum(r<0) > 0.: print('WARNING! out of range values!')
    code = -C*np.log(1.-r)
    #code[r==1.] = 100#np.inf
    return code

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
        # try:
        indices = (p_c * P_cum.shape[1]).astype(np.int)  # (floor) index of each p_c in the respective line of P_cum
        p = p_c * P_cum.shape[1] - indices  # ratio between floor and ceil
        floor = P_cum.ravel()[indices - (p_c == 1) + stick]  # floor, accounting for extremes, and moved on the raveled P_cum matrix
        ceil = P_cum.ravel()[indices + 1 - (p_c == 0) - (p_c == 1) -
                            (indices >= P_cum.shape[1] - 1) + stick]  # ceiling,  accounting for both extremes, and moved similarly
        # except IndexError as e: # TODO : remove this debugging HACK
        #     print (e)
        #     print(P_cum.shape, np.prod(P_cum.shape), p_c, floor, p,
        #           indices - (p_c == 1) + stick,
        #           indices + 1 - (p_c == 0) - (p_c == 1) - (indices >= stick + P_cum.shape[1] - 1) + stick)
        return (1 - p) * floor + p * ceil
    else:
        code_bins = np.linspace(0., 1., P_cum.shape[1], endpoint=True)
        q_i = np.zeros_like(p_c)
        for i in range(P_cum.shape[0]):
            q_i[i] = np.interp(p_c[i], code_bins, P_cum[i, :], left=0., right=1.)

        #q_i[p_c==0.] = 0.
        return q_i

def inv_quantile(P_cum, q, do_fast=False):
    # TODO : do_fast=True
    n_dictionary, nb_quant = P_cum.shape
    code_bins = np.linspace(0., 1., nb_quant, endpoint=True)
    r = np.zeros_like(q)
    for i in range(n_dictionary):
        r[:, i] = np.interp(q[:, i], P_cum[i, :-1], code_bins[:-1], left=0., right=1.-.5/nb_quant)
    return r

def mp(X, dictionary, precision=None, l0_sparseness=10, fit_tol=None, alpha_MP=1.,
       do_sym=False, P_cum=None, do_fast=True, C=5., verbose=0, gain=None):
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

    # #if fit_tol is None: fit_tol = 0.

    # starting Matching Pursuit
    if precision is None:
        corr = (X @ dictionary.T)
        Xcorr = (dictionary @ dictionary.T)
        #SE_0 = np.sum(X*2, axis=1)
    else:
        print('dooh precision not implemented')
        corr = (X @ (precision*dictionary).T)
        Xcorr = (dictionary @ (precision*dictionary).T)
        #SE_0 = np.sum(X*2, axis=1)

    # COMP
    if gain is None: # SLOW
        nb_quant = P_cum.shape[1]
        stick = np.arange(n_dictionary)*nb_quant

        for i_sample in range(n_samples):
            c = corr[i_sample, :].copy()
            #while (i_l0 < l0_sparseness) or (SE > fit_tol * SE_0):
            for i_l0 in range(int(l0_sparseness)) :
                r = rescaling(c, C=C, do_sym=do_sym)
                q = quantile(P_cum, r, stick, do_fast=do_fast)

                ind = np.argmax(q)
                c_ind = alpha_MP * c[ind] / Xcorr[ind, ind]

                sparse_code[i_sample, ind] += c_ind
                c -= c_ind * Xcorr[ind, :]
    else: # FAST
        gain = gain[np.newaxis, :] * np.ones_like(corr)
        line = np.arange(n_samples)
        for i_l0 in range(int(l0_sparseness)):
                q = rectify(corr, do_sym=do_sym) * gain
                ind = np.argmax(q, axis=1)
                sparse_code[line, ind] += corr[line, ind]
                corr = corr - (Xcorr[ind, :] * corr[line, ind][:, np.newaxis])

    if verbose>0:
        duration=time.time()-t0
        print('coding duration : {0}'.format(duration))

    return sparse_code
