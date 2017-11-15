#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
import numpy as np
import time

def sparse_encode(X, dictionary, algorithm='mp', fit_tol=None,
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
        sparse_code = mp(X, dictionary, l0_sparseness=l0_sparseness, fit_tol=fit_tol,
                            P_cum=P_cum, C=C, do_sym=do_sym, verbose=verbose)
    else:
        raise ValueError('Sparse coding method must be "mp", "lasso_lars" '
                         '"lasso_cd",  "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    return sparse_code

def get_rescaling(code, nb_quant, do_sym=False, verbose=False):
    if do_sym:
        code = np.abs(code)
    else:
        code *= code>0

    sorted_coeffs = np.sort(code.ravel())
    indices = [int(q*(sorted_coeffs.size-1) ) for q in np.linspace(0, 1, nb_quant, endpoint=True)]
    C = sorted_coeffs[indices]
    return C

def rescaling(code, C=0., do_sym=False, verbose=False):
    """
    See

    - http://blog.invibe.net/posts/2017-11-07-meul-with-a-non-parametric-homeostasis.html

    for a derivation of the following function.

    """
    if isinstance(C, np.float):
        if C==0.: print('WARNING! C is equal to zero!')
        if do_sym:
            return 1.-np.exp(-np.abs(code)/C)
        else:
            return (1.-np.exp(-code/C))*(code>0)
    elif isinstance(C, np.ndarray):
        if do_sym:
            code = np.abs(code)
        else:
            code *= code>0

        #code_bins = np.linspace(0., 1., C.size, endpoint=True)
        #return np.interp(code, C, code_bins) * (code > 0.)
        p_c = np.zeros_like(code)
        ind_nz = code>0.
        code_bins = np.linspace(0., 1, C.size, endpoint=True)
        p_c[ind_nz] = np.interp(code[ind_nz], C, code_bins)
        return p_c

def quantile(P_cum, p_c, stick):
    """
    See

    - http://blog.invibe.net/posts/2017-03-29-testing-comps-pcum.html
    - http://blog.invibe.net/posts/2017-03-29-testing-comps-fastpcum.html
    - http://blog.invibe.net/posts/2017-03-29-testing-comps-fastpcum_scripted.html

    for a derivation of the following line.

    """
    return P_cum.ravel()[(p_c*P_cum.shape[1] - (p_c==1)).astype(np.int) + stick]

def mp(X, dictionary, l0_sparseness=10, fit_tol=None, do_sym=True, P_cum=None, C=0., verbose=0):
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

    fit_tol : criterium based on the residual error - not implemented yet

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
    corr = (X @ dictionary.T)
    Xcorr = (dictionary @ dictionary.T)
    #SE_0 = np.sum(X*2, axis=1)

    if not P_cum is None:
        nb_quant = P_cum.shape[1]
        stick = np.arange(n_dictionary)*nb_quant
        if C == 0.:
            C = P_cum[-1, :]
            P_cum = P_cum[:-1, :]

    # TODO: vectorize?
    for i_sample in range(n_samples):
        c = corr[i_sample, :].copy()
        #c_0 = corr_0[i_sample]
        #i_l0, SE = 0, SE_0
        for i_l0 in range(int(l0_sparseness)) :
        #while (i_l0 < l0_sparseness) or (SE > fit_tol * SE_0):
            if P_cum is None:
                if do_sym:
                    ind  = np.argmax(np.abs(c))
                else:
                    ind  = np.argmax(c)
            else:
                ind  = np.argmax(quantile(P_cum, rescaling(c, C=C, do_sym=do_sym), stick))
            #print(i_l0, ind, rescaling(c, C=C, do_sym=do_sym))
            c_ind = c[ind] / Xcorr[ind, ind]
            sparse_code[i_sample, ind] += c_ind
            c -= c_ind * Xcorr[ind, :]
            #SE -= c_ind**2 # pythagora
            #i_l0 += 1
    if verbose>0:
        duration=time.time()-t0
        print('coding duration : {0}'.format(duration))
    return sparse_code
