#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
from shl_scripts import sparse_encode
from shl_scripts import quantile, rescaling
from shl_scripts import inv_quantile, inv_rescaling
from shl_scripts import mutual_coherence, get_record
import time
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

    eta : float or dict
        Gives the learning parameter for the homeostatic gain.

    n_iter : int,
        total number of iterations to perform

    eta_homeo : float
        Gives the learning parameter for the homeostatic gain.

    alpha_homeo : float
        Gives the smoothing exponent  for the homeostatic gain
        If equal to 1 the homeostatic learning rule learns a linear relation to
        variance.

    dictionary : array of shape (n_dictionary, n_pixels),
        initial value of the dictionary for warm restart scenarios
        Use ``None`` for a new learning.

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
    def __init__(self, fit_algorithm='mp', dictionary=None, precision=None,
                 eta=.003, beta1=.9, beta2=.999, epsilon=8,
                 homeo_method = 'HEH',
                 eta_homeo=0.05, alpha_homeo=0.0, C=5., nb_quant=256, P_cum=None,
                 n_dictionary=None, n_iter=10000,
                 batch_size=32,
                 l0_sparseness=None, fit_tol=None, alpha_MP=1.,
                 do_precision=False, eta_precision=0.01, do_sym=False,
                 record_each=200, record_num_batches=2**12,
                 verbose=False, one_over_F=True):
        self.fit_algorithm = fit_algorithm
        self.dictionary = dictionary
        self.precision = precision
        self.n_dictionary = n_dictionary
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.homeo_method = homeo_method
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo
        self.C = C
        self.nb_quant = nb_quant
        self.P_cum = P_cum
        self.n_iter = n_iter
        self.do_sym = do_sym
        self.batch_size = batch_size
        self.l0_sparseness = l0_sparseness
        self.alpha_MP = alpha_MP
        self.fit_tol = fit_tol
        self.do_precision = do_precision
        self.eta_precision = eta_precision
        self.record_each = record_each
        self.record_num_batches = record_num_batches
        self.verbose = verbose
        self.one_over_F = one_over_F

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
        return_fn = dict_learning(X, dictionary=self.dictionary,
                                  do_precision=self.do_precision, eta_precision=self.eta_precision,
                                  eta=self.eta, beta1=self.beta1, beta2=self.beta2,
                                  epsilon=self.epsilon,
                                  homeo_method=self.homeo_method,
                                  eta_homeo=self.eta_homeo,
                                  alpha_homeo=self.alpha_homeo,
                                  C=self.C,
                                  nb_quant=self.nb_quant,
                                  P_cum=self.P_cum,
                                  n_dictionary=self.n_dictionary,
                                  l0_sparseness=self.l0_sparseness, alpha_MP=self.alpha_MP,
                                  n_iter=self.n_iter, method=self.fit_algorithm,
                                  do_sym=self.do_sym, one_over_F=self.one_over_F,
                                  batch_size=self.batch_size,
                                  record_each=self.record_each,
                                  record_num_batches=self.record_num_batches,
                                  verbose=self.verbose
                                  )

        if self.record_each==0:
            self.dictionary, self.precision, self.P_cum = return_fn
        else:
            self.dictionary, self.precision, self.P_cum, self.record = return_fn

    def transform(self, X, algorithm=None, l0_sparseness=None, fit_tol=None, alpha_MP=None):
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
        if alpha_MP is None:  alpha_MP = self.alpha_MP
        return sparse_encode(X, self.dictionary, self.precision, algorithm=algorithm, P_cum=self.P_cum,
                                fit_tol=fit_tol, l0_sparseness=l0_sparseness, alpha_MP=alpha_MP)


def dict_learning(X, dictionary=None, precision=None,
                  eta=.003, beta1=.9, beta2=.999, epsilon=8,
                  homeo_method = 'HEH',
                  eta_homeo=0.05, alpha_homeo=0.0,  C=5., nb_quant=256, P_cum=None,
                  n_dictionary=2, l0_sparseness=10, fit_tol=None, alpha_MP=1.,
                  do_precision=False, eta_precision=0.001,
                  n_iter=100, one_over_F=True,
                  batch_size=100, record_each=0, record_num_batches=2**12, verbose=False,
                  method='mp', do_sym=False):
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

    dictionary: array of shape (n_dictionary, n_pixels)
        Initialization for the dictionary. If None, we chose Gaussian noise.

    precision: array of shape (n_dictionary, n_pixels)
        Initialization for the precision matrix. If None, we chose a uniform matrix.

    n_dictionary : int,
        Number of dictionary atoms to extract.

    eta : float or dict
        Gives the learning parameter for the dictionary.

    n_iter : int,
        total number of iterations to perform

    eta_homeo : float
        Gives the learning parameter for the homeostatic gain.

    alpha_homeo : float
        Gives the smoothing exponent  for the homeostatic gain
        If equal to 1 the homeostatic learning rule learns a linear relation to
        variance.
        If equal to zero, we use COMP

    nb_quant : int,
        number of bins for the quantification used in the homeostasis

    C : float
        characteristic scale for the quantization.
        Use C=0. to have an adaptive scaling.

    dictionary : array of shape (n_dictionary, n_pixels),
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

    do_precision: boolean
        Switch to perform or not a precision-weighted learning.

    eta_precision: float
        Gives the learning parameter for computing the precision.

    do_mask: boolean
        A switch to learn the filters just on a disk. This allows to control for potential
        problems with the fact that images atre sampled on a square grid.

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
    # print('n_dictionary=', n_dictionary)
    # print('n_pixels=', n_pixels)
    # print('n_samples=', n_samples)
    if dictionary is None:
        if not one_over_F:
            dictionary = np.random.randn(n_dictionary, n_pixels)
        else:
            from shl_scripts.shl_tools import ovf_dictionary
            dictionary = ovf_dictionary(n_dictionary, n_pixels)

    norm = np.sqrt(np.sum(dictionary**2, axis=1))
    dictionary /= norm[:, np.newaxis]

    variance = (X**2).mean() * np.ones((n_dictionary, n_pixels))
    precision = 1 / variance if do_precision else None

    if verbose==1:
        print('[dict_learning]', end=' ')

    #initializing parameters
    if beta1 == 0:
        do_adam = False
    else:
        do_adam = True
        moment = energy = np.zeros_like(dictionary)

    # default homeostasis parameters
    mean_measure = None
    if homeo_method=='HEH':
        # we do not use a gain
        # but instead do the equalitarian homeostasis
        gain = None
    else:
        gain = np.ones(n_dictionary)

    #initializing cumulative probability
    if P_cum is None:
        P_cum = np.linspace(0., 1., nb_quant, endpoint=True)[np.newaxis, :] * np.ones((n_dictionary, 1))

    # splits the whole dataset into batches
    n_batches = n_samples // batch_size
    X_train = X.copy()
    # Modifies the sequence in-place by shuffling its contents; Multi-dimensional arrays are only shuffled along the first axis:
    np.random.shuffle(X_train)
    # Splits into ``n_batches`` batches of size batch_size
    batches = np.array_split(X_train, n_batches)
    import itertools
    # Return elements from list of batches until it is exhausted. Then repeat the sequence indefinitely.
    batches = itertools.cycle(batches)

    # cycle over all batches
    for ii, this_X in zip(range(n_iter), batches):

        # Sparse coding
        sparse_code = sparse_encode(this_X, dictionary, precision if do_precision else None,
                                    algorithm=method, fit_tol=fit_tol,
                                    P_cum=P_cum, C=C, do_sym=do_sym, l0_sparseness=l0_sparseness,
                                    gain=gain, alpha_MP=alpha_MP)
        residual = this_X - sparse_code @ dictionary

        # compute variance
        if eta_precision>0:
            # precision *= 1-eta
            variance *= 1-eta_precision
            if True:
                variance_ = np.zeros((n_dictionary, n_pixels))
                for i in range(n_dictionary):
                    if True:
                        # rec_i = (sparse_code[:, i][:, None]) @ (dictionary[i, :][None, :])
                        rec_i = sparse_code[:, i][:, None] * dictionary[i, :][None, :]
                        #print (rec_i.shape)
                        variance_[i, :] = ((this_X - rec_i)**2).mean(axis=0)
                        # variance_[i, :] = (sparse_code[:, i][:, None]**2 * (this_X - rec_i)**2).mean(axis=0)
                        # m = (sparse_code[:, i]**2).mean()
                        # if m>0:
                        #     variance_[i, :] /= m
                    # else:
                    #     # rec_i = (sparse_code[:, i][:, None]) @ (dictionary[i, :][None, :])
                    #     #print (rec_i.shape)
                    #     variance_[i, :] = ((this_X / sparse_code[:, i][:, None] - dictionary[i, :][None, :])**2).mean(axis=0)
            # else:
            #     # variance_ = ((this_X[:, np.newaxis, :] - dictionary[np.newaxis, :, :]*sparse_code[:, :, np.newaxis])**2).mean(axis=0)
            # #variance_[i, :] = (residual**2).mean(axis=0)
            #     variance_ = - sparse_code.T @ (residual**2)
            #     variance_ /= batch_size

            # print('minmax variance_', variance_.min(axis=1), variance_.max(axis=1), variance_.max(axis=1).shape)
            # print('minmax variance_', variance_.min(axis=0).reshape((12, 12)), variance_.max(axis=0).reshape((12, 12)), variance_.max(axis=0).shape)
            # print('minmax variance_', variance_.min(), variance_.max(), variance_.shape)
            variance += eta_precision    * variance_
            # print('minmax variance', variance.min(), variance.max(), variance.shape)
            # precision += eta * 1/variance_
            precision = 1./(variance + 1.e-16)
            # print('minmax precision', precision.min(), precision.max(), precision.shape)

        # homeostasis : we compute P_cum and define different strategies
        mean_measure, P_cum, gain = homeostasis(mean_measure, P_cum, gain,
                        homeo_method, sparse_code, eta_homeo, alpha_homeo,
                        nb_quant=nb_quant, verbose=verbose, C=C, do_sym=do_sym)

        # Update dictionary: Hebbian learning
        gradient = - sparse_code.T @ residual
        # divide by the batch size to get the average in the Hebbian learning:
        gradient /= batch_size

        # if do_precision: # modulates gradient by the precision
        #     gradient *= precision / precision.mean()

        if do_adam:
            # ADAM https://arxiv.org/pdf/1412.6980.pdf
            #  biased first moment estimate
            moment = beta1 * moment + (1 - beta1) * gradient
            # biased second raw moment estimate
            energy = beta2 * energy + (1 - beta2) * (gradient**2)
            energy_ = (np.sqrt(energy / (1-beta2**(ii+1))) + 10**(-epsilon))
            dictionary -= eta * (moment / (1-beta1**(ii+1))) / energy_
        else:
            dictionary -= eta * gradient

        # we normalise filters
        if do_precision:
            squared_norm = np.sqrt(np.diagonal(dictionary @ (precision*dictionary).T))
        else:
            squared_norm = np.sqrt(np.sum(dictionary**2, axis=1)).T
        dictionary /= squared_norm[:, np.newaxis]

        cputime = (time.time() - t0)

        if verbose > 0:
            if ii % int(record_each)==0:
                print ("Iteration % 3i /  % 3i (elapsed time: % 3is, % 3imn % 3is)"
                       % (ii + 1, n_iter, cputime, cputime//60, cputime%60))

        if record_each>0:
            if ii % int(record_each)==0:
                MC = mutual_coherence(dictionary)

                indx = np.random.permutation(X_train.shape[0])[:record_num_batches]
                sparse_code_rec = sparse_encode(X_train[indx, :], dictionary,
                                                precision if do_precision else None,
                                                algorithm=method, fit_tol=fit_tol,
                                                 # P_cum=P_cum, gain=gain,
                                                 P_cum=None, gain=np.ones(n_dictionary),
                                                 C=C, do_sym=do_sym,
                                                 l0_sparseness=l0_sparseness)

                from shl_scripts.shl_tools import get_logL
                logL = get_logL(sparse_code_rec).mean()
                from shl_scripts.shl_tools import get_MI
                MI = get_MI(sparse_code_rec)

                # calculation of relative entropy
                p_ = np.count_nonzero(sparse_code_rec, axis=0) / (sparse_code_rec.shape[1])
                p_ /= p_.sum()
                rel_ent = np.sum(-p_ * np.log(p_)) / np.log(sparse_code_rec.shape[1])
                
                # relative error
                SD = np.linalg.norm(X_train[indx, :])/record_num_batches
                error = np.linalg.norm(X_train[indx, :] - (sparse_code_rec @ dictionary))/record_num_batches

                # calculation of quantization error
                # stick = np.arange(n_dictionary)*nb_quant
                # q = quantile(P_cum, rescaling(sparse_code_rec, C=C), stick)
                P_cum_mean = P_cum.mean(axis=0)[np.newaxis, :] * np.ones((n_dictionary, nb_quant))
                # q_sparse_code = inv_rescaling(inv_quantile(P_cum_mean, q), C=C)
                # qerror = np.linalg.norm(X_train[indx, :] - (q_sparse_code @ dictionary))/record_num_batches
                from shl_scripts import get_KS
                qerror = np.sum(get_KS(P_cum_mean, P_cum))

                from shl_scripts.shl_tools import get_perror
                perror = get_perror(X_train[indx, :], dictionary, precision,
                                            algorithm=method, fit_tol=fit_tol,
                                             P_cum=P_cum, gain=gain,
                                             C=C, do_sym=do_sym,
                                             l0_sparseness=l0_sparseness)

                from scipy.stats import kurtosis
                record_one = pd.DataFrame([{'kurt':kurtosis(sparse_code_rec, axis=0),
                                            'prob_active':np.mean(np.abs(sparse_code_rec)>0, axis=0),
                                            'var':np.mean(sparse_code_rec**2, axis=0),
                                            'error':error/SD,
                                            'logL':logL,
                                            'MI':MI,
                                            'MC':MC,
                                            'qerror':qerror,
                                            # 'aerror':aerror,
                                            'perror':perror,
                                            'cputime':cputime,
                                            'entropy':rel_ent}],
                                            index=[ii])
                record = pd.concat([record, record_one])

    if verbose > 1:
        print('done (total time: % 3is, % 4.1fmn)' % (cputime, cputime / 60))

    if record_each==0:
        return dictionary, precision, P_cum
    else:
        return dictionary, precision, P_cum, record


def homeostasis(mean_measure, P_cum, gain,
                homeo_method, sparse_code, eta_homeo, alpha_homeo,
                nb_quant, verbose, C, do_sym):

    # homeostasis : we compute P_cum and define different strategies
    P_cum = update_P_cum(P_cum, sparse_code, eta_homeo,
                         nb_quant=nb_quant, verbose=verbose, C=C, do_sym=do_sym)

    n_dictionary = sparse_code.shape[1]
    # compute statistics on the activation probability
    if mean_measure is None:
        mean_measure = update_measure(np.zeros(n_dictionary), sparse_code,
                                        eta_homeo=1., verbose=verbose,
                                        do_HAP=True)
    else:
        mean_measure = update_measure(mean_measure, sparse_code, eta_homeo,
                                      verbose=verbose, do_HAP=True)

    if homeo_method=='None':
        # do nothing
        assert(gain[0]==1.)

    elif homeo_method=='HEH':
        assert(gain is None)

    elif homeo_method in ['EXP', 'HAP', 'eHAP', 'EMP']:
        target = (sparse_code>0).mean()
        # apply different heuristics on the gain
        if homeo_method=='EXP':
            # cf. https://github.com/VictorBoutin/CHAMP/blob/caa2a77cc65d0043db1aeb11eedb707633a93df4/CHAMP/CHAMP_Layer.py#L365
            gain = np.exp(-(1 / alpha_homeo) * (mean_measure-target))
        elif homeo_method=='HAP':
            gain = np.log(mean_measure)/np.log(target)
        elif homeo_method=='eHAP':
            gain = (mean_measure/target)**(-alpha_homeo)#
            # gain /= gain.mean()
        elif homeo_method=='EMP':
            p_threshold = target*(1+alpha_homeo)
            gain = 1. * (mean_measure < p_threshold)

    elif homeo_method=='OLS':
        # compute statistics on the variance of coefficients
        if mean_measure is None:
            mean_measure = update_measure(np.zeros(n_dictionary), sparse_code,
                                          eta_homeo=1., verbose=verbose,
                                          do_HAP=False)
        else:
            mean_measure = update_measure(mean_measure, sparse_code, eta_homeo,
                                            verbose=verbose, do_HAP=False)
        # apply heuristics on the gain
        gain = mean_measure**(-alpha_homeo)

    else:
        raise ValueError('Homeostasis method must be "EXP", "None", "HAP", "Olshausen" '
                         '"EMP" or "HEH", got %s.'
                         % homeo_method)

    return mean_measure, P_cum, gain

def update_measure(mean_measure, code, eta_homeo, verbose=False, do_HAP=False):
    """Update the estimated statistics of coefficients in place.

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
    mean_measure: array of shape (n_dictionary)
        Value of the dictionary' norm at the previous iteration.

    code: array of shape (n_dictionary, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    eta_homeo: float
        Gives the learning parameter for the gain.

    verbose:
        Degree of output the procedure will print.

    do_HAP: boolean
        Switch to compute the variance (if False) or the activation probability (if True)

    Returns
    -------
    gain: array of shape (n_dictionary)
        Updated value of the dictionary' norm.

    """
    if code.ndim==1:
        code = code[:, np.newaxis]
    if eta_homeo>0.:
        if not do_HAP:
            measure_ = np.mean(code**2, axis=0)
            mean_measure_ = measure_ / measure_.mean()
        else:
            measure_ = 1. * np.count_nonzero(code, axis=0)
            mean_measure_ = measure_ / code.shape[0]
        mean_measure = (1 - eta_homeo)*mean_measure + eta_homeo * mean_measure_

    return mean_measure


def update_P_cum(P_cum, code, eta_homeo, C, nb_quant=100, do_sym=False, verbose=False):
    """Update the estimated modulation function in place.

    Parameters
    ----------
    P_cum: array of shape (n_samples, n_components)
        Value of the modulation function at the previous iteration.

    dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    code: array of shape (n_samples, n_features)
        Data matrix.

    eta_homeo: float
        Gives the learning parameter for the mod.

    verbose:
        Degree of output the procedure will print.

    Returns
    -------
    P_cum: array of shape (n_samples, n_components)
        Updated value of the modulation function.

    """
    if eta_homeo>0.:
        P_cum_ = get_P_cum(code, nb_quant=nb_quant, C=C, do_sym=do_sym, verbose=verbose)
        P_cum = (1 - eta_homeo)*P_cum + eta_homeo * P_cum_
    else:
        print('dooh, eta_homeo is nil')
    return P_cum

def get_P_cum(code, C, nb_quant=256, do_sym=False, verbose=False):
    from shl_scripts.shl_encode import rescaling
    p_c = rescaling(code, C, do_sym=do_sym, verbose=verbose)

    n_samples, nb_filter = code.shape
    code_bins = np.linspace(0., 1., nb_quant, endpoint=True)
    P_cum = np.zeros((nb_filter, nb_quant))

    for i in range(nb_filter):
        # note: the last bin includes 1, while this value is impossible (pc==1 <=> C=infinite)
        p, bins = np.histogram(p_c[:, i], bins=code_bins, density=True)
        p /= p.sum()
        P_cum[i, :] = np.hstack((0, np.cumsum(p)))
    return P_cum
