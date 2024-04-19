#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
from shl_scripts import get_data, touch
from shl_scripts import sparse_encode

"""

========================================================
Learning filters from natural images using sparse coding
========================================================

* When imposing a code representing patches from natural images to be sparse,
one observes the formation of filters resembling the receptive field of simple
cells in primates primary visual cortex. This was first proposed in the
framework of the SparseNet algorithm from Bruno Olshausen
(http://redwood.berkeley.edu/bruno/sparsenet/).

* This particular implementation has been published as Perrinet, Neural
Computation (2010) (see https://laurentperrinet.github.io/publication/perrinet-10-shl )::

   @article{Perrinet10shl,
        Author = {Perrinet, Laurent U.},
        Title = {Role of homeostasis in learning sparse representations},
        Year = {2010}
        Url = {https://laurentperrinet.github.io/publication/perrinet-10-shl},
        Doi = {10.1162/neco.2010.05-08-795},
        Journal = {Neural Computation},
        Volume = {22},
        Number = {7},
        Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images,
        Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit,
        Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
        Month = {July},
        }


"""

import time
toolbar_width = 40
import numpy as np

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

import os
home = os.environ['HOME']

class SHL(object):
    """

    Base class to define SHL experiments:
        - initialization
        - coding and learning
        - visualization
        - quantitative analysis

    """
    def __init__(self,
                 height=256, # of image
                 width=256, # of image
                 patch_width=21,
                 N_patches=2**16,
                 datapath='../database/',
                 name_database='kodakdb', # TODO : fing a larger, more homogeneous database?
                 #name_database='laurent',
                 do_mask=True, do_bandpass=True,
                 over_patches=16,
                 patch_ds=1,
                 n_dictionary=26**2,
                 learning_algorithm='mp',
                 fit_tol=None,
                 l0_sparseness=21,
                 alpha_MP=.95,
                 one_over_F=True,
                 n_iter=2**12 + 1,
                 eta=0.02, beta1=.990, beta2=.990, epsilon=10,
                 do_precision=False, eta_precision=0.000,
                 homeo_method='HAP',
                 eta_homeo=0.01, alpha_homeo=.05,
                 C=3., nb_quant=128, P_cum=None,
                 do_sym=False,
                 seed=42,
                 patch_norm=False,
                 batch_size=2**12,
                 record_each=2**5,
                 record_num_batches=2**10,
                 n_image=None,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 cache_dir='cache_dir',
                 n_jobs=1,
                ):
        self.height = height
        self.width = width
        self.patch_width = patch_width
        self.patch_ds = patch_ds
        self.N_patches = int(N_patches/DEBUG_DOWNSCALE)
        self.datapath = datapath
        self.name_database = name_database
        self.n_dictionary = n_dictionary
        self.learning_algorithm = learning_algorithm
        self.n_iter = int(n_iter/DEBUG_DOWNSCALE)
        self.seed = seed
        self.patch_norm = patch_norm
        if not n_image is None:
            self.n_image = int(n_image/DEBUG_DOWNSCALE)
        else:
            self.n_image = None
        self.batch_size = batch_size
        self.fit_tol = fit_tol
        self.do_mask = do_mask
        self.do_bandpass = do_bandpass
        self.over_patches = over_patches

        self.l0_sparseness = l0_sparseness
        self.alpha_MP = alpha_MP
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_precision = do_precision
        self.eta_precision = eta_precision
        self.homeo_method = homeo_method
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo
        self.C = C
        self.nb_quant = nb_quant
        self.P_cum = P_cum

        self.do_sym = do_sym
        self.record_each = int(record_each/DEBUG_DOWNSCALE)
        self.record_num_batches = record_num_batches
        self.verbose = verbose
        # assigning and create a folder for caching data
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs

        if not self.cache_dir is None:
            try:
                os.mkdir(self.cache_dir)
            except:
                pass

        self.one_over_F = one_over_F

        # creating a tag related to this process
        PID, HOST = os.getpid(), os.uname()[1]
        self.LOCK = '_lock' + '_pid-' + str(PID) + '_host-' + HOST

    def get_data(self, matname=None, patch_width=None):
        if patch_width is None: patch_width= self.patch_width
        from shl_scripts.shl_tools import get_data
        return get_data(height=self.height, width=self.width, n_image=self.n_image,
                    patch_size=(patch_width, patch_width), patch_ds=self.patch_ds, datapath=self.datapath,
                    N_patches=self.N_patches, verbose=self.verbose,
                    cache_dir=self.cache_dir, seed=self.seed, do_bandpass=self.do_bandpass,
                    do_mask=self.do_mask, over_patches = self.over_patches, patch_norm=self.patch_norm,
                    name_database=self.name_database, matname=matname)


    def code(self, data, dico, coding_algorithm='mp', matname=None, P_cum=None, fit_tol=None, l0_sparseness=None, gain=None):
        if gain is None:
            gain = np.ones(self.n_dictionary)
        if l0_sparseness is None:
            l0_sparseness = self.l0_sparseness
        if matname is None:
            if self.verbose:
                print('Coding data with algorithm ', coding_algorithm,  end=' ')
                t0 = time.time()
            from shl_scripts.shl_encode import sparse_encode

            sparse_code = sparse_encode(data, dico.dictionary, dico.precision,
                                        fit_tol=fit_tol,
                                        l0_sparseness=l0_sparseness, alpha_MP=self.alpha_MP,
                                        algorithm=self.learning_algorithm,
                                        P_cum=P_cum, do_sym=self.do_sym, verbose=0,
                                        gain=gain)

        else:
            fmatname = os.path.join(self.cache_dir, matname) + '_coding.npy'
            if not(os.path.isfile(fmatname)):
                if not(os.path.isfile(fmatname + '_lock')):
                    touch(fmatname + '_lock')
                    touch(fmatname + self.LOCK)
                    if self.verbose: print('No cache found {}: Coding with algo = {} \n'.format(fmatname, self.learning_algorithm), end=' ')
                    sparse_code = self.code(data, dico,
                                            fit_tol=fit_tol, l0_sparseness=l0_sparseness, matname=None)
                    np.save(fmatname, sparse_code)
                    try:
                        os.remove(fmatname + self.LOCK)
                        os.remove(fmatname + '_lock')
                    except:
                        print('Coud not remove ', fmatname + self.LOCK)
                else:
                    print('the computation is locked', fmatname + self.LOCK)
            else:
                if self.verbose: print("loading the code called : {0}".format(fmatname))
                sparse_code = np.load(fmatname)

        return sparse_code

    def decode(self, sparse_code, dico):
        return sparse_code @ dico.dictionary

    def learn_dico(self, dictionary=None, precision=None, P_cum=None,
                   data=None, matname=None, record_each=None, folder_exp=None,
                   list_figures=[], fig_kwargs={'fig':None, 'ax':None}):

        if data is None: data = self.get_data(matname=matname)

        if matname is None:
            # Learn the dictionary from reference patches
            t0 = time.time()
            from shl_scripts.shl_learn import SparseHebbianLearning
            dico = SparseHebbianLearning(fit_algorithm=self.learning_algorithm,
                                         dictionary=dictionary, precision=precision,
                                         do_sym=self.do_sym,
                                         n_dictionary=self.n_dictionary,
                                         eta=self.eta,
                                         beta1=self.beta1,
                                         beta2=self.beta2,
                                         epsilon=self.epsilon,
                                         do_precision=self.do_precision,
                                         eta_precision=self.eta_precision,
                                         homeo_method=self.homeo_method,
                                         eta_homeo=self.eta_homeo,
                                         alpha_homeo=self.alpha_homeo,
                                         C=self.C,
                                         nb_quant=self.nb_quant,
                                         P_cum=P_cum,
                                         n_iter=self.n_iter,
                                         l0_sparseness=self.l0_sparseness,
                                         alpha_MP=self.alpha_MP,
                                         one_over_F=self.one_over_F,
                                         batch_size=self.batch_size,
                                         verbose=self.verbose,
                                         fit_tol=self.fit_tol,
                                         record_each=self.record_each,
                                         #record_snapshot=self.record_snapshot,
                                         record_num_batches=self.record_num_batches,
)

            if self.verbose: print('Training on %d patches' % len(data))
            dico.fit(data)

            if self.verbose:
                dt = time.time() - t0
                print('done in %.2fs.' % dt)

        else:
            dico = 'lock'
            fmatname = os.path.join(self.cache_dir, matname) + '_dico.pkl'
            import pickle
            if not(os.path.isfile(fmatname)):
                time.sleep(np.random.rand()*0.01)
                if not(os.path.isfile(fmatname + '_lock')):
                    touch(fmatname + '_lock')
                    touch(fmatname + self.LOCK)
                    try:
                        if self.verbose != 0 :
                            print('No cache found {}: Learning the dictionary with algo = {} \n'.format(fmatname, self.learning_algorithm), end=' ')
                            
                            if not dictionary is None or not P_cum is None:
                                print("resuming the learning on : {0}".format(fmatname))

                        dico = self.learn_dico(data=data, dictionary=dictionary, precision=precision, P_cum=P_cum,
                                               matname=None)
                        with open(fmatname, 'wb') as fp:
                            pickle.dump(dico, fp)
                    except ImportError as e:
                        print('Error', e)
                    finally:
                        try:
                            os.remove(fmatname + self.LOCK)
                            os.remove(fmatname + '_lock')
                        except:
                            if self.verbose: print('Coud not remove ', fmatname + self.LOCK)
                else:
                    dico = 'lock'
                    if self.verbose: print('the computation is locked', fmatname + self.LOCK)
            else:
                if self.verbose: print("loading the dico called : {0}".format(fmatname))
                # Une seule fois mp ici
                with open(fmatname, 'rb') as fp:
                    dico = pickle.load(fp)

        if not dico == 'lock':
            if 'show_dico' in list_figures:
                fig, ax = self.show_dico(dico, data=data, title=matname, **fig_kwargs)
            if 'show_Pcum' in list_figures:
                fig, ax = self.show_Pcum(dico, **fig_kwargs)
            if 'plot_error' in list_figures:
                fig, ax = self.plot_error(dico)
            if 'show_dico_in_order' in list_figures:
                fig,ax = self.show_dico_in_order(dico, data=data, title=matname, **fig_kwargs)
            if 'plot_variance' in list_figures:
                sparse_code = self.code(data, dico,
                                            fit_tol=self.fit_tol, l0_sparseness=self.l0_sparseness, matname=matname)
                fig, ax = self.plot_variance(sparse_code, **fig_kwargs)
            if 'plot_variance_histogram' in list_figures:
                sparse_code = self.code(data, dico, matname=matname)
                fig, ax = self.plot_variance_histogram(sparse_code, **fig_kwargs)
            if 'time_plot_var' in list_figures:
                fig, ax = self.time_plot(dico, variable='var', **fig_kwargs)
            if 'time_plot_kurt' in list_figures:
                fig, ax = self.time_plot(dico, variable='kurt', **fig_kwargs)
            if 'time_plot_prob' in list_figures:
                fig, ax = self.time_plot(dico, variable='prob_active', **fig_kwargs)
            if 'time_plot_error' in list_figures:
                fig, ax = self.time_plot(dico, variable='error', **fig_kwargs)
            if 'time_plot_qerror' in list_figures:
                fig, ax = self.time_plot(dico, variable='qerror', **fig_kwargs)
            if 'time_plot_perror' in list_figures:
                fig, ax = self.time_plot(dico, variable='perror', **fig_kwargs)
            if 'time_plot_aerror' in list_figures:
                fig, ax = self.time_plot(dico, variable='aerror', **fig_kwargs)
            if 'time_plot_entropy' in list_figures:
                fig, ax = self.time_plot(dico, variable='entropy', **fig_kwargs)
            if 'time_plot_logL' in list_figures:
                fig, ax = self.time_plot(dico, variable='logL', **fig_kwargs)
            if 'time_plot_MC' in list_figures:
                fig, ax = self.time_plot(dico, variable='MC', **fig_kwargs)
            try:
                #if fname is None:
                fig.show()
            except:
                pass

        return dico

    def plot_variance(self, sparse_code, **fig_kwargs):
        from shl_scripts.shl_tools import plot_variance
        return plot_variance(self, sparse_code, **fig_kwargs)

    def plot_variance_histogram(self, sparse_code, **fig_kwargs):
        from shl_scripts.shl_tools import plot_variance_histogram
        return plot_variance_histogram(self, sparse_code, **fig_kwargs)

    def time_plot(self, dico, variable='kurt', N_nosample=1, **fig_kwargs):
        from shl_scripts.shl_tools import time_plot
        return time_plot(self, dico, variable=variable, N_nosample=N_nosample, **fig_kwargs)

    def show_dico(self, dico, data=None, title=None, **fig_kwargs):
        from shl_scripts.shl_tools import show_dico
        return show_dico(self, dico=dico, data=data, title=title, **fig_kwargs)

    def show_dico_in_order(self, dico, data=None, title=None, **fig_kwargs):
        from shl_scripts.shl_tools import show_dico_in_order
        return show_dico_in_order(self, dico=dico, data=data, title=title, **fig_kwargs)

    def show_Pcum(self, dico, title=None, verbose=False, n_yticks=21, alpha=.05, c='g', **fig_kwargs):
        from shl_scripts.shl_tools import plot_P_cum
        ymin = 1 - 1.5 * self.l0_sparseness/self.n_dictionary
        return plot_P_cum(dico.P_cum, ymin=ymin, title=title, verbose=verbose, n_yticks=n_yticks, alpha=alpha, c=c, **fig_kwargs)

    def plot_error(self, dico, **fig_kwargs):
        from shl_scripts.shl_tools import plot_error
        return plot_error(dico, **fig_kwargs)


from copy import deepcopy
from shl_scripts import get_record

class SHL_set(object):
    """

    Base class to define a set of SHL experiments:
        - initialization
        - run: coding and learning
        - scan: visualization
        - quantitative analysis

    """
    def __init__(self, opts, tag='default', data_matname='data', base=4., N_scan=9, do_run=True):
        self.opts = deepcopy(opts)
        self.tag = tag
        self.N_scan = N_scan
        self.base = base
        self.shl = SHL(**deepcopy(self.opts))
        self.data = self.shl.get_data(matname=data_matname)
        self.do_run = do_run

    def matname(self, variable, value):
        value = check_type(variable, value)
        if not isinstance(value, int):
            label = '%.5f' % value
        else:
            label = '%d' % value
        return  self.tag + f'_{variable}={label}'

    def get_values(self, variable, median, N_scan, verbose=0):
        if variable is 'alpha_MP':
            values = np.logspace(-1., 0., N_scan, base=self.base, endpoint=True)
        elif variable in ['beta1', 'beta2']:
            values = 1. - np.logspace(-1, 1., N_scan, base=self.base)*(1-median)
        elif variable in ['seed']:
            values = median + np.arange(N_scan)
        else:
            values = np.logspace(-1., 1., N_scan, base=self.base)*median
        values = [check_type(variable, value) for value in values]
        if verbose>1: print('DEBUG: variable, median, values = ', variable, median, values)
        return values

    def run(self, N_scan=None, variables=['eta'], 
            list_figures=[], fig_kwargs={}, verbose=0):
        # defining  the range of the scan
        if N_scan is None: N_scan = self.N_scan

        # gather all tuples
        variables_, values_ = [], np.zeros(N_scan*len(variables))
        for i, variable in enumerate(variables):
            variables_.extend([variable] * N_scan)
            values = self.get_values(variable, self.shl.__dict__[variable], N_scan, verbose=verbose)
            values_[(i*N_scan):((i+1)*N_scan)] = values

        if self.n_jobs == 1:
            for variable, value in zip(variables_, values_):
                shl = prun(variable, value, self.data, self.opts,
                            self.matname(variable, value), list_figures, fig_kwargs, verbose)
                dico = shl.learn_dico(data=self.data,
                            matname=self.matname(variable, value),
                            list_figures=list_figures, fig_kwargs=fig_kwargs)
                # print('shape =', dico.)
        else:
            # We will use the ``joblib`` package do distribute this computation on different CPUs.
            from joblib import Parallel, delayed
            # , backend="threading"
            Parallel(n_jobs=self.n_jobs, verbose=15)(delayed(prun)(variable, value, self.data, self.opts, self.matname(variable, value), list_figures, fig_kwargs, verbose) for (variable, value) in zip(variables_, values_))

    def scan(self, N_scan=None, variable='eta', list_figures=[],
                display='', display_variable='logL',
                alpha=.6, color=None, label=None, fname=None,
                fig=None, ax=None, fig_kwargs={}, verbose=0):
        # defining  the range of the scan
        if N_scan is None: N_scan = self.N_scan
        # running all jobs (run the self.run function before to perform multi-processing)
        self.run(N_scan=N_scan, variables=[variable], verbose=0, fig_kwargs=fig_kwargs)

        if display == 'dynamic':
            import matplotlib.pyplot as plt
            fig_error, ax_error = None, None
        elif display == 'final':
            import matplotlib.pyplot as plt
            results = []
            if fig is None:
                fig = plt.figure(figsize=(16, 4))
            if ax is None:
                ax = fig.add_subplot(111)

        values = self.get_values(variable, self.shl.__dict__[variable], N_scan, verbose=verbose)
        for value in values:
            # one run to retrieve the SHL object
            shl = prun(variable, value, self.data, self.opts, self.matname(variable, value), [], {}, verbose)
            # now performing all computations on that object
            dico = shl.learn_dico(data=self.data, matname=self.matname(variable, value),
                            list_figures=list_figures, fig_kwargs=fig_kwargs)
            # using that object to perform quantitative visualizations.
            if display == 'dynamic':
                if not isinstance(value, int):
                    label = '%s=%.4f' % (variable, value)
                else:
                    label = '%s=%d' % (variable, value)
                try:
                    fig_error, ax_error = shl.time_plot(dico, variable=display_variable,
                        fig=fig_error, ax=ax_error, label=label)
                except Exception as e:
                    print('While doing the time plot for ', self.matname(variable, value), self.shl.LOCK)
                    print('We encountered error', e, ' with', dico)
            elif display == 'final':
                try:
                    learning_time, A = get_record(dico, display_variable, 0)
                    results.append(A[-1])
                except Exception as e:
                    print('While processing ', self.matname(variable, value), self.shl.LOCK)
                    print('We encountered error', e, ' with', dico)
                    results.append(np.nan)
            else:
                if len(list_figures)>0:
                    import matplotlib.pyplot as plt
                    plt.show()

            del shl

        if display == 'dynamic':
            ax_error.legend()
            if display_variable in ['aerror']:
                ax_error.set_ylim(0)
            # if display_variable in ['error', 'qerror']:
            #     ax_error.set_ylim(0, 1)
            # elif display_variable in ['perror']:
            #     ax_error.set_ylim(1.0)
            # elif display_variable in ['cputime']:
            #     ax_error.set_yscale('log')
            return fig_error, ax_error
        elif display == 'final':
            try:
                ax.plot(values, results, '-', lw=1, alpha=alpha, color=color, label=label)
                ax.set_ylabel(display_variable)
                ax.set_xlabel(variable)
                # ax.set_xlim(values.min(), values.max())
                if display_variable in ['error']:
                    ax.set_ylim(0, 1)
                # elif display_variable in ['perror']:
                #     ax.set_ylim(1.0)
                elif display_variable in ['cputime']:
                    #ax.axis('tight')
                    #ax.set_yscale('log')
                    ax.set_ylim(0)
                ax.set_xscale('log')
            except Exception as e:
                print('While processing ', self.matname(variable, value), self.shl.LOCK)
                print('We encountered error', e, ' with', dico)
            return fig, ax

def check_type(variable, value):
    if variable in ['seed', 'n_iter', 'nb_quant', 'l0_sparseness', 'patch_width', 'n_dictionary', 'batch_size']:
        value = int(value)
    return value

def prun(variable, value, data, opts, matname, list_figures, fig_kwargs, verbose):
    if isinstance(value, int):
        value_str = str(value)
    else:
        value_str = '%.4f' % value
    if verbose: print('Running variable', variable, 'with value', value_str)
    value = check_type(variable, value)

    shl = SHL(**deepcopy(opts))
    shl.__dict__[variable] = value
    # if verbose: print('DEBUG:', shl.__dict__, self.shl.__dict__)
    if variable in ['patch_width']:
        data = shl.get_data(**{variable:value})
    dico = shl.learn_dico(data=data, matname=matname,
                list_figures=list_figures, fig_kwargs=fig_kwargs)
    return shl

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 10

    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', homeo_method='HAP', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt
    fig, ax = shl.show_dico(dico, order=False)
    plt.savefig('../probe/shl_HAP.png')
    fig, ax = shl.show_Pcum(dico)
    plt.savefig('../probe/shl_HAP_Pcum.png')

    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', homeo_method='HEH', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt
    fig, ax = shl.show_dico(dico, order=False)
    plt.savefig('../probe/shl_HEH.png')
    fig, ax = shl.show_Pcum(dico)
    plt.savefig('../probe/shl_HEH_Pcum.png')

    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', homeo_method='None', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt
    fig, ax = shl.show_dico(dico, order=False)
    plt.savefig('../probe/shl_nohomeo.png')
    fig, ax = shl.show_Pcum(dico)
    plt.savefig('../probe/shl_nohomeo_Pcum.png')
