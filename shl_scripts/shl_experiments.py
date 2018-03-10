#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
from shl_scripts.shl_tools import get_data, touch
from shl_scripts.shl_encode import sparse_encode
from shl_scripts import shl_tools

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
                 patch_ds=1,
                 N_patches=2**16,
                 datapath='../database/',
                 name_database='kodakdb',
                 #name_database='laurent',
                 n_dictionary=23**2,
                 learning_algorithm='mp',
                 fit_tol=None,
                 do_precision=False,
                 do_mask=True, do_bandpass=True,
                 over_patches=16,
                 l0_sparseness=16,
                 one_over_F=True,
                 n_iter=2**10 + 1,
                 eta=.006, beta1=.9, beta2=.999, epsilon=1.e-8,
                 homeo_method='HAP',
                 eta_homeo=0.005, alpha_homeo=.4,
                 C=2., nb_quant=64, P_cum=None,
                 do_sym=False,
                 seed=42,
                 patch_norm=False,
                 batch_size=1024,
                 record_each=128,
                 n_image=None,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 data_cache='data_cache',
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
        self.do_precision = do_precision
        self.do_mask = do_mask
        self.do_bandpass = do_bandpass
        self.over_patches = over_patches

        self.l0_sparseness = l0_sparseness
        # self.l0_sparseness_end = l0_sparseness_end
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

        self.do_sym = do_sym
        self.record_each = int(record_each/DEBUG_DOWNSCALE)
        self.verbose = verbose
        # assigning and create a folder for caching data
        self.data_cache = data_cache

        self.one_over_F = one_over_F

        if not self.data_cache is None:
            try:
                os.mkdir(self.data_cache)
            except:
                pass

        # creating a tag related to this process
        PID, HOST = os.getpid(), os.uname()[1]
        self.LOCK = '_lock' + '_pid-' + str(PID) + '_host-' + HOST

    def get_data(self, matname=None, patch_width=None):
        if patch_width is None: patch_width= self.patch_width
        from shl_scripts.shl_tools import get_data
        return get_data(height=self.height, width=self.width, n_image=self.n_image,
                    patch_size=(patch_width, patch_width), patch_ds=self.patch_ds, datapath=self.datapath,
                    N_patches=self.N_patches, verbose=self.verbose,
                    data_cache=self.data_cache, seed=self.seed, do_bandpass=self.do_bandpass,
                    do_mask=self.do_mask, over_patches = self.over_patches, patch_norm=self.patch_norm,
                    name_database=self.name_database, matname=matname)


    def code(self, data, dico, coding_algorithm='mp', matname=None, P_cum=None, fit_tol=None, l0_sparseness=None):
        if l0_sparseness is None:
            l0_sparseness = self.l0_sparseness
        if matname is None:
            if self.verbose:
                print('Coding data with algorithm ', coding_algorithm,  end=' ')
                t0 = time.time()
            from shl_scripts.shl_encode import sparse_encode

            sparse_code = sparse_encode(data, dico.dictionary, dico.precision,
                                        fit_tol=fit_tol,
                                        l0_sparseness=l0_sparseness,
                                        algorithm=self.learning_algorithm,
                                        P_cum=None, do_sym=self.do_sym, verbose=0,
                                        gain=np.ones(self.n_dictionary))

        else:
            fmatname = os.path.join(self.data_cache, matname) + '_coding.npy'
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

    def learn_dico(self, dictionary=None, precision=None, P_cum=None, data=None,
                   matname=None, record_each=None, folder_exp=None, list_figures=[], fname=None):

        if data is None: data = self.get_data(matname=matname)

        if matname is None:
            # Learn the dictionary from reference patches
            t0 = time.time()
            from shl_scripts.shl_learn import SparseHebbianLearning
            dico = SparseHebbianLearning(dictionary=dictionary, precision=precision,
                                         fit_algorithm=self.learning_algorithm,
                                         do_sym=self.do_sym,
                                         n_dictionary=self.n_dictionary,
                                         eta=self.eta,
                                         beta1=self.beta1,
                                         beta2=self.beta2,
                                         epsilon=self.epsilon,
                                         homeo_method=self.homeo_method,
                                         eta_homeo=self.eta_homeo,
                                         alpha_homeo=self.alpha_homeo,
                                         C=self.C,
                                         nb_quant=self.nb_quant,
                                         P_cum=self.P_cum,
                                         n_iter=self.n_iter,
                                         l0_sparseness=self.l0_sparseness, one_over_F=self.one_over_F,
                                        #  l0_sparseness_end=self.l0_sparseness_end,
                                         batch_size=self.batch_size, verbose=self.verbose,
                                         fit_tol=self.fit_tol,
                                         do_precision=self.do_precision, record_each=self.record_each,
)

            if self.verbose: print('Training on %d patches' % len(data))
            dico.fit(data)

            if self.verbose:
                dt = time.time() - t0
                print('done in %.2fs.' % dt)

        else:
            dico = 'lock'
            fmatname = os.path.join(self.data_cache, matname) + '_dico.pkl'
            import pickle
            if not(os.path.isfile(fmatname)):
                time.sleep(np.random.rand()*0.01)
                if not(os.path.isfile(fmatname + '_lock')):
                    touch(fmatname + '_lock')
                    touch(fmatname + self.LOCK)
                    try:
                        if self.verbose != 0 :
                            print('No cache found {}: Learning the dictionary with algo = {} \n'.format(fmatname, self.learning_algorithm), end=' ')

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
                if not dictionary is None or not P_cum is None:
                    if self.verbose: print("resuming the learning on : {0}".format(fmatname))
                    if not (os.path.isfile(fmatname + '_lock')):
                        touch(fmatname + '_lock')
                        touch(fmatname + self.LOCK)
                        dico = self.learn_dico(data=data, dictionary=dictionary, precision=precision, P_cum=P_cum,
                                               matname=None)
                        with open(fmatname, 'wb') as fp:
                            pickle.dump(dico, fp)
                        try:
                            os.remove(fmatname + self.LOCK)
                            os.remove(fmatname + '_lock')
                        except:
                            if self.verbose: print('Coud not remove ', fmatname + self.LOCK)

        if not dico == 'lock':
            if 'show_dico' in list_figures:
                fig, ax = self.show_dico(dico, title=matname, fname=fname)
            if 'show_Pcum' in list_figures:
                fig, ax = self.show_Pcum(dico, fname=fname)
            if 'plot_error' in list_figures:
                fig, ax = self.plot_error(dico)
            if 'show_dico_in_order' in list_figures:
                fig,ax = self.show_dico_in_order(dico, title=matname, fname=fname)
            if 'plot_variance' in list_figures:
                sparse_code = self.code(data, dico,
                                            fit_tol=self.fit_tol, l0_sparseness=self.l0_sparseness, matname=matname)
                fig, ax = self.plot_variance(sparse_code, fname=fname)
            if 'plot_variance_histogram' in list_figures:
                sparse_code = self.code(data, dico, matname=matname)
                fig, ax = self.plot_variance_histogram(sparse_code, fname=fname)
            if 'time_plot_var' in list_figures:
                fig, ax = self.time_plot(dico, variable='var', fname=fname)
            if 'time_plot_kurt' in list_figures:
                fig, ax = self.time_plot(dico, variable='kurt', fname=fname)
            if 'time_plot_prob' in list_figures:
                fig, ax = self.time_plot(dico, variable='prob_active', fname=fname)
            if 'time_plot_error' in list_figures:
                fig, ax = self.time_plot(dico, variable='error', fname=fname)
            if 'time_plot_qerror' in list_figures:
                fig, ax = self.time_plot(dico, variable='qerror', fname=fname)
            if 'time_plot_perror' in list_figures:
                fig, ax = self.time_plot(dico, variable='perror', fname=fname)
            if 'time_plot_aerror' in list_figures:
                fig, ax = self.time_plot(dico, variable='aerror', fname=fname)
            if 'time_plot_entropy' in list_figures:
                fig, ax = self.time_plot(dico, variable='entropy', fname=fname)
            if 'time_plot_logL' in list_figures:
                fig, ax = self.time_plot(dico, variable='logL', fname=fname)
            try:
                #if fname is None:
                fig.show()
            except:
                pass

        return dico

    def plot_variance(self, sparse_code, fname=None, fig=None, ax=None):
        from shl_scripts.shl_tools import plot_variance
        return plot_variance(self, sparse_code, fname=fname, fig=fig, ax=ax)

    def plot_variance_histogram(self, sparse_code, fname=None, fig=None, ax=None):
        from shl_scripts.shl_tools import plot_variance_histogram
        return plot_variance_histogram(self, sparse_code, fname=fname, fig=fig, ax=ax)

    def time_plot(self, dico, variable='kurt', fname=None, N_nosample=1, color=None, label=None, fig=None, ax=None):
        from shl_scripts.shl_tools import time_plot
        return time_plot(self, dico, variable=variable, fname=fname, N_nosample=N_nosample, color=color, label=label, fig=fig, ax=ax)

    def show_dico(self, dico, data=None, title=None, fname=None, dpi=200, fig=None, ax=None):
        from shl_scripts.shl_tools import show_dico
        return show_dico(self, dico=dico, data=data, title=title, fname=fname, dpi=dpi, fig=fig, ax=ax)

    def show_Pcum(self, dico, title=None, fname=None, verbose=False, n_yticks=21, alpha=.05, c='g', fig=None, ax=None):
        from shl_scripts.shl_tools import plot_P_cum
        ymin = 1 - 1.5 * self.l0_sparseness/self.n_dictionary
        return plot_P_cum(dico.P_cum, ymin=ymin, title=title, verbose=verbose, n_yticks=n_yticks, alpha=alpha, c=c, fig=None, ax=None)

    def plot_error(self, dico, fig=None, ax=None):
        from shl_scripts.shl_tools import plot_error
        return plot_error(dico, fig=fig, ax=ax)

    def show_dico_in_order(self, dico, data=None, title=None, fname=None, dpi=200, fig=None, ax=None):
        from shl_scripts.shl_tools import show_dico_in_order
        return show_dico_in_order(self, dico=dico, data=data, title=title, fname=fname, dpi=dpi, fig=fig, ax=ax)

from copy import deepcopy
class SHL_set(object):
    """

    Base class to define a set of SHL experiments:
        - initialization
        - coding and learning
        - visualization
        - quantitative analysis

    """
    def __init__(self, opts, tag='default', data_matname='data', N_scan=7):
        self.opts = deepcopy(opts)
        self.tag = tag
        self.N_scan = N_scan
        self.shl = SHL(**deepcopy(self.opts))
        self.data = self.shl.get_data(matname='data')

    def matname(self, variable, value):
        value = check_type(variable, value)
        if not isinstance(value, int):
            label = '%.5f' % value
        else:
            label = '%d' % value
        return  self.tag + ' - {}={}'.format(variable, label)

    def run(self, N_scan=None, variables=['eta'], base=4, n_jobs=14, list_figures=[], verbose=0):
        # defining  the range of the scan
        if N_scan is None: N_scan = self.N_scan

        if n_jobs == 1:
            for variable in variables:
                median = self.shl.__dict__[variable]
                values = np.logspace(-1., 1., N_scan, base=base)*median
                values = [check_type(variable, value) for value in values]
                for value in values:
                    shl = run(variable, value, self.data, self.opts, self.matname(variable, value), list_figures)
                    dico = shl.learn_dico(data=self.data, matname=self.matname(variable, value),
                                list_figures=list_figures)
        else:
            variables_, values_ = [], np.zeros(N_scan*len(variables))
            for i, variable in enumerate(variables):
                variables_.extend([variable] * N_scan)
                median = self.shl.__dict__[variable]
                values_[(i*N_scan):((i+1)*N_scan)] = [check_type(variable, value) for value in np.logspace(-1., 1., N_scan, base=base)*median]

            # We will use the ``joblib`` package do distribute this computation on different CPUs.
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs, verbose=15)(delayed(run)(variable, value, self.data, self.opts, self.matname(variable, value), list_figures) for (variable, value) in zip(variables_, values_))

    def scan(self, N_scan=None, variable='eta', list_figures=[], base=4,
                display='', display_variable='logL',
                alpha=.6, color=None, label=None, fname=None, fig=None, ax=None, verbose=0):
        # defining  the range of the scan
        if N_scan is None: N_scan = self.N_scan
        median = self.shl.__dict__[variable]
        values = np.logspace(-1., 1., N_scan, base=base)*median
        values = [check_type(variable, value) for value in values]

        self.run(N_scan=N_scan, variables=[variable], base=base, n_jobs=1, verbose=verbose)

        if verbose: print('DEBUG:', variable, median, values)
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

        for value in values:
            shl = run(variable, value, self.data, self.opts, self.matname(variable, value), list_figures)
            dico = shl.learn_dico(data=self.data, matname=self.matname(variable, value),
                            list_figures=list_figures)

            if display == 'dynamic':
                if not isinstance(value, int):
                    label = '%s=%.3f' % (variable, value)
                else:
                    label = '%s=%d' % (variable, value)
                fig_error, ax_error = shl.time_plot(dico, variable=display_variable,
                        fig=fig_error, ax=ax_error, label=label)
            elif display == 'final':
                try:
                    # print (dico.record['cputime'])
                    df_variable = dico.record[display_variable]
                    # learning_time = np.array(df_variable.index)
                    results.append(df_variable[df_variable.index[-1]])
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
            if display_variable in ['error', 'qerror']:
                ax_error.set_ylim(0, 1)
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
                if display_variable in ['error', 'qerror']:
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
    if variable in ['n_iter', 'nb_quant', 'l0_sparseness', 'patch_width', 'n_dictionary', 'batch_size']:
        value = int(value)
    return value

def run(variable, value, data, opts, matname, list_figures):
    value = check_type(variable, value)

    shl = SHL(**deepcopy(opts))
    shl.__dict__[variable] = value
    # if verbose: print('DEBUG:', shl.__dict__, self.shl.__dict__)
    if variable in ['patch_width']:
        data = shl.get_data(**{variable:value})

    dico = shl.learn_dico(data=data, matname=matname,
                list_figures=list_figures)
    return shl

if __name__ == '__main__':

    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 0
    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt

    fig, ax = dico.show_dico()
    plt.savefig('../probe/assc.png')
    plt.show()
