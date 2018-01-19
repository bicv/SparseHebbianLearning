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
                 patch_size=(16, 16),
                 datapath='database/',
                 name_database='kodakdb',
                 n_dictionary=24**2,
                 learning_algorithm='mp',
                 fit_tol=None,
                 do_precision=False,
                 do_mask=True,
                 l0_sparseness=50,
                 l0_sparseness_end=None,
                 one_over_F=True,
                 n_iter=2**12,
                 # Standard
                 #eta=.01, # or equivalently
                 eta = dict(eta=.05, beta1=0),
                 # ADAM https://arxiv.org/pdf/1412.6980.pdf
                 #eta=dict(eta=.002, beta1=.9, beta2=.999, epsilon=1.e-8),
                 homeo_method='HAP',
                 homeo_params=dict(eta_homeo=0.05, alpha_homeo=0.02),
                 do_sym=False,
                 max_patches=4096,
                 seed=42,
                 patch_norm=False,
                 batch_size=512,
                 record_each=128,
                 n_image=None,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 data_cache='/tmp/data_cache',
                ):
        self.height = height
        self.width = width
        self.datapath = datapath
        self.patch_size = patch_size
        self.n_dictionary = n_dictionary
        self.n_iter = int(n_iter/DEBUG_DOWNSCALE)
        self.max_patches = int(max_patches/DEBUG_DOWNSCALE)
        self.name_database = name_database #[0],
        self.seed = seed,
        self.patch_norm = patch_norm,
        if not n_image is None:
            self.n_image = int(n_image/DEBUG_DOWNSCALE)
        else:
            self.n_image = None
        self.batch_size = batch_size
        self.learning_algorithm = learning_algorithm
        self.fit_tol = fit_tol
        self.do_precision = do_precision
        self.do_mask = do_mask

        self.l0_sparseness = l0_sparseness
        self.l0_sparseness_end = l0_sparseness_end
        self.eta = eta
        self.do_sym = do_sym
        self.record_each = int(record_each/DEBUG_DOWNSCALE)
        self.verbose = verbose
        # assigning and create a folder for caching data
        self.data_cache = data_cache
        self.homeo_method = homeo_method
        self.homeo_params = homeo_params
        self.one_over_F = one_over_F

        if not self.data_cache is None:
            try:
                os.mkdir(self.data_cache)
            except:
                pass

        # creating a tag related to this process
        PID, HOST = os.getpid(), os.uname()[1]
        self.LOCK = '_lock' + '_pid-' + str(PID) + '_host-' + HOST

    def get_data(self, matname=None):
        from shl_scripts.shl_tools import get_data
        return get_data(height=self.height, width=self.width, n_image=self.n_image,
                    patch_size=self.patch_size, datapath=self.datapath,
                    max_patches=self.max_patches, verbose=self.verbose,
                    data_cache=self.data_cache, seed=self.seed,
                    do_mask=self.do_mask, patch_norm=self.patch_norm,
                    name_database=self.name_database, matname=matname)


    def code(self, data, dico, coding_algorithm='mp', matname=None, fit_tol=None, l0_sparseness=None):
        if l0_sparseness is None:
            l0_sparseness = self.l0_sparseness
        if matname is None:
            if self.verbose:
                print('Coding data with algorithm ', coding_algorithm,  end=' ')
                t0 = time.time()
            from shl_scripts.shl_encode import sparse_encode
            if 'C' in self.homeo_params.keys():
                C = self.homeo_params['C']
            else:
                C = 0.
            if 'P_cum' in self.homeo_params.keys():
                P_cum = self.homeo_params['P_cum']
            else:
                P_cum = None

            if self.l0_sparseness_end is not None:
                l0_sparseness = self.l0_sparseness_end
            else:
                l0_sparseness = self.l0_sparseness

            sparse_code = sparse_encode(data, dico.dictionary, dico.precision,
                                        fit_tol=fit_tol,
                                        l0_sparseness=l0_sparseness,
                                        algorithm=self.learning_algorithm,
                                        C=C, P_cum=P_cum, do_sym=self.do_sym, verbose=0, gain=None)
            if self.verbose:
                dt = time.time() - t0
                print('done in %.2fs.' % dt)
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
                                         eta=self.eta, n_iter=self.n_iter,
                                         l0_sparseness=self.l0_sparseness, one_over_F=self.one_over_F,
                                         l0_sparseness_end=self.l0_sparseness_end,
                                         batch_size=self.batch_size, verbose=self.verbose,
                                         fit_tol=self.fit_tol,
                                         do_precision=self.do_precision, record_each=self.record_each,
                                         homeo_method=self.homeo_method, homeo_params=self.homeo_params)

            if self.verbose: print('Training on %d patches' % len(data), end='... ')
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
                    except ImportError: #Exception as e:
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
            if 'time_plot_entropy' in list_figures:
                fig, ax = self.time_plot(dico, variable='entropy', fname=fname)
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

    def plot_error(self, dico, fig=None, ax=None):
        from shl_scripts.shl_tools import plot_error
        return plot_error(dico, fig=fig, ax=ax)

    def show_dico_in_order(self, dico, data=None, title=None, fname=None, dpi=200, fig=None, ax=None):
        from shl_scripts.shl_tools import show_dico_in_order
        return show_dico_in_order(self, dico=dico, data=data, title=title, fname=fname, dpi=dpi, fig=fig, ax=ax)

if __name__ == '__main__':

    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 0
    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt

    fig, ax = dico.show_dico()
    plt.savefig('../probe/assc.png')
    plt.show()
