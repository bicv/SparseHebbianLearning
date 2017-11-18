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
import os

toolbar_width = 40

# import matplotlib

import numpy as np
# see https://github.com/bicv/SLIP/blob/master/SLIP.ipynb
from SLIP import Image

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
                 n_dictionary=18**2,
                 learning_algorithm='mp',
                 fit_tol=None,
                 l0_sparseness=15,
                 n_iter=2**14,
                 eta=.025,
                 eta_homeo=.01, nb_quant=128, C=0., do_sym=False,
                 alpha_homeo=0.,
                 max_patches=4096,
                 name_database='kodakdb', seed=None, patch_norm=True,
                 batch_size=128,
                 record_each=128,
                 n_image=200,
                 DEBUG_DOWNSCALE=1, # set to 10 to perform a rapid experiment
                 verbose=0,
                 data_cache=os.path.join(home, 'tmp/data_cache'),
                 ):
        self.height = height
        self.width = width
        self.datapath = datapath
        self.patch_size = patch_size
        self.n_dictionary = n_dictionary
        self.n_iter = int(n_iter/DEBUG_DOWNSCALE)
        self.max_patches = int(max_patches/DEBUG_DOWNSCALE)
        self.name_database=name_database,
        self.seed=seed,
        self.patch_norm=patch_norm,
        self.n_image = int(n_image/DEBUG_DOWNSCALE)
        self.batch_size = batch_size
        self.learning_algorithm = learning_algorithm
        self.fit_tol = fit_tol

        self.l0_sparseness = l0_sparseness
        self.eta = eta
        self.eta_homeo = eta_homeo
        self.alpha_homeo = alpha_homeo
        self.nb_quant = nb_quant
        self.C = C
        self.do_sym = do_sym

        self.record_each = int(record_each/DEBUG_DOWNSCALE)
        self.verbose = verbose
        # assigning and create a folder for caching data
        self.data_cache = data_cache
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
        # height=256, width=256, n_image=200, patch_size=(12,12),
        #     datapath='database/', name_database='serre07_distractors',
        #     max_patches=1024, seed=None, patch_norm=True, verbose=0,
        #     data_cache='/tmp/data_cache', matname=None
        return get_data(height=self.height, width=self.width, n_image=self.n_image,
                    patch_size=self.patch_size, datapath=self.datapath,
                    max_patches=self.max_patches, verbose=self.verbose,
                    data_cache=self.data_cache, seed=seed, patch_norm=patch_norm,
                        seed=self.seed,
                        patch_norm=self.patch_norm,
                        name_database=self.name_database, matname=matname)


    def code(self, data, dico, coding_algorithm='mp', matname=None, l0_sparseness=None):
        if l0_sparseness is None:
            l0_sparseness = self.l0_sparseness
        if matname is None:
            if self.verbose:
                print('Coding data with algorithm ', coding_algorithm,  end=' ')
                t0 = time.time()
            from shl_scripts.shl_encode import sparse_encode
            sparse_code = sparse_encode(data, dico.dictionary,
                                        algorithm=self.learning_algorithm,
                                        fit_tol=None,
                                        l0_sparseness=l0_sparseness,
                                        C=self.C, P_cum=dico.P_cum, do_sym=self.do_sym, verbose=0)
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
                    sparse_code = self.code(data, dico, matname=None)
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

    def learn_dico(self, dictionary=None, P_cum=None, data=None, name_database='serre07_distractors',
                   matname=None, record_each=None, folder_exp=None, list_figures=[], fname=None):

        if data is None: data = self.get_data(name_database, matname=matname)

        if matname is None:
            # Learn the dictionary from reference patches
            t0 = time.time()
            from shl_scripts.shl_learn import SparseHebbianLearning
            dico = SparseHebbianLearning(dictionary=dictionary, P_cum=P_cum,
                                         fit_algorithm=self.learning_algorithm,
                                         nb_quant=self.nb_quant, C=self.C, do_sym=self.do_sym,
                                         n_dictionary=self.n_dictionary, eta=self.eta, n_iter=self.n_iter,
                                         eta_homeo=self.eta_homeo, alpha_homeo=self.alpha_homeo,
                                         l0_sparseness=self.l0_sparseness,
                                         batch_size=self.batch_size, verbose=self.verbose,
                                         fit_tol=self.fit_tol,
                                         record_each=self.record_each)
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
                time.sleep(np.random.rand()*0.1)
                if not(os.path.isfile(fmatname + '_lock')):
                    touch(fmatname + '_lock')
                    touch(fmatname + self.LOCK)
                    try:
                        if self.verbose != 0 :
                            print('No cache found {}: Learning the dictionary with algo = {} \n'.format(fmatname, self.learning_algorithm), end=' ')

                        dico = self.learn_dico(data=data, dictionary=dictionary, P_cum=P_cum, name_database=name_database,
                                               record_each=self.record_each, matname=None)
                        with open(fmatname, 'wb') as fp:
                            pickle.dump(dico, fp)
                    except AttributeError:
                        print('Attribute Error')
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
                    if not (os.path.isfile(fmatname + '_lock')):
                        touch(fmatname + '_lock')
                        touch(fmatname + self.LOCK)
                        dico = self.learn_dico(data=data, dictionary=dictionary, P_cum=P_cum, name_database=name_database,
                                           record_each=self.record_each, matname=None)
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
            if 'show_dico_in_order' in list_figures:
                fig,ax = self.show_dico_in_order(dico, title=matname, fname=fname)
            if 'plot_variance' in list_figures:
                sparse_code = self.code(data, dico, matname=matname)
                fig, ax = self.plot_variance(sparse_code, data=data, fname=fname)
            if 'plot_variance_histogram' in list_figures:
                sparse_code = self.code(data, dico, matname=matname)
                fig, ax = self.plot_variance_histogram(sparse_code, data=data, fname=fname)
            if 'time_plot_var' in list_figures:
                fig, ax = self.time_plot(dico, variable='var', fname=fname);
            if 'time_plot_kurt' in list_figures:
                fig, ax = self.time_plot(dico, variable='kurt', fname=fname);
            if 'time_plot_prob' in list_figures:
                fig, ax = self.time_plot(dico, variable='prob_active', fname=fname);
            if 'time_plot_error' in list_figures:
                fig, ax = self.time_plot(dico, variable='error', fname=fname)
            if 'time_plot_entropy' in list_figures:
                fig, ax = self.time_plot(dico, variable='entropy', fname=fname)
            try:
                #if fname is None:
                fig.show()
            except:
                pass

        return dico

    def plot_variance(self, sparse_code, data=None, algorithm=None, fname=None):
        from shl_scripts.shl_tools import plot_variance
        return plot_variance(self, sparse_code, data=data, fname=fname, algorithm=algorithm)

    def plot_variance_histogram(self, sparse_code, data=None, algorithm=None, fname=None):
        from shl_scripts.shl_tools import plot_variance_histogram
        return plot_variance_histogram(self, sparse_code, data=data, fname=fname, algorithm=algorithm)

    def time_plot(self, dico, variable='kurt', fname=None, N_nosample=1):
        from shl_scripts.shl_tools import time_plot
        return time_plot(self, dico, variable=variable, fname=fname, N_nosample=N_nosample)

    def show_dico(self, dico, data=None, title=None, fname=None, dpi=200):
        from shl_scripts.shl_tools import show_dico
        return show_dico(self, dico=dico, data=data, title=title, fname=fname, dpi=dpi)

    def show_dico_in_order(self, dico, data=None, title=None, fname=None, dpi=200):
        from shl_scripts.shl_tools import show_dico_in_order
        return show_dico_in_order(self, dico=dico, data=data, title=title, fname=fname, dpi=dpi)

if __name__ == '__main__':

    DEBUG_DOWNSCALE, verbose = 10, 100 #faster, with verbose output
    DEBUG_DOWNSCALE, verbose = 1, 0
    shl = SHL(DEBUG_DOWNSCALE=DEBUG_DOWNSCALE, learning_algorithm='mp', verbose=verbose)
    dico = shl.learn_dico()
    import matplotlib.pyplot as plt

    fig, ax = dico.show_dico()
    plt.savefig('../probe/assc.png')
    plt.show()
