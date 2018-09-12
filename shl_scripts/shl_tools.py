#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
import time
import numpy as np
from SLIP import Image
from shl_scripts import sparse_encode

toolbar_width = 40


def touch(filename):
    open(filename, 'w').close()


def preprocessing(image, height=256, width=256, patch_size=(12, 12),
                  do_bandpass=True, seed=None):
    slip = Image({'N_X': height, 'N_Y': width,
                  'white_n_learning': 0,
                  'seed': seed,
                  'white_N': .07,
                  'white_N_0': .0,  # olshausen = 0.
                  'white_f_0': .4,  # olshausen = 0.2
                  'white_alpha': 1.4,
                  'white_steepness': 4.,
                  'do_mask': True})
    image = slip.whitening(image)
    if do_bandpass:
        # # print(2*.5*max(height, width)/max(patch_size))
        # # slip.f_mask = slip.retina(sigma=2*.5*max(height, width)/max(patch_size))
        df = .07
        slip.f_mask = (1-np.exp((slip.f-.5)/(.5*df)))*(slip.f < .5)
        # removing low frequencies
        cutoff, slope = 1./max(patch_size), 42
        # useful for debuging : slip.f_mask *= .5*(np.tanh(-slope*(slip.f-cutoff))+1)
        slip.f_mask *= .5*np.tanh(slope*(slip.f-cutoff)) + .5
        image = slip.preprocess(image)

    return image


def ovf_dictionary(n_dictionary, n_pixels, height=256, width=256, seed=None, do_bandpass=True):
    N_f = np.sqrt(n_pixels).astype(int)
    fx, fy = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    spectra = 1/np.sqrt(fx**2+fy**2)  # FIX : may be infinite!
    phase = np.random.uniform(0, 2 * np.pi, (height, width))
    image = np.real(np.fft.ifft2(np.fft.fftshift(spectra*np.exp(1j*phase))))

    image = preprocessing(image, height=height, width=height, patch_size=(N_f, N_f))

    slip = Image({'N_X': height, 'N_Y': width, 'do_mask': False})
    dictionary = slip.extract_patches_2d(image, (N_f, N_f), N_patches=n_dictionary)
    dictionary -= np.mean(dictionary)
    #dictionary /= np.std(dictionary)
    dictionary = dictionary.reshape(dictionary.shape[0], -1)

    return dictionary


def get_data(height=256, width=256, n_image=200, patch_size=(12, 12), patch_ds=1,
             datapath='database/', name_database='kodakdb', do_bandpass=True,
             N_patches=1024, seed=None, do_mask=True, patch_norm=False, verbose=0,
             cache_dir='/tmp/cache_dir', over_patches=8, matname=None):
    """
    Extract data:

    Extract from a given database composed of image of size (height, width) a
    series a random patches.

    """
    if verbose:
        import sys
        # setup toolbar
        sys.stdout.write('Extracting data...')
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
        t0 = time.time()
    if matname is None:
        # Load natural images and extract patches
        # see https://github.com/bicv/SLIP/blob/master/SLIP.ipynb
        slip = Image({'datapath': datapath, 'N_image': n_image,
                      'seed': seed, 'N_X': height, 'N_Y': width})
        slip_us = Image({'N_X': height*patch_ds, 'N_Y': width*patch_ds,
                         'datapath': datapath,
                         'do_mask': True,
                         'N_image': n_image,
                         'seed': seed})
        import os

        if do_mask:
            x, y = np.meshgrid(
                np.linspace(-1, 1, patch_size[0], endpoint=True), np.linspace(-1, 1, patch_size[1], endpoint=True))
            mask = (np.sqrt(x ** 2 + y ** 2) < 1).astype(np.float).ravel()

        imagelist = slip_us.make_imagelist(name_database=name_database)
        for filename, croparea in imagelist:
            image, filename_, croparea_ = slip_us.patch(
                name_database, filename=filename, croparea=croparea, center=False)
            if patch_ds > 1:
                from skimage.measure import block_reduce
                image = block_reduce(image, block_size=(patch_ds, patch_ds), func=np.mean)

            # whitening
            image = preprocessing(image, height=height, width=width,
                                  patch_size=patch_size, do_bandpass=do_bandpass)

            # Extract all reference patches and ravel them
            data_ = slip.extract_patches_2d(
                image, patch_size, N_patches=over_patches*int(N_patches/len(imagelist)))
            indices_most_energy = np.argsort(-np.std(data_, axis=(1, 2)))
            data_ = data_[indices_most_energy[:int(N_patches/len(imagelist))], :, :]

            data_ -= np.mean(data_)
            if patch_norm:
                data_ /= np.std(data_)

            data_ = data_.reshape(data_.shape[0], -1)
            if do_mask:
                data_ = data_ * mask[np.newaxis, :]

            # collect everything as a matrix
            try:
                data = np.vstack((data, data_))
            except Exception:
                data = data_.copy()
            if verbose:
                # update the bar
                sys.stdout.write(filename + ", ")
                sys.stdout.flush()

    else:
        import os
        fmatname = os.path.join(cache_dir, matname)
        if not(os.path.isfile(fmatname + '_data.npy')):
            if not(os.path.isfile(fmatname + '_data' + '_lock')):
                touch(fmatname + '_data' + '_lock')
                try:
                    if verbose:
                        print('No cache found {}: Extracting data...'.format(
                            fmatname + '_data'), end=' ')
                    data = get_data(height=height, width=width, n_image=n_image,
                                    patch_size=patch_size, datapath=datapath,
                                    name_database=name_database, N_patches=N_patches,
                                    seed=seed, patch_norm=patch_norm, verbose=verbose,
                                    matname=None)
                    np.save(fmatname + '_data.npy', data)
                finally:
                    try:
                        os.remove(fmatname + '_data' + '_lock')
                    except:
                        print('Coud not remove ', fmatname + '_data')
            else:
                print('the data extraction is locked', fmatname + '_data')
                return 'lock'
        else:
            if verbose:
                print("loading the data called : {0}".format(fmatname + '_data'))
            data = np.load(fmatname + '_data.npy')
    if verbose:
        dt = time.time() - t0
        sys.stdout.write("Data is of shape : " + str(data.shape))
        sys.stdout.write(' - done in %.2fs.' % dt)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return data


def generate_sparse_vector(N_image, l0_sparseness, nb_dico, N_boost=0,
                           K_boost=2., C_0=3., rho_coeff=.85, seed=420, do_sym=False):
    np.random.seed(seed)
    coeff = np.zeros((N_image, nb_dico))
    for i in range(N_image):
        # indices of non-zero coefficients
        ind = np.random.permutation(np.arange(nb_dico))[:l0_sparseness]
        coeff[i, ind] = C_0 * rho_coeff**(np.random.rand(l0_sparseness)*l0_sparseness)  # activities
        coeff[i, :N_boost] *= K_boost  # perturbation
        if do_sym:
            coeff[i, ind] *= np.sign(np.random.randn(l0_sparseness))

    return coeff


def get_logL(sparse_code):
    record_num_batches = sparse_code.shape[0]
    rho_hat = np.count_nonzero(sparse_code, axis=0).mean()/record_num_batches
    #rho = shl.l0_sparseness / shl.n_dictionary
    sd = np.sqrt(rho_hat*(1-rho_hat)*record_num_batches)

    measures = np.count_nonzero(sparse_code, axis=0)

    # likelihood = 1 / np.sqrt(2*np.pi) / sd *  np.exp(-.5 * (measures - rho)**2 / sd**2)
    logL = -.5 * (measures - rho_hat*record_num_batches)**2 / sd**2
    logL -= np.log(np.sqrt(2*np.pi) * sd)
    #print(np.log(np.sum(np.exp(logL))), np.log(np.sqrt(2*np.pi) * sd))
    #logL -= np.log(np.sum(np.exp(logL)))
    return logL

def get_perror(X_train, dictionary, precision, algorithm, fit_tol,
                  P_cum, gain, C, do_sym, l0_sparseness):
    # calculation of generalization error
    sparse_code_bar = sparse_encode(X_train, dictionary, precision,
                            algorithm=algorithm, fit_tol=fit_tol,
                             P_cum=P_cum, gain=gain,
                             C=C, do_sym=do_sym,
                             l0_sparseness=l0_sparseness)
    # sparse_code_bar = sparse_code_bar.T
    # np.random.shuffle(sparse_code_bar)
    # sparse_code_bar = sparse_code_bar.T
    # np.random.shuffle(sparse_code_bar)
    # sparse_code_bar = np.random.permutation(sparse_code_bar.ravel()).reshape(sparse_code_bar.shape)
    sparse_code_bar = np.random.permutation(sparse_code_bar)

    patches_bar = sparse_code_bar @ dictionary
    sparse_code_rec = sparse_encode(patches_bar, dictionary, precision,
                             algorithm=algorithm, fit_tol=fit_tol,
                             P_cum=P_cum, gain=gain,
                                C=C, do_sym=do_sym,
                             l0_sparseness=l0_sparseness)

    # thr = np.percentile(sparse_code_bar.ravel(), 100 * (1 - l0_sparseness_high/n_dictionary ), axis=0)
    # sparse_code_bar *= (sparse_code_bar > thr)

    # q = quantile(P_cum, rescaling(sparse_code_rec, C=C), stick, do_fast=False)
    # q_bar = quantile(P_cum, rescaling(sparse_code_bar, C=C), stick, do_fast=False)
    # aerror = np.mean(np.abs(q_bar-q))
    perror = np.mean((sparse_code_bar > 0) == (sparse_code_rec > 0))
    perror = 1. - perror
    return perror

def get_MI(sparse_code):
    b_ij = (sparse_code[:, np.newaxis, :] > 0) * (sparse_code[:, :, np.newaxis] > 0)
    P_ij = 1.*np.count_nonzero(b_ij, axis=0)
    P_ij /= P_ij.sum()
    P_i = 1.*np.count_nonzero(sparse_code, axis=0)
    P_i /= P_i.sum()
    MI_ = - P_ij * np.log(P_i[:, np.newaxis]*P_i[np.newaxis, :] / P_ij)
    MI_[P_ij == 0] = 0
    MI_[(P_i[:, np.newaxis]*P_i[np.newaxis, :]) == 0] = 0
    return MI_.sum()

def get_KL(p_true, p_obs, do_scipy=False):
    if p_true.sum()==0. or p_obs.sum()==0.:
        return '>X>X>X KL function:  problem with null histograms! <X<X<X<'
    # if not (p_true.sum()==1. and p_obs.sum()==1.):
    #     print(p_true.sum(), p_obs.sum())
    #     return '>X>X>X KL function:  problem with non normalized histograms! <X<X<X<'
    if not do_scipy:
        p_true /= p_true.sum()
        p_obs /= p_obs.sum()
        surprise = np.log2(p_true/p_obs)
        surprise[p_true == 0] = 0
        surprise[p_obs == 0] = 0
        kl = np.sum(p_true*surprise, axis=-1)
        return kl
    else:
        from scipy.stats import entropy
        return entropy(p_obs.T, p_true.T, base=2)


def get_KS(P_true, P_obs):
    return np.abs(P_true-P_obs).sum(axis=1)


def print_stats(data, dictionary, sparse_code, max_patches=10, N_show=120, display=True, verbose=True):
    if display:
        import matplotlib.pyplot as plt
    if verbose:
        print(42*'🐒')

        print('number of codes, size of codewords = ', sparse_code.shape)
        print('average of codewords = ', sparse_code.mean())
        print('average std of codewords = ', sparse_code.std())
        # , ' ~= l0/M =', shl.l0_sparseness/shl.n_dictionary)
        print('l0-sparseness of codewords = ', (sparse_code > 0).mean())
        print('std of the average of individual patches = ', sparse_code.mean(axis=0).std())

    if display:
        plt.matshow(sparse_code[:N_show, :])
        plt.show()
        fig, axs = show_data(data[:max_patches, :])
        plt.show()

    patches = sparse_code @ dictionary
    error = data - patches

    if verbose:
        print('number of codes, size of reconstructed images = ', patches.shape)

    if display:
        fig, axs = show_data(patches[:max_patches, :])
        plt.show()
        fig, axs = show_data(error[:max_patches, :], cmax=np.max(np.abs(patches[:max_patches, :])))
        plt.show()

    SD = np.sqrt(np.mean(data**2, axis=1))

    if verbose:
        print('average of data patches = ', data.mean(), '+/-', data.mean(axis=1).std())
        print('average of residual patches = ', error.mean(), '+/-', error.mean(axis=1).std())
        print('median energy of data = ', np.median(SD))
        print('average energy of data = ', SD.mean(), '+/-', SD.std())
        print('average error = ', error.mean(), '+/-', error.std())

    SE = np.sqrt(np.mean(error**2, axis=1))

    if verbose:
        print('average energy of residual = ', SE.mean(), '+/-', SE.std())
        print('median energy of residual = ', np.median(SE))
        print('average gain of coding = ', (SD/SE).mean(), '+/-', (SD/SE).std())

    return SD, SE


def compute_RMSE(data, dico):
    """
    Compute the Root Mean Square Error between the image and it's encoded representation
    """
    a = dico.transform(data)
    residual = data - a @ dico.dictionary
    mse = np.sum(residual**2, axis=1)/np.sqrt(np.sum(data**2, axis=1))
    rmse = np.sqrt(np.mean(mse))
    return rmse


def compute_distance(data, dico):
    """
    Compute the distance of a distribution from its gaussian equivalent.
    if the KL is close to 1, the studied distribution is closed to a gaussian
    """
    sparse_code = dico.transform(data)
    N = dico.dictionary.shape[0]
    P_norm = np.mean(sparse_code**2, axis=0)  # /Z
    mom1 = np.sum(P_norm)/dico.dictionary.shape[0]
    mom2 = np.sum((P_norm-mom1)**2)/(dico.dictionary.shape[0]-1)
    distance = 1/N * np.sum((P_norm-mom1)**2 / mom2**2)
    return distance


def compute_kurto(data, dico):
    """
    Compute the kurtosis
    """
    sparse_code = dico.transform(data)
    P_norm = np.mean(sparse_code**2, axis=0)  # /Z
    from scipy.stats import kurtosis
    kurto = kurtosis(P_norm, axis=0)
    return kurto

# To adapt with shl_exp


def show_dico_in_order(shl_exp, dico, data=None, title=None, fname=None, dpi=200, **kwargs):
    """
    Displays the dictionary of filter in order of probability of selection.
    Filter which are selected more often than others are located at the end

    """
    return show_dico(shl_exp, dico=dico, data=data, dim_graph=None,
                     order=True, title=title, fname=fname, dpi=dpi, **kwargs)


def show_dico(shl_exp, dico,  data=None, order=False, title=None, dim_graph=None,
                 do_tiles=True, fname=None, dpi=200, fig=None, ax=None):
    """
    display the dictionary in a random order
    """
    import matplotlib
    subplotpars = matplotlib.figure.SubplotParams(
        left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)

    n_dictionary = dico.dictionary.shape[0]
    if dim_graph is None:
        dim_graph = int(np.ceil(np.sqrt(n_dictionary)))
        dim_graph = (dim_graph, dim_graph)
    dim_patch = int(np.sqrt(dico.dictionary.shape[1]))

    if order:
        # order by activation probability
        sparse_code = shl_exp.code(data=data, dico=dico)
        res_lst = np.count_nonzero(sparse_code, axis=0)
        indices = res_lst.argsort()
    else:
        indices = range(np.prod(dim_graph))

    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)
    if ax is None:
        ax = fig.add_subplot(111)

    if do_tiles:
        for i in range(np.prod(dim_graph)):
            ax = fig.add_subplot(dim_graph[0], dim_graph[1], i + 1)
            dico_to_display = dico.dictionary[indices[i]]
            cmax = np.max(np.abs(dico_to_display))
            if not dico.precision is None:
                dico_to_display = dico_to_display.reshape((dim_patch, dim_patch))/cmax
                precision_to_display = dico.precision[indices[i]].reshape((dim_patch, dim_patch))
                precision_cmax = np.max(precision_to_display)

                image = np.dstack((.5 + .5*dico_to_display, .5 + .5 *
                                   dico_to_display, .5 + .5*dico_to_display))
                image *= np.dstack((np.ones_like(precision_to_display), precision_to_display /
                                    precision_cmax, np.ones_like(precision_to_display)))
                ax.imshow(image, interpolation='nearest')
                # DEBUG:
                # ax.imshow(precision_to_display/precision_cmax,
                #             cmap=plt.cm.gray_r, vmin=0, vmax=+precision_cmax,
                #             interpolation='nearest')

            else:
                ax.imshow(dico_to_display.reshape((dim_patch, dim_patch)),
                          cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                          interpolation='nearest')
            ax.set_xticks(())
            ax.set_yticks(())
    else:
        # backgroung image
        image = -np.ones((dim_graph[0]*(dim_patch+1)+1, dim_graph[1]*(dim_patch+1)+1))
        for i in range(np.prod(dim_graph)):
            dico_to_display = dico.dictionary[indices[i]].reshape((dim_patch, dim_patch))
            cmax = np.max(np.abs(dico_to_display))
            i_col, i_row = i % dim_graph[1], i // dim_graph[1]
            image[(i_row*(dim_patch+1)+1):((i_row+1)*(dim_patch+1)), (i_col*(dim_patch+1)+1):((i_col+1)*(dim_patch+1))] = dico_to_display / cmax

        if not dico.precision is None:
            print('not implemented') # TODO
            assert(False)
        else:
            ax.imshow(image,
                      cmap=plt.cm.gray_r, vmin=-1, vmax=+1,
                      interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_axis_off()

    if title is not None:
        fig.suptitle(title, fontsize=12, backgroundcolor='white', color='k')
    if not fname is None:
        fig.savefig(fname, dpi=dpi)
    return fig, ax


def show_data(data, fname=None, dpi=200, cmax=None, fig=None, axs=None):
    """
    display the data in a line

    """
    N_patches, n_pixels = data.shape
    N_pix = np.sqrt(n_pixels).astype(int)
    # subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(15, 3))  # , subplotpars=subplotpars
        fig, axs = plt.subplots(1, N_patches, figsize=(15, 2))
    # if ax is None:
    #     ax = fig.add_subplot(111)

    if cmax is None:
        cmax = np.max(np.abs(data))
    for j in range(N_patches):
        axs[j].imshow(data[j, :].reshape((N_pix, N_pix)),
                      cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                      interpolation='nearest')
        axs[j].set_xticks(())
        axs[j].set_yticks(())

    if not fname is None:
        fig.savefig(fname, dpi=dpi)
    return fig, axs


def plot_coeff_distribution(dico, data, title=None, algorithm=None, fname=None, fig=None, ax=None):
    """
    Plot the coeff distribution of a given dictionary
    """

    if algorithm is not None:
        sparse_code = sparse_encode(data, dico.dictionary, algorithm=algorithm)

    else:
        sparse_code = dico.transform(data)
    res_lst = np.count_nonzero(sparse_code, axis=0)
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(res_lst, columns=['Coeff'])
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)

    with sns.axes_style("white"):
        ax = sns.distplot(df['Coeff'], kde=False)  # , fit=gamma,  fit_kws={'clip':(0., 5.)})
    if title is not None:
        ax.set_title('distribution of coefficients, ' + title)
    else:
        ax.set_title('distribution of coefficients')
    ax.set_ylabel('pdf')
    ax.set_xlim(0)
    if not fname is None:
        fig.savefig(fname, dpi=200)
    return fig, ax


def bins_step(mini, maxi, nb_step):
    """
    doing a range of non integer number to make histogram more beautiful
    """
    step = (maxi-mini)/nb_step
    out = list()
    a = mini
    for i in range(nb_step+1):
        out.append(a)
        a = a+step
    out.append(a)
    return out


def plot_dist_max_min(shl_exp, dico, data=None, algorithm=None, fname=None, fig=None, ax=None):
    """
    plot the coefficient distribution of the filter which is selected the more,
    and the one which is selected the less
    """
    if (algorithm is not None) and (data is not None):
        sparse_code = sparse_encode(data, dico.dictionary, algorithm=algorithm)
    else:
        sparse_code = shl_exp.coding
    nb_filter_selection = np.count_nonzero(sparse_code, axis=0)

    index_max = np.argmax(nb_filter_selection)
    index_min = np.argmin(nb_filter_selection)
    color, label = ['r', 'b'], ['most selected filter : {0}'.format(
        index_max), 'less selected filter : {0}'.format(index_min)]
    coeff_max = np.abs(sparse_code[:, index_max])
    coeff_min = np.abs(sparse_code[:, index_min])
    bins_max = bins_step(0.0001, np.max(coeff_max), 20)
    bins_min = bins_step(0.0001, np.max(coeff_min), 20)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    with sns.axes_style("white"):
        n_max, bins1 = np.histogram(coeff_max, bins_max)
        n_min, bins2 = np.histogram(coeff_min, bins_min)
        ax.semilogy(bins1[:-1], n_max, label=label[0], color=color[0])
        ax.semilogy(bins2[:-1], n_min, label=label[1], color=color[1])
    ax.set_title('distribution of coeff in the most & less selected filters')
    ax.set_ylabel('number of selection')
    ax.set_xlabel('value of coefficient')
    plt.legend()
    if not fname is None:
        fig.savefig(fname, dpi=200)
    return fig, ax


def mutual_coherence(dictionary):
    """
    returns the mutual-coherence of matrix dico

    http://stemblab.github.io/mutual-coherence/

    """
    norm = np.sqrt(np.sum(dictionary**2, axis=1))
    dictionary /= norm[:, np.newaxis]
    Xcorr = (dictionary @ dictionary.T)
    Xcorr[np.diag_indices_from(Xcorr)] = 0.
    return Xcorr.max()

# To adapt with shl_exp


def plot_variance_and_proxy(dico, data, title, algorithm=None, fname=None, fig=None, ax=None):
    """
    Overlay of 2 histogram, the histogram of the variance of the coefficient,
    and the corresponding gaussian one
    """
    if algorithm is not None:
        sparse_code = sparse_encode(data, dico.dictionary, algorithm=algorithm)
    else:
        sparse_code = shl_encode.code(data)
    Z = np.mean(sparse_code**2)
    P_norm = np.mean(sparse_code**2, axis=0)/Z
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(P_norm, columns=['P'])
    mom1 = np.mean(P_norm)
    mom2 = (1/(dico.dictionary.shape[0]-1))*np.sum((P_norm-mom1)**2)
    Q = np.random.normal(mom1, mom2, dico.dictionary.shape[0])
    df1 = pd.DataFrame(Q, columns=['Q'])

    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)

    #bins=[0, 10, 20, 30, 40, 50, 100]
    mini = min(np.min(P_norm), np.min(Q))
    maxi = max(np.max(P_norm), np.max(Q))
    bins = bins_step(mini, maxi, 20)
    with sns.axes_style("white"):
        ax = sns.distplot(df['P'], bins=bins, kde=False)
        ax = sns.distplot(df1['Q'], bins=bins, kde=False)
        # ax = sns.distplot(df['P'], bins=frange(0.0,4.0,0.2),kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
        #ax = sns.distplot(df1['Q'],bins=frange(0.0,4.0,0.2), kde=False)
    if title is not None:
        ax.set_title('distribution of the mean variance of coefficients, ' + title)
    else:
        ax.set_title('distribution of the mean variance of coefficients ')
    ax.set_ylabel('pdf')
    if not fname is None:
        fig.savefig(fname, dpi=200)
    # print(mom1,mom2)
    return fig, ax


def plot_proba_histogram(coding, verbose=False, fig=None, ax=None):
    n_dictionary = coding.shape[1]

    p = np.count_nonzero(coding, axis=0)/coding.shape[1]
    p /= p.sum()

    rel_ent = np.sum(-p * np.log(p)) / np.log(n_dictionary)
    if verbose:
        print('Entropy / Entropy_max=', rel_ent)

    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)

    ax.bar(np.arange(n_dictionary), p*n_dictionary)
    ax.set_title('distribution of the selection probability - entropy= ' + str(rel_ent))
    ax.set_ylabel('pdf')
    ax.axis('tight')
    ax.set_xlim(0, n_dictionary)
    return fig, ax


def plot_error(dico, fig=None, ax=None):
    # TODO : show SE as a function of l0
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)
    n = np.arange(dico.n_iter)
    err = dico.rec_error
    ax.plot(n, err)
    ax.set_title('Reconstruction error')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Iteration')
    return fig, ax


def plot_variance(shl_exp, sparse_code, fname=None, fig=None, ax=None):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)
    n_dictionary = sparse_code.shape[1]
    Z = np.mean(sparse_code**2)
    # , yerr=np.std(code**2/Z, axis=0))
    ax.bar(np.arange(n_dictionary), np.mean(sparse_code**2, axis=0)/Z)
    ax.set_title('Variance of coefficients')
    ax.set_ylabel('Variance')
    ax.set_xlabel('#')
    ax.set_xlim(0, n_dictionary)
    if not fname is None:
        fig.savefig(fname, dpi=200)
    return fig, ax


def plot_variance_histogram(shl_exp, sparse_code, fname=None, fig=None, ax=None):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)
    variance = np.mean(sparse_code**2, axis=0)
    ax.hist(variance, bins=np.linspace(0, variance.max(), 20, endpoint=True))
    ax.set_title('distribution of the mean variance of coefficients')
    ax.set_ylabel('pdf')
    ax.axis('tight')
    ax.set_xlim(0)
    if not fname is None:
        fig.savefig(fname, dpi=200)
    return fig, ax


def plot_P_cum(P_cum, ymin=0.95, ymax=1.001, title='non-linear functions', suptitle=None, ylabel='quantile', verbose=False, n_yticks=21, alpha=.05, c='g', fig=None, ax=None):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 8))
    if ax is None:
        ax = fig.add_subplot(111)
    coefficients = np.linspace(0, 1, P_cum.shape[1])
    if ymax>1.:
        ax.plot(coefficients, np.ones_like(coefficients), '--')
    ax.plot(coefficients, P_cum.T, c=c, alpha=alpha)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('rescaled coefficients')
    ax.set_ylabel(ylabel)
    # ax.set_xlim(0)
    ax.set_yticks(np.linspace(0, 1, n_yticks))
    ax.axis('tight')
    ax.set_ylim(ymin, ymax)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12, backgroundcolor='white', color='k')
    return fig, ax

#import seaborn as sns
#import pandas as pd


def plot_scatter_MpVsTrue(sparse_vector, my_sparse_code, alpha=.01, xlabel='True', ylabel='MP', fig=None, ax=None):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 16))
    if ax is None:
        ax = fig.add_subplot(111)

    a_min = np.min((sparse_vector.min(), my_sparse_code.min()))
    a_max = np.max((sparse_vector.max(), my_sparse_code.max()))
    ax.plot(np.array([a_min, a_max]), np.array([a_min, a_max]), 'k--', lw=2)
    ax.scatter(sparse_vector.ravel(), my_sparse_code.ravel(), alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.axis('equal')
    return fig, ax

def get_record(dico, variable, N_nosample):
    if variable=='F':
        # print('HACK')
        learning_time, A1 = get_record(dico, 'error', N_nosample)
        learning_time, A2 = get_record(dico, 'qerror', N_nosample)
        A = 61.803*A1 + .1*A2
    else:
        # try:
        df_variable = dico.record[variable]
        learning_time = np.array(df_variable.index)  # np.arange(0, dico.n_iter, dico.record_each)
        # if df_variable.ndim==1:
        #     print(df_variable.index, variable, df_variable.ndim)
        try:
            A = np.zeros((len(df_variable.index)))
            for ii, ind in enumerate(df_variable.index):
                # print(df_==variable[ind].shape)
                A[ii] = df_variable[ind]
        except:
            A = np.zeros((len(df_variable.index), dico.n_dictionary))
            for ii, ind in enumerate(df_variable.index):
                A[ii, :] = df_variable[ind]
            A = A[:, :-N_nosample]

    return learning_time, A

def time_plot(shl_exp, dico, variable='kurt', unit=None, N_nosample=0, alpha=.6,
                color=None, label=None, fname=None, fig=None, ax=None):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(16, 4))
    if ax is None:
        ax = fig.add_subplot(111)

    learning_time, A = get_record(dico, variable, N_nosample)

    # print(learning_time, A[:, :-N_nosample].shape, df_variable[ind])
    ax.plot(learning_time, A, '-', lw=1, alpha=alpha, color=color, label=label)
    if unit is None:
        ax.set_ylabel(variable)
    else:
        ax.set_ylabel(variable + ' (' + unit + ')')
    ax.set_xlabel('Learning step')
    ax.set_xlim(0, dico.n_iter)
    # if variable=='entropy' :
    #    ax.set_ylim(0.95)
    # else
    #print('label', label)
    #if not label is None: ax.legend(loc='best')
    if variable in ['error', 'qerror']:
        ax.set_ylim(0)

    # except ImportError:
    #     ax.set_title('record not available')
    #     ax.set_ylabel(variable)
    #     ax.set_xlabel('Learning step')
    #     ax.set_xlim(0, shl_exp.n_iter)
    if not fname is None:
        fig.savefig(fname, dpi=200)
    return fig, ax
