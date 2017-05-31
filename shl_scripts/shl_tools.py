#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import division, print_function, absolute_import
import time
import numpy as np
from shl_scripts.shl_encode import sparse_encode
import matplotlib
import matplotlib.pyplot as plt

toolbar_width = 40


def touch(filename):
    open(filename, 'w').close()


def get_data(height=256, width=256, n_image=200, patch_size=(12,12),
            datapath='database/', name_database='serre07_distractors',
            max_patches=1024, seed=None, patch_norm=True, verbose=0,
            data_cache='/tmp/data_cache', matname=None):
    """
    Extract data:

    Extract from a given database composed of image of size (height, width) a
    series a random patches.

    """
    if matname is None:
        from SLIP import Image

        slip = Image({'N_X':height, 'N_Y':width,
                'white_n_learning' : 0,
                'seed': seed,
                'white_N' : .07,
                'white_N_0' : .0, # olshausen = 0.
                'white_f_0' : .4, # olshausen = 0.2
                'white_alpha' : 1.4,
                'white_steepness' : 4.,
                'datapath': datapath,
                'do_mask':True,
                'N_image': n_image})

        if verbose:
            import sys
            # setup toolbar
            sys.stdout.write('Extracting data...')
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
            t0 = time.time()
        imagelist = slip.make_imagelist(name_database=name_database)#, seed=seed)
        for filename, croparea in imagelist:
            # whitening
            image, filename_, croparea_ = slip.patch(name_database, filename=filename, croparea=croparea, center=False)#, seed=seed)
            image = slip.whitening(image)
            # Extract all reference patches and ravel them
            data_ = slip.extract_patches_2d(image, patch_size, N_patches=int(max_patches))#, seed=seed)
            data_ = data_.reshape(data_.shape[0], -1)
            data_ -= np.mean(data_, axis=0)
            if patch_norm:
                data_ /= np.std(data_, axis=0)
            # collect everything as a matrix
            try:
                data = np.vstack((data, data_))
            except Exception:
                data = data_.copy()
            if verbose:
                # update the bar
                sys.stdout.write(filename + ", ")
                sys.stdout.flush()
        if verbose:
            dt = time.time() - t0
            sys.stdout.write("\n")
            sys.stdout.write("Data is of shape : "+ str(data.shape))
            sys.stdout.write(' - done in %.2fs.' % dt)
            sys.stdout.flush()
    else:
        import os
        fmatname = os.path.join(data_cache, matname)
        if not(os.path.isfile(fmatname + '_data.npy')):
            if not(os.path.isfile(fmatname + '_data' + '_lock')):
                touch(fmatname + '_data' + '_lock')
                try:
                    if verbose: print('No cache found {}: Extracting data...'.format(fmatname + '_data'), end=' ')
                    data = get_data(height=height, width=width, n_image=n_image,
                                    patch_size=patch_size, datapath=datapath,
                                    name_database=name_database, max_patches=max_patches,
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
        else:
            if verbose: print("loading the data called : {0}".format(fmatname + '_data'))
            # Une seule fois mp ici
            data = np.load(fmatname + '_data.npy')
    return data

def generate_sparse_vector(N_image, l0_sparseness, nb_dico, N_boost=0,
                           K_boost=2., C_0=3., rho_coeff=.85, seed=420, do_sym=False):
    np.random.seed(seed)
    coeff = np.zeros((N_image, nb_dico))
    rho = np.zeros((N_image, nb_dico))
    for i in range(N_image):
        ind = np.random.permutation(np.arange(nb_dico))[:l0_sparseness] # indices of non-zero coefficients
        coeff[i, ind] = C_0 * rho_coeff**np.arange(l0_sparseness) # activities
        coeff[i, :N_boost] *= K_boost  # perturbation
        if do_sym:
            coeff[i, ind] *= np.sign(np.random.randn(l0_sparseness))

        rho[i, ind] = 1 - np.arange(l0_sparseness)/nb_dico # 1 - relative rank
    return coeff, rho


def compute_RMSE(data, dico):
    """
    Compute the Root Mean Square Error between the image and it's encoded representation
    """
    a = dico.transform(data)
    residual = data - a @ dico.dictionary
    mse = np.sum(residual**2, axis=1)/np.sqrt(np.sum(data**2, axis=1))
    rmse = np.sqrt(np.mean(mse))
    return rmse

def compute_KL(data, dico):
    """
    Compute the Kullback Leibler ratio to compare a distribution to its gaussian equivalent.
    if the KL is close to 1, the studied distribution is closed to a gaussian
    """
    sparse_code = dico.transform(data)
    N = dico.dictionary.shape[0]
    P_norm = np.mean(sparse_code**2, axis=0)#/Z
    mom1 = np.sum(P_norm)/dico.dictionary.shape[0]
    mom2 = np.sum((P_norm-mom1)**2)/(dico.dictionary.shape[0]-1)
    KL = 1/N * np.sum( (P_norm-mom1)**2 / mom2**2 )
    return KL

def compute_kurto(data, dico):
    """
    Compute the kurtosis
    """
    sparse_code= dico.transform(data)
    P_norm = np.mean(sparse_code**2, axis=0)#/Z
    from scipy.stats import kurtosis
    kurto = kurtosis(P_norm, axis=0)
    return kurto

# To adapt with shl_exp
def show_dico_in_order(shl_exp, dico, data=None, title=None, fname=None, dpi=200, **kwargs):
    """
    Displays the dictionary of filter in order of probability of selection.
    Filter which are selected more often than others are located at the end

    """
    return show_dico(shl_exp, dico=dico, data=data, order=True, title=title, fname=fname, dpi=dpi, **kwargs)

def show_dico(shl_exp, dico,  data=None, order=False, title=None, fname=None, dpi=200, **kwargs):
    """
    display the dictionary in a random order
    """
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
    fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)

    dim_graph = dico.dictionary.shape[0]
    if order:
        sparse_code = shl_exp.code(data=data, dico=dico)
        res_lst = np.count_nonzero(sparse_code, axis=0)
        indices = res_lst.argsort()
    else:
        indices = range(dim_graph)
    dim_patch = int(np.sqrt(dico.dictionary.shape[1]))

    for i in range(dim_graph):
        ax = fig.add_subplot(np.sqrt(dim_graph), np.sqrt(dim_graph), i + 1)
        dico_to_display = dico.dictionary[indices[i]]
        cmax = np.max(np.abs(dico_to_display))
        ax.imshow(dico_to_display.reshape((dim_patch,dim_patch)),
                     cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                     interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
    if title is not None:
        fig.suptitle(title, fontsize=12, backgroundcolor = 'white', color = 'k')
    if not fname is None: fig.savefig(fname, dpi=dpi)
    return fig, ax

def plot_coeff_distribution(dico, data, title=None,algorithm=None,fname=None):
    """
    Plot the coeff distribution of a given dictionary
    """

    if algorithm is not None :
        sparse_code = shl_encode.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    res_lst=np.count_nonzero(sparse_code,axis=0)
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(res_lst, columns=['Coeff'])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    with sns.axes_style("white"):
        ax = sns.distplot(df['Coeff'], kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
    if title is not None:
        ax.set_title('distribution of coefficients, ' + title)
    else:
        ax.set_title('distribution of coefficients')
    ax.set_ylabel('pdf')
    ax.set_xlim(0)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax


def bins_step(mini, maxi, nb_step):
    """
    doing a range of non integer number to make histogram more beautiful
    """
    step=(maxi-mini)/nb_step
    out=list()
    a=mini
    for i in range(nb_step+1):
        out.append(a)
        a=a+step
    out.append(a)
    return out

def plot_dist_max_min(shl_exp, dico, data=None, algorithm=None, fname=None):
    """
    plot the coefficient distribution of the filter which is selected the more,
    and the one which is selected the less
    """
    if (algorithm is not None) and (data is not None)  :
        sparse_code = shl_encode.sparse_encode(data, dico.dictionary,algorithm=algorithm)
    else :
        sparse_code = shl_exp.coding
    nb_filter_selection=np.count_nonzero(sparse_code, axis=0)

    index_max=np.argmax(nb_filter_selection)
    index_min=np.argmin(nb_filter_selection)
    color,label=['r', 'b'], ['most selected filter : {0}'.format(index_max),'less selected filter : {0}'.format(index_min)]
    coeff_max = np.abs(sparse_code[:,index_max])
    coeff_min = np.abs(sparse_code[:,index_min])
    bins_max = bins_step(0.0001,np.max(coeff_max),20)
    bins_min = bins_step(0.0001,np.max(coeff_min),20)
    fig = plt.figure(figsize=(6, 10))
    ax = plt.subplot(2,1,1)
    with sns.axes_style("white"):
        n_max,bins1=np.histogram(coeff_max,bins_max)
        n_min,bins2=np.histogram(coeff_min,bins_min)
        ax.semilogy(bins1[:-1],n_max,label=label[0],color=color[0])
        ax.semilogy(bins2[:-1],n_min,label=label[1],color=color[1])
    ax.set_title('distribution of coeff in the most & less selected filters')
    ax.set_ylabel('number of selection')
    ax.set_xlabel('value of coefficient')
    plt.legend()
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

## To adapt with shl_exp
def plot_variance_and_proxy(dico, data, title, algorithm=None, fname=None):
    """
    Overlay of 2 histogram, the histogram of the variance of the coefficient,
    and the corresponding gaussian one
    """
    if algorithm is not None :
        sparse_code = shl_encode.sparse_encode(data, dico.dictionary, algorithm=algorithm)
    else :
        sparse_code = shl_encode.code(data)
    Z = np.mean(sparse_code**2)
    P_norm=np.mean(sparse_code**2, axis=0)/Z
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(P_norm, columns=['P'])
    mom1= np.mean(P_norm)
    mom2 = (1/(dico.dictionary.shape[0]-1))*np.sum((P_norm-mom1)**2)
    Q=np.random.normal(mom1,mom2,dico.dictionary.shape[0])
    df1=pd.DataFrame(Q, columns=['Q'])

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    #bins=[0, 10, 20, 30, 40, 50, 100]
    mini=min(np.min(P_norm),np.min(Q))
    maxi=max(np.max(P_norm),np.max(Q))
    bins=bins_step(mini,maxi,20)
    with sns.axes_style("white"):
        ax = sns.distplot(df['P'],bins=bins,kde=False)
        ax = sns.distplot(df1['Q'],bins=bins, kde=False)
        #ax = sns.distplot(df['P'], bins=frange(0.0,4.0,0.2),kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
        #ax = sns.distplot(df1['Q'],bins=frange(0.0,4.0,0.2), kde=False)
    if title is not None :
        ax.set_title('distribution of the mean variance of coefficients, ' + title)
    else :
        ax.set_title('distribution of the mean variance of coefficients ')
    ax.set_ylabel('pdf')
    if not fname is None: fig.savefig(fname, dpi=200)
    #print(mom1,mom2)
    return fig, ax

def plot_proba_histogram(coding, verbose=False):
    n_dictionary=coding.shape[1]

    p = np.count_nonzero(coding, axis=0)/coding.shape[1]
    p /= p.sum()

    rel_ent = np.sum( -p * np.log(p)) / np.log(n_dictionary)
    if verbose: print('Entropy / Entropy_max=', rel_ent )

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(n_dictionary), p*n_dictionary)
    ax.set_title('distribution of the selection probability - entropy= ' + str(rel_ent)  )
    ax.set_ylabel('pdf')
    ax.set_xlim(0)
    ax.axis('tight')
    return fig, ax

def plot_variance(shl_exp, sparse_code, data=None, algorithm=None, fname=None):
    n_dictionary = shl_exp.n_dictionary
    Z = np.mean(sparse_code**2)
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(n_dictionary), np.mean(sparse_code**2, axis=0)/Z)#, yerr=np.std(code**2/Z, axis=0))
    ax.set_title('Variance of coefficients')
    ax.set_ylabel('Variance')
    ax.set_xlabel('#')
    ax.set_xlim(0, n_dictionary)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def plot_variance_histogram(shl_exp, sparse_code, data=None, algorithm=None, fname=None):
    from scipy.stats import gamma

    Z = np.mean(sparse_code**2)
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(np.mean(sparse_code**2, axis=0)/Z, columns=['Variance'])
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    with sns.axes_style("white"):
        ax = sns.distplot(df['Variance'], kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
    ax.set_title('distribution of the mean variance of coefficients')
    ax.set_ylabel('pdf')
    ax.set_xlim(0)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax


def plot_P_cum(P_cum, verbose=False, n_yticks= 21, alpha=.05):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    coefficients = np.linspace(0, 1, P_cum.shape[1])
    ax.plot(coefficients, np.ones_like(coefficients), '--')
    ax.plot(coefficients, P_cum.T, c='g', alpha=alpha)
    ax.set_title(' non-linear functions ')
    ax.set_xlabel('normalized coefficients')
    ax.set_ylabel('z-score')
    #ax.set_xlim(0)
    ax.set_yticks( np.linspace(0, 1, n_yticks))
    ax.axis('tight')
    return fig, ax

#import seaborn as sns
#import pandas as pd
def plot_scatter_MpVsTrue(sparse_vector, my_sparse_code, alpha=.01, xlabel='True', ylabel='MP'):

    fig = plt.figure(figsize=(16, 16))
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


def time_plot(shl_exp, dico, variable='kurt', N_nosample=1, alpha=.3, fname=None):
    try:
        df_variable = dico.record[variable]
        learning_time = np.array(df_variable.index) #np.arange(0, dico.n_iter, dico.record_each)
        A = np.zeros((len(df_variable.index), dico.n_dictionary))
        for ii, ind in enumerate(df_variable.index):
            A[ii, :] = df_variable[ind]

        #print(learning_time, A[:, :-N_nosample].shape)
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        ax.plot(learning_time, A[:, :-N_nosample], '-', lw=1, alpha=alpha)
        ax.set_ylabel(variable)
        ax.set_xlabel('Learning step')
        ax.set_xlim(0, dico.n_iter)
        #if variable=='entropy' :
        #    ax.set_ylim(0.95)
        #else :
        #    ax.set_ylim(0)
        if not fname is None: fig.savefig(fname, dpi=200)
        return fig, ax

    except AttributeError:
        fig = plt.figure(figsize=(12, 1))
        ax = fig.add_subplot(111)
        ax.set_title('record not available')
        ax.set_ylabel(variable)
        ax.set_xlabel('Learning step')
        ax.set_xlim(0, dico.n_iter)
        if not fname is None: fig.savefig(fname, dpi=200)
        return fig, ax
