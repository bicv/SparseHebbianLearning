
from SLIP import Image
from scipy.stats import kurtosis
import sys
import time
import numpy as np
import encode_shl
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gamma

toolbar_width = 40

def bins_step(mini,maxi,nb_step):
    '''doing a range of non integer number to make histogramm more beautiful'''
    step=(maxi-mini)/10
    out=list()
    a=mini
    for i in range(nb_step+1):
        out.append(a)
        a=a+step
    return out


def get_data(height=256, width=256, n_image=200, patch_size=(12,12),
            datapath='database/', name_database='serre07_distractors',
            max_patches=1024, seed=None, patch_norm=True, verbose=0):
    ''' Extract database
    Extract from a given database composed of image of size (height,width) a series a random patch
    '''
    slip = Image({'N_X':height, 'N_Y':width,
                'white_n_learning' : 0,
                'seed': None,
                'white_N' : .07,
                'white_N_0' : .0, # olshausen = 0.
                'white_f_0' : .4, # olshausen = 0.2
                'white_alpha' : 1.4,
                'white_steepness' : 4.,
                'datapath': datapath,
                'do_mask':True,
                'N_image': n_image})

    if verbose:
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
    return data

def compute_RMSE(data, dico):
    ''' Compute the Root Mean Square Error between the image and it's encoded representation'''
    a=dico.transform(data)
    residual=data - a@dico.dictionary
    b=np.sum(residual**2,axis=1)/np.sqrt(np.sum(data**2,axis=1))
    rmse=math.sqrt(np.mean(b))
    return rmse

def compute_KL(data, dico):
    '''Compute the Kullback Leibler ratio to compare a distribution to its gaussian equivalent.
    if the KL is close to 1, the studied distribution is closed to a gaussian'''
    sparse_code= dico.transform(data)
    N=dico.dictionary.shape[0]
    P_norm = np.mean(sparse_code**2, axis=0)#/Z
    mom1 = np.sum(P_norm)/dico.dictionary.shape[0]
    mom2 = np.sum((P_norm-mom1)**2)/(dico.dictionary.shape[0]-1)
    KL = 1/N * np.sum( (P_norm-mom1)**2 / mom2**2 )
    return KL

def Compute_kurto(data, dico):
    '''Compute the kurtosis'''
    sparse_code= dico.transform(data)
    P_norm = np.mean(sparse_code**2, axis=0)#/Z
    kurto = kurtosis(P_norm, axis=0)
    return kurto

def show_dico_in_order(dico, data, algorithm=None,title=None, fname=None):
    '''Display a the dictionary of filter in order of probability of selection.
    Filter which are selected more often than others are located at the end'''
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
    fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    dim_graph=dico.dictionary.shape[0]
    nb_of_patch=data.shape[0]
    res=0
    i=0
    res_lst=list()
    for j in range(dim_graph):
        res=0
        while i<nb_of_patch:
            if sparse_code[i,j]!=0 : res+=1
            i+=1
        res_lst.append(res)
        i=0

    a=np.asarray(res_lst).argsort()
    dim_graph=dico.dictionary.shape[0]
    dim_patch=int(np.sqrt(data.shape[1]))

    for i in range(dim_graph):
        ax = fig.add_subplot(np.sqrt(dim_graph), np.sqrt(dim_graph), i + 1)
        index_to_consider=a[i]
        dico_to_display=dico.dictionary[index_to_consider]
        cmax = np.max(np.abs(dico_to_display))
        ax.imshow(dico_to_display.reshape((dim_patch,dim_patch)), cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
    if title is not None:
        fig.suptitle(title, fontsize=12, backgroundcolor = 'white', color = 'k')
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def show_dico(dico, title=None, fname=None, **kwargs):
    '''
    display the dictionary in a random order
    '''
    dim_graph = dico.dictionary.shape[0]
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
    fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)
    dim_patch = int(np.sqrt(dico.dictionary.shape[1]))

    for i, component in enumerate(dico.dictionary):
        ax = fig.add_subplot(np.sqrt(dim_graph), np.sqrt(dim_graph), i + 1)
        cmax = np.max(np.abs(component))
        ax.imshow(component.reshape((dim_patch,dim_patch)), cmap=plt.cm.gray_r, vmin=-cmax, vmax=+cmax,
                interpolation='nearest')
        ax.set_xticks(())
        ax.set_yticks(())
    if title is not None:
        fig.suptitle(title, fontsize=12, backgroundcolor = 'white', color = 'k')
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def plot_coeff_distribution(dico, data, title,algorithm=None,fname=None):
    '''Plot the coeff distribution of a given dictionary'''
    nb_dico=dico.dictionary.shape[0]
    nb_of_patch=data.shape[0]
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    res=0
    i=0
    res_lst=list()

    for j in range(nb_dico):
        res=0
        while i<nb_of_patch:
            if sparse_code[i,j]!=0 : res+=1
            i+=1
        res_lst.append(res)
        i=0

    df=pd.DataFrame(res_lst, columns=['Coeff'])
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


def plot_dist_max_min(dico, data, algorithm=None,fname=None):
    '''plot the coefficient distribution of the filter which is selected the more, and the one which is selected the less'''
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    nb_dico=dico.dictionary.shape[0]
    nb_of_patch=data.shape[0]
    res=0
    i=0
    res_lst=list()
    for j in range(nb_dico):
        res=0
        while i<nb_of_patch:
            if sparse_code[i,j]!=0 : res+=1
            i+=1
        res_lst.append(res)
        i=0
    a=np.asarray(res_lst)
    index_max=np.argmax(a)
    index_min=np.argmin(a)
    coeff_max = np.abs(sparse_code[:,index_max])
    coeff_min = np.abs(sparse_code[:,index_min])
    bins_max = bins_step(0.0001,np.max(coeff_max),20)
    bins_min = bins_step(0.0001,np.max(coeff_min),20)
    fig = plt.figure(figsize=(6, 8))
    with sns.axes_style("white"):
        ax = plt.subplot(2,1,1)
        ax = sns.distplot(coeff_max,bins=bins_max, kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
        ax1=plt.subplot(2,1,2)
        ax1= sns.distplot(coeff_min,bins=bins_min, kde=False)
    ax.set_title('distribution of max')
    ax.set_ylabel('Probability')
    ax.set_xlim(0)
    ax1.set_title('distribution of min')
    ax1.set_ylabel('Probability')
    ax1.set_xlim(0)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def plot_variance_and_proxy(dico, data, title, algorithm=None, fname=None):
    '''Overlay of 2 histogram, the histogram of the variance of the coefficient, and the corresponding gaussian one'''
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    Z = np.mean(sparse_code**2)
    P_norm=np.mean(sparse_code**2, axis=0)/Z
    df = pd.DataFrame(P_norm, columns=['P'])
    mom1= np.mean(P_norm)
    mom2 = (1/(dico.dictionary.shape[0]-1))*np.sum((P_norm-mom1)**2)
    Q=np.random.normal(mom1,mom2,dico.dictionary.shape[0])
    df1=pd.DataFrame(Q, columns=['Q'])
    #code = self.code(data, dico)
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

def plot_variance(dico, data, algorithm=None, fname=None):
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    n_dictionary=dico.dictionary.shape[0]
    # code = self.code(data, dico)
    Z = np.mean(sparse_code**2)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(n_dictionary), np.mean(sparse_code**2, axis=0)/Z)#, yerr=np.std(code**2/Z, axis=0))
    ax.set_title('Variance of coefficients')
    ax.set_ylabel('Variance')
    ax.set_xlabel('#')
    ax.set_xlim(0, n_dictionary)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def plot_variance_histogram(dico, data, algorithm=None, fname=None):
    from scipy.stats import gamma
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    Z = np.mean(sparse_code**2)
    df = pd.DataFrame(np.mean(sparse_code**2, axis=0)/Z, columns=['Variance'])
    #code = self.code(data, dico)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    with sns.axes_style("white"):
        ax = sns.distplot(df['Variance'], kde=False)#, fit=gamma,  fit_kws={'clip':(0., 5.)})
    ax.set_title('distribution of the mean variance of coefficients')
    ax.set_ylabel('pdf')
    ax.set_xlim(0)
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax

def time_plot(dico, variable='kurt', fname=None, N_nosample=1, alpha=.3):

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
    if not fname is None: fig.savefig(fname, dpi=200)
    return fig, ax
