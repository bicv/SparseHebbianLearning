
from SLIP import Image
import sys
import time
import numpy as np
import encode_shl
import math
import matplotlib
import matplotlib.pyplot as plt

toolbar_width = 40
''' Extract database
Extract from a given database composed of image of size (height,width) a series a random patch
'''
def get_data(height=256,width=256,n_image=200,patch_size=(12,12),
            datapath='database/',name_database='serre07_distractors',
            max_patches=1024,seed=None,patch_norm=True,verbose=0):
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
        image, filename_, croparea_ = slip.patch(name_database, filename=filename, croparea=croparea, center=False)#, , seed=seed)
        image = slip.whitening(image)
        # Extract all reference patches and ravel them

        ### Modification temporaire pour faire fonctionner la fonction extract_patches_2d
        #data_ = extract_patches_2d(self.height,self.width,image, self.patch_size, N_patches=int(self.max_patches))
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

''' Compute the Root Mean Square Error between the image and it's encoded representation'''
def compute_RMSE(data, dico, algorithm=None):
    if algorithm is not None :
        a=encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        a=dico.transform(data)
    residual=data - a@dico.dictionary
    b=np.sum(residual**2,axis=1)/np.sqrt(np.sum(data**2,axis=1))
    rmse=math.sqrt(np.mean(b))
    return rmse

'''Compute the Kullback Leibler ratio to compare a distribution to its gaussian equivalent.
    if the KL is close to 1, the studied distribution is closed to a gaussian'''
def compute_KL(data, dico, algorithm=None):
    if algorithm is not None :
        sparse_code = encode_shl.sparse_encode(data,dico.dictionary,algorithm=algorithm)
    else :
        sparse_code= dico.transform(data)
    N=dico.dictionary.shape[0]
    P_norm = np.mean(sparse_code**2, axis=0)#/Z
    mom1 = np.sum(P_norm)/dico.dictionary.shape[0]
    mom2 = np.sum((P_norm-mom1)**2)/(dico.dictionary.shape[0]-1)
    KL = 1/N * np.sum( (P_norm-mom1)**2 / mom2**2 )
    return KL

'''Display a the dictionary of filter in order of probability of selection.
    Filter which are selected more often than others are located at the end'''
def show_dico_in_order(dico,data,algorithm=None,title=None, fname=None):
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

'''display the dictionary in a random order '''
def show_dico(dico,data, title=None, fname=None, **kwargs):
    dim_graph=dico.dictionary.shape[0]
    subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,)
    fig = plt.figure(figsize=(10, 10), subplotpars=subplotpars)
    dim_patch=int(np.sqrt(data.shape[1]))
    
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
