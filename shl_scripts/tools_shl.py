
from SLIP import Image
import sys
import time
import numpy as np

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
