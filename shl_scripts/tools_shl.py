
''' Extract database
Extract from a given database composed of image of size (height,width) a series a random patch 
'''
def get_data(height=256,width=256,n_image=200,patch_size=(12,12),
            datapath='database/',name_database='serre07_distractors',
            seed=None,patch_norm=True):
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
