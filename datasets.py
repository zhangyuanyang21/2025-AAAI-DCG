import os, sys, random
import numpy as np
import scipy.io as sio


def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    print("shuffle")
    if data_name in ['CUB']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','cub_googlenet_doc2vec_c10.mat'))
        X_list.append(mat['X'][0][0].astype('float32'))
        X_list.append(mat['X'][0][1].astype('float32'))
        Y_list.append(np.squeeze(mat['gt']))

    return X_list, Y_list

