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
    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2]:
            x = train_x[view][index]
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')[index]
        Y_list.append(y)
    elif data_name in ['HandWritten']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'handwritten.mat'))
        X_list.append(mat['X'][1][0].astype('float32'))
        X_list.append(mat['X'][4][0].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['Multi-Fashion']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','Fashion.mat'))
        X_list.append(mat['X1'].reshape(-1,784).astype('float32'))
        X_list.append(mat['X2'].reshape(-1,784).astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
       
    return X_list, Y_list

