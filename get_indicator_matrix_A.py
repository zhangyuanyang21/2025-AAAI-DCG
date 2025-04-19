import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete multi-view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: e.g., 0.1, 0.3, 0.5, 0.7
        Returns:
          indicator matrix A

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix    # indicator matrix A

