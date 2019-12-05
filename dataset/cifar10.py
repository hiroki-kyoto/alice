from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import matplotlib.pyplot as plt


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename, n_class):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        XX = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") / 255.0
        Y = np.array(Y)
        YY = np.zeros([Y.shape[0], n_class])
        YY[np.arange(Y.shape[0]), Y] = 1.0
        return XX, YY


def Load_CIFAR10(path):
    data_train = dict()
    data_test = dict()
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f, 10)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_batch'), 10)
    data_train['input'] = Xtr
    data_train['output'] = Ytr
    data_test['input'] = Xte
    data_test['output'] = Yte
    return data_train, data_test
