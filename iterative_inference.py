# iterative_inference.py
# NN inference in an iterative manner, instead of a forward single shot.
from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def Load_CIFAR10(path):
    data_train = dict()
    data_test = dict()
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_batch'))
    data_train['input'] = Xtr
    data_train['output'] = Ytr
    data_test['input'] = Xte
    data_test['output'] = Yte
    return data_train, data_test


class IINN(object):
    def __init__(self, dim_x, dim_y):
        pass
    def attention(self, x, y):
        pass
    def inference(self, x, a):
        pass
    def getInputPlaceHolder(self):
        pass
    def getFeedbackPlaceHolder(self):
        pass
    def getOutputTensor(self):
        pass


def Build_IINN(n_class):
    dim_x = [None, None, None, 3]
    dim_y = [n_class]
    return IINN(dim_x, dim_y)


def Train_IINN(iinn_, data, model_path):
    xx = data['input']
    yy = data['output']
    print(xx.shape)


def Test_IINN(iinn_, data, model_path):
    xx = data['input']
    yy = data['output']
    print(xx.shape)
    for i in range(xx.shape[0]):
        x = xx[i]
        y = yy[i]
        y_trivial = np.ones(n_class)  # start from a trivial solution
        a = iinn_.attention(x, y_trivial)
        y = iinn_.inference(x, a)
        a = iinn_.attention(x, y)
        y = iinn_.inference(x, a)
        # ... this procedure goes on and on until converged
        pass


if __name__ == "__main__":
    n_class = 10
    iinn_ = Build_IINN(n_class)

    # training with CIFAR-10 dataset
    data_train, data_test = Load_CIFAR10('../Datasets/CIFAR10/')
    model_path = '../Models/CIFAR10-IINN/'
    Train_IINN(iinn_, data_train, model_path)
    # test the trained model with test split of the same dataset
    Test_IINN(iinn_, data_test, model_path)