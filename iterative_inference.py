# iterative_inference.py
# NN inference in an iterative manner, instead of a forward single shot.

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import dataset


def get_conv_weights(w, h, chn_in, chn_out):
    dim = [w, h, chn_in, chn_out]
    init_op = tf.truncated_normal(dim, 0.02)
    return tf.get_variable(
        name='weights',
        initializer=init_op,
        reuse=tf.AUTO_REUSE)


def get_conv_layer_without_bias(inputs, kernel_size, strides, filters):
    w = kernel_size[0]
    h = kernel_size[1]
    chn_in = inputs.shape[-1]
    chn_out = filters
    weights = get_conv_weights(w, h, chn_in, chn_out)
    return tf.nn.conv2d(inputs,
                        weights,
                        strides,
                        padding='same',
                        bias=None,
                        activation=None)



class IINN(object):
    def __init__(self, dim_x, dim_y,
                 conv_dims, fc_dims, att_dims):
        self.layers = []
        self.in_port = tf.placeholder(dim_x)
        self.out_port = tf.placeholder(dim_y)
        self.layers.append(in_port)
        with tf.name_scope('recognition'):
            for i in range(conv_dims):
                with tf.name_scope('layer%d' % i):
                    get_conv_layer_without_bias(
                        self.layers[-1],
                        conv_dims[i]['ksize'],
                        conv_dims[i]['strides'],
                        conv_dims[i]['filters'])

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
    dim_x = [1, None, None, 3]
    dim_y = [1, n_class]
    # configure the convolution dimensions
    conv_dims = [dict()] * 4
    conv_dims['ksize'] = (3, 3)
    conv_dims['strides'] = (2, 2)
    conv_dims['filters'] = 8

    return IINN(dim_x, dim_y,
                conv_dims,
                fc_dims,
                att_dims)


def Train_IINN(iinn_, data, model_path):
    xx = data['input']
    yy = data['output']
    plt.imshow(xx[0])
    plt.show()
    print(yy[0])
    print(xx.shape)
    print(yy.shape)


def Test_IINN(iinn_, data, model_path):
    xx = data['input']
    yy = data['output']

    plt.imshow(xx[0])
    plt.show()
    print(yy[0])
    print(xx.shape)
    print(yy.shape)

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
    data_train, data_test = \
        dataset.cifar10.Load_CIFAR10('../Datasets/CIFAR10/')
    model_path = '../Models/CIFAR10-IINN/'
    Train_IINN(iinn_, data_train, model_path)
    # test the trained model with test split of the same dataset
    Test_IINN(iinn_, data_test, model_path)

    # TO-DO
    # method 1: use label y to control bias for each channel in each layer(leaky relu)
    # method 2: use label y to control the mask for each channel in each layer(Non-zero)
    # method 3: use both x and y to control the attention mask for only input