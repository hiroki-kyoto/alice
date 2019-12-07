# iterative_inference.py
# NN inference in an iterative manner, instead of a forward single shot.

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import dataset

import tensorflow as tf


def get_conv_weights(w, h, chn_in, chn_out):
    dim = [w, h, chn_in, chn_out]
    init_op = tf.truncated_normal(dim, 0.02)
    return tf.get_variable(
        name='weights',
        initializer=init_op)


def get_fc_weights(chn_in, chn_out):
    dim = [chn_in, chn_out]
    init_op = tf.truncated_normal(dim, 0.02)
    return tf.get_variable(
        name='weights',
        initializer=init_op)


def get_bias(filters):
    init_op = tf.zeros([filters], dtype=tf.float32)
    return tf.get_variable(
        name='bias',
        initializer=init_op)


def get_conv_layer(inputs, kernel_size, strides, filters):
    w = kernel_size[0]
    h = kernel_size[1]
    chn_in = inputs.shape.as_list()[-1]
    chn_out = filters
    weights = get_conv_weights(w, h, chn_in, chn_out)
    bias = get_bias(chn_out)
    layer = tf.nn.conv2d(inputs, weights, strides, padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    return layer


def get_fc_layer(inputs, units):
    chn_in = inputs.shape.as_list()[-1]
    chn_out = units
    weights = get_fc_weights(chn_in, chn_out)
    bias = get_bias(chn_out)
    layer = tf.matmul(inputs, weights)
    layer = tf.nn.bias_add(layer, bias)
    return layer


def convert_tensor_conv2fc(tensor): # issue: use max or mean for pooling?
    return tf.reduce_mean(tensor, axis=[1, 2])


class IINN(object):
    def __init__(self, dim_x, dim_y,
                 conv_config, fc_config, att_config):
        self.layers = []
        self.in_port = tf.placeholder(shape=dim_x, dtype=tf.float32)
        self.out_port = tf.placeholder(shape=dim_y, dtype=tf.float32)
        self.layers.append(self.in_port)

        scope = 'recognition'
        with tf.name_scope(scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                sub_scope = 'conv_%d'
                for i in range(len(conv_config)):
                    with tf.name_scope(sub_scope % i):
                        with tf.variable_scope(sub_scope % i):
                            conv_ = get_conv_layer(
                                self.layers[-1],
                                conv_config[i]['ksize'],
                                conv_config[i]['strides'],
                                conv_config[i]['filters'])
                            self.layers.append(conv_)
                # bridge tensor between conv and fc to let it flow thru
                layer = convert_tensor_conv2fc(self.layers[-1])
                self.layers.append(layer)
                sub_scope = 'fc_%d'
                for i in range(len(fc_config)):
                    with tf.name_scope(sub_scope % i):
                        with tf.variable_scope(sub_scope % i):
                            fc_ = get_fc_layer(
                                self.layers[-1],
                                fc_config[i]['units'])
                            self.layers.append(fc_)

        # network self check
        print("===================== VARIABLES ======================")
        vars = tf.global_variables()
        for i in range(len(vars)):
            print("var#%d:\t%s\t%s" % (i, vars[i].name, vars[i].shape))
        print("======================================================")
        print("\n")
        print("===================== OPERATORS ======================")
        for i in range(len(self.layers)):
            print("op#%d:\t%s" % (i, self.layers[i]))
        print("======================================================")

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


def new_conv_config(k_w, k_h, s_w, s_h, filters):
    demo_config = dict()
    demo_config['ksize'] = (k_w, k_h)
    demo_config['strides'] = (1, s_w, s_h, 1)
    demo_config['filters'] = filters
    return demo_config

def new_fc_config(units):
    demo_config = dict()
    demo_config['units'] = units
    return demo_config


def Build_IINN(n_class):
    dim_x = [1, None, None, 3]
    dim_y = [1, n_class]

    # configure the convolution layers
    n_conv = 4
    conv_config = [None] * n_conv
    for i in range(n_conv):
        conv_config[i] = new_conv_config(3, 3, 2, 2, 8 << i)

    # configure the fully connectied layers
    n_fc = 3
    fc_config = [None] * n_fc
    for i in range(n_fc):
        fc_config[i] = new_fc_config(16 << i)

    # configure the special module : feedback attention
    n_att = 3
    att_config = [None] * n_att
    for i in range(n_att):
        att_config[i] = new_fc_config(64 >> i)

    return IINN(dim_x, dim_y,
                conv_config,
                fc_config,
                att_config)


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

    # question remained:
    # Is it feasible to replace the bias with attention output or simply add a new bias before activation?
    # [solved] add new bias instead of replace: activate(conv + conv bias + attention bias)