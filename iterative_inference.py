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


def get_nonlinear_layer(inputs):
    return tf.nn.leaky_relu(inputs, alpha=0.2)


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


def get_controlled_layer(inputs, control): # define your own control strategy here
    return tf.nn.bias_add(inputs, control)


def get_loss(outputs, feedbacks):
    return tf.nn.softmax_cross_entropy_with_logits_v2(None, feedbacks, outputs)


def convert_tensor_conv2fc(tensor): # issue: use max or mean for pooling?
    return tf.reduce_mean(tensor, axis=[1, 2])


class IINN(object):
    def __init__(self, dim_x, dim_y,
                 conv_config, fc_config, att_config):
        self.inputs = tf.placeholder(shape=dim_x, dtype=tf.float32)
        self.feedbacks = tf.placeholder(shape=dim_y, dtype=tf.float32)

        self.rec_layers = []
        self.rec_layers.append(self.inputs)

        self.att_layers = []
        self.att_layers.append(self.feedbacks)

        self.ctl_layers = []

        scope = 'attention'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # attention module
            sub_scope = 'fc_%d'
            for i in range(len(att_config)):
                with tf.variable_scope(sub_scope % i, reuse=tf.AUTO_REUSE):
                    fc_ = get_fc_layer(
                        self.att_layers[-1],
                        att_config[i]['units'])
                    fc_ = get_nonlinear_layer(fc_)
                    self.att_layers.append(fc_)
            # bridge tensor between attention to biases of conv
            num_biases = 0
            for i in range(len(conv_config)):
                num_biases += conv_config[i]['filters']
            with tf.variable_scope(sub_scope % len(att_config)):
                fc_ = get_fc_layer(
                    self.att_layers[-1],
                    num_biases)
                assert fc_.shape.as_list()[0] == 1
                self.att_layers.append(fc_[0])

        scope = 'control'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # creating sub operations with back-prop killed
            conv_bias_ctl = []
            offset = 0
            for i in range(len(conv_config)):
                ctl_grad_free = tf.stop_gradient(
                    self.att_layers[-1][offset:offset + conv_config[i]['filters']])
                self.ctl_layers.append(ctl_grad_free)
                assert conv_config[i]['filters'] == ctl_grad_free.shape.as_list()[0]
                offset += ctl_grad_free.shape.as_list()[0]

        scope = 'recognition'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sub_scope = 'conv_%d'
            for i in range(len(conv_config)):
                with tf.variable_scope(sub_scope % i):
                    conv_ = get_conv_layer(
                        self.rec_layers[-1],
                        conv_config[i]['ksize'],
                        conv_config[i]['strides'],
                        conv_config[i]['filters'])
                    conv_ = get_controlled_layer(conv_, self.ctl_layers[i])
                    conv_ = get_nonlinear_layer(conv_)
                    self.rec_layers.append(conv_)
            # bridge tensor between conv and fc to let it flow thru
            layer = convert_tensor_conv2fc(self.rec_layers[-1])
            self.rec_layers.append(layer)

            # creating classifier using fc layers
            sub_scope = 'fc_%d'
            for i in range(len(fc_config)):
                with tf.variable_scope(sub_scope % i):
                    fc_ = get_fc_layer(
                        self.rec_layers[-1],
                        fc_config[i]['units'])
                    fc_ = get_nonlinear_layer(fc_)
                    self.rec_layers.append(fc_)
            # the last classifier layer -- using fc without nonlinearization
            with tf.variable_scope(sub_scope % len(fc_config)):
                self.outputs = get_fc_layer(self.rec_layers[-1], dim_y[1])
            self.rec_layers.append(self.outputs)

            # calculate the loss
            self.rec_loss = get_loss(self.outputs, self.feedbacks)
            self.att_loss = None


        # network self check
        print("============================ VARIABLES ===============================")
        vars = tf.global_variables()
        for i in range(len(vars)):
            print("var#%03d:%32s %16s %12s" %
                  (i, vars[i].name[:-2], vars[i].shape, str(vars[i].dtype)[9:-6]))
        print("======================================================================")
        print("\n")
        print("============================ OPERATORS ===============================")
        ops = self.rec_layers
        for i in range(len(ops)):
            print("opr#%03d:%32s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        ops = self.att_layers
        for i in range(len(ops)):
            print("opr#%03d:%32s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        ops = self.ctl_layers
        for i in range(len(ops)):
            print("opr#%03d:%32s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        print("======================================================================")

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

    # PLEASE ADD ACTIVATION FUNCTION TO EACH LAYER!!!

    # Use dual path to train with or without attention

    # Two approach:
    # 1st- inspired by the concept of co-activated neural group, attention is a phase locked loop.
    # 2nd- impsired by the system of yinyang-GAN, the HU system, decoder pass grads to encoder.

    # for the 1st approach:
    # 1> when argmax(y) == argmax(y_{gt}), attention module is trained to converge at y = y_{gt}
    # 2> otherwise, attention module is trained to output y_{n+1}=y-y_n till y = \vec{0}.

    # Next, add optimizer and begin to train this network!