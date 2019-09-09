# Using stacked Self Organizing Maps to classifier and generate images

import numpy as np
from components import utils
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import glob
import abc


class Model(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def getInputPlaceHolder(self):
        pass

    @abc.abstractmethod
    def getFeedbackPlaceHolder(self):
        pass

    @abc.abstractmethod
    def getOutputOp(self):
        pass

    @abc.abstractmethod
    def getTrainableVariables(self):
        pass

    @abc.abstractmethod
    def getGlobalVariables(self):
        pass

    @abc.abstractmethod
    def getLocalVariables(self):
        pass


class AutoEncoder(Model):
    def __init__(self, input_chn, kernels, sizes, strides):
        super(AutoEncoder, self).__init__()
        # building blocks
        self.layers = []
        layer = tf.placeholder(
            shape=[None, None, None, input_chn],
            dtype=tf.float32)
        self.layers.append(layer)
        for i in range(len(kernels)):
            layer = tf.layers.conv2d(
                inputs=self.layers[-1],
                filters=kernels[i],
                kernel_size=sizes[i],
                strides=strides[i],
                use_bias=True,
                activation=tf.nn.relu)



def TrainModel(model, path, opt='SGD', lr=1e-4):
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
        feed = model.getFeedbackPlaceHolder()
        vars = model.getTrainableVariables()
        saver = tf.train.Saver(var_list=vars)
        sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    # train the network with unlabeled examples, actually, the label is also a kind of input
    files = glob.glob('E:/Gits/Datasets/Umbrella/seq-in/*.jpg')[0:30:10]
    files += glob.glob('E:/Gits/Datasets/Umbrella/seq-out/*.jpg')[0:30:10]
    images = [None] * len(files)
    for i in range(len(files)):
        images[i] = np.array(Image.open(files[i]), np.float32) / 255.0
    print('Dataset Loaded!')

    # create a AutoEncoder
    auto_encoder = AutoEncoder(input_chn=4, kernels=[8, 8, 8, 8], sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2])
    # train the AE with unlabeled samples
    TrainModel(auto_encoder, '../../Models/SemanticSegmentation/' 'SGD', 1e-3)
