import numpy as np
import tensorflow as tf
from PIL import Image
import glob

INPUT_SIZE = (256, 256)
BATCH_SIZE = 6

class Classifier(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.layers = []
        with self.graph.as_default():
            # build a graph...
            self.sess = tf.Session()
            self.input_ = tf.placeholder(
                dtype=tf.float32,
                shape=[-1, INPUT_SIZE[0], INPUT_SIZE[1], 3])
            # layer 1, 3x3 2x2 with relu, 8 channels
            _layer = tf.layers.conv2d(
                inputs=self.input_,
                filters=8,
                kernel_size=3,
                strides=2,
                padding='valid',
                dilation_rate=1,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_constraint=tf.initializers.truncated_normal(0.0, 0.04))
            self.layers.append(_layer)
            # layer 2, 3x3 2z2 with relu, 16 channels
            _layer = tf.layers.conv2d(
                inputs=_layer,
                filters=16,
                kernel_size=3,
                strides=2,
                padding='valid',
                dilation_rate=1,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_constraint=tf.initializers.truncated_normal(0.0, 0.04))
            self.layers.append(_layer)
            # 3x3 2x2 batch norm, relu, 16 channels
            _layer = tf.layers.conv2d(
                inputs=_layer,
                filters=16,
                kernel_size=3,
                strides=2,
                padding='valid',
                dilation_rate=1,
                activation=None,
                use_bias=True,
                kernel_constraint=tf.initializers.truncated_normal(0.0, 0.04))
            _layer = tf.layers.batch_normalization(_layer, trainable=True)
            _layer = tf.nn.relu(_layer)
            self.layers.append(_layer)
            # another block here...
        pass

    def load_model(self, path):
        pass

    def learn(self, data):
        pass

    def inference(self, data):
        pass

    def finalize(self):
        pass


if __name__ == '__main__':
    path_ = "../Datasets/Hands"
    hands = glob.glob(path_ + "/with-hand/*.JPG")
    blank = glob.glob(path_ + "/without-hand/*.JPG")
    print("hands=" + str(len(hands)))
    print("blank=" + str(len(blank)))
    for fn_ in blank:
        im_ = Image.open(fn_)
        im_ = im_.resize(INPUT_SIZE)
        pass
    # build a classifier network
    cf = Classifier()
