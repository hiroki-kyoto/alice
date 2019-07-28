import numpy as np
import tensorflow as tf
import json
import components.utils as utils

''''
Net configuration demo:
net_conf = dict(classes=2,
                inputs=[1, 256, 256, 3],
                filters=[8, 16, 16, 8],
                ksizes=[3, 3, 3, 3],
                strides=[2, 2, 2, 2],
                relus=[0, 1, 0, 1],
                links=[[], [], [], [0]],
                fc=[8, 32, 8],
                tanh=[0, 1, 0])
'''


class Classifier(object):
    def __init__(self, conf, input_=None):
        if isinstance(conf, dict):
           pass
        elif isinstance(conf, str):
            conf = json.loads(conf)
        else:
            assert False
        self.graph = tf.Graph()
        self.layers = []
        with self.graph.as_default():
            self.sess = tf.Session()
            if input_ is None:
                self.input_ = tf.placeholder(dtype=tf.float32, shape=conf['inputs'])
            else:
                self.input_ = input_
            _layer = self.input_
            self.layers.append(_layer)
            # add convolution layers
            assert len(conf['filters']) > 0
            for _conv_idx in range(len(conf['filters'])):
                for _layer_id in conf['links'][_conv_idx]:
                    down_scale = (self.layers[_layer_id].shape.as_list()[1] / _layer.shape.as_list()[1],
                                  self.layers[_layer_id].shape.as_list()[2] / _layer.shape.as_list()[2])
                    _layer = tf.concat([_layer, utils.down_sample(self.layers[_layer_id], down_scale)], axis=-1)
                _layer = tf.layers.conv2d(
                    inputs=_layer,
                    filters=conf['filters'][_conv_idx],
                    kernel_size=conf['ksizes'][_conv_idx],
                    strides=conf['strides'][_conv_idx],
                    padding='same',
                    dilation_rate=1,
                    activation=[None, tf.nn.relu][conf['relus'][_conv_idx]],
                    use_bias=True,
                    kernel_constraint=tf.initializers.truncated_normal(0.0, 0.04))
                self.layers.append(_layer)
            # add fully connected layers
            # to reshape the convolution 4-D tensor into 2-D matrix
            _shape = _layer.shape.as_list()
            _layer = tf.reshape(_layer, shape=[_shape[0], _shape[1]*_shape[2]*_shape[3]])
            for _fc_id in range(len(conf['fc'])):
                _layer = tf.layers.dense(
                    inputs=_layer,
                    units=conf['fc'][_fc_id],
                    activation=[None, tf.nn.tanh][conf['tanh'][_fc_id]],
                    use_bias=True,
                    kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04))
                self.layers.append(_layer)
            # the last fc layer for categorical presentation
            _layer = tf.layers.dense(
                inputs=_layer,
                units=conf['classes'],
                activation=None,
                use_bias=True,
                kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04))
            # add a softmax layer to determine the class of the input
            _layer = tf.nn.softmax(_layer, axis=-1)
            self.layers.append(_layer)

    def load_model(self, path):
        pass

    def learn(self, data):
        pass

    def inference(self, data):
        pass

    def finalize(self):
        pass