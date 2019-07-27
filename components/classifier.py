import numpy as np
import tensorflow as tf
import json

net_conf = dict(classes=2,
                filters=[8, 16, 16, 8],
                ksizes=[3, 3, 3, 3],
                strides=[2, 2, 2, 2],
                relus=[0, 1, 0, 1],
                links=[[], [], [], [0]],
                fc=[8, 32, 8])


def down_sample(x, down_scale):
    return tf.layers.average_pooling2d(x, down_scale + down_scale//2, down_scale, 'same')



class Classifier(object):
    def __init__(self, conf):

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
            self.input_ = tf.placeholder(dtype=tf.float32)
            _layer = self.input_
            self.layers.append(_layer)
            # add convolution layers
            assert len(conf['filters']) > 0
            for _conv_idx in range(conf['filters']):
                for _layer_id in conf['links'][_conv_idx]:
                    down_scale = (self.layers[_layer_id].shape[1] / _layer.shape[1],
                                  self.layers[_layer_id].shape[2] / _layer.shape[2])
                    _layer = tf.concat([_layer, down_sample(self.layers[_layer_id], down_scale)], axis=-1)
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
            pass
        pass

    def load_model(self, path):
        pass

    def learn(self, data):
        pass

    def inference(self, data):
        pass

    def finalize(self):
        pass