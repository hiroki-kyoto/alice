import numpy as np
import tensorflow as tf
import json
import components.utils as utils
import matplotlib.pyplot as plt

from abc import ABCMeta


# conf: a JSON format text to build a network
# input_: the input tensor, could be a variable, or None(by this way, a default variable with
# shape specified in JSON string will be created, and its setter will be created).
# feedback: the feedback tensor, could be a placeholder or None(by this way, a default placeholder
# with shape specified as the same as that of the output tensor).
# optimize_param: True means the optimizer will update the parameters on training, otherwise not.
# optimize_input: True means the optimizer will update the input variable on training. On which,
# the type of input_ will be checked, if it is not a variable, it will set as False automatically.


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, conf, optimize_input=False, lr=1e-4):
        self.sess = None
        self.saver_train = None
        self.saver_test = None
        self._input = None
        self.input_ = None
        self.optimize_input = optimize_input
        self.input_setter = None
        self.output = None
        self.lr = lr
        self.layers = []

        if isinstance(conf, dict):
            pass
        elif isinstance(conf, str):
            conf = json.loads(conf)
        else:
            assert False
        self.sess = tf.Session()
        self.optimize_input = optimize_input

        if self.optimize_input:
            self._input = utils.create_variable(name='input', shape=conf['inputs'], trainable=True)
            self.input_ = tf.placeholder(dtype=tf.float32, shape=conf['inputs'])
            self.input_setter = tf.assign(self._input, self.input_)  # dependency is required later
        else:
            self._input = tf.placeholder(name='input', dtype=tf.float32, shape=conf['inputs'])

        # add the input setter to the layer collection
        self.layers.append(self._input)

    def init_blank_model(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self, path):  # load only inference parameters, excluding the optimizer parameters
        if tf.train.checkpoint_exists(path):
            self.saver_test.restore(self.sess, path)
            # initialize the rest uninitialized variables
            utils.initialize_uninitialized(self.sess)
        else:
            assert False

    def recover(self, path):  # load the trained inference parameters, including optimizer parameters
        if tf.train.checkpoint_exists(path):
            self.saver_train.restore(self.sess, path)
            # initialize the rest uninitialized variables
            utils.initialize_uninitialized(self.sess)
        else:
            assert False

    def save(self, path):
        self.saver_test.save(self.sess, path)

    def dump(self, path):
        self.saver_train.save(self.sess, path)

    def train(self,
              train_images,
              train_labels,
              stop_precision=1e-3,
              max_epoc=100,
              valid_images=None,
              valid_labels=None):
        print('Not implemented!')
        assert False

    def test(self, im):  # SINGLE INSTANCE INFERENCE IS SUPPORTED ONLY!
        assert im.shape[0] == 1
        assert self._input.shape.as_list()[0] == 1  # MODEL ESTABLISHED IN NO BATCH MODE
        if self.optimize_input:
            self.sess.run(self.input_setter, feed_dict={
                self.input_: im
            })
            return self.sess.run(self.output)
        else:
            return self.sess.run(self.output, feed_dict={self._input: im})

    def close(self):
        self.sess.close()

    def info(self):
        _desc = ''
        for layer in self.layers:
            _desc += (layer.name + ": " + str(layer.shape) + '\n')
        return _desc
