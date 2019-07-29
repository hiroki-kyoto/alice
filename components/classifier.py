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

# conf: a JSON format text to build a network
# input_: the input tensor, could be a variable, or None(by this way, a default variable with
# shape specified in JSON string will be created, and its setter will be created).
# feedback: the feedback tensor, could be a placeholder or None(by this way, a default placeholder
# with shape specified as the same as that of the output tensor).
# optimize_param: True means the optimizer will update the parameters on training, otherwise not.
# optimize_input: True means the optimizer will update the input variable on training. On which,
# the type of input_ will be checked, if it is not a variable, it will set as False automatically.
class Classifier(object):
    def __init__(self, conf, input_=None, feedback=None, optimize_param=True, optimize_input=False, lr=1e-3):
        if isinstance(conf, dict):
           pass
        elif isinstance(conf, str):
            conf = json.loads(conf)
        else:
            assert False
        self.layers = []
        self.lr = lr
        with tf.Session() as self.sess:
            if input_ is None:
                # the internal state of input
                self._input = utils.create_variable("input", shape=conf['inputs'], trainable=True)
            else:
                self._input = input_
            # check the input type
            if not isinstance(self._input, tf.Variable):
                optimize_input = False # by no means can we optimize a non-variable tensor
            else:
                self.input_ = tf.placeholder(dtype=self._input.dtype, shape=self._input.shape)
                self.input_setter = tf.assign(self._input, self.input_) # dependency is required later
            # add the input setter to the layer collection
            _layer = self._input
            self.layers.append(_layer)
            # add convolution layers
            assert len(conf['filters']) > 0
            for _conv_idx in range(len(conf['filters'])):
                for _layer_id in conf['links'][_conv_idx]:
                    down_scale = (int(self.layers[_layer_id].shape.as_list()[1] // _layer.shape.as_list()[1]),
                                  int(self.layers[_layer_id].shape.as_list()[2] // _layer.shape.as_list()[2]))
                    _layer = tf.concat([_layer, utils.down_sample(self.layers[_layer_id], down_scale)], axis=-1)
                _layer = tf.layers.conv2d(
                    inputs=_layer,
                    filters=int(conf['filters'][_conv_idx]),
                    kernel_size=conf['ksizes'][_conv_idx],
                    strides=conf['strides'][_conv_idx],
                    padding='same',
                    dilation_rate=1,
                    activation=[None, tf.nn.relu][conf['relus'][_conv_idx]],
                    use_bias=True,
                    kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04))
                self.layers.append(_layer)
            # add fully connected layers
            # to reshape the convolution 4-D tensor into 2-D matrix
            _shape = _layer.shape.as_list()
            _layer = tf.reshape(_layer, shape=[_shape[0], int(_shape[1]*_shape[2]*_shape[3])])
            for _fc_id in range(len(conf['fc'])):
                _layer = tf.layers.dense(
                    inputs=_layer,
                    units=conf['fc'][_fc_id],
                    activation=[None, tf.nn.tanh][conf['tanh'][_fc_id]],
                    use_bias=True,
                    kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04))
                self.layers.append(_layer)
            # the last fc layer for categorical presentation
            self.output = tf.layers.dense(
                inputs=_layer,
                units=conf['classes'],
                activation=None,
                use_bias=True,
                kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04))
            _layer = self.output
            self.layers.append(_layer)
            # deal with feedback
            if feedback is not None:
                assert not isinstance(feedback, tf.Variable)
                self.feedback = feedback
            else:
                self.feedback = tf.placeholder(dtype=tf.float32, shape=_layer.shape.as_list())
            # add optimizer
            if not optimize_param:
                all_vars = [self._input]
            else:
                all_vars = tf.trainable_variables()
            vars = []
            if not optimize_input:
                for v in all_vars:
                    if v.name != self._input.name:
                        vars.append(v)
            else:
                vars = all_vars
            # calculate the cost of a training instance
            self.cost = tf.losses.softmax_cross_entropy(self.feedback, self.output)
            if optimize_param or optimize_input:
                self.cost_minimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr).minimize(self.cost, var_list=vars)

    def load(self, path): # load only inference parameters, excluding the optimizer parameters
        saver = tf.train.Saver(var_list=)
        pass

    def recover(self, path): # load the trained inference parameters, including optimizer parameters
        saver = tf.train.Saver(var_list=)
        pass

    def dump(self, path):
        pass

    def train(self, data):
        pass

    def test(self, data):
        pass

    def close(self):
        self.sess.close()
        pass
