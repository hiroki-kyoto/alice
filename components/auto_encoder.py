import numpy as np
import tensorflow as tf
import json
import components.utils as utils
import components.model as model
import matplotlib.pyplot as plt

'''
Net Configuration Demo:

encoder_conf = dict(
    inputs=[1, 256, 256, 3],
    filters=[8, 16, 32, 16, 8],
    ksizes=[3, 3, 3, 3, 3],
    strides=[2, 2, 2, 2, 2],
    conv_relus=[0, 1, 0, 1, 0],
    links=[[], [], [], [], []],
    fc=[16, 32, 16],
    fc_relus=[1, 0, 1]
)
                
decoder_conf = dict(
    inputs=[1, 16],
    fc=[16, 32, 16],
    fc_relus=[1, 0, 1],
    filters=[8, 16, 32, 16, 8],
    ksizes=[3, 3, 3, 3, 3],
    strides=[2, 2, 2, 2, 2],
    conv_relus=[0, 1, 0, 1, 0],
    links=[[], [], [], [], []],
)
'''


class AutoEncoder(model.Model):
    def __init__(self, conf, optimize_input=False, lr=1e-4):
        super(AutoEncoder, self).__init__(conf, optimize_input, lr)

        _layer = self._input
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
        self.layers.append(_layer)

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

        self.lr = lr
        self.feedback = tf.placeholder(dtype=tf.float32, shape=self.layers[-1].shape.as_list())
        self.cost = tf.losses.softmax_cross_entropy(self.feedback, self.output)

        # add optimizer
        if self.optimize_input:
            vars = [self._input]
        else:
            all_vars = tf.trainable_variables()
            vars = []
            for v in all_vars:
                if v.name != self._input.name:
                    vars.append(v)
        self.cost_minimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost, var_list=vars)

        # set up the model saver except the training components
        all_vars = tf.trainable_variables()
        vars = []
        for v in all_vars:
            if v.name != self._input.name:
                vars.append(v)
        self.saver_test = tf.train.Saver(var_list=vars)
        # set up the model saver with the training components
        all_vars = tf.global_variables()
        vars = []
        for v in all_vars:
            if v.name != self._input.name:
                vars.append(v)
        self.saver_train = tf.train.Saver(var_list=vars)

    def init_blank_model(self):
        self.sess.run(tf.global_variables_initializer())

    def load(self, path): # load only inference parameters, excluding the optimizer parameters
        if tf.train.checkpoint_exists(path):
            self.saver_test.restore(self.sess, path)
            # initialize the rest uninitialized variables
            utils.initialize_uninitialized(self.sess)
        else:
            assert False

    def recover(self, path): # load the trained inference parameters, including optimizer parameters
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

    def train(self, train_images, train_labels, stop_precision=1e-3, max_epoc=100, valid_images=None, valid_labels=None):
        assert len(train_images) == len(train_labels)
        assert len(valid_images) == len(valid_labels)
        batch_size = self._input.shape[0]
        if not self.optimize_input:
            epoc = 0
            batch_num = len(train_images) // batch_size
            batch_num_valid = len(valid_images) // batch_size
            assert batch_num > 0
            loss_hist = np.zeros([batch_num]) + 1.0
            loss_hist_valid = np.zeros([batch_num_valid]) + 1.0
            loss_epoc = np.zeros([max_epoc])
            loss_epoc_valid = np.zeros([max_epoc])
            while np.mean(loss_hist) > stop_precision and epoc < max_epoc:
                seq = np.random.permutation(len(train_images))
                for i in range(batch_num):
                    batch_im = train_images[seq[i*batch_size:(i+1)*batch_size]]
                    batch_lb = train_labels[seq[i*batch_size:(i+1)*batch_size]]
                    _, output_ = self.sess.run([self.cost_minimizer, self.output], feed_dict={
                        self._input: batch_im,
                        self.feedback: batch_lb
                    })
                    loss_hist[i] = np.mean(np.argmax(output_, axis=-1) != np.argmax(batch_lb, axis=-1))
                # check the validation error, which is defined by 1-accuracy
                for i in range(batch_num_valid):
                    batch_im = valid_images[i*batch_size:(i+1)*batch_size]
                    batch_lb = valid_labels[i*batch_size:(i+1)*batch_size]
                    output_ = self.sess.run(self.output, feed_dict={
                        self._input: batch_im
                    })
                    loss_hist_valid[i] = np.mean(np.argmax(output_, axis=-1) != np.argmax(batch_lb, axis=-1))
                loss_epoc[epoc] = np.mean(loss_hist)
                loss_epoc_valid[epoc] = np.mean(loss_hist_valid)
                print('EPOC#%d\tLOSS=%.5f/%.5f VALID=%.5f' % (epoc, loss_epoc[epoc], stop_precision, loss_epoc_valid[epoc]))
                epoc += 1
                plt.clf()
                plt.plot(loss_epoc[:epoc], 'r-')
                plt.plot(loss_epoc_valid[:epoc], 'b--')
                plt.xticks(np.arange(0, max_epoc, max_epoc / 10))
                plt.yticks(np.arange(0, 1.0, 1.0 / 10))
                plt.axis([0, max_epoc, 0, 1.0])
                plt.legend(['train', 'valid'])
                plt.pause(0.01)
        else:
            assert batch_size == len(train_images)
            epoc = 0
            loss_epoc = np.zeros([max_epoc])
            loss_ = stop_precision + 1.0
            self.sess.run(self.input_setter, feed_dict={
                self.input_: train_images
            })
            while np.mean(loss_) > stop_precision and epoc < max_epoc:
                _, loss_ = self.sess.run([self.cost_minimizer, self.cost], feed_dict={
                    self.feedback: train_labels
                })
                loss_epoc[epoc] = loss_
                print('ITER#%d\tLOSS=%.5f/%.5f' % (epoc, loss_, stop_precision))
                epoc += 1
                plt.clf()
                plt.plot(loss_epoc[:epoc], 'r-')
                plt.xticks(np.arange(0, max_epoc, max_epoc / 10))
                plt.yticks(np.arange(0, 1.0, 1.0 / 10))
                plt.axis([0, max_epoc, 0, 1.0])
                plt.legend(['train'])
                plt.pause(0.01)

    def test(self, im):     # SINGLE INSTANCE INFERENCE IS SUPPORTED ONLY!
        assert im.shape[0] == 1
        assert self._input.shape.as_list()[0] == 1    # MODEL ESTABLISHED IN NO BATCH MODE
        if self.optimize_input:
            self.sess.run(self.input_setter, feed_dict={
                self.input_: im
            })
            return self.sess.run(self.output)
        else:
            return self.sess.run(self.output, feed_dict={self._input: im})

    def close(self):
        self.sess.close()
        pass

    def info(self):
        _desc = ''
        for layer in self.layers:
            _desc += (layer.name + ": " + str(layer.shape) + '\n')
        return _desc
