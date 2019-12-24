# iinn.py
# iterative inference neural network
# NN inference in an iterative manner, instead of a forward single shot.

import numpy as np
import os
import platform
import matplotlib.pyplot as plt

import dataset
import components.utils as utils

import tensorflow as tf


def get_conv_weights(h, w, chn_in, chn_out):
    dim = [h, w, chn_in, chn_out]
    init_op = tf.truncated_normal(dim, mean=0.0, stddev=0.1)
    return tf.get_variable(
        name='weights',
        initializer=init_op)


def get_deconv_weights(h, w, chn_in, chn_out):
    dim = [h, w, chn_out, chn_in]
    init_op = tf.truncated_normal(dim, mean=0.0, stddev=0.1)
    return tf.get_variable(
        name='weights',
        initializer=init_op)


def get_fc_weights(chn_in, chn_out):
    dim = [chn_in, chn_out]
    init_op = tf.truncated_normal(dim, mean=0.0, stddev=0.1)
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


def get_deconv_layer(inputs, kernel_size, strides, filters):
    k_h = kernel_size[0]
    k_w = kernel_size[1]
    chn_in = inputs.shape.as_list()[-1]
    ch_out = filters
    weights = get_deconv_weights(k_h, k_w, chn_in, chn_out)
    bias = get_bias(chn_out)
    out_shape = [inputs.shape.as_list()[1] * strides[0],
                 inputs.shape.as_list()[2] * strides[1]]
    layer = tf.nn.deconv2d(inputs, weights, out_shape, strides, padding='SAME')
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


def get_XEntropy_distance(outputs, feedbacks):
    return tf.nn.softmax_cross_entropy_with_logits_v2(None, feedbacks, outputs)


def get_Manhattan_distance(outputs, feedbacks):
    return tf.reduce_mean(tf.abs(outputs - feedbacks))


def get_Euclidean_distance(outputs, feedbacks):
    return tf.reduce_mean(tf.square(outputs - feedbacks))


def get_Chebyshev_distance(outputs, feedbacks):
    return tf.reduce_max(tf.abs(outputs - feedbacks))


def get_Minkowski_distance(outputs, feedbacks, norm):
    return tf.reduce_mean(tf.pow(tf.abs(feedbacks - outputs), norm))


def get_Cosine_distance(outputs, feedbacks):
    assert len(outputs.shape) == 2
    assert len(feedbacks.shape) == 2
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(outputs), axis=1) + 1E-8)
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(feedbacks), axis=1) + 1E-8)
    a_dot_b = tf.reduce_sum(a * b, axis=1)
    return tf.reduce_mean(1 - a_dot_b / (a_norm * b_norm))


def mat2vec(mat): # convert 4D tensor into 2D
    n, h, w, c = mat.shape.as_list()[0:4]
    return tf.reshape(tf.space2depth(mat, [h, w]), [n, h*w*c])


def vec2mat(vec, h, w): # convert 2D tensor into 4D
    assert vec.shape.as_list()[1] % (h*w) == 0
    n, d = vec.shape.as_list()[0:2]
    c = d // (h * w)
    return tf.reshape(tf.depth2space(vec, [h, w]), [n, h, w, c])


class IINN(object):
    def decodeA(self, conv_config):
        self.decoderA.append(self.inputA)
        scope = 'decoderA'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sub_scope = 'conv_%d'
            for i in range(len(conv_config)):
                with tf.variable_scope(sub_scope % i):
                    conv_ = get_conv_layer(
                        self.decoderA[-1],
                        conv_config[i]['ksize'],
                        conv_config[i]['strides'],
                        conv_config[i]['filters'])
                    conv_ = get_nonlinear_layer(conv_)
                    self.decoderA.append(conv_)
            # bridge tensors between conv and fc
            self.codeA = mat2vec(self.decoderA[-1])
            self.decoderA.append(self.codeA)


    def encodeA(self, conv_config):
        self.encoderA.append(self.codeA)
        scope = 'encoderA'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # bridge: convert tensors from 2D to 4D ( extended on W x H )
            base_h, base_w = self.decoderA[-2].shape.as_list()[1:3]
            mat_ = vec2mat(self.encoderA[-1], base_h, base_w)
            self.encoderA.append(mat_)

            sub_scope = 'deconv_%d'
            for i in range(len(conv_config)-1):
                with tf.variable_scope(sub_scope % i):
                    deconv_ = get_deconv_layer(
                        self.encoderA[-1],
                        conv_config[len(conv_config) - 1 - i]['ksize'],
                        conv_config[len(conv_config) - 1 - i]['strides'],
                        conv_config[len(conv_config) - 2 - i]['filters'])
                    deconv_ = get_nonlinear_layer(deconv_)
                    self.encoderA.append(deconv_)
            with tf.variable_scope(sub_scope % (len(conv_config) - 1)):
                self.inputA2A = get_deconv_layer(
                    self.encoderA[-1],
                    conv_config[0]['ksize'],
                    conv_config[0]['strides'],
                    self.inputA.shape.as_list()[3])
                self.encoderA.append(self.inputA2A)


    def transformA2B(self, fc_config):
        self.transA2B.append(self.codeA)
        scope = 'transA2B'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sub_scope = 'fc_%d'
            for i in range(len(fc_config)):
                with tf.variable_scope(sub_scope % i):
                    fc_ = get_fc_layer(
                        self.transA2B[-1],
                        fc_config[i]['units'])
                    fc_ = get_nonlinear_layer(fc_)
                    self.transA2B.append(fc_)
            # the last classifier layer -- using fc without nonlinearization
            with tf.variable_scope(sub_scope % len(fc_config)):
                self.codeA2B = get_fc_layer(self.transA2B[-1], self.dim_y[1])
            self.transA2B.append(self.codeA2B)


    def transformB2A(self, fc_config):
        self.transB2A.append(self.codeB)
        scope = 'transB2A'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            sub_scope = 'fc_%d'
            for i in range(len(fc_config) - 1):
                with tf.variable_scope(sub_scope % i):
                    fc_ = get_fc_layer(
                        self.transB2A[-1],
                        fc_config[len(fc_config) - 2 - i]['units'])
                    fc_ = get_nonlinear_layer(fc_)
                    self.transB2A.append(fc_)
            # the last classifier layer -- using fc without nonlinearization
            with tf.variable_scope(sub_scope % (len(fc_config) - 1)):
                self.codeB2A = get_fc_layer(self.transB2A[-1], self.codeA.shape[1])
            self.transB2A.append(self.codeB2A)


    def __init__(self, dim_x, dim_y,
                 conv_config, fc_config, att_config):
        # dimension of input and output should be stored
        self.dim_x = dim_x
        self.dim_y = dim_y
        # sample from space A, given input
        self.inputA = tf.placeholder(shape=dim_x, dtype=tf.float32)
        # expected latent code from space B, given feedback
        self.codeB = tf.get_variable(name='codeB', initializer=tf.zeros(dim_y, dtype=tf.float32))
        self.codeB_setter = tf.placeholder(shape=dim_y, dtype=tf.float32)
        self.codeB_assign = self.codeB.assign(self.codeB_setter)

        self.decoderA = []
        self.encoderA = []
        self.transA2B = []
        self.transB2A = []

        # the optimizer : Learning rate in 2 stages: 1E-4, 1E-5.
        self.optimzer = tf.train.AdamOptimizer(learning_rate=1E-4)

        # build decoder of A
        self.decodeA(conv_config)
        # build encoder of A
        self.encodeA(conv_config)

        # build transformer from A to B
        self.transformA2B(fc_config)
        # build transformer from B to A
        self.transformB2A(fc_config)

        # calculate losses
        self.lossA2A = get_Manhattan_distance(self.inputA2A, self.inputA)
        self.lossA2B = get_XEntropy_distance(self.codeA2B, self.codeB)
        self.lossB2A = get_Manhattan_distance(self.codeB2A, self.codeA)

        # Creating minimizers for different training purpose
        # group the variables by its namespace
        vars = tf.global_variables()
        vars_decoderA = []
        vars_encoderA = []
        vars_transA2B = []
        vars_transB2A = []

        for i in range(len(vars)):
            if vars[i].name.find('decoderA') != -1:
                vars_decoderA.append(vars[i])
            elif vars[i].name.find('encoderA') != -1:
                vars_encoderA.append(vars[i])
            elif vars[i].name.find('transA2B') != -1:
                vars_transA2B.append(vars[i])
            elif vars[i].name.find('transB2A') != -1:
                vars_transB2A.append(vars[i])
            elif vars[i].name.find('codeB') != -1:
                pass
            else:
                raise NameError('unknown variables: %s' % vars[i].name)

        self.opt_A2A = self.optimzer.minimize(
            self.lossA2A, var_list=utils.flatten([vars_decoderA, vars_encoderA]), name='opt_A2A')
        self.opt_A2B = self.optimzer.minimize(
            self.lossA2B, var_list=vars_transA2B, name='opt_A2B')
        self.opt_B2A = self.optimzer.minimize(
            self.lossB2A, var_list=vars_transB2A, name='opt_B2A')
        self.opt_codeB = self.optimizer.minimize(
            self.lossB2A, var_list=self.codeB, name='opt_codeB')

        # network self check
        print("================================ VARIABLES ===================================")
        vars = tf.global_variables()
        for i in range(len(vars)):
            print("var#%03d:%40s %16s %12s" %
                  (i, vars[i].name[:-2], vars[i].shape, str(vars[i].dtype)[9:-6]))
        print("==============================================================================")
        print("\n")
        print("================================ OPERATORS ===================================")
        ops = self.rec_layers
        for i in range(len(ops)):
            print("opr#%03d:%40s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        ops = self.att_layers
        for i in range(len(ops)):
            print("opr#%03d:%40s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        ops = self.ctl_layers
        for i in range(len(ops)):
            print("opr#%03d:%40s %16s %12s" %
                  (i, ops[i].name[:-2], ops[i].shape, str(ops[i].dtype)[9:-2]))
        print("==============================================================================")

    def getInput(self):
        return self.inputs
    def getAttentionInput(self):
        return self.att_inputs
    def getFeedback(self):
        return self.feedbacks
    def getOutput(self):
        return self.outputs
    def getControl(self):
        return self.ctl_layers
    def getLoss(self):
        return self.rec_loss
    def getOptRec(self):
        return self.minimizer_rec
    def getOptAtt(self):
        return self.minimizer_att


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
    n_conv = 8
    conv_config = [None] * n_conv
    for i in range(n_conv):
        conv_config[i] = new_conv_config(3, 3, i%2+1, i%2+1, 8<<(i//2))

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


def Train_IINN(iinn_: IINN,
               data: dict,
               model_path: str,
               train_stage: int) -> float:
    xx = data['input']
    yy = data['output']

    x_t = iinn_.getInput() # tensor of inputs
    a_t = iinn_.getAttentionInput() # tensor of inputs of attention
    y_t = iinn_.getOutput() # tensor of outputs
    c_t = iinn_.getControl() # tensor of all control signals
    f_t = iinn_.getFeedback() # tensor of feedback

    loss_t = iinn_.getLoss()
    opt_rec = iinn_.getOptRec()
    opt_att = iinn_.getOptAtt()

    # set up all the control signals to 0
    ctl_sig = []
    for i in range(len(c_t)):
        ctl_sig.append(np.array([1.0] * c_t[i].shape.as_list()[0]))

    # batch size should be always 1 because of control module limit
    BAT_NUM = 1024
    MAX_ITR = 100000 * BAT_NUM
    CVG_EPS = 1e-7
    itr = 0
    eps = 1E10
    loss = np.zeros([BAT_NUM], dtype=np.float32)

    # set up the global step counter
    global_step = tf.get_variable(name="global_step", initializer=0)
    step_next = tf.assign_add(global_step, 1, use_locking=True)

    # establish the training context
    sess = tf.Session()
    vars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=vars, max_to_keep=5)
    # load the pretrained model if exists
    if tf.train.checkpoint_exists(model_path):
        saver.restore(sess, model_path)
        utils.initialize_uninitialized(sess)
    else:
        sess.run(tf.global_variables_initializer())

    if train_stage == 1: # stage 1: train without attention
        while itr < MAX_ITR and  eps > CVG_EPS:
            idx = np.random.randint(xx.shape[0])
            feed_in = dict()
            feed_in[x_t] = xx[idx:idx+1, :, :, :]
            feed_in[f_t] = yy[idx:idx+1, :]
            for i in range(len(c_t)):
                feed_in[c_t[i]] = ctl_sig[i]
            loss[itr % BAT_NUM], _, _ = \
                sess.run([loss_t, opt_rec, step_next], feed_dict=feed_in)
            itr += 1
            if itr % BAT_NUM == 0:
                eps = np.mean(loss)
                print("batch#%05d loss=%12.8f" % (itr / BAT_NUM, eps))
            if itr % (BAT_NUM * 16) == 0:
                saver.save(sess, model_path, global_step=global_step)
    elif train_stage == 2: # training with attention, try the 3 approaches
        # approach # 3: train the entire model with attention
        while itr < MAX_ITR and eps > CVG_EPS:
            idx = np.random.randint(xx.shape[0])
            # first shot:
            # get the input of attention module, ie, the output of last shot
            feed_in = dict()
            feed_in[x_t] = xx[idx:idx + 1, :, :, :]
            for j in range(len(c_t)):
                feed_in[c_t[j]] = ctl_sig[j]
            y = sess.run(y_t, feed_dict=feed_in)
            # second shot:
            # use the outputs of last shot to control the second shot
            feed_in = dict()
            feed_in[x_t] = xx[idx:idx + 1, :, :, :]
            feed_in[a_t] = np.copy(y)
            feed_in[f_t] = yy[idx:idx + 1, :]

            loss[itr % BAT_NUM], _, _, _ = \
                sess.run([loss_t, opt_att, opt_rec, step_next],
                         feed_dict=feed_in)
            itr += 1
            if itr % BAT_NUM == 0:
                eps = np.mean(loss)
                print("batch#%05d loss=%12.8f" % (itr / BAT_NUM, eps))
            if itr % (BAT_NUM * 16) == 0:
                saver.save(sess, model_path, global_step=global_step)
    elif train_stage >= 3:
        # training in turn
        pass
    else:
        raise NameError("unrecognized stage parameter!")
    return eps


def Test_IINN(iinn_: IINN, data: dict, model_path: str, stage: int) -> float:
    xx = data['input']
    yy = data['output']

    x_t = iinn_.getInput()  # tensor of inputs
    y_t = iinn_.getOutput()  # tensor of outputs
    c_t = iinn_.getControl()  # tensor of all control signals
    a_t = iinn_.getAttentionInput()  # tensor of inputs of attention

    # set up all the control signals to 0
    ctl_sig = []
    for i in range(len(c_t)):
        ctl_sig.append(np.array([1.0] * c_t[i].shape.as_list()[0]))

    sess = tf.Session()
    vars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=vars)
    # load the pretrained model if exists
    if tf.train.checkpoint_exists(model_path):
        saver.restore(sess, model_path)
    else:
        raise NameError("failed to load checkpoint from path %s" %model_path)

    # inference
    labels_gt = np.argmax(yy, axis=-1)
    num_correct = 0

    if stage == 1: # test without attention control
        for i in range(xx.shape[0]):
            feed_in = dict()
            feed_in[x_t] = xx[i:i+1, :, :, :]
            for j in range(len(c_t)):
                feed_in[c_t[j]] = ctl_sig[j]
            y = sess.run(y_t, feed_dict=feed_in)
            if np.argmax(y[0]) == labels_gt[i]:
                num_correct += 1
    elif stage == 2: # test double-shot with attention control
        for i in range(xx.shape[0]):
            # first shot:
            # get the input of attention module, ie, the output of last shot
            feed_in = dict()
            feed_in[x_t] = xx[i:i+1, :, :, :]
            for j in range(len(c_t)):
                feed_in[c_t[j]] = ctl_sig[j]
            y = sess.run(y_t, feed_dict=feed_in)
            # second shot:
            # use the outputs of last shot to control the second shot
            feed_in = dict()
            feed_in[x_t] = xx[i:i+1, :, :, :]
            feed_in[a_t] = np.copy(y)
            y, ctl_ = sess.run([y_t, c_t], feed_dict=feed_in)
            print(ctl_[5])
            if np.argmax(y[0]) == labels_gt[i]:
                num_correct += 1
    elif stage >= 3: # test multiple shot with attention control
        for i in range(xx.shape[0]):
            # first shot:
            # get the input of attention module, ie, the output of last shot
            feed_in = dict()
            feed_in[x_t] = xx[i:i+1, :, :, :]
            for j in range(len(c_t)):
                feed_in[c_t[j]] = ctl_sig[j]
            y = sess.run(y_t, feed_dict=feed_in)
            # second or latter shot:
            # use the outputs of last shot to control the next shot
            feed_in = dict()
            feed_in[x_t] = xx[i:i + 1, :, :, :]
            for shot in range(stage - 1):
                feed_in[a_t] = np.copy(y)
                y = sess.run(y_t, feed_dict=feed_in)
            if np.argmax(y[0]) == labels_gt[i]:
                num_correct += 1
    return float(num_correct) / float(labels_gt.shape[0])


if __name__ == "__main__":
    n_class = 10
    iinn_ = Build_IINN(n_class)

    # training with CIFAR-10 dataset
    data_train, data_test = \
        dataset.cifar10.Load_CIFAR10('../Datasets/CIFAR10/')

    print('image shape: (%d, %d)' % (data_train['input'].shape[1],
                                     data_train['input'].shape[2]))

    model_path = '../Models/CIFAR10-IINN/stage2/ckpt_iinn_cifar10-16678912'
    loss = Train_IINN(iinn_, data_train, model_path, 1)
    print('Final Training Loss = %12.8f' % loss)

    #acc = Test_IINN(iinn_, data_test, model_path, 1)
    #print("Accuracy = %6.5f" % acc)

    # TODO reference here: [https://www.cnblogs.com/thisisajoke/p/12054290.html]
    #   1. define an image generator and train with YinYang model.(sparsest AE)
    #   2. Pick up the idea of creating GAME of AI