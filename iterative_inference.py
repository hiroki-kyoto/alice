# iterative_inference.py
# NN inference in an iterative manner, instead of a forward single shot.

import numpy as np
import os
import platform
import matplotlib.pyplot as plt

import dataset
import components.utils as utils

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


def get_controlled_layer(inputs, control): # define your own control strategy
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
        self.att_inputs = tf.placeholder(shape=dim_y, dtype=tf.float32)

        self.rec_layers = []
        self.rec_layers.append(self.inputs)

        self.att_layers = []
        self.att_layers.append(self.att_inputs)

        self.ctl_layers = []

        # the optimizer
        # Learning rate stages: 1E-4, 1E-5.
        self.optimzer = tf.train.AdamOptimizer(learning_rate=1E-4)

        scope = 'attention'
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # normalize input (VERY IMPORTANT)
            in_norm = tf.nn.softmax(self.att_layers[-1])
            self.att_layers.append(in_norm)
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
                ctl_grad_free = \
                    self.att_layers[-1][offset:offset + conv_config[i]['filters']]
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

        # Creating minimizers for different training purpose
        # group the variables by its namespace
        vars = tf.global_variables()
        rec_vars = []
        att_vars = []
        for i in range(len(vars)):
            if vars[i].name.find('recognition') != -1:
                rec_vars.append(vars[i])
            elif vars[i].name.find('attention') != -1:
                att_vars.append(vars[i])
            else:
                raise NameError('unknown variables: %s' % vars[i].name)

        self.minimizer_rec = self.optimzer.minimize(
            self.rec_loss, var_list=rec_vars, name='opt_rec')
        self.minimizer_att = self.optimzer.minimize(
            self.rec_loss, var_list=att_vars, name='opt_att')

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

    def attention(self, x, y):
        pass
    def inference(self, x, a):
        pass
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
        ctl_sig.append(np.array([0] * c_t[i].shape.as_list()[0]))

    # batch size should be always 1 because of control module limit
    BAT_NUM = 1024
    MAX_ITR = 100000 * BAT_NUM
    CVG_EPS = 1e-2
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
                print("batch#%05d loss=%3.5f" % (itr / BAT_NUM, eps))
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
                print("batch#%05d loss=%3.5f" % (itr / BAT_NUM, eps))
            if itr % (BAT_NUM * 16) == 0:
                saver.save(sess, model_path, global_step=global_step)
    elif train_stage == 3:
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
        ctl_sig.append(np.array([0] * c_t[i].shape.as_list()[0]))

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
            y = sess.run(y_t, feed_dict=feed_in)
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

    ''' 
    # iterative inference demo
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
    '''


if __name__ == "__main__":
    n_class = 10
    iinn_ = Build_IINN(n_class)

    # training with CIFAR-10 dataset
    data_train, data_test = \
        dataset.cifar10.Load_CIFAR10('../Datasets/CIFAR10/')

    # inspect the dataset
    print('image shape: (%d, %d)' % (data_train['input'].shape[1],
                                     data_train['input'].shape[2]))

    model_path = '../Models/CIFAR10-IINN/ckpt_iinn_cifar10-3424256-6356992-9011200-21626880'
    #loss = Train_IINN(iinn_, data_train, model_path, 2)
    #print('Final Training Loss = %6.5f' % loss)
    # test the trained model with test split of the same dataset
    acc = Test_IINN(iinn_, data_train, model_path, 4)
    print("Accuracy = %6.5f" % acc)

    # TODO
    #   method 1: use label y to control bias for each channel in each layer(leaky relu)
    #   method 2: use label y to control the mask for each channel in each layer(Non-zero)
    #   method 3: use both x and y to control the attention mask for only input

    # TODO Two approachs:
    #   1st- inspired by the concept of co-activated neural group, attention is a phase locked loop.
    #   2nd- inspired by the system of yinyang-GAN, the HU system, decoder pass grads to encoder.

    # for the 1st approach:
    # 1> when argmax(y) == argmax(y_{gt}), attention module is trained to converge at y = y_{gt}
    # 2> otherwise, attention module is trained to output y_{n+1}=y-y_n till y = \vec{0}.

    # Next, add optimizer and begin to train this network!
    # Test the model trained without attention control.
    # Train with attention control in 4 ways:
    #   1. train the model with attention alone using groundtruth labels;
    #   2. train with attention alone using output labels;
    #   3. train together of rec and att with groundtruth labels;
    #   4. train attention(using output/gt) and recognition modules in turns;