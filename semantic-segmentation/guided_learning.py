# Using stacked Self Organizing Maps to classifier and generate images

import numpy as np
from components import utils
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import glob
from abc import abstractmethod, ABCMeta


class Model(metaclass=ABCMeta):
    @abstractmethod
    def getInputPlaceHolder(self):pass

    @abstractmethod
    def getOutputOp(self):pass

class AutoEncoder(Model):
    def __init__(self, nchannels, nclasses, kernels, sizes, strides):
        # building blocks
        self.layers = []
        layer = tf.placeholder(
            shape=[None, None, None, nchannels],
            dtype=tf.float32)
        self.layers.append(layer)
        # encoder
        for i in range(len(kernels)):
            layer = tf.layers.conv2d(
                inputs=self.layers[-1],
                filters=kernels[i],
                kernel_size=sizes[i],
                strides=strides[i],
                padding='same',
                use_bias=True,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04),
                bias_initializer=tf.zeros_initializer())
            self.layers.append(layer)
        # decoder
        for i in range(len(kernels) - 1):
            layer = tf.layers.conv2d_transpose(
                inputs=self.layers[-1],
                filters=kernels[len(kernels)- 2 - i],
                kernel_size=sizes[len(kernels)- 1 - i],
                strides=strides[len(kernels)- 1 - i],
                padding='same',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04),
                bias_initializer=tf.zeros_initializer())
            self.layers.append(layer)
        layer = tf.layers.conv2d_transpose(
            inputs=self.layers[-1],
            filters=nchannels + nclasses,
            kernel_size=sizes[0],
            strides=strides[0],
            padding='same',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.initializers.truncated_normal(0.0, 0.04),
            bias_initializer=tf.zeros_initializer())
        self.layers.append(layer)

    def getInputPlaceHolder(self):
        return self.layers[0]

    def getOutputOp(self):
        return self.layers[-1]


def TrainModel(model, path, images, labels, opt='SGD', lr=1e-4):
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
        feed = tf.placeholder(
            shape=out_.shape.as_list(),
            dtype=out_.dtype)
        h, w, c = images[0].shape[0:3]
        unsupervised_loss = tf.reduce_mean(tf.square(
            tf.reduce_mean(tf.abs(out_[:, :, :, :c] - in_), axis=-1)))
        supervised_loss = tf.reduce_mean(tf.square(
            tf.reduce_mean(tf.abs(out_ - feed), axis=-1)))
        optimizer = None
        if opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        supervised_minimizer = optimizer.minimize(supervised_loss)
        unsupervised_minimizer = optimizer.minimize(unsupervised_loss)
        # establish the training context
        vars = tf.trainable_variables()
        saver = tf.train.Saver(var_list=vars)
        # load the pretrained model if exists
        if tf.train.checkpoint_exists(path):
            saver.restore(sess, path)
            utils.initialize_uninitialized(sess)
        else:
            sess.run(tf.global_variables_initializer())
        # start training thread
        max_epoc = 1000
        stop_avg_loss = 1e-3
        loss_ = np.zeros([len(images)], np.float32) + stop_avg_loss * 1000
        loss_acc = np.zeros([max_epoc], np.float32)
        for epoc in range(max_epoc):
            if np.mean(loss_) < stop_avg_loss:
                print('training success.')
                break
            else:
                loss_acc[epoc] = np.mean(loss_)
                print('Average Loss for epoc#%d: %.5f' % (epoc, loss_acc[epoc]))
            # forward the tensor stream
            ids = np.random.permutation(len(images))
            for id_ in ids:
                x = np.reshape(images[id_], [1, h, w, c])
                y = labels[id_]
                if y is None: # unsupervised learning
                    loss_[id_], _ = sess.run(
                        [unsupervised_loss, unsupervised_minimizer],
                        feed_dict={in_: x})
                else:
                    feed_ = np.zeros([1, h, w, feed.shape.as_list()[-1]], np.float32)
                    feed_[0, :, :, :c] = x
                    if y > 0: # class 0 stands for no object, object ID starts from #1.
                        feed_[0, :, :, c + (y - 1)] = 1.0
                    loss_[id_], _ = sess.run(
                        [supervised_loss, supervised_minimizer],
                        feed_dict={in_: x, feed: feed_})
            # visualize the training process
            plt.clf()
            plt.plot(loss_acc[:epoc], 'r-')
            plt.xticks(np.arange(0, max_epoc, max_epoc / 10))
            plt.yticks(np.arange(0, 1.0, 1.0 / 10))
            plt.axis([0, max_epoc, 0, 1.0])
            plt.legend(['train'])
            plt.pause(0.01)
        # save the model into files
        saver.save(sess, path)
        print('model saved.')


def TestModel(model, path, images, labels):
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
        h, w, c = images[0].shape[0:3]
        # establish the training context
        vars = tf.trainable_variables()
        saver = tf.train.Saver(var_list=vars)
        # load the pretrained model if exists
        if tf.train.checkpoint_exists(path):
            saver.restore(sess, path)
            utils.initialize_uninitialized(sess)
        else:
            print('Model not found at : %s' % path)
            assert False
        # start testing
        # forward the tensor stream
        ids = np.arange(len(images))
        for id_ in ids:
            x = np.reshape(images[id_], [1, h, w, c])
            y = sess.run(out_, feed_dict={in_: x})
            y = np.maximum(np.minimum(y, 1.0), 0)
            if labels[id_] is None:
                # check if the construction is okay
                utils.show_rgb(x[0, :, :, :])
                utils.show_rgb(y[0, :, :, :c])
            else:
                # check if the semantic segmentation is okay
                utils.show_rgb(x[0, :, :, :])
                utils.show_rgb(y[0, :, :, :c])
                for cid in range(y.shape[-1] - c):
                    #utils.show_rgb(y[0, :, :, :c] * y[0, :, :, c + cid:c+cid+1])
                    utils.show_gray(y[0, :, :, c + cid], min=0, max=1.0)





if __name__ == '__main__':
    # train the network with unlabeled examples, actually, the label is also a kind of input
    files = glob.glob('E:/Gits/Datasets/Umbrella/seq-in/*.jpg')[0:300]
    files += glob.glob('E:/Gits/Datasets/Umbrella/seq-out/*.jpg')[0:300]
    images = [None] * len(files)
    labels = [None] * len(files)
    assert len(images) % 2 == 0
    for i in range(len(files)):
        images[i] = np.array(Image.open(files[i]), np.float32) / 255.0
        # if i < len(images) / 2:
        #     labels[i] = 1
        # else:
        #     labels[i] = 0
    print('Dataset Loaded!')

    # create a AutoEncoder
    auto_encoder = AutoEncoder(
        nchannels=3,
        nclasses=1,
        kernels=[8, 16, 32, 64],
        sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 2])

    # train the AE with unlabeled samples
    # TrainModel(
    #     model=auto_encoder,
    #     path='../../Models/SemanticSegmentation/umbrella.ckpt',
    #     images=images,
    #     labels=labels,
    #     opt='Adam',
    #     lr=1e-4)

    TestModel(
        model=auto_encoder,
        path='../../Models/SemanticSegmentation/umbrella.ckpt',
        images=images,
        labels=labels)
