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


def print_error(str_):
    print('\033[1;31m' + str_ + '\033[0m')


def TrainModel(model, path, images, labels, opt='SGD', lr=1e-4):
    LAMBDA = 0.1
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
        feed = tf.placeholder(
            shape=out_.shape.as_list(),
            dtype=out_.dtype)
        h, w, c = images[0].shape[0:3]
        unsupervised_loss = tf.reduce_mean(
            tf.square(
                tf.reduce_mean(
                    tf.abs(out_[:, :, :, :c] - in_),
                    axis=-1
                )
            )
        )
        supervised_loss = LAMBDA * tf.reduce_mean(
            tf.square(
                tf.reduce_mean(
                    tf.abs(out_[:, :, :, c:] - feed[:, :, :, c:]),
                    axis=-1
                )
            )
        ) + (1 - LAMBDA) * unsupervised_loss
        optimizer = None
        if opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            print_error('Unsupported Optimizer!')
            assert False
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
        loss_ = np.zeros([len(images)], np.float32)
        loss_acc = np.zeros([max_epoc], np.float32)
        for epoc in range(max_epoc):
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
            plt.xticks(np.arange(0, max_epoc + max_epoc / 10, max_epoc / 10))
            #plt.yticks(np.arange(0, 1.0, 1.0 / 10))
            #plt.axis(xmin=0, xmax=max_epoc, ymin=0, ymax=1.0)
            plt.axis(xmin=0, xmax=max_epoc)
            plt.legend(['train'])
            plt.pause(0.01)

            if np.mean(loss_) < stop_avg_loss:
                print('training success.')
                break
            else:
                loss_acc[epoc] = np.mean(loss_)
                print('Average Loss for epoc#%d: %.5f' % (epoc, loss_acc[epoc]))
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



# Valid convolution
def extract_patches(output_ = None, input_=None, ksize=3):
    h, w, c = input_.shape[0], input_.shape[1], input_.shape[2]
    assert h > ksize and w > ksize
    if output_ is None:
        output_ = np.zeros(shape=[h - (ksize - 1), w - (ksize - 1), ksize*ksize, c])
    for i in range(ksize):
        for j in range(ksize):
            output_[:, :, i*ksize + j, :] = input_[i:h - (ksize - 1 - i), j:w - (ksize - 1 - j), :]
    return output_


# occlusion simulation
def occlude_mask(im_, bbox):
    pass

# Full size is not changed but the size and position on the image are changed
# new size has to be smaller than the original one
def resize_foreground(im_, mask, down_scale):
    assert down_scale < 1.0
    assert im_.shape[0] == mask.shape[0] and im_.shape[1] == mask.shape[1]
    h, w = im_.shape[0], im_.shape[1]
    h_ = int(h * down_scale)
    w_ = int(w * down_scale)
    y = h // 2 - h_ // 2
    x = w // 2 - w_ // 2
    im_rgb = Image.fromarray(np.uint8(im_*255))
    im_rgb = im_rgb.resize((w_, h_), Image.LINEAR)
    im_rgb = np.array(im_rgb)
    im_grey = Image.fromarray(np.uint8(mask*255))
    im_grey = im_grey.resize((w_, h_), Image.NEAREST)
    im_grey = np.array(im_grey)
    im_[y:y+h_, x:x+w_, :] = im_rgb[:, :, :] / 255.0
    mask[:, :] = 0
    mask[y:y+h_, x:x+w_] = im_grey[:, :] / 255.0
    return im_, mask


def move_foreground(im_, mask, offset):
    assert offset[0] < w // 3
    assert offset[1] < h // 3
    x_min = max(0, offset[0])
    x_max = min(h, h + offset[0])
    y_min = max(0, offset[1])
    y_max = min(w, w + offset[1])
    #x_ind = np.arange(x_min, x_max, 1)
    #y_ind = np.arange(y_min, y_max, 1)
    pad_mask = np.zeros([x_max - x_min, y_max - y_min], np.float32)
    pad_rgb = np.zeros([x_max - x_min, y_max - y_min, 3], np.float32)
    pad_mask[:, :] = mask[0:x_max-x_min, 0:y_max-y_min]
    pad_rgb[:, :, :] = im_[0:x_max-x_min, 0:y_max-y_min, :]
    im_[:, :, :] = 0
    mask[:, :] = 0
    im_[x_min:x_max, y_min:y_max, :] = pad_rgb
    mask[x_min:x_max, y_min:y_max] = pad_mask
    return im_, mask


# rotate around a point in 2D plane with equation:
# r <- sqrt(x^2+y^2)
# x <- r * cos(theta)
# y <- r * sin(theta)
# Using remap theory: mapping form a to b,
# to update each pixel in new coordinates,
# one has to inverse this process,
# I.E. to calcalate the source of which has rotated.
def rotate_foreground(im_, mask, theta):
    theta = -theta
    h, w = im_.shape[0], im_.shape[1]
    x = np.arange(0, w, 1) - w / 2
    y = h / 2 - np.arange(0, h, 1)
    xx, yy = np.meshgrid(x, y)
    xx_new = xx * np.cos(theta) - yy * np.sin(theta)
    yy_new = yy * np.cos(theta) + xx * np.sin(theta)
    xx_new = np.int32(np.minimum(np.maximum(xx_new + w / 2, 0), w - 1))
    yy_new = np.int32(np.minimum(np.maximum(h / 2 - yy_new, 0), h - 1))
    tmp_rgb = np.zeros(im_.shape, im_.dtype)
    tmp_mask = np.zeros(mask.shape, mask.dtype)
    tmp_rgb[:, :, :] = im_[:, :, :]
    tmp_mask[:, :] = mask[:, :]
    im_[:, :, :] = tmp_rgb[yy_new, xx_new, :]
    mask[:, :] = tmp_mask[yy_new, xx_new]
    return im_, mask



if __name__ == '__main__':
    # abstract object from white wall
    files_fg = glob.glob('E:/Gits/Datasets/Umbrella/WhiteWall/fg/*.jpg')
    files_bg = glob.glob('E:/Gits/Datasets/Umbrella/WhiteWall/bg/*.jpg')
    new_size = (400, 300)
    images_bg = [None] * len(files_bg)
    #for i in range(len(files_bg)):
    for i in range(1):
        im_ = Image.open(files_bg[i])
        im_ = im_.resize(new_size)
        images_bg[i] = np.array(im_, np.float32) / 255.0

    images_fg = [None] * len(files_fg)
    masks_fg = [None] * len(files_fg)
    patches = None
    mask_patches = None

    for i in range(len(images_fg)):
        im_ = Image.open(files_fg[i])
        im_ = im_.resize(new_size)
        images_fg[i] = np.array(im_, np.float32) / 255.0
        #utils.show_rgb(images_fg[i])

        # using robust technique to separate object
        patches = extract_patches(output_=patches, input_=images_fg[i], ksize=3)
        miu = np.mean(patches, axis=(2, 3), keepdims=True)
        sigma = np.mean(np.abs(patches - miu), axis=(2, 3)) / miu[:,:,0,0]
        utils.show_gray(sigma, min=0, max=1)
        h, w = sigma.shape[0], sigma.shape[1]
        masks_fg[i] = np.zeros([images_fg[i].shape[0], images_fg[i].shape[1]], np.float32)
        masks_fg[i][1:h+1, 1:w+1] = sigma > 0.08

        # erode the mask a little bit to fit the edge of object
        mask = np.expand_dims(masks_fg[i], axis=-1)
        mask_patches = extract_patches(output_=mask_patches, input_=mask, ksize=3)
        mask = np.min(mask_patches, axis=2)
        masks_fg[i][1:h + 1, 1:w + 1] = mask[:, :, 0]

        # apply a few data augumentation ops to the mask and the foreground
        images_fg[i], masks_fg[i] = resize_foreground(images_fg[i], masks_fg[i], 0.5)
        images_fg[i], masks_fg[i] = move_foreground(images_fg[i], masks_fg[i], [50, 20])
        images_fg[i], masks_fg[i] = rotate_foreground(images_fg[i], masks_fg[i], np.pi / 6)

        mask = np.expand_dims(masks_fg[i], axis=-1)
        merg = mask * images_fg[i] + (1 - mask) * images_bg[0]
        utils.show_rgb(merg)
        input()
    exit(0)

    # train the network with unlabeled examples, actually, the label is also a kind of input
    # files = glob.glob('E:/Gits/Datasets/Umbrella/seq-in/*.jpg')[0:300:10]
    # files += glob.glob('E:/Gits/Datasets/Umbrella/seq-out/*.jpg')[0:300:10]
    files = glob.glob('E:/Gits/Datasets/Umbrella/seq-in/I_800.jpg')
    files += glob.glob('E:/Gits/Datasets/Umbrella/seq-out/O_800.jpg')
    print(files)
    images = [None] * len(files)
    labels = [None] * len(files)
    assert len(images) % 2 == 0
    for i in range(len(files)):
        #images[i] = np.array(Image.open(files[i]), np.float32) / 255.0
        im_ = Image.open(files[i])
        im_ = im_.resize((144, 144))
        images[i] = np.array(im_, np.float32) / 255.0
        #utils.show_rgb(images[i])
        if i < len(images) / 2:
            labels[i] = 1
        else:
            labels[i] = 0
    print('Dataset Loaded!')
    # create a AutoEncoder
    auto_encoder = AutoEncoder(
        nchannels=3,
        nclasses=1,
        kernels=[8, 16, 32, 64],
        sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 2])

    # train the AE with unlabeled samples
    TrainModel(
        model=auto_encoder,
        path='../../Models/SemanticSegmentation/umbrella.ckpt',
        images=images,
        labels=labels,
        opt='Adam',
        lr=1e-4)

    TestModel(
        model=auto_encoder,
        path='../../Models/SemanticSegmentation/umbrella.ckpt',
        images=images,
        labels=labels)
