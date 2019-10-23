# Using stacked Self Organizing Maps to classifier and generate images

import numpy as np
from components import utils
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import glob
import time
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
            shape=[None, None, None, nchannels + nclasses],
            dtype=tf.float32)
        self.layers.append(layer) # the input layer
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
    assert False


# define the training targets
TARGET_VISUAL_LOSS = 0
TARGET_SEMANTIC_LOSS = 1
TARGET_OVERALL_LOSS = 2
TARGET_FOREGND_LOSS = 3


def TrainModel(model, path, samples, opt='SGD', lr=1e-4, target=TARGET_OVERALL_LOSS):
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
        feed = tf.placeholder(
            shape=out_.shape.as_list(),
            dtype=out_.dtype)
        c = 3
        if target==TARGET_OVERALL_LOSS:
            loss = tf.reduce_mean(tf.abs(out_ - feed))
        elif target==TARGET_VISUAL_LOSS:
            loss = tf.reduce_mean(tf.abs(out_[:, :, :, :c] - feed[:, :, :, :c]))
        elif target==TARGET_SEMANTIC_LOSS:
            loss = tf.reduce_mean(tf.abs(out_[:, :, :, c:] - feed[:, :, :, c:]))
        elif target==TARGET_FOREGND_LOSS:
            loss = tf.reduce_mean(tf.abs(out_[:, :, :, :c] - feed[:, :, :, :c]), axis=-1) * feed[:, :, :, c]
            loss = tf.reduce_mean(loss)
            loss = loss + tf.reduce_mean(tf.abs(out_[:, :, :, c] - feed[:, :, :, c]))
        else:
            print_error('Unrecognized target for training!')
        optimizer = None
        if opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            print_error('Unsupported Optimizer!')
        minimizer = optimizer.minimize(loss)
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
        batch_size = 32
        max_epoc = 1000
        stop_avg_loss = 1e-2
        loss_ = np.zeros([batch_size], np.float32)
        loss_acc = np.zeros([max_epoc], np.float32)
        mask, im, mask_occ, im_occ = next(samples)
        h, w, c_= im.shape[0:3]
        assert c_ == c
        nclasses = feed.shape.as_list()[-1] - c
        x = np.zeros([1, h, w, c + nclasses], dtype=np.float32)
        y = np.copy(x)

        # control the training difficulty
        difficulty = 0.0

        for epoc in range(max_epoc):
            for id_ in range(batch_size):
                # occlusion cases are 3 for inputs:
                # case 0. mask full, im occluded;
                # case 1. mask occluded, im full;
                # case 2. both full.
                # output should be always both full
                mask, im, mask_occ, im_occ = samples.send(difficulty)
                occ_case = np.random.randint(0, 2)
                if occ_case == 0:
                    x[0, :, :, :c] = im_occ[:, :, :]
                    for cid_ in range(nclasses):
                        x[0, :, :, c + cid_] = (mask == cid_)
                    y[0, :, :, :c] = im[:, :, :]
                    for cid_ in range(nclasses):
                        y[0, :, :, c + cid_] = (mask == cid_)
                elif occ_case == 1:
                    x[0, :, :, :c] = im[:, :, :]
                    for cid_ in range(nclasses):
                        x[0, :, :, c + cid_] = (mask_occ == cid_)
                    y[0, :, :, :c] = im[:, :, :]
                    for cid_ in range(nclasses):
                        y[0, :, :, c + cid_] = (mask == cid_)
                elif occ_case == 2:
                    x[0, :, :, :c] = im[:, :, :]
                    for cid_ in range(nclasses):
                        x[0, :, :, c + cid_] = (mask == cid_)
                    y[0, :, :, :c] = im[:, :, :]
                    for cid_ in range(nclasses):
                        y[0, :, :, c + cid_] = (mask == cid_)
                else:
                    print_error('occlusion case id invalid!')
                loss_[id_], _ = sess.run(
                    [loss, minimizer],
                    feed_dict={in_: x, feed: y})
                print('Epoc:%5d\tBatch:%5d\tLoss:%8.5f' % (epoc, id_, loss_[id_]))
            # visualize the training process
            plt.clf()
            plt.plot(loss_acc[:epoc], 'r-')
            plt.xticks(np.arange(0, max_epoc + max_epoc / 10, max_epoc / 10))
            #plt.yticks(np.arange(0, 1.0, 1.0 / 10))
            #plt.axis(xmin=0, xmax=max_epoc, ymin=0, ymax=1.0)
            plt.axis(xmin=0, xmax=max_epoc)
            plt.legend(['train'])
            plt.pause(0.01)

            loss_acc[epoc] = np.mean(loss_)
            print('Average Loss for epoc#%d: %.5f' % (epoc, loss_acc[epoc]))

            if loss_acc[epoc] < stop_avg_loss:
                saver.save(sess, path)
                print('tmp model saved.')
                difficulty += 0.1
                print('training goes harder...')

        # save the model into files
        saver.save(sess, path)
        print('model saved.')


def TestModel(model, path, samples, test_num):
    with tf.Session() as sess:
        in_ = model.getInputPlaceHolder()
        out_ = model.getOutputOp()
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
        mask, im, mask_occ, im_occ = next(samples)
        h, w, c = im.shape[0:3]
        nclasses = out_.shape.as_list()[-1] - 3
        x = np.zeros([1, h, w, c + nclasses], dtype=np.float32)
        y = np.copy(x)
        test_num = 10

        for i in range(test_num):
            mask, im, mask_occ, im_occ = samples.send(0.5)
            #occ_case = np.random.randint(2)
            occ_case = 0
            if occ_case == 0:
                x[0, :, :, :c] = im_occ[:, :, :]
                for cid_ in range(nclasses):
                    x[0, :, :, c + cid_] = (mask == cid_)
                y[0, :, :, :c] = im[:, :, :]
                for cid_ in range(nclasses):
                    y[0, :, :, c + cid_] = (mask == cid_)
            elif occ_case == 1:
                x[0, :, :, :c] = im[:, :, :]
                for cid_ in range(nclasses):
                    x[0, :, :, c + cid_] = (mask_occ == cid_)
                y[0, :, :, :c] = im[:, :, :]
                for cid_ in range(nclasses):
                    y[0, :, :, c + cid_] = (mask == cid_)
            else:
                print_error('occlusion case id invalid!')
            y_out = sess.run(out_, feed_dict={in_: x})
            print(np.max(np.max(y_out[0], axis=0), axis=0))
            y_out = np.maximum(np.minimum(y_out, 1.0), 0)
            plt.clf()
            plt.title(str(occ_case))
            if occ_case == 0:
                plt.figure(0)
                plt.imshow(x[0, :, :, :c])
                plt.figure(1)
                plt.imshow(y_out[0, :, :, :c])
                plt.figure(2)
                plt.imshow(y[0, :, :, :c])
            else:
                plt.figure(0)
                plt.imshow(x[0, :, :, c])
                plt.figure(1)
                plt.imshow(y_out[0, :, :, c])
                plt.figure(2)
                plt.imshow(y[0, :, :, c])
            plt.pause(10)





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
# bbox : [begin_x, begin_y, width, height]
def occlude_mask(mask, bbox):
    mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0
    return mask


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
    im_rgb = np.array(im_rgb, np.float32)
    im_grey = Image.fromarray(mask)
    im_grey = im_grey.resize((w_, h_), Image.NEAREST)
    im_grey = np.array(im_grey, np.uint8)
    im_[:, :, :] = 0
    im_[y:y+h_, x:x+w_, :] = im_rgb[:, :, :] / 255.0
    mask[:, :] = 0
    mask[y:y+h_, x:x+w_] = im_grey[:, :]
    return im_, mask


# move by offset: (dx, dy) -> dx is horizontal, dy is vertical.
def move_foreground(im_, mask, offset):
    h, w = im_.shape[0], im_.shape[1]
    x_min = max(0, offset[0])
    x_max = min(w, w + offset[0])
    y_min = max(0, offset[1])
    y_max = min(h, h + offset[1])
    pad_mask = np.zeros(mask.shape, mask.dtype)
    pad_rgb = np.zeros(im_.shape, im_.dtype)
    if offset[0] >= 0:
        if offset[1] >= 0:
            pad_mask[y_min:y_max, x_min:x_max] = mask[0:y_max - y_min, 0:x_max - x_min]
            pad_rgb[y_min:y_max, x_min:x_max, :] = im_[0:y_max - y_min, 0:x_max - x_min, :]
        else:
            pad_mask[y_min:y_max, x_min:x_max] = mask[-offset[1]:h, 0:x_max-x_min]
            pad_rgb[y_min:y_max, x_min:x_max, :] = im_[-offset[1]:h, 0:x_max - x_min, :]
    else:
        if offset[1] >= 0:
            pad_mask[y_min:y_max, x_min:x_max] = mask[0:y_max - y_min, -offset[0]:w]
            pad_rgb[y_min:y_max, x_min:x_max, :] = im_[0:y_max - y_min, -offset[0]:w, :]
        else:
            pad_mask[y_min:y_max, x_min:x_max] = mask[-offset[1]:h, -offset[0]:w]
            pad_rgb[y_min:y_max, x_min:x_max, :] = im_[-offset[1]:h, -offset[0]:w, :]
    im_[:, :, :] = pad_rgb[:, :, :]
    mask[:, :] = pad_mask[:, :]
    return im_, mask


# rotate around a point in 2D plane with equation:
# x <- cos(theta) * x - sin(theta) * y
# y <- sin(theta) * x + cos(theta) * y
# Using remap theory: mapping form a to b,
# to update each pixel in new coordinates,
# one has to inverse this process,
# I.E. to calcalate the source of which has rotated.
def rotate_foreground(im_, mask, theta):
    im_rgb = Image.fromarray(np.uint8(im_ * 255))
    im_gray = Image.fromarray(mask)
    im_rgb = im_rgb.rotate(theta)
    im_gray = im_gray.rotate(theta)
    im_ = np.array(im_rgb, np.float32) / 255.0
    mask = np.array(im_gray, np.uint8)
    return im_, mask


def preprocess_dataset(input_prefix, output_prefix, output_size):
    files = glob.glob(input_prefix + '/*.jpg')
    for i in range(len(files)):
        im_ = Image.open(files[i])
        im_ = im_.resize(output_size)
        files[i] = files[i].replace(input_prefix, output_prefix)
        im_.save(files[i])


def load_dataset(path, target_size=(512, 384)):
    files_fg = glob.glob(path + '/fg/*.jpg')
    files_bg = glob.glob(path + '/bg/*.jpg')
    images_bg = [None] * len(files_bg)
    images_fg = [None] * len(files_fg)
    masks_fg = [None] * len(files_fg)
    patches = None
    mask_patches = None

    for i in range(len(files_bg)):
        im_ = Image.open(files_bg[i])
        im_ = im_.resize(target_size)
        images_bg[i] = np.array(im_, np.float32) / 255.0

    for i in range(len(images_fg)):
        im_ = Image.open(files_fg[i])
        im_ = im_.resize(target_size)
        images_fg[i] = np.array(im_, np.float32) / 255.0
        # utils.show_rgb(images_fg[i])

        # using robust technique to separate object
        patches = extract_patches(output_=patches, input_=images_fg[i], ksize=3)
        miu = np.mean(patches, axis=(2, 3), keepdims=True)
        sigma = np.mean(np.abs(patches - miu), axis=(2, 3)) / miu[:, :, 0, 0]

        h, w = sigma.shape[0], sigma.shape[1]
        masks_fg[i] = np.zeros([images_fg[i].shape[0], images_fg[i].shape[1]], np.uint8)
        masks_fg[i][1:h + 1, 1:w + 1] = sigma > 0.05

        # erode the mask a little bit to fit the edge of object
        mask = np.expand_dims(masks_fg[i], axis=-1)
        mask_patches = extract_patches(output_=mask_patches, input_=mask, ksize=3)
        mask = np.min(mask_patches, axis=2)
        masks_fg[i][1:h + 1, 1:w + 1] = mask[:, :, 0]

    return images_fg, masks_fg, images_bg


def generate_random_sample(images_fg, masks_fg, images_bg):
    assert len(images_fg) > 0
    h, w = masks_fg[0].shape[0:2]
    fg = np.copy(images_fg[0])
    mask = np.zeros([h, w], dtype=np.uint8)
    mask_occ = np.copy(mask)
    bg = np.copy(images_bg[0])
    merg = np.copy(images_bg[0])
    merg_occ = np.copy(merg)

    num_fg = len(images_fg)
    num_bg = len(images_bg)
    h, w = mask.shape[0], mask.shape[1]

    print('foreground units: %d.' % num_fg)
    print('background units: %d.' % num_bg)

    # the global setting controled outside this coroutine
    occ_ratio = 0.0

    while True:
        id_fg = np.random.randint(num_fg)
        id_bg = np.random.randint(num_bg)
        fg[:, :, :] = images_fg[id_fg][:, :, :]
        mask[:, :] = masks_fg[id_fg][:, :]
        bg[:, :, :] = images_bg[id_bg][:, :, :]

        # data augumentation method 1: resize
        min_ratio = 0.3
        resize_ratio = min_ratio + np.random.rand() * (1 - min_ratio)
        fg, mask = resize_foreground(fg, mask, resize_ratio)

        # data augumentation method 2: rotate
        theta = 360 * np.random.rand()
        fg, mask = rotate_foreground(fg, mask, theta)

        # data augumentaiton method 3: move
        max_move = 0.5
        move_x = int(((2 * np.random.rand() - 1.0) * max_move) * w)
        move_y = int(((2 * np.random.rand() - 1.0) * max_move) * h)
        fg, mask = move_foreground(fg, mask, [move_x, move_y])

        merg = np.expand_dims(mask==1, axis=-1) * fg + np.expand_dims(mask==0, axis=-1) * bg

        # data augumentation method 4: crop
        crop_box = [None] * 4
        crop_box[2] = int(w*occ_ratio)
        crop_box[3] = int(h*occ_ratio)
        crop_box[0] = np.random.randint(0, w - crop_box[2])
        crop_box[1] = np.random.randint(0, h - crop_box[3])

        mask_occ[:, :] = mask[:, :]
        mask_occ = occlude_mask(mask_occ, crop_box)
        merg_occ = np.expand_dims(mask_occ==1, axis=-1) * fg + np.expand_dims(mask_occ==0, axis=-1) * bg

        occ_ratio = yield (mask, merg, mask_occ, merg_occ)


def InspectDataset(TRAIN_VOLUME):
    fg, mask, bg = load_dataset('../../Datasets/Umbrella/')
    print('Dataset loaded!')
    sample_generator = generate_random_sample(fg, mask, bg)
    next(sample_generator)
    for i in range(TRAIN_VOLUME):
        mask, im, mask_occ, im_occ = sample_generator.send(np.random.rand())
        plt.clf()
        plt.figure(0)
        plt.imshow(im_occ)
        plt.figure(1)
        plt.imshow(im)
        plt.pause(1)


if __name__ == '__main__':
    #InspectDataset(100)
    #exit(0)

    fg, mask, bg = load_dataset('../../Datasets/Umbrella/')
    print('Dataset loaded!')
    sample_generator = generate_random_sample(fg, mask, bg)
    # train the network with unlabeled examples, actually, the label is also a kind of input
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
        samples=sample_generator,
        opt='Adam',
        lr=1e-4,
        target=TARGET_FOREGND_LOSS)
    exit(0)
    TestModel(
        model=auto_encoder,
        path='../../Models/SemanticSegmentation/umbrella.ckpt',
        samples=sample_generator,
        test_num=10)

# to do list:
# 1. train with semantic loss
# 2. visual loss changed to the difference of foreground