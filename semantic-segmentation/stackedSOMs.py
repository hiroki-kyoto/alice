# Using stacked Self Organizing Maps to classifier and generate images

import numpy as np
from components import utils
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import glob

MIN_SUPPORT = 100
PATTERN_DENSITY = 0.1 # control the net volume of long term memory
MAX_IMMATURITY = 0.7 # control the minimum activation requirement of absorbing patterns into long term memory
ITER_TIMES = 300 # control the absorbing cycles per instance

def allocate_winners(responses_shape):
    return np.zeros([responses_shape[0], responses_shape[1]], dtype=np.int32)


def allocate_responses(patches_shape, patterns_shape):
    assert len(patches_shape) == 5
    assert len(patterns_shape) == 4
    assert patches_shape[2] == patterns_shape[1]
    assert patches_shape[3] == patterns_shape[2]
    assert patches_shape[4] == patterns_shape[3]
    return np.zeros([patches_shape[0], patches_shape[1], patterns_shape[0]], dtype=np.float32)


def allocate_patches(input_shape, ksize, stride):
    assert len(input_shape) == 3
    assert input_shape[0] % stride == 0
    assert input_shape[1] % stride == 0
    h_ = input_shape[0] // stride
    w_ = input_shape[1] // stride
    c_ = input_shape[2]
    return np.zeros([h_, w_, ksize, ksize, c_], dtype=np.float32)


def allocate_immaturity(num_patterns):
    return np.zeros([num_patterns], dtype=np.float32) + 1.0


def allocate_acceptance(num_patterns):
    return np.zeros([num_patterns], dtype=np.float32)


def extract_patches(output_, input_, ksize, stride):
    assert len(input_.shape) == 3
    assert input_.shape[0] % stride == 0
    assert input_.shape[1] % stride == 0

    h_ = input_.shape[0] // stride
    w_ = input_.shape[1] // stride
    c_ = input_.shape[2]

    assert len(output_.shape) == 5
    assert h_ == output_.shape[0]
    assert w_ == output_.shape[1]
    assert ksize == output_.shape[2]
    assert ksize == output_.shape[3]
    assert c_ == output_.shape[4]

    h_i = input_.shape[0]
    w_i = input_.shape[1]
    r_ = ksize // 2

    for i in range(ksize):
        y_coords = np.maximum(i - r_ + np.arange(0, h_i, stride), 0)
        y_coords = np.minimum(y_coords, h_i - 1)
        for j in range(ksize):
            x_coords = np.maximum(j - r_ + np.arange(0, w_i, stride), 0)
            x_coords = np.minimum(x_coords, w_i - 1)
            x_ids, y_ids = np.meshgrid(x_coords, y_coords)
            output_[:, :, i, j] = input_[y_ids, x_ids]

    return output_


def merge_patches(imag_, patches):
    assert len(imag_.shape) == 3
    assert len(patches.shape) == 5

    h_ = patches.shape[0]
    w_ = patches.shape[1]
    c_ = patches.shape[4]
    ksize = patches.shape[2]
    stride = imag_.shape[0] // h_

    h_i = imag_.shape[0]
    w_i = imag_.shape[1]
    r_ = ksize // 2

    mask = np.zeros(imag_.shape, dtype=np.float32)
    for i in range(ksize):
        y_coords = np.maximum(i - r_ + np.arange(0, h_i, stride), 0)
        y_coords = np.minimum(y_coords, h_i - 1)
        for j in range(ksize):
            x_coords = np.maximum(j - r_ + np.arange(0, w_i, stride), 0)
            x_coords = np.minimum(x_coords, w_i - 1)
            x_ids, y_ids = np.meshgrid(x_coords, y_coords)
            imag_[y_ids, x_ids] += patches[:, :, i, j]
            mask[y_ids, x_ids] += 1
    mask = np.maximum(mask, 1)
    imag_[:, :, :] = imag_[:, :, :] / mask[:, :, :]

    return imag_


def pattern_response(responses, winners, patches, patterns, immaturity, acceptance, discount, max_iter, logs):
    assert len(responses.shape) == 3
    assert len(patches.shape) == 5
    assert len(patterns.shape) == 4
    assert len(winners.shape) == 2
    if logs is not None:
        assert len(logs.shape) == 2

    assert patches.shape[2] == patterns.shape[1]
    assert patches.shape[3] == patterns.shape[2]
    assert patches.shape[4] == patterns.shape[3]

    assert responses.shape[0] == patches.shape[0]
    assert responses.shape[1] == patches.shape[1]
    assert responses.shape[2] == patterns.shape[0]

    assert winners.shape[0] == responses.shape[0]
    assert winners.shape[1] == responses.shape[1]

    if logs is not None:
        assert logs.shape[0] == max_iter
        assert logs.shape[1] == patterns.shape[0]

    instant_response = None

    # reshape tensors into vectors
    x_ = np.reshape(patches,
                    newshape=[patches.shape[0] * patches.shape[1],
                              patches.shape[2] * patches.shape[3] * patches.shape[4]])
    w_ = np.reshape(patterns,
                    newshape=[patterns.shape[0],
                              patterns.shape[1] * patterns.shape[2] * patterns.shape[3]])
    y_ = np.reshape(responses,
                    newshape=[responses.shape[0] * responses.shape[1], responses.shape[2]])
    z_ = np.reshape(winners, newshape=[winners.shape[0] * winners.shape[1]])
    # update response with constant input
    if logs is not None:
        logs[0, :] = immaturity
    for iter_ in range(max_iter):
        for i in range(patterns.shape[0]):
            y_[:, i] = 1.0 / (1 + np.sum(np.abs(x_[:, :] - w_[i, :]), axis=1))
        # winner takes all
        z_[:] = np.argmax(y_, axis=1)
        # update the patterns in memory
        for i in range(patterns.shape[0]):
            input_mask = np.float32(z_ == i)
            count = np.sum(input_mask)
            pattern_mask = count >= MIN_SUPPORT
            attractiveness = pattern_mask * np.sum(input_mask * y_[:, i]) / np.maximum(count, MIN_SUPPORT)
            input_mask = np.reshape(input_mask, newshape=[input_mask.shape[0], 1])
            delta_w = pattern_mask * (np.sum(input_mask * x_, axis=0) / np.maximum(count, MIN_SUPPORT))
            w_[i] = (1 - immaturity[i]) * w_[i] + immaturity[i] * delta_w
            # update the global states of patterns
            immaturity[i] = discount * immaturity[i] + (1 - discount) * (1 - attractiveness)
            acceptance[i] = discount * acceptance[i] + (1 - discount) * count / z_.shape[0]
            if logs is not None:
                logs[min(iter_ + 1, max_iter-1), i] = immaturity[i]
            if iter_ == 0: # response at the first shot
                y_[:, :] = 0.0
                y_[np.arange(y_.shape[0]), z_[:]] = 1
                instant_response = np.copy(responses)

    return instant_response


def allocate_imagination(output_shape, stride, channels, pattern_num):
    assert len(output_shape) == 3
    assert pattern_num == output_shape[2]
    h_i = output_shape[0] * stride
    w_i = output_shape[1] * stride
    c_i = channels
    return np.zeros([h_i, w_i, c_i], dtype=np.float32)


def allocate_patterns(pattern_num, kernel_size, channels):
    patterns = np.random.uniform(0.0, 1.0, [pattern_num, kernel_size, kernel_size, channels])
    vectors = np.reshape(patterns, newshape=[pattern_num, kernel_size * kernel_size * channels])
    return patterns


def recall_from_output(imag_, output_, patterns, patches):
    assert len(imag_.shape) == 3
    assert len(output_.shape) == 3
    assert len(patterns.shape) == 4
    assert len(patches.shape) == 5
    assert patterns.shape[0] == output_.shape[2]
    ksize = patches.shape[2]
    stride = imag_.shape[0] // patches.shape[0]
    y_ = np.reshape(output_, newshape=[output_.shape[0] * output_.shape[1], output_.shape[2]])
    # convert the imag_ to an one-hot vector based array
    max_ids = np.argmax(y_, axis=1)
    y_[:, :] = 0
    y_[np.arange(y_.shape[0]), max_ids] = 1
    w_ = np.reshape(patterns, newshape=[patterns.shape[0],
                                        patterns.shape[1] * patterns.shape[2] * patterns.shape[3]])
    x_ = np.reshape(patches, newshape=[patches.shape[0] * patches.shape[1],
                                       patches.shape[2] * patches.shape[3] * patches.shape[4]])
    x_[:, :] = np.dot(y_, w_)
    # upscale the patches into higher resolution images
    merge_patches(imag_, patches)
    return imag_


def pattern_distance(x, y):
    return np.mean(np.abs(x - y))


# padding type is full
class PatternLayer(object):
    def __init__(self, pattern_num=8, kernel_size=3, stride=2, channels=3, discount=0.98):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pattern_num = pattern_num
        self.channels = channels

        self.patterns = None
        self.patches = None
        self.responses = None
        self.winners = None
        self.immaturity = None
        self.acceptance = None
        self.imagination = None
        self.discount = discount

    def observe(self, input_, max_iter=300, logs=None):
        shape_ = input_.shape
        assert len(shape_) == 3

        if self.patterns is None:
            self.patterns = allocate_patterns(self.pattern_num, self.kernel_size, self.channels)

        if self.patches is None:
            self.patches = allocate_patches(input_.shape, self.kernel_size, self.stride)

        if self.responses is None:
            self.responses = allocate_responses(self.patches.shape, self.patterns.shape)

        if self.winners is None:
            self.winners = allocate_winners(self.responses.shape)

        if self.immaturity is None:
            self.immaturity = allocate_immaturity(self.pattern_num)

        if self.acceptance is None:
            self.acceptance = allocate_acceptance(self.pattern_num)

        # handle the situation when the input dimension changed
        if self.channels > input_.shape[2]:
            print("Pattern Pruning is not supported yet!")
            assert False
        elif self.channels < input_.shape[2]:
            self.channels = input_.shape[2]
            new_patterns = allocate_patterns(self.pattern_num, self.kernel_size, self.channels)
            new_patterns[:, :, :, :self.patterns.shape[3]] = self.patterns[:, :, :, :]
            new_patterns[:, :, :, self.patterns.shape[3]:] = 0.5 # to keep the old memory equal to new ones
            del(self.patterns)
            self.patterns = new_patterns

            # update the working memory
            del(self.patches)
            self.patches = allocate_patches(input_.shape, self.kernel_size, self.stride)

        # NOW RUNNING THE OBSERVATION CORE PROCEDURE
        extract_patches(self.patches, input_, self.kernel_size, self.stride)
        # long term memory should not be directly affected, only short term memory is mutable
        self.patterns_ = np.copy(self.patterns)
        instant_response = pattern_response(self.responses, self.winners, self.patches, self.patterns_,
                         self.immaturity, self.acceptance, self.discount, max_iter, logs)

        return instant_response

    def update(self):
        # check if new patterns found
        new_found = []
        for i in range(len(self.patterns)):
            if self.immaturity[i] < MAX_IMMATURITY and pattern_distance(self.patterns[i], self.patterns_[i]) > PATTERN_DENSITY:
                new_found.append(i)
        print('====== New pattern found: %d =======' % len(new_found))
        # append new pattern to long term memory
        new_patterns = np.zeros([len(self.patterns) + len(new_found), self.kernel_size, self.kernel_size, self.channels])
        new_patterns[:len(self.patterns)] = self.patterns[:]
        for i in range(len(new_found)):
            new_patterns[len(self.patterns) + i] = self.patterns_[new_found[i]]
        del(self.patterns)
        self.patterns = new_patterns

        # update the working memory
        self.pattern_num = len(self.patterns)
        del(self.responses)
        self.responses = allocate_responses(self.patches.shape, self.patterns.shape)
        del(self.winners)
        self.winners = allocate_winners(self.responses.shape)
        # the two states are a little bit different, they are not temporary but long lasting
        new_immaturity = allocate_immaturity(self.pattern_num)
        new_immaturity[:len(self.immaturity)] = self.immaturity[:]
        new_immaturity[len(self.immaturity):] = self.immaturity[new_found]
        del(self.immaturity)
        self.immaturity = new_immaturity
        new_acceptance = allocate_acceptance(self.pattern_num)
        new_acceptance[:len(self.acceptance)] = self.acceptance[:]
        new_acceptance[len(self.acceptance):] = self.acceptance[new_found]
        del(self.acceptance)
        self.acceptance = new_acceptance

    def update_force(self):
        self.patterns = self.patterns_

    def recall(self, output_):
        if self.imagination is None:
            self.imagination = allocate_imagination(output_.shape, self.stride, self.channels, self.pattern_num)
        else:
            self.imagination[:, :, :] = 0
        if self.patches is None:
            self.patches = allocate_patches(self.imagination.shape, self.kernel_size, self.stride)
        recall_from_output(self.imagination, output_, self.patterns, self.patches)
        return self.imagination

    def save(self, path):
        machine_state = dict()
        machine_state['patterns'] = self.patterns
        machine_state['immaturity'] = self.immaturity
        machine_state['acceptance'] = self.acceptance
        with open(path, 'wb') as f:
            pickle.dump(machine_state, f)

    def load(self, path):
        with open(path, 'rb+') as f:
            machine_state = pickle.load(f)
            self.patterns = machine_state['patterns']
            self.immaturity = machine_state['immaturity']
            self.acceptance = machine_state['acceptance']

            self.pattern_num = self.patterns.shape[0]
            self.kernel_size = self.patterns.shape[1]
            self.channels = self.patterns.shape[3]


class PatternNetwork(object):
    def __init__(self, pattern_nums, kernels, strides, channels=3, discount=0.01):
        self.layers = []
        for i in range(len(pattern_nums)):
            if i==0:
                chns = channels
            else:
                chns = self.layers[i-1].pattern_num
            self.layers.append(PatternLayer(pattern_nums[i], kernels[i], strides[i], chns, discount))

    def glance(self, input_):
        res = self.layers[0].observe(input_, 1, None)
        for i in range(len(self.layers) - 1):
            res = self.layers[i + 1].observe(res, 1, None)
        return res

    def observe(self, input_, max_iter=300, logs=None, force_update=False):
        if logs is None:
            logs = [None] * len(self.layers)
        self.layers[0].observe(input_, max_iter, logs[0])
        if force_update:
            self.layers[0].update_force()
        else:
            self.layers[0].update()
        res = self.layers[0].observe(input_, 1, None)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].observe(res, max_iter, logs[i + 1])
            if force_update:
                self.layers[i + 1].update_force()
            else:
                self.layers[i + 1].update()
            res = self.layers[i + 1].observe(res, 1, None)
        return res

    def update(self):
        for i in range(len(self.layers)):
            self.layers[i].update()

    def update_force(self):
        for i in range(len(self.layers)):
            self.layers[i].update_force()

    def recall(self, output_):
        imag = self.layers[len(self.layers) - 1].recall(output_)
        for i in range(len(self.layers) - 1):
            imag = self.layers[len(self.layers) - 2 - i].recall(imag)
        return imag

    def save(self, path_): # the given path should be a directory that stores the parameters of all layers
        for i in range(len(self.layers)):
            self.layers[i].save(path_ + '/PL_%d.npy' % i)

    def load(self, path_):
        for i in range(len(self.layers)):
            self.layers[i].load(path_ + '/PL_%d.npy' % i)


if __name__ == '__main__':
    # train the network with unlabeled examples, actually, the label is also a kind of input
    files = glob.glob('E:/Gits/Datasets/Umbrella/seq-in/*.jpg')[0:30:10]
    files += glob.glob('E:/Gits/Datasets/Umbrella/seq-out/*.jpg')[0:30:10]
    images = [None] * len(files)
    for i in range(len(files)):
        images[i] = np.array(Image.open(files[i]), np.float32) / 255.0
    print('Dataset Loaded!')

    '''
    # Force updating test
    layer = PatternLayer(pattern_num=8, kernel_size=3, stride=2, channels=3, discount=0.98)
    layer_1 = PatternLayer(pattern_num=8, kernel_size=3, stride=2, channels=8, discount=0.98)
    # training first layer
    logs = np.zeros([300, layer.pattern_num])
    layer.observe(images[0], max_iter=300, logs=logs)
    plt.plot(logs)
    plt.show()
    layer.update_force()
    # test first layer
    instant_res = layer.observe(images[0], max_iter=1, logs=None)
    # train second layer
    logs = np.zeros([300, layer_1.pattern_num])
    layer_1.observe(instant_res, max_iter=300, logs=logs)
    plt.plot(logs)
    plt.show()
    layer_1.update_force()
    # test second layer
    instant_res = layer_1.observe(instant_res, max_iter=1, logs=None)
    imag = layer_1.recall(instant_res)
    imag = layer.recall(imag)
    utils.show_rgb(imag)
    exit(0)
    '''

    ''' train together failed!
    net = PatternNetwork(pattern_nums=[8, 8], kernels=[3, 3], strides=[2, 2], channels=3, discount=0.98)

    for sample_id in range(len(images)):
        net.observe(images[sample_id], max_iter=ITER_TIMES, logs=None, force_update=sample_id==0)

    # check if old memory is kept
    for sample_id in range(len(images)):
        utils.show_rgb(images[sample_id])
        res = net.observe(images[sample_id], max_iter=1, logs=None)
        imag = net.recall(res)
        utils.show_rgb(imag)
    
    net.save('../../Models/PatternLayers/')
    '''

    # train layer by layer
    net = PatternNetwork(pattern_nums=[8, 8], kernels=[3, 3], strides=[2, 2], channels=3, discount=0.98)

    # train the first layer
    layer = net.layers[0]
    for sample_id in range(len(images)):
        layer.observe(images[sample_id], max_iter=ITER_TIMES, logs=None)
        if sample_id == 0:
            layer.update_force()
        else:
            layer.update()
    # test first layer
    for sample_id in range(len(images)):
        instant_res = layer.observe(images[sample_id], max_iter=1, logs=None)
        imag = layer.recall(instant_res)
        utils.show_rgb(imag)

    # train the second layer
    layer = net.layers[0]
    for sample_id in range(len(images)):
        layer.observe(images[sample_id], max_iter=ITER_TIMES, logs=None)
        if sample_id == 0:
            layer.update_force()
        else:
            layer.update()
    # test second layer
    for sample_id in range(len(images)):
        instant_res = layer.observe(images[sample_id], max_iter=1, logs=None)
        imag = layer.recall(instant_res)
        utils.show_rgb(imag)


