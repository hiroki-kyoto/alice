# Using stacked Self Organizing Maps to classifier and generate images

import numpy as np
from components import utils
from PIL import Image
import matplotlib.pyplot as plt

MIN_NORM = 1e-3

### input has to be bounded between 0 and 1 as a float32 tensor

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


def allocate_states(num_patterns):
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


def normalize_patterns(x):
    for i in range(x.shape[0]):
        norm_ = np.sqrt(np.sum(np.square(x[i, :])))
        while norm_ < MIN_NORM:
            x[i, :] = np.random.uniform(0.0, 1.0, x.shape[1])
            norm_ = np.sqrt(np.sum(np.square(x[i, :])))
        x[i, :] = x[i, :] / norm_
    return x


# patterns are all normalized vectors of unit length
def pattern_response(responses, winners, states, patches, patterns, learning_rate):
    assert len(responses.shape) == 3
    assert len(patches.shape) == 5
    assert len(patterns.shape) == 4
    assert len(winners.shape) == 2
    assert len(states.shape) == 1

    assert patches.shape[2] == patterns.shape[1]
    assert patches.shape[3] == patterns.shape[2]
    assert patches.shape[4] == patterns.shape[3]

    assert responses.shape[0] == patches.shape[0]
    assert responses.shape[1] == patches.shape[1]
    assert responses.shape[2] == patterns.shape[0]

    assert winners.shape[0] == responses.shape[0]
    assert winners.shape[1] == responses.shape[1]

    assert states.shape[0] == responses.shape[2]

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
    # normalize the input patches
    x_[:, :] = np.maximum(x_[:, :], MIN_NORM)
    x_[:, :] = x_[:, :] / np.sqrt(np.sum(np.square(x_[:, :]), axis=1, keepdims=True))
    for i in range(patterns.shape[0]):
        y_[:, i] = np.sum(w_[i, :] * x_[:, :], axis=1)
    # winner takes all
    z_[:] = np.argmax(y_, axis=1)
    # update the patterns in memory
    for i in range(patterns.shape[0]):
        #mask = np.reshape(np.float32(z_ == i) * y_[:, i], newshape=[len(z_), 1])
        mask = np.reshape(np.float32(z_ == i), newshape=[len(z_), 1])
        count = np.maximum(np.sum(z_ == i), 1)
        states[i] = 1.0 - np.sum(np.float32(z_ == i) * y_[:, i]) / count
        w_[i, :] = (1.0 - learning_rate * states[i]) *  w_[i, :] + learning_rate * states[i] * np.sum(mask * x_, axis=0) / count
    # return the responses as one-hot vectors
    y_[:, :] = 0.0
    y_[np.arange(y_.shape[0]), z_[:]] = 1.0


def allocate_imagination(output_shape, stride, channels, pattern_num):
    assert len(output_shape) == 3
    assert pattern_num == output_shape[2]
    h_i = output_shape[0] * stride
    w_i = output_shape[1] * stride
    c_i = channels
    return np.zeros([h_i, w_i, c_i], dtype=np.float32)


def recall_from_output(imag_, output_, patterns, patches):
    assert len(imag_.shape) == 3
    assert len(output_.shape) == 3
    assert len(patterns.shape) == 4
    assert len(patches.shape) == 5
    assert patterns.shape[0] == output_.shape[2]
    ksize = patches.shape[2]
    stride = imag_.shape[0] // patches.shape[0]
    y_ = np.reshape(output_, newshape=[output_.shape[0] * output_.shape[1], output_.shape[2]])
    w_ = np.reshape(patterns, newshape=[patterns.shape[0],
                                        patterns.shape[1] * patterns.shape[2] * patterns.shape[3]])
    x_ = np.reshape(patches, newshape=[patches.shape[0] * patches.shape[1],
                                       patches.shape[2] * patches.shape[3] * patches.shape[4]])
    x_[:, :] = np.dot(y_, w_)
    # upscale the patches into higher resolution images
    imag_[] # reverse the process as extract patches but caring about the overlapping problems.


# padding type is full
class PatternLayer(object):
    def __init__(self, pattern_num=8, kernel_size=3, stride=2, channels=3, learning_rate=1e-4):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pattern_num = pattern_num
        self.channels = channels
        self.patterns = np.random.uniform(0.0, 1.0, [pattern_num, kernel_size, kernel_size, channels])
        vectors = np.reshape(self.patterns, newshape=[pattern_num, kernel_size * kernel_size * channels])
        normalize_patterns(vectors)
        self.patches = None
        self.responses = None
        self.winners = None
        self.states = None
        self.imagination = None
        self.learning_rate = learning_rate

    def observe(self, input_):
        shape_ = input_.shape
        assert len(shape_) == 3
        assert shape_[2] == self.patterns.shape[3]

        if self.patches is None:
            self.patches = allocate_patches(input_.shape, self.kernel_size, self.stride)

        if self.responses is None:
            self.responses = allocate_responses(self.patches.shape, self.patterns.shape)

        if self.winners is None:
            self.winners = allocate_winners(self.responses.shape)

        if self.states is None:
            self.states = allocate_states(self.pattern_num)

        extract_patches(self.patches, input_, self.kernel_size, self.stride)
        pattern_response(self.responses, self.winners, self.states, self.patches, self.patterns, self.learning_rate)
        vectors = np.reshape(self.patterns, newshape=[self.pattern_num, self.kernel_size * self.kernel_size * self.channels])
        normalize_patterns(vectors)
        return self.responses

    def recall(self, output_):
        if self.imagination is None:
            self.imagination = allocate_imagination(output_.shape, self.stride, self.channels, self.pattern_num)
        if self.patches is None:
            self.patches = allocate_patches(self.imagination.shape, self.kernel_size, self.stride)
        recall_from_output(self.imagination, output_, self.patterns, self.patches)
        return self.imagination

    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == '__main__':
    pattens = [256, 16, 128, 16, 64, 16, 32, 16, 16, 8]
    kernels = [3, 1, 3, 1, 3, 1, 3, 1, 1, 1]
    # train the stacked SOMs layer by layer
    layers = []

    input_ = np.array(Image.open('E:/Gits/Datasets/Umbrella/seq-in/I_0.jpg'))
    utils.show_rgb(input_)

    layer = PatternLayer(pattern_num=8, kernel_size=3, stride=2, channels=3, learning_rate=1e0)
    res = layer.observe(input_)
    utils.show_gray(layer.winners, min=0, max=res.shape[2]-1)
    ITER_TIMES = 500
    losses = np.zeros(ITER_TIMES)
    for i in range(ITER_TIMES):
        res = layer.observe(input_)
        losses[i] = np.mean(layer.states)
    utils.show_gray(layer.winners, min=0, max=res.shape[2]-1)
    plt.plot(losses)
    plt.show()

    layer.recall(res)


