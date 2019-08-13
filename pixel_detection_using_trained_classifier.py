import numpy as np
import tensorflow as tf
from PIL import Image
import glob

from components import classifier, utils


class BoundingBox(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def area(self):
        return self.w * self.h

    def empty(self):
        return self.area()==0


# Using Fix-Flex Ranging Search strategy
# im_: input image
# func_check: a local function to be called to check the existence of given
# bounding box, it is supposed to satisfy the interfaces:
#   func_check(sub_im, bbox)
def search_bbox(im_, func_check):
    # For the first round, flex both dimension
    DIM_FIX = 0
    DIM_FLEX = 1

    bbox = BoundingBox()
    flag_dim = [DIM_FLEX, DIM_FLEX]

    # check if the given image contains the given object
    bbox_cand = [0, 0, im_.shape[0], im_shape[1]]
    if not func_check(im_, bbox_cand):
        return bbox
    else:
        # search inside this region
        pass
    return bbox


if __name__ == '__main__':
    tf.reset_default_graph()
    graph_classifier = tf.Graph()
    with graph_classifier.as_default():
        # build a classifier network
        json_conf = open('classifier.json', 'rt').read()
        cf = classifier.Classifier(json_conf, optimize_input=False)
        print(cf.info())
        batch_size, height, width, channel = cf._input.shape.as_list()
        _, classes = cf.output.shape.as_list()
        assert channel == 3

    object_id_file = '../Datasets/Umbrella/rand_seq_object.txt'
    blank_id_file = '../Datasets/Umbrella/rand_seq_blank.txt'

    f_object_ids = open(object_id_file, 'rt')
    f_blank_ids = open(blank_id_file, 'rt')

    objects = f_object_ids.readlines()
    blank = f_blank_ids.readlines()

    objects = [f[:-1] for f in objects]
    blank = [f[:-1] for f in blank]

    print("objects=" + str(len(objects)))
    print("blank=" + str(len(blank)))

    # splitting the dataset into training dataset and validation dataset
    OBJECT_ID = 0
    BLANK_ID = 1

    train_splits = [len(objects) // 2, len(blank) // 2]
    valid_splits = [len(objects) // 2, len(blank) // 2]

    train_images = np.zeros([train_splits[0] + train_splits[1], height, width, channel], dtype=np.float32)
    train_labels = np.zeros([train_splits[0] + train_splits[1], classes], dtype=np.float32)
    valid_images = np.zeros([valid_splits[0] + valid_splits[1], height, width, channel], dtype=np.float32)
    valid_labels = np.zeros([valid_splits[0] + valid_splits[1], classes], dtype=np.float32)

    idx = 0
    for i in range(train_splits[0]):
        im_ = Image.open(objects[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, OBJECT_ID] = 1.0
        idx += 1
    for i in range(train_splits[1]):
        im_ = Image.open(blank[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, BLANK_ID] = 1.0
        idx += 1

    idx = 0
    for i in range(valid_splits[0]):
        im_ = Image.open(objects[train_splits[0] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, OBJECT_ID] = 1.0
        idx += 1
    for i in range(valid_splits[1]):
        im_ = Image.open(blank[train_splits[1] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, BLANK_ID] = 1.0
        idx += 1

    n, h, w, c = train_images.shape
    ksizes = 1*h//3, 1*w//3
    stride = 3, 3
    step_y = (h - ksizes[0]) // stride[0]
    step_x = (w - ksizes[1]) // stride[1]
    with graph_classifier.as_default():
        cf.load('./models/umbrella_classifier.ckpt')
        idx = 259
        attention_map = np.zeros([h, w], np.float32)
        for i in range(step_y):
            for j in range(step_x):
                mask_ = np.zeros([1, h, w, 1])
                mask_[0, i*stride[0]:i*stride[0] + ksizes[0], j*stride[1]:j*stride[1] + ksizes[1], 0] = 1.0
                im_ = mask_ * train_images[idx]
                #utils.show_rgb(im_[0])
                guess_label = cf.test(im_)[0]
                if (np.argmax(guess_label) == np.argmax(train_labels[idx])):
                    attention_map[i*stride[0]:i*stride[0] + ksizes[0], j*stride[1]:j*stride[1] + ksizes[1]] += 1.0
                else:
                    print(['++++', '----'][np.argmax(guess_label)])
        # normalize the attention map
        attention_map = np.minimum(utils.normalize(attention_map) + 0.1, 1.0)
        attention_map = np.reshape(attention_map, [h, w, 1])
        utils.show_rgb(attention_map * train_images[idx])
        utils.show_gray(attention_map[:, :, 0])

