import numpy as np
import tensorflow as tf
from PIL import Image
import glob

from components import classifier, utils


if __name__ == '__main__':
    tf.reset_default_graph()
    # build a classifier network
    json_conf = open('net_hand_detect.json', 'rt').read()
    cf = classifier.Classifier(json_conf)
    for layer in cf.layers:
        print(layer.name + ": " + str(layer.shape))
    batch_size, height, width, channel = cf.input_.shape.as_list()
    # load the data
    path_ = "../Datasets/Hands"
    hands = glob.glob(path_ + "/with-hand/*.JPG")
    blank = glob.glob(path_ + "/without-hand/*.JPG")
    print("hands=" + str(len(hands)))
    print("blank=" + str(len(blank)))
    for fn_ in blank:
        im_ = Image.open(fn_)
        im_ = im_.resize((height, width))
    cf.sess.close()


