import numpy as np
import tensorflow as tf
from PIL import Image
import glob

from components import classifier

INPUT_SIZE = (256, 256)
BATCH_SIZE = 6


if __name__ == '__main__':
    path_ = "../Datasets/Hands"
    hands = glob.glob(path_ + "/with-hand/*.JPG")
    blank = glob.glob(path_ + "/without-hand/*.JPG")
    print("hands=" + str(len(hands)))
    print("blank=" + str(len(blank)))
    for fn_ in blank:
        im_ = Image.open(fn_)
        im_ = im_.resize(INPUT_SIZE)
        pass
    # build a classifier network
    json_conf = '{"classes": 2, \
    "inputs": [1, 256, 256, 3], \
    "filters": [8, 16, 16, 8], \
    "ksizes": [3, 3, 3, 3], \
    "strides": [2, 2, 2, 2], \
    "relus": [0, 1, 0, 1], \
    "links":[[], [], [], [0]],\
    "fc": [8, 32, 8]}'
    cf = classifier.Classifier(json_conf)
    print(cf.layers)


