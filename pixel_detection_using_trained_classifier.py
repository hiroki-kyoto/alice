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
    cf = classifier.Classifier(2)


