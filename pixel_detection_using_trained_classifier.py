import numpy as np
import tensorflow as tf
from PIL import Image
import glob

class Classifier(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # build a graph...
            self.sess = tf.Session()
        pass

    def load_model(self, path):
        pass

    def learn(self, data):
        pass

    def inference(self, data):
        pass

    def finalize(self):
        pass


if __name__ == '__main__':
    path_ = "../Datasets/PixelLevelDetectionUsingTrainedClassifier"
    hands = glob.glob(path_ + "/with-hand/*.JPG")
    blank = glob.glob(path_ + "/without-hand/*.JPG")
    print("hands=" + str(len(hands)))
    print("blank=" + str(len(blank)))
    # build a classifier network
    cf = Classifier()
