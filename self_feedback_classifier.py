import numpy as np
import tensorflow as tf
from PIL import Image
import glob

from components import classifier, utils


if __name__ == '__main__':
    tf.reset_default_graph()
    # build a classifier network
    json_conf = open('classifier.json', 'rt').read()
    cf = classifier.Classifier(json_conf, optimize_input=False)
    for layer in cf.layers:
        print(layer.name + ": " + str(layer.shape))
    batch_size, height, width, channel = cf._input.shape.as_list()
    _, classes = cf.output.shape.as_list()
    assert channel == 3

    f_plant_ids = open('../Datasets/Plants/rand_seq_plant.txt', 'rt')
    f_blank_ids = open('../Datasets/Plants/rand_seq_blank.txt', 'rt')

    plant = f_plant_ids.readlines()
    blank = f_blank_ids.readlines()

    plant = [f[:-1] for f in plant]
    blank = [f[:-1] for f in blank]

    print("plant=" + str(len(plant)))
    print("blank=" + str(len(blank)))

    # splitting the dataset into training dataset and validation dataset
    PLANT_ID = 0
    BLANK_ID = 1

    train_splits = [len(plant) // 2, len(blank) // 2]
    valid_splits = [len(plant) // 2, len(blank) // 2]

    train_images = np.zeros([train_splits[0] + train_splits[1], height, width, channel])
    train_labels = np.zeros([train_splits[0] + train_splits[1], classes])
    valid_images = np.zeros([valid_splits[0] + valid_splits[1], height, width, channel])
    valid_labels = np.zeros([valid_splits[0] + valid_splits[1], classes])

    idx = 0
    for i in range(train_splits[0]):
        im_ = Image.open(plant[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, PLANT_ID] = 1.0
        idx += 1
    for i in range(train_splits[1]):
        im_ = Image.open(blank[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, BLANK_ID] = 1.0
        idx += 1

    idx = 0
    for i in range(valid_splits[0]):
        im_ = Image.open(plant[train_splits[0] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, PLANT_ID] = 1.0
        idx += 1
    for i in range(valid_splits[1]):
        im_ = Image.open(blank[train_splits[1] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, BLANK_ID] = 1.0
        idx += 1

    cf.init_blank_model()
    cf.train(train_images, train_labels, 1e-3, 300, valid_images, valid_labels)
    cf.save('./models/plant_classifier.ckpt')
    cf.close()
    input()






