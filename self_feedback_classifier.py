import numpy as np
import tensorflow as tf
from PIL import Image
import glob

from components import classifier, utils, generator


if __name__ == '__main__':
    tf.reset_default_graph()
    graph_classifier = tf.Graph()
    with graph_classifier.as_default():
        # build a classifier network
        json_conf = open('classifier.json', 'rt').read()
        cf = classifier.Classifier(json_conf, optimize_input=False, lr=1e-7)
        print(cf.info())
        batch_size, height, width, channel = cf._input.shape.as_list()
        _, classes = cf.output.shape.as_list()
        assert channel == 3

    # process the dataset
    object_id_file = '../Datasets/Umbrella/rand_seq_object.txt'
    blank_id_file = '../Datasets/Umbrella/rand_seq_blank.txt'

    '''
    # save the dataset for the first time
    object_files = glob.glob('../Datasets/Umbrella/seq-in/*.jpg')
    blank_files = glob.glob('../Datasets/Umbrella/seq-out/*.jpg')

    f_object_ids = open(object_id_file, 'wt')
    f_blank_ids = open(blank_id_file, 'wt')

    object_ids = np.random.permutation(len(object_files))
    blank_ids = np.random.permutation(len(blank_files))

    for i in range(len(object_ids)):
        f_object_ids.write(object_files[object_ids[i]] + '\n')
    for i in range(len(blank_ids)):
        f_blank_ids.write(blank_files[blank_ids[i]] + '\n')
    exit(0)
    '''

    # read the dataset with configure file
    f_object_ids = open(object_id_file, 'rt')
    f_blank_ids = open(blank_id_file, 'rt')

    objects = f_object_ids.readlines()
    blanks = f_blank_ids.readlines()

    objects = [f[:-1] for f in objects]
    blanks = [f[:-1] for f in blanks]

    print("objects=" + str(len(objects)))
    print("blanks=" + str(len(blanks)))

    # splitting the dataset into training dataset and validation dataset
    PLANT_ID = 0
    BLANK_ID = 1

    train_splits = [len(objects) // 2, len(blanks) // 2]
    valid_splits = [len(objects) // 2, len(blanks) // 2]

    train_images = np.zeros([train_splits[0] + train_splits[1], height, width, channel], dtype=np.float32)
    train_labels = np.zeros([train_splits[0] + train_splits[1], classes], dtype=np.float32)
    valid_images = np.zeros([valid_splits[0] + valid_splits[1], height, width, channel], dtype=np.float32)
    valid_labels = np.zeros([valid_splits[0] + valid_splits[1], classes], dtype=np.float32)

    idx = 0
    for i in range(train_splits[0]):
        im_ = Image.open(objects[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, PLANT_ID] = 1.0
        idx += 1
    for i in range(train_splits[1]):
        im_ = Image.open(blanks[i])
        train_images[idx, :, :, :] = im_.resize((height, width))
        train_labels[idx, BLANK_ID] = 1.0
        idx += 1

    idx = 0
    for i in range(valid_splits[0]):
        im_ = Image.open(objects[train_splits[0] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, PLANT_ID] = 1.0
        idx += 1
    for i in range(valid_splits[1]):
        im_ = Image.open(blanks[train_splits[1] + i])
        valid_images[idx, :, :, :] = im_.resize((height, width))
        valid_labels[idx, BLANK_ID] = 1.0
        idx += 1

    # train the classifier
    with graph_classifier.as_default():
         #cf.init_blank_model()
        cf.recover('./models/umbrella_classifier.ckpt')
        cf.train(train_images, train_labels, 1e-3, 150, valid_images, valid_labels)
        cf.save('./models/umbrella_classifier.ckpt')

    with graph_classifier.as_default():
        # test the model
        cf.load('./models/umbrella_classifier.ckpt')
        score = 0.0
        for i in range(len(valid_images)):
            guess_label = cf.test(valid_images[i:i+1])[0]
            if np.argmax(guess_label) == np.argmax(valid_labels[i]):
                score += 1.0
        print('Validation Accuracy is : %.3f' % (score/len(valid_labels)))

    print('classifier is done.')
    print('========================================')
    exit(0)

    graph_generator = tf.Graph()
    with graph_generator.as_default():
        # build a classifier network
        json_conf = open('generator.json', 'rt').read()
        gen = generator.Generator(json_conf, optimize_input=False)
        print(gen.info())
        batch_size, classes = gen._input.shape.as_list()
        _, h, w, c = gen.output.shape.as_list()
        assert channel == 3
        assert c == 3
        assert classes == 2

    # now run the classifier to get the latent code, and feed them into the generator
    # to retrieve images of the related classes
    train_latents = np.zeros([len(train_labels), classes], dtype=np.float32)
    valid_latents = np.zeros([len(valid_labels), classes], dtype=np.float32)
    train_feedbacks = np.zeros([len(train_images), h, w, c], dtype=np.float32)
    valid_feedbacks = np.zeros([len(valid_images), h, w, c], dtype=np.float32)
    num_train = 0
    num_valid = 0

    with graph_classifier.as_default():
        print('Generating the training dataset for training the generator...')
        cf.load('./models/plant_classifier.ckpt')
        for i in range(len(train_images)):
            latent_code = cf.test(train_images[i:i+1])
            # save these latent code for generation training
            if np.argmax(latent_code[0]) == np.argmax(train_labels[i]):
                train_latents[num_train, :] = latent_code[0]
                train_feedbacks[num_train, :, :, :] = train_images[i]/255.0
                num_train += 1
        print('Datasets for training the generator is done with %d samples.' % num_train)

        print('Generating the training dataset for validating the generator...')
        for i in range(len(valid_images)):
            latent_code = cf.test(valid_images[i:i + 1])
            # save these latent code for generation training
            if np.argmax(latent_code[0]) == np.argmax(valid_labels[i]):
                valid_latents[num_valid, :] = latent_code[0]
                valid_feedbacks[num_valid, :, :, :] = valid_images[i]/255.0
                num_valid += 1
        print('Datasets for validating the generator is done with %d samples.' % num_valid)

    print('Training the generator started...')
    train_latents = train_latents[:num_train]
    train_feedbacks = train_feedbacks[:num_train]
    valid_latents = valid_latents[:num_valid]
    valid_feedbacks = valid_feedbacks[:num_valid]

    print(train_latents[:6])
    exit(0)

    with graph_generator.as_default():
        # train the generator
        gen.init_blank_model()
        gen.train(train_latents, train_feedbacks, 0.01, 300, valid_latents, valid_feedbacks)
        gen.save('./models/plant_generator.ckpt')
        gen.close()
    print('Training the generator done.')


