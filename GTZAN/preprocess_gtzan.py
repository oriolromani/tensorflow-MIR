__author__ = 'oriol'

import essentia.standard as es
import os
from collections import defaultdict
import timeit
import cPickle as pickle
import numpy as np
from random import shuffle
import numpy as np


#  help functions
def create_hot_vector(num_classes, position):
    hot_vector = np.zeros(num_classes, dtype=int)
    hot_vector[position] = 1
    return hot_vector


# extract spectrograms form all the files contained in path,
# store them in pickle file
def extract_spectrograms(path):
    subfolders = os.listdir(path)

    # init variables and algorithms
    fs = 22050
    # a 100 ms window size
    window_size = 2048
    hop_size = 1024
    spectrum = es.Spectrum()
    window = es.Windowing(type='blackmanharris62')
    mel = es.MelBands(numberBands=40)
    Results = defaultdict(list)

    for folder in subfolders:
        if folder.find('features') == -1:
            for filename in os.listdir(os.path.join(path, folder)):
                print "processing file: %s" % filename
                start_time = timeit.default_timer()
                filepath = os.path.join(path, folder, filename)
                name, extension = os.path.splitext(filepath)
                if extension == '.au':
                    os.rename(filepath, name + ".wav")
                    filepath = name + ".wav"
                audio_file = es.EasyLoader(filename=filepath, sampleRate=fs)
                audio = audio_file.compute()
                # take just a third part of the sound (about 10 s)
                for frame in es.FrameGenerator(audio, window_size, window_size):
                    spec = spectrum(window(frame))
                    me = mel(spec)
                    Results[folder].append(me)
                elapsed = timeit.default_timer() - start_time
                print 'file processed in %.4f s' % elapsed
    with open('features.json', 'w') as outfile:
        pickle.dump(Results, outfile)


def pre_process_data():
    print "---Pre_processing data in the tensorflow shape---"
    with open('features.json', 'rb') as fp:
        data = pickle.load(fp)
    train = []
    train_tmp = []
    _y_train = []
    test = []
    test_tmp = []
    _y_test = []
    labels = {}

    for index, key in enumerate(data.keys()):
        label = create_hot_vector(len(data.keys()), index)
        labels[key] = label
        for i, instance in enumerate(data[key]):
            if i < 25:
                test_tmp.append([instance, label])
            else:
                train_tmp.append([instance, label])
    # randomly distribute the training instances
    # shuffle(train_tmp)
    # store labels and instances in different lists
    for instance in train_tmp:
        train.append(instance[0])
        _y_train.append(instance[1])
    for instance in test_tmp:
        test.append(instance[0])
        _y_test.append(instance[1])
    print "---Data already processed---"
    return np.asarray(train), np.asarray(_y_train), np.asarray(test), np.asarray(_y_test)








