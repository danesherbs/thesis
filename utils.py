'''
Utilities package for autoencoders
'''

from PIL import Image
import numpy as np
import os
from time import gmtime, strftime
from keras.datasets import mnist
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from scipy.io import loadmat
import matplotlib.pyplot as plt


'''
Loading ALE data
'''
# Contains screen-grabs from most recent run of ALE
RECORD_PATH = './atari_agents/record/'

'''
Dataset loaders
'''
# General Atari loader
def atari_generator(directory, batch_size=64, shuffle=True):
    '''
    Takes directory of Atari 2600 images (3, 210, 160)
    and returns a generator of (1, 84, 84) images.
    '''
    image_names = os.listdir(directory)
    if shuffle:
        np.random.shuffle(image_names)
    dataset_size = len(image_names)
    counter = 0
    # yield data forever
    while True:
        # calculate new counter
        counter = np.mod(counter, dataset_size)
        counter_max = counter + batch_size
        # reset counter if end extends over end of dataset
        if counter_max > dataset_size:
            print("Reached end of dataset. Re-setting counter.")
            counter = 0
            counter_max = batch_size
        # extract next batch of images
        batch_image_names = image_names[counter : counter_max]
        X = []
        for batch_image_name in batch_image_names:
            batch_image = Image.open(os.path.join(directory, batch_image_name))
            batch_array = np.asarray(batch_image)
            X.append(batch_array)
        # reshape and normalise data
        X = np.asarray(X)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        X = X.astype('float32')
        X = (X - np.min(X)) / np.max(X)
        # yield next batch
        yield (X, X)
        # slide counter by batch_size
        counter += batch_size

# MNIST loader
def load_mnist(shuffle=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    input_shape = (1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    if shuffle:
        np.random.shuffle(X_train)
        np.random.shuffle(X_test)
    return (X_train, y_train), (X_test, y_test)

# Frey face loader
def load_frey(shuffle=True):
    '''
    Adapted script from http://dohmatob.github.io/research/2016/10/22/VAE.html.
    '''
    url = "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_filename = os.path.basename(url)
    if not os.path.exists(data_filename):
        __fetch_file(url)
    else:
        print("Data file {} exists.".format(data_filename))
    # reshape data for later convenience
    img_rows, img_cols = 28, 20
    ff = loadmat(data_filename, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T.reshape((-1, img_rows, img_cols))
    X_train = ff[:1800]
    X_test = ff[1800:1900]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    if shuffle:
        np.random.shuffle(X_train)
        np.random.shuffle(X_test)
    return (X_train, None), (X_test, None)


'''
Miscellaneous
'''
def count_images(directory):
    return len(os.listdir(directory))

def make_generator(X, batch_size=64):
    counter = 0
    dataset_size = len(X)
    while True:
        # calculate new counter
        counter = np.mod(counter, dataset_size)
        counter_max = counter + batch_size
        # reset counter if end extends over end of dataset        
        if counter_max > dataset_size:
            # print("Reached end of dataset. Re-setting counter.")
            counter = 0
            counter_max = batch_size
        # yield next batch
        yield (X[counter:counter_max], X[counter:counter_max])
        # slide counter by batch_size
        counter += batch_size

'''
Helpers
'''
def __fetch_file(url):
    '''
    Downloads a file from a URL.
    Adapted script from http://dohmatob.github.io/research/2016/10/22/VAE.html.
    '''
    try:
        f = urlopen(url)
        print("Downloading data file " + url + " ...")
        # Open our local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
        print("Done.")
    #handle errors
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)

def __show_image(X, sample_num=0):
    x = X[sample_num][0]
    plt.imshow(x)
    plt.gray()
    plt.show()

'''
Hyperparameter searching
'''
def build_hyperparameter_string(name, hp_dictionary):
    hyperparameters = sorted(hp_dictionary.keys())
    out_string = name + '_' + __date_and_time_string()
    for hp in hyperparameters:
        hp_value = hp_dictionary[hp]
        out_string += "_" + str(hp) + "_" + str(hp_value)
    return out_string

def __date_and_time_string():
    return strftime("%d_%b_%H_%M_%S", gmtime())


'''
MAIN
'''
if __name__ == '__main__':
    # image_path = './atari_agents/record/100.png'
    # image_array = image_to_array(image_path, rgb=False)
    # print(np.min(image_array))
    # print(np.max(image_array))

    # directory = './atari_agents/record/test/'
    # test_generator = atari_generator(directory, batch_size=64)
    # [x for x in test_generator]

    (X_train, _), (X_test, _) = load_frey()
    __show_image(X_train, sample_num=0)