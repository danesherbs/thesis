'''
Utilities package for autoencoders
'''
from PIL import Image
import numpy as np
import os
from keras.datasets import mnist



'''
Loading ALE data
'''
# Contains screen-grabs from most recent run of ALE
RECORD_PATH = './atari_agents/record/'

'''
Dataset loaders
'''

# General Atari loader
def atari_generator(directory, batch_size=64):
    '''
    Takes directory of Atari 2600 images (3, 210, 160)
    and returns a generator of (1, 84, 84) images.
    '''
    image_names = os.listdir(directory)
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

def count_images(directory):
    return len(os.listdir(directory))

def mnist_generator(batch_size=64):
    (X_train, _), (X_test, _) = mnist.load_data()


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
    from time import gmtime, strftime
    return strftime("%d_%b_%H_%M_%S", gmtime())


'''
MAIN
'''
if __name__ == '__main__':

    # __demo_plot_data()

    # image_path = './atari_agents/record/100.png'
    # image_array = image_to_array(image_path, rgb=False)
    # print(np.min(image_array))
    # print(np.max(image_array))

    directory = './atari_agents/record/test/'
    test_generator = atari_generator(directory, batch_size=64)
    [x for x in test_generator]

