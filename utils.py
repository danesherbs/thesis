'''
Utilities package for autoencoders
'''
from PIL import Image
import numpy as np
import os


'''
Loading ALE data
'''
# Contains screen-grabs from most recent run of ALE
RECORD_PATH = './atari_agents/record/'

def image_to_array(image_path, rgb=False):
    '''
    Converts image to numpy array
    '''
    image = Image.open(image_path)
    if not rgb:
        image = image.convert('L')  # extract luminance
    return np.asarray(image)

def __example_image_to_array():
    '''
    Takes random screen-grab in RECORD_PATH and shows array and image
    '''
    image_path = './atari_agents/record/0.png'
    image_array = image_to_array(image_path, rgb=False)
    print(image_array)
    Image.fromarray(np.uint8(image_to_array(image_path))).show()

def array_to_image(image_array):
    '''
    Converts numpy array to image
    '''
    return Image.fromarray(np.uint8(image_array))

def __example_array_to_image():
    '''
    Shows output of array_to_image for random screen-grab from RECORD_PATH
    '''
    image_path = './atari_agents/record/020377.png'
    image_array = image_to_array(image_path, rgb=False)
    image = array_to_image(image_array)
    image.show()

def load_data():
    '''
    Makes (X_train, X_test, y_train, y_test) from images in RECORD_PATH
    '''
    from scipy.misc import imresize
    from sklearn.model_selection import train_test_split
    import os
    X = []
    print('Loading training data...', end=' ')
    for filename in os.listdir(RECORD_PATH)[::100]:
        if not filename.endswith('.png'):
            continue  # skip non-png files
        image = Image.open(os.path.join(RECORD_PATH, filename))
        image = image.convert('L')
        image = image.resize((84, 84), Image.ANTIALIAS)
        X.append(np.asarray(image))
    X = np.asarray(X)
    y = np.asarray([-1]*len(X))  # psuedo labels
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.10, random_state=42)
    print('done.')
    return (X_train, None), (X_test, None)

def __demo_plot_data():
    # load data and print shape
    (X_train, _), (X_test, _) = load_data()
    print(X_train.shape)
    # plot sample image
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(X_train[30], cmap='inferno')
    plt.show()

def data_generator(directory, batch_size=64):
    image_names = os.listdir(directory)
    dataset_size = len(image_names)
    counter = 0
    # yield data forever
    while True:
        # extract next batch of images
        counter = np.mod(counter, dataset_size)
        counter_max = min(counter + batch_size, dataset_size)
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
        # increment counter for next call
        counter += batch_size

def count_images(directory):
    return len(os.listdir(directory))


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
    data_generator(directory, batch_size=5)

