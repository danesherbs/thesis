'''
Utilities package for autoencoders
'''
from PIL import Image
import numpy as np


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
    
    Implements pre-processing function phi as described in "Human-level control through deep reinforcement learning"
    https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

    Takes images in dataset (all of shape (3, 210, 160)) and returns Y channel resized to (1, 84, 84).
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

def get_data_directory():
    return RECORD_PATH
    

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

    __demo_plot_data()

    # image_path = './atari_agents/record/100.png'
    # image_array = image_to_array(image_path, rgb=False)
    # print(np.min(image_array))
    # print(np.max(image_array))
