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
        image = image.convert('L')  # convert to black and white
    return np.asarray(image, dtype=np.uint8)  # uint8 \in [0,255]

def __example_image_to_array():
    '''
    Takes random screen-grab in RECORD_PATH and shows array and image
    '''
    image_path = './atari_agents/record/020377.png'
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

def load_data(down_scale_factor=0.5):
    '''
    Makes (X_train, X_test, y_train, y_test) from images in RECORD_PATH
    '''
    from scipy.misc import imresize
    from sklearn.model_selection import train_test_split
    import os
    X = []
    print('Loading training data...')
    for filename in os.listdir(RECORD_PATH):
        if not filename.endswith('.png'):
            continue  # skip non-png files
        image_array = image_to_array(os.path.join(RECORD_PATH, filename))
        down_sampled = imresize(image_array, down_scale_factor)
        X.append(down_sampled)
    X = np.asarray(X, dtype='uint8')
    y = np.asarray([-1]*len(X))  # psuedo labels
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.10, random_state=42)
    return (X_train, None), (X_test, None)

def __demo_plot_data():
    # load data and print shape
    (X_train, _), (X_test, _) = load_data(down_scale_factor=0.5)
    print(X_train.shape)
    # plot sample image
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(X_train[1], cmap='inferno')
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
    from time import gmtime, strftime
    return strftime("%d_%b_%H_%M_%S", gmtime())


'''
Graphing utilities
'''
def show_subplot(images):
    '''
    Plots monochrome images in a subplot figure
    '''
    from matplotlib import pyplot as plt
    nrows = 4
    ncols = len(images) / nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i in xrange(nrows):
        for j in xrange(ncols):
            if ncols * i + j < len(images):
                axs[i][j].matshow(images[nrows+ncols])
    plt.tight_layout()
    plt.show()

def __example_show_subplot():
    import cPickle as pickle
    imgs = pickle.load(open('encoded_imgs.pickle', 'rb'))
    show_subplot(imgs)

'''
MAIN
'''
if __name__ == '__main__':
    # input_img = image_to_array('./atari_agents/record/020377.png', rgb=False)
    # output_img = image_to_array('./atari_agents/record/000170.png', rgb=False)
    __demo_plot_data()
