'''
Utilities package for autoencoders
'''

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

RECORD_PATH = './atari_agents/record/'


def image_to_array(image_path, rgb=False):
    '''
    Converts image to numpy array
    '''
    image = Image.open(image_path)
    if not rgb:
        image = image.convert('1')  # convert to black and white
    return np.asarray(image, dtype=np.uint8)  # uint8 \in [0,255]

def __example_image_to_array():
    '''
    Takes random screen-grab in RECORD_PATH and shows image
    '''
    image_path = './atari_agents/record/020377.png'
    image_array = image_to_array(image_path)
    print image_to_array(image_path).shape
    Image.fromarray(np.uint8(image_to_array(image_path))).show()

def load_data():
    '''
    Makes (X_train, X_test, y_train, y_test) from images in RECORD_PATH
    '''
    X = []
    print 'Loading training data...',
    for filename in os.listdir(RECORD_PATH)[::200]:
        if not filename.endswith('.png'):
            continue  # skip non-png files
        X.append(image_to_array(os.path.join(RECORD_PATH, filename)))
    print 'done.'
    X = np.asarray(X, dtype='uint8')
    y = np.asarray([-1]*len(X))  # psuedo labels
    return train_test_split(X, y, test_size=0.10)

def show_subplot(images):
    '''
    Plots images in a subplot figure
    '''
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
    __example_image_to_array()
