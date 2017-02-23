'''
Utilities package for autoencoders
'''

from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from matplotlib import pyplot as plt
from keras import backend as K
from keras import objectives
from PIL import Image
import numpy as np
import os


'''
Contains screen-grabs from most recent run of ALE
'''
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
    print image_array
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

def load_data(down_sample=False):
    '''
    Makes (X_train, X_test, y_train, y_test) from images in RECORD_PATH
    '''
    X = []
    print 'Loading training data...',
    for filename in os.listdir(RECORD_PATH):
    # for filename in os.listdir(RECORD_PATH)[::100]:
        if not filename.endswith('.png'):
            continue  # skip non-png files
        image_array = image_to_array(os.path.join(RECORD_PATH, filename))
        if down_sample:
            down_sampled = block_reduce(image_array, block_size=(5, 5), func=np.max)
            X.append(down_sampled)
        else:
            X.append(image)
    print 'done.'
    X = np.asarray(X, dtype='uint8')
    Y = np.asarray([-1]*len(X))  # psuedo labels
    return train_test_split(X, Y, test_size=0.10)

def show_subplot(images):
    '''
    Plots monochrome images in a subplot figure
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

def vae_loss(input_img, output_img, mean, log_var, beta=1.0):
        # compute the reconstruction loss (binary cross entropy)
        reconstruction_loss = objectives.binary_crossentropy(input_img.flatten(), output_img.flatten())
        # compute the KL divergence between approximate and latent posteriors
        kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
        return reconstruction_loss + beta * kl_loss

def sampling(args, latent_dim):
        '''
        Inputs: mean and std vectors
        Output: sampled vector
        '''
        z_mean, z_log_var = args  # unpack args
        epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, std=1.0)  # sample from standard normal
        return z_mean + K.exp(z_log_var / 2) * epsilon

'''
MAIN
'''
if __name__ == '__main__':
    # input_img = image_to_array('./atari_agents/record/020377.png', rgb=False)
    # output_img = image_to_array('./atari_agents/record/000170.png', rgb=False)
    # print vae_loss(input_img, output_img)
    load_data()