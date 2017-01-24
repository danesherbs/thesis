import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Convolution2D, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


# load data
from utils import load_data
X_train, X_test, _, _ = load_data()

# specify depth for Theano
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# cast to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise values
X_train /= np.max(X_train)
X_test /= np.max(X_test)

# set constants
original_dim = X_train.shape[1:]
intermediate_dim = 100
latent_dim_sqrt = 4  # < sqrt(intermediate_dim) else not an autoencoder
latent_dim = latent_dim_sqrt**2
nb_epoch = 75

# define architecture
input_img = Input(shape=original_dim)

y = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=X_train.shape[1:])(input_img)
y = MaxPooling2D((2, 2), border_mode='same', name='encoded')(y)
y = Flatten()(y)

h = Dense(intermediate_dim, activation='relu')(y)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    '''
    Inputs: mean and std vectors
    Output: sampled vector
    '''
    z_mean, z_log_var = args  # unpack args
    epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, std=1.0)  # sample from standard normal
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(np.prod(original_dim), activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
decoded_img = Reshape(original_dim)(x_decoded_mean)

def vae_loss(input_img, output_img):
    xent_loss = np.prod(original_dim) * objectives.binary_crossentropy(input_img, output_img)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# build model and train
vae = Model(input_img, decoded_img)
# vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=['accuracy'])
vae.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
vae.summary()
vae.fit(X_train, X_train, shuffle=True, nb_epoch=nb_epoch, verbose=1)

# show encoded images
encoder = Model(input_img, z)
encoded_imgs = encoder.predict(X_test)
print X_test.shape
print encoded_imgs.shape
plt.matshow(encoded_imgs[0].reshape(latent_dim_sqrt,latent_dim_sqrt))
plt.show()

# show decoded images
decoded_imgs = vae.predict(X_test)
plt.imshow(decoded_imgs[0][0])
plt.show()
