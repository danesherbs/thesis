import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, Flatten, Convolution2D, Deconvolution2D, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras import objectives
import utils



'''
Required functions for latent space and training
'''
def sampling(args):
    z_mean, z_log_var = args  # unpack args
    epsilon = K.random_normal(shape=(1, 30, 52), mean=0.0, std=1.0)  # sample from standard normal
    return z_mean + K.exp(z_log_var / 2) * epsilon

def vae_loss(input_img, output_img):
    reconstruction_loss = objectives.binary_crossentropy(input_img.flatten(), output_img.flatten())
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconstruction_loss + beta * kl_loss


'''
Define parameters
'''
batch_size = 32
nb_epoch = 10
nb_filters = 8
pool_size = (2, 2)
kernal_size = (7, 7)
intermediate_dim = 500
latent_dim = 36
beta = 1.0


'''
Load data
'''
X_train, X_test, _, _ = utils.load_data(down_sample=True)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
input_shape = X_train.shape[1:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


'''
Define model
'''
input_img = Input(shape=input_shape)

# 'kernal_size' convolution with 'nb_filters' output filters and stride 1x1 on a ?x? image
conv = Convolution2D(nb_filters, kernal_size[0], kernal_size[1], border_mode='valid', input_shape=input_shape)(input_img)

# separate dense layers for mu and log(sigma), both of size latent_dim
z_mean = Convolution2D(1, kernal_size[0], kernal_size[1], border_mode='valid')(conv)
z_log_var = Convolution2D(1, kernal_size[0], kernal_size[1], border_mode='valid')(conv)

# sample from normal with z_mean and z_log_var
z = Lambda(sampling, output_shape=(1, 30, 52))([z_mean, z_log_var])

# 'kernal_size' transposed convolution with 32 output filters and stride 1x1 on a 8 16x16 images
deconv = Deconvolution2D(nb_filters, kernal_size[0], kernal_size[1], output_shape=(None, 8, 36, 58), border_mode='valid', input_shape=(1, 30, 52))(z)

# 'kernal_size' transposed convolution with 1 output filter and stride 1x1 on a 32 22x22 images
decoded_img = Deconvolution2D(1, kernal_size[0], kernal_size[1], output_shape=(None, 1, 42, 64), border_mode='valid', input_shape=(8, 36, 58))(deconv)

# define model
encoder = Model(input_img, z)
vae = Model(input_img, decoded_img)

vae.summary()


'''
Compile and fit
'''
vae.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_crossentropy'])
vae.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, nb_epoch=nb_epoch)


'''
Compare reconstruction
'''
latent_imgs = encoder.predict(X_test)
decoded_imgs = vae.predict(X_test)
import matplotlib.pyplot as plt
plt.matshow(X_test[0][0])
plt.matshow(latent_imgs[0][0])
plt.matshow(decoded_imgs[0][0])
plt.show()