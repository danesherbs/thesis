import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, Flatten, Convolution2D, Deconvolution2D, Reshape
from keras.models import Model
from keras.datasets import mnist
# from utils import sampling, vae_loss



from keras import backend as K
from keras import objectives

def sampling(args):
    '''
    Inputs: mean and std vectors
    Output: sampled vector
    '''
    z_mean, z_log_var = args  # unpack args
    epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, std=1.0)  # sample from standard normal
    return z_mean + K.exp(z_log_var / 2) * epsilon

def vae_loss(input_img, output_img):
    # compute the reconstruction loss
    # reconstruction loss is given by cross entropy 
    reconstruction_loss = objectives.binary_crossentropy(input_img.flatten(), output_img.flatten())
    # compute the KL divergence between approximate and latent posteriors
    # KL(q,p) = -0.5 * sum( 1 + log(var_z) - mu_z^2 - var_z)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconstruction_loss + beta * kl_loss


'''
Define parameters
'''
batch_size = 128
nb_epoch = 10
input_w = 28
input_h = 28
nb_filters = 32
pool_size = (2, 2)
kernal_size = (7, 7)
intermediate_dim = 500
latent_dim = 36
beta = 1.0


'''
Load data
'''
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, input_h, input_w)
X_test = X_test.reshape(X_test.shape[0], 1, input_h, input_w)
input_shape = (1, input_h, input_w)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# X_train = X_train[::100]
# X_test  = X_test[::100]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


'''
Define model
'''
input_img = Input(shape=input_shape)

# 'kernal_size' convolution with 'nb_filters' output filters and stride 1x1 on a ?x? image
conv_1 = Convolution2D(32, kernal_size[0], kernal_size[1], border_mode='valid', input_shape=input_shape)(input_img)

# 'kernal_size' convolution with 'nb_filters' output filters and stride 1x1 on a ?x? image
conv_2 = Convolution2D(8, kernal_size[0], kernal_size[1], border_mode='valid')(conv_1)

# low dimensional dense layer
x = Flatten()(conv_2)
x = Dense(intermediate_dim)(x)

# separate dense layers for mu and log(sigma), both of size latent_dim
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# sample from normal with z_mean and z_log_var
# z = Lambda(lambda args: sampling(args, latent_dim), output_shape=(latent_dim,))([z_mean, z_log_var])
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# scale up to dense layer of size 'intermediate_dim'
x = Dense(intermediate_dim)(z)

# scale to original flattened size
x = Dense(2048)(x)

# reshape into original volume
x = Reshape((8, 16, 16))(x)

# 'kernal_size' transposed convolution with 32 output filters and stride 1x1 on a 8 16x16 images
deconv_1 = Deconvolution2D(32, kernal_size[0], kernal_size[1], output_shape=(None, 32, 22, 22), border_mode='valid', input_shape=(8, 16, 16))(x)

# 'kernal_size' transposed convolution with 1 output filter and stride 1x1 on a 32 22x22 images
decoded_img = Deconvolution2D(1, kernal_size[0], kernal_size[1], output_shape=(None, 1, 28, 28), border_mode='valid', input_shape=(32, 22, 22))(deconv_1)

# define model
vae = Model(input_img, decoded_img)

vae.summary()


'''
Compile and fit
'''
# net_vae_loss = lambda input_img, output_img: vae_loss(input_img, output_img, z_mean, z_log_var, beta=beta)
# vae.compile(loss=net_vae_loss, optimizer='adam')
vae.compile(loss=vae_loss, optimizer='rmsprop', metrics=['binary_crossentropy'])
# vae.compile(loss=vae_loss, optimizer='adam')
history = vae.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, nb_epoch=nb_epoch)

# from keras.utils.visualize_util import plot
# plot(vae, to_file='convolutional_variational_autoencoder.png', show_shapes=True)

description = 'beta = ' + str(beta).replace(".", "-") + ', nb_epoch = ' + str(nb_epoch) + ', optimizer = ' + 'rmsprop' + ', loss = ' + 'vae_loss'


'''
Compare reconstruction
'''
decoded_imgs = vae.predict(X_test)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(X_test[0][0])
plt.figure()
plt.imshow(decoded_imgs[0][0])
plt.show()