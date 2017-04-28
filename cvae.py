import keras
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import losses
from keras import optimizers
from keras import initializers
from keras.callbacks import EarlyStopping
import numpy as np
import utils



'''
Required functions for latent space and training
'''
def sampling(args):
    # unpack arguments
    z_mean, z_log_var = args
    # need mean and std for each point
    assert z_mean.shape[1:] == z_log_var.shape[1:]
    # output shape is same as mean and log_var
    output_shape = z_mean.shape[1:]
    # sample from standard normal
    epsilon = K.random_normal(shape=output_shape, mean=0.0, stddev=1.0)
    # reparameterization trick
    return z_mean + K.exp(z_log_var) * epsilon

def vae_loss(y_true, y_pred):
    # y_true is of shape (batch_size,) + input_shape
    # y_pred is of shape (batch_size,) + output_shape
    # For example, training an autoencoder on MNIST with batch_size = 64 gives
    # y_true and y_pred both a shape of of shape (64, 1, 28, 28).

    # Flatten y_true and y_pred of shape (batch_size, 1, 28, 28) to (batch_size, 1 * 28 * 28).
    # Elements are in the interval [0, 1], which can be interpreted as probabilities.
    y_true = K.reshape(y_true, (batch_size, np.prod(input_shape)))
    y_pred = K.reshape(y_pred, (batch_size, np.prod(input_shape)))

    # Take the sum of the binary cross entropy for each image in the batch.
    # Reconstruction loss is of the shape (batch_size, 1).
    reconstruction_loss = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
    # reconstruction_loss = K.mean(np.prod(input_shape) * K.binary_crossentropy(y_pred, y_true), axis=-1)

    # Get latent shape
    latent_shape = z.get_shape().as_list()[1:]
    # Flatten latent space into shape (batch_size,) + flattened_latent_space
    z_mean_flat = K.reshape(z_mean, (batch_size, np.prod(latent_shape)))
    z_log_var_flat = K.reshape(z_log_var, (batch_size, np.prod(latent_shape)))
    # Take the KL divergence between q(z|x) and the standard multivariate Gaussian
    # for each image in the batch. KL loss is of the shape (batch_size, 1).
    kl_loss = -0.5 * K.sum(1 + z_log_var_flat - K.square(z_mean_flat) - K.exp(z_log_var_flat), axis=-1)

    # return the mean weighted sum of reconstruction loss and KL divergence across the
    return reconstruction_loss + beta * kl_loss


'''
Initialisation
'''
# constants
batch_size = 128
epochs = 50
filters = 8
latent_filters = 4
kernal_size = (3, 3)
pool_size = (2, 2)
beta = 1.0
loss_function = 'vae_loss'
optimizer = 'rmsprop'

# initialisers
weight_seed = None
kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=weight_seed)
bias_initializer = initializers.TruncatedNormal(mean=1.0, stddev=0.5, seed=weight_seed)


'''
Define filename
'''
# define name of run
name = 'cvae'

# builder hyperparameter dictionary
hp_dictionary = {
	'batch_size': batch_size,
	'epochs': epochs,
	'beta': beta,
	'loss': loss_function,
	'optimizer': optimizer
}

# define log directory
log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'


'''
Load data
'''
# import dataset
from keras.datasets import mnist
(X_train, _), (X_test, _) = mnist.load_data()
# X_train = X_train[::20]
# X_test = X_test[::20]

# reshape into (num_samples, num_channels, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# record input shape
input_shape = X_train.shape[1:]

# cast pixel values to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise pixel values
X_train /= 255.0
X_test /= 255.0

# print data information
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


'''
Encoder
'''
# define input with 'channels_first'
input_encoder = Input(shape=input_shape, name='encoder_input')

# 'kernal_size' convolution with 'filters' output filters, stride 1x1 and 'valid' border_mode
conv2D_1 = Conv2D(filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
max_pooling_1 = MaxPooling2D(pool_size, name='encoder_max_pooling_1')(conv2D_1)

# separate dense layers for mu and log(sigma), both of size latent_dim
z_mean = Conv2D(latent_filters, kernal_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(max_pooling_1)
z_log_var = Conv2D(latent_filters, kernal_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(max_pooling_1)

# sample from normal with z_mean and z_log_var
z = Lambda(sampling, name='latent_space')([z_mean, z_log_var])


'''
Decoder
'''
# define input with 'channels_first'
encoder_out_shape = tuple(z.get_shape().as_list())
input_decoder = Input(shape=(encoder_out_shape[1], encoder_out_shape[2], encoder_out_shape[3]), name='decoder_input')

# transposed convolution and up sampling
conv2DT_1 = Conv2DTranspose(filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')(input_decoder)
up_sampling_1 = UpSampling2D(pool_size, name='decoder_up_sampling_1')(conv2DT_1)

# transposed convolution
conv2DT_2 = Conv2DTranspose(1, kernal_size, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')(up_sampling_1)

# define decoded image to be image in last layer
decoded_img = conv2DT_2


'''
Train model
'''
# define and save models
encoder = Model(input_encoder, z)
decoder = Model(input_decoder, decoded_img)
cvae = Model(input_encoder, decoder(encoder(input_encoder)))

# print model summary
encoder.summary()
decoder.summary()
cvae.summary()

# compile and train
cvae.compile(loss=vae_loss, optimizer=optimizer)

# define callbacks
tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False)
checkpointer = keras.callbacks.ModelCheckpoint(filepath=log_dir + 'weights.{epoch:03d}-{val_loss:.4f}.hdf5', verbose=1, monitor='val_loss', mode='auto', period=1, save_best_only=True)
callbacks = [tensorboard, checkpointer]

# fit model and record in TensorBoard
cvae.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, callbacks=callbacks)


'''
Save model architectures and weights of encoder/decoder
'''
# model architectures
model_json = cvae.to_json()
with open(log_dir + 'model.json', 'w') as json_file:
	json_file.write(model_json)

encoder_json = encoder.to_json()
with open(log_dir + 'encoder.json', 'w') as json_file:
	json_file.write(encoder_json)

decoder_json = decoder.to_json()
with open(log_dir + 'decoder.json', 'w') as json_file:
	json_file.write(decoder_json)

# weights of encoder and decoder
encoder.save_weights(log_dir + "encoder_weights.hdf5")
decoder.save_weights(log_dir + "decoder_weights.hdf5")
