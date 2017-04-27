import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
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
def vae_loss(x, x_decoded_mean):
    # input image dimensions
    img_rows, img_cols = input_shape[1], input_shape[2]
    # flatten tensors
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    # compute binary crossentropy
    xent_loss = img_rows * img_cols * losses.binary_crossentropy(x, x_decoded_mean)
    # compute KL divergence
    # kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # kl_loss = - 0.5 * K.sum(484 + 2*z_log_var - K.square(z_mean) - K.exp(2*z_log_var), axis=-1)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # return linear combination of losses
    return K.mean(xent_loss + beta*kl_loss)


'''
Initialisation
'''
# constants
batch_size = 128
epochs = 1
filters = 64
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
name = 'cvae_chollet'

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

conv_1 = Conv2D(1,
                kernel_size=(2, 2),
                padding='same', activation='relu')(input_encoder)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=(3, 3),
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=(3, 3),
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(128, activation='relu')(flat)

z_mean = Dense(16)(hidden)
z_log_var = Dense(16)(hidden)

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

# sample from normal with z_mean and z_log_var
z = Lambda(sampling, name='latent_space')([z_mean, z_log_var])


'''
Decoder
'''
# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(128, activation='relu')
decoder_upsample = Dense(filters * 14 * 14, activation='relu')
output_shape = (batch_size, filters, 14, 14)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, (3, 3),
                                   padding='same',
                                   strides=1,
                                   activation='relu')

output_shape = (batch_size, filters, 29, 29)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(1,
                             kernel_size=(2, 2),
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean = decoder_mean_squash(x_decoded_relu)



'''
Train model
'''
# encoder
encoder = Model(input_encoder, z)

# decoder
decoder_input = Input(shape=(16,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
decoder = Model(decoder_input, _x_decoded_mean_squash)

# full model
cvae = Model(input_encoder, x_decoded_mean)

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
