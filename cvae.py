from keras.layers import Input, Dense, Lambda, Convolution2D, Deconvolution2D
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras import objectives
from keras.callbacks import EarlyStopping
import utils
import cPickle


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
batch_size = 64
nb_epoch = 50
nb_filters = 8
latent_features = nb_filters / 2
kernal_size = (7, 7)
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

# 'kernal_size' convolution with 'nb_filters' output filters and stride 1x1
conv = Convolution2D(nb_filters, kernal_size[0], kernal_size[1], border_mode='valid', input_shape=input_shape)(input_img)

# separate dense layers for mu and log(sigma), both of size latent_dim
z_mean = Convolution2D(latent_features, kernal_size[0], kernal_size[1], border_mode='valid')(conv)
z_log_var = Convolution2D(latent_features, kernal_size[0], kernal_size[1], border_mode='valid')(conv)

# sample from normal with z_mean and z_log_var
z_output_shape = output_shape=(latent_features, 30, 52)
z = Lambda(sampling, output_shape=z_output_shape, name='latent_space')([z_mean, z_log_var])

# 'kernal_size' transposed convolution with 'nb_filters' output filters and stride 1x1
deconv = Deconvolution2D(nb_filters, kernal_size[0], kernal_size[1], output_shape=(None, 8, 36, 58), border_mode='valid', input_shape=z_output_shape)(z)

# 'kernal_size' transposed convolution with 1 output filter and stride 1x1 on a 32 22x22 images
decoded_img = Deconvolution2D(1, kernal_size[0], kernal_size[1], output_shape=(None, 1, 42, 64), border_mode='valid', input_shape=(8, 36, 58))(deconv)


'''
Train model
'''

# define and save models
vae = Model(input_img, decoded_img)
encoder = Model(input=vae.input, output=vae.get_layer('latent_space').output)
saved_models = 'saved_models/'
description = '_epoch=' + str(nb_epoch) + '_beta=' + str(int(beta)) + '_latent=' + str(latent_features)
vae.save(saved_models + 'cvae' + description + '.h5')
encoder.save(saved_models + 'encoder' + description + '.h5')

# print model summary
vae.summary()

# compile and train
vae.compile(loss=vae_loss, optimizer='adam', metrics=['binary_crossentropy'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
history = vae.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[earlyStopping])

# save learnt weights
vae.save_weights(saved_models + 'cvae' + description + '_weights' + '.h5')
encoder.save_weights(saved_models + 'encoder' + description + '_weights' + '.h5')

# save history
cPickle.dump(saved_models + 'history' + description + '.pickle', 'w')