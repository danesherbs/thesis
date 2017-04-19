from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import losses
from keras import optimizers
from keras import initializers
from keras.callbacks import EarlyStopping


'''
Define parameters
'''
weight_seed = 123
batch_size = 64
epochs = 5
filters = 16
intermediate_filters = 8
latent_filters = 2
kernal_size = (3, 3)
pool_size = (2, 2)


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

# normalise pixel values between [0, 1]
X_train /= 255.0
X_test /= 255.0

# print data information
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


'''
Define model
'''
# weight initaliser
kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=weight_seed)
bias_initializer = initializers.TruncatedNormal(mean=1.0, stddev=0.5, seed=weight_seed)

# define input with 'channels_first'
input_img = Input(shape=input_shape)

# encoder
x = Conv2D(filters, kernal_size, activation='relu', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_img)
x = MaxPooling2D(pool_size, padding='same', name='encoder_max_pooling_1')(x)
x = Conv2D(intermediate_filters, kernal_size, activation='relu', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
x = MaxPooling2D(pool_size, padding='same', name='encoder_max_pooling_2')(x)
x = Conv2D(intermediate_filters, kernal_size, activation='relu', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')(x)
x = MaxPooling2D(pool_size, padding='same', name='encoder_max_pooling_3')(x)

# decoder
x = Conv2D(intermediate_filters, kernal_size, activation='relu', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2D_1')(x)
x = UpSampling2D(pool_size, name='decoder_up_sampling_1')(x)
x = Conv2D(intermediate_filters, kernal_size, activation='relu', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2D_2')(x)
x = UpSampling2D(pool_size, name='decoder_up_sampling_2')(x)
x = Conv2D(filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2D_3')(x)
x = UpSampling2D(pool_size, name='decoder_up_sampling_3')(x)
x = Conv2D(1, kernal_size, activation='sigmoid', padding='same', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2D_4')(x)

# define decoded image to be image in last layer
decoded_img = x


'''
Train model
'''
# define and save models
cae = Model(input_img, decoded_img)

# print model summary
cae.summary()

# compile and train
cae.compile(loss=losses.binary_crossentropy, optimizer='adadelta')

from keras.callbacks import TensorBoard
cae.fit(X_train, X_train, validation_data=(X_test, X_test), shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
