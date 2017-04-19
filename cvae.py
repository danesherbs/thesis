from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import losses
from keras import optimizers
from keras import initializers
from keras.callbacks import EarlyStopping


'''
Required functions for latent space and training
'''
def sampling(args):
    z_mean, z_log_sigma = args  # unpack args
    assert z_mean.shape[1:] == z_log_sigma.shape[1:]  # need mean and std for each point
    output_shape = z_mean.shape[1:]  # same shape as mean and log_var
    epsilon = K.random_normal(shape=output_shape, mean=0.0, stddev=1.0)  # sample from standard normal
    return z_mean + K.exp(z_log_sigma) * epsilon  # reparameterization trick

def vae_loss(x, x_decoded_mean):
    xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + beta*kl_loss


'''
Define parameters
'''
weight_seed = None
batch_size = 128
epochs = 1
filters = 8
latent_filters = 4
kernal_size = (3, 3)
pool_size = (2, 2)
beta = 1.0


'''
Load data
'''
# import dataset
from keras.datasets import mnist
(X_train, _), (X_test, _) = mnist.load_data()
# X_train, X_test, _, _ = utils.load_data(down_sample=True)

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
Define model
'''
# weight initaliser
kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.5, seed=weight_seed)
bias_initializer = initializers.TruncatedNormal(mean=1.0, stddev=0.5, seed=weight_seed)

# define input with 'channels_first'
input_img = Input(shape=input_shape)

# 'kernal_size' convolution with 'filters' output filters, stride 1x1 and 'valid' border_mode
conv2D_1 = Conv2D(filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_img)
max_pooling_1 = MaxPooling2D(pool_size, name='encoder_max_pooling_1')(conv2D_1)

# separate dense layers for mu and log(sigma), both of size latent_dim
z_mean = Conv2D(latent_filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(max_pooling_1)
z_log_sigma = Conv2D(latent_filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_sigma')(max_pooling_1)

# sample from normal with z_mean and z_log_sigma
z = Lambda(sampling, name='latent_space')([z_mean, z_log_sigma])

# 'kernal_size' transposed convolution with 'filters' output filters and stride 1x1
conv2DT_1 = Conv2DTranspose(filters, kernal_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')(z)
up_sampling_1 = UpSampling2D(pool_size, name='decoder_up_sampling_1')(conv2DT_1)
conv2DT_2 = Conv2DTranspose(1, kernal_size, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')(up_sampling_1)

# define decoded image to be image in last layer
decoded_img = conv2DT_2


'''
Train model
'''
# define and save models
vae = Model(input_img, decoded_img)

# print model summary
vae.summary()

# compile and train
vae.compile(loss=losses.binary_crossentropy, optimizer='adadelta')
vae.fit(X_train, X_train, validation_data=(X_test, X_test), batch_size=batch_size, epochs=epochs)
