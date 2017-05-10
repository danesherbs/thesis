from vae import VAE
import numpy as np
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


class myVAE(VAE):

    def __init__(self, input_shape, optimizer):
        VAE.__init__(self, input_shape, optimizer)

    def set_model(self):
        # constants
        batch_size = 32
        epochs = 1
        filters = 32
        latent_filters = 4
        kernel_size = (6, 6)
        pool_size = (2, 2)
        beta = 1.0

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=input_shape, name='encoder_input')

        conv_1 = Conv2D(1,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu')(input_encoder)
        conv_2 = Conv2D(filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu')(conv_1)
        conv_3 = Conv2D(filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu',
                        strides=1)(conv_2)
        conv_4 = Conv2D(filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu')(conv_3)
        conv4_out_shape = tuple(conv_4.get_shape().as_list())
        flat = Flatten()(conv_4)

        hidden = Dense(128, activation='relu')(flat)

        z_mean = Dense(16)(hidden)
        z_log_var = Dense(16)(hidden)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='latent_space')([z_mean, z_log_var])
        encoder_out_shape = tuple(z.get_shape().as_list())

        # we instantiate these layers separately so as to reuse them later
        input_decoder = Input(shape=encoder_out_shape[1:])
        decoder_hid = Dense(128, activation='relu')
        print(conv4_out_shape)
        decoder_upsample = Dense(np.prod(conv4_out_shape[1:]), activation='relu')
        output_shape = (batch_size, filters, conv4_out_shape[2], conv4_out_shape[3])

        print(output_shape[1:])
        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Conv2DTranspose(filters,
                                           kernel_size=kernel_size,
                                           padding='same',
                                           activation='relu')
        decoder_deconv_2 = Conv2DTranspose(filters,
                                           kernel_size=kernel_size,
                                           padding='same',
                                           activation='relu')

        decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                  kernel_size=kernel_size,
                                                  padding='same',
                                                  activation='relu')
        decoder_mean_squash = Conv2D(1,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     activation='sigmoid')

        hid_decoded = decoder_hid(input_decoder)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean = decoder_mean_squash(x_decoded_relu)

        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, x_decoded_mean)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



if __name__ == '__main__':
    # inputs
    input_shape = (1, 28, 28)
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    # make VAE
    vae = myVAE(input_shape, optimizer)
    # compile VAE
    vae.compile()
    # get dataset
    from keras.datasets import mnist
    (X_train, _), (X_test, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    input_shape = X_train.shape[1:]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train - np.min(X_train)) / np.max(X_train)
    X_test = (X_test - np.min(X_test)) / np.max(X_test)
    X_train = X_train[::100]
    X_test = X_test[::100]
    vae.fit(X_train, validation_data=X_test)