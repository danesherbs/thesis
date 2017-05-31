from vae import VAE
from autoencoder import Autoencoder
import numpy as np
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model


class CholletVAE(VAE):

    def __init__(self, input_shape, log_dir):
        VAE.__init__(self, input_shape, log_dir)

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


class DenseAutoencoder(Autoencoder):
    def __init__(self, input_shape, log_dir):
        self.input_shape = input_shape
        self.log_dir = log_dir
        Autoencoder.__init__(self, input_shape, log_dir)

    def set_model(self):
        '''
        Constants
        '''
        input_size = np.prod(self.input_shape)
        latent_size = 16

        '''
        Encoder
        '''
        input_encoder = Input(shape=(self.input_shape), name='encoder_input')

        x = Flatten(name='encoder_flatten_1')(input_encoder)
        x = Dense(512, activation='relu', name='encoder_dense_1')(x)
        z = Dense(latent_size, activation='relu', name='encoder_dense_3')(x)

        '''
        Decoder
        '''
        input_decoder = Input(shape=(latent_size,), name='decoder_input')

        x = Dense(512, activation='relu', name='decoder_dense_2')(input_decoder)
        x = Dense(input_size, name='decoder_dense_3')(x)
        x = Reshape(self.input_shape)(x)

        '''
        For parent class
        '''
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, x)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))


if __name__ == '__main__':
    
    # inputs
    input_shape = (1, 28, 28)
    epochs = 10
    batch_size = 1
    log_dir = './summaries/test_dir/'

    # make VAE
    vae = CholletVAE(input_shape, log_dir)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)
    
    # get dataset
    import utils
    (X_train, _), (X_test, _) = utils.load_mnist()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    
    # print summaries
    vae.print_model_summaries()
    
    # fit VAE
    steps_per_epoch = int(len(X_train) / batch_size)
    validation_steps = int(len(X_test) / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)