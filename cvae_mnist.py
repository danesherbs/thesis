from vae import VAE
from autoencoder import Autoencoder
import numpy as np
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model


class CholletVAE(VAE):

    def __init__(self,
                input_shape,
                log_dir,
                filters=32,
                pool_size=(2,2),
                kernel_size=6,
                pre_latent_size=128,
                latent_size=32,
                beta=1.0):
        self.input_shape = input_shape
        self.filters = filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')

        conv_1 = Conv2D(1,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu')(input_encoder)
        conv_2 = Conv2D(self.filters,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu')(conv_1)
        conv_3 = Conv2D(self.filters,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu',
                        strides=1)(conv_2)
        conv_4 = Conv2D(self.filters,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu')(conv_3)
        conv4_out_shape = tuple(conv_4.get_shape().as_list())
        flat = Flatten()(conv_4)

        hidden = Dense(self.pre_latent_size, activation='relu')(flat)

        z_mean = Dense(self.latent_size)(hidden)
        z_log_var = Dense(self.latent_size)(hidden)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='latent_space')([z_mean, z_log_var])
        encoder_out_shape = tuple(z.get_shape().as_list())

        # we instantiate these layers separately so as to reuse them later
        input_decoder = Input(shape=encoder_out_shape[1:])
        decoder_hid = Dense(self.pre_latent_size, activation='relu')
        decoder_upsample = Dense(np.prod(conv4_out_shape[1:]), activation='relu')
        output_shape = (self.filters, conv4_out_shape[2], conv4_out_shape[3])

        decoder_reshape = Reshape(output_shape)
        decoder_deconv_1 = Conv2DTranspose(self.filters,
                                           kernel_size=self.kernel_size,
                                           padding='same',
                                           activation='relu')
        decoder_deconv_2 = Conv2DTranspose(self.filters,
                                           kernel_size=self.kernel_size,
                                           padding='same',
                                           activation='relu')

        decoder_deconv_3_upsamp = Conv2DTranspose(self.filters,
                                                  kernel_size=self.kernel_size,
                                                  padding='same',
                                                  activation='relu')
        decoder_mean_squash = Conv2D(1,
                                     kernel_size=self.kernel_size,
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




class ShallowDenseMNIST(VAE):

    def __init__(self,
                input_shape,
                log_dir,
                filters=32,
                pool_size=(2,2),
                kernel_size=6,
                pre_latent_size=128,
                latent_size=32,
                beta=1.0):
        self.input_shape = input_shape
        self.filters = filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')

        conv_1 = Conv2D(1,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu')(input_encoder)
        conv_2 = Conv2D(self.filters,
                        kernel_size=self.kernel_size,
                        padding='same',
                        activation='relu')(conv_1)
        conv2_out_shape = tuple(conv_2.get_shape().as_list())
        flat = Flatten()(conv_2)

        hidden = Dense(self.pre_latent_size, activation='relu')(flat)

        z_mean = Dense(self.latent_size)(hidden)
        z_log_var = Dense(self.latent_size)(hidden)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='latent_space')([z_mean, z_log_var])
        encoder_out_shape = tuple(z.get_shape().as_list())

        # we instantiate these layers separately so as to reuse them later
        input_decoder = Input(shape=encoder_out_shape[1:])
        decoder_hid = Dense(self.pre_latent_size, activation='relu')
        decoder_upsample = Dense(np.prod(conv2_out_shape[1:]), activation='relu')
        output_shape = (self.filters, conv2_out_shape[2], conv2_out_shape[3])

        decoder_reshape = Reshape(output_shape)
        decoder_deconv_1 = Conv2DTranspose(self.filters,
                                           kernel_size=self.kernel_size,
                                           padding='same',
                                           activation='relu')
        decoder_mean_squash = Conv2D(1,
                                     kernel_size=self.kernel_size,
                                     padding='same',
                                     activation='sigmoid')

        hid_decoded = decoder_hid(input_decoder)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        x_decoded_mean = decoder_mean_squash(deconv_1_decoded)

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
        latent_size = 32

        '''
        Encoder
        '''
        input_encoder = Input(shape=(self.input_shape), name='encoder_input')

        x = Flatten(name='encoder_flatten_1')(input_encoder)
        z = Dense(latent_size, activation='relu', name='encoder_dense_1')(x)

        '''
        Decoder
        '''
        input_decoder = Input(shape=(latent_size,), name='decoder_input')

        x = Dense(input_size, activation='sigmoid', name='decoder_dense_1')(input_decoder)
        x = Reshape(self.input_shape)(x)

        '''
        For parent class
        '''
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, x)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))



class ConvolutionalAutoencoder(Autoencoder):
    def __init__(self, input_shape, log_dir):
        self.input_shape = input_shape
        self.log_dir = log_dir
        Autoencoder.__init__(self, input_shape, log_dir)

    def set_model(self):
        '''
        Constants
        '''
        input_size = np.prod(self.input_shape)
        intermediate_filters = 32
        latent_filters = 4
        kernel_size = 2

        '''
        Encoder
        '''
        input_encoder = Input(shape=(self.input_shape), name='encoder_input')

        x = Conv2D(intermediate_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu')(input_encoder)
        x = MaxPooling2D(pool_size=(2, 2),
                strides=None,
                padding='valid')(x)
        x = Conv2D(latent_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu')(x)
        z = MaxPooling2D(pool_size=(2, 2),
                strides=None,
                padding='valid')(x)

        '''
        Decoder
        '''
        input_decoder = Input(shape=(latent_filters, 7, 7), name='decoder_input')

        x = UpSampling2D(size=(2, 2))(input_decoder)
        x = Conv2DTranspose(intermediate_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)        
        x = Conv2DTranspose(1,
                kernel_size=kernel_size,
                padding='same',
                activation='sigmoid')(x)

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