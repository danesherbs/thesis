from vae import VAE
from cae import CAE
import numpy as np
import utils
from keras import backend as K
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf



class HigginsVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=2, pre_latent_size=64, latent_size=2):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=input_shape, name='encoder_input')
        x = Conv2D(filters, kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        # x = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(x)
        x = Conv2D(2*filters, kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        # x = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(x)
        x = Conv2D(2*filters, kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(pre_latent_size, activation='relu', name='encoder_dense_1')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(latent_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Dense(latent_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(pre_latent_size, activation='relu')(input_decoder)
        x = Dense(np.prod(before_flatten_shape[1:]), activation='relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*filters, kernel_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(x)
        x = Conv2DTranspose(filters, kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_2')(x)
        decoded_img = Conv2DTranspose(1, kernel_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


'''
For optimal Pong network experiment: experiment_optimal_network_convolutional_latent_pong.py
'''
class PongEntangledConvolutionalLatentVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=None, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        if latent_filters is None:
            self.latent_filters = 2*self.filters
        else:
            self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



'''
For optimal Pong network experiment: experiment_optimal_network_convolutional_latent_pong.py
'''
class PongEntangledConvolutionalLatentNoBatchNormVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=None, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        if latent_filters is None:
            self.latent_filters = 2*self.filters
        else:
            self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPong(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, strides=(2,2), pre_latent_size=64, latent_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPongNoBatchNorm(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, pre_latent_size=64, latent_size=2, strides=(2,2), beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=self.strides,
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=self.strides,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPongBatchNormBeforeLatent(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, pre_latent_size=64, latent_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPongBatchNormAfterLatent(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, pre_latent_size=64, latent_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPongBatchNormBeforeAndAfterLatent(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, pre_latent_size=64, latent_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


'''
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPongBatchNormEverywhereLatent(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=6, pre_latent_size=64, latent_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.pre_latent_size = pre_latent_size
        self.latent_size = latent_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size,
                activation=None,
                name='encoder_dense_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size,
                activation=None,
                name='decoder_dense_1')(input_decoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]),
                activation=None,
                name='decoder_dense_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters,
                self.kernel_size,
                strides=(2, 2),
                padding='valid',
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,
                self.kernel_size,
                strides=(2, 2),
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='decoder_conv2DT_3')(x)
        x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


'''
For optimal Pong network experiment: experiment_convolutional_image_latent_space_invaders_different_betas.py
'''
class ConvolutionalLatentNoBatchNormVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=None, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        if latent_filters is None:
            self.latent_filters = 2*self.filters
        else:
            self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        # x = BatchNormalization()(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z



'''
For experiment_different_loss_functions.py
'''
class ConvolutionalLatentShallowAverageFilterVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=8, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z_layer = Lambda(self.sampling, name='encoder_z')
        z = z_layer([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), padding='same', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z_layer = z_layer
        self.z = z

    def kl_loss(self, y_true, y_pred):
        '''
        Override KL-loss function to be KL-loss over average activations in each filter
        '''
        z_mean_average_filters = K.mean(self.z_mean, axis=(2, 3))  # (batch_size, filters)
        z_log_var_average_filters = K.mean(self.z_log_var, axis=(2, 3))  # (batch_size, filters)
        loss = -0.5 * K.sum(1 + z_log_var_average_filters - K.square(z_mean_average_filters) - K.exp(z_log_var_average_filters), axis=1)  # (batch_size,)
        return K.mean(loss)  # float



'''
For experiment_different_loss_functions.py
'''
class ConvolutionalLatentShallowWeightedAverageFilterVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=8, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z_layer = Lambda(self.sampling, name='encoder_z')
        z = z_layer([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), padding='same', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z_layer = z_layer
        self.z = z

    def kl_loss(self, y_true, y_pred):
        '''
        Override KL-loss function to be KL-loss over average activations in each filter
        '''
        weight = np.prod(self.z_mean.shape[2:])  # float = width * height
        z_mean_average_filters = K.mean(self.z_mean, axis=(2, 3))  # (batch_size, filters)
        z_log_var_average_filters = K.mean(self.z_log_var, axis=(2, 3))  # (batch_size, filters)
        loss = -0.5 * K.sum(1 + z_log_var_flat - K.square(z_mean_average_filters) - K.exp(z_log_var_average_filters), axis=1)  # (batch_size,)
        return K.mean(loss)  # float


'''
For experiment_different_loss_functions.py
'''
class ConvolutionalLatentShallowOrthoRegCAE(CAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=8, kernel_size=2, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        # call parent constructor
        CAE.__init__(self, input_shape, log_dir)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        z_layer = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')
        z = z_layer(x)

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_2')(x)
        decoded_img = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), padding='same', activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For OrthoReg
        self.z_layer = z_layer

    # def cae_loss(self, y_pred, y_true):
    #     return self.reconstruction_loss(y_true, y_pred) + self.ortho_reg(self.z_layer.get_weights())


# import keras
# from keras.regularizers import Regularizer

# class OrthoReg(Regularizer):
#     """
#     Regularizer convolutional weights
#     """

#     """Regularizer for L1 and L2 regularization.

#     # Arguments
#         l1: Float; L1 regularization factor.
#         l2: Float; L2 regularization factor.
#     """

#     def __init__(self, l1=0., l2=0.):
#         self.l1 = K.cast_to_floatx(l1)
#         self.l2 = K.cast_to_floatx(l2)

#     def __call__(self, x):
#         regularization = 0.
#         if self.l1:
#             regularization += K.sum(self.l1 * K.abs(x))
#         if self.l2:
#             regularization += K.sum(self.l2 * K.square(x))
#         return regularization

#     def get_config(self):
#         return {'l1': float(self.l1),
#                 'l2': float(self.l2)}

    # def __call__(self, w):
    #     # regularise w to be orthogonal
    #     # L = sum_{w,w'} log(1 + exp( l * (cos(w,w') - 1) ))  (see eq. 7)
    #     # set  l = 10

    #     w_new = tf.reshape(w, [-1, self.latent_filters])

    #     # normalized w
    #     norm = tf.sqrt(tf.reduce_sum(tf.square(w_new), 0, keep_dims=True))
    #     w_new = w_new / ( norm + 1e-10 )

    #     # compute product
    #     prod = tf.matmul( K.transpose(w_new), w_new )
       
    #     l = 10
    #     L = tf.log( 1 + tf.exp(l * (prod - 1) ))
    #     diag_mat = tf.diag(tf.diag_part(L))
    #     L = L - diag_mat
    #     Loss = tf.reduce_sum( tf.reduce_sum(L) )

    #     return Loss

    # def get_config(self):
    #         return {'name': self.__class__.__name__}


'''
For experiment_different_filter_sizes_and_strides.py
'''
class ConvolutionalLatentAverageFilterShallowVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=8, kernel_size=2, img_channels=1, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        self.img_channels = img_channels
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = Conv2DTranspose(self.img_channels, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z

    def kl_loss(self, y_true, y_pred):
        '''
        Override KL-loss function to be KL-loss over average activations in each filter
        '''
        z_mean_average_filters = K.mean(self.z_mean, axis=(2, 3))  # (batch_size, filters)
        z_log_var_average_filters = K.mean(self.z_log_var, axis=(2, 3))  # (batch_size, filters)
        loss = -0.5 * K.sum(1 + z_log_var_average_filters - K.square(z_mean_average_filters) - K.exp(z_log_var_average_filters), axis=1)  # (batch_size,)
        return K.mean(loss)  # float



class WeightedAverageFilters(VAE):

    def __init__(self, input_shape, log_dir, filters=32, latent_filters=8, kernel_size=2, img_channels=1, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
        self.img_channels = img_channels
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = Conv2DTranspose(self.img_channels, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z

    def kl_loss(self, y_true, y_pred):
        '''
        Override KL-loss function to be KL-loss over average activations in each filter
        '''
        z_mean_average_filters = K.mean(self.z_mean, axis=(2, 3))  # (batch_size, filters)
        z_log_var_average_filters = K.mean(self.z_log_var, axis=(2, 3))  # (batch_size, filters)
        loss = -0.5 * K.sum(1 + z_log_var_average_filters - K.square(z_mean_average_filters) - K.exp(z_log_var_average_filters), axis=1)  # (batch_size,)
        return K.sum(loss)  # float





class LatentImage(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=2, img_channels=1, beta=1.0):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = 1
        self.kernel_size = kernel_size
        self.img_channels = img_channels
        # call parent constructor
        VAE.__init__(self, input_shape, log_dir, beta=beta)

    def set_model(self):
        '''
        Initialisers
        '''
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        '''
        Encoder
        '''
        # define input with 'channels_first'
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())

        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(input_decoder)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), padding='same', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
        x = Conv2DTranspose(self.img_channels, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_4')(x)
        decoded_img = Activation('sigmoid')(x)

        '''
        Necessary definitions
        '''
        # For parent fitting function
        self.encoder = Model(input_encoder, z)
        self.decoder = Model(input_decoder, decoded_img)
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))
        # For parent loss function
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z






'''
Main
'''
if __name__ == '__main__':
    # inputs
    input_shape = (1, 84, 84)
    epochs = 10
    batch_size = 1
    beta = 1.0
    filters = 32
    kernel_size = 6
    pre_latent_size = 512
    latent_size = 16
    
    # define filename
    name = 'cvae_atar_higgins'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'pre_latent_size': pre_latent_size,
        'latent_size': latent_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = HigginsVAE(input_shape, 
                log_dir,
                filters=filters,
                kernel_size=kernel_size,
                pre_latent_size=pre_latent_size,
                latent_size=latent_size)
    
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)
    
    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)
    
    # print summaries
    vae.print_model_summaries()
    
    # fit VAE
    steps_per_epoch = int(train_size / batch_size)
    validation_steps = int(test_size / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)