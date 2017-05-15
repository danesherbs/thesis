from vae import VAE
import numpy as np
import utils
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model



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
For experiment_optimal_network_dense_latent_pong.py
'''
class DenseLatentPong(VAE):

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
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        x = Dense(self.pre_latent_size, activation=None, name='encoder_dense_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Dense(self.latent_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Dense(self.latent_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Dense(self.pre_latent_size, activation=None)(input_decoder)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(before_flatten_shape[1:]), activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(2*self.filters, self.kernel_size, strides=(2, 2), padding='valid', activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)
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