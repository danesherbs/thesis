from vae import VAE
import numpy as np
import utils
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model



class FreyVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=2, pre_latent_size=64, latent_size=2):
        # initialise FreyVAE specific variables
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
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        # x = Dense(self.pre_latent_size, activation='relu', name='encoder_dense_1')(x)

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
        # x = Dense(self.pre_latent_size, activation='relu')(input_decoder)
        x = Dense(np.prod(before_flatten_shape[1:]), activation='relu')(input_decoder)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_2')(x)
        decoded_img = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)

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
For convolutional latent space experiment
'''
class FreyDenseLatentVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=2, pre_latent_size=64, latent_size=2):
        # initialise FreyVAE specific variables
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
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = Conv2D(2*self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')(x)
        before_flatten_shape = tuple(x.get_shape().as_list())
        x = Flatten()(x)
        # x = Dense(self.pre_latent_size, activation='relu', name='encoder_dense_1')(x)

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
        # x = Dense(self.pre_latent_size, activation='relu')(input_decoder)
        x = Dense(np.prod(before_flatten_shape[1:]), activation='relu')(input_decoder)
        x = Reshape(before_flatten_shape[1:])(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_2')(x)
        decoded_img = Conv2DTranspose(1, self.kernel_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2DT_3')(x)

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
For convolutional latent space experiment
'''
class FreyConvolutionalLatentVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=32, kernel_size=2, latent_channels=1, pool_size=2):
        # initialise FreyVAE specific variables
        self.filters = filters
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.pool_size = pool_size
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
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = Conv2D(self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        x = MaxPooling2D(self.pool_size, name='encoder_max_pooling_1')(x)
        x = Conv2D(2*self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')(x)
        x = MaxPooling2D(self.pool_size, name='encoder_max_pooling_2')(x)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_channels, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(x)
        z_log_var = Conv2D(self.latent_channels, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(x)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='encoder_z')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # take encoder output shape
        encoder_out_shape = tuple(z.get_shape().as_list())
        # define rest of model
        input_decoder = Input(shape=encoder_out_shape[1:], name='decoder_input')
        x = Conv2DTranspose(2*self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')(input_decoder)
        x = UpSampling2D(self.pool_size, name='decoder_up_sampling_1')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')(x)
        x = UpSampling2D(self.pool_size, name='decoder_up_sampling_2')(x)
        x = Conv2DTranspose(self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_3')(x)
        decoded_img = Conv2DTranspose(1, self.kernel_size, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_4')(x)

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
Main function
'''
if __name__ == '__main__':
    
    # inputs
    input_shape = (1, 28, 20)
    epochs = 10
    batch_size = 1
    beta = 2.0
    filters = 32
    kernel_size = 2
    pre_latent_size = 64
    latent_size = 8
    
    # define filename
    name = 'cvae_frey'

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
    vae = FreyVAE(input_shape, 
                log_dir,
                filters=filters,
                kernel_size=kernel_size,
                pre_latent_size=pre_latent_size,
                latent_size=latent_size)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)

    # print summaries
    vae.print_model_summaries()
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

    # fit VAE
    steps_per_epoch = int(train_size / batch_size)
    validation_steps = int(test_size / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)