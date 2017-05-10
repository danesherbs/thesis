from vae import VAE
import numpy as np
import utils
from keras import initializers
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model



class ConvolutionalLatentVAE(VAE):

    def __init__(self, input_shape, log_dir, filters=8, latent_filters=4, kernel_size=3, pool_size=2):
        # initialise HigginsVAE specific variables
        self.filters = filters
        self.latent_filters = latent_filters
        self.kernel_size = kernel_size
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
        input_encoder = Input(shape=input_shape, name='encoder_input')
        conv2D_1 = Conv2D(self.filters, self.kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')(input_encoder)
        max_pooling_1 = MaxPooling2D(self.pool_size, name='encoder_max_pooling_1')(conv2D_1)

        # separate dense layers for mu and log(sigma), both of size latent_dim
        z_mean = Conv2D(self.latent_filters, self.kernel_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_mean')(max_pooling_1)
        z_log_var = Conv2D(self.latent_filters, self.kernel_size, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_z_log_var')(max_pooling_1)

        # sample from normal with z_mean and z_log_var
        z = Lambda(self.sampling, name='latent_space')([z_mean, z_log_var])

        '''
        Decoder
        '''
        # define input with 'channels_first'
        encoder_out_shape = tuple(z.get_shape().as_list())
        input_decoder = Input(shape=(encoder_out_shape[1], encoder_out_shape[2], encoder_out_shape[3]), name='decoder_input')

        # transposed convolution and up sampling
        conv2DT_1 = Conv2DTranspose(self.filters, kernel_size, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')(input_decoder)
        up_sampling_1 = UpSampling2D(self.pool_size, name='decoder_up_sampling_1')(conv2DT_1)

        # transposed convolution
        conv2DT_2 = Conv2DTranspose(1, self.kernel_size, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')(up_sampling_1)

        # define decoded image to be image in last layer
        decoded_img = conv2DT_2

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
    input_shape = (1, 28, 28)
    epochs = 1
    batch_size = 1
    filters = 32
    latent_filters = 1
    kernel_size = 3
    pool_size = 2
    beta = 1.0
    
    # define filename
    name = 'cvae_convolutional_latent'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'loss': 'vae_loss',
        'optimizer': 'adam',
        'beta': beta
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = ConvolutionalLatentVAE(input_shape, 
                log_dir,
                filters=filters,
                latent_filters=latent_filters,
                kernel_size=kernel_size,
                pool_size=pool_size)
    
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)
    
    # save architecure
    vae.save_model_architecture()
    
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
    
    # save encoder and decoder weights
    vae.save_weights()