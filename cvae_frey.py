from vae import VAE
import numpy as np
import utils
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model



class FreyVAE(VAE):

    def __init__(self, input_shape, log_dir):
        VAE.__init__(self, input_shape, log_dir)

    def set_model(self):
        '''
        Constants
        '''
        filters = 32
        kernel_size = 2
        pre_latent_size = 64
        latent_size = 2

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
Main function
'''
if __name__ == '__main__':
    
    # inputs
    input_shape = (1, 28, 20)
    epochs = 10
    batch_size = 1
    beta = 1.0
    
    # define filename
    name = 'cvae_frey'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'batch_size': batch_size,
        'epochs': epochs,
        'beta': beta,
        'loss': 'vae_loss',
        'optimizer': 'adam',
        'latent_size': 2
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = HigginsVAE(input_shape, log_dir)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
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