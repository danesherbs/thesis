import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras import initializers
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.objectives import binary_crossentropy
from abc import ABCMeta, abstractmethod


class VAE(metaclass=ABCMeta):
    '''
    Class to handle building and training VAE models.
    '''

    def __init__(self, input_shape, optimizer):
        '''
        Parameters
        ----------
        input_shape : (num_channels, num_rows, num_cols)
        '''
        self.optimizer = optimizer
        self.input_shape = input_shape

    def fit(self, x_train, epochs=1, batch_size=1, val_split=.1,
            learning_rate=1e-3, reset_model=True):
        """
        Training model
        """
        self.batch_size = batch_size
        self.epochs = epochs
        print('Setting up model...', end=' ')
        self.set_model()
        print('done.')
        self.print_model_summaries()

        if x_train.shape[0] % batch_size != 0:
            raise(RuntimeError("Training data shape {} is not divisible by batch size {}".format(x_train.shape[0], self.batch_size)))

        # compile parameters
        self.model.compile(optimizer=self.optimizer, loss=self.vae_loss)

        self.model.fit(x_train, x_train,
                        nb_epoch=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=val_split)

    def fit_generator(self, train_generator, test_generator, steps_per_epoch, validation_steps, epochs=10):
        self.model.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=int(len(X_train)/batch_size),
                   validation_data=test_generator,
                   validation_steps=int(len(X_test)/batch_size),
                   callbacks=callbacks)

    @abstractmethod
    def set_model(self):
        '''
        Must define the following for training:
            - self.encoder
            - self.decoder
            - self.model
        and the following for the loss function:
            - self.z_mean
            - self.z_log_var
            - self.z
        '''
        pass

    def sampling(self, args):
        '''
        Sampling function used in encoder
        '''
        # unpack arguments
        z_mean, z_log_var = args
        # need mean and std for each point
        assert z_mean.shape[1:] == z_log_var.shape[1:]
        # output shape is same as mean and log_var
        output_shape = z_mean.shape[1:]
        # sample from standard normal
        epsilon = K.random_normal(shape=output_shape, mean=0.0, stddev=1.0)
        # reparameterization trick
        return z_mean + K.exp(z_log_var) * epsilon

    def vae_loss(self, y_true, y_pred):
        '''
        Variational autoencoder loss function
        '''
        beta = 1.0
        y_true = K.reshape(y_true, (-1, np.prod(self.input_shape)))
        y_pred = K.reshape(y_pred, (-1, np.prod(self.input_shape)))
        reconstruction_loss = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        latent_shape = self.z.get_shape().as_list()[1:]
        z_mean_flat = K.reshape(self.z_mean, (-1, np.prod(latent_shape)))
        z_log_var_flat = K.reshape(self.z_log_var, (-1, np.prod(latent_shape)))
        kl_loss = -0.5 * K.sum(1 + z_log_var_flat - K.square(z_mean_flat) - K.exp(z_log_var_flat), axis=-1)
        return K.mean(reconstruction_loss + beta * kl_loss)

    def print_model_summaries(self):
        '''
        Prints model summaries of encoder, decoder and entire model
        '''
        self.__print_model_summary(self.encoder)
        self.__print_model_summary(self.decoder)
        self.__print_model_summary(self.model)

    def __print_model_summary(self, model):
        '''
        Helper for print_model_summaries
        '''
        if model is not None:
            model.summary()


class myVAE(VAE):

    def __init__(self, input_shape, optimizer):
        # call parent constructor
        VAE.__init__(self, input_shape, optimizer)

    def set_model(self):
        # initalise important parameters
        kernal_size = (6, 6)
        filters = 32
        hidden_dim = 256
        latent_dim = 32

        # initialisers
        weight_seed = None
        kernel_initializer = initializers.glorot_uniform(seed = weight_seed)
        bias_initializer = initializers.glorot_uniform(seed = weight_seed)

        # Encoder layers
        encoder_conv2D_1 = Conv2D(self.filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')
        encoder_conv2D_2 = Conv2D(2*self.filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')
        encoder_conv2D_3 = Conv2D(2*self.filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')
        encoder_flatten_1 = Flatten()
        encoder_dense_1 = Dense(self.hidden_dim, activation='relu', name='encoder_dense_1')
        encoder_z_mean = Dense(self.latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_mean')
        encoder_z_log_var = Dense(self.latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_log_var')
        encoder_z = Lambda(self.sampling, name='z')

        # Connect encoder layers
        input_encoder = Input(shape=self.input_shape, name='encoder_input')
        x = encoder_conv2D_1(input_encoder)
        x = encoder_conv2D_2(x)
        x = encoder_conv2D_3(x)
        x_flattened = encoder_flatten_1(x)
        x = encoder_dense_1(x_flattened)
        z_mean = encoder_z_mean(x)
        z_log_var = encoder_z_log_var(x)
        z = encoder_z([z_mean, z_log_var])

        # Define encoder model
        self.encoder = Model(input_encoder, z)

        # Define useful shapes for decoder (for symmetry)
        before_flatten_shape = tuple(x_flattened.get_shape().as_list())

        # Decoder layers
        decoder_dense_1 = Dense(self.hidden_dim, activation='relu', name="decoder_dense_1")
        decoder_dense_2 = Dense(np.prod((64, 7, 7)), activation='relu', name="decoder_dense_2")
        decoder_reshape_1 = Reshape((64, 7, 7), name="decoder_reshape_1")
        decoder_conv2DT_1 = Conv2DTranspose(2*self.filters, kernal_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')
        decoder_conv2DT_2 = Conv2DTranspose(self.filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')
        decoder_conv2DT_3 = Conv2DTranspose(1, kernal_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_3')

        # Connect decoder layers
        input_decoder = Input(shape=(self.latent_dim,), name='decoder_input')
        x = decoder_dense_1(input_decoder)
        x = decoder_dense_2(x)
        x = decoder_reshape_1(x)
        x = decoder_conv2DT_1(x)
        x = decoder_conv2DT_2(x)
        x = decoder_conv2DT_3(x)

        # Define decoder model
        self.decoder = Model(input_decoder, x)

        # Define entire model
        self.model = Model(input_encoder, self.decoder(self.encoder(input_encoder)))

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z


if __name__ == '__main__':
    input_shape = (1, 84, 84)
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae = myVAE(input_shape, optimizer)
    vae.set_model()