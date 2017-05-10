from keras import backend as K
from keras.objectives import binary_crossentropy
from abc import ABCMeta, abstractmethod


class VAE(metaclass=ABCMeta):
    '''
    Class to handle building and training VAE models.
    '''

    def __init__(self, input_shape, optimizer):
        '''
        input_shape : (num_channels, num_rows, num_cols)
        '''
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.callbacks = None  # TODO: define callbacks
        self.set_model()

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

    def fit(self, X_train, **kwargs):
        '''
        Wrapper for Keras fit method
        '''
        self.set_model()
        self.print_model_summaries()
        self.model.fit(X_train, X_train, kwargs)

    def fit_generator(self, train_generator, test_generator, **kwargs):
        '''
        Wrapper for Keras fit_generator method
        '''
        self.set_model()
        self.print_model_summaries()
        self.model.fit_generator(train_generator, kwargs)

    def compile(self, **kwargs):
        '''
        Compiles Keras model
        '''
        loss = kwargs.get('loss', self.loss)
        optimizer = kwargs.get('optimizer', self.optimizer)
        self.model.compile(loss=loss, optimizer=optimizer)

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