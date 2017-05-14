import pickle
import numpy as np
import os
import keras
from keras.models import load_model
from keras import backend as K
from keras.objectives import binary_crossentropy
from keras.models import model_from_json
from abc import ABCMeta, abstractmethod
from custom_callbacks import CustomModelCheckpoint


class VAE(object, metaclass=ABCMeta):
    '''
    Class to handle helper and training functions of VAEs.
    '''

    def __init__(self, input_shape, log_dir, beta=1.0):
        '''
        input_shape : (num_channels, num_rows, num_cols)
        '''
        self.input_shape = input_shape
        self.log_dir = log_dir
        self.beta = beta
        self.set_model()
        self.__define_callbacks()  # call after set_model()

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

    def fit_generator(self, train_generator, **kwargs):
        '''
        Wrapper for Keras fit_generator method
        '''
        callbacks = kwargs.get('callbacks', self.callbacks)  # default callbacks
        self.model.fit_generator(train_generator, callbacks=callbacks, **kwargs)

    def compile(self, **kwargs):
        '''
        Wrapper for Keras compile method
        '''
        loss = kwargs.get('loss', self.vae_loss)  # default loss
        self.model.compile(loss=loss, **kwargs)

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
        y_true = K.reshape(y_true, (-1, np.prod(self.input_shape)))
        y_pred = K.reshape(y_pred, (-1, np.prod(self.input_shape)))
        reconstruction_loss = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        latent_shape = self.z.get_shape().as_list()[1:]
        z_mean_flat = K.reshape(self.z_mean, (-1, np.prod(latent_shape)))
        z_log_var_flat = K.reshape(self.z_log_var, (-1, np.prod(latent_shape)))
        kl_loss = -0.5 * K.sum(1 + z_log_var_flat - K.square(z_mean_flat) - K.exp(z_log_var_flat), axis=-1)
        return K.mean(reconstruction_loss + self.beta * kl_loss)

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

    def save_model_architecture(self):
        '''
        Saves model architecture of encoder, decoder and entire model
        '''
        self.__save_model_architecture(self.model, 'model')
        self.__save_model_architecture(self.encoder, 'encoder')
        self.__save_model_architecture(self.decoder, 'decoder')

    def __save_model_architecture(self, model, name):
        '''
        Helper for save_model_architecture
        '''
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        model_json = model.to_json()
        with open(self.log_dir + name + '.json', 'w') as json_file:
            json_file.write(model_json)

    def load_model_architecture(self):
        '''
        Loads model architectures
        '''
        self.model = self.__load_model_architecture('model')
        self.encoder = self.__load_model_architecture('encoder')
        self.decoder = self.__load_model_architecture('decoder')

    def __load_model_architecture(self, name):
        '''
        Helper for load_model_architecture
        '''
        json_file = open(self.log_dir + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, {'sampling': self.sampling})
        return model

    def load_model(self):
        '''
        Loads encoder, decoder and entire model.
        Expects files to be named 'encoder', 'decoder' and 'model'.
        '''
        self.encoder = load_model(self.log_dir + 'encoder.hdf5', {'sampling': self.sampling, 'vae_loss': self.vae_loss})
        self.decoder = load_model(self.log_dir + 'decoder.hdf5', {'sampling': self.sampling, 'vae_loss': self.vae_loss})
        self.model = load_model(self.log_dir + 'model.hdf5', {'sampling': self.sampling, 'vae_loss': self.vae_loss})

    def __define_callbacks(self):
        '''
        Helper for __init__
        '''
        tensorboard = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_images=False)
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.5,
                                                patience=2,
                                                mode='auto',
                                                verbose=0)
        model_checkpointer = CustomModelCheckpoint(filepath=self.log_dir + 'model' + '.hdf5',
                                                verbose=1,
                                                monitor='val_loss',
                                                mode='auto',
                                                period=1,
                                                save_best_only=True,
                                                other_models={'encoder': self.encoder, 'decoder': self.decoder})
        csv_logger = keras.callbacks.CSVLogger(self.log_dir + 'log.csv',
                                                separator=',',
                                                append=True)
        self.callbacks = [tensorboard, earlystopping, model_checkpointer, csv_logger]

    '''
    Getters
    '''
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_model(self):
        return self.model