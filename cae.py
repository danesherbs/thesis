import pickle
import numpy as np
import os
import keras
import tensorflow as tf

from keras.models import load_model
from keras import backend as K
from keras.objectives import binary_crossentropy
from keras.models import model_from_json
from custom_callbacks import CustomModelCheckpoint

from abc import ABCMeta, abstractmethod



class CAE(object, metaclass=ABCMeta):
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
        and the following if ortho_reg is used:
            - self.latent_filters
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
        loss = kwargs.get('loss', self.cae_loss)  # default loss
        self.model.compile(loss=loss, **kwargs)

    def cae_loss(self, y_true, y_pred):
        '''
        Variational autoencoder loss function
        '''
        return self.reconstruction_loss(y_true, y_pred)

    def reconstruction_loss(self, y_true, y_pred):
        '''
        Binary cross-entropy
        '''
        y_true = K.reshape(y_true, (-1, np.prod(self.input_shape)))
        y_pred = K.reshape(y_pred, (-1, np.prod(self.input_shape)))
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)

    def ortho_reg(self, w):
        # regularise w to be orthogonal
        # L = sum_{w,w'} log(1 + exp( l * (cos(w,w') - 1) ))  (see eq. 7)
        # set  l = 10

        w_new = tf.reshape(w, [-1, self.latent_filters])

        print("")
        print("")
        print("")
        print("GOT HERE")
        print("")
        print("")
        print("")

        # normalized w
        norm = tf.sqrt(tf.reduce_sum(tf.square(w_new), 0, keep_dims=True))
        w_new = w_new / ( norm + 1e-10 )

        # compute product
        prod = tf.matmul( K.transpose(w_new), w_new )
       
        l = 10
        L = tf.log( 1 + tf.exp(l * (prod - 1) ))
        diag_mat = tf.diag(tf.diag_part(L))
        L = L - diag_mat
        Loss = tf.reduce_sum( tf.reduce_sum(L) )

        return Loss

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

    def load_model(self):
        '''
        Loads encoder, decoder and entire model.
        Expects files to be named 'encoder', 'decoder' and 'model'.
        '''
        # specify location of custom functions to properly load serialized model
        custom_functions = {'cae_loss': self.cae_loss,
                            'reconstruction_loss': self.reconstruction_loss}
        # load relevant models
        self.encoder = load_model(self.log_dir + 'encoder.hdf5', custom_objects=custom_functions)
        self.decoder = load_model(self.log_dir + 'decoder.hdf5', custom_objects=custom_functions)
        self.model = load_model(self.log_dir + 'model.hdf5', custom_objects=custom_functions)

    def __define_callbacks(self):
        '''
        Helper for __init__
        '''
        tensorboard = keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_images=False)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.5,
                                                patience=8,
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
        reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.7,
                                                patience=0,
                                                verbose=1,
                                                mode='auto',
                                                epsilon=0.3,
                                                cooldown=2,
                                                min_lr=1e-6)
        self.callbacks = [tensorboard, early_stopping, model_checkpointer, csv_logger, reduce_lr_on_plateau]

    '''
    Getters
    '''
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_model(self):
        return self.model