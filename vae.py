import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Deconvolution2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.objectives import binary_crossentropy


class VAE():
    """
    Class to handle building and training VAE models.
    """
    def __init__(self, input_shape=(1, 84, 84), latent_dim=32, hidden_dim=256, filters=32):
        """
        Setting up everything.
        Parameters
        ----------
        input_shape : Array of shape (num_rows, num_cols, num_channels)
            Shape of image.
        latent_dim : int
            Dimension of latent distribution.
        latent_disc_dim : int
            Dimension of discrete latent distribution.
        hidden_dim : int
            Dimension of hidden layer.
        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of depth.
        """
        self.optimizer = None
        self.model = None
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.filters = filters

    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,
            learning_rate=1e-3, reset_model=True):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self.__set_model()

        if x_train.shape[0] % batch_size != 0:
            raise(RuntimeError("Training data shape {} is not divisible by batch size {}".format(x_train.shape[0], self.batch_size)))

        # Update parameters
        K.set_value(self.optimizer.lr, learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=self.__vae_loss)

        self.model.fit(x_train, x_train,
                        nb_epoch=self.num_epochs,
                        batch_size=self.batch_size,
                        validation_split=val_split)

    def __set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...", end=' ')

        # Encoder layers
        encoder_conv2D_1 = Conv2D(filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')
        encoder_conv2D_2 = Conv2D(2*filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')
        encoder_conv2D_3 = Conv2D(2*filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')
        encoder_flatten_1 = Flatten()
        encoder_dense_1 = Dense(self.hidden_dim, activation='relu', name='encoder_dense_1')
        encoder_z_mean = Dense(self.latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_mean')
        encoder_z_log_var = Dense(self.latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_log_var')
        encoder_z = Lambda(self.__sampling, name='z')

        # Connect encoder layers
        input_encoder = Input(shape=(self.batch_size,) + self.input_shape, name='encoder_input')
        x = encoder_conv2D_1(input_encoder)
        x = encoder_conv2D_2(x)
        x = encoder_conv2D_3(x)
        x = encoder_flatten_1(x)
        x = encoder_dense_1(x)
        z_mean = encoder_z_mean(x)
        z_log_var = encoder_z_log_var(x)
        z = encoder_z([z_mean, z_log_var])

        # Define encoder model
        self.encoder = Model(input_encoder, z)

        # Decoder layers
        decoder_dense_1 = Dense(pre_latent_size, activation='relu', name="decoder_dense_1")
        decoder_dense_2 = Dense(self.hidden_dim, activation='relu', name="decoder_dense_2")
        decoder_reshape_1 = Reshape(before_flatten_shape[1:], name="decoder_reshape_1")
        decoder_conv2DT_1 = Conv2DTranspose(2*filters, kernal_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')
        decoder_conv2DT_2 = Conv2DTranspose(filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')
        decoder_conv2DT_3 = Conv2DTranspose(1, kernal_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_3')(x)

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

        # Compile model
        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer, loss=self.__vae_loss)

        print("done.")
        return None

    def __sampling(args):
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

    def __vae_loss(y_true, y_pred):
        # y_true is of shape (batch_size,) + input_shape
        # y_pred is of shape (batch_size,) + output_shape
        # For example, training an autoencoder on MNIST with batch_size = 64 gives
        # y_true and y_pred both a shape of of shape (64, 1, 28, 28).

        # Flatten y_true and y_pred of shape (batch_size, 1, 28, 28) to (batch_size, 1 * 28 * 28).
        # Elements are in the interval [0, 1], which can be interpreted as probabilities.
        y_true = K.reshape(y_true, (-1, np.prod(input_shape)))
        y_pred = K.reshape(y_pred, (-1, np.prod(input_shape)))

        # Take the sum of the binary cross entropy for each image in the batch.
        # Reconstruction loss is of the shape (batch_size, 1).
        reconstruction_loss = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        # reconstruction_loss = np.prod(input_shape) * K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

        # Get latent shape
        latent_shape = self.z.get_shape().as_list()[1:]
        # Flatten latent space into shape (batch_size,) + flattened_latent_space
        z_mean_flat = K.reshape(self.z_mean, (-1, np.prod(latent_shape)))
        z_log_var_flat = K.reshape(self.z_log_var, (-1, np.prod(latent_shape)))
        # Take the KL divergence between q(z|x) and the standard multivariate Gaussian
        # for each image in the batch. KL loss is of the shape (batch_size, 1).
        kl_loss = -0.5 * K.sum(1 + z_log_var_flat - K.square(z_mean_flat) - K.exp(z_log_var_flat), axis=-1)

        # Return the mean weighted sum of reconstruction loss and KL divergence.
        # The output loss is therefore scalar after taking the mean of vector of shape (batch_size,).
        return K.mean(reconstruction_loss + beta * kl_loss)