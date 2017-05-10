from vae import VAE
import numpy as np
from keras import initializers
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model


class myVAE(VAE):

    def __init__(self, input_shape, optimizer):
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
        encoder_conv2D_1 = Conv2D(filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_1')
        encoder_conv2D_2 = Conv2D(2*filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_2')
        encoder_conv2D_3 = Conv2D(2*filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='encoder_conv2D_3')
        encoder_flatten_1 = Flatten()
        encoder_dense_1 = Dense(hidden_dim, activation='relu', name='encoder_dense_1')
        encoder_z_mean = Dense(latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_mean')
        encoder_z_log_var = Dense(latent_dim, activation=None, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='z_log_var')
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
        decoder_dense_1 = Dense(hidden_dim, activation='relu', name="decoder_dense_1")
        decoder_dense_2 = Dense(np.prod((64, 7, 7)), activation='relu', name="decoder_dense_2")
        decoder_reshape_1 = Reshape((64, 7, 7), name="decoder_reshape_1")
        decoder_conv2DT_1 = Conv2DTranspose(2*filters, kernal_size, strides=(2, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_1')
        decoder_conv2DT_2 = Conv2DTranspose(filters, kernal_size, strides=(2, 2), activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_2')
        decoder_conv2DT_3 = Conv2DTranspose(1, kernal_size, strides=(2, 2), activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='decoder_conv2DT_3')

        # Connect decoder layers
        input_decoder = Input(shape=(latent_dim,), name='decoder_input')
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
    vae.print_model_summaries()