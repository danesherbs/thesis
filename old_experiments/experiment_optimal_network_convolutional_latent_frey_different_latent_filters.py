'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.

'''

from cvae_frey import FreyConvolutionalLatentSpaceNoBatchNormVAE
import sampling
import utils
import numpy as np


experiment = 'experiment_optimal_network_convolutional_latent_frey_different_latent_filters'


def train_reconstruction_only_frey_network_with_image_latent_space(latent_filters):
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    kernel_size = 2
    pool_size = 2
    beta = 1.0
    lr = 1e-4
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
        'beta': beta,
        'lr': lr,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyConvolutionalLatentSpaceNoBatchNormVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_filters=latent_filters,
                                            kernel_size=kernel_size,
                                            pool_size=pool_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=lr)
    vae.compile(optimizer=optimizer)

    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

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


def main():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_filters = 1
    kernel_size = 2
    pool_size = 2
    lr = 1e-4
    beta = 1.0
    batch_size = 1

    # log directory
    run = 'cvae_frey_20_May_17_33_37_batch_size_1_beta_1.0_epochs_20_filters_32_kernel_size_2_latent_filters_2_loss_vae_loss_lr_0.0001_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # define model
    vae = FreyConvolutionalLatentSpaceNoBatchNormVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_filters=latent_filters,
                                            kernel_size=kernel_size,
                                            pool_size=pool_size,
                                            beta=beta)

    # load weights
    vae.load_model()

    # extract models
    model = vae.get_model()
    decoder = vae.get_decoder()
    encoder = vae.get_encoder()

    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

    # show original and reconstruction
    sampling.encode_decode_sample(X_test, model)

    # plot filters
    # sampling.show_convolutional_layers(X_test, encoder, 8, 8)

    # sample from prior
    # sampling.decode_prior_samples(5, decoder, latent_shape=(1, 64, 1, 1))

    # sample from posterior
    # num_iter = 1000
    # sampling.sample_posterior(X_test, model, num_iter, show_every=5)


'''
Main
'''
if __name__ == '__main__':
    for latent_filters in 2**np.arange(0, 1):
        train_reconstruction_only_frey_network_with_image_latent_space(latent_filters)
    # main()