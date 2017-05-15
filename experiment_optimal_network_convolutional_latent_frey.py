'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.

'''

from cvae_frey import FreyOptimalConvolutionalLatentExperimentVAE, \
                      FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE, \
                      FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterSameBordersVAE, \
                      FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterSameBordersNoPoolingVAE
import sampling
import utils
import numpy as np



def train_reconstruction_only_frey_network_with_image_latent_space():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_channels = 1
    kernel_size = 2
    beta = 0.0  # reconstruction loss only
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_latent_image'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
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



def train_entangled_frey_network_with_image_latent_space():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_channels = 1
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_latent_image'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
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


def train_entangled_frey_network_with_fully_connected_filters():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_channels = 64
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_fully_connected_filters'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
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


def train_entangled_frey_network_with_fully_connected_filters_same_borders():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_channels = 64
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_fully_connected_filters_same_borders'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
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



def train_entangled_frey_network_with_fully_connected_filters_same_borders_no_pooling():
    # inputs
    input_shape = (1, 28, 20)
    filters = 32
    latent_channels = 64
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_fully_connected_filters_same_borders_no_pooling'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-2)
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



def train_entangled_frey_network_with_fully_connected_filters_same_borders_no_pooling_less_filters():
    # inputs
    input_shape = (1, 28, 20)
    filters = 8
    latent_channels = 16
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 20
    batch_size = 1

    # define filename
    name = 'cvae_frey_entangled_with_fully_connected_filters_same_borders_no_pooling_less_filters'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_channels': latent_channels,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/experiment_optimal_network_convolutional_latent_frey/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
                                            beta=beta)

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
    latent_channels = 1
    kernel_size = 2
    beta = 1.0  # entangled latent space
    epochs = 10
    batch_size = 1

    # log directory
    experiment = 'experiment_optimal_network_convolutional_latent_frey'
    run = 'cvae_frey_entangled_with_fully_connected_filters_same_borders_15_May_13_50_05_batch_size_1_beta_1.0_epochs_20_filters_32_kernel_size_2_latent_channels_64_loss_vae_loss_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # define model
    vae = FreyOptimalConvolutionalLatentExperimentFullyConnectedFilterVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_channels=latent_channels,
                                            kernel_size=kernel_size,
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
    # sampling.encode_decode_sample(X_test, model)

    # plot filters
    # sampling.show_convolutional_layers(X_test, encoder, 8, 8)

    # sample from prior
    # sampling.decode_prior_samples(5, decoder, latent_shape=(1, 64, 1, 1))

    # sample from posterior
    num_iter = 1000
    sampling.sample_posterior(X_test, model, num_iter, show_every=5)


'''
Main
'''
if __name__ == '__main__':
    # train_reconstruction_only_frey_network_with_image_latent_space()
    # train_entangled_frey_network_with_image_latent_space()
    # train_entangled_frey_network_with_fully_connected_filters()
    # train_entangled_frey_network_with_fully_connected_filters_same_borders()
    # train_entangled_frey_network_with_fully_connected_filters_same_borders_no_pooling()
    # train_entangled_frey_network_with_fully_connected_filters_same_borders_no_pooling_less_filters()
    main()