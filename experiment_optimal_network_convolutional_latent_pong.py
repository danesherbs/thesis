'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.
'''

from cvae_atari import PongEntangledConvolutionalLatentVAE, \
                       PongEntangledConvolutionalLatentNoBatchNormVAE
import sampling
import utils
import numpy as np


experiment = 'experiment_optimal_network_convolutional_latent_pong'


def train_only_reconstruction_loss_pong_network():
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    kernel_size = 6
    beta = 0.0  # only reconstruction loss
    epochs = 10
    batch_size = 1

    # define filename
    name = 'cvae_atari_only_reconstruction_loss_pong'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = PongEntangledConvolutionalLatentVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)

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


def train_entangled_pong_network():
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    kernel_size = 6
    beta = 1.0  # entangled latent space
    epochs = 10
    batch_size = 1

    # define filename
    name = 'cvae_atari_entangled_pong'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = PongEntangledConvolutionalLatentVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)

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


def train_entangled_pong_network_with_image_latent_space():
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    latent_filters = 1
    kernel_size = 6
    beta = 1.0  # entangled latent space
    epochs = 100
    batch_size = 1

    # define filename
    name = 'cvae_atari_entangled_pong_with_latent_image'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = PongEntangledConvolutionalLatentVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_filters=latent_filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)

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



def train_only_reconstruction_loss_pong_network_no_batch_norm():
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    kernel_size = 6
    beta = 0.0  # only reconstruction loss
    epochs = 10
    batch_size = 1

    # define filename
    name = 'cvae_atari_only_reconstruction_loss_pong_no_batch_norm'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = PongEntangledConvolutionalLatentNoBatchNormVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-1)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)

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



def train_pong_network_no_batch_norm(beta):
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    kernel_size = 6
    epochs = 20
    batch_size = 1
    lr = 1e-4

    # define filename
    name = 'cvae_atari_pong_no_batch_norm'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'lr': lr,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = PongEntangledConvolutionalLatentNoBatchNormVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=lr)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size)
    train_size = utils.count_images(train_directory)
    test_size = utils.count_images(test_directory)

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
    input_shape = (1, 84, 84)
    filters = 32
    kernel_size = 6
    epochs = 20
    batch_size = 1

    # log directory
    run = 'cvae_atari_only_reconstruction_loss_pong_no_batch_norm_16_May_14_53_56_batch_size_1_beta_1_epochs_10_filters_32_kernel_size_6_loss_vae_loss_lr_0.0001_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # define model
    vae = PongEntangledConvolutionalLatentNoBatchNormVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            kernel_size=kernel_size)

    # load weights
    vae.load_model()

    # extract models
    model = vae.get_model()
    decoder = vae.get_decoder()
    encoder = vae.get_encoder()

    # load testing data
    test_directory = './atari_agents/record/test/'
    test_generator = utils.atari_generator(test_directory, batch_size=1)
    X_test_size = 100
    X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])

    # show original and reconstruction
    # sampling.encode_decode_sample(X_test, model)

    # plot filters
    # sampling.show_convolutional_layers(X_test, encoder, 8, 8)

    # sample from prior
    # sampling.decode_prior_samples(5, decoder, latent_shape=(1, 64, 7, 7))

    # sample from posterior
    num_iter = 100
    sampling.sample_posterior(X_test, model, num_iter, show_every=1)

    # change latent variable
    # latent_shape = (1, 64, 7, 7)
    # filter_index = 30
    # sampling.change_latent_filter(X_test, latent_shape, filter_index, encoder, decoder, num_samples=5, init_sample_num=0, noise_factor=15)

'''
Main
'''
if __name__ == '__main__':
    # train_only_reconstruction_loss_pong_network()
    # train_entangled_pong_network()
    # train_entangled_pong_network_with_image_latent_space()
    # train_only_reconstruction_loss_pong_network_no_batch_norm()
    train_pong_network_no_batch_norm(1)
    # main()