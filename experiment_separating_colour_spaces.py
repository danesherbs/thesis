'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.
'''

from cvae_atari import LatentImage, ConvolutionalLatentAverageFilterShallowVAE, WeightedAverageFilters
import utils
import numpy as np


experiment = 'experiment_separating_colour_spaces'


def train_latent_image(beta):
    # inputs
    input_shape = (3, 84, 84)
    filters = 32
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    img_channels = 3

    # define filename
    name = 'latent_image'

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
    vae = LatentImage(input_shape, 
                    log_dir,
                    filters=filters,
                    kernel_size=kernel_size,
                    img_channels=img_channels,
                    beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=lr)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size, img_channels=img_channels)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size, img_channels=img_channels)
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





def train_naive_average(beta):
    # inputs
    input_shape = (3, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    img_channels = 3

    # define filename
    name = 'naive_average_filter'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
        'img_channels': img_channels,
        'lr': lr,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = ConvolutionalLatentAverageFilterShallowVAE(input_shape, 
                                                log_dir,
                                                filters=filters,
                                                latent_filters=latent_filters,
                                                kernel_size=kernel_size,
                                                img_channels=img_channels,
                                                beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=lr)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size, img_channels=img_channels)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size, img_channels=img_channels)
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





def train_weighted_average(beta):
    # inputs
    input_shape = (3, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    img_channels = 3

    # define filename
    name = 'weighted_average_filter'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
        'img_channels': img_channels,
        'lr': lr,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = WeightedAverageFilters(input_shape, 
                                log_dir,
                                filters=filters,
                                latent_filters=latent_filters,
                                kernel_size=kernel_size,
                                img_channels=img_channels,
                                beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=lr)
    vae.compile(optimizer=optimizer)

    # get dataset
    train_directory = './atari_agents/record/train/'
    test_directory = './atari_agents/record/test/'
    train_generator = utils.atari_generator(train_directory, batch_size=batch_size, img_channels=img_channels)
    test_generator = utils.atari_generator(test_directory, batch_size=batch_size, img_channels=img_channels)
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
    for beta in 2**np.arange(3):
        train_latent_image(beta)
        train_naive_average(beta)
        train_weighted_average(beta)


'''
Main
'''
if __name__ == '__main__':
    main()