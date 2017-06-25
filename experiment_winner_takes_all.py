'''
This experiment trains a winner takes all network for different beta.
'''

from cvae_atari import WinnerTakesAll
import sampling
import utils
import numpy as np
import matplotlib.pyplot as plt


experiment = 'experiment_winner_takes_all'


def train_winner_takes_all(beta):
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 6
    epochs = 5
    batch_size = 1
    lr = 1e-4
    img_channels = 1

    # define filename
    name = 'cvae_atari_winner_takes_all'

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


def main():
    for beta in [1, 4, 16]:
        train_winner_takes_all(beta)


'''
Main
'''
if __name__ == '__main__':
    main()
