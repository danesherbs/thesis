'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.
'''

from cvae_mnist import CholletVAE, ShallowDenseMNIST
import utils
import numpy as np


experiment = 'experiment_dense_mnist'


def train_deep(beta):
    # inputs
    input_shape = (1, 28, 28)
    epochs = 20
    batch_size = 1
    filters = 32
    kernel_size = 6
    pre_latent_size = 128
    latent_size = 32

    # define filename
    name = 'deep'

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
    vae = CholletVAE(input_shape, 
                    log_dir,
                    filters=filters,
                    kernel_size=kernel_size,
                    pre_latent_size=pre_latent_size,
                    latent_size=latent_size,
                    beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-4)
    vae.compile(optimizer=optimizer)

    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
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



def train_shallow(beta):
    # inputs
    input_shape = (1, 28, 28)
    epochs = 30
    batch_size = 1
    filters = 32
    kernel_size = 6
    pre_latent_size = 128
    latent_size = 32

    # define filename
    name = 'shallow'

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
    vae = ShallowDenseMNIST(input_shape, 
                            log_dir,
                            filters=filters,
                            kernel_size=kernel_size,
                            pre_latent_size=pre_latent_size,
                            latent_size=latent_size,
                            beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-4)
    vae.compile(optimizer=optimizer)

    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
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


def train_deep_two_latent_variables(beta):
    # inputs
    input_shape = (1, 28, 28)
    epochs = 20
    batch_size = 1
    filters = 32
    kernel_size = 6
    pre_latent_size = 128
    latent_size = 2

    # define filename
    name = 'deep'

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
    vae = CholletVAE(input_shape, 
                    log_dir,
                    filters=filters,
                    kernel_size=kernel_size,
                    pre_latent_size=pre_latent_size,
                    latent_size=latent_size,
                    beta=beta)

    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-4)
    vae.compile(optimizer=optimizer)

    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
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
    for beta in range(1, 4):
        # train_deep(beta)
        train_shallow(beta)
    train_deep_two_latent_variables(1)


'''
Main
'''
if __name__ == '__main__':
    main()