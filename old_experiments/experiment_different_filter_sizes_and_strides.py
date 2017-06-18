'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.
'''

from cvae_atari import ConvolutionalLatentAverageFilterShallowVAE
import sampling
import utils
import numpy as np
import matplotlib.pyplot as plt


experiment = 'experiment_different_filter_sizes_and_strides'


def train_average_filter(beta):
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 7
    epochs = 12
    batch_size = 1
    lr = 1e-4

    # define filename
    name = 'cvae_atari'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'latent_filters': latent_filters,
        'kernel_size': kernel_size,
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
    latent_filters = 8
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    beta = 1.0

    # log directory
    run = 'cvae_atari_28_May_16_14_52_batch_size_1_beta_1_epochs_12_filters_32_kernel_size_7_latent_filters_8_loss_vae_loss_lr_0.0001_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # make VAE
    vae = ConvolutionalLatentAverageFilterShallowVAE(input_shape, 
                                            log_dir,
                                            filters=filters,
                                            latent_filters=latent_filters,
                                            kernel_size=kernel_size,
                                            beta=beta)

    # load weights
    vae.load_model()

    # extract models
    model = vae.get_model()
    decoder = vae.get_decoder()
    encoder = vae.get_encoder()

    # load testing data
    test_directory = './atari_agents/record/test/'
    test_generator = utils.atari_generator(test_directory, batch_size=1, shuffle=False)
    X_test_size = 330
    X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])

    # show original and reconstruction
    sampling.encode_decode_sample(X_test, model)
    plt.show()

    # plot filters
    # sampling.show_convolutional_layers(X_test, encoder, 4, 2, threshold=True, threshold_val=0.0)
    # plt.show()

    # sample from prior
    # sampling.decode_prior_samples(5, decoder, latent_shape=(1, 8, 7, 7))
    # plt.show()

    # sample from posterior
    # num_iter = 100
    # sampling.sample_posterior(X_test, model, num_iter, show_every=1)

    # change latent variable
    # latent_shape = (1, 8, 8, 8)
    # filter_index = 6
    # sampling.change_latent_filter(X_test,
    #                             latent_shape,
    #                             filter_index,
    #                             encoder,
    #                             decoder,
    #                             num_samples=10,
    #                             init_sample_num=0,
    #                             noise_factor=1.0,
    #                             std_dev=1.0,
    #                             mean=0.0)

    # plot mean activation over latent space
    # sampling.plot_mean_latent_activation(X_test, encoder, 4, 2, threshold=True, threshold_val=0.0)
    # plt.show()

    # change single latent neuron
    # latent_shape = (1, 8, 7, 7)
    # latent_variable_pos = (0, 5, 2, 2)
    # range_values = np.arange(0, 4, 0.1)
    # sampling.change_latent_variable_over_range(X_test, latent_variable_pos, encoder, decoder, range_values)

    # latent_variable_pos = (0, 4, 7, 3)
    # utils.make_slider(lambda val: sampling.change_latent_variable(X_test,
    #                                                             latent_variable_pos,
    #                                                             encoder,
    #                                                             decoder,
    #                                                             val,
    #                                                             init_sample_num=0,
    #                                                             show=True))

'''
Main
'''
if __name__ == '__main__':
    # for beta in range(1, 4):
    #     train_average_filter(beta)
    main()