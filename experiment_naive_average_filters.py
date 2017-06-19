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


experiment = 'experiment_naive_average'


def train_average_filter(beta):
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    img_channels = 1

    # define filename
    name = 'cvae_atari_average_filter'

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
    # inputs
    input_shape = (1, 84, 84)
    filters = 32
    latent_filters = 8
    kernel_size = 6
    epochs = 10
    batch_size = 1
    lr = 1e-4
    img_channels = 1
    beta = 1.0

    # log directory
    # run = 'cvae_atari_average_filter_18_Jun_15_40_50_batch_size_1_beta_1_epochs_10_filters_32_img_channels_1_kernel_size_6_latent_filters_8_loss_vae_loss_lr_0.0001_optimizer_adam'
    # run = 'cvae_atari_average_filter_18_Jun_18_07_12_batch_size_1_beta_2_epochs_10_filters_32_img_channels_1_kernel_size_6_latent_filters_8_loss_vae_loss_lr_0.0001_optimizer_adam'
    run = 'cvae_atari_average_filter_18_Jun_20_33_08_batch_size_1_beta_4_epochs_10_filters_32_img_channels_1_kernel_size_6_latent_filters_8_loss_vae_loss_lr_0.0001_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # make VAE
    vae = ConvolutionalLatentAverageFilterShallowVAE(input_shape, 
                                log_dir,
                                filters=filters,
                                latent_filters=latent_filters,
                                kernel_size=kernel_size,
                                img_channels=img_channels,
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
    X_test_size = 1000
    X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])

    # show original and reconstruction
    # for sample_number in range(4):
        # sampling.encode_decode_sample(X_test, model, sample_number=sample_number, save=True, save_path='/home/dane/Documents/Thesis/thesis/figures/results/naive_average/', base='beta_4_')

    # plot filters
    # sampling.show_convolutional_layers(X_test, encoder, 2, 4, init_sample_num=3, save=True, save_path='/home/dane/Documents/Thesis/thesis/figures/results/naive_average/', base='beta_4_')
    # plt.show()

    # sample from prior
    # sampling.decode_prior_samples(5, decoder, latent_shape=(1, 8, 8, 8), save=True, save_path='/home/dane/Documents/Thesis/thesis/figures/results/naive_average/', base='beta_4_')
    # plt.show()

    # sample from posterior
    # num_iter = 100
    # sampling.sample_posterior(X_test, model, num_iter, show_every=1, save=True, save_path='/home/dane/Documents/Thesis/thesis/figures/results/naive_average/', base='beta_4_')

    # change latent variable
    # latent_shape = (1, 8, 8, 8)
    # filter_index = 0
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
    sampling.plot_mean_latent_activation(X_test, encoder, 2, 4, threshold=True, threshold_val=0.1)
    plt.subplots_adjust(left=0.14, bottom=0.63, right=0.5, top=1, wspace=0.14, hspace=0)
    plt.savefig('/home/dane/Documents/Thesis/thesis/figures/results/naive_average/' + 'beta_4_' + 'average_activation' + '.png', bbox_inches='tight')
    # plt.show()



'''
Main
'''
if __name__ == '__main__':
    # main()
    for beta in 2**np.arange(3):
        train_average_filter(beta)