import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils
from PIL import Image


def decode_prior_samples(num_samples, decoder, latent_shape=(1, 4, 11, 11)):
    # take num_sample samples
    for i in range(num_samples):
        # sample from prior
        prior_sample = np.random.normal(size=latent_shape, loc=0.0, scale=1.0)  # sample from standard normal
        # decode sample
        sample_decoded = decoder.predict(prior_sample)
        # plot decoded sample
        plt.figure(i)
        plt.title("Sample " + str(i))
        plt.imshow(sample_decoded[0][0])
        plt.gray()
    # show all plots at once

def encode_decode_sample(X, model, sample_number=0, img_channels=1):
    # predict a sample
    decoded_imgs = model.predict(np.asarray([X[sample_number]]))
    # plot actual
    plt.figure(1)
    plt.title('Original')
    plt.imshow(np.reshape(X[sample_number], (84, 84, img_channels)))
    plt.gray()
    # plot predicted
    plt.figure(2)
    plt.title('Reconstructed')
    plt.imshow(np.reshape(decoded_imgs[0], (84, 84, img_channels)))
    plt.gray()

def sample_posterior(X, model, num_iter, show_every=1, init_sample_num=0):
    # seed with initial sample
    x = np.asarray([[X[init_sample_num][0]]])
    # MCMC samlping from P^(Z)
    for iter in range(num_iter):
        # pass through CVAE
        x = model.predict(x)
        # plot result
        if np.mod(iter, show_every) == 0:
            plt.figure()
            plt.title("Iteration " + str(iter))
            plt.imshow(x[0][0])
            plt.gray()
            plt.show()

def show_convolutional_layers(X, encoder, rows, columns, init_sample_num=0, threshold=False, threshold_val=0.5):
    '''
    Takes x_encoded volume of shape (num_filters, width, height)
    and plots activations of each (num_filters in total).
    '''
    # encode image
    x_encoded = encoder.predict(np.asarray([X[init_sample_num]]))  # shape (1, num_filters, width, height)
    x_encoded = x_encoded[0]  # shape (num_filters, width, height)

    if threshold:
        apply_threshold(x_encoded, threshold_val)

    # plot in grid of shape (rows, columns)
    plt.figure()
    filter = 0
    for row in range(rows):
        for column in range(columns):
            ax = plt.subplot2grid((rows, columns), (row, column))
            ax.set_title('Filter ' + str(filter))
            ax.imshow(x_encoded[filter])
            filter += 1

def __demo_sample_posterior():
    num_iter = 500
    show_every = 5
    init_sample_num = 0
    __sample_posterior(num_iter, show_every=show_every, init_sample_num=init_sample_num)


def change_latent_variable_over_range(X, latent_variable_pos, encoder, decoder, range_values, init_sample_num=0):
    for val in range_values:
        change_latent_variable(X, latent_variable_pos, encoder, decoder, val, init_sample_num=init_sample_num, show=True)

def change_latent_variable(X, latent_variable_pos, encoder, decoder, val, init_sample_num=0, show=True):
    # initialise latent_vector with encoded images
    latent_vector = encoder.predict(np.asarray([[X[init_sample_num][0]]]))

    print(latent_vector.shape)
    # change only the latent value at latent_index
    latent_vector[latent_variable_pos] = val
    decoded_latent_vector = decoder.predict(latent_vector)
    plt.figure(1)
    plt.title("Latent value " + str(val))
    plt.imshow(decoded_latent_vector[0][0])
    plt.gray()
    if show:
        plt.show()


def change_latent_filter(X, latent_shape, filter_index, encoder, decoder, num_samples=5, init_sample_num=0, noise_factor=1.0, std_dev=1.0, mean=0.0):
    # initialise latent_vector with encoded image
    latent_vector = encoder.predict(np.asarray([[X[init_sample_num][0]]]))

    # add noise to filter filter_index of latent_vector and decode result
    for i in range(num_samples):
        # create noisy filter
        noise = np.zeros(latent_shape)
        # noise[0][filter_index] = np.random.randn(latent_shape[2], latent_shape[3])
        noise[0][filter_index] = np.random.normal(size=(latent_shape[2], latent_shape[3]), scale=std_dev, loc=mean)
        # add noise to filter
        latent_vector_noisy = latent_vector + noise_factor * noise
        # decode result
        decoded_latent_vector = decoder.predict(latent_vector_noisy)
        plt.figure()
        plt.title("Filter " + str(filter_index) + ", sample " + str(i))
        plt.imshow(decoded_latent_vector[0][0])
        plt.gray()
        plt.show()

def plot_mean_latent_activation(X, encoder, rows, columns, threshold=False, threshold_val=0.8):
    # get mean latent vector
    mean_encoded = mean_latent_activation(X, encoder)  # shape (64, 7, 7)
    if threshold:
        apply_threshold(mean_encoded, threshold_val)
    # plot in grid of shape (rows, columns)
    plt.figure()
    filter = 0
    for row in range(rows):
        for column in range(columns):
            ax = plt.subplot2grid((rows, columns), (row, column))
            ax.imshow(mean_encoded[filter])
            filter += 1

def mean_latent_activation(X, encoder):
    # get encoded images
    X_encoded = encoder.predict(X)
    # return mean over all encoded images
    return np.mean(X_encoded, axis=0)

def apply_threshold(np_array, thresh_val):
    low_value_indicies = np_array < thresh_val
    np_array[low_value_indicies] = 0