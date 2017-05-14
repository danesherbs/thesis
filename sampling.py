import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils


def decode_prior_samples(num_samples, latent_shape=(1, 4, 11, 11)):
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
	plt.show()

def encode_decode_sample(X, model, sample_number=0):
	# predict a sample
	decoded_imgs = model.predict(np.asarray([[X[sample_number][0]]]))
	# plot actual
	plt.figure(1)
	plt.title('Original')
	plt.imshow(X[sample_number][0])
	plt.gray()
	# plot predicted
	plt.figure(2)
	plt.title('Reconstructed')
	plt.imshow(decoded_imgs[0][0])
	plt.gray()
	# show both at same time
	plt.show()

def sample_posterior(num_iter, show_every=1, init_sample_num=0):
	# seed with initial sample
	x = np.asarray([[X_test[init_sample_num][0]]])
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

def show_convolutional_layers(X, encoder, rows, columns, init_sample_num=0):
	'''
	Takes x_encoded volume of shape (num_filters, width, height)
	and plots activations of each (num_filters in total).
	'''
	# encode image
	x_encoded = encoder.predict(np.asarray([[X[init_sample_num][0]]]))  # shape (1, num_filters, width, height)
	x_encoded = x_encoded[0]  # shape (num_filters, width, height)

	# plot in grid of shape (rows, columns)
	filter = 0
	for row in range(rows):
		for column in range(columns):
			ax = plt.subplot2grid((rows, columns), (row, column))
			ax.imshow(x_encoded[filter])
			filter += 1
	plt.show()


def __demo_sample_posterior():
	num_iter = 500
	show_every = 5
	init_sample_num = 0
	__sample_posterior(num_iter, show_every=show_every, init_sample_num=init_sample_num)