import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils


'''
Log directory
'''
name = 'cvae_frey_11_May_14_57_52_batch_size_1_beta_2.0_epochs_10_filters_32_kernel_size_2_latent_size_8_loss_vae_loss_optimizer_adam_pre_latent_size_64'
log_dir = './summaries/' + name + '/'


'''
Load models
'''
# import model class
from cvae_frey import FreyVAE

# define model
vae = FreyVAE((1, 28, 20), log_dir)

# load weights
encoder = vae.load_model()

# extract models
model = vae.get_model()
decoder = vae.get_decoder()
encoder = vae.get_encoder()


'''
Load data
'''
dataset = 'frey'
if dataset == 'atari':
	test_directory = './atari_agents/record/test/'
	test_generator = utils.atari_generator(test_directory, batch_size=1)
	X_test_size = 100
	X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])
elif dataset == 'mnist':
	# import dataset
	(_, _), (X_test, _) = utils.load_keras()
elif dataset == 'frey':
	# import dataset
	(_, _), (X_test, _) = utils.load_frey()
else:
	print("Dataset not found.")

'''
Sampling functions
'''
def __decode_prior_samples(num_samples, latent_shape=(1, 4, 11, 11)):
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


def __encode_decode_sample(sample_number=0):
	# predict a sample
	decoded_imgs = model.predict(np.asarray([[X_test[sample_number][0]]]))
	# plot actual
	plt.figure(1)
	plt.title('Original')
	plt.imshow(X_test[sample_number][0])
	plt.gray()
	# plot predicted
	plt.figure(2)
	plt.title('Reconstructed')
	plt.imshow(decoded_imgs[0][0])
	plt.gray()
	# show both at same time
	plt.show()

def __sample_posterior(num_iter, show_every=1, init_sample_num=0):
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

def __demo_sample_posterior():
	num_iter = 500
	show_every = 5
	init_sample_num = 0
	__sample_posterior(num_iter, show_every=show_every, init_sample_num=init_sample_num)

'''
Main
'''
if __name__ == '__main__':
	# __decode_prior_samples(5, latent_shape=(1, 8))
	# __encode_decode_sample(sample_number=0)
	__demo_sample_posterior()