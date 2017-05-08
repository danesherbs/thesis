import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils


'''
Inputs
'''
name = 'cvae_higgins_07_May_10_47_56_batch_size_1_beta_1.0_epochs_20_latent_size_10_loss_vae_loss_optimizer_adam'
model_weights = 'weights.018-953.9656.hdf5'


'''
Log directory
'''
log_dir = './summaries/' + name + '/'


'''
Load models
'''
# load model json file
json_file = open(log_dir + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load encoder json file
json_file = open(log_dir + 'encoder.json', 'r')
loaded_encoder_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_encoder_json)

# load decoder json file
json_file = open(log_dir + 'decoder.json', 'r')
loaded_decoder_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_decoder_json)


'''
Load weights
'''
model.load_weights(log_dir + model_weights)
encoder.load_weights(log_dir + 'encoder_weights.hdf5')
decoder.load_weights(log_dir + 'decoder_weights.hdf5')


'''
Load data
'''
custom_data = True
if custom_data:
	test_directory = './atari_agents/record/test/'
	test_generator = utils.atari_data_generator(test_directory, batch_size=1)
	X_test_size = 100
	X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])
else:
	# import dataset
	from keras.datasets import mnist
	(_, _), (X_test, _) = mnist.load_data()
	# reshape into (num_samples, num_channels, width, height)
	X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
	# record input shape
	input_shape = X_train.shape[1:]
	# cast pixel values to floats
	X_test = X_test.astype('float32')
	# normalise pixel values
	X_test /= 255.0

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
	num_iter = 1000
	show_every = 100
	init_sample_num = 99
	__sample_posterior(num_iter, show_every=show_every, init_sample_num=init_sample_num)

'''
Main
'''
if __name__ == '__main__':
	# __decode_prior_samples(5, latent_shape=(1, 10))
	# __encode_decode_sample(sample_number=0)
	__demo_sample_posterior()