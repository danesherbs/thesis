import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils


'''
Inputs
'''
name = 'cvae_frey_11_May_10_08_48_batch_size_1_beta_2.0_epochs_10_filters_32_kernel_size_2_latent_size_8_loss_vae_loss_optimizer_adam_pre_latent_size_64'
model_weights = 'weights.007-351.1501.hdf5'


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
(_, _), (X_test, _) = utils.load_frey()


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


def traverse_latent_space(latent_size):
	stddevs = 1
	subplot_width = len(range(-stddevs, stddevs+1))
	subplot_height = latent_size
	fig_counter = 1
	fig = plt.figure()
	for latent_variable in range(latent_size):
		for std in range(-stddevs, stddevs+1):
			# construct latent sample
			encoded_sample = np.zeros((1, latent_size))
			encoded_sample[0][latent_variable] = std
			# decode latent sample
			decoded_sample = decoder.predict(encoded_sample)
			# plot reconstruction
			subplot_number = int(str(subplot_height) + str(subplot_width) + str(fig_counter))
			print(subplot_number)
			print('Figure', subplot_number)
			ax = fig.add_subplot(330 + fig_counter)
			# ax.title('Latent variable = ' + str(latent_variable) + ', std = ' + str(std))
			ax.imshow(decoded_sample[0][0])
			# increment fig_counter
			fig_counter += 1
	plt.show()
	plt.gray()

# def plot_manifold():
# 	width = 5
# 	height = 5
# 	for i in range(width):
# 		for j in range(height):



'''
Main
'''
if __name__ == '__main__':
	latent_size = 2
	traverse_latent_space(latent_size)