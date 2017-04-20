import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import utils


'''
Inputs
'''
name = 'cvae14_batch_size_64_beta_1.0_epochs_5_loss_binary_crossentropy_optimizer_adadelta'
model_weights = 'weights.004-3.0960.hdf5'


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
# import dataset
from keras.datasets import mnist
(X_train, _), (X_test, _) = mnist.load_data()

# reshape into (num_samples, num_channels, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# record input shape
input_shape = X_train.shape[1:]

# cast pixel values to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise pixel values
X_train /= 255.0
X_test /= 255.0


'''
Sampling functions
'''
def __decode_prior_samples(num_samples, latent_shape=(1, 4, 11, 11)):
	# take num_sample samples
	for i in range(num_samples):
		# sample from prior
		prior_sample = np.random.normal(size=latent_shape)  # sample from standard normal
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


'''
Main
'''
if __name__ == '__main__':
	num_samples = 5
	latent_shape = (1, 4, 11, 11)
	__decode_prior_samples(num_samples, latent_shape=latent_shape)