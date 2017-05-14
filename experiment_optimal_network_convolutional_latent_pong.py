'''
This experiment will try to find an entangled convolutional latent space of Pong.
This is done so that we may see if objects may be recognised by the existing method
(that is, obtaining the spectrum of activations across features).

Dataset: 300,000 screenshots of Pong, 10% used for validation.
'''

from cvae_atari import PongEntangledConvolutionalLatentVAE
from sampling import encode_decode_sample
import utils
import numpy as np


# inputs
input_shape = (1, 84, 84)
epochs = 10
batch_size = 1
beta = 0.0  # entangled latent space
filters = 32
kernel_size = 6

def train_pong_network():
	# define filename
	name = 'cvae_atari_pong'

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
	log_dir = './summaries/experiment_optimal_network_convolutional_latent_pong/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

	# make VAE
	vae = PongEntangledConvolutionalLatentVAE(input_shape, 
	            log_dir,
	            filters=filters,
	            kernel_size=kernel_size,
	            beta=beta)

	# compile VAE
	from keras import optimizers
	optimizer = optimizers.Adam(lr=1e-3)
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


# '''
# Main
# '''

# if __name__ == '__main__':
# 	# log directory
# 	name = 'cvae_atari_pong_14_May_12_43_49_batch_size_1_beta_0.0_epochs_10_filters_32_kernel_size_6_loss_vae_loss_optimizer_adam'
# 	log_dir = './summaries/experiment_optimal_network_convolutional_latent_pong/' + name + '/'

# 	# define model
# 	vae = PongEntangledConvolutionalLatentVAE(input_shape, 
# 	            log_dir,
# 	            filters=filters,
# 	            kernel_size=kernel_size)    

# 	# load weights
# 	vae.load_model()

# 	# extract models
# 	model = vae.get_model()
# 	decoder = vae.get_decoder()
# 	encoder = vae.get_encoder()

# 	# load testing data
# 	test_directory = './atari_agents/record/test/'
# 	test_generator = utils.atari_generator(test_directory, batch_size=1)
# 	X_test_size = 100
# 	X_test = np.asarray([next(test_generator)[0][0] for i in range(X_test_size)])

# 	# plot a sample and its reconstruction
# 	encode_decode_sample(X_test, model)