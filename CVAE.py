from keras import initializers
import utils


class CVAE(object):
    '''
    Convolutional Variational Autoencoder
    '''

    def __init__(self,
                filename = 'CVAE',
                batch_size = 128,
                epochs = 50,
                beta = 1.0,
                loss_function = 'binary_crossentropy',
                optimizer = 'rmsprop',
                weight_seed = None,
                init_kernel_mean = 0.0,
                init_kernal_stddev = 0.5,
                init_bias_mean = 1.0,
                init_bias_stddev = 0.5):

        # hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta

        # optimisation
        self.loss_function = loss_function
        self.optimizer = optimizer

        # initalise summary path
        self.log_dir = self.__build_log_dir(filename)

        # initialisers
        self.weight_seed = weight_seed
        self.kernel_initializer = initializers.TruncatedNormal(mean=init_kernel_mean, stddev=init_kernal_stddev, seed=weight_seed)
        self.bias_initializer = initializers.TruncatedNormal(mean=init_bias_mean, stddev=init_bias_stddev, seed=weight_seed)

        filters = 8
        latent_filters = 4
        kernal_size = (3, 3)
        pool_size = (2, 2)

    def __build_log_dir(self, filename):
        # builder hyperparameter dictionary
        hp_dictionary = {
        	'batch_size': self.batch_size,
        	'epochs': self.epochs,
        	'beta': self.beta,
        	'loss': self.loss_function,
        	'optimizer': self.optimizer
        }
        # define log directory
        return './summaries/' + utils.build_hyperparameter_string(filename, hp_dictionary) + '/'


    def encoder(self):
        pass

    def decoder(self):
        pass

    def print_log_dir(self):
        print(self.log_dir)
        return None


if __name__ == '__main__':
    # build CVAE
    cvae = CVAE()

    # call function
    cvae.print_log_dir()
