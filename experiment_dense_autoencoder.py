from cvae_mnist import DenseAutoencoder
from keras import optimizers
import utils


experiment = 'experiment_dense_autoencoder'


def train_dense_autoencoder():

    # inputs
    input_shape = (1, 28, 28)
    epochs = 15
    batch_size = 1
    lr = 1e-4

    # define filename
    name = 'autoencoder_mnist'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + experiment + '/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    autoencoder = DenseAutoencoder(input_shape, log_dir)
    
    # compile VAE
    optimizer = optimizers.Adam(lr=lr)
    autoencoder.compile(optimizer=optimizer)
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    
    # print summaries
    autoencoder.print_model_summaries()
    
    # fit VAE
    steps_per_epoch = int(len(X_train) / batch_size)
    validation_steps = int(len(X_test) / batch_size)
    autoencoder.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)


if __name__ == '__main__':
    train_dense_autoencoder()