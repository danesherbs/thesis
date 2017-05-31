from cvae_mnist import DenseAutoencoder
from keras import optimizers
import utils
import sampling
import matplotlib.pyplot as plt


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


def main():
    # inputs
    img_channels = 1
    input_shape = (img_channels, 28, 28)
    epochs = 50
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
    run = 'autoencoder_mnist_31_May_19_54_30_batch_size_1_epochs_60_loss_vae_loss_lr_0.0001_optimizer_adam'
    log_dir = './summaries/' + experiment + '/' + run + '/'

    # make VAE
    autoencoder = DenseAutoencoder(input_shape, log_dir)
    
    # load weights
    autoencoder.load_model()

    # extract models
    model = autoencoder.get_model()
    decoder = autoencoder.get_decoder()
    encoder = autoencoder.get_encoder()
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_mnist()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    
    # show original and reconstruction
    sampling.encode_decode_sample(X_test, model, input_shape=(1, 28, 28))
    plt.show()


if __name__ == '__main__':
    # train_dense_autoencoder()
    main()