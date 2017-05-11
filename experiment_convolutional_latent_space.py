import utils
from cvae_frey import FreyDenseLatentVAE, FreyConvolutionalLatentVAE


'''
All experiments run with:
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.5, patience=6, mode='auto', verbose=0)
'''


'''
EXPERIMENT 1
'''
def run_dense_latent_space():
    '''
    Latent shape: (15,)
    Number of parameters in network: 119,839
    Number of parameters in latent space: 33,615
    '''

    # inputs
    input_shape = (1, 28, 20)
    epochs = 1
    batch_size = 1
    beta = 1.0
    filters = 32
    kernel_size = 2
    pre_latent_size = 64
    latent_size = 15
    
    # define filename
    name = 'cvae_frey_dense_latent'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'pre_latent_size': pre_latent_size,
        'latent_size': latent_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyDenseLatentVAE(input_shape, 
                log_dir,
                filters=filters,
                kernel_size=kernel_size,
                pre_latent_size=pre_latent_size,
                latent_size=latent_size)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)

    # print summaries
    vae.print_model_summaries()
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

    # fit VAE
    steps_per_epoch = int(train_size / batch_size)
    validation_steps = int(test_size / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)


'''
EXPERIMENT 2
'''
def run_convolutional_latent_space_with_same_number_of_parameters():
    '''
    Latent shape: (128, 5, 3)
    Number of parameters in network: 119,521
    Number of parameters in latent space: 32,896
    '''

    # inputs
    input_shape = (1, 28, 20)
    epochs = 1
    batch_size = 1
    beta = 1.0
    filters = 32
    kernel_size = 2
    latent_channels = 128
    pool_size = 2
    
    # define filename
    name = 'cvae_frey_convolutional_latent'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'latent_channels': latent_channels,
        'pool_size': pool_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyConvolutionalLatentVAE(input_shape, 
                log_dir,
                filters=filters,
                kernel_size=kernel_size,
                latent_channels=latent_channels,
                pool_size=pool_size)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)

    # print summaries
    vae.print_model_summaries()
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

    # fit VAE
    steps_per_epoch = int(train_size / batch_size)
    validation_steps = int(test_size / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)


'''
EXPERIMENT 3
'''
def run_convolutional_latent_space_with_same_number_of_neurons():
    '''
    Latent shape: (1, 5, 3)
    Number of parameters in network: 21,731
    Number of parameters in latent space: 257
    '''

    # inputs
    input_shape = (1, 28, 20)
    epochs = 1
    batch_size = 1
    beta = 1.0
    filters = 32
    kernel_size = 2
    latent_channels = 1
    pool_size = 2
    
    # define filename
    name = 'cvae_frey_convolutional_latent_space'

    # builder hyperparameter dictionary
    hp_dictionary = {
        'epochs': epochs,
        'batch_size': batch_size,
        'beta': beta,
        'filters': filters,
        'kernel_size': kernel_size,
        'latent_channels': latent_channels,
        'pool_size': pool_size,
        'loss': 'vae_loss',
        'optimizer': 'adam'
    }

    # define log directory
    log_dir = './summaries/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

    # make VAE
    vae = FreyConvolutionalLatentVAE(input_shape, 
                log_dir,
                filters=filters,
                kernel_size=kernel_size,
                latent_channels=latent_channels,
                pool_size=pool_size)
    
    # compile VAE
    from keras import optimizers
    optimizer = optimizers.Adam(lr=1e-3)
    vae.compile(optimizer=optimizer)

    # print summaries
    vae.print_model_summaries()
    
    # get dataset
    (X_train, _), (X_test, _) = utils.load_frey()
    train_generator = utils.make_generator(X_train, batch_size=batch_size)
    test_generator = utils.make_generator(X_test, batch_size=batch_size)
    train_size = len(X_train)
    test_size = len(X_test)

    # fit VAE
    steps_per_epoch = int(train_size / batch_size)
    validation_steps = int(test_size / batch_size)
    vae.fit_generator(train_generator,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch,
                   validation_data=test_generator,
                   validation_steps=validation_steps)


def run_experiments():
    run_dense_latent_space()
    run_convolutional_latent_space_with_same_number_of_parameters()
    run_convolutional_latent_space_with_same_number_of_neurons()


if __name__ == '__main__':
    # run experiments 5 times
    for _ in range(5):
        run_experiments()