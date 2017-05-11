# inputs
input_shape = (1, 28, 28)
epochs = 1
batch_size = 1
filters = 32
latent_filters = 1
kernel_size = 3
pool_size = 2
beta = 1.0

# define filename
name = 'cvae_convolutional_latent'

# builder hyperparameter dictionary
hp_dictionary = {
    'epochs': epochs,
    'batch_size': batch_size,
    'filters': filters,
    'latent_filters': latent_filters,
    'kernel_size': kernel_size,
    'pool_size': pool_size,
    'loss': 'vae_loss',
    'optimizer': 'adam',
    'beta': beta
}

# define log directory
log_dir = './summaries/experiment_convolutional_latent_space/' + utils.build_hyperparameter_string(name, hp_dictionary) + '/'

# make VAE
vae = ConvolutionalLatentDeepVAE(input_shape, 
            log_dir,
            filters=filters,
            latent_filters=latent_filters,
            kernel_size=kernel_size,
            pool_size=pool_size)


# compile VAE
from keras import optimizers
optimizer = optimizers.Adam(lr=1e-3)
vae.compile(optimizer=optimizer)

# get dataset
(X_train, _), (X_test, _) = utils.load_mnist()
train_generator = utils.make_generator(X_train, batch_size=batch_size)
test_generator = utils.make_generator(X_test, batch_size=batch_size)
train_size = len(X_train)
test_size = len(X_test)

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