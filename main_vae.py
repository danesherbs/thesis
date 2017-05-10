from vae import VAE
import utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
Initialise VAE
'''
vae = VAE()

'''
Load data
'''
# import dataset
(X_train, _), (X_test, _) = utils.load_data()

# reshape into (num_samples, num_channels, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# record input shape
input_shape = X_train.shape[1:]

# cast pixel values to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise pixel values
X_train = (X_train - np.min(X_train)) / np.max(X_train)
X_test = (X_test - np.min(X_test)) / np.max(X_test)

# print data information
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# initialise data generator
train_generator = ImageDataGenerator()
train_generator.fit(X_train)
test_generator = ImageDataGenerator()
test_generator.fit(X_test)


'''
Fit VAE to data
'''
vae.fit(X_train)