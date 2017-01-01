# use Theano convention for dimension ordering
from keras import backend
backend.set_image_dim_ordering('th')

# import model and layers
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# load data
from utils import load_data
X_train, X_test, _, _ = load_data()

# set seed for reproducibility
import numpy as np
np.random.seed(123)

# specify depth for Theano
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# cast to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise values
X_train /= np.max(X_train)
X_test /= np.max(X_test)

# define architecture
input_img = Input(shape=X_train.shape[1:])

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=X_train.shape[1:])(input_img)
encoded = MaxPooling2D((2, 2), border_mode='same', name='encoded')(x)

x = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(encoded)
decoded = UpSampling2D((2, 2), name='decoded')(x)

# define models
autoencoder = Model(input_img, decoded)
encoder = Model(autoencoder.input, autoencoder.get_layer('encoded').output)

# # print summary of model (debugging)
# print autoencoder.summary()
# print encoder.summary()

# choose loss function and optimiser
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# fit model on training data
autoencoder.fit(X_train, X_train, nb_epoch=10, verbose=1)
 
# evaluate model on test data
score = autoencoder.evaluate(X_test, X_test, verbose=0)	

# use matplotlib
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

# plot encoded images
from utils import show_subplot
show_subplot(encoded_imgs[1])

# plot decoded image
from matplotlib import pyplot as plt
plt.imshow(decoded_imgs[1][0])
plt.show()