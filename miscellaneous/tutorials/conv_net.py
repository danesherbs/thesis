# set random seed for preproducability
import numpy as np
np.random.seed(3141)

# use Theano convention for dimension ordering
from keras import backend
backend.set_image_dim_ordering('th')

# linear stack of neural network layers
from keras.models import Sequential

# layers that are used in almost any neural network
from keras.layers import Dense, Dropout, Activation, Flatten

# convolutional layers that will help us efficiently train on image data
from keras.layers import Convolution2D, MaxPooling2D

# help us transform our data later
from keras.utils import np_utils

# MNIST dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# should print (60000, 28, 28)
# 60000 samples, each 28x28
# print X_train.shape


# pre-processing for Keras:
#    - specify depth of image for Theano
#    - cast to float32 
#    - normalise values
#    - make class labels categorical

# specify depth of image for Theano
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# cast to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise values
X_train /= np.max(X_train)
X_test /= np.max(X_test)

# make class labels categorical
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)



# start stacking layers
model = Sequential()

# 32 kernels, each 3x3
# the step size is (1,1) by default, and it can be tuned using the 'subsample' parameter
print X_train.shape
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))


# now add layers like building legos
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# add fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# print summary of model
print model.summary()

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fit to data
model.fit(X_train, y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

