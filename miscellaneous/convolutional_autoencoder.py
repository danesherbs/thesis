from keras.models import Sequential
from keras.layers import Convolution2D, Deconvolution2D, Reshape
from keras.datasets import mnist
from keras import backend as K
from keras.utils import np_utils


'''
Define parameters
'''
batch_size = 128
nb_epoch = 10
input_w = 28
input_h = 28
nb_filters = 32
pool_size = (2, 2)
kernal_size = (3, 3)


'''
Load data
'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, input_h, input_w)
    X_test = X_test.reshape(X_test.shape[0], 1, input_h, input_w)
    input_shape = (1, input_h, input_w)
else:
    X_train = X_train.reshape(X_train.shape[0], input_h, input_w, 1)
    X_test = X_test.reshape(X_test.shape[0], input_h, input_w, 1)
    input_shape = (input_h, input_w, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train[::10]
X_test  = X_test[::10]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


'''
Define model
'''
model = Sequential()

# 3x3 convolution with 'nb_filters' output filters and stride 1x1 on a 28x28 image
model.add(Convolution2D(nb_filters, kernal_size[0], kernal_size[1], border_mode='valid', input_shape=input_shape))

# 3x3 convolution on top, with 16 output filters
model.add(Convolution2D(8, kernal_size[0], kernal_size[1], border_mode='valid'))

# 3x3 transposed convolution with 32 output filters and stride 1x1 on a 16 24x24 images
model.add(Deconvolution2D(32, kernal_size[0], kernal_size[1], output_shape=(None, 32, 22, 22), border_mode='valid', input_shape=(8, 16, 16)))

# 3x3 transposed convolution with 1 output filter and stride 1x1 on a 32 26x26 images
model.add(Deconvolution2D(1, kernal_size[0], kernal_size[1], output_shape=(None, 1, 28, 28), border_mode='valid', input_shape=(32, 22, 22)))

model.summary()


'''
Compile and fit
'''
model.compile(loss='binary_crossentropy', optimizer='adadelta')
model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, X_test))


'''
Compare reconstruction
'''
decoded_imgs = model.predict(X_test)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(X_test[1][1])
plt.figure()
plt.imshow(decoded_imgs[1][1])
plt.show()