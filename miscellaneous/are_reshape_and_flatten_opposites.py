import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist



'''



Yes, they are.




'''



'''
Define parameters
'''
batch_size = 128
nb_epoch = 10
input_w = 28
input_h = 28


'''
Load data
'''
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, input_h, input_w)
X_test = X_test.reshape(X_test.shape[0], 1, input_h, input_w)
input_shape = (1, input_h, input_w)

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
input_img = Input(shape=input_shape)
x = Flatten()(input_img)
output_img = Reshape(input_shape)(x)
model = Model(input_img, output_img)

model.summary()


'''
Compile and fit
'''
model.compile(loss='binary_crossentropy', optimizer='adadelta')
model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch)


'''
Compare reconstruction
'''
decoded_imgs = model.predict(X_test)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(X_test[0][0])
plt.figure()
plt.imshow(decoded_imgs[0][0])
plt.show()