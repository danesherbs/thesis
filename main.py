import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json


'''
Constants
'''
log_dir = './summaries/cvae12/'


'''
Load models and weights
'''
# load model json file
json_file = open(log_dir + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load model and weights
cvae = model_from_json(loaded_model_json)
cvae.load_weights(log_dir + 'weights.39-0.07.hdf5')


'''
Load data
'''
# X_train, X_test, _, _ = utils.load_data(down_sample=True)

# import dataset
from keras.datasets import mnist
(X_train, _), (X_test, _) = mnist.load_data()

# reshape into (num_samples, num_channels, width, height)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# record input shape
input_shape = X_train.shape[1:]

# cast pixel values to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalise pixel values
X_train /= 255.0
X_test /= 255.0


'''
View test image
'''
# choose sample number of dataset
sample_number = 2

# predict a sample
decoded_imgs = cvae.predict(np.asarray([[X_test[sample_number][0]]]))

# plot actual
plt.figure(1)
plt.title('Original')
plt.imshow(X_test[sample_number][0])
plt.gray()

# plot predicted
plt.figure(2)
plt.title('Reconstructed')
plt.imshow(decoded_imgs[0][0])
plt.gray()

# show both at same time
plt.show()