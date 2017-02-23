from keras.models import load_model
import utils


'''
Load models and weights
'''
vae = load_model('saved_models/cvae_epoch=10_beta=1_latent=1.h5')
vae.load_weights('saved_models/cvae_epoch=10_beta=1_latent=1_weights.h5')

encoder = load_model('saved_models/encoder_epoch=10_beta=1_latent=16.h5')
encoder.load_weights('saved_models/encoder_epoch=10_beta=1_latent=16_weights.h5')


'''
Load data
'''
X_train, X_test, _, _ = utils.load_data(down_sample=True)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
input_shape = X_train.shape[1:]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


'''
Main
'''
latent_imgs = encoder.predict(X_test)
decoded_imgs = vae.predict(X_test)

import matplotlib.pyplot as plt
plt.matshow(X_test[0][0])
plt.title('Original image')
# utils.show_subplot(latent_imgs[0])
plt.matshow(latent_imgs[0][0])
plt.title('Latent feature maps')
plt.matshow(decoded_imgs[0][0])
plt.title('Reconstructed image')
plt.show()