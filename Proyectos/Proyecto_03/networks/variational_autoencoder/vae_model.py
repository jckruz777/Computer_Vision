from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras import backend as K

from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE:
    def __init__(self, original_dim, batch_size, epochs, img_width, img_height):
        # network parameters
        self._original_dim = original_dim
        self._input_shape = (original_dim, )
        self._intermediate_dim = 512
        self._batch_size = batch_size
        self._latent_dim = 8
        self._epochs = epochs
        self._img_width = img_width - 2
        self._img_height = img_height - 2

    def build(self):
        chanDim = -1
        # VAE model = encoder + decoder
        # build encoder model
        self._inputs = Input(shape=self._input_shape, name='encoder_input')
        x = Dense(self._intermediate_dim, activation='relu')(self._inputs)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        x = Dense(self._intermediate_dim, activation='relu')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        self._z_mean = Dense(self._latent_dim, name='z_mean')(x)
        self._z_log_var = Dense(self._latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self._latent_dim,), name='z')([self._z_mean, self._z_log_var])

        # instantiate encoder model
        self._encoder = Model(self._inputs, [self._z_mean, self._z_log_var, z], name='encoder')
        #self._encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(self._latent_dim,), name='z_sampling')
        x = Dense(self._intermediate_dim, activation='relu')(latent_inputs)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        x = Dense(self._intermediate_dim, activation='relu')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)
        self._outputs = Dense(self._original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self._decoder = Model(latent_inputs, self._outputs, name='decoder')
        #self._decoder.summary()

        # instantiate VAE model
        self._outputs = self._decoder(self._encoder(self._inputs)[2])
        self._vae = Model(self._inputs, self._outputs, name='vae_mlp')

    def setReconstructionError(self, loss):
        if loss == 'mse':
            self._reconstruction_loss = mse(self._inputs, self._outputs)
        else:
            self._reconstruction_loss = binary_crossentropy(self._inputs,
                                                            self._outputs)

        self._reconstruction_loss *= self._original_dim
        kl_loss = 1 + self._z_log_var - K.square(self._z_mean) - K.exp(self._z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        self._vae_loss = K.mean(self._reconstruction_loss + kl_loss)

    def compile(self):
        self._vae.add_loss(self._vae_loss)
        self._vae.compile(optimizer='adam')
        #self._vae.summary()

    def loadWeights(self, weights):
        self._vae.load_weights(weights)

    def train(self, x_train, x_val):
        # train the autoencoder
        self._history = self._vae.fit(x_train,
                epochs=self._epochs,
                batch_size=self._batch_size,
                validation_data=(x_val, None))
        self._vae.save_weights('vae_mlp.h5')

    def prediction(self, orig):
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._img_width, self._img_height))
        orig = img
        img = img.astype('float32') / 255

        images = np.array([img])
        images = np.reshape(images, [-1, self._img_width*self._img_height])

        rec = self._vae.predict(images)
        rec = rec * 255
        rec = rec.astype('int32')
        rec = np.reshape(rec, [-1, self._img_width, self._img_height, 3])

        loss = self._vae.evaluate(images, verbose=0)
        ssimg = ssim(orig, rec[0], multichannel=True)
        rec_img = rec[0]
        return (loss, ssimg, rec_img)

    def plot(self):
        plot_model(self._encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        plot_model(self._decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        plot_model(self._vae, to_file='vae_mlp.png', show_shapes=True)

        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('vae_mlp_error.png')
        plt.show()
