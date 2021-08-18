import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
################################################################################
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
###############################################################################
class VariationalAE:
    """ VAE for outlier detection using tf.keras """
    ############################################################################
    def __init__(self,
        input_dimensions:'int',
        encoder_units:'list', latent_dimensions:'int', decoder_units:'list',
        batch_size:'int', epochs:'int', learning_rate:'float')->'None':
        """
        Creates a variational auto encoder

        INPUTS
            input_dimensions:
            encoder_units: python list containing the number of units in
                each layer of the encoder
            latent_dimensions: number of  dimensions for the latent
                representation
            decoder_units: python list containing the number of units in
                each layer of the decoder
            batch_size: number of batches for the training set
            epochs: maximum number of epochs to train the algorithm
            learning_rate: value for the learning rate
        """

        self.input_dimensions = input_dimensions

        self.input_layer = Input(
            shape=(self.input_dimensions,),
            name='input_layer')

        self.encoder_units = encoder_units
        self.latent_dimensions = latent_dimensions
        self.decoder_units = decoder_units

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.vae = self.build_vae()
    ############################################################################
    def build_vae(self)->'keras.model':
        """
        Builds and returns a compiled variational auto encoder using keras API
        """

        output = self.decoder(self.encoder(self.input_layer)[2])

        vae = Model(self.input_layer, output,
            name='variational_auto_encoder')

        adam_optimizer = Adam(learning_rate=self.learning_rate)

        loss = self.standard_loss
        vae.compile(loss=loss , optimizer=adam_optimizer,
            metrics=['accuracy'])

        return vae
    ############################################################################
    def standard_loss(self, y_true, y_pred)->'keras.custom_loss':
        """
        Standard loss function for the variational auto encoder, that is,
        mean squared error for de reconstruction loss and the KL divergence.

        INPUTS
            y_true : input datapoint
            y_pred : predicted value by the network

        OUTPUT
            loss function with the keras format that is compatible with the
            .compile API
        """

        mse = MeanSquaredError(y_true, y_pred)
        kl = KLDivergence(y_true, y_pred)

        return K.sum(mse, kl)
    ############################################################################
    def build_encoder(self)-> 'keras.model':
        """
        Builds and returns a compiled variational encoder using keras API
        """
        X = self.input_layer
        standard_deviation = np.sqrt(2. / self.input_dimensions)

        for idx, units in enumerate(self.encoder_units):

            initial_weights = tf.keras.initializers.RandomNormal(
                mean=0.,
                stddev=standard_deviation)

            layer = Dense(units, activation='relu',
                kernel_initializer=initial_weights,
                name=f'encoder_{idx+1}')(X)

            X = layer

            standard_deviation = np.sqrt(2. / units)

            if units == self.encoder_units[-1]:

                z_mean = Dense(self.latent_dimensions, name='z_mean',
                    kernel_initializer=initial_weights)(X)

                z_log_sigma = Dense(self.latent_dimensions,
                    name='z_log_sigma',
                    kernel_initializer=initial_weights)(X)

                z = Lambda(self.sampling)([z_mean, z_log_sigma])

        encoder = Model(self.input_layer, [z_mean, z_log_sigma, z],
            name='variational_encoder')

        return encoder
    ############################################################################
    def sampling(self, arguments:'list')->'keras.custom_function':
        """
        function to sample into the latent representation

        INPUTS
            arguments: list containing z_mean and z_log_sigma.
                Better to have in this wayfor the keras.lambda layer

        OUTPUT
            z : sampled latent variable
        """

        z_mean, z_log_sigma = arguments

        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], self.latent_dimensions),
            mean=0., stddev=0.1)

        z = z_mean + K.exp(z_log_sigma) * epsilon

        return z
    ############################################################################
    def build_decoder(self)->'keras.model':
        """
        Builds and returns a compiled decoder using keras API
        """
        decoder_input = Input(shape=(self.latent_dimensions,),
            name='z_sampling')

        standard_deviation = np.sqrt(2. / self.latent_dimensions)

        X = decoder_input

        for idx, units in enumerate(self.decoder_units):

            initialization_weights = tf.keras.initializers.RandomNormal(
                mean=0., stddev=standard_deviation)

            layer = Dense(units, activation='relu',
                kernel_initializer=initialization_weights,
                name=f'decoder_{idx+1}')(X)

            X = layer
            standard_deviation = np.sqrt(2./units)

            if units == self.decoder_units[-1]:
                decoder_output = self.decoder_output_layer(
                    standard_deviation, X)

        decoder = Model(decoder_input, decoder_output,
            name='variational_decoder')

        return decoder
    ###########################################################################
    def decoder_output_layer(self,
        standard_deviation:'float',
        X:'keras.layer')->'keras.Dense':
        """
        Auxiliary method to create a decoder output layer with a linear
        activation function to account for high emission lines
        think how to use it to my advantage

        INPUTS
            standard_deviation: For the purpose of initialization of the weights
            X: previous hidden layer to the output of the decoder

        OUTPUTS
            output_layer: keras.Dense.Layer, the output layer
        """
        initialization_weights = tf.keras.initializers.RandomNormal(
            mean=0., stddev=standard_deviation)

        output_layer = Dense(self.input_dimensions, name='decoder_output',
        kernel_initializer=initialization_weights)(X)

        return output_layer
    ############################################################################
    def custom_loss(self):
        pass
    ############################################################################
    def fit(self, spectra:'2D np.array')-> 'None':

        self.vae.fit(x=spectra, y=spectra, epochs=self.epochs,
            batch_size=self.batch_size, verbose=2)
    ############################################################################
    def predict(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.vae.predict(spectra)
    ############################################################################
    def encode(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        return self.encoder(spectra)
    ############################################################################
    def decode(self, coding:'2D np.array')->'2D np.aray':

        if coding.ndim==1:
            coding = coding.reshape(1,-1)

        return self.decoder(coding)
    ############################################################################
    def save_vae(self, fpath:'str'):

        self.vae.save(f'{fpath}')
    # ############################################################################
    def save_encoder(self, fpath:'str'):

        self.encoder.save(f'{fpath}')
    ############################################################################
    def save_decoder(self, fpath:'str'):

        self.decoder.save(f'{fpath}')
    ############################################################################
    def plot_model(self):

        plot_model(self.vae, to_file='DenseVAE.png', show_shapes='True')
        plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
        plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
    # ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.vae.summary()
###############################################################################
