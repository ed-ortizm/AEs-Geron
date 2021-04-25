import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
################################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
###############################################################################
class Base36:
    ############################################################################
    def decode(self, sub_class:'str'):

        if ' ' in sub_class:
            sub_class = sub_class.replace(' ', '')

        elif sub_class == '':
            sub_class = 'EMPCLS'

        return int(sub_class, 36)
    ############################################################################
    def encode(self, sub_class:'int'):

        alphabet, base36 = ['0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', '']

        sub_class = abs(int(sub_class))

        while sub_class:
            sub_class, i = divmod(sub_class, 36)
            base36 = alphabet[i] + base36

        return base36 or alphabet[0]
    ###########################################################################################################################################################
def plot_history(history:'tf.keras.callback.History', save_to:'str'):

    for key, value in history.history.items():

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(value)

        ax.set_title(f'Model {key}')
        ax.set_xlabel(f'epochs')
        ax.set_ylabel(f'{key}')

        fig.savefig(f'{save_to}_{key}.png')
        plt.close()

        np.save(f'{save_to}_{key}.npy', value)
###############################################################################
def load_data(file_name, file_path):

    if os.path.exists(file_path):

        print(f'Loading: {file_name}')

        return np.load(f'{file_path}')

    else:
        print(f'There is no file: {file_name}')
        sys.exit()
###############################################################################
class LoadAE:
    """ Load AE for outlier detection using tf.keras """
    ############################################################################
    def __init__(self, ae_path, encoder_path, decoder_path)->'None':

        self.ae = keras.models.load_model(f'{ae_path}')
        self.encoder = keras.models.load_model(f'{encoder_path}')
        self.decoder = keras.models.load_model(f'{decoder_path}')
    ############################################################################
    def predict(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.ae.predict(spectra)
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
    def plot_model(self):

        plot_model(self.ae, to_file='DenseVAE.png', show_shapes='True')
        plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
        plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
    ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.ae.summary()
###############################################################################
class AEDense:
    """ VAE for outlier detection using tf.keras """
    ############################################################################
    def __init__(self, n_input_dimensions:'int', n_layers_encoder: 'list',
        n_latent_dimensions:'int', n_layers_decoder: 'list', batch_size:'int',
        epochs:'int', learning_rate:'float', loss:'str')->'None':


        self.n_input_dimensions = n_input_dimensions

        self.n_layers_encoder = n_layers_encoder
        self.n_latent_dimensions = n_latent_dimensions
        self.n_layers_decoder = n_layers_decoder
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss

        self.inputs = Input(shape=(self.n_input_dimensions,),
                            name='ae_input_layer')

        self.encoder = self.build_encoder()

        self.decoder = self.build_decoder()



        self.learning_rate = learning_rate
        self.ae = self.build_ae()
    ############################################################################
    def build_ae(self):

        ae = Model(self.inputs, self.decoder(self.encoder(self.inputs)),
            name='DenseAE')

        adam_optimizer = Adam(learning_rate=self.learning_rate)

        ae.compile(loss=self.loss, optimizer=adam_optimizer,
            metrics=['accuracy'])

        return ae
    ############################################################################
    def build_encoder(self)->'tf.keras.model':

        X = self.inputs
        std_dev = np.sqrt(2. / self.n_input_dimensions)

        for idx, n_units in enumerate(self.n_layers_encoder):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

            layer = Dense(n_units, name=f'encoder_layer_{idx+1}',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            std_dev = np.sqrt(2. / n_units)

            if n_units == self.n_layers_encoder[-1]:
                latent_layer = self._latent_layer(n_units, X)

        encoder = Model(self.inputs, latent_layer, name='DenseEncoder')

        return encoder
    ###########################################################################
    def _latent_layer(self, n_units:'int', X:'tf.keras.Dense')->'tf.keras.Dense':

        std_dev = np.sqrt(2./n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        latent_layer = Dense(self.n_latent_dimensions, name='latent_layer',
        activation='relu', kernel_initializer=w_init)(X)

        return latent_layer
    ###########################################################################
    def build_decoder(self)->'tf.keras.model':

        input_decoder = Input(shape=(self.n_latent_dimensions,),
            name='decoder_input'
        )

        std_dev = np.sqrt(2. / self.n_latent_dimensions)

        X = input_decoder

        for idx, n_units in enumerate(self.n_layers_decoder):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)
            std_dev = np.sqrt(2./n_units)

            layer = Dense(n_units, name=f'layer_{idx+1}_decoder',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            if n_units == self.n_layers_decoder[-1]:
                output_layer = self._output_layer(n_units, X)

        decoder = Model(input_decoder, output_layer, name='DenseDecoder')

        return decoder
    ###########################################################################
    def _output_layer(self, n_units:'int', X:'tf.keras.Dense')->'tf.keras.Dense':

        std_dev = np.sqrt(2./n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        output_layer = Dense(self.n_input_dimensions, name='decoder_output',
        kernel_initializer=w_init)(X)

        return output_layer
    ############################################################################
    def fit(self, spectra:'2D np.array')-> 'tf.keras.callback.History':

        return self.ae.fit(x=spectra, y=spectra, epochs=self.epochs,
            batch_size=self.batch_size, shuffle=True, verbose=2)
    ############################################################################
    def save_ae(self, fpath:'str'):

        self.ae.save(f'{fpath}')
    ############################################################################
    def save_encoder(self, fpath:'str'):

        self.encoder.save(f'{fpath}')
    ############################################################################
    def save_decoder(self, fpath:'str'):

        self.decoder.save(f'{fpath}')
    ############################################################################
    def plot_model(self):

        plot_model(self.ae, to_file='DenseVAE.png', show_shapes='True')
        plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
        plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
    ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.ae.summary()
    ############################################################################
    def predict(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.ae.predict(spectra)
###############################################################################
class VAEDense:
    """ VAE for outlier detection using tf.keras """
    ############################################################################
    def __init__(self, n_input_dimensions:'int', n_layers_encoder: 'list',
        n_latent_dimensions:'int', n_layers_decoder: 'list', batch_size:'int',
        epochs:'int', learning_rate:'float')->'None':


        self.n_input_dimensions = n_input_dimensions

        self.n_layers_encoder = n_layers_encoder
        self.n_latent_dimensions = n_latent_dimensions
        self.n_layers_decoder = n_layers_decoder
        self.batch_size = batch_size
        self.epochs = epochs

        self.inputs = Input(shape=(self.n_input_dimensions,),
                            name='vae_input_layer')

        self.latent_mu = None
        self.latent_ln_sigma = None

        self.encoder = self.build_encoder()

        self.decoder = self.build_decoder()


        self.loss = self.vae_loss()

        self.learning_rate = learning_rate
        self.vae = self.build_vae()
    ############################################################################
    def build_vae(self):

        vae = Model(self.inputs, self.decoder(self.encoder(self.inputs)),
            name='DenseVAE')
        adam_optimizer = Adam(learning_rate=self.learning_rate)
#        vae.compile(loss='mse', optimizer=adam_optimizer)
        vae.compile(loss=self.loss, optimizer=adam_optimizer)
        return vae
    ############################################################################
    def build_encoder(self)->'None':

        X = self.inputs
        std_dev = np.sqrt(2. / self.n_input_dimensions)

        for idx, n_units in enumerate(self.n_layers_encoder):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

            layer = Dense(n_units, name=f'encoder_layer_{idx+1}',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            std_dev = np.sqrt(2. / n_units)

            if n_units == self.n_layers_encoder[-1]:
                latent = self._stochastic_layer(n_units, X)

        encoder = Model(self.inputs, latent, name='DenseEncoder')

        return encoder
    ############################################################################
    def _stochastic_layer(self, n_units:'int', X:'tf.keras.Dense')->'tf.keras.Lambda':

        std_dev = np.sqrt(2. / n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        self.latent_mu = Dense(self.n_latent_dimensions, name='latent_mu',
            kernel_initializer=w_init)(X)

        self.latent_ln_sigma = Dense(self.n_latent_dimensions,
            name='latent_ln_sigma')(X) #, kernel_initializer=w_init)(X)

        ########################################################################
        # def _sample_latent_features(self, distribution):
        def _sample_latent_features(distribution):

            z_m, z_s = distribution
            batch = K.shape(z_m)[0]
            dim = K.int_shape(z_m)[1]
            epsilon = K.random_normal(shape=(batch, dim))

            return z_m + K.exp(0.5 * z_s) * epsilon
        ########################################################################
        latent = Lambda(_sample_latent_features,
                        output_shape=(self.n_latent_dimensions,),
                        name='latent')([self.latent_mu, self.latent_ln_sigma])

        return latent
    # ###########################################################################
    def build_decoder(self)->'None':

        input_layer = Input(shape=(self.n_latent_dimensions,),
            name='decoder_input'
        )

        std_dev = np.sqrt(2. / self.n_latent_dimensions)

        X = input_layer

        for idx, n_units in enumerate(self.n_layers_decoder):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)
            std_dev = np.sqrt(2./n_units)

            layer = Dense(n_units, name=f'layer_{idx+1}_decoder',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            if n_units == self.n_layers_decoder[-1]:
                output_layer = self._output_layer(n_units, X)

        decoder = Model(input_layer, output_layer, name='DenseDecoder')

        return decoder
    ###########################################################################
    def _output_layer(self, n_units, X)->'tf.keras.Dense':

        std_dev = np.sqrt(2./n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        output_layer = Dense(self.n_input_dimensions, name='decoder_output',
        activation='relu', kernel_initializer=w_init)(X)

        return output_layer
    ############################################################################
    def vae_loss(self):
        return self.vae_loss_aux
    ############################################################################
    def vae_loss_aux(self, y_true, y_pred):

        kl_loss = self.kl_loss()
        rec_loss = self.rec_loss(y_true, y_pred)
        return K.mean(kl_loss + rec_loss)
    ############################################################################
    def kl_loss(self):

        z_m = self.latent_mu
        z_s = self.latent_ln_sigma

        kl_loss = 1 + z_s - K.square(z_m) - K.exp(z_s)

        return -0.5 * K.sum(kl_loss, axis=-1)
    ############################################################################
    def rec_loss(self, y_true, y_pred):

        return keras.losses.mse(y_true, y_pred)
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
class DenseDecoder:

    ###########################################################################
    def __init__(self, n_latent_dimensions: 'int', n_output_dimensions: 'int',
                 n_hiden_layers: 'list') -> 'None':

        self.n_latent_dimensions = n_latent_dimensions
        self.n_output_dimensions = n_output_dimensions
        self.n_hiden_layers = n_hiden_layers

        self.decoder = self.build_decoder()
    ############################################################################

    def plot_model(self):

        plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
    ###########################################################################

    def summary(self):

        self.decoder.summary()
    ###########################################################################

    def build_decoder(self):

        input_layer = Input(shape=(self.n_latent_dimensions,), name='decoder_input')
        std_dev = np.sqrt(2. / self.n_latent_dimensions)

        n_layers = len(self.n_hiden_layers)
        X = input_layer
        for idx, n_units in enumerate(self.n_hiden_layers):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)
            std_dev = np.sqrt(2. / n_units)

            layer = Dense(n_units, name=f'layer_{idx+1}_decoder',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            if n_units == self.n_hiden_layers[-1]:
                output_layer = self._output_layer(n_units, X)

        decoder = Model(input_layer, output_layer, name='DenseDecoder')

        return decoder
    ###########################################################################
    def _output_layer(self, n_units, X):

        std_dev = np.sqrt(2. / n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        output_layer = Dense(self.n_output_dimensions, name='decoder_output',
                             kernel_initializer=w_init)(X)

        return output_layer
################################################################################
class DenseEncoder:

    ###########################################################################
    def __init__(self, n_input_dimensions: 'int', n_hiden_layers: 'list',
                 n_latent_dimensions: 'int') -> 'None':

        self.n_input_dimensions = n_input_dimensions
        self.n_hiden_layers = n_hiden_layers
        self.n_latent_dimensions = n_latent_dimensions

        self.inputs = Input(shape=(self.n_input_dimensions,),
                            name='encoder_input_layer')

        self.latent_mu = None
        self.latent_ln_sigma = None

        self.encoder = self.build_encoder()
    ############################################################################

    def plot_model(self):

        plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
    ###########################################################################

    def summary(self):

        self.encoder.summary()
    ###########################################################################

    def build_encoder(self):

        X = self.inputs
        std_dev = np.sqrt(2. / self.n_input_dimensions)

        for idx, n_units in enumerate(self.n_hiden_layers):

            w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

            layer = Dense(n_units, name=f'encoder_layer_{idx+1}',
                          activation='relu', kernel_initializer=w_init)(X)

            X = layer

            std_dev = np.sqrt(2. / n_units)

            if n_units == self.n_hiden_layers[-1]:
                latent = self.stochastic_layer(n_units, X)

        encoder = Model(self.inputs, latent, name='DenseEncoder')

        return encoder

    ###########################################################################
    def stochastic_layer(self, n_units, X):

        std_dev = np.sqrt(2. / n_units)

        w_init = keras.initializers.RandomNormal(mean=0., stddev=std_dev)

        self.latent_mu = Dense(self.n_latent_dimensions, name='latent_mu',
                               kernel_initializer=w_init)(X)

        self.latent_ln_sigma = Dense(self.n_latent_dimensions,
                                     name='latent_ln_sigma', kernel_initializer=w_init)(X)

        latent = Lambda(self._sample_latent_features,
                        output_shape=(self.n_latent_dimensions,),
                        name='latent')([self.latent_mu, self.latent_ln_sigma])

        return latent
    ###########################################################################
    def _sample_latent_features(self, distribution):

        z_m, z_s = distribution
        batch = K.shape(z_m)[0]
        dim = K.int_shape(z_m)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_m + K.exp(0.5 * z_s) * epsilon
################################################################################
