from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from constants_AEs import sdss_data_proc

class VAE:
    """ VAE for outlier detection using tf.keras

    References:
    Portillo et al. 2020
    """

    def __init__(self,
    in_dim=1_000,
    lat_dim=10,
    hid_dim=[549, 110, 10, 110, 549],
    batch_size=32,
    epochs=10,
    lr= 1e-4):

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.lat_dim = lat_dim
        self.epochs = epochs
        self.lr = lr
        self.encoder = None
        self.decoder = None
        self.VAE = None
        self._create_VAE()

        #
        # def _init_VAE(self):
        #     # blabla
        #
        #     # Create AE
        #     self._create_VAE()

    def _create_VAE(self):

        # Build Encoder
        inputs = Input(shape=(self.in_dim,), name='encoder_input')

        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.in_dim))
        hidden_0 = Dense(self.hid_dim[0], name='hidden_0', activation='relu',
        kernel_initializer=w_init)(inputs)

        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.hid_dim[0]))
        hidden_1 = Dense(self.hid_dim[1], name='hidden_1', activation='relu',
        kernel_initializer=w_init)(hidden_0)

        # Stocastic layer
        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.hid_dim[1]))
        latent_mu = Dense(self.hid_dim[2], name='latent_mu',
        kernel_initializer=w_init)(hidden_1)
        latent_ln_sigma = Dense(self.hid_dim[2], name='latent_ln_sigma',
        kernel_initializer=w_init)(hidden_1)

        latent = Lambda(self._sample_latent_features,
        output_shape=(self.hid_dim[2],),
        name='latent')([latent_mu, latent_ln_sigma])


        self.encoder = Model(inputs, latent, name='encoder')
        self.encoder.summary()
        # plot_model(self.encoder, to_file='encoder.png', show_shapes='True')

        # Build Decoder
        latent_in = Input(shape=(self.hid_dim[2],), name='decoder_input')

        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.hid_dim[2]))
        hidden_3 = Dense(self.hid_dim[3], name='hidden_3', activation='relu',
        kernel_initializer=w_init)(latent_in)

        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.hid_dim[3]))
        hidden_4 = Dense(self.hid_dim[4], name='hidden_4', activation='relu',
        kernel_initializer=w_init)(hidden_3)

        w_init = keras.initializers.RandomNormal(mean=0.,
        stddev=np.sqrt(2./self.hid_dim[4]))
        outputs = Dense(self.in_dim, name='decoder_output',
        kernel_initializer=w_init)(hidden_4)

        self.decoder = Model(latent_in, outputs, name='decoder')
        self.decoder.summary()
        # plot_model(self.decoder, to_file='decoder.png', show_shapes='True')

        # VAE = Encoder + Decoder
        vae = Model(inputs, self.decoder(self.encoder
                            (inputs)), name='VAE')
        vae.summary()
        # plot_model(vae, to_file='VAE.png', show_shapes=True)

        # Mean square error loss function with Adam optimizer
        # loss = self._vae_loss(z_m=latent_mu, z_s=latent_ln_sigma,
        # y_true=inputs, y_pred=outputs)
        loss = self._vae_loss(z_m=latent_mu, z_s=latent_ln_sigma)
        vae.compile(loss=loss, optimizer='adam') #, lr = self.lr)

        self.VAE = vae

    def _sample_latent_features(self, distribution):

        z_m, z_s = distribution
        batch = K.shape(z_m)[0]
        dim = K.int_shape(z_m)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_m + K.exp(0.5*z_s)*epsilon

    # def _vae_loss(self, z_m, z_s, y_true, y_pred):
    def _vae_loss(self, z_m, z_s):

        # vae_loss = self._reconstruction_loss(y_true, y_pred) +\
        # self._kl_loss(z_m, z_s)
        def rec_loss(y_true, y_pred):

            return keras.losses.mse(y_true, y_pred)

        def kl_loss(self, z_m, z_s):

            kl_loss = 1 + z_s - K.square(z_m) - K.exp(z_s)

            return -0.5*K.sum(kl_loss, axis=-1)

        def vae_loss(y_true, y_pred):

            kl_loss = kl_loss(z_m, z_s)
            rec_loss = rec_loss(y_true, y_pred)
            return K.mean(kl_loss + rec_loss)

        # return K.mean(vae_loss)
        return vae_loss

class AEpca:

    def __init__(self, in_dim, lat_dim=2, batch_size=32, epochs=10, lr= 1e-4):
        self.in_dim = in_dim
        self.batch_size = batch_size
        self.lat_dim = lat_dim
        self.epochs = epochs
        self.lr = lr
        self.encoder = None
        self.decoder = None
        self.AE = None
        self._init_AE()

    def _init_AE(self):

        # Build Encoder
        inputs = Input(shape=(self.in_dim,), name='encoder_input')
        latent = Dense(self.lat_dim, name='latent_vector')(inputs)
        self.encoder = Model(inputs, latent, name='encoder')
        self.encoder.summary()
#        plot_model(self.encoder, to_file='encoder.png', show_shapes='True')

        # Build Decoder
        latent_in = Input(shape=(self.lat_dim,), name='decoder_input')
        outputs = Dense(self.in_dim, name='decoder_output')(latent_in)
        self.decoder = Model(latent_in, outputs, name='decoder')
        self.decoder.summary()
#        plot_model(self.decoder, to_file='decoder.png', show_shapes='True')

        # AE = Encoder + Decoder
        autoencoder = Model(inputs, self.decoder(self.encoder
                            (inputs)), name='autoencoder')
        autoencoder.summary()
#        plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

        # Mean square error loss function with Adam optimizer
        autoencoder.compile(loss='mse', optimizer='adam') #, lr = self.lr)

        self.AE = autoencoder

    def fit(self, spectra):
        self.AE.fit(spectra, spectra, epochs=self.epochs,
                    batch_size=self.batch_size)

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)
        return self.AE.predict(test_spec)

    def encode(self, spec):
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        return self.encoder(spec)

    def decode(self, lat_val):
        return self.decoder(lat_val)

    def save(self):
        self.encoder.save('encoder')
        self.decoder.save('decoder')
        self.AE.save('AutoEncoder')

class Outlier:

    def __init__(self, dbPath=sdss_data_proc, N=20):
        self.scores = None
        self.N = N
        self.fnames = glob(f'{dbPath}/*')

    def chi2(self, O, P):
        self.scores = (np.square(P-O)*(1/np.abs(P))).mean(axis=1)
        np.save('chi2_outlier_scores.npy', self.scores)
        self.retrieve(metric='chi2')

    def mse(self, O, P):
        self.scores = np.square(P-O).mean(axis=1)
        np.save('mse_outlier_scores.npy', self.scores)
        self.retrieve(metric='mse')


    def mad(self, O, P):
        self.scores = np.abs(P-O).mean(axis=1)
        np.save('mad_outlier_scores.npy', self.scores)
        self.retrieve(metric='mad')

    def lp(self, O, P, p=1):
        self.scores = (np.sum((np.abs(P-O))**p, axis=1))**(1/p)
        np.save(f'lp_{p}_outlier_scores.npy', self.scores)
        self.retrieve(metric=f'lp_{p}')

    def area(self, O, P):
        self.scores = np.trapz(np.square(P-O), axis=1)
        np.save('area_outlier_scores.npy', self.scores)
        self.retrieve(metric=f'area')

    def lpf(self, O, P, p=1):
        self.scores = (np.trapz((np.abs(P-O))**p, axis=1))**(1/p)
        np.save(f'lpf_{p}_outlier_scores.npy', self.scores)
        self.retrieve(metric=f'lpf_{p}')


    def retrieve(self, metric=None, fraction=1):
        print('Loading outlier scores!')

        ids_proc_specs = np.argpartition(self.scores, -1*int(self.N*fraction))[-1*int(self.N*fraction):]
        np.save(f'outlier_ids_{metric}', ids_proc_specs)
        print('Outlier scores for the weirdest spectra')

        names = open(f'{metric}_fnames.txt', 'w+')

        for n, idx in enumerate(ids_proc_specs):
            print(f'ID:{idx} --> {self.scores[idx]}. File name: {self.fnames[idx].split("/")[-1]}', end='\r')
            names.write(f'{self.fnames[idx].split("/")[-1][:-4]}\n')
        names.close()

class PcA:

    def __init__(self, n_comps = False):
        if not(n_comps):
            self.PCA = PCA()
        else:
            self.n_comps = n_comps
            self.PCA = PCA(self.n_comps)

    def fit(self, spec):
        return self.PCA.fit(spec)

    def components(self):
        return self.PCA.components_

    def expvar(self):
        return self.PCA.explained_variance_ratio_

    def inverse(self, trf_spec):
        if trf_spec.ndim == 1:
            trf_spec = trf_spec.reshape(1, -1)

        return self.PCA.inverse_transform(trf_spec)

    def predict(self, test_spec):
        if test_spec.ndim == 1:
            test_spec = test_spec.reshape(1, -1)

        return self.PCA.transform(test_spec)

def plt_spec_pca(flx,pca_flx,componets):
    '''Comparative plot to see how efficient is the PCA compression'''
    plt.figure(figsize=(8,4));

    # Original Image
    plt.subplot(1, 2, 1);
    plt.plot(flx)
    plt.xlabel(f'{flx.size} components', fontsize = 14)
    plt.title('Original Spectra', fontsize = 20)

    # principal components
    plt.subplot(1, 2, 2);
    plt.plot(pca_flx)
    plt.xlabel(f'{componets} componets', fontsize = 14)
    plt.title('Reconstructed spectra', fontsize = 20)
    plt.show()
    plt.close()

def plot_2D(data, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(data[:,0], data[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.savefig(f'{title}.png')
    plt.show()
    plt.close()
