#!/usr/bin/env python3.8
from configparser import ConfigParser, ExtendedInterpolation
import os
import time

import numpy as np

from ae_library import VariationalAE
###############################################################################
ti = time.time()
###############################################################################
# configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('vae.ini')
############################################################################
# Relevant directories
data_directory = parser.get('directories', 'train')
output_directory = parser.get('directories', 'output')

# save the model once it is trained
models_directory = parser.get('directories', 'models')
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
############################################################################
# network architecture
encoder_str = parser.get('architecture', 'encoder')
encoder = [int(units) for units in encoder_str.split('_')]

latent_dimensions = parser.getint('architecture', 'latent_dimensions')

decoder_str = parser.get('architecture', 'decoder')
decoder = [int(units) for units in decoder_str.split('_')]

architecture_str = f'{encoder_str}_{latent_dimensions}_{decoder_str}'
############################################################################
# network hyperparameters
learning_rate = parser.getfloat('hyper-parameters', 'learning_rate')
batch_size = parser.getint('hyper-parameters', 'batch_size')
epochs = parser.getint('hyper-parameters', 'epochs')
############################################################################
# data parameters
number_spectra = parser.getint('data-parameters', 'number_spectra')
################################################################################
number_input_dimensions = 3000
vae = VariationalAE(input_dimensions=number_input_dimensions,
        encoder_units=encoder,
        latent_dimensions=latent_dimensions,
        decoder_units=decoder,
        batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

vae.summary()
###############################################################################
# Loading training data
###############################################################################
# Training the model

# history = vae.fit(spectra=training_set[:, :-5])
###############################################################################
# Defining directorie to save the model once it is trained
# vae_name = 'DenseVAE'
# encoder_name = 'DenseEncoder'
# decoder_name = 'DenseDecoder'
#
# vae.save_vae(f'{models_dir}/{vae_name}')
# vae.save_encoder(f'{models_dir}/{encoder_name}')
# vae.save_decoder(f'{models_dir}/{decoder_name}')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
