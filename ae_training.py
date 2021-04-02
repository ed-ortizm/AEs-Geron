#!/usr/bin/env python3.8

import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import AEDense
from lib_VAE_outlier import input_handler
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
###############################################################################
# Loading training data
fname = f'spectra_{n_spectra}_{normalization_type}.npy'
fpath = f'{training_data_dir}/{fname}'

if os.path.exists(fpath):

    print(f'Loading training set: {fname}')

    training_set =  np.load(f'{fpath}')
    np.random.shuffle(training_set)

else:
    print(f'There is no file: {fname}')
    sys.exit()
###############################################################################
# Parameters for the AEDense
n_galaxies = training_set.shape[0]
n_input_dimensions = training_set[:, :-5].shape[1]
n_latent_dimensions = 5
###########################################
# encoder
n_layers_encoder = [500, 50]

# decoder
n_layers_decoder = [50, 500]

# Other parameters
# 1% to take advantage of stochastic part of stochastic gradient descent
batch_size = int(n_galaxies*0.01)
print(f'Batch size is: {batch_size}')

if local:
    epochs = 5
else:
    epochs = 20

learning_rate = 0.001 # default: 0.001
loss = 'mse'

ae = AEDense(n_input_dimensions, n_layers_encoder, n_latent_dimensions,
    n_layers_decoder, batch_size, epochs, learning_rate, loss)

ae.summary()
###############################################################################
# Training the model

ae.fit(spectra=training_set[:, :-5])
# PEnding to track history
###############################################################################
# Defining directorie to save the model once it is trained
models_dir = f'{working_dir}/models/AE'

if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)
# Models names
# layers for name
layers_encoder_str = '_'.join(str(unit) for unit in n_layers_encoder)
layers_decoder_str = '_'.join(str(unit) for unit in n_layers_decoder)
layers_str = f'{layers_encoder_str}_{n_latent_dimensions}_{layers_decoder_str}'

ae_name = f'DenseAE_{loss}_{n_galaxies}_{layers_str}'
encoder_name = f'DenseEncoder_{loss}_{n_galaxies}_{layers_str}'
decoder_name = f'DenseDecoder_{loss}_{n_galaxies}_{layers_str}'

if local:

    print('Saving model trained in local machine')
    ae.save_ae(f'{models_dir}/{ae_name}_{local}')
    ae.save_encoder(f'{models_dir}/{encoder_name}_local')
    ae.save_decoder(f'{models_dir}/{decoder_name}_local')

else:
    ae.save_ae(f'{models_dir}/{ae_name}')
    ae.save_encoder(f'{models_dir}/{encoder_name}')
    ae.save_decoder(f'{models_dir}/{decoder_name}')
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
