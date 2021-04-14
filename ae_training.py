#!/usr/bin/env python3.8
from argparse import ArgumentParser
import os
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import AEDense
from lib_VAE_outlier import load_data
from lib_VAE_outlier import plot_history
###############################################################################
ti = time.time()
###############################################################################
parser = ArgumentParser()

parser.add_argument('--learning_rate', '-lr', type=float)
parser.add_argument('--batch_size','-bs', type=int)
parser.add_argument('--server', '-s', type=str)
parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--normalization_type', '-n_type', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--loss', type=str)

script_arguments = parser.parse_args()

##############################################################################
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
local = script_arguments.server == 'local'

number_latent_dimensions = script_arguments.latent_dimensions

layers_encoder = [int(number_units) for number_units
    in script_arguments.encoder_layers.split('_')]

layers_decoder = [int(number_units) for number_units
    in script_arguments.decoder_layers.split('_')]

epochs = script_arguments.epochs
loss = script_arguments.loss
batch_size = script_arguments.batch_size
learning_rate = script_arguments.learning_rate
################################################################################
# Relevant directories
train_data_dir = f'{spectra_dir}/normalized_data'
################################################################################
# Loading training data
train_set_name = f'spectra_{number_spectra}_{normalization_type}'
train_set_path = f'{train_data_dir}/{train_set_name}.npy'
training_set = load_data(train_set_name, train_set_path)
np.random.shuffle(training_set)
################################################################################
# Parameters for the AEDense
number_input_dimensions = training_set[:, :-5].shape[1]
###########################################
ae = AEDense(number_input_dimensions, layers_encoder, number_latent_dimensions,
    layers_decoder, batch_size, epochs, learning_rate, loss)

ae.summary()
###############################################################################
# Training the model
history = ae.fit(spectra=training_set[:, :-5])
################################################################################
# Defining directorie to save the model once it is trained
models_dir = f'{working_dir}/models/AE'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Models names
encoder_str = script_arguments.encoder_layers
decoder_str = script_arguments.decoder_layers

layers_str = f'{encoder_str}_{number_latent_dimensions}_{decoder_str}'

models_dir = f'{models_dir}/{layers_str}'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

tail_model_name = (f'{loss}_{layers_str}_nSpectra_{number_spectra}_nType'
    f'_{normalization_type}')

ae_name = f'DenseAE_{tail_model_name}'
encoder_name = f'DenseEncoder_{tail_model_name}'
decoder_name = f'DenseDecoder_{tail_model_name}'

if local:

    print('Saving model trained in local machine')

    ae_name = f'DenseAE_{tail_model_name}_local'
    encoder_name = f'DenseEncoder_{tail_model_name}_local'
    decoder_name = f'DenseDecoder_{tail_model_name}_local'

ae.save_ae(f'{models_dir}/{ae_name}')
ae.save_encoder(f'{models_dir}/{encoder_name}')
ae.save_decoder(f'{models_dir}/{decoder_name}')

plot_history(history, f'{models_dir}/{ae_name}')
################################################################################
#Reconstructed data for outlier detection
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{number_spectra}'

if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)

tail_reconstructed = (f'{loss}_{layers_str}_nType_{normalization_type}')

reconstructed_set_name = (
    f'{train_set_name}_{tail_reconstructed}_reconstructed')

if local:
    reconstructed_set_name = f'{reconstructed_set_name}_local'

reconstructed_set_path = f'{generated_data_dir}/{reconstructed_set_name}.npy'

reconstructed_set = ae.predict(training_set[:, :-5])

print(f'Saving reconstructed training set')

np.save(f'{reconstructed_set_path}', reconstructed_set)

################################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
