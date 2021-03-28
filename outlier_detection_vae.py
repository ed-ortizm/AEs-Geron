#!/usr/bin/env python3.8
# https://github.com/tensorflow/tensorflow/issues/47311
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import os
import sys
import time

import numpy as np
from tensorflow import keras

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import DenseVAE
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

    training_set =  np.load(f'{fpath}', mmap_mode='r')

else:
    print(f'There is no file: {fname}')
    sys.exit()
###############################################################################
# Parameters for the DenseVAE
n_galaxies = training_set.shape[0]
n_input_dimensions = training_set[:, :-5].shape[1]
n_latent_dimensions = 10
###############################################################################
# Loading VAE trained model
models_dir = f'{working_dir}/models'
model_name = f'DenseVAE'
model_path = f'{models_dir}/{model_name}'
vae = keras.models.load_model(model_path, compile=False)
###############################################################################
#
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
