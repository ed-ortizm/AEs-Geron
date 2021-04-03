#!/usr/bin/env python3.8
import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import input_handler, LoadAE, Outlier
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
layers_str = sys.argv[-1]
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{n_spectra}'
if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)

models_dir = f'{working_dir}/models/AE/{layers_str}'
###############################################################################
# Loading training data
training_set_name = f'spectra_{n_spectra}_{normalization_type}'
training_set_path = f'{training_data_dir}/{training_set_name}.npy'

if os.path.exists(training_set_path):

    print(f'Loading training set: {training_set_name}.npy\n')

    training_set =  np.load(f'{training_set_path}', mmap_mode='r')

else:
    print(f'There is no file: {training_set_name}.npy\n')
    sys.exit()
###############################################################################
# Loading AEs predicted data for outlier detection

reconstructed_set_name = f'{training_set_name}_reconstructed_{layers_str}'
if local:
    reconstructed_set_name = f'{reconstructed_set_name}_local'

reconstructed_set_path = f'{generated_data_dir}/{reconstructed_set_name}.npy'

if os.path.exists(f'{reconstructed_set_path}'):

    print(f'Loading reconstructed data set: {reconstructed_set_name}\n')

    reconstructed_set =  np.load(f'{reconstructed_set_path}')

else:

    print(f'There is no file: {reconstructed_set_name}\n')
    print(f'Generateing reconstructed set\n')


    # Loading models
    ae_name = f'DenseAE_mse_{n_spectra}_{layers_str}'
    encoder_name = f'DenseEncoder_mse_{n_spectra}_{layers_str}'
    decoder_name = f'DenseDecoder_mse_{n_spectra}_{layers_str}'

    if local:

        ae_name = f'{ae_name}_local'
        encoder_name = f'{encoder_name}_local'
        decoder_name = f'{decoder_name}_local'


    ae_path = f'{models_dir}/{ae_name}'
    encoder_path = f'{models_dir}/{encoder_name}'
    decoder_path = f'{models_dir}/{decoder_name}'

    ae = LoadAE(ae_path, encoder_path, decoder_path)

    ############################################################################
    reconstructed_set = ae.predict(training_set[:, :-5])
    np.save(f'{reconstructed_set_path}', reconstructed_set)
###############################################################################
# Outlier detection
if local:
    n_top_spectra = 100
else:
    n_top_spectra = 10_000

metrics = ['mse']
for metric in metrics:

    if metric == "lp":
        p = 0.1
        outlier = Outlier(metric=metric, p=p)

    else:
        outlier = Outlier(metric=metric)
    ############################################################################
    print(f'Loading outlier scores')

    o_score_name = f'{metric}_o_score_{n_spectra}_{layers_str}'
    fname_normal = f'most_normal_ids_{metric}_{n_spectra}_{layers_str}'
    fname_outliers = f'most_outlying_ids_{metric}_{n_spectra}_{layers_str}'

    if local:
        o_score_name = f'{o_score_name}_local'
        fname_normal = f'most_normal_ids_{metric}_local'
        fname_outliers = f'most_outlying_ids_{metric}_local'

    print(f'This is the path to o_score \n {generated_data_dir}/{o_score_name}')
    if os.path.exists(f'{generated_data_dir}/{o_score_name}.npy'):

        o_scores = np.load(f'{generated_data_dir}/{o_score_name}.npy')

    else:

        print(f'Computing {o_score_name}')

        o_scores = outlier.score(O=training_set[:, :-5], R=reconstructed_set)
        np.save(f'{generated_data_dir}/{o_score_name}.npy',
            o_scores)
    ############################################################################
    #Selecting top outliers
    print(f'Computing top reconstructions for {metric} metric\n')
    most_normal_ids, most_outlying_ids = outlier.top_reconstructions(
        scores=o_scores, n_top_spectra=n_top_spectra)

    print('Saving top outliers data')

    np.save(f'{generated_data_dir}/{fname_normal}.npy',
        most_normal_ids)
    np.save(f'{generated_data_dir}/{fname_outliers}.npy',
        most_outlying_ids)

###############################################################################
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
