#!/usr/bin/env python3.8
from argparse import ArgumentParser
import os
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import load_data, Outlier
###############################################################################
ti = time.time()
###############################################################################
parser = ArgumentParser()

parser.add_argument('--server', '-s', type=str)
parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--normalization_type', '-n_type', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--metrics', type=str, nargs='+')
parser.add_argument('--model', type=str)

script_arguments = parser.parse_args()
################################################################################
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
local = script_arguments.server == 'local'

number_latent_dimensions = script_arguments.latent_dimensions
layers_encoder = script_arguments.encoder_layers
layers_decoder = script_arguments.decoder_layers

metrics = script_arguments.metrics
model = script_arguments.model
################################################################################
# Relevant directories
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'

training_data_dir = f'{spectra_dir}/normalized_data'
generated_data_dir = f'{spectra_dir}/AE_outlier/{layers_str}/{number_spectra}'
###############################################################################
# Loading training data
training_set_name = f'spectra_{number_spectra}_{normalization_type}'
training_set_path = f'{training_data_dir}/{training_set_name}.npy'

training_set = load_data(training_set_name, training_set_path)
# ###############################################################################
# # Outlier detection
# if local:
#     n_top_spectra = 100
# else:
#     n_top_spectra = 10_000
#
# metrics = ['mse']
# for metric in metrics:
#
#     if metric == "lp":
#         p = 0.1
#         outlier = Outlier(metric=metric, p=p)
#
#     else:
#         outlier = Outlier(metric=metric)
#     ############################################################################
#     print(f'Loading outlier scores')
#
#     o_score_name = f'{metric}_o_score_{n_spectra}_{layers_str}'
#     fname_normal = f'most_normal_ids_{metric}_{n_spectra}_{layers_str}'
#     fname_outliers = f'most_outlying_ids_{metric}_{n_spectra}_{layers_str}'
#
#     if local:
#         o_score_name = f'{o_score_name}_local'
#         fname_normal = f'most_normal_ids_{metric}_local'
#         fname_outliers = f'most_outlying_ids_{metric}_local'
#
#     print(f'This is the path to o_score \n {generated_data_dir}/{o_score_name}')
#     if os.path.exists(f'{generated_data_dir}/{o_score_name}.npy'):
#
#         o_scores = np.load(f'{generated_data_dir}/{o_score_name}.npy')
#
#     else:
#
#         print(f'Computing {o_score_name}')
#
#         o_scores = outlier.score(O=training_set[:, :-5], R=reconstructed_set)
#         np.save(f'{generated_data_dir}/{o_score_name}.npy',
#             o_scores)
#     ############################################################################
#     #Selecting top outliers
#     print(f'Computing top reconstructions for {metric} metric\n')
#     most_normal_ids, most_outlying_ids = outlier.top_reconstructions(
#         scores=o_scores, n_top_spectra=n_top_spectra)
#
#     print('Saving top outliers data')
#
#     np.save(f'{generated_data_dir}/{fname_normal}.npy',
#         most_normal_ids)
#     np.save(f'{generated_data_dir}/{fname_outliers}.npy',
#         most_outlying_ids)
#
# ###############################################################################
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
