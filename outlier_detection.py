#!/usr/bin/env python3.8
from argparse import ArgumentParser
import glob
import os
import sys
import time

import numpy as np

from constants_VAE_outlier import models_dir, spectra_dir, working_dir
from library_outlier import Outlier
from lib_VAE_outlier import load_data, LoadAE

###############################################################################
ti = time.time()
################################################################################
parser = ArgumentParser()
############################################################################
parser.add_argument('--server', '-s', type=str)
parser.add_argument('--number_spectra','-n_spec', type=int)
parser.add_argument('--normalization_type', '-n_type', type=str)
############################################################################
parser.add_argument('--model', type=str)
parser.add_argument('--encoder_layers', type=str)
parser.add_argument('--latent_dimensions', '-lat_dims', type=int)
parser.add_argument('--decoder_layers', type=str)
parser.add_argument('--loss', type=str)
############################################################################
parser.add_argument('--test_set', '-t_set', type=str)
############################################################################
parser.add_argument('--metrics', type=str, nargs='+')
parser.add_argument('--top_spectra', '-top', type=int)
############################################################################
parser.add_argument('--percentages', '-%', type=int, nargs='+')
############################################################################
script_arguments = parser.parse_args()
################################################################################
local = script_arguments.server == 'local'
number_spectra = script_arguments.number_spectra
normalization_type = script_arguments.normalization_type
############################################################################
model = script_arguments.model
layers_encoder = script_arguments.encoder_layers
number_latent_dimensions = script_arguments.latent_dimensions
layers_decoder = script_arguments.decoder_layers
loss = script_arguments.loss
layers_str = f'{layers_encoder}_{number_latent_dimensions}_{layers_decoder}'
############################################################################
test_set_name = script_arguments.test_set
############################################################################
metrics = script_arguments.metrics
number_top_spectra = script_arguments.top_spectra
############################################################################
percentages = script_arguments.percentages
################################################################################
# Relevant directories
data_dir = f'{spectra_dir}/processed_spectra'
generated_data_dir = (
    f'{spectra_dir}/{model}_outlier/{layers_str}/{number_spectra}')

if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)
################################################################################
# Loading data
test_set_path = f'{data_dir}/{test_set_name}.npy'
test_set = load_data(test_set_name, test_set_path)
################################################################################
# Reconstructed data
tail_model_name = (f'{model}_{layers_str}_loss_{loss}_nTrain_{number_spectra}_'
    f'nType_{normalization_type}')

if local:
    tail_model_name = f'{tail_model_name}_local'

tail_reconstructed = f'reconstructed_{tail_model_name}'
############################################################################
reconstructed_test_set_name = f'{test_set_name}_{tail_reconstructed}'

reconstructed_test_set_path = (
    f'{generated_data_dir}/{reconstructed_test_set_name}.npy')

if os.path.exists(reconstructed_test_set_path):

    reconstructed_test_set = load_data(reconstructed_test_set_name,
        reconstructed_test_set_path)
else:

    #os.mkdirs(reconstructed)
    model_head = f'{models_dir}/{model}/{layers_str}/Dense'
    model_tail = (f'{layers_str}_loss_{loss}_nTrain_{number_spectra}_'
        f'nType_{normalization_type}')
    #    f'nType_{normalization_type}')
    if local:
        model_tail = f'{model_tail}_local'

    ae_path = f'{model_head}{model}_{model_tail}'
    encoder_path = f'{model_head}Encoder_{model_tail}'
    decoder_path = f'{model_head}Decoder_{model_tail}'

    ae = LoadAE(ae_path, encoder_path, decoder_path)

    reconstructed_test_set = ae.predict(test_set[:, :-8])

    np.save(reconstructed_test_set_path, reconstructed_test_set)
################################################################################
# Outlier detection

for metric in metrics:
    outlier = Outlier(metric=metric)
################################################################################
    scores_test = outlier.score(O=test_set[:, :-8],
        R=reconstructed_test_set, percentages=percentages)

    for idx, scores in enumerate(scores_test):

        percent_str = f'{percentages[idx]}_percent'
        scores_dir = f'{generated_data_dir}/{test_set_name}_{metric}_score_{percent_str}'

        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
    ############################################################################
        scores_name = f'{test_set_name}_{metric}_score_{percent_str}_{tail_model_name}'

        np.save(f'{scores_dir}/{scores_name}.npy', scores)
    ############################################################################
        normal_ids, outlier_ids = outlier.top_reconstructions(
            scores=scores, n_top_spectra=number_top_spectra)

        print('Saving top outliers IDs')
        ########################################################################
        normal_name = f'normal_IDS_nTop_{number_top_spectra}_{scores_name}'
        np.save(f'{scores_dir}/{normal_name}.npy', normal_ids)
        ########################################################################
        outlier_name = f'outlier_IDS_nTop_{number_top_spectra}_{scores_name}'
        np.save(f'{scores_dir}/{outlier_name}.npy', outlier_ids)
    ############################################################################
        spec_top_outliers_name = (
            f'outlier_nTop_{number_top_spectra}_{scores_name}')

        spec_top_outliers = test_set[outlier_ids]
        spec_top_outliers = np.insert(spec_top_outliers, 0, outlier_ids, axis=1)

        np.save(f'{scores_dir}/{spec_top_outliers_name}.npy', spec_top_outliers)
        ########################################################################
        R_top_outliers_name = (
            f'reconstructed_outlier_nTop_{number_top_spectra}_{scores_name}')

        R_top_outliers = reconstructed_test_set[outlier_ids]

        np.save(f'{scores_dir}/{R_top_outliers_name}.npy', R_top_outliers)
        ########################################################################
        spec_top_normal_name = (
            f'normal_nTop_{number_top_spectra}_{scores_name}')

        spec_top_normal = test_set[normal_ids]

        spec_top_normal = np.insert(spec_top_normal, 0, normal_ids, axis=1)

        np.save(f'{scores_dir}/{spec_top_normal_name}.npy', spec_top_normal)
        ########################################################################
        R_top_normal_name = (
            f'reconstructed_normal_nTop_{number_top_spectra}_{scores_name}')

        R_top_normal = reconstructed_test_set[normal_ids]

        np.save(f'{scores_dir}/{R_top_normal_name}.npy', R_top_normal)
###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
def nothing():
    # data_set_name = f'spectra_{number_spectra}_{normalization_type}'
    # train_set_name = f'star_forming_spectra_{number_spectra}_{normalization_type}'
    # ############################################################################
    # # train_set_name = f'{data_set_name}_nSnr_{number_snr}_SF_train'
    # train_set_path = f'{data_dir}/{train_set_name}.npy'
    # train_set = load_data(train_set_name, train_set_path)
    ############################################################################
    # test_set_name = f'{data_set_name}_nSnr_{number_snr}_noSF_test'

    # reconstructed_train_set_name = f'{train_set_name}_{tail_reconstructed}'
############################################################################
# reconstructed_train_set_path = (
#     f'{generated_data_dir}/{reconstructed_train_set_name}.npy')
#
# reconstructed_train_set = load_data(reconstructed_train_set_name,
#     reconstructed_train_set_path)
    ################################################################################
########TTTTTTTTTTTTTTTTTTTEEEEEEEEEEEEESSSSSSSSSSTTTTTTTTTTTTT
    # scores_test = outlier.score(O=test_set[:, :-8],
    #     R=reconstructed_test_set, percentages=percentages)
    #
    # for idx, scores in enumerate(scores_test):
    #
    #     percent_str = f'{percentages[idx]}_percent'
    #     scores_dir = f'{generated_data_dir}/{metric}_score_{percent_str}'
    # ############################################################################
    #     scores_name = f'{metric}_score_{percent_str}_test_{tail_model_name}'
    #
    #     np.save(f'{scores_dir}/{scores_name}.npy', scores)
    # ############################################################################
    #     normal_ids, outlier_ids = outlier.top_reconstructions(
    #         scores=scores, n_top_spectra=number_top_spectra)
    #
    #     print('Saving top outliers IDs')
    #     ########################################################################
    #     normal_name = f'normal_IDS_nTop_{number_top_spectra}_{scores_name}'
    #     np.save(f'{scores_dir}/{normal_name}.npy', normal_ids)
    #     ########################################################################
    #     outlier_name = f'outlier_IDS_nTop_{number_top_spectra}_{scores_name}'
    #     np.save(f'{scores_dir}/{outlier_name}.npy', outlier_ids)
    # ############################################################################
    #     spec_top_outliers_name = (
    #         f'outlier_nTop_{number_top_spectra}_{scores_name}')
    #
    #     spec_top_outliers = test_set[outlier_ids]
    #     spec_top_outliers = np.insert(spec_top_outliers, 0, outlier_ids, axis=1)
    #
    #     np.save(f'{scores_dir}/{spec_top_outliers_name}.npy', spec_top_outliers)
    #     ########################################################################
    #     R_top_outliers_name = (
    #         f'reconstructed_outlier_nTop_{number_top_spectra}_{scores_name}')
    #
    #     R_top_outliers = reconstructed_test_set[outlier_ids]
    #
    #     np.save(f'{scores_dir}/{R_top_outliers_name}.npy', R_top_outliers)
    #     ########################################################################
    #     spec_top_normal_name = (
    #         f'normal_nTop_{number_top_spectra}_{scores_name}')
    #
    #     spec_top_normal = test_set[normal_ids]
    #
    #     spec_top_normal = np.insert(spec_top_normal, 0, normal_ids, axis=1)
    #
    #     np.save(f'{scores_dir}/{spec_top_normal_name}.npy', spec_top_normal)
    #     ########################################################################
    #     R_top_normal_name = (
    #         f'reconstructed_normal_nTop_{number_top_spectra}_{scores_name}')
    #
    #     R_top_normal = reconstructed_test_set[normal_ids]
    #
    #     np.save(f'{scores_dir}/{R_top_normal_name}.npy', R_top_normal)
    pass
