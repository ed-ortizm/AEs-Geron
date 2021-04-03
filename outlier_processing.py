#!/usr/bin/env python3.8
import glob
import os
import sys
import time

import numpy as np

from constants_VAE_outlier import spectra_dir, working_dir
from lib_VAE_outlier import input_handler
###############################################################################
ti = time.time()
###############################################################################
n_spectra, normalization_type, local = input_handler(script_arguments=sys.argv)
###############################################################################
# Relevant directories
training_data_dir = f'{spectra_dir}/normalized_data'
generated_data_dir = f'{spectra_dir}/AE_outlier'
layers_data_dirs = glob.glob(f'{generated_data_dir}/*/*')
###############################################################################
# Loading AEs predicted data for outlier detection

metric = 'mse'

for layers_data_dir in layers_data_dirs:

    n_spectra = layers_data_dir.split('/')[-1]
    layers_str = layers_data_dir.split('/')[-2]
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

    reconstructed_set_name = f'{training_set_name}_reconstructed_{layers_str}'

    if local:
        reconstructed_set_name = f'{reconstructed_set_name}_local'

    reconstructed_set_path = f'{layers_data_dir}/{reconstructed_set_name}.npy'

    print(f'Loading reconstructed data set: {reconstructed_set_name}\n')

    reconstructed_set =  np.load(f'{reconstructed_set_path}')
    ############################################################################
    ## Outlier processing
    ############################################################################
    # Data for outlier detection

    o_score_name = f'{metric}_o_score_{n_spectra}_{layers_str}'
    fname_normal = f'most_normal_ids_{metric}_{n_spectra}_{layers_str}'
    fname_outliers = f'most_outlying_ids_{metric}_{n_spectra}_{layers_str}'

    if local:
        o_score_name = f'{o_score_name}_local'
        fname_normal = f'{fname_normal}_local'
        fname_outliers = f'{fname_outliers}_local'

    o_scores = np.load(f'{layers_data_dir}/{o_score_name}.npy')
    outlier_ids = np.load(f'{layers_data_dir}/{fname_outliers}.npy')
    normal_ids = np.load(f'{layers_data_dir}/{fname_normal}.npy')
    ###############################################################################
    # top reconstructions

    O_top_outliers = training_set[outlier_ids]
    R_top_outliers = reconstructed_set[outlier_ids]
    O_top_normal = training_set[normal_ids]
    R_top_normal = reconstructed_set[normal_ids]
    ############################################################################
    # saving data
    O_top_outliers_name = f'outliers_spectra_{outlier_ids.size}_{n_spectra}_{layers_str}'
    R_top_outliers_name = f'outliers_reconstructed_{outlier_ids.size}_{n_spectra}_{layers_str}'
    O_top_normal_name = f'normal_spectra_{normal_ids.size}_{n_spectra}_{layers_str}'
    R_top_normal_name = f'normal_reconstructed_{normal_ids.size}_{n_spectra}_{layers_str}'

    if local:

        O_top_outliers_name = f'{O_top_outliers_name}_local'
        R_top_outliers_name = f'{R_top_outliers_name}_local'
        O_top_normal_name = f'{O_top_normal_name}_local'
        R_top_normal_name = f'{R_top_normal_name}_local'

    np.save(f'{layers_data_dir}/{O_top_outliers_name}.npy', O_top_outliers)
    np.save(f'{layers_data_dir}/{R_top_outliers_name}.npy', R_top_outliers)
    np.save(f'{layers_data_dir}/{O_top_normal_name}.npy', O_top_normal)
    np.save(f'{layers_data_dir}/{R_top_normal_name}.npy', R_top_normal)

###############################################################################
tf = time.time()
print(f'Running time: {tf-ti:.2f}')
