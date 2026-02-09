# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_from_disk
import json
import pandas as pd
import os
import argparse
import random
from copy import deepcopy
import numpy as np
from utils import reformat_dataset

hf_repo = "Monash-University/monash_tsf"
config = "solar_weekly"

scoring_column = "label_target"


def main(
    global_shared_data_dir,
    output_directory
):
    """
    Loads data from global_shared_data_dir and saves a gold_submission.csv to output_directory, e.g:
    ds = load_from_disk(os.path.join(global_shared_data_dir, f'{hf_repo}/{config}'))
    data = ds[f'{test_split}']
    rows = [json.dumps(d[f'{scoring_column}']) for d in data]
    pd.Series(rows).to_csv(os.path.join(output_directory, 'gold_submission.csv'), index=False, header=[f'{scoring_column}'])
    """
    ds = load_from_disk(os.path.join(global_shared_data_dir, f'{hf_repo}/{config}'))

    test = ds['test']
    validation = ds['validation']

    test_set = reformat_dataset(validation, test)

    # Extract only the forecast portion from each sequence (5 weekly values)
    # This matches what agents are expected to predict for solar_weekly
    rows = []
    for d in test_set:
        full_sequence = d['label_target']  # Extended sequence length
        input_sequence = d['target']       # Base sequence length
        forecast_portion = full_sequence[len(input_sequence):]  # Forecast steps ahead
        rows.append(json.dumps(forecast_portion))

    pd.Series(rows).to_csv(os.path.join(output_directory, 'gold_submission.csv'), index=False, header=[f'{scoring_column}'])

    # permute the gold labels randomly to create different versions
    # Get dataset size for permutation logic
    dataset_size = len(rows)

    # Permutation 1: Little shuffling - swap only adjacent elements
    permutation_1 = deepcopy(rows)
    if dataset_size >= 2:
        # Swap first two elements only (minimal change)
        permutation_1[0], permutation_1[1] = permutation_1[1], permutation_1[0]

    # Permutation 2: A lot of shuffling - extensive randomization
    permutation_2 = deepcopy(rows)
    if dataset_size > 3:
        # Set random seed for reproducible shuffling
        random.seed(42)
        # Shuffle approximately 70% of the dataset extensively
        shuffle_count = max(2, int(dataset_size * 0.7))
        indices = list(range(dataset_size))
        random.shuffle(indices)

        # Apply the shuffled indices to reorder elements extensively
        for i in range(shuffle_count):
            if i < len(indices) and indices[i] < dataset_size:
                # Swap current position with shuffled position
                j = indices[i]
                if i != j:
                    permutation_2[i], permutation_2[j] = permutation_2[j], permutation_2[i]

    pd.Series(permutation_1).to_csv(os.path.join(output_directory, 'gold_submission_permuted_1.csv'), index=False, header=[f'{scoring_column}'])
    pd.Series(permutation_2).to_csv(os.path.join(output_directory, 'gold_submission_permuted_2.csv'), index=False, header=[f'{scoring_column}'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV from dataset.")
    parser.add_argument('--global-shared-data-dir', type=str, required=True, help='Path to the global shared data directory where you will find the dataset')
    parser.add_argument('--output-directory', type=str, required=True, help='Directory to save the output CSV')

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
