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
from utils import reformat_dataset, combine_lists

hf_repo = "Monash-University/monash_tsf"
config = "rideshare"

scoring_column = "label_target"


def main(global_shared_data_dir, output_directory):
    """
    Loads data from global_shared_data_dir and saves a gold_submission.csv to output_directory, e.g:
    ds = load_from_disk(os.path.join(global_shared_data_dir, f'{hf_repo}/{config}'))
    data = ds[f'{test_split}']
    rows = [json.dumps(d[f'{scoring_column}']) for d in data]
    pd.Series(rows).to_csv(os.path.join(output_directory, 'gold_submission.csv'), index=False, header=[f'{scoring_column}'])
    """
    ds = load_from_disk(os.path.join(global_shared_data_dir, f"{hf_repo}/{config}"))

    test = ds["test"]
    validation = ds["validation"]

    test = test.map(combine_lists)
    validation = validation.map(combine_lists)

    test_set = reformat_dataset(validation, test)

    # Extract forecast portion from each individual time series
    rows = []
    for d in test_set:
        full_sequences = d["label_target"]
        input_sequences = d["target"]

        for i in range(len(full_sequences)):
            full_seq = full_sequences[i]
            input_seq = input_sequences[i]

            # Extract forecast portion - keep NaNs in original positions
            available_forecast = full_seq[len(input_seq) :]

            # Take exactly 48 values, padding at the END with NaN if needed
            forecast_48 = available_forecast[:48]  # Take up to 48 values
            while len(forecast_48) < 48:  # Pad at the END if shorter
                forecast_48.append(np.nan)

            # Handle NaN values for JSON serialization - convert to 'NaN' strings
            forecast_clean = []
            for val in forecast_48:
                if isinstance(val, float) and np.isnan(val):
                    forecast_clean.append("NaN")
                else:
                    forecast_clean.append(val)

            # Each individual time series becomes one row with 48 forecasts
            rows.append(json.dumps(forecast_clean))

    pd.Series(rows).to_csv(
        os.path.join(output_directory, "gold_submission.csv"), index=False, header=[f"{scoring_column}"]
    )

    # permute the gold labels randomly to create different versions
    # BUT only swap rows that have the same length to preserve data structure
    dataset_size = len(rows)

    # Group rows by their length to ensure we only swap compatible rows
    length_groups = {}
    for i, row in enumerate(rows):
        data = json.loads(row)
        length = len(data)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(i)

    # Permutation 1: Little shuffling - swap only first two elements if they have same length
    permutation_1 = deepcopy(rows)
    if dataset_size >= 2:
        data_0 = json.loads(rows[0])
        data_1 = json.loads(rows[1])
        if len(data_0) == len(data_1):
            # Swap first two elements only if they have same length
            permutation_1[0], permutation_1[1] = permutation_1[1], permutation_1[0]

    # Permutation 2: A lot of shuffling - but only within same-length groups
    permutation_2 = deepcopy(rows)
    if dataset_size > 3:
        # Set random seed for reproducible shuffling
        random.seed(42)

        # Shuffle within each length group to preserve structure
        for length, indices in length_groups.items():
            if len(indices) > 1:
                # Shuffle indices within this length group
                shuffled_indices = indices.copy()
                random.shuffle(shuffled_indices)

                # Apply swaps within the group (70% of group size)
                swap_count = max(1, int(len(indices) * 0.7))
                for k in range(swap_count):
                    if k < len(indices) and k < len(shuffled_indices):
                        i = indices[k]
                        j = shuffled_indices[k]
                        if i != j and i < len(permutation_2) and j < len(permutation_2):
                            permutation_2[i], permutation_2[j] = permutation_2[j], permutation_2[i]

    pd.Series(permutation_1).to_csv(
        os.path.join(output_directory, "gold_submission_permuted_1.csv"), index=False, header=[f"{scoring_column}"]
    )
    pd.Series(permutation_2).to_csv(
        os.path.join(output_directory, "gold_submission_permuted_2.csv"), index=False, header=[f"{scoring_column}"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV from dataset.")
    parser.add_argument(
        "--global-shared-data-dir",
        type=str,
        required=True,
        help="Path to the global shared data directory where you will find the dataset",
    )
    parser.add_argument("--output-directory", type=str, required=True, help="Directory to save the output CSV")

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
