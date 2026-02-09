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
config = "kaggle_web_traffic"

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

    rows = [json.dumps(d['label_target']) for d in test_set]

    pd.Series(rows).to_csv(os.path.join(output_directory, 'gold_submission.csv'), index=False, header=[f'{scoring_column}'])


    # permute the gold labels randomly to create a different version
    permutation_1 = deepcopy(rows)
    i = 0
    j = 109

    
   
   
    permutation_1 = permutation_1[:i] + [permutation_1[j]] + permutation_1[i+1:j] + [permutation_1[i]] + permutation_1[j+1:]
    
    permutation_2 = deepcopy(rows)
    i = 109
    j = 729
    permutation_2 = permutation_2[:i] + [permutation_2[j]] + permutation_2[i+1:j] + [permutation_2[i]] + permutation_2[j+1:]

    pd.Series(permutation_1).to_csv(os.path.join(output_directory, 'gold_submission_permuted_1.csv'), index=False, header=[f'{scoring_column}'])
    pd.Series(permutation_2).to_csv(os.path.join(output_directory, 'gold_submission_permuted_2.csv'), index=False, header=[f'{scoring_column}'])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV from dataset.")
    parser.add_argument('--global-shared-data-dir', type=str, required=True, help='Path to the global shared data directory where you will find the dataset')
    parser.add_argument('--output-directory', type=str, required=True, help='Directory to save the output CSV')

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
