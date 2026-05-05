# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd





from datasets import load_from_disk
import json
import pandas as pd
import os
import argparse

hf_repo = 'Pavithree/eli5'
config = 'default'
test_split = 'test'
scoring_column = 'answers'


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
    def load_test_set():
        dataset = load_from_disk(os.path.join(global_shared_data_dir, 'Pavithree/eli5', 'default'))['test']
        ground_truth = dataset.map(lambda x: {scoring_column: x[scoring_column]["text"][0] if x[scoring_column]["text"] else ''})[scoring_column]
        return ground_truth

    processed_test_set = load_test_set()

    golden_submissions = pd.DataFrame({scoring_column: processed_test_set})
    golden_submissions.to_csv(os.path.join(output_directory, "gold_submission.csv"), index = False)

    permuted_one = np.random.permutation(golden_submissions[scoring_column].values)
    permuted_golden_submissions = pd.DataFrame({scoring_column: permuted_one})
    permuted_golden_submissions.to_csv(os.path.join(output_directory, "gold_submission_permuted_1.csv"), index = False)

    permuted_two = np.random.permutation(golden_submissions[scoring_column].values)
    permuted_golden_submissions_ = pd.DataFrame({scoring_column: permuted_two})
    permuted_golden_submissions_.to_csv(os.path.join(output_directory, "gold_submission_permuted_2.csv"), index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV from dataset.")
    parser.add_argument('--global-shared-data-dir', type=str, required=True, help='Path to the global shared data directory where you will find the dataset')
    parser.add_argument('--output-directory', type=str, required=True, help='Directory to save the output CSV')

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
