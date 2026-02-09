# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pandas as pd
from datasets import load_from_disk


def main(
    global_shared_data_dir,
    output_directory
):
    """
    Creates gold_submission.csv files with scaled G labels (multiplied by 1000)
    """

    # Load the QM9 dataset from the shared data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, 'nimashoghi/qm9/default')
    dataset = load_from_disk(dataset_source_fpath)
    test = dataset['test']

    # Scale G labels by 1000
    scaled_labels = [label * 1000 for label in test['G']]

    # Save as gold_submission.csv
    output_file = os.path.join(output_directory, 'gold_submission.csv')
    pd.Series(scaled_labels).to_csv(output_file, index=False, header=['G'])
    print(f"Saved scaled G labels to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV with scaled labels.")
    parser.add_argument('--global-shared-data-dir', type=str, required=True, help='Path to the global shared data directory')
    parser.add_argument('--output-directory', type=str, required=True, help='Directory to save the output CSV')

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
