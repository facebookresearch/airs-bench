# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
import pandas as pd
from random import random
from datasets import load_from_disk
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make gold labels for the APPS dataset for Code Generation."
    )
    parser.add_argument(
        "--global-shared-data-dir",
        required=True,
        help="Path to the sharded raw data directory, e.g at /checkpoint/maui/shared/airsbench-raw-data"
    )
    parser.add_argument(
        "--output-directory",
        default=Path(__file__).resolve().parent,
        help="Path to the agent data mount directory, e.g ~/aira-dojo/data"
    )
    return parser.parse_args()


def main(args):
    dataset_source_fpath = os.path.join(
        Path(args.global_shared_data_dir),
        'codeparrot/apps/all'
    )
    dataset = load_from_disk(dataset_source_fpath)
    test = dataset["test"]
    codes = []
    codes_perm_1 = []
    codes_perm_2 = []

    incorrect_solution = "print('invalid')"

    for idx in range(len(test)):
        sample = test[idx]

        try:
            # problematic code - errors if correct sample code doesn't exist
            sample_solution = json.loads(sample["solutions"])[0]
        except:
            sample_solution = incorrect_solution

        codes.append(sample_solution)
        rng = random()
        if rng < 0.33:
            codes_perm_1.append(sample_solution)
            codes_perm_2.append(sample_solution)
        elif rng < 0.66:
            codes_perm_1.append(incorrect_solution)
            codes_perm_2.append(sample_solution)
        else:
            codes_perm_1.append(incorrect_solution)
            codes_perm_2.append(incorrect_solution)

    df = pd.DataFrame({f"code{i}": codes for i in range(1, 6)})
    df_perm_1 = pd.DataFrame({f"code{i}": codes_perm_1 for i in range(1, 6)})
    df_perm_2 = pd.DataFrame({f"code{i}": codes_perm_2 for i in range(1, 6)})


    # Save to CSV
    save_path = Path(args.output_directory).expanduser() / "gold_submission.csv"
    df.to_csv(save_path, index=False)
    save_path = Path(args.output_directory).expanduser() / "gold_submission_permuted_1.csv"
    df_perm_1.to_csv(save_path, index=False)
    save_path = Path(args.output_directory).expanduser() / "gold_submission_permuted_2.csv"
    df_perm_2.to_csv(save_path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
