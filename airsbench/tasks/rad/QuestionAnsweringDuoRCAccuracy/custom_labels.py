# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import numpy as np
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make gold labels for the DuoRC dataset for Question Answering."
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
        'ibm-research/duorc/ParaphraseRC'
    )
    dataset = load_from_disk(dataset_source_fpath)
    test = dataset["test"]

    df = test.to_pandas()
    df_labels = df[["no_answer", "answers"]].rename(
        columns={"no_answer": "has_answer"}
    )
    df_labels["answers"] = df_labels["answers"].apply(lambda x: "" if len(x) == 0 else x[0])
    df_labels["has_answer"] = np.logical_not(df_labels["has_answer"])
    save_path = (
        Path(args.output_directory).expanduser() /
        "gold_submission.csv"
    )
    df_labels.to_csv(save_path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
