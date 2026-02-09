# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
import argparse
from datasets import load_from_disk
from utils import reformat_dataset, combine_lists


def main(global_shared_data_dir: str, agent_data_mount_dir: str, agent_log_dir: str) -> None:
    """
    Loads test set from airsbench_raw_data_dir into agent_data_mount_dir.
    Loads submission.csv from agent_log_dir into agent_data_mount_dir.
    :param airsbench_raw_data_dir: Path to the AIRS-Bench raw data directory.
    :param agent_data_mount_dir: Path to the agent data mount directory.
    :param agent_log_dir: Path to an agents log directory.
    """

    # Load test with labels from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, "Monash-University/monash_tsf/rideshare")
    dataset = load_from_disk(dataset_source_fpath)
    test = dataset["test"]
    validation = dataset["validation"]

    test = test.map(combine_lists)
    validation = validation.map(combine_lists)

    test_set = reformat_dataset(validation, test)

    test_set.save_to_disk(os.path.join(agent_data_mount_dir, "test_with_labels"))

    # Load submission.csv from the agent log directory
    submission_fpath = os.path.join(agent_log_dir, "submission.csv")
    shutil.copy2(submission_fpath, os.path.join(agent_data_mount_dir, "submission.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test set with labels")
    parser.add_argument("--global-shared-data-dir")
    parser.add_argument("--agent-data-mount-dir")
    parser.add_argument("--agent-log-dir")
    args = parser.parse_args()

    # Validate that the provided directories exist
    for path in [args.global_shared_data_dir, args.agent_data_mount_dir, args.agent_log_dir]:
        if path is not None and not os.path.isdir(path):
            print(f"Error: Directory not found: {path}", file=sys.stderr)
            sys.exit(1)

    main(args.global_shared_data_dir, args.agent_data_mount_dir, args.agent_log_dir)
