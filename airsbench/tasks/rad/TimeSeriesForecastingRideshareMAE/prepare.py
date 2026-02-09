# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import logging

from datasets import load_dataset, load_from_disk
from utils import reformat_dataset, combine_lists

# Configure logger with custom prefix
logger = logging.getLogger("dataset_code")
handler = logging.StreamHandler()
formatter = logging.Formatter("[Running provided `dataset_code`] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main(global_shared_data_dir: str, agent_data_mount_dir: str, agent_log_dir: str) -> None:
    # Load from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, "Monash-University/monash_tsf/rideshare")
    dataset = load_from_disk(dataset_source_fpath)
    train = dataset["train"]  # Base length time steps per series (339 series, minutely measurements)
    validation = dataset["validation"]  # Base+60 time steps per series (339 series, 1 hour ahead)
    test = dataset["test"]  # Base+120 time steps per series (339 series, 2 hours ahead)

    train = train.select_columns(["target", "feat_dynamic_real"])
    test = test.select_columns(["target", "feat_dynamic_real"])
    validation = validation.select_columns(["target", "feat_dynamic_real"])

    train = train.map(combine_lists)
    test = test.map(combine_lists)
    validation = validation.map(combine_lists)

    # TRAINING DATA PREPARATION:
    # Input: train split (base steps) → Target: validation split (base+60 steps)
    # Model learns to forecast 60 steps ahead (consistent 60-step horizon, 1 hour)
    train_set = reformat_dataset(train, validation)

    # TEST DATA PREPARATION:
    # Input: validation split (base+60 steps) → Target: test split (base+120 steps)
    # Model must forecast 60 steps ahead for evaluation (consistent 60-step horizon, 1 hour)
    test_set = reformat_dataset(validation, test)

    # Remove labels from test set (agent shouldn't see ground truth)
    test_set = test_set.remove_columns(["label_target"])

    # Save to the agent data mount directory
    train_set.save_to_disk(os.path.join(agent_data_mount_dir, "train"))
    test_set.save_to_disk(os.path.join(agent_data_mount_dir, "test"))


if __name__ == "__main__":
    # Boilerplate code from prepare_boilerplate.py

    parser = argparse.ArgumentParser(
        description="Ensure the script is run with the required data directory arguments."
    )
    parser.add_argument(
        "--global-shared-data-dir",
        required=True,
        help="Path to the sharded raw data directory, e.g at /checkpoint/maui/shared/airsbench-raw-data",
    )
    parser.add_argument(
        "--agent-data-mount-dir", required=True, help="Path to the agent data mount directory, e.g ~/aira-dojo/data"
    )
    parser.add_argument(
        "--agent-log-dir",
        required=False,
        help="Path to a specific agents log directory, e.g /checkpoint/maui/shared/agent-log-dirs/agent-428391. This can be used e.g to pull in data created by a previous agent.",
    )

    args = parser.parse_args()

    # Validate that the provided directories exist
    for path in [args.global_shared_data_dir, args.agent_data_mount_dir, args.agent_log_dir]:
        if path is not None and not os.path.isdir(path):
            print(f"Error: Directory not found: {path}", file=sys.stderr)
            sys.exit(1)

    logger.info(f"AirsBench raw data directory: {args.global_shared_data_dir}")
    logger.info(f"Agent data mount directory: {args.agent_data_mount_dir}")
    if args.agent_log_dir:
        logger.info(f"Agent log directory: {args.agent_log_dir}")
    else:
        logger.info("No agent log directory provided.")

    main(
        global_shared_data_dir=args.global_shared_data_dir,
        agent_data_mount_dir=args.agent_data_mount_dir,
        agent_log_dir=args.agent_log_dir,
    )
