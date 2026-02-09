#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import logging

import shutil
from datasets import load_from_disk

# Configure logger with custom prefix
SCRIPT_NAME = 'evaluate_prepare'
logger = logging.getLogger(SCRIPT_NAME)
handler = logging.StreamHandler()
formatter = logging.Formatter('[Running provided `SCRIPT_NAME`] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main(global_shared_data_dir: str, agent_data_mount_dir: str, agent_log_dir: str) -> None:
    """
    Loads test set from airsbench_raw_data_dir into agent_data_mount_dir.
    Loads submission.csv from agent_log_dir into agent_data_mount_dir.

    :param airsbench_raw_data_dir: Path to the AIRS-Bench raw data directory.
    :param agent_data_mount_dir: Path to the agent data mount directory.
    :param agent_log_dir: Path to an agents log directory.
    """
    # Load test with labels from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, 'Yelp/yelp_review_full', 'yelp_review_full')
    dataset = load_from_disk(dataset_source_fpath)
    test = dataset['test']
    test.save_to_disk(os.path.join(agent_data_mount_dir, 'test_with_labels'))

    # Load submission.csv from the agent log directory
    submission_fpath = os.path.join(agent_log_dir, 'submission.csv')
    shutil.copy2(submission_fpath, os.path.join(agent_data_mount_dir, 'submission.csv'))


if __name__ == "__main__":
    # Boilerplate code from prepare_boilerplate.py

    parser = argparse.ArgumentParser(
        description="Ensure the script is run with the required data directory arguments."
    )
    parser.add_argument(
        "--global-shared-data-dir",
        required=True,
        help="Path to the sharded raw data directory, e.g at /checkpoint/maui/shared/airsbench-raw-data"
    )
    parser.add_argument(
        "--agent-data-mount-dir",
        required=True,
        help="Path to the agent data mount directory, e.g ~/aira-dojo/data"
    )
    parser.add_argument(
        "--agent-log-dir",
        required=False,
        help="Path to a specific agents log directory, e.g /checkpoint/maui/shared/agent-log-dirs/agent-428391. This can be used e.g to pull in data created by a previous agent."
    )

    args = parser.parse_args()

    # Validate that the provided directories exist
    for path in [args.global_shared_data_dir, args.agent_data_mount_dir, args.agent_log_dir]:
        if path is not None and not os.path.isdir(path):
            print(f"Error: Directory not found: {path}", file=sys.stderr)
            sys.exit(1)

    logger.info(f"AIRSBench global_shared_data_dir: {args.global_shared_data_dir}")
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
    