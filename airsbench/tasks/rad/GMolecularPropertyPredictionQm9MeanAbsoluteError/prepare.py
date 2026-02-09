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

# Configure logger with custom prefix
logger = logging.getLogger('dataset_code')
handler = logging.StreamHandler()
formatter = logging.Formatter('[Running provided `dataset_code`] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main(global_shared_data_dir: str, agent_data_mount_dir: str, agent_log_dir: str) -> None:
    """
    Main function for processing data directories.

    :param global_shared_data_dir: Path to the AIRS-Bench raw data directory.
    :param agent_data_mount_dir: Path to the agent data mount directory.
    :param agent_log_dir: Path to an agents log directory.
    """

    # Load from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, 'nimashoghi/qm9/default')
    dataset = load_from_disk(dataset_source_fpath)
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']

    # Scale G labels by 1000
    train = train.map(lambda example: {'G': example['G'] * 1000})
    val = val.map(lambda example: {'G': example['G'] * 1000})
    test = test.map(lambda example: {'G': example['G'] * 1000})

    # Remove all scoring columns except G from train set (keep G for training)
    train = train.remove_columns([
        'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps',
        'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'c_v',
        'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM'
    ])
    val = val.remove_columns([
        'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps',
        'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'c_v',
        'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM'
    ])
    # Save to the agent data mount directory
    train.save_to_disk(os.path.join(agent_data_mount_dir, 'train'))
    val.save_to_disk(os.path.join(agent_data_mount_dir, 'val'))

    # Remove all scoring columns including G from test set (G is the prediction target)
    test = test.remove_columns([
        'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps',
        'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'G', 'c_v',
        'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM'
    ])

    # Save test dataset to disk
    test.save_to_disk(os.path.join(agent_data_mount_dir, 'test'))


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
