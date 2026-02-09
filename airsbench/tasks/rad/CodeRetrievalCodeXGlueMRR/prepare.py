# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import logging
import re

from datasets import load_dataset, load_from_disk

# Configure logger with custom prefix
logger = logging.getLogger('dataset_code')
handler = logging.StreamHandler()
formatter = logging.Formatter('[Running provided `dataset_code`] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def select_columns(dds, columns):
        to_drop = [c for c in dds.column_names if c not in columns]
        dds_small = dds.remove_columns(to_drop)
        return dds_small


def main(global_shared_data_dir: str, agent_data_mount_dir: str, agent_log_dir: str) -> None:
    """
    For each split we create:
        - <split>_search_corpus: A dataset containing the code snippets to be searched.
        - <split>_queries_with_labels: A dataset containing the natural language queries and the corresponding code snippet labels.

    We also provide test_queries which is a dataset containing only the natural language queries without labels.
    The agent has to generate predictions for these queries and return them in the submission.csv file.

    :param global_shared_data_dir: Path to the AIRS-Bench raw data directory.
    :param agent_data_mount_dir: Path to the agent data mount directory.
    :param agent_log_dir: Path to an agents log directory.
    """

    # Load from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, 'google/code_x_glue_tc_nl_code_search_adv/default')
    dataset = load_from_disk(dataset_source_fpath)

    for split in ['train', 'validation', 'test']:
        dds = dataset[split]

        # Search corpus is just id and code with docstring removed from code
        search_corpus = select_columns(dds, ["id", "code_tokens"])
        search_corpus = search_corpus.map(
            lambda example: {
                "code": ' '.join(example['code_tokens'])
            },
            remove_columns=["code_tokens"]
        )
        search_corpus.save_to_disk(os.path.join(agent_data_mount_dir, f'{split}/search_corpus'))

        # Queries are just docstring and the resulting code id
        queries_with_labels = select_columns(dds, ["docstring_tokens", "id"])
        queries_with_labels = queries_with_labels.map(
            lambda example: {
                "query": " ".join(example["docstring_tokens"])
            },
            remove_columns=["docstring_tokens"]
        )

        # shuffle the queries
        queries_with_labels = queries_with_labels.shuffle(seed=42)

        if split == 'test':
            # For the test set we do not provide the labels
            queries = select_columns(queries_with_labels, ["query"])
            queries.save_to_disk(os.path.join(agent_data_mount_dir, f'test/queries'))
        else:
            queries_with_labels.save_to_disk(os.path.join(agent_data_mount_dir, f'{split}/queries_with_labels'))


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