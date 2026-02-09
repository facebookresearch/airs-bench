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
import re
import copy
import random

hf_repo = 'google/code_x_glue_tc_nl_code_search_adv'
config = 'default'
test_split = 'test'

def select_columns(dds, columns):
        to_drop = [c for c in dds.column_names if c not in columns]
        dds_small = dds.remove_columns(to_drop)
        return dds_small

def save_as_csv(dds, output_fpath):
    """
    Takes a huggingface dataset with columns query: str, rankings: [list of ranked code ids]
    Saves as a CSV with header: query,rankings
    where rankings is json.dumps([list of ranked code ids])
    """
    dds = dds.map(
        lambda example: {
            "rankings": json.dumps(example["rankings"])
        }
    )
    df = dds.to_pandas()
    df.to_csv(output_fpath, index=False, header=["query", "rankings"])


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
        
    # Load from the raw data directory
    dataset_source_fpath = os.path.join(global_shared_data_dir, f"{hf_repo}/{config}")
    dataset = load_from_disk(dataset_source_fpath)
    dds = dataset[test_split]

    n_docs = len(dds)
    print(f"Loaded {n_docs} documents from the {test_split} split of the dataset.")

    # Submission format is a CSV with columns: query: str, rankings: [list of ranked code ids]
    queries_with_labels = select_columns(dds, ["docstring_tokens", "id"])
    queries_with_labels = queries_with_labels.map(
        lambda example: {
            "query": " ".join(example["docstring_tokens"]),
            "rankings": [example["id"]] + random.sample([i for i in range(n_docs) if i != example["id"]], random.randint(1, 200))  # 1 correct + random incorrect ids
        },
        remove_columns=["docstring_tokens", "id"]
    )
    # Save as CSV instead of a HuggingFace dataset
    csv_fpath = os.path.join(output_directory, 'gold_submission.csv')
    save_as_csv(queries_with_labels, csv_fpath)

    # Produce a worse summision by shuffling the rankings
    worse_queries = queries_with_labels.map(
        lambda example: {
             "rankings": random.sample(example["rankings"], len(example["rankings"]))
        },
    )
    # Save as CSV instead of a HuggingFace dataset
    csv_fpath = os.path.join(output_directory, 'gold_submission_permuted_1.csv')
    save_as_csv(worse_queries, csv_fpath)

    # And another worse submission by reversing the rankings
    worse_queries_2 = queries_with_labels.map(
        lambda example: {
                "rankings": list(reversed(example["rankings"]))
        },
    )
    # Save as CSV instead of a HuggingFace dataset
    csv_fpath = os.path.join(output_directory, 'gold_submission_permuted_2.csv')
    save_as_csv(worse_queries_2, csv_fpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold submission CSV from dataset.")
    parser.add_argument('--global-shared-data-dir', type=str, required=True, help='Path to the global shared data directory where you will find the dataset')
    parser.add_argument('--output-directory', type=str, required=True, help='Directory to save the output CSV')

    args = parser.parse_args()

    main(args.global_shared_data_dir, args.output_directory)
