#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from utils import calculate_scores


def load_test_set():
    return load_from_disk('/home/agent/workspace/data/test_with_labels')


def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    # Predictions should be pd.DataFrame with columns: query: str, rankings: json.dumps([list of ranked code ids])
    # Labels should be hf Dataset with keys query: str, id: code id

    # First json.loads the rankings column of predictions
    predictions['rankings'] = predictions['rankings'].apply(json.loads)

    # Map to format for calculate_scores
    # Predictions are {url: str -> [list of ranked code ids]}
    # Labels are {url: str -> code id}
    # We'll use the query as the url for both
    formatted_predictions = {
        q: pred.tolist() if isinstance(pred, np.ndarray) else pred
        for q, pred in zip(predictions['query'], predictions['rankings'])
    }
    formatted_labels = {
        q: label
        for q, label in zip(labels['query'], labels['id'])
    }
    return calculate_scores(formatted_labels, formatted_predictions)


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions"
    )
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    labels = load_test_set()
    n_test_samples = len(labels)
    print(f"Loaded {n_test_samples} labels.")

    print(f"Loading predictions from: {a.submission_file}")
    try:
        # Assuming CSV has no header and contains only prediction values
        # Adjust if your submission format is different (e.g., has headers, specific columns)
        preds = pd.read_csv(a.submission_file, header=0)
        if preds.shape[0] != n_test_samples:
            raise ValueError(
                f"Submission file row count ({preds.shape[0]}) "
                f"does not match test set size ({n_test_samples})."
            )
        # Further shape validation could be added here against spec['shape_per_item']
        # e.g., if preds.shape[1:] != spec['shape_per_item']
    except FileNotFoundError:
        p.error(f"Submission file not found: {a.submission_file}")
    except Exception as e:
        p.error(f"Error loading submission_file: {e}")

    print("Evaluating predictions...")
    result = evaluate(preds, labels)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    _cli()
