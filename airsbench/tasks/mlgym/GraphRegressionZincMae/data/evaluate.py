#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json
import ast
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk


def load_test_set():

    dataset = load_from_disk('/home/agent/workspace/data/test_with_labels')
    return np.array(dataset["y"])



def evaluate(predictions, labels):
    """
    Compute Mean Absolute Error (MAE) for graph regression on ZINC.
    Returns only {"Mae": score}.
    """
    # Convert to numeric numpy arrays
    clean_predictions = []
    for p in predictions:
        if isinstance(p, str):
            parsed = ast.literal_eval(p)  # safely turns "[0.95]" into a Python list [0.95]
            if isinstance(parsed, list):
                clean_predictions.append(parsed[0])  # take the first element if it's a single-item list
            else:
                clean_predictions.append(float(parsed))
        else:
            clean_predictions.append(float(p))
    predictions = clean_predictions
    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)

    # Squeeze trailing singleton dims (e.g., shape (N,1) -> (N,))
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(-1)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: predictions {y_pred.shape} vs labels {y_true.shape}"
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions contain non-finite values (NaN/Inf).")
    if not np.all(np.isfinite(y_true)):
        raise ValueError("Labels contain non-finite values (NaN/Inf).")

    mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size > 0 else 0.0
    return {"MAE": mae}



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
        submission_df = pd.read_csv(a.submission_file, header=0)
        preds = submission_df.values.squeeze()
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
