#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset, load_from_disk
from sklearn.metrics import mean_absolute_error


def load_test_set():

    dataset = load_from_disk('/home/agent/workspace/data/test_with_labels')
    return dataset["label_target"]

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    all_preds = []
    all_labels = []
    test_ds = load_from_disk('/home/agent/workspace/data/test_with_labels')
    train_targets = test_ds["target"]

    for pred, label, train_target in zip(predictions, labels, train_targets):
        # Handle NaN values in prediction strings by replacing them with np.nan
        pred_str = pred.replace('NaN', 'null').replace('nan', 'null')
        try:
            pred_list = json.loads(pred_str)
            # Convert null values back to np.nan
            pred_list = [np.nan if x is None else x for x in pred_list]
            pred = np.array(pred_list)
        except json.JSONDecodeError:
            # Fallback to ast.literal_eval if JSON parsing fails
            pred = np.array(ast.literal_eval(pred))

        label = np.array(label)
        train_size = np.array(train_target).shape[0]

        # Extract forecast portion from full label sequence
        label_forecast = label[train_size:]

        # Predictions should already be 5-step forecasts from custom_labels.py
        if pred.shape != label_forecast.shape:
            raise ValueError(
                f"Invalid sample: Prediction shape {pred.shape} does not match "
                f"forecast label shape {label_forecast.shape}. Expected {5} forecast steps."
            )

        all_preds.append(pred)
        all_labels.append(label_forecast)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Flatten arrays
    all_preds_flat = all_preds.flatten()
    all_labels_flat = all_labels.flatten()

    # Remove NaN values - only evaluate on valid data points
    valid_mask = ~(np.isnan(all_preds_flat) | np.isnan(all_labels_flat))

    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points found for evaluation")

    valid_preds = all_preds_flat[valid_mask]
    valid_labels = all_labels_flat[valid_mask]

    mae = mean_absolute_error(valid_labels, valid_preds)
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
