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
from utils import parse_and_validate_predictions_and_labels, process_predictions_and_labels_for_evaluation


def load_test_set():
    """Load test labels and extract forecast portions to match custom_labels.py output"""
    import json
    from utils import reformat_dataset

    dataset = load_from_disk("./data/test_with_labels")

    # The dataset contains the full sequences, but we need just the forecast portions
    # like custom_labels.py generates - 48 values per individual time series
    validation_targets = dataset["target"]  # Base sequences (validation length)
    full_test_targets = dataset["label_target"]  # Full test sequences

    forecast_labels = []
    for val_target, full_target in zip(validation_targets, full_test_targets):
        for i in range(len(full_target)):  # Process each series individually
            val_series_len = len(val_target[i])
            full_series = full_target[i]

            # Extract available forecast portion - keep NaNs in original positions
            available_forecast = full_series[val_series_len:]

            # Take exactly 48 values, padding at the END with NaN if needed
            forecast_48 = available_forecast[:48]  # Take up to 48 values
            while len(forecast_48) < 48:  # Pad at the END if shorter
                forecast_48.append(np.nan)

            # Convert to JSON string like custom_labels.py does
            # Handle NaN values
            forecast_clean = []
            for val in forecast_48:
                if isinstance(val, float) and np.isnan(val):
                    forecast_clean.append("NaN")
                else:
                    forecast_clean.append(val)

            # Each individual time series becomes one row with 48 forecasts
            forecast_labels.append(json.dumps(forecast_clean))

    return forecast_labels


def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """

    all_preds_flat, all_labels_flat = parse_and_validate_predictions_and_labels(predictions, labels)
    valid_preds, valid_labels = process_predictions_and_labels_for_evaluation(all_preds_flat, all_labels_flat)
    mae = mean_absolute_error(valid_labels, valid_preds)
    return {"MAE": mae}


def _cli():
    p = argparse.ArgumentParser(description="Evaluate predictions")
    p.add_argument("--submission-file", default="submission.csv", help="Path to CSV file containing predictions.")
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
                f"Submission file row count ({preds.shape[0]}) does not match test set size ({n_test_samples})."
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


if __name__ == "__main__":
    _cli()
