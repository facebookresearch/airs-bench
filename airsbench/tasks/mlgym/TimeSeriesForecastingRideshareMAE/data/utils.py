# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import Dataset
import numpy as np


def combine_lists(example):
    """Combine target (10 lists) and feat_dynamic_real (5 lists) into 15 lists."""
    combined = example["target"] + example["feat_dynamic_real"]
    return {"combined": combined}


def reformat_dataset(validation_ds, test_ds):
    """Transform time series data for forecasting evaluation.

    Args:
        validation_ds: Dataset with base sequence length (for input)
        test_ds: Dataset with extended sequence length (containing forecasts)

    Returns:
        Dataset with 'target' (input) and 'label_target' (extended with forecasts) columns
    """
    test_set = Dataset.from_dict({"target": validation_ds["combined"], "label_target": test_ds["combined"]})
    return test_set


def process_predictions_and_labels_for_evaluation(predictions, labels):
    """
    Process prediction and label arrays, handling NaN values for evaluation.

    Args:
        predictions: Array of flattened predictions
        labels: Array of flattened labels

    Returns:
        tuple: (valid_predictions, valid_labels) with NaN values removed
    """
    # Convert string 'NaN' to actual NaN for proper masking
    preds_clean = []
    labels_clean = []

    for pred_val, label_val in zip(predictions, labels):
        # Convert string 'NaN' to numpy NaN, handle other cases
        try:
            if pred_val == "NaN" or pred_val is None:
                pred_clean = np.nan
            else:
                pred_clean = float(pred_val)
        except (ValueError, TypeError):
            pred_clean = np.nan

        try:
            if label_val == "NaN" or label_val is None:
                label_clean = np.nan
            else:
                label_clean = float(label_val)
        except (ValueError, TypeError):
            label_clean = np.nan

        preds_clean.append(pred_clean)
        labels_clean.append(label_clean)

    preds_clean = np.array(preds_clean)
    labels_clean = np.array(labels_clean)

    # Remove NaN values - only evaluate on valid data points
    valid_mask = ~(np.isnan(preds_clean) | np.isnan(labels_clean))

    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points found for evaluation")

    valid_preds = preds_clean[valid_mask]
    valid_labels = labels_clean[valid_mask]

    return valid_preds, valid_labels


def parse_and_validate_predictions_and_labels(predictions, labels):
    """
    Parse JSON predictions and labels, handle NaN values, validate shapes.

    Args:
        predictions: List of JSON-encoded prediction strings
        labels: List of JSON-encoded label strings or arrays

    Returns:
        tuple: (flattened_predictions, flattened_labels) ready for evaluation
    """
    import ast
    import json

    all_preds = []
    all_labels = []

    for pred, label in zip(predictions, labels):
        # Handle NaN values in prediction strings by replacing them with np.nan
        pred_str = pred.replace("NaN", "null").replace("nan", "null")
        try:
            pred_list = json.loads(pred_str)
            # Convert null values back to np.nan
            pred_list = [np.nan if x is None else x for x in pred_list]
            pred = np.array(pred_list)
        except json.JSONDecodeError:
            # Fallback to ast.literal_eval if JSON parsing fails
            pred = np.array(ast.literal_eval(pred))

        # Handle NaN values in labels the same way as predictions
        if isinstance(label, str):
            label_str = label.replace("NaN", "null").replace("nan", "null")
            try:
                label_list = json.loads(label_str)
                # Convert null values back to np.nan
                label_list = [np.nan if x is None else x for x in label_list]
                label = np.array(label_list)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval if JSON parsing fails
                label = np.array(ast.literal_eval(label))
        else:
            label = np.array(label)

        # Both predictions and labels are already forecast portions from custom_labels.py
        # Just validate they have the same shape
        if pred.shape != label.shape:
            raise ValueError(
                f"Invalid sample: Prediction shape {pred.shape} does not match label shape {label.shape}."
            )

        all_preds.append(pred)
        all_labels.append(label)

    # Flatten and process predictions and labels, handling NaN values
    all_preds_flat = np.concatenate(all_preds).flatten()
    all_labels_flat = np.concatenate(all_labels).flatten()

    return all_preds_flat, all_labels_flat
