#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset, load_from_disk
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error


def load_test_set():
    
    dataset = load_from_disk('/home/agent/workspace/data/test_with_labels')
    return dataset["label_target"]

def safe_literal_eval_with_nan(s):
    import ast
    import math
    s_fixed = s.replace('NaN', 'None')
    lst = ast.literal_eval(s_fixed)
    return lst

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    mases = []
    test_ds = load_from_disk('/home/agent/workspace/data/test_with_labels')
    train_targets = test_ds["target"]

    for pred, label, train_target in zip(predictions, labels, train_targets):
        try:
            pred = np.array(safe_literal_eval_with_nan(pred))
        except Exception as e:
            raise ValueError(f"Error parsing prediction: {pred}, with error {e}") from e
        label = np.array(label)
        
        if pred.shape != label.shape:
            raise ValueError(
                "Invalid sample: "
                f"Prediction shape {pred.shape} does not match "
                f"label shape {label.shape}"
            )

        train_target = np.array(train_target)
        train_size = train_target.shape[0]
        # some starting values in train_target can be NaN, remove them
        train_target = train_target[~np.isnan(train_target)]
        # remove first train_size elements from pred and label
        pred = pred[train_size:]
        label = label[train_size:]

        #find any nans in label
        mask = ~np.isnan(label)
        pred = pred[mask]
        label = label[mask]

        # there is one sample that seems to have all nans in the label after filtering
        if label.shape[0] == 0:
            continue
    
        mase = mean_absolute_scaled_error(y_true=label, y_pred=pred, y_train=train_target)
        mases.append(mase)

    return {"MASE": np.mean(mases)}




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
