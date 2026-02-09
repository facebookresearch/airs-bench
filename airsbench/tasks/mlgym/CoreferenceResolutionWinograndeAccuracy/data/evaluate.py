#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json
import numpy as np
import pandas as pd
from datasets import load_from_disk


def load_test_answers(path='/home/agent/workspace/data/test_with_labels'):
    """
    Load answers for evaluation (string labels).
    """
    ds = load_from_disk(path)
    if "answer" not in ds.features:
        raise RuntimeError(
            "Test data does not contain 'label'. "
            "Use a labeled copy like '/home/agent/workspace/data/test_with_labels'."
        )
    labels = list(ds["answer"])
    return labels


def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    correct = np.fromiter(
        (int(p) == int(y) for p, y in zip(predictions, labels)), dtype=bool
    )
    return {"Accuracy": float(correct.mean())}


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for Winogrande test split using Accuracy."
    )
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    labels = load_test_answers()
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
