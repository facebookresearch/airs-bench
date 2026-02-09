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


def load_test_set():

    dataset = load_from_disk('./data/test_with_labels')
    return np.array(dataset["answer"])



def evaluate(predictions, labels):
    """
    Compute QA accuracy for dreamerdeo/finqa.
    Returns only {"Accuracy": accuracy}.

    Matching rules:
      - If both prediction and label parse as numbers (after stripping currency
        symbols, commas, spaces; handling negatives in parentheses; converting
        percents), compare numerically with tolerance.
      - Otherwise, compare normalized strings (lowercased, trimmed, collapsed
        whitespace).
    """

    def is_nan(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    def normalize_text(s: str) -> str:
        s = str(s).strip().lower()
        # Collapse internal whitespace
        s = " ".join(s.split())
        return s

    def to_number(s: str):
        """
        Try to parse a string into a float.
        - Removes currency symbols and commas.
        - Handles negatives in parentheses: (123) -> -123
        - Handles percent: '5%' -> 0.05
        Returns float or None if parsing fails.
        """
        if s is None:
            return None
        ss = str(s).strip()
        if ss == "":
            return None

        neg = False
        # Handle negatives in parentheses, e.g., "(1,234.56)"
        if ss.startswith("(") and ss.endswith(")"):
            neg = True
            ss = ss[1:-1].strip()

        # Remove currency symbols and spaces
        ss = ss.replace("$", "").replace("£", "").replace("€", "")
        ss = ss.replace(",", "").replace(" ", "")

        is_percent = False
        if ss.endswith("%"):
            is_percent = True
            ss = ss[:-1]

        # Allow leading +/-
        try:
            val = float(ss)
        except Exception:
            return None

        if neg:
            val = -val
        if is_percent:
            val = val / 100.0
        return val

    # Coerce inputs to lists of strings; handle None/NaN
    preds = ["" if is_nan(p) else str(p) for p in np.asarray(predictions, dtype=object)]
    gts   = ["" if is_nan(t) else str(t) for t in np.asarray(labels, dtype=object)]

    if len(preds) != len(gts):
        raise ValueError(
            f"Number of predictions ({len(preds)}) does not match number of labels ({len(gts)})."
        )

    correct = 0
    n = len(gts)

    # Tolerances for numeric comparison
    ABS_TOL = 1e-4
    REL_TOL = 1e-4

    for p, t in zip(preds, gts):
        # Try numeric compare first
        pn = to_number(p)
        tn = to_number(t)

        if pn is not None and tn is not None:
            if abs(pn - tn) <= max(ABS_TOL, REL_TOL * max(1.0, abs(tn))):
                correct += 1
            continue

        # Fall back to normalized string exact match
        if normalize_text(p) == normalize_text(t):
            correct += 1

    acc = correct / n if n > 0 else 0.0
    return {"Accuracy": float(acc)}



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
