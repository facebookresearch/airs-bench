#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json, re, string
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
import os

def load_test_set():
    dataset = load_from_disk('./data/test_with_labels')
    return [x["text"] for x in dataset["answers"]]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    Matches the official SQuAD evaluate-v1.1.py (Rajpurkar et al. 2016)."""
    s = str(s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    return s

def compute_exact(a_gold, a_pred):
    """Check whether normalized prediction matches normalized gold answer."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def format_prediction(pred):
    # handle edge case
    if pred == '"None"':
        return "None"
    return pred

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    Uses normalized exact match following the official SQuAD evaluate-v1.1.py.
    For each example, EM = max over gold answers of compute_exact(gold, pred).
    """
    exact_matches = 0
    for pred, label_list in zip(predictions, labels):
        pred_text = format_prediction(pred)
        # Max over all acceptable gold answers (official SQuAD protocol)
        if any(compute_exact(gold, pred_text) for gold in label_list):
            exact_matches += 1

    exact_match_score = exact_matches / len(labels)

    return {"ExactMatch": exact_match_score}
    


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
