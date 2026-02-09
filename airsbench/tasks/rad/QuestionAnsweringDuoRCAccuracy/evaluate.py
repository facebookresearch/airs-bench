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


def load_test_answers(path='./data/test_with_labels'):
    """
    Load answers for evaluation (string labels).
    """
    ds = load_from_disk(path)
    labels = list(ds["answers"])
    no_answers = list(ds["no_answer"])
    return labels, no_answers

def is_correct_answer(submission, candidate_answers):
    submission = submission.lower()
    for candidate_answer in candidate_answers:
        if candidate_answer.lower() == submission:
            return True
    return False

def evaluate(submission_answers, submission_has_answers, label_answers, label_no_answers):
    """
    Returns a dict of metric_name -> value
    """
    correct = 0
    total = 0
    for submission_answer, submission_has_answer, label_answer, label_no_answer in zip(
        submission_answers, submission_has_answers, label_answers, label_no_answers
    ):
        total += 1
        if not submission_has_answer:
            correct += label_no_answer
        else:
            is_correct = is_correct_answer(submission_answer, label_answer)
            correct += is_correct
    return {"Accuracy": correct/total}


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for DuoRC test split using Accuracy."
    )
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    label_answers, label_no_answers = load_test_answers()
    n_test_samples = len(label_answers)
    print(f"Loaded {n_test_samples} labels.")

    print(f"Loading predictions from: {a.submission_file}")
    try:
        # Assuming CSV has no header and contains only prediction values
        # Adjust if your submission format is different (e.g., has headers, specific columns)
        submission_df = pd.read_csv(
            a.submission_file,
            header=0,
            keep_default_na=False
        )
        submission_answers = list(submission_df["answer"])
        submission_has_answers = list(submission_df["has_answer"])
        if len(submission_answers) != n_test_samples:
            raise ValueError(
                f"Submission file row count ({len(submission_answers)}) "
                f"does not match test set size ({n_test_samples})."
            )
    except FileNotFoundError:
        p.error(f"Submission file not found: {a.submission_file}")
    except Exception as e:
        p.error(f"Error loading submission_file: {e}")

    print("Evaluating predictions...")
    result = evaluate(submission_answers, submission_has_answers, label_answers, label_no_answers)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    _cli()
