# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json
import pandas as pd
import json
from pathlib import Path
from datasets import load_from_disk
from utils import evaluate_all_testcases


def load_testcases(path="/home/agent/workspace/data/test_with_labels"):
    """
    Load testcases for evaluation.
    """
    ds = load_from_disk(path)
    return ds


def evaluate(submissions, testcases):
    """
    Compute Pass@5 metric for a list of submissions and testcases.
    """
    passAt5 = evaluate_all_testcases(submissions, testcases)
    return {"Pass@5": passAt5}


def _cli():
    p = argparse.ArgumentParser(description="Evaluate Pass@5 using submission.csv")
    p.add_argument("--submission-file", required=True,
                   help="Path to CSV with columns code1..code5")
    a = p.parse_args()

    print("Loading test set…")
    testcases = load_testcases()
    n_test_samples = len(testcases)

    print(f"Loading submissions from: {a.submission_file}")
    submission_df = pd.read_csv(a.submission_file, header=0)
    submission_scripts = submission_df[[f'code{i}' for i in range(1, 6)]].values.tolist()
    n_submissions = len(submission_scripts)
    assert n_submissions == n_test_samples, f"Submission file row count ({n_submissions}) does not match test set size ({n_test_samples})."

    print("Evaluating Pass@5…")
    result = evaluate(submission_scripts, testcases)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
