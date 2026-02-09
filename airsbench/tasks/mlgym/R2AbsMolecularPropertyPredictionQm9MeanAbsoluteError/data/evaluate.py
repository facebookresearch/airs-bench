#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json, numpy as np, pandas as pd
# Conditional torch import - might be added by generated functions if needed
# import torch
from datasets import load_dataset, load_from_disk


from datasets import load_dataset
import numpy as np

def load_test_set():
    dataset = load_from_disk('/home/agent/workspace/data/test_with_labels')
    return np.array(dataset["R_2_Abs"])


import torch
import numpy as np
from torchmetrics import MeanAbsoluteError

def evaluate(predictions, labels):
    # Convert numpy arrays to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Ensure inputs are torch tensors with float type
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    predictions = predictions.float()
    labels = labels.float()

    # Initialize and compute metric
    metric = MeanAbsoluteError(num_outputs=1)
    score = metric(predictions, labels)

    return {"MeanAbsoluteError": score.item()}


import numpy as np
import random
import string

def generate_dummy_predictions(n_samples):
    # Given nature is "unknown", we'll return scalar values as a reasonable default
    predictions = np.random.uniform(0, 1, size=(n_samples, 1))
    return predictions.squeeze()



def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for qm9 test split using MeanAbsoluteError."
    )
    p.add_argument("--dummy-submission", action="store_true",
                   help="Evaluate with randomly generated dummy predictions.")
    p.add_argument("--submission-file", default="submission.csv",
                   help="Path to CSV file containing predictions.")
    a = p.parse_args()

    print("Loading test set labels...")
    labels = load_test_set()
    n_test_samples = len(labels)
    print(f"Loaded {n_test_samples} labels.")

    if a.dummy_submission:
        print(f"Generating {n_test_samples} dummy predictions...")
        preds = generate_dummy_predictions(n_test_samples)
    else:
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
