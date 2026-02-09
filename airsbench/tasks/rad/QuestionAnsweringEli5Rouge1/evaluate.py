#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, json, numpy as np, pandas as pd
# Conditional torch import - might be added by generated functions if needed
#import torch
from datasets import load_dataset, load_from_disk
import torch
import numpy as np
import evaluate
import numpy
from typing import List, Dict
from rouge_score import rouge_scorer, scoring


def compute_rouge(preds, refs, types=("rouge1",), stemmer=True):
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=stemmer)
    agg = scoring.BootstrapAggregator()
    for p, r in zip(preds, refs):
        agg.add_scores(scorer.score(r, p))  # (reference, prediction)
    out = agg.aggregate()
    # return mid F1 like TorchMetrics
    return {k: v.mid.fmeasure for k, v in out.items()}


def load_test_set():
    # dataset = load_dataset('Pavithree/eli5', 'default', split='test')
    dataset = load_from_disk('./data/test_with_labels')
    ground_truth = dataset.map(
        lambda x: {"answers": x["answers"]["text"][0] if x["answers"]["text"] else None},
        load_from_cache_file=False,
        keep_in_memory=True,
    )['answers'] # dict['answers'] = list
    return ground_truth


def evaluate_function(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Computes the ROUGE-1 metric for a molecule captioning task.

    Args:
        predictions (List[str]): A list of predicted captions.
        labels (List[str]): A list of ground truth captions.

    Returns:
        Dict[str, float]: A dictionary containing the ROUGE-1 F-measure score.
    """

    print(f"Calling evaluate on {len(predictions)} predictions and {len(labels)} labels...")

    results = compute_rouge(predictions, labels)
    rouge1_fmeasure = float(results["rouge1"])

    return {"Rouge1": rouge1_fmeasure}


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for eli5 test split using ROUGE-1."
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
    result = evaluate_function(preds, labels)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    _cli()
