# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import multiprocessing
import numpy as np
from tqdm import tqdm
from testing_util import run_test
from concurrent.futures import ThreadPoolExecutor, as_completed


def solves_testcases(submission, testcases, verbose=False):
    """
    Write submission once to a temp file and run it against all testcases.
    """
    timeout = 10

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(testcases, submission, verbose, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(testcases["input_output"])
        # consider that all tests failed
        result = [[0 for i in range(len(in_outs["inputs"]))]]
        if verbose:
            print("global timeout")
    result = result[0]
    fixed = []
    for e in result:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return np.all(fixed)


def _passes_any_submission(submissions, testcases, verbose=False):
    for submission in submissions:
        if solves_testcases(submission, testcases, verbose):
            return 1  # counts as correct
    return 0


def evaluate_all_testcases(submissions_all, testcases_all, verbose=False, max_workers=None):
    assert len(submissions_all) == len(testcases_all), "Number of submissions and testcases do not match"
    total = len(submissions_all)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_passes_any_submission, submissions, testcases, verbose)
            for submissions, testcases in zip(submissions_all, testcases_all)
        ]
        correct = 0
        for fut in tqdm(as_completed(futures), total=total):
            correct += fut.result()

    return correct / total
