#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import json
import re
import argparse
import sys
import tempfile
import math
import pandas as pd
import numpy as np
import yaml
import ast

# Globals to be set in __main__
GLOBAL_SHARED_DATA_DIR = None
AGENT_DATA_MOUNT_DIR = None
AGENT_LOG_DIR = None
INPUT_FOLDER = None


def run_script(script_name, args_list, cwd=None, debug=False):
    """
    Runs a Python script from INPUT_FOLDER with given args_list, returns stdout.
    Raises RuntimeError on failure. Can specify working directory cwd.
    """
    cmd = [sys.executable, os.path.join(INPUT_FOLDER, script_name)] + args_list
    if not debug:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    else:
        result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


def assert_dir_nonempty(path, name):
    if not os.path.isdir(path):
        raise AssertionError(f"{name} directory was not created at {path}")
    if not os.listdir(path):
        raise AssertionError(f"{name} directory at {path} is empty")


def test_prepare(expected_resultant_folders=["train", "test"]):
    """
    Test that prepare.py creates non-empty train and test directories.
    """
    args = [
        "--global-shared-data-dir",
        GLOBAL_SHARED_DATA_DIR,
        "--agent-data-mount-dir",
        AGENT_DATA_MOUNT_DIR,
        "--agent-log-dir",
        AGENT_LOG_DIR,
    ]
    run_script("prepare.py", args)

    for folder in expected_resultant_folders:
        assert_dir_nonempty(os.path.join(AGENT_DATA_MOUNT_DIR, folder), folder)

    print("✅ prepare.py test passed.")


def test_evaluate_prepare(
    submission_files=[("example_submission.csv", "submission.csv")],
    expected_resultant_folders=["test_with_labels"],
    expected_resultant_files=["submission.csv"],
):
    """
    Test that evaluate_prepare.py copies submission and creates test_with_labels.
    """
    for submission_file_source, submission_file_dest in submission_files:
        example_csv = os.path.join(INPUT_FOLDER, submission_file_source)
        if not os.path.isfile(example_csv):
            raise AssertionError(f"{submission_file_source} is missing in {INPUT_FOLDER}")

        dst = os.path.join(AGENT_LOG_DIR, submission_file_dest)
        shutil.copy(example_csv, dst)
        if not os.path.isfile(dst):
            raise AssertionError(f"Failed to copy {submission_file_dest} to {AGENT_LOG_DIR}")

    # Copy submission_files directory if it exists (for multimodal tasks)
    submission_files_src = os.path.join(INPUT_FOLDER, "submission_files")
    if os.path.exists(submission_files_src) and os.path.isdir(submission_files_src):
        submission_files_dst = os.path.join(AGENT_LOG_DIR, "submission_files")
        if os.path.exists(submission_files_dst):
            shutil.rmtree(submission_files_dst)
        shutil.copytree(submission_files_src, submission_files_dst)
        print(f"Copied submission_files directory from {submission_files_src} to {submission_files_dst}")

    args = [
        "--global-shared-data-dir",
        GLOBAL_SHARED_DATA_DIR,
        "--agent-data-mount-dir",
        AGENT_DATA_MOUNT_DIR,
        "--agent-log-dir",
        AGENT_LOG_DIR,
    ]
    run_script("evaluate_prepare.py", args)

    for folder in expected_resultant_folders:
        assert os.path.isdir(os.path.join(AGENT_DATA_MOUNT_DIR, folder)), (
            f"{folder} directory not created at {os.path.join(AGENT_DATA_MOUNT_DIR, folder)}"
        )

    for file in expected_resultant_files:
        assert os.path.isfile(os.path.join(AGENT_DATA_MOUNT_DIR, file)), f"{file} not found in {AGENT_DATA_MOUNT_DIR}"

    print(f"✅ evaluate_prepare.py test passed for {submission_files}.")


def test_evaluate(expected_metric_name: str):
    """
    Simulate container environment: mount AGENT_DATA_MOUNT_DIR at ./data, then run evaluate.py without
    """
    with tempfile.TemporaryDirectory(prefix="container_") as container_dir:
        # simulate bind-mount: create 'data' symlink to AGENT_DATA_MOUNT_DIR
        data_path = os.path.join(container_dir, "data")
        os.symlink(AGENT_DATA_MOUNT_DIR, data_path)
        # ensure log directory is available if needed
        logs_path = os.path.join(container_dir, "logs")
        os.symlink(AGENT_LOG_DIR, logs_path)

        # run evaluate.py inside container_dir
        stdout = run_script("evaluate.py", ["--submission-file", "./data/submission.csv"], cwd=container_dir)

    pattern = re.compile(r"--- EVALUATION RESULT ---\s*(\{[\s\S]*?\})", re.DOTALL)
    match = pattern.search(stdout)
    if not match:
        raise AssertionError(f"Failed to extract results dict from stdout: {stdout}")

    result_str = match.group(1)
    try:
        result_dict = json.loads(result_str)
    except json.JSONDecodeError:
        raise AssertionError(f"Failed to parse JSON: {result_str}")
    if not result_dict:
        raise AssertionError("Results dict is empty")
    assert expected_metric_name in result_dict, (
        f"Expected metric '{expected_metric_name}' not found in results: {result_dict}"
    )
    values = list(result_dict.values())
    if len(values) != 1:
        raise AssertionError(f"Expected single result, got {len(values)}: {result_dict}")
    result = values[0]
    if not isinstance(result, (int, float)):
        raise AssertionError(f"Result is not numeric: {result}")
    print(f"✅ evaluate.py test passed (result = {result})")
    # print(stdout)
    return result


def check_gold_score(task_full_path, task_name, gold_score, lower_is_better):
    if task_name == "KeywordExtractionInspecF1at10":
        assert math.isclose(gold_score, 0.9297514511638763), (
            f"❌ gold score for {task_full_path} is not the maximum of 0.92975"
        )
    elif task_name == "CodeGenerationAPPSPassAt5":
        assert math.isclose(gold_score, 0.753), f"❌ gold score for {task_full_path} is not the maximum of 0.753"
    elif task_name == "ProteinDesignCATH43Perplexity" or task_name == "ProteinDesignCATH42Perplexity":
        assert math.isclose(gold_score, 1.0), (
            f"❌ gold score for {task_full_path} is not the optimal perplexity of 1.0: {gold_score}"
        )
    elif lower_is_better:
        assert gold_score == 0.0, (
            f"❌ lower_is_better is {lower_is_better} for task and gold score for {task_full_path} is not 0.0: {gold_score}"
        )
    elif not lower_is_better:
        assert gold_score == 1.0, (
            f"❌ lower_is_better is {lower_is_better} for task and gold score for {task_full_path} is not 1.0: {gold_score}"
        )
    else:
        raise AssertionError(f"Unable to extract metric from task name: {task_full_path}")
    print(f"✅ gold score for {task_name} is correct")


def check_metadata(metadata, input_folder_name):
    required_fields = [
        "metric_lower_is_better",
        "file_export_globs",
        "container_python_requirements",
        "evaluate_container_python_requirements",
        "logging_info",
    ]
    for field in required_fields:
        if field not in metadata:
            print(f"Error: {field} not found in metadata", file=sys.stderr)
            raise AssertionError(f"{field} not found in metadata.yaml")
        if field == "metric_lower_is_better":
            if not isinstance(metadata[field], bool):
                print(f"Error: metric_lower_is_better must be a boolean in metadata", file=sys.stderr)
                raise AssertionError("metric_lower_is_better must be a boolean")

    required_logging_info_fields = [
        'name',
        'dataset',
        'category',
        'research_problem',
        'output_type',
        'config',
        'train_split',
        'test_split',
        'input_columns',
        'scoring_column',
        'custom_gold_labels',
        'custom_rad_class',
        'metric',
        'sota',
        'dataset_paper_url'
    ]

    required_sota_fields = [
        'sota_paper_url',
        'sota_score'
    ]


    if 'submission.csv' in metadata.get('file_export_globs', []):
        required_logging_info_fields.append('shape')
    else:
        print("Note: 'submission.csv' not in file_export_globs, skipping 'shape' check")

    for field in required_logging_info_fields:
        if field not in metadata["logging_info"]:
            print(f"Error: logging_info.{field} not found in metadata", file=sys.stderr)
            raise AssertionError(f"logging_info.{field} not found in metadata.yaml")
        value = metadata["logging_info"][field]
        if (
            value is None
            or (isinstance(value, str) and value.strip() == "")
            or (isinstance(value, list) and len(value) == 0)
        ):
            print(f"Error: logging_info.{field} is empty in metadata", file=sys.stderr)
            raise AssertionError(f"logging_info.{field} is empty in metadata.yaml")
        if field == "input_columns":
            if not isinstance(metadata["logging_info"][field], list) or not all(
                isinstance(col, str) for col in metadata["logging_info"][field]
            ):
                print(f"Error: logging_info.input_columns must be a list of strings in metadata", file=sys.stderr)
                raise AssertionError("logging_info.input_columns must be a list of strings")
        if field == "shape":
            val = ast.literal_eval(value) if isinstance(value, str) else value
            # Ensure its a tuple of atleast 1 integer(s)
            if not (
                isinstance(val, (list, tuple))
                and all(isinstance(dim, int) and dim > 0 for dim in val)
                and len(val) >= 1
            ):
                print(
                    f"Error: logging_info.shape must be a tuple of at least one positive integer in metadata",
                    file=sys.stderr,
                )
                raise AssertionError("logging_info.shape must be a tuple of at least one positive integer")
        if field == 'sota':
            for i in range(len(metadata["logging_info"]['sota'])):
                for k in required_sota_fields:
                    if k not in metadata["logging_info"]['sota'][i]:
                        print(f"Error: logging_info.sota.{i}.{k} not found in metadata", file=sys.stderr)
                        raise AssertionError(f"logging_info.sota.{i}.{k} not found in metadata.yaml")
                    value = metadata["logging_info"]['sota'][i][k]
                    if value is None or (isinstance(value, str) and value.strip() == ""):
                        print(f"Error: logging_info.sota.{i}.{k} is empty in metadata")
                        raise ValueError(f"Error: logging_info.sota.{i}.{k} is empty in metadata")

    assert metadata["logging_info"]["name"] == input_folder_name, (
        f"logging_info.name '{metadata['logging_info']['name']}' does not match folder name '{input_folder_name}'"
    )

    print("✅ Metadata check passed")


def check_project_description(assert_ends_with_evaluate):
    # Check that project_description.md exists in input folder
    project_description_path = os.path.join(INPUT_FOLDER, "project_description.md")
    if not os.path.isfile(project_description_path):
        print(f"Error: project_description.md not found at {project_description_path}", file=sys.stderr)
        raise AssertionError("project_description.md not found in input folder")

    if assert_ends_with_evaluate:
        with open(project_description_path, "r") as f:
            project_description = f.read()

        # Check that it ends in evaluate.py
        evaluate_path = os.path.join(INPUT_FOLDER, "evaluate.py")
        if not os.path.isfile(evaluate_path):
            print(f"Error: evaluate.py not found at {evaluate_path}", file=sys.stderr)
            raise AssertionError("evaluate.py not found in input folder")

        with open(evaluate_path, "r") as f:
            evaluate_code = f.read()

        evaluate_code = f"```py\n{evaluate_code.strip()}\n```"

        if evaluate_code not in project_description:
            print(f"Error: project_description.md does not contain the evaluate.py code", file=sys.stderr)

            print(project_description)

            print(f"[{INPUT_FOLDER}] Should i insert it for you? (y/n)")
            to_continue = input()

            if to_continue.lower() == "y":
                with open(project_description_path, "a") as f:
                    f.write("\n\n")
                    f.write(evaluate_code)
                print(f"Inserted evaluate.py code into project_description.md")
            else:
                raise AssertionError("project_description.md does not contain the evaluate.py code")


def has_gold_submission_csv():
    if not os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission.csv")):
        return False
    return True


def has_all_gold_submission_csvs():
    if not os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission.csv")):
        return False
    if not os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_permuted_1.csv")):
        return False
    if not os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_permuted_2.csv")):
        return False
    return True


def has_all_gold_submission_folders():
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission")):
        return False
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission_random_1")):
        return False
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission_random_2")):
        return False
    return True


def has_all_gold_submission_LRA_folders():
    """Check for LRA-specific gold submission folder structure."""
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission_trained")):
        return False
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission_random_1")):
        return False
    if not os.path.isdir(os.path.join(INPUT_FOLDER, "gold_submission_random_2")):
        return False
    return True


def switch_rows(submission, switch_count):
    permuted_submission = submission.copy()
    if len(permuted_submission) > 1:
        for _ in range(switch_count):
            i, j = np.random.choice(len(permuted_submission), 2, replace=False)
            permuted_submission.iloc[[i, j]] = permuted_submission.iloc[[j, i]].values
    return permuted_submission


def create_gold_permutations_csvs():
    # load up the gold submission and permute the labels
    gold_submission = pd.read_csv(os.path.join(INPUT_FOLDER, "gold_submission.csv"))

    permuted_1_path = os.path.join(INPUT_FOLDER, "gold_submission_permuted_1.csv")
    permuted_2_path = os.path.join(INPUT_FOLDER, "gold_submission_permuted_2.csv")

    if not os.path.isfile(permuted_1_path):
        gold_submission_permuted_1 = switch_rows(gold_submission, len(gold_submission) // 5)
        gold_submission_permuted_1.to_csv(permuted_1_path, index=False)
        print("gold_submission_permuted_1.csv created (few random switches)")
    else:
        print("gold_submission_permuted_1.csv already exists, skipping creation")

    if not os.path.isfile(permuted_2_path):
        gold_submission_permuted_2 = switch_rows(gold_submission, len(gold_submission))
        gold_submission_permuted_2.to_csv(permuted_2_path, index=False)
        print("gold_submission_permuted_2.csv created (many random switches)")
    else:
        print("gold_submission_permuted_2.csv already exists, skipping creation")

    assert os.path.isfile(permuted_1_path)
    assert os.path.isfile(permuted_2_path)


def create_gold_submission_from_test_labels(metadata):
    from save_test_labels_as_example_submission import main as save_test_labels_as_example_submission

    save_test_labels_as_example_submission(
        dataset_location=os.path.join(GLOBAL_SHARED_DATA_DIR, metadata["logging_info"]["dataset"]),
        target_column=metadata["logging_info"]["scoring_column"],
        output_directory=os.path.join(INPUT_FOLDER),
        config=metadata["logging_info"]["config"],
        test_split=metadata["logging_info"]["test_split"],
    )
    assert os.path.isfile(os.path.join(INPUT_FOLDER, "gold_submission.csv"))
    print("save_test_labels_as_example_submission.py created gold_submission.csv")

    create_gold_permutations_csvs()
    print("All gold submission files are now present")


def check_gold_submission_csvs(metadata):
    assert has_all_gold_submission_csvs(), "Not all gold_submission_*.csv files are present"
    print("All gold_submission_*.csv files are present")

    try:
        metric_name = metadata["logging_info"]["metric"]
        test_prepare(expected_resultant_folders=["train", "test"])
        test_evaluate_prepare(
            submission_files=[("gold_submission_permuted_1.csv", "submission.csv")],
            expected_resultant_folders=["test_with_labels"],
            expected_resultant_files=["submission.csv"],
        )
        example_result = test_evaluate(metric_name)
        test_evaluate_prepare(
            submission_files=[("gold_submission_permuted_2.csv", "submission.csv")],
            expected_resultant_folders=["test_with_labels"],
            expected_resultant_files=["submission.csv"],
        )
        example_result_permuted = test_evaluate(metric_name)
        test_evaluate_prepare(
            submission_files=[("gold_submission.csv", "submission.csv")],
            expected_resultant_folders=["test_with_labels"],
            expected_resultant_files=["submission.csv"],
        )
        gold_result = test_evaluate(metric_name)

        if example_result > 0:
            assert example_result != example_result_permuted, "❌ Test Failed - Metric should change with permutation"
            print("✅ Test Passed - Metric changed with permutation")
        if metadata["metric_lower_is_better"]:
            assert example_result > gold_result, (
                "❌ Test Failed - gold_submission_permuted_1 should be worse (higher) than gold"
            )
            print("✅ Test Passed - gold_submission_permuted_1 is worse (higher) than gold")
        else:
            assert example_result < gold_result, (
                "❌ Test Failed - gold_submission_permuted_1 should be worse (lower) than gold"
            )
            print("✅ Test Passed - gold_submission_permuted_1 is worse (lower) than gold")

        lower_is_better = metadata["metric_lower_is_better"]
        check_gold_score(INPUT_FOLDER, metadata["logging_info"]["name"], gold_result, lower_is_better)

    except AssertionError as e:
        print(f"Test failed: {e}", file=sys.stderr)
        raise e


def check_gold_submission_folders(metadata):
    assert has_all_gold_submission_folders(), "Not all gold_submission_* directories are present"
    print("All gold_submission_* directories are present")

    try:
        metric_name = metadata["logging_info"]["metric"]
        exported_files = metadata["file_export_globs"]
        gold_submission_random_1_files = [(f"gold_submission_random_1/{f}", f) for f in exported_files]
        test_prepare(expected_resultant_folders=["train"])
        test_evaluate_prepare(
            submission_files=gold_submission_random_1_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        example_result = test_evaluate(metric_name)
        gold_submission_random_2_files = [(f"gold_submission_random_2/{f}", f) for f in exported_files]
        test_evaluate_prepare(
            submission_files=gold_submission_random_2_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        example_result_random = test_evaluate(metric_name)
        gold_submission_files = [(f"gold_submission/{f}", f) for f in exported_files]
        test_evaluate_prepare(
            submission_files=gold_submission_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        gold_result = test_evaluate(metric_name)

        if example_result > 0:
            assert example_result != example_result_random, (
                "❌ Test Failed - Metric should change between random folders"
            )
            print("✅ Test Passed - Metric changed with random weights")
        if metadata["metric_lower_is_better"]:
            assert example_result > gold_result, (
                "❌ Test Failed - gold_submission_random_1 should be worse (higher) than gold"
            )
            print("✅ Test Passed - gold_submission_random_1 is worse (higher) than gold")
        else:
            assert example_result < gold_result, (
                "❌ Test Failed - gold_submission_random_1 should be worse (lower) than gold"
            )
            print("✅ Test Passed - gold_submission_random_1 is worse (lower) than gold")

    except AssertionError as e:
        print(f"Test failed: {e}", file=sys.stderr)
        raise e


def check_gold_submission_LRA(metadata):
    """Check LRA-specific gold submission folders (trained vs random models)."""
    assert has_all_gold_submission_LRA_folders(), "Not all gold_submission_LRA directories are present"
    print("All gold_submission_LRA directories are present")

    try:
        metric_name = metadata["logging_info"]["metric"]
        # LRA tasks need to prepare datasets from TSV files
        test_prepare(expected_resultant_folders=["train"])

        # Test random model 1
        submission_files = [
            ("gold_submission_random_1/evaluate.py", "evaluate.py"),
            ("gold_submission_random_1/model.py", "model.py"),
            ("gold_submission_random_1/train.py", "train.py"),
            ("gold_submission_random_1/input_pipeline.py", "input_pipeline.py"),
            ("gold_submission_random_1/train_utils.py", "train_utils.py"),
        ]

        # Always add build_vocab.py and vocab_file.subwords if they exist in this submission directory
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_random_1/build_vocab.py")):
            submission_files.append(("gold_submission_random_1/build_vocab.py", "build_vocab.py"))
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_random_1/vocab_file.subwords")):
            submission_files.append(("gold_submission_random_1/vocab_file.subwords", "vocab_file.subwords"))

        test_evaluate_prepare(
            submission_files=submission_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        # Copy configs and models directories
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_random_1/configs"),
            os.path.join(AGENT_LOG_DIR, "configs"),
            dirs_exist_ok=True,
        )
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_random_1/models"),
            os.path.join(AGENT_LOG_DIR, "models"),
            dirs_exist_ok=True,
        )
        print(f"🔍 DEBUG: About to run random_1 evaluation with metric_name='{metric_name}'")
        # Run evaluation (untrained model)
        random_result_1 = test_evaluate(metric_name)
        print(f"🔍 DEBUG: random_result_1 = {random_result_1}")

        # Test random model 2
        submission_files = [
            ("gold_submission_random_2/evaluate.py", "evaluate.py"),
            ("gold_submission_random_2/model.py", "model.py"),
            ("gold_submission_random_2/train.py", "train.py"),
            ("gold_submission_random_2/input_pipeline.py", "input_pipeline.py"),
            ("gold_submission_random_2/train_utils.py", "train_utils.py"),
        ]

        # Always add build_vocab.py and vocab_file.subwords if they exist in this submission directory
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_random_2/build_vocab.py")):
            submission_files.append(("gold_submission_random_2/build_vocab.py", "build_vocab.py"))
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_random_2/vocab_file.subwords")):
            submission_files.append(("gold_submission_random_2/vocab_file.subwords", "vocab_file.subwords"))

        test_evaluate_prepare(
            submission_files=submission_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        # Copy configs and models directories
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_random_2/configs"),
            os.path.join(AGENT_LOG_DIR, "configs"),
            dirs_exist_ok=True,
        )
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_random_2/models"),
            os.path.join(AGENT_LOG_DIR, "models"),
            dirs_exist_ok=True,
        )
        print(f"🔍 DEBUG: About to run random_2 evaluation with metric_name='{metric_name}'")
        # Run evaluation (untrained model)
        random_result_2 = test_evaluate(metric_name)
        print(f"🔍 DEBUG: random_result_2 = {random_result_2}")

        # Test trained model
        submission_files = [
            ("gold_submission_trained/evaluate.py", "evaluate.py"),
            ("gold_submission_trained/model.py", "model.py"),
            ("gold_submission_trained/train.py", "train.py"),
            ("gold_submission_trained/input_pipeline.py", "input_pipeline.py"),
            ("gold_submission_trained/train_utils.py", "train_utils.py"),
        ]

        # Always add build_vocab.py and vocab_file.subwords if they exist in this submission directory
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_trained/build_vocab.py")):
            submission_files.append(("gold_submission_trained/build_vocab.py", "build_vocab.py"))
        if os.path.exists(os.path.join(INPUT_FOLDER, "gold_submission_trained/vocab_file.subwords")):
            submission_files.append(("gold_submission_trained/vocab_file.subwords", "vocab_file.subwords"))

        test_evaluate_prepare(
            submission_files=submission_files, expected_resultant_files=[], expected_resultant_folders=[]
        )
        # Copy configs and models directories
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_trained/configs"),
            os.path.join(AGENT_LOG_DIR, "configs"),
            dirs_exist_ok=True,
        )
        shutil.copytree(
            os.path.join(INPUT_FOLDER, "gold_submission_trained/models"),
            os.path.join(AGENT_LOG_DIR, "models"),
            dirs_exist_ok=True,
        )
        # Copy marker files
        trained_marker = os.path.join(INPUT_FOLDER, "gold_submission_trained/TRAINED_SUBMISSION_MARKER.txt")
        if os.path.exists(trained_marker):
            shutil.copy2(trained_marker, os.path.join(AGENT_LOG_DIR, "TRAINED_SUBMISSION_MARKER.txt"))

        # Run evaluation with training enabled by using subprocess call with --train-model flag
        try:
            result = subprocess.run(
                [sys.executable, "evaluate.py", "--submission-file", "submission.csv", "--train-model"],
                capture_output=True,
                text=True,
                timeout=3600 * 24,
                cwd=AGENT_LOG_DIR,
            )

            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split("\n")
                json_started = False
                json_lines = []

                for line in lines:
                    if line.strip() == "--- EVALUATION RESULT ---":
                        json_started = True
                        continue
                    elif json_started:
                        json_lines.append(line)

                if json_lines:
                    json_str = "\n".join(json_lines)
                    result_dict = json.loads(json_str)
                    print(f"🔍 DEBUG: metric_name='{metric_name}', result_dict keys={list(result_dict.keys())}")
                    print(f"🔍 DEBUG: result_dict={result_dict}")

                    # Get the result - ensure we get the actual value, not 0.0 default
                    if metric_name in result_dict:
                        trained_result = float(result_dict[metric_name])
                    else:
                        # Fallback: try to get any numeric value from the dict
                        trained_result = 0.0
                        for key, value in result_dict.items():
                            if isinstance(value, (int, float)) and value > 0:
                                trained_result = float(value)
                                print(f"🔍 DEBUG: Using fallback key '{key}' with value {value}")
                                break

                    print(f"🔍 DEBUG: trained_result={trained_result}")
                else:
                    trained_result = 0.0
                    print(f"🔍 DEBUG: No JSON lines found, setting trained_result=0.0")
            else:
                trained_result = 0.0
        except:
            trained_result = 0.0

        # Validate results
        if random_result_1 > 0:
            assert random_result_1 != random_result_2, "❌ Test Failed - Metric should change between random models"
            print("✅ Test Passed - Metric changed between random models")

        if metadata["metric_lower_is_better"]:
            assert random_result_1 > trained_result, (
                "❌ Test Failed - random model should be worse (higher) than trained"
            )
            print("✅ Test Passed - random model is worse (higher) than trained")
        else:
            assert random_result_1 < trained_result, (
                "❌ Test Failed - random model should be worse (lower) than trained"
            )
            print("✅ Test Passed - random model is worse (lower) than trained")

    except AssertionError as e:
        print(f"LRA test failed: {e}", file=sys.stderr)
        raise e


def main(
    global_shared_data_dir: str,
    agent_data_mount_dir: str,
    agent_log_dir: str,
    input_folder: str,
    assert_ends_with_evaluate: bool = False,
):
    global GLOBAL_SHARED_DATA_DIR, AGENT_DATA_MOUNT_DIR, AGENT_LOG_DIR, INPUT_FOLDER  # forgive me father, for I have sinned

    GLOBAL_SHARED_DATA_DIR = os.path.abspath(global_shared_data_dir)
    INPUT_FOLDER = os.path.abspath(input_folder)
    input_folder_name = os.path.basename(INPUT_FOLDER)

    print(f"Testing INPUT_FOLDER: {INPUT_FOLDER}")

    if agent_data_mount_dir:
        AGENT_DATA_MOUNT_DIR = os.path.abspath(agent_data_mount_dir)
    else:
        AGENT_DATA_MOUNT_DIR = tempfile.mkdtemp(prefix="agent_data_mount_")

    if agent_log_dir:
        AGENT_LOG_DIR = os.path.abspath(agent_log_dir)
    else:
        AGENT_LOG_DIR = tempfile.mkdtemp(prefix="agent_log_")

    # Clean/Recreate mount & log dirs only
    for d in (AGENT_DATA_MOUNT_DIR, AGENT_LOG_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    if not os.path.isdir(GLOBAL_SHARED_DATA_DIR):
        print(f"Error: GLOBAL_SHARED_DATA_DIR not found at {GLOBAL_SHARED_DATA_DIR}", file=sys.stderr)
        raise AssertionError("GLOBAL_SHARED_DATA_DIR not found")

    # Check that metadata.yaml exists in input folder
    # and that the logging_info.dataset and metric_lower_is_better fields are set
    metadata_path = os.path.join(INPUT_FOLDER, "metadata.yaml")
    if not os.path.isfile(metadata_path):
        print(f"Error: metadata.yaml not found at {metadata_path}", file=sys.stderr)
        raise AssertionError("metadata.yaml not found in input folder")

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    check_metadata(metadata, input_folder_name)
    check_project_description(assert_ends_with_evaluate)

    # First check if we have all the gold_submission_*.csv's or gold_submission/ folders
    if has_all_gold_submission_csvs():
        check_gold_submission_csvs(metadata)
    elif has_all_gold_submission_folders():
        check_gold_submission_folders(metadata)
    elif has_all_gold_submission_LRA_folders():
        check_gold_submission_LRA(metadata)

    # If not, see if we have custom_labels.py to create them
    elif os.path.isfile(os.path.join(INPUT_FOLDER, "custom_labels.py")):
        print(
            f"Not all gold_submission_*.csv's or gold_submission/ folders found, but running 'custom_labels.py' to create them..."
        )
        run_script(
            os.path.join(INPUT_FOLDER, "custom_labels.py"),
            ["--global-shared-data-dir", GLOBAL_SHARED_DATA_DIR, "--output-directory", INPUT_FOLDER],
        )
        if has_gold_submission_csv():
            create_gold_permutations_csvs()
            check_gold_submission_csvs(metadata)
        elif has_all_gold_submission_folders():
            check_gold_submission_folders(metadata)
        elif has_all_gold_submission_LRA_folders():
            check_gold_submission_LRA(metadata)
        else:
            print(
                f"Error: custom_labels.py did not create the required gold_submission.csv or gold_submission/ folders",
                file=sys.stderr,
            )
            raise AssertionError(
                "custom_labels.py did not create the required gold_submission.csv or gold_submission/ folders"
            )

    # If not, see if we can create gold_submission.csv from test labels
    else:
        print(
            f"Not all gold_submission_*.csv's or gold_submission/ folders found, and no custom_labels.py found, so creating gold_submission.csv from test labels..."
        )
        create_gold_submission_from_test_labels(metadata)
        check_gold_submission_csvs(metadata)

    print("All tests passed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone test runner for INPUT_FOLDER scripts.")
    parser.add_argument(
        "--global-shared-data-dir", required=True, help="Path to GLOBAL_SHARED_DATA_DIR (will not be modified)"
    )
    parser.add_argument(
        "--agent-data-mount-dir", help="Path to AGENT_DATA_MOUNT_DIR (if omitted, a temp dir will be used)"
    )
    parser.add_argument("--agent-log-dir", help="Path to AGENT_LOG_DIR (if omitted, a temp dir will be used)")
    parser.add_argument("--input-folder", required=True, help="Path to the INPUT_FOLDER containing scripts")
    parser.add_argument(
        "--assert-ends-with-evaluate",
        action="store_true",
        help="Assert that project_description.md ends with evaluate.py code block",
    )
    args = parser.parse_args()

    main(
        global_shared_data_dir=args.global_shared_data_dir,
        agent_data_mount_dir=args.agent_data_mount_dir,
        agent_log_dir=args.agent_log_dir,
        input_folder=args.input_folder,
        assert_ends_with_evaluate=args.assert_ends_with_evaluate,
    )