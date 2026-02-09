#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced script to convert aira-dojo (rad) tasks to the MLGYM format.
Handles tasks with custom data preparation by pre-running prepare.py and evaluate_prepare.py.

Usage:
    python converter_rad_mlgym_enhanced.py ../airsbench/tasks/rad/TextualClassificationSickAccuracy
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

import pandas as pd
import yaml

from datasets import Features, Sequence, Value, load_from_disk


EXCLUDE_DIRS: Set[str] = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}

# Base paths for data storage
SHARED_TEXT_ONLY_DATA_DIR = "../datasets/datasets_download_location"
SHARED_TEXT_ONLY_PREPARED_DATA_DIR = "../datasets/datasets_download_location_prepared"

# Known standard library and third-party packages to skip when detecting auxiliary files
# Generated from comprehensive analysis of all RAD Python files
KNOWN_LIBRARIES = {
    "__future__",
    "absl",
    "anytree",
    "apted",
    "argparse",
    "ast",
    "builtins",
    "collections",
    "copy",
    "csv",
    "datasets",
    "datetime",
    "enum",
    "faulthandler",
    "flax",
    "functools",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "jax",
    "json",
    "logging",
    "math",
    "ml_collections",
    "mteb",
    "multiprocessing",
    "nltk",
    "numpy",
    "optax",
    "os",
    "pandas",
    "pathlib",
    "pickle",
    "platform",
    "pyext",
    "random",
    "re",
    "records",
    "resource",
    "rouge_score",
    "sentence_transformers",
    "shutil",
    "signal",
    "six",
    "sklearn",
    "sqlite3",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "tensorflow",
    "tensorflow_datasets",
    "tensorflow_text",
    "time",
    "tokenize",
    "torch",
    "torchmetrics",
    "tqdm",
    "traceback",
    "transformers",
    "typing",
    "unittest",
    "yaml",
    "threading",
    "concurrent",
    "asyncio",
}

_DEFAULT_TASK_ENTRYPOINT = "CSVSubmissionTasks"


def escape_curly_braces(text: str) -> str:
    """Escape curly braces in text for YAML formatting."""
    return text.replace("{", "{{").replace("}", "}}")


def make_serializable(obj: Any) -> Any:
    """Convert HuggingFace dataset objects to serializable format."""
    if isinstance(obj, Value):
        return {
            "type": "Value",
            "dtype": obj.dtype,
            "id": obj.id,
        }
    elif isinstance(obj, Sequence):
        return {
            "type": "Sequence",
            "feature": make_serializable(obj.feature),
            "length": obj.length,
            "id": obj.id,
        }
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        try:
            return str(obj)
        except Exception:
            return None


def is_task_directory(path: Path) -> bool:
    """
    Check if a directory is a task directory (contains evaluate.py and metadata.yaml).
    """
    return (path / "evaluate.py").exists() and (path / "metadata.yaml").exists()


def find_files(root: Path) -> Iterator[Path]:
    """
    Walk directory tree and yield directories containing evaluate.py and metadata.yaml.
    If root itself is a task directory, yield only that.
    """
    # Check if root itself is a task directory
    if is_task_directory(root):
        yield root
        return

    # Otherwise, walk the tree to find task directories
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if "evaluate.py" in filenames and "metadata.yaml" in filenames:
            yield Path(dirpath)


def read_evaluate(p: Path) -> str:
    """Read evaluate.py file contents."""
    try:
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return p.read_text()
    except Exception as e:
        return f"<<ERROR reading {p}: {e}>>"


def read_metadata(p: Path) -> Dict[str, Any]:
    """Load metadata.yaml file."""
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_config_list(metadata: Dict[str, Any]) -> List[str]:
    config = metadata["logging_info"].get("config")
    if isinstance(config, str):
        return [config]
    elif isinstance(config, list):
        if not all(isinstance(c, str) for c in config):
            raise ValueError("All elements in 'config' list must be strings")
        return config
    else:
        raise TypeError("'config' must be either a string or a list of strings")


def needs_data_preparation(task_path: Path) -> bool:
    """
    Check if a task needs custom data preparation.

    Returns:
        True if prepare.py exists, False otherwise.
    """
    return (task_path / "prepare.py").exists()


def run_prepare_script(task_path: Path, global_data_dir: str, output_dir: str, task_inputs: list[str]) -> bool:
    """
    Run prepare.py to pre-process data for tasks with custom preparation.

    Args:
        task_path: Path to the RAD task directory
        global_data_dir: Path to raw HuggingFace datasets
        output_dir: Directory to save prepared train/test data

    Returns:
        True if successful, False otherwise
    """
    prepare_py = task_path / "prepare.py"

    if not prepare_py.exists():
        print(f"  ⚠️  prepare.py not found in {task_path}")
        return False

    # Create temporary directories for data preparation
    with (
        tempfile.TemporaryDirectory(prefix="agent_data_") as agent_data_dir,
        tempfile.TemporaryDirectory(prefix="agent_log_") as agent_log_dir,
    ):
        # Copy prepare.py and ALL its dependencies to temp directory
        temp_task_dir = None
        auxiliary_files = find_all_auxiliary_files_recursive(task_path, "prepare.py")
        if auxiliary_files:
            temp_task_dir = tempfile.mkdtemp(prefix="task_")
            shutil.copy(prepare_py, os.path.join(temp_task_dir, "prepare.py"))
            # Copy all auxiliary files that prepare.py depends on
            for aux_file in auxiliary_files:
                src = task_path / aux_file
                if src.exists():
                    shutil.copy(src, os.path.join(temp_task_dir, aux_file))
            prepare_script = os.path.join(temp_task_dir, "prepare.py")
        else:
            prepare_script = str(prepare_py)

        try:
            # Run prepare.py
            cmd = [
                sys.executable,
                prepare_script,
                "--global-shared-data-dir",
                global_data_dir,
                "--agent-data-mount-dir",
                agent_data_dir,
                "--agent-log-dir",
                agent_log_dir,
            ]

            print(f"  🔄 Running prepare.py for {task_path.name}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode != 0:
                print(f"  ❌ prepare.py failed: {result.stderr}")
                return False

            # Copy prepared input data to output directory
            for directory in task_inputs:
                path_to_dir = os.path.join(agent_data_dir, directory)
                if not os.path.exists(path_to_dir):
                    print(f"  ❌ prepare.py did not create {directory} directory in {path_to_dir}.")
                    return False

                os.makedirs(output_dir, exist_ok=True)
                shutil.copytree(path_to_dir, os.path.join(output_dir, directory), dirs_exist_ok=True)

            # Also copy val or validation if it exists
            for filename in ["val", "validation"]:
                val_dir = os.path.join(agent_data_dir, filename)
                if os.path.exists(val_dir):
                    shutil.copytree(val_dir, os.path.join(output_dir, filename), dirs_exist_ok=True)

            print(f"  ✅ Data prepared and saved to {output_dir}")
            return True

        except subprocess.TimeoutExpired:
            print("  ❌ prepare.py timed out after 15 minutes")
            return False
        except Exception as e:
            print(f"  ❌ Error running prepare.py: {e}")
            return False
        finally:
            if temp_task_dir:
                shutil.rmtree(temp_task_dir, ignore_errors=True)


def run_evaluate_prepare(
    task_path: Path, global_data_dir: str, prepared_data_dir: str, metadata: Dict[str, Any], task_outputs: list[str]
) -> List[str]:
    """
    Run evaluate_prepare.py to create test_with_labels and other evaluation directories.

    Args:
        task_path: Path to the RAD task directory
        global_data_dir: Path to raw HuggingFace datasets
        prepared_data_dir: Directory where prepared data is stored
        metadata: Task metadata

    Returns:
        List of directory names created (e.g., ['test_with_labels', 'db_schema'])
    """
    evaluate_prepare_py = task_path / "evaluate_prepare.py"

    if not evaluate_prepare_py.exists():
        return []

    with (
        tempfile.TemporaryDirectory(prefix="agent_data_") as agent_data_dir,
        tempfile.TemporaryDirectory(prefix="agent_log_") as agent_log_dir,
    ):
        for task_output_file in task_outputs:
            if task_output_file == "submission.csv":
                # Create dummy submission.csv
                scoring_col = metadata["logging_info"]["scoring_column"]
                # Handle both string and list formats (e.g., time series tasks use lists)
                if isinstance(scoring_col, list):
                    scoring_col = scoring_col[0]
                dummy_csv = os.path.join(agent_log_dir, "submission.csv")
                pd.DataFrame({scoring_col: [0]}).to_csv(dummy_csv, index=False)
            else:
                # Just write an empty file.
                dummy_file = os.path.join(agent_log_dir, task_output_file)
                with open(dummy_file, "w") as f:
                    f.write("")

        # Copy evaluate_prepare.py and ALL its dependencies to temp directory
        temp_task_dir = None
        auxiliary_files = find_all_auxiliary_files_recursive(task_path, "evaluate_prepare.py")
        if auxiliary_files:
            temp_task_dir = tempfile.mkdtemp(prefix="task_")
            shutil.copy(evaluate_prepare_py, os.path.join(temp_task_dir, "evaluate_prepare.py"))
            # Copy all auxiliary files that evaluate_prepare.py depends on
            for aux_file in auxiliary_files:
                src = task_path / aux_file
                if src.exists():
                    shutil.copy(src, os.path.join(temp_task_dir, aux_file))
            evaluate_prepare_script = os.path.join(temp_task_dir, "evaluate_prepare.py")
        else:
            evaluate_prepare_script = str(evaluate_prepare_py)

        try:
            # Run evaluate_prepare.py
            cmd = [
                sys.executable,
                evaluate_prepare_script,
                "--global-shared-data-dir",
                global_data_dir,
                "--agent-data-mount-dir",
                agent_data_dir,
                "--agent-log-dir",
                agent_log_dir,
            ]

            print(f"  🔄 Running evaluate_prepare.py for {task_path.name}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout
            )

            if result.returncode != 0:
                print(f"  ⚠️  evaluate_prepare.py failed: {result.stderr}")
                return []

            # Discover ALL directories created (except submission.csv)
            created_dirs = []
            for item in os.listdir(agent_data_dir):
                item_path = os.path.join(agent_data_dir, item)
                if os.path.isdir(item_path):
                    created_dirs.append(item)
                    # Copy to prepared data location
                    dest_path = os.path.join(prepared_data_dir, item)
                    shutil.copytree(item_path, dest_path, dirs_exist_ok=True)
                    print(f"    ✅ Created directory: {item}")

            return created_dirs

        except subprocess.TimeoutExpired:
            print("  ⚠️  evaluate_prepare.py timed out after 15 minutes")
            return []
        except Exception as e:
            print(f"  ⚠️  Error running evaluate_prepare.py: {e}")
            return []
        finally:
            if temp_task_dir:
                shutil.rmtree(temp_task_dir, ignore_errors=True)


def find_all_auxiliary_files_recursive(
    task_path: Path, start_file: str, visited: Optional[Set[str]] = None
) -> List[str]:
    """
    Recursively find all Python file dependencies starting from a file.

    Args:
        task_path: Path to task directory
        start_file: Starting file (e.g., 'evaluate.py')
        visited: Set of already visited files to avoid cycles

    Returns:
        List of all auxiliary .py files needed (transitive dependencies included)
    """
    if visited is None:
        visited = set()

    if start_file in visited:
        return []

    visited.add(start_file)
    auxiliary_files = []

    file_path = task_path / start_file
    if not file_path.exists():
        return []

    try:
        content = file_path.read_text()
    except Exception:
        return []

    # Find all local imports
    import_pattern = r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    for match in re.finditer(import_pattern, content, re.MULTILINE):
        module_name = match.group(1)

        # Skip standard library and known third-party packages
        if module_name in KNOWN_LIBRARIES:
            continue

        module_file = f"{module_name}.py"
        module_path = task_path / module_file

        if module_path.exists():
            # Add this file if it's not a standard task file
            if module_file not in ["evaluate.py", "prepare.py", "evaluate_prepare.py", "custom_labels.py"]:
                if module_file not in auxiliary_files:
                    auxiliary_files.append(module_file)

                # Recursively find dependencies of this file (transitive imports)
                sub_deps = find_all_auxiliary_files_recursive(task_path, module_file, visited)
                for dep in sub_deps:
                    if dep not in auxiliary_files:
                        auxiliary_files.append(dep)

    return auxiliary_files


def find_auxiliary_python_files(task_path: Path, evaluate_content: str) -> List[str]:
    """
    Find all local .py files that evaluate.py imports (including transitive dependencies).

    Args:
        task_path: Path to the RAD task directory
        evaluate_content: Content of evaluate.py file (not used, kept for compatibility)

    Returns:
        List of auxiliary Python file names (e.g., ['utils.py', 'testing_util.py'])
    """
    return find_all_auxiliary_files_recursive(task_path, "evaluate.py")


def copy_auxiliary_files(task_path: Path, mlgym_data_dir: str, auxiliary_files: List[str]) -> None:
    """
    Copy auxiliary Python files to mlgym data directory.

    Args:
        task_path: Path to the RAD task directory
        mlgym_data_dir: Destination directory in mlgym
        auxiliary_files: List of auxiliary file names to copy
    """
    for filename in auxiliary_files:
        src = task_path / filename
        dst = os.path.join(mlgym_data_dir, filename)
        shutil.copy(src, dst)
        print(f"    📋 Copied auxiliary file: {filename}")


def convert_evaluate(evaluate_rad: str, metadata: Dict[str, Any], uses_prepared_data: bool = False) -> Optional[str]:
    """
    Convert evaluate.py from RAD format to MLGYM format using regex to replace ALL path occurrences.

    Args:
        evaluate_rad: Original evaluate.py content
        metadata: Task metadata
        uses_prepared_data: If True, paths point to prepared data location

    Returns:
        Modified evaluate.py content
    """
    # Pattern matches: './data/something' or "./data/something"
    # Captures the quote style and path separately
    pattern = r"(['\"])\.\/data\/([^'\"]+)\1"

    def replace_path(match):
        quote = match.group(1)  # ' or "
        path_part = match.group(2)  # e.g., test_with_labels, db_schema, test
        # Replace with absolute path in agent workspace
        return f"{quote}/home/agent/workspace/data/{path_part}{quote}"

    evaluate_mlgym = re.sub(pattern, replace_path, evaluate_rad)
    return evaluate_mlgym


def build_starter_code_list(task_id: str, auxiliary_files: List[str], extra_data_dirs: List[str]) -> List[str]:
    """
    Build list of files/dirs to include in starter_code for task yaml.

    Args:
        task_id: Task identifier
        auxiliary_files: List of auxiliary Python files (e.g., ['utils.py'])
        extra_data_dirs: List of extra data directories (e.g., ['test_with_labels', 'db_schema'])

    Returns:
        List of paths for starter_code
    """
    starter_code = [f"data/{task_id}/evaluate.py"]

    # Add auxiliary Python files
    for aux_file in auxiliary_files:
        starter_code.append(f"data/{task_id}/{aux_file}")

    # Note: We don't add extra_data_dirs to starter_code because they're data directories,
    # not code files. MLGYM mounts the entire data directory, so these will be accessible.

    return starter_code


def write_requirements_file(task_id: str, mlgym_path: str | Path, reqs: Iterable[str]) -> None:
    """Generate requirements.txt file for the task."""
    requirements_file = os.path.join(mlgym_path, task_id, "data", "requirements.txt")

    # Base requirements (common packages)
    base_requirements = [
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "scikit-learn",
        "tqdm",
        "datasets",
        "gymnasium",
        "transformers[torch]",
        "matplotlib",
        "torchmetrics",
        "torcheval",
        "xgboost",
        "lightgbm",
        "catboost",
        "omegaconf",
        "evaluate",
        "rouge-score",
        "sacrebleu",
        "sentencepiece",
        "tokenizers",
        "sentence-transformers",
        "accelerate",
        "peft",
        "huggingface-hub",
        "opencv-python",
        "nltk",
        "records",
        "apted",
        "anytree",
        "babel",
    ]

    # Helper function to extract package name from requirement string
    def get_package_name(req: str) -> str:
        """Extract package name from requirement (handles ==, >=, <=, etc.)"""
        # Split on common version specifiers
        for separator in ["==", ">=", "<=", ">", "<", "!=", "~=", "["]:
            if separator in req:
                return req.split(separator)[0].strip()
        return req.strip()

    # Build a dict mapping package name to requirement spec
    requirements_dict = {}

    # Add base requirements first
    for req in base_requirements:
        pkg_name = get_package_name(req)
        requirements_dict[pkg_name] = req

    # Override with task-specific requirements
    # Task-specific versioned requirements take precedence
    for req in reqs:
        pkg_name = get_package_name(req)
        requirements_dict[pkg_name] = req

    # Write to file
    requirements_content = "\n".join(requirements_dict.values())
    with open(requirements_file, "w") as f:
        f.write(requirements_content)


def write_evaluate(task_id: str, mlgym_path: str | Path, content: Optional[str]) -> None:
    """Write converted evaluate.py to mlgym directory."""
    mlgym_evaluate_path = os.path.join(mlgym_path, task_id, "data", "evaluate.py")
    if content is not None:
        with open(mlgym_evaluate_path, "w") as f:
            f.write(content)
    else:
        print(f"  ⚠️  Evaluate not saved (content is None) for task: {task_id}")


def write_dataset_config(
    task_id: str,
    mlgym_path: str | Path,
    metadata: Dict[str, Any],
    global_data_dir: str,
    uses_prepared_data: bool = False,
    prepared_data_path: Optional[str] = None,
) -> None:
    """
    Create dataset config YAML for the task.
    """
    dataset_name = metadata["logging_info"]["dataset"].split("/")[-1]
    features = metadata["logging_info"]["features"]
    metadata["logging_info"]["features"] = features.to_dict() if isinstance(features, Features) else features
    # --- Use config_list instead of single config ---
    config_list = get_config_list(metadata)
    for cfg in config_list:
        if uses_prepared_data and prepared_data_path:
            data_path = prepared_data_path
        else:
            data_path = os.path.join(
                global_data_dir,
                metadata["logging_info"]["dataset"],
                cfg,
            )
        dataset_config_file = os.path.join(
            mlgym_path,
            task_id,
            "configs",
            "datasets",
            f"{dataset_name}_{cfg}.yaml" if cfg != "default" else f"{dataset_name}.yaml",
        )
        dataset_config = {
            "data_path": data_path,
            "description": metadata["logging_info"]["features"],
            "is_local": True,
            "name": dataset_name,
        }
        with open(dataset_config_file, "w") as f:
            yaml.safe_dump(dataset_config, f, default_flow_style=False)


def write_task_config(task_id: str, mlgym_path: str | Path, metadata: Dict[str, Any], starter_code: List[str]) -> None:
    """
    Create task config YAML for the task.

    Args:
        task_id: Task identifier
        mlgym_path: Root path to mlgym project
        metadata: Task metadata
        starter_code: List of files/dirs to include in starter_code
    """
    task_config_file = os.path.join(mlgym_path, task_id, "configs", "tasks", f"{task_id}.yaml")

    features = metadata["logging_info"]["features"]
    metadata["logging_info"]["features"] = features.to_dict() if isinstance(features, Features) else features

    description = metadata["project_description"]
    description += (
        "If a baseline is given, your task is to train a new model that improves "
        "performance on the given dataset as much as possible. If you fail to produce "
        "a valid submission artefact evaluation file will give you a score of 0."
    )

    description = escape_curly_braces(description)
    description += "\n{dataset_docs}"

    config_list = get_config_list(metadata)
    task_config = {
        "id": task_id,
        "name": task_id,
        "description": description,
        "dataset_configs": [
            f"datasets/{metadata['logging_info']['dataset'].split('/')[1]}_{c}.yaml" for c in config_list
        ]
        if config_list[0] != "default"
        else [f"datasets/{metadata['logging_info']['dataset'].split('/')[1]}.yaml"],
        "task_entrypoint": metadata["task_entrypoint"] if "task_entrypoint" in metadata else _DEFAULT_TASK_ENTRYPOINT,
        "training_timeout": 3600,
        "use_generic_conda": False,
        "starter_code": starter_code,  # Use dynamic list
        "evaluation_paths": ["evaluate.py"],
        "metric_lower_is_better": metadata["metric_lower_is_better"],
        "evaluation_read_only": True,
        "memory_path": f"data/{task_id}/memory.json",
        "requirements_path": f"data/{task_id}/requirements.txt",
        "use_separate_eval_container": True,
        "cleanup_eval_on_failure": False,
        "eval_timeout": 3600,
    }

    with open(task_config_file, "w") as f:
        yaml.safe_dump(task_config, f, default_flow_style=False)


def read_project_description(rad_path: Path) -> str:
    """Read project_description.md from RAD task directory."""
    with open(rad_path / "project_description.md", "r") as f:
        return f.read()


def prepare_mlgym_dir(path: str | Path, task_id: str) -> None:
    """Ensure mlgym directory structure exists."""
    os.makedirs(path, exist_ok=True)
    base = os.path.join(path, task_id)
    os.makedirs(str(base) + "/data", exist_ok=True)
    os.makedirs(str(base) + "/configs", exist_ok=True)
    os.makedirs(str(base) + "/configs/datasets", exist_ok=True)
    os.makedirs(str(base) + "/configs/tasks", exist_ok=True)


def load_features_from_prepared_data(prepared_data_path: str) -> Optional[Features]:
    """Load dataset features from prepared data directory."""
    try:
        train_path = os.path.join(prepared_data_path, "train")
        if os.path.exists(train_path):
            dataset = load_from_disk(train_path)
            return dataset.features
    except Exception as e:
        print(f"  ⚠️  Could not load features from prepared data: {e}")
    return None


def main(
    root: Path | str,
    task_inputs: list[str],
    task_outputs: list[str],
    global_data_dir: str = SHARED_TEXT_ONLY_DATA_DIR,
    prepared_data_dir: str | None = None,
    pretty_print_flag: bool = False,
    overwrite: bool = False,
    task_entrypoint: str | None = None,
    task_type: str = "text_only",
) -> None:
    """
    Discover RAD tasks, convert to MLGYM format, handling custom data preparation.

    Workflow:
        1) Find all RAD tasks with evaluate.py and metadata.yaml
        2) Check if task uses custom data preparation (prepare.py exists)
        3) If custom:
           - Run prepare.py to pre-process data (creates train/val/test)
           - Run evaluate_prepare.py to create test_with_labels + extra dirs
           - Save all to prepared data directory
           - Point mlgym config to prepared location
        4) Find and copy auxiliary files (utils.py, etc.)
        5) Convert evaluate.py with regex-based path replacement
        6) Build dynamic starter_code list
        7) Create mlgym configs with all necessary files

    Args:
        root: Root directory to search for RAD tasks
        global_data_dir: Path to raw HuggingFace datasets
        prepared_data_dir: Path to prepared data directory
        pretty_print_flag: Reserved for optional stdout printing
        overwrite: If True, force re-processing even if outputs already exist
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        print(f"❌ Root path is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    task_type_normalized = (task_type or "text_only").lower()
    if prepared_data_dir is None:
        if task_type_normalized == "text_only":
            prepared_data_dir = SHARED_TEXT_ONLY_PREPARED_DATA_DIR

    found_any = False
    for idx, file_path in enumerate(find_files(root), start=1):
        found_any = True
        print("\n" + "=" * 80)
        print(f"[{idx}] Processing: {file_path.name}")
        print("=" * 80)

        # Prepare paths
        file_path_metadata = file_path / "metadata.yaml"
        rad_evaluate_path = file_path / "evaluate.py"
        mlgym_path = "/".join(str(rad_evaluate_path).split("/")[:-2]).replace("rad", "mlgym")

        # Load RAD contents
        rad_evaluate = read_evaluate(rad_evaluate_path)
        metadata = read_metadata(file_path_metadata)
        task_id = metadata["logging_info"]["name"]

        # Check if task needs custom data preparation
        uses_custom_preparation = needs_data_preparation(file_path)
        extra_data_dirs = []

        if uses_custom_preparation:
            print("  🔧 Task uses custom data preparation")

            # Step 1: Run prepare.py
            prepared_data_path = os.path.join(prepared_data_dir, task_id)

            if overwrite or not os.path.exists(prepared_data_path):
                if overwrite and os.path.exists(prepared_data_path):
                    print(
                        f"  🔄 Overwrite mode: removing existing data at {prepared_data_path} before re-running prepare.py"
                    )
                    # Remove existing directory to avoid NFS permission issues when overwriting files
                    shutil.rmtree(prepared_data_path, ignore_errors=False)
                    print("  🗑️  Removed existing directory")
                success = run_prepare_script(file_path, global_data_dir, prepared_data_path, task_inputs)
                if not success:
                    print(f"  ❌ Skipping {task_id} - data preparation failed")
                    continue
            else:
                print(f"  ℹ️  Using existing prepared data at {prepared_data_path}")

            # Step 2: Run evaluate_prepare.py
            extra_data_dirs = run_evaluate_prepare(
                file_path, global_data_dir, prepared_data_path, metadata, task_outputs
            )
            if extra_data_dirs:
                print(f"  ✅ Created extra directories: {', '.join(extra_data_dirs)}")
            else:
                print("  ℹ️  No extra data directories created")

            # Load features from prepared data
            config_list = get_config_list(metadata)
            if len(config_list) > 1:
                print("  ✅  Skipping feature loading from prepared data, due to multiple dataset configs.")
            else:
                # For single config (whether 'default' or named config), try loading from prepared data
                features = load_features_from_prepared_data(prepared_data_path)
                if features:
                    metadata["logging_info"]["features"] = features
                    print("  ✅ Loaded features from prepared data")
                else:
                    print("  ⚠️  Could not load features from prepared data, will try standard method")
        else:
            print("  ℹ️  Task uses standard HuggingFace data loading")
            prepared_data_path = None

        # Step 3: Convert evaluate.py with regex-based path replacement
        mlgym_evaluate = convert_evaluate(rad_evaluate, metadata, uses_prepared_data=uses_custom_preparation)

        if mlgym_evaluate is None:
            print("  ❌ Could not convert evaluate.py")
            continue

        # Prepare mlgym task folder
        prepare_mlgym_dir(mlgym_path, task_id)

        # Step 4: Find and copy auxiliary files
        auxiliary_files = find_auxiliary_python_files(file_path, rad_evaluate)
        if auxiliary_files:
            print(f"  📋 Found auxiliary files: {', '.join(auxiliary_files)}")
            mlgym_data_dir = os.path.join(mlgym_path, task_id, "data")
            copy_auxiliary_files(file_path, mlgym_data_dir, auxiliary_files)

        # Step 5: Build dynamic starter_code list
        starter_code = build_starter_code_list(task_id, auxiliary_files, extra_data_dirs)
        print(f"  📝 Starter code: {starter_code}")

        # Write mlgym data/evaluate.py
        write_evaluate(task_id, mlgym_path, mlgym_evaluate)

        # Write mlgym data/requirements.txt
        reqs = set(metadata["evaluate_container_python_requirements"] + metadata["container_python_requirements"])
        write_requirements_file(task_id, mlgym_path, reqs)

        # Load features if not already loaded
        if "features" not in metadata["logging_info"]:
            from datasets import get_dataset_config_info, load_dataset

            try:
                # --- Use config_list for feature loading ---
                config_list = get_config_list(metadata)
                features = None
                for cfg in config_list:
                    dataset_path = f"{global_data_dir}/{metadata['logging_info']['dataset']}"
                    info = get_dataset_config_info(dataset_path, config_name=cfg)
                    if info.features is not None:
                        features = info.features
                        break
                if features is None:
                    for cfg in config_list:
                        train = load_dataset(os.path.join(dataset_path, cfg), split="train")
                        features = train.features
                        break
            except Exception as e1:
                try:
                    features = get_dataset_config_info(
                        metadata["logging_info"]["dataset"], config_list[0], trust_remote_code=True
                    ).features
                except Exception as e2:
                    try:
                        dataset = load_dataset(
                            metadata["logging_info"]["dataset"],
                            metadata["logging_info"]["config"],
                            split="train",
                            trust_remote_code=True,
                        )
                        features = dataset.features
                    except Exception as e3:
                        print(f"  ❌ Could not load features for {metadata['logging_info']['dataset']}")
                        print(f"     Error 1: {e1}")
                        print(f"     Error 2: {e2}")
                        print(f"     Error 3: {e3}")
                        print(f"  ⚠️  Skipping {task_id}")
                        continue

            metadata["logging_info"]["features"] = features

        # --- Write dataset configs for all configs ---
        write_dataset_config(
            task_id,
            mlgym_path,
            metadata,
            global_data_dir,
            uses_prepared_data=uses_custom_preparation,
            prepared_data_path=prepared_data_path,
        )
        if "project_description" not in metadata:
            project_description = read_project_description(file_path)
            metadata["project_description"] = project_description
        if task_entrypoint:
            metadata["task_entrypoint"] = task_entrypoint
        write_task_config(task_id, mlgym_path, metadata, starter_code)
        print(f"  ✅ {task_id} completed successfully")
    if pretty_print_flag and not found_any:
        print("No matching tasks found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RAD tasks to MLGYM format with enhanced data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a specific task
  python converter_rad_mlgym_enhanced.py tasks/rad/TextualClassificationSickAccuracy
        """,
    )
    parser.add_argument("batch", help="Batch name or specific task path (e.g., batch_4 or batch_4/TaskName)")
    parser.add_argument(
        "--global-data-dir",
        help="Path to global shared data directory)",
    )
    parser.add_argument(
        "--prepared-data-dir",
        default=None,
        help="Path to prepared data directory (default: auto-selected based on task type)",
    )
    parser.add_argument(
        "--task-inputs",
        nargs="+",
        default=["train", "test"],
        help="Task input files/directories that are created by prepare.py and should be copied to task directory (default: train test)",
    )
    parser.add_argument(
        "--task-outputs",
        nargs="+",
        default=["submission.csv"],
        help="Task expected output files for submission, these are used as dummy examples for agent (default: submission.csv)",
    )
    parser.add_argument(
        "--disable-overwrite",
        action="store_false",
        help="Disable force re-processing of all steps when outputs already exist",
    )
    parser.add_argument(
        "--task-entrypoint",
        choices=["CSVSubmissionTasks", "SubmissionFolderTasks"],
        default=_DEFAULT_TASK_ENTRYPOINT,
        help=f"Task entrypoint: CSVSubmissionTasks or SubmissionFolderTasks (default: {_DEFAULT_TASK_ENTRYPOINT})",
    )
    args = parser.parse_args()
    aira_bench_data = "aira-bench-data"
    root: str = os.path.realpath(__file__).split(aira_bench_data)[0]
    batch = args.batch
    path: str = f"{root}/{aira_bench_data}/airsbench/tasks/rad"
    if args.global_data_dir is not None:
        global_data_dir = args.global_data_dir
    else:
        global_data_dir = SHARED_TEXT_ONLY_DATA_DIR
    if args.prepared_data_dir is not None:
        prepared_data_dir = args.prepared_data_dir
    else:
        prepared_data_dir = SHARED_TEXT_ONLY_PREPARED_DATA_DIR
    print(f"Expected inputs: {', '.join(args.task_inputs)}")
    print(f"Expected outputs: {', '.join(args.task_outputs)}")
    main(
        path,
        task_inputs=args.task_inputs,
        task_outputs=args.task_outputs,
        global_data_dir=global_data_dir,
        prepared_data_dir=prepared_data_dir,
        pretty_print_flag=True,
        overwrite=args.disable_overwrite,
        task_entrypoint=args.task_entrypoint,
    )
