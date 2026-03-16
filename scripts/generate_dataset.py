# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate airs_bench_tasks.csv for upload to HuggingFace datasets.

Each row corresponds to one task from airsbench/tasks/rad/.
Columns:
  task, category, research_problem, dataset, metric,
  metadata.yaml, project_description.md, prepare.py,
  evaluate_prepare.py, evaluate.py, custom_labels.py, utils.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "airsbench" / "tasks" / "rad"
OUTPUT_CSV = REPO_ROOT / "airs_bench_tasks.csv"

# ---------------------------------------------------------------------------
# Column definitions (ordered as they should appear in the CSV)
# ---------------------------------------------------------------------------

# Files whose *content* becomes a CSV column (in desired column order)
FILE_COLUMNS: list[str] = [
    "metadata.yaml",
    "project_description.md",
    "prepare.py",
    "evaluate_prepare.py",
    "evaluate.py",
    "custom_labels.py",
    "utils.py",
]

# Files to silently ignore (not included as columns)
EXCLUDED_FILES: set[str] = {
    "gold_submission.csv",
    "gold_submission_permuted_1.csv",
    "gold_submission_permuted_2.csv",
    "testing_util.py",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_file(path: Path) -> str:
    """Return the text content of *path*, or an empty string if absent."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        warnings.warn(f"Skipping non-UTF-8 file: {path}", stacklevel=2)
        return ""


def extract_metadata_fields(task_dir: Path) -> dict[str, str]:
    """Parse metadata.yaml and return the four scalar logging_info fields."""
    meta_path = task_dir / "metadata.yaml"
    fields: dict[str, str] = {
        "category": "",
        "research_problem": "",
        "dataset": "",
        "metric": "",
    }
    if not meta_path.exists():
        warnings.warn(f"metadata.yaml missing in {task_dir}", stacklevel=2)
        return fields

    with meta_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logging_info: dict = data.get("logging_info", {}) or {}
    for key in fields:
        value = logging_info.get(key, "")
        fields[key] = str(value) if value is not None else ""
    return fields


def check_unexpected_files(task_dir: Path) -> None:
    """Warn if any files exist that are neither in FILE_COLUMNS nor EXCLUDED_FILES."""
    known = set(FILE_COLUMNS) | EXCLUDED_FILES
    for path in task_dir.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(task_dir))
            if rel not in known:
                warnings.warn(
                    f"Unexpected file not captured in any column: {task_dir.name}/{rel}",
                    stacklevel=2,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataframe() -> pd.DataFrame:
    task_dirs = sorted(
        [d for d in TASKS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not task_dirs:
        raise FileNotFoundError(f"No task directories found under {TASKS_DIR}")

    rows: list[dict[str, str]] = []

    for task_dir in task_dirs:
        check_unexpected_files(task_dir)

        row: dict[str, str] = {"task": task_dir.name}

        # Metadata-derived scalar columns
        row.update(extract_metadata_fields(task_dir))

        # File-content columns
        for filename in FILE_COLUMNS:
            row[filename] = read_file(task_dir / filename)

        rows.append(row)

    column_order = (
        ["task", "category", "research_problem", "dataset", "metric"]
        + FILE_COLUMNS
    )
    df = pd.DataFrame(rows, columns=column_order)
    return df


def main() -> None:
    print(f"Reading tasks from: {TASKS_DIR}")
    df = build_dataframe()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} columns to: {OUTPUT_CSV}")
    print(f"Columns: {df.columns.tolist()}")

    # Quick sanity checks
    assert df["task"].nunique() == len(df), "Duplicate task names found!"
    assert df["evaluate.py"].ne("").all(), "Some tasks are missing evaluate.py!"
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
