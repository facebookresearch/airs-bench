#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Usage:
#   ./download_hf_datasets_text.sh /path/to/datasets_download_location
#
# If omitted, defaults to ./datasets_download_location (relative to this script's directory).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DOWNLOAD_LOCATION="${1:-"$SCRIPT_DIR/datasets_download_location"}"
DATASETS_CSV_PATH="${DATASETS_CSV_PATH:-"$SCRIPT_DIR/hf_datasets.csv"}"

mkdir -p "$DATASETS_DOWNLOAD_LOCATION"

python "$SCRIPT_DIR/prepare_hf_datasets_text.py" \
  --datasets_download_location "$DATASETS_DOWNLOAD_LOCATION" \
  --datasets_csv_path "$DATASETS_CSV_PATH"

