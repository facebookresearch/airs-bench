# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to download datasets from HuggingFace using a list of dataset URLs.

Given a list of HuggingFace dataset URLs, this script extracts the dataset identifiers
and downloads them using the `datasets` library, saving the data to a specified directory.

Example:
    python prepare_hf_datasets_text.py --datasets_download_location /path/to/output_dir --datasets_csv_path hf_datasets.csv

Requirements:
    pip install datasets==3.6.0
"""



import os
from pathlib import Path
import time
import csv
import asyncio
import requests
import zipfile
from typing import Optional
from datasets import load_dataset, get_dataset_config_names
import argparse


async def download_one_dataset(
    url, output_dir, token, retries=3, use_token=False, config="default"
):
    dataset_name = "/".join(url.rstrip("/").split("/")[-2:])
    print(f"Downloading {dataset_name} ...")
    for attempt in range(retries):
        try:
            if use_token:
                dataset = await asyncio.to_thread(
                    load_dataset, dataset_name, name=config, token=token, trust_remote_code=True
                )
            else:
                dataset = await asyncio.to_thread(load_dataset, dataset_name, name=config, trust_remote_code=True)
            print(f"Available splits: {list(dataset.keys())}")
            save_path = os.path.join(output_dir, dataset_name, config)
            await asyncio.to_thread(dataset.save_to_disk, save_path)
            print(f"Dataset saved to {save_path}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {dataset_name}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(10)
    print(f"Failed to download {dataset_name} after {retries} attempts.")
    return False


async def download_hf_datasets_async(urls, output_dir, token, failed_log_path, config=None, use_token=False):
    failed_datasets = []
    tasks = [
        download_one_dataset(
            hf_dataset_id, output_dir, token, retries=3, use_token=use_token, config=config
        )
        for hf_dataset_id, config in dataset_pairs
    ]
    results = await asyncio.gather(*tasks)
    for url, success in zip(urls, results):
        if not success:
            failed_datasets.append(url)
    if failed_datasets:
        with open(failed_log_path, "w") as f:
            for failed_url in failed_datasets:
                f.write(f"{failed_url}\n")
        print(f"Failed datasets logged to {failed_log_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets.")
    parent_folder = Path(__file__).parent
    parser.add_argument(
        "--datasets_download_location",
        type=str,
        default= parent_folder / "datasets_download_location",
        help="Directory to save downloaded datasets",
    )
    parser.add_argument(
        "--datasets_csv_path",
        type=str,
        default=parent_folder / "hf_datasets.csv",
        help="Path to the CSV file containing dataset IDs and configs",
    )
    args = parser.parse_args()
    output_dir = args.datasets_download_location
    csv_path = args.datasets_csv_path
    failed_log_path = "failed_datasets.txt"

    hf_token = None
    
    dataset_pairs = []
    # Parse CSV file
    with open(csv_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip header row if present
        try:
            header = next(csv_reader)
        except StopIteration:
            print("CSV file is empty")

        # Create pairs from the first two columns
        for row in csv_reader:
            if len(row) >= 2:  # Ensure the row has at least 2 columns
                hf_dataset_id1 = row[0].strip()
                config = row[1].strip()
                dataset_pairs.append((hf_dataset_id1, config))

    print(f"Found {len(dataset_pairs)} datasets to download")
    config = None

    # download HF datasets
    asyncio.run(download_hf_datasets_async(dataset_pairs, output_dir, hf_token, failed_log_path, use_token=False))
    print("All downloads completed.")