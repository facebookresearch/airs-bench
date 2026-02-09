# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from datasets import Dataset


def reformat_dataset(input_split, target_split):
    """
    Reformat time series dataset for forecasting task.

    Args:
        input_split: Dataset containing input time series data
        target_split: Dataset containing target time series data (extended sequences)

    Returns:
        Dataset with input sequences and corresponding forecast targets
    """
    input_data = input_split.to_pandas()
    target_data = target_split.to_pandas()

    # Create reformatted dataset
    reformatted_data = {
        'target': input_data['target'].tolist(),
        'label_target': target_data['target'].tolist()
    }

    return Dataset.from_dict(reformatted_data)
