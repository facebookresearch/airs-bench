# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

def reformat_dataset(input_dataset, output_dataset):
    dataset = deepcopy(input_dataset)

    if "target" in output_dataset.column_names:
        dataset = dataset.add_column(f"label_target", output_dataset["target"])
    else:
        raise ValueError(f"Output dataset must contain 'target' column but has columns: {output_dataset.column_names}")

    return dataset

    


    