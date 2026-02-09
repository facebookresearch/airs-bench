# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a Time Series task to perform time series forecasting on Monash-University/monash_tsf. You will be provided with time series data (`target`), and will need to forecast the future values. Your predictions will be scored against the `label_target` column of the test set.

## Data
### Dataset Description
The rideshare dataset contains various hourly time series representations of attributes related to Uber and Lyft rideshare services for various locations in New York between 26/11/2018 and 18/12/2018. The dataset contains **2304 individual time series**, each capturing different aspects of rideshare demand and pricing, including pickup requests, pricing variations, and service availability across different geographic zones and time periods. The dataset is organized into 156 samples, each containing up to 15 time series, resulting in a total of 2304 individual time series to forecast.

The dataset includes **4 dynamic features** (covariates) that provide additional context for forecasting, such as temporal patterns, weather conditions, traffic indicators, or demand signals that vary over time alongside the main rideshare metrics.

The forecasting task is to predict **48 timesteps into the future** for each of the 2304 time series using historical hourly data. This corresponds to predicting the next 48 hours of rideshare service metrics for each individual time series based on historical patterns and covariate information.

**Note on Missing Values**: This dataset contains NaN (Not a Number) values in the time series, representing missing or unavailable data points. During evaluation, only valid (non-NaN) data points are considered for calculating the Mean Absolute Error. Predictions for time steps with NaN ground truth values are ignored in the evaluation.

### Dataset Structure
The rideshare config of `Monash-University/monash_tsf` has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
'label_target': List(Value('float32')), # true future values for the time series (This is the scoring column to predict). It includes both the historical and future values concatenated together.
'target': List(Value('float32')), # historical values for the time series (a string representation of a list of floats)
'feat_dynamic_real': List(Value('float32')), # additional dynamic real-valued features that can be used alongside the target time series
}
```
Train data can be loaded using load_from_disk('./data/train') and test data can be loaded using load_from_disk('./data/test').
Note that the scoring column has been removed from the test data.

**Data Shape and Format**: The rideshare dataset treats each time series individually, resulting in **2304 individual time series** each requiring 48 future timestep predictions. The submission format is (2304, 48) where each row contains 48 forecast values for one time series.

**NaN Handling in Evaluation**: During evaluation, NaN values in both predictions and ground truth labels are handled gracefully. Only valid (non-NaN) data points are considered when calculating the Mean Absolute Error. If a ground truth value is NaN, the corresponding prediction is ignored in the evaluation, ensuring robust performance measurement even with missing data.

### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` label_target ```

And it should be of shape `(2304,)` where each row contains a JSON-encoded list of 48 forecast values.

### Evaluation Code
The evaluation metric is Mean Absolute Error (MAE), calculated as follows:

```python
#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset, load_from_disk
from sklearn.metrics import mean_absolute_error
from utils import parse_and_validate_predictions_and_labels, process_predictions_and_labels_for_evaluation


def load_test_set():
    """Load test labels and extract forecast portions to match custom_labels.py output"""
    import json
    from utils import reformat_dataset

    dataset = load_from_disk("./data/test_with_labels")

    # The dataset contains the full sequences, but we need just the forecast portions
    # like custom_labels.py generates - 48 values per individual time series
    validation_targets = dataset["target"]  # Base sequences (validation length)
    full_test_targets = dataset["label_target"]  # Full test sequences

    forecast_labels = []
    for val_target, full_target in zip(validation_targets, full_test_targets):
        for i in range(len(full_target)):  # Process each series individually
            val_series_len = len(val_target[i])
            full_series = full_target[i]

            # Extract available forecast portion - keep NaNs in original positions
            available_forecast = full_series[val_series_len:]

            # Take exactly 48 values, padding at the END with NaN if needed
            forecast_48 = available_forecast[:48]  # Take up to 48 values
            while len(forecast_48) < 48:  # Pad at the END if shorter
                forecast_48.append(np.nan)

            # Convert to JSON string like custom_labels.py does
            # Handle NaN values
            forecast_clean = []
            for val in forecast_48:
                if isinstance(val, float) and np.isnan(val):
                    forecast_clean.append("NaN")
                else:
                    forecast_clean.append(val)

            # Each individual time series becomes one row with 48 forecasts
            forecast_labels.append(json.dumps(forecast_clean))

    return forecast_labels


def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """

    all_preds_flat, all_labels_flat = parse_and_validate_predictions_and_labels(predictions, labels)
    valid_preds, valid_labels = process_predictions_and_labels_for_evaluation(all_preds_flat, all_labels_flat)
    mae = mean_absolute_error(valid_labels, valid_preds)
    return {"MAE": mae}


def _cli():
    p = argparse.ArgumentParser(description="Evaluate predictions")
    p.add_argument("--submission-file", default="submission.csv", help="Path to CSV file containing predictions.")
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
                f"Submission file row count ({preds.shape[0]}) does not match test set size ({n_test_samples})."
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


if __name__ == "__main__":
    _cli()

```
