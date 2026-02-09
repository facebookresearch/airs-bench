# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a Time Series task to perform time series forecasting (5 weeks ahead) on Monash-University/monash_tsf. You will be provided with time series data (`target`), and will need to forecast the future values (5 weekly values). Your predictions will be scored against the `label_target` column of the test set.

## Data
### Dataset Description
The solar_weekly dataset provides weekly aggregated solar power generation and forecast data for a large set of simulated photovoltaic (PV) plants across the United States. The dataset captures the complex dynamics of solar power generation, including seasonal variations, weather-dependent fluctuations, and geographic diversity across different climate zones. The forecasting task is to predict the next 5 weeks of solar power generation for each plant using historical weekly generation data.

### Dataset Structure
The solar_weekly config of `Monash-University/monash_tsf` has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
{
'label_target': List(Value('float32')), # true future values for the time series (This is the scoring column to predict). It includes both the historical and future values concatenated together.
'target': List(Value('float32')), # historical values for the time series (a string representation of a list of floats)
}
```
Train data can be loaded using load_from_disk('./data/train') and test data can be loaded using load_from_disk('./data/test').
Note that the scoring column has been removed from the test data.

### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` label_target ```

And it should be of shape `(137,)`.

### Evaluation
The evaluation will be performed on the `submission.csv` file you have submitted using the Mean Absolute Error (MAE) metric. Here is the evaluation script that will be used:

```py
#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import ast
from datasets import load_dataset, load_from_disk
from sklearn.metrics import mean_absolute_error


def load_test_set():

    dataset = load_from_disk('./data/test_with_labels')
    return dataset["label_target"]

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """
    all_preds = []
    all_labels = []
    test_ds = load_from_disk('./data/test_with_labels')
    train_targets = test_ds["target"]

    for pred, label, train_target in zip(predictions, labels, train_targets):
        # Handle NaN values in prediction strings by replacing them with np.nan
        pred_str = pred.replace('NaN', 'null').replace('nan', 'null')
        try:
            pred_list = json.loads(pred_str)
            # Convert null values back to np.nan
            pred_list = [np.nan if x is None else x for x in pred_list]
            pred = np.array(pred_list)
        except json.JSONDecodeError:
            # Fallback to ast.literal_eval if JSON parsing fails
            pred = np.array(ast.literal_eval(pred))

        label = np.array(label)
        train_size = np.array(train_target).shape[0]

        # Extract forecast portion from full label sequence
        label_forecast = label[train_size:]

        # Predictions should already be 5-step forecasts from custom_labels.py
        if pred.shape != label_forecast.shape:
            raise ValueError(
                f"Invalid sample: Prediction shape {pred.shape} does not match "
                f"forecast label shape {label_forecast.shape}. Expected {5} forecast steps."
            )

        all_preds.append(pred)
        all_labels.append(label_forecast)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Flatten arrays
    all_preds_flat = all_preds.flatten()
    all_labels_flat = all_labels.flatten()

    # Remove NaN values - only evaluate on valid data points
    valid_mask = ~(np.isnan(all_preds_flat) | np.isnan(all_labels_flat))

    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points found for evaluation")

    valid_preds = all_preds_flat[valid_mask]
    valid_labels = all_labels_flat[valid_mask]

    mae = mean_absolute_error(valid_labels, valid_preds)
    return {"MAE": mae}




def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions"
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
    result = evaluate(preds, labels)

    print("\n--- EVALUATION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    _cli()
```
