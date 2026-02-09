
# Overview
## Task Description
This is a Machine Learning project and your goal is to complete the project's TASK following the instructions provided below.

    TASK: This is a NLP task to perform Sentiment Analysis on Yelp/yelp_review_full/. Your predictions will be scored against the `label` column of the test set.

## Data
### Dataset Structure
The default config of Yelp/yelp_review_full/ has the following structure.
Here is a description of the contents of each column including their name, what they contain, and the data type:
```
Column types:
  label: ClassLabel(names=['1 star', '2 star', '3 stars', '4 stars', '5 stars']): The given rating (0 , 1, 2, 3, or 4).
  text: Value('string'): The review text provided by the user.
```
Train data can be loaded using load_from_disk('./data/train') and test data can be loaded using load_from_disk('./data/test').
Note that the scoring column has been removed from the test data.
### Submission file
The submission file should be a csv file named `submission.csv` with the following header:
``` label ```

And it should be of shape `(50000)`.

### Evalution
The evaluation will be performed on the `submission.csv` file you have submitted using the Accuracy metric. Here is the evaluation script that will be used:

```py
#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk


def load_test_set():
    ds = load_from_disk("./data/test_with_labels")
    ground_truth = [int(x) for x in ds["label"]]
    return ground_truth

def evaluate(predictions, labels):
    """
    Returns a dict of metric_name -> value
    """

    # Convert inputs to int if they are strings (purposefully for string only)
    labels = [int(x) if isinstance(x, str) else x for x in labels]
    predictions = [int(x) if isinstance(x, str) else x for x in predictions]

    accuracy = float(np.mean(np.array(predictions) == np.array(labels)))

    return {"Accuracy": accuracy}
    


def _cli():
    p = argparse.ArgumentParser(
        description="Evaluate predictions for qm9 test split using MeanSquaredError."
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